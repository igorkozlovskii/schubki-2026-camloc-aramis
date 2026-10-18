"""One-command offline release run: config in, CSV out. Inference side: opens the bundle, one camera store, one first-pose file.

Reuses the frozen S02 reader/runner (allowlisted store, causal scheduling, append-only CSV, single first-pose gate). The raw-MCAP
path is refused with the exact requirements of the S02 exporter (see `raw_mcap_requirements`).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from camloc import runner
from camloc.contracts import CLOCK_CAMERA, STALE_CUTOFF_NS, ContractError, FirstPose
from camloc.io import CameraStore
from camloc.util import canonical_hash, code_hash, sha256_file

from .bundle import BundleError, verify_bundle
from .codeid import camloc_root, check_code, executing_mode, unpinned_modules
from .config import ReleaseConfigError, load_release_config, resolve_model_config
from .validate import validate_prediction_csv

class ReleaseInputError(ContractError):
    pass


OPTIONAL_SET_KEYS = ("hold_policy",)    # accepted by the composites although the shipped configs do not spell them out (default = unchanged behaviour)


def deep_merge(base: dict, over: dict, path: str = "", allow_new=OPTIONAL_SET_KEYS) -> dict:
    """`--set` semantics (S08e): nested dicts are merged key by key (an override of `e04` or `filter` keeps its other keys), everything else replaces.
    A key that does not exist in the model's params is refused (a typo must not silently run the unchanged model), except `allow_new`."""
    out = dict(base)
    for k, v in over.items():
        here = f"{path}.{k}" if path else k
        if k not in base and k not in allow_new:
            raise ReleaseInputError(f"--set {here}: not a parameter of the model (known: {sorted(base)[:12]})")
        out[k] = deep_merge(base[k], v, here, ()) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def set_nested(d: dict, dotted: str, value) -> None:
    """`--set e04.top_k=3` -> {"e04": {"top_k": 3}} (dotted keys build nested overrides; merged by deep_merge)."""
    parts = dotted.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
        if not isinstance(d, dict):
            raise ReleaseInputError(f"--set {dotted}: {p!r} is given both as a value and as a prefix")
    d[parts[-1]] = value


def resolve_input(path) -> tuple[Path, bool]:
    """(path, is_raw_mcap). A directory is a safe camera store; a *.mcap file is exported first (see `prepare_raw_input`)."""
    p = Path(path)
    if p.suffix.lower() == ".mcap":
        if not p.is_file():
            raise ReleaseInputError(f"input MCAP not found: {p}")
        return p, True
    if p.is_file() and p.suffix.lower() == ".db3":
        raise ReleaseInputError("REFUSED raw input: only MCAP (*.mcap) recordings are supported")
    if not p.is_dir():
        raise ReleaseInputError(f"input must be a safe camera store directory or a raw *.mcap file, got {p}")
    return p, False


def find_exporter_python(repo_root: Path, explicit=None) -> str:
    """Interpreter that can import mcap + mcap_ros2 (the generic exporter needs them; the vision environment does not ship them).
    Order: --exporter-python, the current interpreter, <repo>/.pixi/envs/default/bin/python."""
    import importlib.util
    cands = []
    if explicit:
        cands.append(str(explicit))
    if importlib.util.find_spec("mcap") and importlib.util.find_spec("mcap_ros2"):
        cands.append(sys.executable)
    cands.append(str(repo_root / ".pixi/envs/default/bin/python"))
    for c in cands:
        if Path(c).is_file():
            r = subprocess.run([c, "-c", "import mcap, mcap_ros2, jsonschema"], capture_output=True, timeout=60)
            if r.returncode == 0:
                return c
            if explicit and c == str(explicit):
                raise ReleaseInputError(f"--exporter-python {c} cannot import mcap, mcap_ros2 and jsonschema")
    raise ReleaseInputError("a raw MCAP needs an interpreter with mcap, mcap-ros2-support and jsonschema (the vision environment has none): "
                            "run `pixi install` for the default environment of this repository or pass --exporter-python <python>, or export first with "
                            "scripts/export_test_input.py and pass the store directory")


def prepare_raw_input(mcap: Path, *, bundle: Path, repo_root: Path, workdir: Path, rng, split: Path | None, cfg: dict, exporter_python=None) -> dict:
    """Raw MCAP -> safe store under `workdir` via the generic allowlisting exporter (a subprocess of scripts/export_test_input.py,
    code pinned by the bundle's code snapshot). Returns {store, report, exporter_python, command, exit_code, source_sha256, dev}.
    The supplied dev recording is recognised by its sha256 BEFORE anything is exported: it needs --split and --range, and the widened
    window is audit-guarded first, so no audit-lap byte is ever read for it."""
    from camloc import runner
    src_sha = sha256_file(mcap)
    dev = src_sha == cfg["dev_recording_sha256"]
    export_rng = None
    if dev:
        if split is None:
            raise ReleaseInputError("input is the supplied dev recording: --split (audit guard) is mandatory")
        if rng is None:
            raise ReleaseInputError("raw dev-recording input needs an explicit --range (only that window is exported, audit laps are never read)")
        export_rng = (int(rng[0]) - 1_000_000_000, int(rng[1]) + 1_000_000_000)    # log_time lags the stamp by < 0.1 s; 1 s margin
        runner.audit_guard(runner.load_split(split), export_rng[0], export_rng[1], cfg)
    store = Path(workdir) / (mcap.stem + "_store")
    report = store.with_name(store.name + ".export_report.json")
    if store.exists() or report.exists() or store.with_name(store.name + ".partial").exists():
        raise ReleaseInputError(f"REFUSED: {store} (or its report/.partial) exists in --workdir; delete it or choose another --workdir")
    Path(workdir).mkdir(parents=True, exist_ok=True)
    py = find_exporter_python(repo_root, exporter_python)
    cmd = [py, "-B", str(repo_root / "scripts/export_test_input.py"), "--mcap", str(mcap), "--out", str(store),
           "--compare-calib", str(bundle / "calibration/calib.json")]
    if cfg["runtime"].get("model_compares_full_calibration_sha"):      # legacy: only a model that compares the whole calib.json hash needs the normalised bytes
        cmd.append("--canonical-calib")
    if export_rng:
        cmd += ["--range-ns", str(export_rng[0]), str(export_rng[1])]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if r.returncode not in (0, 3):
        shutil.rmtree(store, ignore_errors=True)
        raise ReleaseInputError(f"raw MCAP export failed (exit {r.returncode}): {r.stderr.strip()[-600:]}")
    return {"store": store, "report": report if report.is_file() else None, "exporter_python": py, "command": " ".join(cmd[2:]),
            "exit_code": r.returncode, "source_sha256": src_sha, "dev": dev, "export_range_ns": list(export_rng) if export_rng else None}


TEL_EXPORT_SCRIPT = "scripts/export_telemetry.py"


def check_telemetry_dir(tdir: Path, required_fields) -> dict:
    """Pre-flight of a telemetry-v1 directory (default-deny, no pandas): header = stamp_ns + allowed fields only, the required fields present, rows > 0.
    Returns {ok, reason, rows, first_stamp_ns, last_stamp_ns, csv_sha256}; never raises."""
    from camloc.io.telemetry import ALLOWED_FIELDS, TELEMETRY_NAME, TelemetryError, assert_no_forbidden
    info = {"ok": False, "reason": None, "rows": 0, "first_stamp_ns": None, "last_stamp_ns": None, "csv_sha256": None, "dir": str(tdir)}
    try:
        p = Path(tdir) / TELEMETRY_NAME
        if not p.is_file():
            info["reason"] = f"no {TELEMETRY_NAME} in {tdir}"
            return info
        with open(p, newline="") as f:
            rd = csv.reader(f)
            header = next(rd, [])
            if "stamp_ns" not in header:
                info["reason"] = "telemetry.csv has no stamp_ns column"
                return info
            cols = [c for c in header if c != "stamp_ns"]
            assert_no_forbidden(cols)
            unknown = [c for c in cols if c not in ALLOWED_FIELDS]
            if unknown:
                info["reason"] = f"telemetry columns outside the telemetry-v1 allowlist: {unknown}"
                return info
            missing = [c for c in required_fields if c not in cols]
            if missing:
                info["reason"] = f"telemetry.csv lacks required fields {missing}"
                return info
            first = last = None
            n = 0
            for row in rd:
                if not row:
                    continue
                if first is None:
                    first = int(row[0])
                last = int(row[0])
                n += 1
        info.update(rows=n, first_stamp_ns=first, last_stamp_ns=last, csv_sha256=sha256_file(p))
        if n == 0:
            info["reason"] = "telemetry.csv has no rows"
            return info
        info["ok"] = True
    except (TelemetryError, ValueError, OSError) as e:
        info["reason"] = f"{type(e).__name__}: {e}"
    return info


def prepare_telemetry(*, mcap: Path | None, telemetry, workdir: Path, repo_root: Path, tel_cfg: dict, exporter_python, export_rng, stem: str) -> dict:
    """Resolve the telemetry input of a telemetry release: --telemetry none -> unavailable; --telemetry DIR -> that directory; else, for a raw MCAP, the separate
    allowlisting exporter (scripts/export_telemetry.py, only /constructor0/state_estimation, only telemetry-v1 fields). Any problem (topic absent, field missing, forbidden
    column, exporter failure, empty file) yields tier T0 with the reason; it never raises. Returns {available, reason, dir, check, export}."""
    out = {"available": False, "reason": None, "dir": None, "check": None, "export": None}
    req = tel_cfg.get("required_fields", [])
    if telemetry is not None and str(telemetry) == "none":
        out["reason"] = "--telemetry none"
        return out
    if telemetry is not None:
        tdir = Path(telemetry)
    elif mcap is not None:
        tdir = Path(workdir) / (stem + "_telemetry")
        if tdir.exists() or tdir.with_name(tdir.name + ".partial").exists():
            out["reason"] = f"{tdir} exists in --workdir; delete it or choose another --workdir"
            return out
        try:
            py = find_exporter_python(repo_root, exporter_python)
        except ReleaseInputError as e:
            out["reason"] = f"no interpreter for the telemetry exporter: {e}"
            return out
        Path(workdir).mkdir(parents=True, exist_ok=True)
        cmd = [py, "-B", str(repo_root / TEL_EXPORT_SCRIPT), "--input", str(mcap), "--out", str(tdir)]
        if export_rng:
            cmd += ["--range-ns", str(export_rng[0]), str(export_rng[1])]
        r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        out["export"] = {"command": " ".join(cmd[2:]), "exit_code": r.returncode, "stderr_tail": r.stderr.strip()[-300:]}
        if r.returncode != 0:
            out["reason"] = ("telemetry topic /constructor0/state_estimation absent from the recording" if r.returncode == 3
                             else f"telemetry export failed (exit {r.returncode}): {r.stderr.strip()[-200:]}")
            shutil.rmtree(tdir, ignore_errors=True)
            return out
    else:
        out["reason"] = "no telemetry supplied (store input without --telemetry)"
        return out
    chk = check_telemetry_dir(tdir, req)
    out["check"] = chk
    if not chk["ok"]:
        out["reason"] = chk["reason"]
        return out
    out.update(available=True, dir=str(tdir))
    return out


def load_first_pose_any(path, out_json: Path) -> tuple[FirstPose, str]:
    """Accept first-pose-v1 JSON, a plain {timestamp_ns,x,y,yaw} JSON, or a one-row CSV with header. Writes a canonical, provenance-free
    JSON to out_json (the file the runner consumes). Returns (pose, sha256 of the ORIGINAL file)."""
    p = Path(path)
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if p.suffix.lower() == ".csv":
        rows = list(csv.reader(raw.decode().splitlines()))
        if len(rows) != 2 or rows[0] != ["timestamp_ns", "x", "y", "yaw"]:
            raise ReleaseInputError("first-pose CSV must be a header 'timestamp_ns,x,y,yaw' plus exactly one row")
        d = {"timestamp_ns": int(rows[1][0]), "x": float(rows[1][1]), "y": float(rows[1][2]), "yaw": float(rows[1][3])}
    else:
        d = json.loads(raw)
        d = {k: v for k, v in d.items() if k not in ("schema", "provenance")}   # provenance (GT row ids etc.) never reaches the model
        extra = set(d) - {"timestamp_ns", "x", "y", "yaw"}
        if extra:
            raise ReleaseInputError(f"first-pose file has unknown/forbidden keys: {sorted(extra)} (only timestamp_ns,x,y,yaw)")
        if not {"timestamp_ns", "x", "y", "yaw"} <= set(d):
            raise ReleaseInputError("first-pose file must carry timestamp_ns,x,y,yaw")
    fp = FirstPose(int(d["timestamp_ns"]), float(d["x"]), float(d["y"]), float(d["yaw"]))
    out_json.write_text(json.dumps({"schema": "first-pose-v1", "timestamp_ns": fp.timestamp_ns, "x": fp.x, "y": fp.y, "yaw": fp.yaw}, indent=1) + "\n")
    return fp, sha


def _git_commit(root: Path) -> str | None:
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:   # noqa: BLE001
        return None


def check_calibration(cfg: dict, bundle: Path, store_calib: Path, store_full_sha: str) -> dict:
    """Camera-only calibration gate. The reference is the bundle's calibration/calib.json (hash-verified with the bundle) and must itself
    equal the config's pinned camera fingerprint. A camera present in both calibrations must be identical; the cameras the model reads
    (runtime.calibration_required_cameras, default fl) must be present and identical. Lidar/radar/gnss/imu extrinsics never matter.
    Returns the record (status 'match' | 'note'); raises ReleaseInputError on a mismatch."""
    from camloc.io import calibfp
    rt = cfg["runtime"]
    ref = calibfp.fingerprints(bundle / "calibration/calib.json")
    pin = rt.get("calibration_camera_fingerprint")
    if pin is not None and ref["camera_fingerprint"] != pin:
        raise BundleError("bundle calibration/calib.json does not match the release config's pinned camera fingerprint")
    got = calibfp.fingerprints(store_calib)
    required = tuple(rt.get("calibration_required_cameras", ["fl"]))
    cmp = calibfp.compare(ref["per_camera"], got["per_camera"], required)
    rec = {"status": cmp["status"], "required_cameras": list(required), "camera_fingerprint": got["camera_fingerprint"],
           "release_camera_fingerprint": ref["camera_fingerprint"], "full_sha256": store_full_sha, "release_full_sha256": ref["full_sha256"],
           "non_camera_differences": store_full_sha != ref["full_sha256"] and cmp["status"] != "mismatch", "comparison": cmp}
    if cmp["status"] == "mismatch":
        raise ReleaseInputError(
            f"calibration mismatch: fitted map extrinsics invalid (camera-only fingerprint {got['camera_fingerprint'][:12]} != release "
            f"{ref['camera_fingerprint'][:12]}; differing cameras {cmp['differing']}, missing required {cmp['missing_required']}). "
            "Run export_test_input.py --compare-calib to see per-camera dK/dT.")
    return rec


def check_split_pin(cfg: dict, split: Path | None) -> dict | None:
    """If the release config pins a split (file sha256 + split_hash), a --split that differs is refused. Limits: this guards against
    a wrong/edited split FILE; it does not prove that the input store is the dev recording (the dev-recording test is the source MCAP
    sha256 in the store manifest, which a store builder controls) - see SETUP.md."""
    pin = cfg.get("split_pin")
    if split is None or pin is None:
        return None
    import yaml
    sha = sha256_file(split)
    sh = yaml.safe_load(Path(split).read_text()).get("split_hash")
    if sha != pin["file_sha256"] or sh != pin["split_hash"]:
        raise ReleaseInputError(f"--split {split} does not match the split pinned in the release config "
                                f"(file sha256 {sha[:12]} vs {pin['file_sha256'][:12]}, split_hash {str(sh)[:12]} vs {pin['split_hash'][:12]})")
    return {"split_id": pin.get("split_id"), "file_sha256": sha, "split_hash": sh, "pinned": True}


def count_flagged_ticks(diag_path: Path) -> dict:
    """Count ticks the model itself labeled as degraded service (S07g): `tracking == "degraded"` and the `coast_limit_hold` flag. Never raises."""
    out = {"steps": 0, "degraded": 0, "coast_limit_hold": 0, "cam_fr_fallback": 0, "first_degraded_t_ns": None, "error": None}
    try:
        with open(diag_path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("event") != "step":
                    continue
                out["steps"] += 1
                if d.get("tracking") == "degraded":
                    out["degraded"] += 1
                    if out["first_degraded_t_ns"] is None:
                        out["first_degraded_t_ns"] = d.get("t")
                if "coast_limit_hold" in (d.get("flags") or ()):
                    out["coast_limit_hold"] += 1
                if "cam_fr_fallback" in (d.get("flags") or ()):
                    out["cam_fr_fallback"] += 1
    except Exception as e:   # noqa: BLE001  never fail the run over a counter
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def run_release(*, bundle: Path, input_path: Path, first_pose: Path, output: Path, rng=None, split: Path | None = None,
                config_override: dict | None = None,
                command: str = "", set_params: dict | None = None, verify_payloads: bool = True,
                instrument: dict | None = None, workdir: Path | None = None, keep_store: bool = False, exporter_python=None, telemetry=None,
                audit_open: bool = False) -> dict:
    """instrument (benchmark only): {'after_row': callable(n, csv_path)}; filled with 'create_model_s' and 'verify_s'. Behaviour is otherwise identical."""
    t_start = time.perf_counter()
    from .netguard import block_network
    block_network()
    bundle, output = Path(bundle), Path(output)
    store_dir, is_raw = resolve_input(input_path)
    if audit_open and is_raw:
        raise ReleaseInputError("--audit-open is not available for a raw MCAP input: export the audit laps with the unlocked exporters and pass the store (AUDIT_RUNBOOK.md)")
    # 1. bundle: strict verification (all hashes), no silent fallbacks
    t_v0 = time.perf_counter()
    vrep = verify_bundle(bundle, check_hash=True)
    if instrument is not None:
        instrument["verify_s"] = time.perf_counter() - t_v0
    man = json.loads((bundle / "bundle_manifest.json").read_text())
    cfg = load_release_config(bundle / "release_config.yaml")
    if cfg["release_id"] != man["release_id"]:
        raise BundleError("release_config.yaml does not belong to this bundle manifest")
    # 2. code identity: the checkout that is executing must equal the bundle's code snapshot, file by file (no escape hatch)
    src_root = camloc_root()
    mode = executing_mode(bundle)
    if mode != "repo":
        raise BundleError("camloc is imported from inside the bundle: bundle-code execution is not a supported mode "
                          "(the bundle's code/ is provenance; run from a repository checkout)")
    crep = check_code(man, src_root)
    if crep["mismatches"]:
        raise BundleError(f"code in use differs from the bundle's code snapshot: {crep['mismatches'][:5]} "
                          "(rebuild the bundle for this commit, or check out the commit the bundle was built from)")
    # 3. output paths (exclusive)
    work = output.with_name(output.stem + ".run")
    diag_out = output.with_name(output.stem + ".diagnostics.jsonl")
    rec_out = output.with_name(output.stem + ".run_record.json")
    exp_out = output.with_name(output.stem + ".export_report.json")
    tman_out = output.with_name(output.stem + ".telemetry_manifest.json")
    for p in (output, diag_out, rec_out, work, exp_out, tman_out):
        if p.exists():
            raise ContractError(f"REFUSED: {p} exists (outputs are never overwritten)")
    # 3b. raw MCAP: export to a scratch store first (generic allowlisting exporter); the store is deleted afterwards unless --keep-store
    # the split pin is checked first: a wrong/edited --split must not drive the audit guard or an export (S08d)
    split_check = check_split_pin(cfg, split)
    raw = None
    wd = Path(workdir) if workdir is not None else output.with_name(output.stem + ".work")
    if is_raw:
        raw = prepare_raw_input(store_dir, bundle=bundle, repo_root=src_root.parent, workdir=wd, rng=rng, split=split, cfg=cfg,
                                exporter_python=exporter_python)
        store_dir = raw["store"]
    tel_dir = tel_scratch = None
    try:
        # 4. input store: the S02 reader enforces the allowlist/hashes; calibration must equal the bundle's
        with CameraStore(store_dir, verify=True) as store:
            calib_sha = sha256_file(store.root / "calib.json")
            src_sha = store.manifest["source"]["sha256"]
            store_id, store_manifest_sha = store.manifest["store_id"], store.manifest_sha256
            fl = store.stamps[CLOCK_CAMERA]
            n_fl = int(len(fl))
            fl_first, fl_last = (int(fl[0]), int(fl[-1])) if n_fl else (None, None)
            input_cameras = store.cameras()
        calib_check = check_calibration(cfg, bundle, store_dir / "calib.json", calib_sha)
        if cfg["runtime"].get("model_compares_full_calibration_sha") and calib_sha != cfg["runtime"]["calibration_sha256"]:
            # the frozen S04 model (E04) compares the WHOLE calib.json hash with its map's; fail early and explain instead of at model init
            raise ReleaseInputError(
                f"camera calibration is acceptable ({calib_check['status']}) but the model compares the whole calib.json hash with its map's "
                f"({calib_sha[:12]} != {cfg['runtime']['calibration_sha256'][:12]}): the input store differs in camera_info order or in non-camera extrinsics. "
                "Pass the raw MCAP to run_release.py (it exports with --canonical-calib) or export with scripts/export_test_input.py "
                "--compare-calib <bundle>/calibration/calib.json --canonical-calib.")
        if n_fl == 0:
            raise ReleaseInputError("input store has no fl images")
        dev = (src_sha == cfg["dev_recording_sha256"])
        if dev and split is None:
            raise ReleaseInputError("input is the supplied dev recording: --split (audit guard) is mandatory; laps 6-7 are never run in S08a")
        # 5. first pose: consumed once, by the runner gate
        output.parent.mkdir(parents=True, exist_ok=True)
        work.mkdir(parents=True)   # created only after every input check passed: a refusal leaves nothing behind
        fp_json = work / "first_pose.json"
        fp, fp_orig_sha = load_first_pose_any(first_pose, fp_json)
        start = fp.timestamp_ns if rng is None else int(rng[0])
        end = (fl_last + 1) if rng is None else int(rng[1])
        if end <= start:
            raise ReleaseInputError("empty range")
        if split is None:   # hidden-test path: a split with no laps (nothing to guard); the real split is used whenever given
            split_path = work / "release_split.yaml"
            split_path.write_text(yaml.safe_dump({"split_id": "release-no-split", "status": "frozen", "laps": []}))
        else:
            split_path = Path(split)
        # 5b. tiers (S08e): a telemetry release (config block `telemetry`) runs its telemetry model at T2 only when the allowed telemetry-v1 input is present and
        # valid; otherwise it falls back, with a stderr warning and a run-record flag, to its camera-only fallback_model (T0). A camera-only release is T0 by definition.
        tel_cfg = cfg.get("telemetry")
        tier = {"tier": "T0", "slot": "model", "fell_back": False, "reason": "camera-only release (no telemetry block)", "telemetry": None}
        tel_dir = None
        if tel_cfg:
            tel_rng = raw["export_range_ns"] if raw is not None else None
            tprep = prepare_telemetry(mcap=Path(input_path) if is_raw else None, telemetry=telemetry, workdir=wd,
                                      repo_root=src_root.parent, tel_cfg=tel_cfg, exporter_python=exporter_python, export_rng=tel_rng, stem=Path(input_path).stem)
            chk = tprep["check"] or {}
            if tprep["available"] and chk.get("first_stamp_ns") is not None and (chk["last_stamp_ns"] < start or chk["first_stamp_ns"] >= end):
                tprep.update(available=False, reason="telemetry stamps do not overlap the run range")
            tier["telemetry"] = {"available": tprep["available"], "reason": tprep["reason"], "export": tprep["export"], "check": tprep["check"]}
            if tprep["available"]:
                tier.update(tier="T2", slot="model", reason="telemetry-v1 present and valid")
                tel_dir = Path(tprep["dir"])
                tel_scratch = tel_dir if (telemetry is None and is_raw) else None
            else:
                tier.update(tier="T0", slot="fallback_model", fell_back=True, reason=tprep["reason"])
        # 5c. audit unlock (S08b round 3): only for a store of the dev recording whose range reaches laps 6-7, only with --audit-open and the orchestrator's token; the
        # ledger line is written here, before the run. The model config of THIS run (and no other) then carries audit_opened/audit_reason for the runner's guard.
        audit_cfg = {}
        if dev and split is not None and audit_open:
            from camloc import runner as _runner
            if any(start < b_ and end > a_ for (a_, b_, _lap) in _runner.audit_intervals(_runner.load_split(split))):
                from camloc.io import unlock as _eg
                try:
                    g = _eg.grant_audit("run", Path(input_path).name, src_sha, [start, end], audit_open=True, command=command, subject=cfg["release_id"])
                except _eg.AuditRefused as e:
                    raise ReleaseInputError(str(e)) from None
                audit_cfg = {"audit_opened": True, "audit_reason": f"S08b audit run (token {g['ledger_entry']['token_sha256'][:12]}, ledgered)"}
        # 6. the bundle's vendored code (LightGlue) must be importable before ANY model is constructed, including the pre-built T2 telemetry model and its
        # fallback (S08b round 2: on a clean machine the vision env PYTHONPATH does not point at it)
        ext_code = bundle / "external/S04-lightglue/LightGlue"
        if ext_code.is_dir() and str(ext_code) not in sys.path:
            sys.path.insert(0, str(ext_code))
        # 6b. resolved model config (absolute bundle paths only in this per-run copy)
        def build_model_cfg(slot):
            mc = {**resolve_model_config(cfg, bundle, slot), **audit_cfg}
            if config_override:
                mc = {**mc, **config_override}
            if set_params:
                mc["params"] = deep_merge(mc["params"], set_params)
            return mc
        model_cfg = build_model_cfg(tier["slot"])
        pre_model = None
        if tier["tier"] == "T2":      # construct the telemetry model now: a telemetry problem that only the model sees (file refused) still ends in T0, never a crash
            from camloc.runner import create_model
            try:
                pre_model = create_model(model_cfg, {**(model_cfg.get("assets") or {}), "telemetry_dir": str(tel_dir)})
            except (ContractError, OSError, ValueError, KeyError, RuntimeError) as e:    # TelemetryError is a RuntimeError
                tier.update(tier="T0", slot="fallback_model", fell_back=True, reason=f"telemetry model refused its input: {type(e).__name__}: {str(e)[:200]}")
                tier["telemetry"]["available"] = False
                tel_dir = None
                model_cfg = build_model_cfg("fallback_model")
        if tier["fell_back"]:
            print(f"WARNING: telemetry unavailable ({tier['reason']}); running the camera-only option (tier T0, model {model_cfg['model_id']}). "
                  "The output is valid but camera-only.", file=sys.stderr)
        cfg_path = work / "model_config.resolved.yaml"
        cfg_path.write_text(yaml.safe_dump(model_cfg))
        spec = runner.RunSpec(store_dir=store_dir, model_config=cfg_path, first_pose=fp_json, out_dir=work / "runner", mode="continuous",
                              start_ns=start, end_ns=end, split_path=split_path, clock=cfg["runtime"]["clock_camera"],
                              stale_ns=int(cfg["runtime"]["stale_cutoff_ms"]) * 1_000_000, verify_payloads=verify_payloads, command=command,
                              allow_telemetry=tier["tier"] == "T2", telemetry_dir=tel_dir)
        t_run = time.perf_counter()
        if pre_model is not None:
            meta = runner.run(spec, model=pre_model, model_cfg=model_cfg)
        elif instrument is None:
            meta = runner.run(spec)
        else:
            from camloc.runner import create_model
            t_m = time.perf_counter()
            model = create_model(model_cfg, model_cfg.get("assets"))
            instrument["create_model_s"] = time.perf_counter() - t_m
            meta = runner.run(spec, after_row=instrument.get("after_row"), model=model, model_cfg=model_cfg)
        t_done = time.perf_counter()
        # 7. the code that really ran: every imported camloc module must be a hash-pinned file of the snapshot
        unpinned = unpinned_modules(man, src_root)
        if unpinned:
            raise BundleError(f"camloc modules executed but not pinned in the bundle code snapshot: {unpinned}")
        # 8. outputs: validate, then publish the three files
        pred = work / "runner/predictions.csv"
        schema = Path(__file__).resolve().parents[3] / "schemas/prediction.schema.json"
        vcsv = validate_prediction_csv(pred, schema if schema.is_file() else None, require_schema=True)   # no silent fallback
        if meta["first_pose_consumed"] != 1:
            raise ContractError(f"first pose consumed {meta['first_pose_consumed']} times")
        shutil.copyfile(pred, output)
        shutil.copyfile(work / "runner/diagnostics.jsonl", diag_out)
        flagged = count_flagged_ticks(diag_out)
        if flagged["degraded"] or flagged["coast_limit_hold"]:
            print(f"WARNING: {flagged['degraded']} of {flagged['steps']} ticks were flagged degraded, {flagged['coast_limit_hold']} held by the coast limit "
                  f"(camera fixes missing for long stretches); the output is valid but low-confidence there. See {diag_out}", file=sys.stderr)
        if tier["tier"] == "T2" and (tel_dir / "telemetry_manifest.json").is_file():
            shutil.copyfile(tel_dir / "telemetry_manifest.json", tman_out)
            tier["telemetry"]["manifest"] = str(tman_out)
            tier["telemetry"]["manifest_sha256"] = sha256_file(tman_out)
        raw_rec = None
        if raw is not None:
            if raw["report"] is not None:
                shutil.copyfile(raw["report"], exp_out)
            raw_rec = {"mcap": str(input_path), "source_sha256": raw["source_sha256"], "exporter_python": raw["exporter_python"], "exporter_command": raw["command"],
                       "exporter_exit_code": raw["exit_code"], "export_range_ns": raw["export_range_ns"], "scratch_store_kept": bool(keep_store),
                       "scratch_store": str(raw["store"]) if keep_store else None,
                       "export_report": str(exp_out) if raw["report"] is not None else None,
                       "export_report_sha256": sha256_file(exp_out) if raw["report"] is not None else None}
        versions = {"python": sys.version.split()[0]}
        try:
            import torch
            versions.update({"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        except Exception:   # noqa: BLE001
            pass
        from .netguard import active
        rec = {
            "schema": "release-run-record-v1", "status": meta["status"], "release_id": cfg["release_id"], "model_id": model_cfg["model_id"], "tier": tier,
            "bundle": {"hash": man["bundle_hash"], "files_verified": vrep["files"], "bytes": vrep["bytes"], "all_hashes_checked": vrep["hashes_checked"]},
            "code": {"mode": mode, "executing_root": str(src_root), "hash_enforced": True, "snapshot_hash": man["code"]["code_hash"],
                     "executing_tree_hash": crep["tree_hash"], "tree_equals_snapshot": crep["tree_hash"] == man["code"]["code_hash"],
                     "check": {k: crep[k] for k in ("code_files_checked", "mismatches")}, "modules_loaded_all_pinned": True,
                     "package_code_hash_in_use": code_hash(src_root / "camloc"), "repo_git_head": _git_commit(src_root.parent)},
            "input": {"store_id": store_id, "store_manifest_sha256": store_manifest_sha, "source_mcap_sha256": src_sha, "dev_recording": dev,
                      "cameras_in_store": input_cameras, "calibration_sha256": calib_sha, "calibration": calib_check, "fl_rows": n_fl, "raw_mcap": raw_rec},
            "first_pose": {"original_sha256": fp_orig_sha, "timestamp_ns": fp.timestamp_ns, "consumed": meta["first_pose_consumed"]},
            "range_ns": [start, end], "audit_guard": ("real split" if split is not None else "none (no split supplied; hidden-test mode)"), "split_pin": split_check,
            "mode": "dev-recording (audit guard active)" if dev else ("release-no-split (hidden/unknown recording: nothing to guard)" if split is None else "unknown recording with split"),
            "network": {"in_process_guard": active(), "note": "IP sockets and name resolution raise; an OS-level proof is the bwrap --unshare-net run in the receipt"},
            "output": {"csv": str(output), "rows": vcsv["rows"], "first_timestamp_ns": vcsv["first_timestamp_ns"], "last_timestamp_ns": vcsv["last_timestamp_ns"],
                       "sha256": sha256_file(output), "schema_validated": vcsv["schema_validated"], "jsonschema_version": vcsv["jsonschema_version"]},
            "diagnostics": str(diag_out), "runner_meta": str(work / "runner/run_meta.json"),
            "flagged_ticks": flagged,
            "latency_ms": meta["latency_ms"], "wall_s": {"total": round(time.perf_counter() - t_start, 3), "verify_and_setup": round(t_run - t_start, 3),
                                                          "runner": round(t_done - t_run, 3)},
            "environment": versions, "command": command,
            "overrides": {"set_params": dict(set_params) if set_params else {}, "config_override": sorted(config_override) if config_override else []},
        }
        rec_out.write_text(json.dumps(rec, indent=1) + "\n")
        return rec
    finally:
        if raw is not None and not keep_store:
            shutil.rmtree(raw["store"], ignore_errors=True)   # scratch store; the small export report stays in the workdir as evidence
            if tel_scratch is not None:
                shutil.rmtree(tel_scratch, ignore_errors=True)     # scratch telemetry export (the manifest copy stays next to the output)
