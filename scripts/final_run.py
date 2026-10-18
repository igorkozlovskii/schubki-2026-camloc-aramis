#!/usr/bin/env python3
"""Release driver of the frozen D-091 final model (final-v1): mc7 + closed-loop borders + poles (S2.10 'mc7BP', k_pole 32) + speed-aware v_max (S4.3 'dyn'),
initialised once from the supplied first pose. Ported from the research scripts/final_run.py (PRG-003 R02b); no parameter, constant or algorithm changed.

    pixi run camloc-run --mcap <bag.mcap> --first-pose <csv|json> --out <new output dir> [--assets <asset root>]
    python scripts/final_run.py run --mcap <bag.mcap> --first-pose <csv|json> --out <new dir>             # same, in the active environment
    python scripts/final_run.py run --store <exported store> --tel <telemetry-v1 dir> --first-pose <csv|json> --out ...   # skips the export

Asset root: --assets, else env CAMLOC_ASSETS, else <release>/assets/final-v1 (build with tools/build_assets.py or unpack the asset tarball; verify with tools/verify_assets.py).
The asset root mirrors the research layout (artifacts/, manifests/v2/, configs/models/, reports/experiments/, derived/, assets/final/); it is exported to the frozen driver modules as
CAMLOC_ASSET_ROOT (their ROOT). Output (--out) is a NEW directory; an existing pred.csv is refused.

`run` does, in this order (all inputs GT-free; the GT guard of S4.2 is installed and refuses ground-truth-class files):
  1 export   scripts/export_test_input.py (store with the file's own /tf_static calibration; exit 0 or 3 accepted) and telemetry:
             state_estimation allowed fields (scripts/export_telemetry.py; exit 3 = topic absent) -> /validation/twist+accel (scripts/v2_s41_telemetry.py) -> none (camera-only: NOT implemented here, refused).
  2 probes   seven E04 probes (fl = configs/models/e04-pnp-rfb-v1.yaml, six CamProbeModel, stride 2) through scripts/run_inference.py --mode continuous, GPU, one process per camera, sequential;
             rl/rr: if the store's calibration differs from the training calibration in the rl/rr extrinsics only, the S4.1 rule T_fitted_test = T_fitted_train inv(T_nominal_train) T_nominal_test is applied in the worker.
  3 streams  fl E04 rows + six meas-v1 parquets.   4 detect  S2.5 pole detections (fl fr cl cr).
  5 filter   v2_s210_lib.run('mc7BP') with the closed-loop LiveBorder (GPU), border fit started from the filter's own state; v_max rule from scripts/v2_s43_vmax.py ('dyn' by default).
The first pose is read once (t0, x, y, yaw) and handed to the filter and to the probes; nothing else carries a pose. Track bounds: <assets>/assets/final/track_bounds.json (hash-checked).
The research subcommands `parity` and `freeze` need research stores and are not part of the release.
GPU: the GPU stages need roughly 9 GB of free VRAM (measured); --wait-vram-mb N optionally waits until that much is free. JIT: no torch extension is compiled by the measured stages;
if one is, set TORCH_EXTENSIONS_DIR (default <out>/work/torch_extensions) and CUDA_HOME for your toolkit.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

CONFIG = ROOT / "configs/final/final-v1.yaml"
BOUNDS_SHA = "28cf2bfd9ce74c2e5f1ededd5d36866221c341609b27a15e3b5bae94b33de0ff"
MARGIN_NS = 1_000_000_000
ASSETS: Path = Path(os.environ.get("CAMLOC_ASSETS") or ROOT / "assets/final-v1")
BOUNDS = ASSETS / "assets/final/track_bounds.json"
REF_CALIB = ASSETS / "artifacts/releases/S2.9-release-mc7-v1/calibration/calib.json"


def install_audit():
    """Optional path audit (env CAMLOC_AUDIT_DIR): every file opened and every module imported by this process (and forked pool workers) is appended to <dir>/<pid>.paths."""
    d = os.environ.get("CAMLOC_AUDIT_DIR")
    if not d:
        return
    Path(d).mkdir(parents=True, exist_ok=True)
    seen: set = set()
    f = open(Path(d) / f"{os.getpid()}.paths", "a", buffering=1)
    pid = os.getpid()

    def hook(event, args):
        nonlocal f, pid
        if event not in ("open", "import"):
            return
        p = args[0] if event == "open" else (args[1] if len(args) > 1 else None)
        if not isinstance(p, (str, bytes, os.PathLike)):
            return
        p = os.path.abspath(os.fsdecode(p))
        if os.getpid() != pid:               # forked worker: own file
            pid = os.getpid()
            f = open(Path(d) / f"{pid}.paths", "a", buffering=1)
            seen.clear()
        if p not in seen:
            seen.add(p)
            f.write(f"{event}\t{p}\n")
    sys.addaudithook(hook)


def set_assets(path) -> Path:
    """Resolve the asset root once and export it to the frozen driver modules (their ROOT) and to child processes."""
    global ASSETS, BOUNDS, REF_CALIB
    ASSETS = Path(path).resolve()
    if not ASSETS.is_dir():
        raise SystemExit(f"REFUSED: asset root {ASSETS} is not a directory (build or unpack the asset bundle; see tools/verify_assets.py)")
    BOUNDS = ASSETS / "assets/final/track_bounds.json"
    REF_CALIB = ASSETS / "artifacts/releases/S2.9-release-mc7-v1/calibration/calib.json"
    os.environ["CAMLOC_ASSET_ROOT"] = str(ASSETS)
    os.environ["CAMLOC_ASSETS"] = str(ASSETS)
    return ASSETS


def asset(p) -> Path:
    """'$ASSETS/x' (release config) or research-relative 'artifacts/x' -> path under the asset root."""
    p = str(p)
    return ASSETS / (p[len("$ASSETS/"):] if p.startswith("$ASSETS/") else p)


def wait_vram(need_mb: int):
    """Optional: block until >= need_mb MiB of GPU memory is free (default off). No shared-machine locking is part of the release."""
    if not need_mb:
        return
    while True:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()
        if out and int(out[0]) >= need_mb:
            return
        print(f"waiting for {need_mb} MiB free VRAM (now {out[0] if out else 'unknown'})", flush=True)
        time.sleep(15)


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cfg_load() -> dict:
    import yaml
    return yaml.safe_load(CONFIG.read_text())


def check_bounds() -> dict:
    if sha(BOUNDS) != BOUNDS_SHA:
        raise SystemExit("REFUSED: assets/final/track_bounds.json changed")
    return json.loads(BOUNDS.read_text())


# ------------------------------------------------------------------------------------------------ first pose (read once)
def read_first_pose(path: Path) -> tuple:
    """csv (timestamp_ns,x,y,yaw) or first-pose-v1 json -> (t0_ns, x, y, yaw)."""
    p = Path(path)
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        return int(d["timestamp_ns"]), float(d["x"]), float(d["y"]), float(d["yaw"])
    row = next(csv.DictReader(open(p)))
    return int(row["timestamp_ns"]), float(row["x"]), float(row["y"]), float(row["yaw"])


# ------------------------------------------------------------------------------------------------ vmax rule and live border
def install_vmax(mode: str):
    import v2_s43_vmax as V
    V.install({"dyn": "dyn", "off": "const40"}[mode])


def patch_harness():
    """The GT-free harness of v2_s42_lib with the packaged track bounds instead of the store's own file."""
    import v2_s42_lib as L

    class FinalHarness(L.GFHarness):
        def __init__(self, store_dir):
            from camloc.contracts import CAMERAS
            from camloc.geometry.borders import BorderMap
            from camloc.geometry.camera import CameraModel
            from camloc.geometry.centerline import Centerline
            from camloc.io import CameraStore, load_static_metadata
            self.store = CameraStore(store_dir, verify=False)
            meta = load_static_metadata(self.store)
            self.bounds = check_bounds()
            self.bmap = BorderMap(self.bounds)
            self.cline = Centerline(self.bounds)
            self.cams = {c: CameraModel.from_static(c, meta.cameras[c]) for c in CAMERAS}
    L.GFHarness = FinalHarness


def run_filter(inp: dict, store_dir: Path, vmax: str) -> dict:
    """mc7BP closed loop on prepared inputs. -> result dict of v2_s210_lib.run plus the replay check and the border-fit rows."""
    import numpy as np
    import v2_s210_lib as Z
    import v2_s42_lib as L
    import v2_s25_filter as F25
    from camloc.geometry.borders import BorderMap
    from camloc.geometry.camera import CameraModel
    from camloc.io import CameraStore, load_static_metadata
    patch_harness()
    install_vmax(vmax)
    st = CameraStore(store_dir, verify=False)
    meta = load_static_metadata(st)
    Z._STATIC["v"] = ({k: CameraModel.from_static(k, meta.cameras[k]) for k in F25.CAMS}, BorderMap(check_bounds()))
    lb = L.make_live_border(store_dir, Z.border_sigma(), inp["ticks"])
    t = time.time()
    r = Z.run("mc7BP", inp, measure=lb.measure)
    r["filter_wall_s"] = time.time() - t
    r["fit_rows"], r["fit_meta"] = lb.rows, {"ckpt": lb.ckpt, "att_map": lb.map, "tag": lb.tag, "fit": lb.fit}
    del lb
    rp = Z.run("mc7BP", inp, border=r["border"])
    r["replay_maxdiff"] = float(max(np.abs(rp["x"] - r["x"]).max(), np.abs(rp["y"] - r["y"]).max(), np.abs(rp["yaw"] - r["yaw"]).max()))
    install_vmax("off")
    return r


# ------------------------------------------------------------------------------------------------ stage 1: export
def stage_export(mcap: Path, work: Path, log: dict) -> tuple:
    store, tel = work / "store", work / "tel"
    if not store.exists():
        p = subprocess.run([sys.executable, str(HERE / "export_test_input.py"), "--mcap", str(mcap), "--out", str(store)], cwd=work, capture_output=True, text=True)
        (work / "export_store.log").write_text(p.stdout + p.stderr)
        if p.returncode not in (0, 3):
            raise SystemExit(f"store export failed rc={p.returncode}: {p.stderr[-400:]}")
        log["store_export_rc"] = p.returncode
    if not tel.exists():
        p = subprocess.run([sys.executable, str(HERE / "export_telemetry.py"), "--input", str(mcap), "--out", str(tel)], cwd=work, capture_output=True, text=True)
        (work / "export_tel_stateest.log").write_text(p.stdout + p.stderr)
        if p.returncode == 0:
            log["telemetry_source"] = "state_estimation (allowed fields)"
        elif p.returncode == 3:
            p2 = subprocess.run([sys.executable, str(HERE / "v2_s41_telemetry.py"), "--mcap", str(mcap), "--out", str(tel)], cwd=work, capture_output=True, text=True)
            (work / "export_tel_validation.log").write_text(p2.stdout + p2.stderr)
            if p2.returncode != 0:
                raise SystemExit("telemetry: state_estimation absent and /validation/twist+accel adapter failed; camera-only fallback is not implemented in this driver "
                                 f"(rc={p2.returncode}): {p2.stderr[-300:]}")
            log["telemetry_source"] = "/validation/twist+accel (v2_s41_telemetry.py)"
        else:
            raise SystemExit(f"telemetry export failed rc={p.returncode}: {p.stderr[-400:]}")
    return store, tel


def calib_diff(store: Path) -> list:
    from camloc.io import calibfp
    ref = calibfp.fingerprints(REF_CALIB)["per_camera"]
    got = calibfp.fingerprints(store / "calib.json")["per_camera"]
    return sorted(c for c in ref if c in got and ref[c] != got[c])


# ------------------------------------------------------------------------------------------------ stage 2: probes
def write_first_pose_json(path: Path, fp: tuple):
    path.write_text(json.dumps({"schema": "first-pose-v1", "timestamp_ns": fp[0], "x": fp[1], "y": fp[2], "yaw": fp[3],
                                "provenance": {"contract": "first-pose-v1", "mode": "continuous", "supplied": True, "source": "final_run.py --first-pose"}}, indent=1))


def stage_probes(store: Path, work: Path, fp: tuple, end_ns: int, log: dict):
    import yaml
    import v2_s42_lib as L

    class Stub:
        audit = False
    pc = L.probe_configs(Stub())
    dif = calib_diff(store)
    log["calibration_differs"] = dif
    if set(dif) - {"rl", "rr"}:
        raise SystemExit(f"REFUSED: the store calibration differs from the training calibration in {dif}; only rl/rr extrinsics are handled (S4.1 rule)")
    (work / "configs").mkdir(exist_ok=True)
    fpj = work / "first_pose.json"
    write_first_pose_json(fpj, fp)
    cmds = []
    for rid, c in pc.items():
        out = work / "probes" / rid
        if out.exists():
            continue
        c = copy.deepcopy(c)
        c["assets"] = {**c.get("assets", {}), "map_dir": str(asset(c["assets"]["map_dir"])), "ext_dir": str(ASSETS / "artifacts/external")}   # absolute: the model joins ROOT / map_dir
        cp = work / "configs" / f"{rid}.yaml"
        cp.write_text(yaml.safe_dump(c, sort_keys=False))
        cam = rid.split("-")[0]
        cmds.append((rid, cam, [sys.executable, str(HERE / "final_run.py"), "probe-worker", "--mode", "continuous", "--start-ns", str(fp[0]), "--end-ns", str(end_ns),
                                "--store", str(store), "--first-pose", str(fpj), "--split", str(ROOT / "configs/splits/release-v1.yaml"), "--model-config", str(cp), "--out", str(out)]))
    if cmds:
        env = {**os.environ, "TORCH_EXTENSIONS_DIR": os.environ.get("TORCH_EXTENSIONS_DIR") or str(work / "torch_extensions")}
        t0 = time.time()
        with open(work / "probes.log", "a") as lf:
            for rid, cam, cmd in cmds:        # sequential, one process per camera (as in F1)
                lf.write(f"start {rid} {time.strftime('%T', time.gmtime())}\n")
                lf.flush()
                p = subprocess.run(cmd, cwd=work, stdout=lf, stderr=subprocess.STDOUT, env={**env, "FINAL_PROBE_CAM": cam})
                lf.write(f"end {rid} rc={p.returncode} {time.strftime('%T', time.gmtime())}\n")
                lf.flush()
                if p.returncode != 0:
                    raise SystemExit(f"probe stage failed ({rid}) rc={p.returncode}, see {work / 'probes.log'}")
        log["probes_wall_s"] = round(time.time() - t0, 1)
    for rid in pc:
        m = json.loads((work / "probes" / rid / "run_meta.json").read_text())
        if m.get("status") != "complete":
            raise SystemExit(f"probe {rid} not complete: {m.get('status')}")


def cmd_probe_worker(argv):
    """In-process run_inference with the S4.1 rl/rr extrinsic rule on E04PnpModel.initialize (only when FINAL_PROBE_CAM is rl or rr and the plain check refuses)."""
    import numpy as np
    from camloc.contracts import ContractError
    from camloc.models import pnp
    from camloc.models.retrieval import PriorState
    cam_name = os.environ.get("FINAL_PROBE_CAM", "")
    orig = pnp.E04PnpModel.initialize

    def init(self, first_pose, sm):
        if cam_name in ("", "fl"):
            return orig(self, first_pose, sm)
        try:
            return orig(self, first_pose, sm)
        except ContractError:
            # the plain check refuses (as for cl/cr/r on this store although their calibration fingerprints equal the training ones, checked in the main process): K, D were set before the check
            self.calibration_check = {"mode": "final_equal_fingerprint", "camera": cam_name}
            self.prior = PriorState(first_pose.x, first_pose.y, first_pose.yaw, first_pose.timestamp_ns)
            if cam_name not in ("rl", "rr"):
                return
            man = self.map.manifest
            Tn_tr = np.array(man["T_base_cam_nominal"], float)
            Tf_tr = np.array(man["T_base_cam_fitted"], float)
            Tn_te = np.asarray(sm.cameras["fl"].T_base_cam.matrix, float)
            self.T_base_cam = Tf_tr @ np.linalg.inv(Tn_tr) @ Tn_te
            self.calibration_check = {"mode": "s41_test_extrinsic", "camera": cam_name}
            self.prior = PriorState(first_pose.x, first_pose.y, first_pose.yaw, first_pose.timestamp_ns)
    pnp.E04PnpModel.initialize = init
    import run_inference
    return run_inference.main(argv)


# ------------------------------------------------------------------------------------------------ stages 3, 4: streams, detections
def stage_streams(store: Path, work: Path):
    import pandas as pd
    import v2_s42_lib as L
    from camloc.filtering import fusion_streams as fs
    from camloc.geometry.centerline import Centerline
    from camloc.mapping.map import load_map
    from camloc.v2.landmark_probe import rows_from_run, validate_meas_v1
    out = work / "streams"
    out.mkdir(exist_ok=True)
    rd = work / "probes" / "fl-e04"
    rows = fs.build_e04_stream(rd, "final:e04")
    fs.validate_rows(rows)
    fs.write_stream(rows, out / "fl.e04.jsonl.gz")
    cl = Centerline(check_bounds())
    for cam in L.CAMS6:
        rd = work / "probes" / f"{cam}-e04"
        meta = json.loads((rd / "run_meta.json").read_text())
        mp = load_map(ROOT / meta["capabilities"]["assets"][0], need_local=False)
        steps = [d for d in map(json.loads, open(rd / "diagnostics.jsonl")) if d["event"] == "step"]
        pred = pd.read_csv(rd / "predictions.csv", dtype={"timestamp_ns": "int64"})
        df = pd.DataFrame(rows_from_run(steps, pred, cam, "landmark", 0, cl, mp.kf_xy, mp.kf.kf_id.to_numpy(), 1.0))
        for c in ("stamp_ns", "avail_ns"):
            df[c] = df[c].astype("int64")
        df["lap"] = df["lap"].astype("int64")
        df["valid"] = df["valid"].astype(bool)
        bad = validate_meas_v1(df)
        assert not bad, bad
        df.to_parquet(out / f"{cam}_landmark.parquet", index=False)
        print("stream", cam, len(df), int(df.valid.sum()), flush=True)


def stage_detect(store: Path, work: Path, procs: int = 4):
    from multiprocessing import Pool
    import v2_s42_lib as L
    from camloc.filtering import fusion_streams as fs
    ticks = [r["t_ns"] for r in fs.read_stream(work / "streams" / "fl.e04.jsonl.gz")]
    with Pool(min(procs, len(L.POLE_CAMS))) as p:
        for r in p.imap_unordered(L._detect_job, [(str(store), c, ticks, str(work / "det")) for c in L.POLE_CAMS]):
            print("detect", *r, flush=True)


# ------------------------------------------------------------------------------------------------ inputs of the filter stage
def build_inputs(store: Path, tel: Path, work: Path, fp: tuple, constants_path: Path) -> dict:
    import numpy as np
    import pandas as pd
    import v2_s116_lib as M
    import v2_s210_lib as Z
    import v2_s42_lib as L
    from camloc.filtering import fusion_streams as fs
    from camloc.filtering.tel_runtime import NEEDED
    rows0 = fs.read_stream(work / "streams" / "fl.e04.jsonl.gz")
    ticks = np.array([r["t_ns"] for r in rows0], np.int64)
    t0, x, y, yaw = fp
    store_t = L.load_tel(tel, NEEDED, int(ticks[-1]) + MARGIN_NS, window=(t0 - MARGIN_NS, int(ticks[-1])))
    streams = {c: pd.read_parquet(work / "streams" / f"{c}_landmark.parquet") for c in L.CAMS6}
    assert all(set(d.stamp_ns) <= set(map(int, ticks)) for d in streams.values()), "extra-camera stamps must lie on the tick grid"
    kc = json.loads(constants_path.read_text())["kappa"]["all7"]
    pole_map = ASSETS / "artifacts/maps/V2-S2.5-poles-full"
    poles = Z.pole_stream_dir(work / "det", pole_map, ASSETS / "artifacts/v2/S1.1-masks/full")
    return {"name": "final", "lap": -1, "fold": None, "rows0": copy.deepcopy(rows0), "fp": (x, y, yaw), "t0": int(t0), "ticks": ticks, "store": store_t, "streams": streams,
            "kappa": {c: dict(kc[c]) for c in M.ALL7}, "poles": poles, "pole_map": pole_map}


def save_csv(path: Path, r: dict):
    import v2_s42_lib as L
    L.save_pred(path, r)


def cmd_run(a):
    import numpy as np
    import v2_s42_lib as L
    cfg = cfg_load()
    check_bounds()
    L.install_gt_guard()
    out = Path(a.out).resolve()
    if (out / "pred.csv").exists():
        raise SystemExit(f"REFUSED: {out / 'pred.csv'} exists")
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    log: dict = {"config_sha256": sha(CONFIG), "vmax": a.vmax, "assets": str(ASSETS)}
    wait_vram(a.wait_vram_mb)
    T = {"start": time.time()}
    fp = read_first_pose(a.first_pose)
    if a.mcap:
        t = time.time()
        store, tel = stage_export(Path(a.mcap), work, log)
        T["export_s"] = round(time.time() - t, 1)
    else:
        store, tel = Path(a.store), Path(a.tel)
        log["telemetry_source"] = f"supplied directory {tel}"
    from camloc.io import CameraStore
    with CameraStore(store, verify=False) as st:
        end_ns = int(st.stamps["fl"][-1])
        n_fl = len(st.stamps["fl"])
    log["store_track_bounds_sha256"] = sha(store / "track_bounds.json") if (store / "track_bounds.json").is_file() else None
    t = time.time()
    stage_probes(store, work, fp, end_ns, log)
    T["probes_s"] = round(time.time() - t, 1)
    t = time.time()
    stage_streams(store, work)
    T["streams_s"] = round(time.time() - t, 1)
    t = time.time()
    stage_detect(store, work)
    T["detect_s"] = round(time.time() - t, 1)
    inp = build_inputs(store, tel, work, fp, asset(cfg["constants"]["kappa"]))
    r = run_filter(inp, store, a.vmax)
    T["filter_s"] = round(r["filter_wall_s"], 1)
    save_csv(out / "pred.csv", r)
    np.savez_compressed(out / "pred.npz", ts=np.asarray(r["ts"], np.int64), x=r["x"], y=r["y"], yaw=r["yaw"], flags=r["flags"])
    n = len(r["ts"])
    T["total_s"] = round(time.time() - T.pop("start"), 1)
    dur = (int(r["ts"][-1]) - int(r["ts"][0])) * 1e-9
    summ = {"rows": n, "finite": bool(np.isfinite(r["x"]).all() and np.isfinite(r["y"]).all() and np.isfinite(r["yaw"]).all()), "span_s": dur, "ticks_expected_10hz": int(round(dur * 10)) + 1,
            "fl_frames": n_fl, "first_pose": fp, "timing_s": T, "filter_ms_per_tick": 1e3 * r["filter_wall_s"] / n, "total_ms_per_tick": 1e3 * T["total_s"] / n, "stats": r["stats"],
            "replay_maxdiff": r.get("replay_maxdiff"), "reanchors": int((r["flags"][:, 2] > 0).sum()), "fits": len(r["fit_rows"]), "fits_valid": sum(1 for x in r["fit_rows"] if x.get("valid")),
            "pred_sha256": sha(out / "pred.csv"), "log": log, "opened_paths": L.opened_paths()}
    (out / "run_summary.json").write_text(json.dumps(summ, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: summ[k] for k in ("rows", "finite", "span_s", "filter_ms_per_tick", "total_ms_per_tick", "pred_sha256", "reanchors")}, default=str))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    install_audit()
    if argv and argv[0] == "probe-worker":
        return cmd_probe_worker(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--mcap", type=Path)
    r.add_argument("--store", type=Path)
    r.add_argument("--tel", type=Path)
    r.add_argument("--first-pose", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--vmax", choices=("dyn", "off"), default="dyn")
    r.add_argument("--assets", type=Path, default=None, help="asset root (default: env CAMLOC_ASSETS, else <release>/assets/final-v1)")
    r.add_argument("--wait-vram-mb", type=int, default=0, help="optionally wait until this much GPU memory is free before the GPU stages (about 9000 needed)")
    a = ap.parse_args(argv)
    if a.cmd == "run" and not (a.mcap or (a.store and a.tel)):
        ap.error("run needs --mcap, or --store and --tel")
    set_assets(a.assets or ASSETS)
    return {"run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
