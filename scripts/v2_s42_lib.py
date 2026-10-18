"""PRG-002 S4.2 shared driver code: the seven declared configurations on ONE lap given its exported store, telemetry and first pose.  Research runner (D-075), NOT a release form.

Declared set (briefs/S4.2.md; D-076):  1 G1 | 2 mc7 | 3 mc7_fr | 4 mc7+borders+poles (S2.10 'mc7BP', k=32) | 5 mc7+borders (S2.10 'mc7B') | 6 telemetry+borders+poles (S2.8 'tbp', k=4) | 7 learned GRU on mc7.

Every accepted piece is imported read-only (v2_s116_lib2.run_arm2, v2_s210_lib.run, v2_s28_lib.TP, v2_s31e_prep.run_mc7 / model_inputs, v2_s31e_train.run_lap, v2_s31d_eval.postprocess);
this module only changes WHERE the inputs come from (an exported store + telemetry directory instead of cameras-v1 / S10a-laps0-5 / the S1.2 streams) and keeps GT out of the process:

* `install_gt_guard()` is an audit hook that REFUSES to open ground-truth-class files (derived/gt*, state_estimation, refline, images.parquet, cases, the raw dev recording, the dev store cameras-v1,
  the S10a dev telemetry, the audit token and ledger, the evaluator-side `*_landmark_eval` parquets) and records every other opened path under data/ derived/ artifacts/ configs/ manifests/.
* telemetry is read by `load_tel` with an explicit stamp limit (the lap end plus the export margin): the shared `tel_runtime.load_store` has no such parameter and refuses audit stamps (kept as is).
* all constants are numbers read from manifests (manifests/v2/S4.2-constants.json, written once by scripts/v2_s42_constants.py in a separate evaluator-side process from TRAINING laps 0/1/3/5).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("CAMLOC_ASSET_ROOT") or Path(__file__).resolve().parents[1])
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

RUNS = ROOT / "artifacts/v2/S4.2-runs"
CONSTANTS = ROOT / "manifests/v2/S4.2-constants.json"
MARGIN_NS = 1_000_000_000          # export margin on both sides of a lap (runbook section 2)
CAMS6 = ("fr", "cl", "cr", "r", "rl", "rr")
ALL7 = ("fl",) + CAMS6
CONFIGS = ("G1", "mc7", "mc7_bp", "mc7_bp_dyn", "tel_bp", "gru_f")       # final declared set (orchestrator, 2026-10-18); mc7_fr / mc7_b / S3.1e GRU stay in the code, not run
LABEL = {"G1": "G1 (fl landmark fixes + telemetry)", "mc7": "mc7 (S1.16 arm C, 7 cameras)", "mc7_fr": "mc7_fr (fl + fr)", "mc7_bp": "mc7 + borders + poles (k = 32)", "mc7_bp_dyn": "mc7 + borders + poles, speed-aware v_max (S4.3 dyn)", "gru_f": "learned GRU with recycling on mc7 + borders + poles (S3.1f release form)", "mc7_b": "mc7 + borders",
         "tel_bp": "telemetry + borders + poles (k = 4)", "gru": "learned GRU on mc7 (release form)"}

# ------------------------------------------------------------------------------------------------ GT guard
DENY = ("derived/gt", "derived/state_estimation", "derived/refline_", "derived/images.parquet", "derived/cases", "c0_2026-10-13_", "AUDIT_OPEN.token", "audit_ledger", "sn_map_state",
        "_landmark_eval", "artifacts/inputs/cameras-v1", "artifacts/telemetry/S10a", "split-v1/frames.csv", "S1.2-streams")
WATCH = ("/data/", "/derived/", "/artifacts/", "/configs/", "/manifests/", "/reports/", "/environments/")
_OPENED: set = set()
_GUARD = {"on": False}


def install_gt_guard():
    """Install once per process; returns the (mutable) set of opened paths."""
    if _GUARD["on"]:
        return _OPENED

    def hook(event, args):
        if event != "open":
            return
        p = args[0]
        if not isinstance(p, (str, bytes, os.PathLike)) or isinstance(p, int):
            return
        ap = os.path.abspath(os.fsdecode(p))
        rel = ap[len(str(ROOT)) + 1:] if ap.startswith(str(ROOT) + "/") else ap
        if any(d in ap for d in DENY) and not ap.endswith((".py", ".md")):
            raise PermissionError(f"S4.2 GT guard: refused to open {rel}")
        if any(w in ap for w in WATCH) and "/.pixi/" not in ap and "__pycache__" not in ap and not ap.endswith((".py", ".pyc", ".so")):
            _OPENED.add(rel)
    sys.addaudithook(hook)
    _GUARD["on"] = True
    return _OPENED


def opened_paths() -> list:
    return sorted(_OPENED)


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ lap configuration
def split_laps() -> dict:
    import yaml
    return {int(L["lap"]): L for L in yaml.safe_load((ROOT / "configs/splits/laps-v1.yaml").read_text())["laps"]}


class LapCfg:
    """One lap run: tag -> lap, role, input paths, output directory."""

    def __init__(self, tag: str):
        self.start_cond = "gt"                       # 'gt' = contract first pose (GT row at the lap start); 'est' = image-based relocalised first pose (D-084)
        base_tag = tag
        if tag.endswith("e") and tag != "dev4s":     # dev4e / aud6e / aud7e
            base_tag, self.start_cond = tag[:-1], "est"
        dev4 = (4, "dev", ROOT / "artifacts/inputs/S4.2-dev-lap4", ROOT / "artifacts/inputs/S4.2-dev-lap4-tel", ROOT / "artifacts/runs/S05-first-poses/lap4.json")
        spec = {"dev4": dev4, "dev4s": dev4,       # dev4s = the same lap with the STORED S1.16 / S07a probe streams as inputs (downstream-only check)
                "aud6": (6, "audit", ROOT / "artifacts/inputs/S4.2-audit-lap6", ROOT / "artifacts/inputs/S4.2-audit-lap6-tel", ROOT / "artifacts/v2/S4.2-audit-firstpose/lap6.json"),
                "aud7": (7, "audit", ROOT / "artifacts/inputs/S4.2-audit-lap7", ROOT / "artifacts/inputs/S4.2-audit-lap7-tel", ROOT / "artifacts/v2/S4.2-audit-firstpose/lap7.json")}
        spec = spec[base_tag]
        self.tag, self.lap, self.role, self.store, self.tel, self.fp_json = tag, *spec
        if self.start_cond == "est":
            self.fp_json = RUNS / tag / "firstpose.json"
        L = split_laps()[self.lap]
        assert L["role"] == self.role, (tag, L["role"])
        self.start_ns, self.end_ns = int(L["start_ns"]), int(L["end_ns"])
        self.audit = self.role == "audit"
        self.tel_limit_ns = self.end_ns + MARGIN_NS            # explicit stamp limit of the telemetry reads (the export range's upper end)
        self.out = RUNS / tag

    def path(self, *p) -> Path:
        return self.out.joinpath(*p)


def constants() -> dict:
    return json.loads(CONSTANTS.read_text())


# ------------------------------------------------------------------------------------------------ telemetry
def load_tel(directory, columns, limit_ns: int, window=None):
    """TelemetryStore of `columns` from <directory>/telemetry.csv; the header is checked default-deny as in camloc.models.e04t.load_telemetry; rows beyond `limit_ns` are refused."""
    import pandas as pd
    from camloc.io.telemetry import ALLOWED_FIELDS, TELEMETRY_MANIFEST_NAME, TELEMETRY_NAME, TelemetryError, TelemetryStore, assert_no_forbidden
    d = Path(directory)
    p = d / TELEMETRY_NAME
    header = list(pd.read_csv(p, nrows=0).columns)
    cols = [c for c in header if c != "stamp_ns"]
    assert_no_forbidden(cols)
    unknown = [c for c in cols if c not in ALLOWED_FIELDS]
    if unknown:
        raise TelemetryError(f"telemetry columns outside the telemetry-v1 allowlist: {unknown}")
    missing = [c for c in columns if c not in cols]
    if missing:
        raise TelemetryError(f"telemetry.csv lacks required fields {missing}")
    df = pd.read_csv(p, usecols=["stamp_ns", *columns], dtype={"stamp_ns": "int64"})
    st = df.stamp_ns.to_numpy(dtype=np.int64)
    if len(st) and int(st.max()) > int(limit_ns):
        raise TelemetryError(f"telemetry rows beyond the declared limit {limit_ns}")
    if window is not None:
        keep = (st >= window[0]) & (st <= window[1])
        df, st = df[keep], st[keep]
    man = json.loads((d / TELEMETRY_MANIFEST_NAME).read_text()) if (d / TELEMETRY_MANIFEST_NAME).is_file() else {}
    return TelemetryStore(st, {c: df[c].to_numpy(dtype=np.float64) for c in columns}, man, TelemetryStore.DEFAULT_STALE_NS)


def first_pose(cfg: LapCfg):
    d = json.loads(cfg.fp_json.read_text())
    assert d["schema"] == "first-pose-v1" and d["provenance"]["lap"] == cfg.lap and d["provenance"]["mode"] == "episode"
    return (float(d["x"]), float(d["y"]), float(d["yaw"])), int(d["timestamp_ns"])


# ------------------------------------------------------------------------------------------------ stage: probes (GPU, run_inference.py episode runs under the lock)
CAM_PARAMS = {"top_k": 6, "lost_s": 5.0, "prior_base_m": 40.0, "vmax_mps": 45.0, "min_inliers": 60, "min_inlier_ratio": 0.2, "max_rms_px": 2.5, "ransac_px": 3.0, "fallback": "retrieval"}
FULL = {"fl": "artifacts/maps/S04-full-v1", "fr": "artifacts/maps/S04d-fr-full-v1", **{c: f"artifacts/maps/V2-S1.16-{c}-full" for c in ("cl", "cr", "r", "rl", "rr")}}
AUDIT_REASON = "S4.2, D-076, Igor: score current best once"


def probe_configs(cfg: LapCfg) -> dict:
    """{run id: (config dict, config path)}: the frozen E04 fl model (configs/models/e04-pnp-rfb-v1.yaml, unchanged params) and the six CamProbeModel probes of S1.16 (stride 2, full-train maps)."""
    import yaml
    base = yaml.safe_load((ROOT / "configs/models/e04-pnp-rfb-v1.yaml").read_text())
    out = {}
    fl = copy.deepcopy(base)
    fl["audit_opened"], fl["audit_reason"] = (True, AUDIT_REASON) if cfg.audit else (False, None)
    out["fl-e04"] = fl
    for cam in CAMS6:
        out[f"{cam}-e04"] = {"schema_version": 1, "model_id": f"v2s42-e04-{cam}-full", "class": "camloc.v2.landmark_probe:CamProbeModel", "assets": {"map_dir": FULL[cam]},
                             "params": {**CAM_PARAMS, "camera": cam, "stride": 2, "pipeline": "e04"}, "audit_opened": bool(cfg.audit), "audit_reason": AUDIT_REASON if cfg.audit else None}
    return out


def stage_probes(cfg: LapCfg, cams=None) -> int:
    import subprocess
    import time
    import yaml
    cd = cfg.path("configs")
    cd.mkdir(parents=True, exist_ok=True)
    cmds, rids = [], []
    for rid, c in probe_configs(cfg).items():
        if cams and rid.split("-")[0] not in cams:
            continue
        out = cfg.path("probes", rid)
        if out.exists():
            print("skip", rid)
            continue
        cp = cd / f"{rid}.yaml"
        cp.write_text(yaml.safe_dump(c, sort_keys=False))
        cmds.append(f"echo start {rid} $(date -u +%T) load $(cat /proc/loadavg); pixi run --manifest-path environments/vision/pixi.toml python scripts/run_inference.py --mode episode --lap {cfg.lap} "
                    f"--store {cfg.store.relative_to(ROOT)} --first-pose {cfg.fp_json.relative_to(ROOT)} --model-config {cp.relative_to(ROOT)} --out {out.relative_to(ROOT)}; echo end {rid} rc=$? $(date -u +%T)")
        rids.append(rid)
    if not cmds:
        return 0
    env = {**os.environ, "TORCH_EXTENSIONS_DIR": "/tmp/torch_extensions_v2s42", "CONDA_OVERRIDE_CUDA": "12.4"}
    (cfg.out / "logs").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(cfg.path("logs", "probes.log"), "a") as lf:
        p = subprocess.run(["/usr/bin/time", "-v", "scripts/gpu_slot.sh", "--need-mb", "6000", "--", "bash", "-c", "; ".join(cmds)], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env)
    print("probes exit", p.returncode, f"wall {time.time() - t0:.0f}s", flush=True)
    return p.returncode


def stage_streams(cfg: LapCfg):
    """probe run dirs -> the fl E04 rows (fusion-stream-v1) and the six meas-v1 landmark parquets. No GT."""
    import pandas as pd
    from camloc.filtering import fusion_streams as fs
    from camloc.geometry.centerline import Centerline
    from camloc.io.store import CameraStore
    from camloc.mapping.map import load_map
    from camloc.v2.landmark_probe import rows_from_run, validate_meas_v1
    rd = cfg.path("probes", "fl-e04")
    meta = json.loads((rd / "run_meta.json").read_text())
    assert meta["status"] == "complete" and meta["lap"] == cfg.lap
    rows = fs.build_e04_stream(rd, f"{cfg.tag}:e04")
    fs.validate_rows(rows)
    (cfg.out / "streams").mkdir(parents=True, exist_ok=True)
    fs.write_stream(rows, cfg.path("streams", "fl.e04.jsonl.gz"))
    cl = Centerline(CameraStore(cfg.store, verify=False).track_bounds())
    for cam in CAMS6:
        rd = cfg.path("probes", f"{cam}-e04")
        meta = json.loads((rd / "run_meta.json").read_text())
        assert meta["status"] == "complete" and meta["lap"] == cfg.lap
        mp = load_map(ROOT / meta["capabilities"]["assets"][0], need_local=False)
        steps = [d for d in map(json.loads, open(rd / "diagnostics.jsonl")) if d["event"] == "step"]
        pred = pd.read_csv(rd / "predictions.csv", dtype={"timestamp_ns": "int64"})
        df = pd.DataFrame(rows_from_run(steps, pred, cam, "landmark", cfg.lap, cl, mp.kf_xy, mp.kf.kf_id.to_numpy(), 1.0))
        for c in ("stamp_ns", "avail_ns"):
            df[c] = df[c].astype("int64")
        df["lap"] = df["lap"].astype("int64")
        df["valid"] = df["valid"].astype(bool)
        bad = validate_meas_v1(df)
        assert not bad, bad
        df.to_parquet(cfg.path("streams", f"{cam}_landmark.parquet"), index=False)
        print(cfg.tag, cam, len(df), int(df.valid.sum()), flush=True)


# ------------------------------------------------------------------------------------------------ inputs of the CPU stages
def load_inputs(cfg: LapCfg, with_poles: bool = False) -> dict:
    """GT-free inputs of one lap, in the dict form of v2_s210_lib.load_input (so Z.run / M2.run_arm2 take them unchanged)."""
    import pandas as pd
    import v2_s116_lib as M
    from camloc.filtering import fusion_streams as fs
    from camloc.filtering.tel_runtime import NEEDED
    rows0 = fs.read_stream(cfg.path("streams", "fl.e04.jsonl.gz"))
    fp, t0 = first_pose(cfg)
    ticks = np.array([r["t_ns"] for r in rows0], np.int64)
    store = load_tel(cfg.tel, NEEDED, cfg.tel_limit_ns, window=(cfg.start_ns - MARGIN_NS, cfg.end_ns))
    streams = {c: pd.read_parquet(cfg.path("streams", f"{c}_landmark.parquet")) for c in CAMS6}
    assert all(set(d.stamp_ns) <= set(map(int, ticks)) for d in streams.values()), "extra-camera stamps must lie on the tick grid"
    kc = constants()["kappa"]
    inp = {"name": cfg.tag, "lap": cfg.lap, "fold": None, "rows0": copy.deepcopy(rows0), "fp": fp, "t0": int(t0), "ticks": ticks, "store": store, "streams": streams,
           "kappa": {c: dict(kc["all7"][c]) for c in M.ALL7}, "poles": None, "pole_map": ROOT / "artifacts/maps/V2-S2.5-poles-full"}
    if with_poles:
        inp["poles"] = pole_stream(cfg)
    return inp


# ------------------------------------------------------------------------------------------------ poles
POLE_CAMS = ("fl", "fr", "cl", "cr")
SYNC_NS, MAX_AGE_NS = 1_000_000, 150_000_000


def _detect_job(a):
    import time
    import cv2
    import pandas as pd
    from camloc.geometry.camera import CameraModel
    from camloc.io import CameraStore, load_static_metadata
    from camloc.v2.poles.detect import COLUMNS, DetConfig, detect_vertical
    store, cam, ticks, out = a
    cv2.setNumThreads(1)
    st = CameraStore(store, verify=False)
    meta = load_static_metadata(st)
    cm = CameraModel.from_static(cam, meta.cameras[cam])
    t0, rows, n = time.time(), [], 0
    for tick in ticks:
        i = int(np.searchsorted(st.stamps[cam], int(tick) + SYNC_NS, side="right")) - 1
        if i < 0 or int(tick) + SYNC_NS - int(st.stamps[cam][i]) > MAX_AGE_NS:
            continue
        n += 1
        img = cv2.imdecode(np.frombuffer(st.jpeg(cam, i), np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
        for r in detect_vertical(img, cm, DetConfig()):
            rows.append((int(tick), int(st.stamps[cam][i]), *r))
    df = pd.DataFrame(rows, columns=["tick_ns", "img_ns", *COLUMNS])
    df["n_frames"] = n
    Path(out).mkdir(parents=True, exist_ok=True)
    df.to_parquet(Path(out) / f"lap2_{cam}.parquet", index=False)         # file name 'lap2_*' is the S2.5/S2.8 schema's name, kept for the shared readers
    return cam, n, len(df), round(time.time() - t0, 1)


def stage_detect(cfg: LapCfg, procs: int = 4):
    """S2.5 vertical-segment detector (unchanged) on the lap store, cameras fl fr cl cr, at the ticks of the E04 stream; same schema as artifacts/v2/S2.8-det."""
    from multiprocessing import Pool
    from camloc.filtering import fusion_streams as fs
    ticks = [r["t_ns"] for r in fs.read_stream(cfg.path("streams", "fl.e04.jsonl.gz"))]
    out = str(cfg.path("det"))
    with Pool(min(procs, len(POLE_CAMS))) as p:
        for r in p.imap_unordered(_detect_job, [(str(cfg.store), c, ticks, out) for c in POLE_CAMS]):
            print(*r, flush=True)


def pole_stream(cfg: LapCfg, det_dir=None, masks_dir=None, pole_map=None) -> list:
    import v2_s210_lib as Z
    return Z.pole_stream_dir(Path(det_dir or cfg.path("det")), Path(pole_map or ROOT / "artifacts/maps/V2-S2.5-poles-full"), Path(masks_dir or ROOT / "artifacts/v2/S1.1-masks/full"))


def static_cams(store_dir):
    """(camera models of the pole cameras, BorderMap) from a store's own calib.json / track_bounds.json (the cache of v2_s210_lib.static_cams is filled with these)."""
    from camloc.geometry.borders import BorderMap
    from camloc.geometry.camera import CameraModel
    from camloc.io import CameraStore, load_static_metadata
    import v2_s25_filter as F25
    st = CameraStore(store_dir, verify=False)
    meta = load_static_metadata(st)
    return {k: CameraModel.from_static(k, meta.cameras[k]) for k in F25.CAMS}, BorderMap(meta.track_bounds)


# ------------------------------------------------------------------------------------------------ border fits (GPU, vision env): closed loop (configs 4, 5) and open loop from poses (config 6)
class GpuLock:
    need_mb = 6000

    def __enter__(self):
        """Shared-slot mode (D-079, the logic of scripts/gpu_slot.sh: one of 3 slot locks /tmp/camloc-gpu-slot.<i> AND need_mb + 2048 MiB free VRAM); S42_GPU=exclusive uses the old global lock."""
        import fcntl
        import subprocess
        import time
        t = time.time()
        if os.environ.get("S42_GPU") == "exclusive":
            self.f = open("/tmp/camloc-gpu.lock", "a")
            fcntl.flock(self.f, fcntl.LOCK_EX)
        else:
            self.f = None
            while self.f is None:
                free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
                if free >= self.need_mb + 2048:
                    for i in (1, 2, 3):
                        f = open(f"/tmp/camloc-gpu-slot.{i}", "a")
                        try:
                            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            self.f = f
                            print(f"gpu slot {i}, free {free} MiB", flush=True)
                            break
                        except OSError:
                            f.close()
                if self.f is None:
                    time.sleep(10)
        self.waited = time.time() - t
        print(f"GPU acquired after {self.waited:.0f} s", flush=True)
        return self

    def __exit__(self, *a):
        import fcntl
        fcntl.flock(self.f, fcntl.LOCK_UN)


class GFHarness:
    """The attributes of v2_s11_common.Harness the border fitters use (store, cline, bmap, cams, camera_frame, context) WITHOUT loading GT or the split frame table."""

    def __init__(self, store_dir):
        from camloc.contracts import CAMERAS
        from camloc.geometry.borders import BorderMap
        from camloc.geometry.camera import CameraModel
        from camloc.geometry.centerline import Centerline
        from camloc.io import CameraStore, load_static_metadata
        self.store = CameraStore(store_dir, verify=False)
        meta = load_static_metadata(self.store)
        self.bounds = meta.track_bounds
        self.bmap = BorderMap(meta.track_bounds)
        self.cline = Centerline(meta.track_bounds)
        self.cams = {c: CameraModel.from_static(c, meta.cameras[c]) for c in CAMERAS}

    def camera_frame(self, cam, tick_ns, max_age_ns=150_000_000):
        st = self.store.stamps[cam]
        i = int(np.searchsorted(st, tick_ns, side="right")) - 1
        if i < 0 or tick_ns - st[i] > max_age_ns:
            return None, None
        return i, int(st[i])

    def context(self, offsets_yaml, mask_dir, cfg=None):
        from camloc.v2 import body_masks
        from camloc.v2.border_probe import ProbeConfig, ProbeContext
        assert offsets_yaml is None
        return ProbeContext(self.bmap, self.cline, dict(self.cams), body_masks.load_masks(mask_dir), cfg or ProbeConfig())


def make_live_border(store_dir, sigma, ticks, tag="full"):
    """v2_s210_fit.LiveBorder with the GT-free harness (same fit code: __init__ is the only difference)."""
    import cv2
    import torch
    from concurrent.futures import ThreadPoolExecutor
    import v2_s210_fit as F
    import v2_s32c_run as R3
    from camloc.v2.border_fast import FastMeasurer
    from camloc.v2.border_seg_att import AttTracker
    from camloc.v2.road_attitude import RoadAttitudeMap
    from camloc.v2.seg import labels as SL
    from camloc.v2.seg.infer import SegPredictor

    class LiveBorderGF(F.LiveBorder):
        def __init__(self):
            cv2.setNumThreads(1)
            self.h = GFHarness(store_dir)
            self.polys = SL.build_road_polys(self.h.bounds)
            n = F.net_for(tag)
            self.cfg = R3.make_cfg(F.FIT)
            self.fit = dict(F.FIT)
            self.meas = FastMeasurer(self.h.context(None, n["masks"]), self.polys, self.cfg, device="cuda", graph=True, prune=1e-6)
            self.pred = SegPredictor.from_checkpoint(str(n["ckpt"]), "cuda")
            self.ram = RoadAttitudeMap.load(str(n["map"]))
            self.tracker = AttTracker(self.cfg, True, sd0=self.cfg.sd0_att_map)
            self.pool = ThreadPoolExecutor(6)
            self.device = "cuda"
            self.ticks = np.asarray(ticks, np.int64)
            self.sigma = float(sigma)
            self.s_hint = None
            self.rows = []
            self.ckpt, self.map, self.tag = str(n["ckpt"]), str(n["map"]), tag
    del torch
    return LiveBorderGF()


def border_openloop(store_dir, start_csv, out_dir, rate_hz: int = 2, max_ticks: int = 0):
    """S1.11 border stream (S3.2e-B network, full-train attitude map and masks, temp 20000 / sd_dpsi 1.5) from a prediction CSV's own poses, 2 Hz: the GT-free port of scripts/v2_s111_border.py
    (scripts/v2_s41_poles.py cmd_border). Takes the GPU lock itself. -> <out_dir>/lap2_multi.parquet"""
    import math
    import time
    import pandas as pd
    import torch
    from concurrent.futures import ThreadPoolExecutor
    import v2_s111_border as B
    import v2_s32d_lib as D
    import v2_s32c_run as R3
    from camloc.geometry.transforms import wrap_angle
    from camloc.v2.border_probe import MEAS_COLUMNS, BorderMeas, meas_row
    from camloc.v2.border_seg_att import AttTracker
    from camloc.v2.border_fast import FastMeasurer
    from camloc.v2.road_attitude import RoadAttitudeMap
    from camloc.v2.seg import labels as SL
    from camloc.v2.seg.infer import SegPredictor, decode_jpeg
    from camloc.v2.meas import write_meas
    h = GFHarness(store_dir)
    ctx = h.context(None, B.MASKS)
    polys = SL.build_road_polys(h.bounds)
    cfg = R3.make_cfg(B.FIT)
    pr = pd.read_csv(start_csv)
    ts, fx, fy, fyaw = (pr[k].to_numpy() for k in ("timestamp_ns", "x", "y", "yaw"))
    ts = ts.astype(np.int64)
    fs_ = np.empty(len(ts))
    hint = None
    for i in range(len(ts)):
        s0, n0, th0, _ = h.cline.project(fx[i], fy[i], s_hint=hint)
        fs_[i] = s0
        hint = s0
    sel = np.arange(0, len(ts), 10 // rate_hz)
    if max_ticks:
        sel = sel[:max_ticks]
    rows = []
    t_all = time.time()
    with GpuLock():
        meas = FastMeasurer(ctx, polys, cfg, device="cuda", graph=True, prune=1e-6)
        pred = SegPredictor.from_checkpoint(str(B.CKPT["s32e"]), "cuda")
        ram = RoadAttitudeMap.load(str(B.MAP))
        pool = ThreadPoolExecutor(6)
        tracker = AttTracker(cfg, True, sd0=cfg.sd0_att_map)
        for i in sel:
            t = int(ts[i])
            t_start = time.perf_counter()
            j = max(i - 1, 0)
            dt_f = max((ts[i] - ts[j]) * 1e-9, 1e-3) if i > 0 else None
            vel = np.zeros(4) if dt_f is None else np.array([fx[i] - fx[j], fy[i] - fy[j], B.wrapd(fyaw[i] - fyaw[j]), fs_[i] - fs_[j]]) / dt_f
            fr = {}
            for c in B.CAMS:
                idx, st = h.camera_frame(c, t, B.MAX_AGE_NS)
                if idx is not None:
                    fr[c] = (idx, st)
            ref_st = t
            if fr:
                cams = list(fr)
                imgs = list(pool.map(lambda c: decode_jpeg(h.store.jpeg(c, fr[c][0])), cams))
                x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().to("cuda")
                with torch.no_grad(), torch.autocast("cuda", enabled=pred.amp):
                    pp = torch.sigmoid(pred.model(x).float())[:, 0]
                items = {}
                for q_, c in enumerate(cams):
                    dtc = (t - fr[c][1]) * 1e-9
                    items[c] = {"prob": pp[q_], "pose0": (fx[i] - vel[0] * dtc, fy[i] - vel[1] * dtc, fyaw[i] - vel[2] * dtc), "s_init": fs_[i] - vel[3] * dtc, "attitude": None}
                ref = "fl" if "fl" in items else cams[0]
                ref_st = fr[ref][1]
                dz_, dpi_, dro_ = ram.lookup(items[ref]["s_init"])
                td = np.array([dro_, dpi_, dz_])
                mm = meas.measure_att(items, ref=ref, att_prior=tracker.prior(t, td), fit_att=True)
                if mm.valid:
                    tracker.update(t, [mm.q["droll"], mm.q["dpitch"], mm.q["dz"]], [mm.q["sd_roll"], mm.q["sd_pitch"], mm.q["sd_z"]], td)
                    s0, n0, th0, _ = h.cline.project(fx[i], fy[i], s_hint=fs_[i])
                    nx, ny = -math.sin(th0), math.cos(th0)
                    x2, y2, yaw2 = fx[i] + mm.q["dn_fit"] * nx, fy[i] + mm.q["dn_fit"] * ny, fyaw[i] + mm.q["dpsi_fit"]
                    s2, n2, th2, _ = h.cline.project(x2, y2, s_hint=s0)
                    m = BorderMeas(True, "", n=float(n2), dpsi=float(wrap_angle(yaw2 - th2)), s=float(s2), x=float(x2), y=float(y2), yaw=float(wrap_angle(yaw2)), q=mm.q)
                else:
                    m = BorderMeas(False, mm.reason, q=mm.q)
                m.q["n_cams"] = float(len(cams))
            else:
                m = BorderMeas(False, "no_frame", q={"n_cams": 0})
            m.q["service_ms"] = 1e3 * (time.perf_counter() - t_start)
            r = meas_row(t, 2, "multi", D.METHOD, m, ref_st, float("nan"), float("nan"), "full")
            r["init"] = "pred_stream"
            r["q_n_points"] = float(m.q.get("n_cams", np.nan))
            r["tick_i"] = int(i)
            rows.append(r)
    st = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_meas(st[MEAS_COLUMNS], out / "lap2_multi.parquet")
    info = {"net": "s32e", "ckpt": str(B.CKPT["s32e"]), "map": str(B.MAP), "masks": str(B.MASKS), "fit": B.FIT, "rate_hz": rate_hz, "start": str(start_csv), "wall_s": time.time() - t_all,
            "n_ticks": len(st), "valid": int(st.valid.sum()), "service_ms_median": float(st.q_service_ms.median())}
    (out / "run.json").write_text(json.dumps(info, indent=1))
    return info


# ------------------------------------------------------------------------------------------------ configurations 1-5 (CPU / GPU)
def save_pred(path: Path, r: dict):
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp_ns": np.asarray(r["ts"], np.int64), "x": np.round(r["x"], 6), "y": np.round(r["y"], 6), "yaw": np.round(r["yaw"], 6)}).to_csv(path, index=False, float_format="%.6f")


def run_arm_cfg(name: str, cfg: LapCfg, inp: dict):
    """configs 1-3 (CPU): v2_s116_lib2.run_arm2 with explicit kappa (numbers). -> result dict of the filter."""
    import v2_s116_lib as M
    import v2_s116_lib2 as M2
    arm = {"G1": "G1", "mc7": "C", "mc7_fr": "C_fr"}[name]
    cams = M.ARMS[arm][0]
    kc = constants()["kappa"]
    kappa = {c: dict(kc["all7"][c]) for c in cams} if arm == "C" else ({c: dict(kc["fr"][c]) for c in cams} if arm == "C_fr" else {c: {"kxy": 1.0, "kyaw": 1.0} for c in cams})
    data = (inp["rows0"], inp["fp"], inp["t0"], inp["ticks"], inp["store"])
    return M2.run_arm2(arm, cfg.lap, "zero", fold=None, kappa=kappa, data=data, streams={c: inp["streams"][c] for c in cams if c != "fl"})


def run_bp(name: str, cfg: LapCfg, inp: dict):
    """configs 4, 5 (GPU for the live border fits): v2_s210_lib.run with the closed-loop border of the GT-free LiveBorder; the stored border rows are replayed and compared (as S2.10)."""
    import v2_s210_lib as Z
    arm = {"mc7_b": "mc7B", "mc7_bp": "mc7BP", "mc7_bp_dyn": "mc7BP"}[name]
    if name.endswith("_dyn"):                       # S4.3 adopted bound: v_max = max(40, 1.3 x max telemetry speed so far); in-process patch of TelFilter.step (own process per run)
        import v2_s43_vmax
        v2_s43_vmax.install("dyn")
    Z._STATIC["v"] = static_cams(cfg.store)
    with GpuLock():
        lb = make_live_border(cfg.store, Z.border_sigma(), inp["ticks"])
        r = Z.run(arm, inp, measure=lb.measure)
        fit_rows, meta = lb.rows, {"ckpt": lb.ckpt, "att_map": lb.map, "tag": lb.tag, "fit": lb.fit}
        del lb
    rp = Z.run(arm, inp, border=r["border"])
    r["replay_maxdiff"] = float(max(np.abs(rp["x"] - r["x"]).max(), np.abs(rp["y"] - r["y"]).max(), np.abs(rp["yaw"] - r["yaw"]).max()))
    r["fit_rows"], r["fit_meta"] = fit_rows, meta
    return r


# ------------------------------------------------------------------------------------------------ configuration 6: telemetry + borders + poles (S2.8 tbp)
def run_tbp(cfg: LapCfg, start_csv, border_parquet, det_dir=None, store_dir=None, tel=None, fp=None, t0=None, tel_limit=None):
    """v2_s28_lib.build(cfg='tbp') with explicit inputs: start poses/ticks = `start_csv` (the G1 poses, as S2.8 used the A' poses), border = rho form (S2.8), poles = S2.5 detections,
    k = k_tbp. Telemetry through `load_tel` (explicit limit). -> (ticks, X (n,3), mode counts, availability)."""
    import math
    import pandas as pd
    import v2_s28_lib as Lb
    from camloc.filtering.tel4_filter import Reanchor4
    from camloc.filtering.tel_filter import TelParams
    from camloc.filtering.tel_runtime import NEEDED
    from camloc.models.e04t import build_stream, resolve_filter  # noqa: F401
    store_dir = store_dir or cfg.store
    c = Lb.consts()
    f = Lb.filter_block()
    b = f["blocks"]
    if fp is None:
        fp, t0 = first_pose(cfg)
    tstore = load_tel(tel or cfg.tel, NEEDED, tel_limit or cfg.tel_limit_ns, window=(t0 - MARGIN_NS, (tel_limit or cfg.tel_limit_ns)))
    s = build_stream(f, fp, t0, tstore, 0.0, 1e6)
    tp, rp = TelParams(**b["tel"]), Reanchor4(**b[f["extra"]["rp"]["from"]])
    pr = pd.read_csv(start_csv)
    ticks = pr.timestamp_ns.to_numpy(np.int64)
    info = {"cfg": "tbp", "n_ticks": int(len(ticks)), "duration_s": float((ticks[-1] - ticks[0]) * 1e-9 + 0.1)}
    bd = pd.read_parquet(border_parquet)
    bd = bd[bd.valid].sort_values("avail_ns")
    sg = c["border_sig0"] / math.sqrt(1.0 - c["rho"] ** 2)
    border = [{"stamp": int(st_), "avail": int(av), "s": float(ss), "z": float(n), "sig": sg, "inc": False} for st_, av, ss, n in zip(bd.stamp_ns, bd.avail_ns, bd.s, bd.n)]
    info["border_rows"] = len(border)
    cams, bmap = static_cams(store_dir)
    lm = pd.read_parquet(Lb.POLE_MAP / "landmarks.parquet")
    off = json.loads((Lb.POLE_MAP / "map.json").read_text()).get("cam_az_offset_rad", {})
    Lb.DET["_s42"] = Path(det_dir or cfg.path("det"))
    poles = Lb.pole_stream("_s42", off)
    k = float(c["k_tbp"])
    s.filter = Lb.TP(tp, fp, t0, rp, border, poles, lm, cams, bmap, k_sig=k)
    info["k_sig"] = k
    ts, X, modes = Lb.drive(s, ticks)
    info["availability"] = Lb.availability(s, info)
    info["mode_counts"] = pd.Series(modes).value_counts().to_dict()
    return {"ts": ts, "x": X[:, 0], "y": X[:, 1], "yaw": X[:, 2], "info": info}


# ------------------------------------------------------------------------------------------------ configuration 7: learned GRU on mc7, release form (vision env, torch, CPU)
def gru_inputs(cfg: LapCfg, inp: dict) -> tuple:
    """-> (X bank dict, mc7 dict) of the lap, built like v2_s31e_prep.build_dev (set 'all', kappa from the four training laps, no side channel): model_inputs + run_mc7. No GT."""
    import v2_s116_lib as M
    import v2_s13_lib as L13
    import v2_s31e_prep as P31
    from camloc.filtering.tel_runtime import NEEDED
    from camloc.v2.learned.features import RAW_FIELDS
    tr = L13.track()
    st_raw = load_tel(cfg.tel, RAW_FIELDS, cfg.tel_limit_ns, window=(cfg.start_ns - MARGIN_NS, cfg.end_ns))
    st_g1 = inp["store"]
    X = P31.model_inputs(inp["ticks"], inp["t0"], st_raw, inp["rows0"], inp["streams"], {"bd": None}, tr)
    mc = P31.run_mc7(inp["rows0"], inp["streams"], "all", inp["kappa"], inp["fp"], inp["t0"], inp["ticks"], st_g1, tr)
    del NEEDED, M
    return X, mc, tr


def gru_apply(X: dict, mc: dict, tr, core: str, runs_dir: Path, device="cpu") -> dict:
    """The 4 fold models of the chosen core on the lap's inputs (closed loop, 'cam' channels), mean over folds, then the v1d run-time rule with mc7 as the default
    (= v2_s31e_eval.dev). Returns the release-form dict."""
    import torch
    import v2_s31e_train as TR
    from camloc.v2.learned.v1d import FallbackCfg, postprocess
    torch.set_num_threads(2)
    dev = torch.device(device)
    z = {k: np.asarray(v) for k, v in X.items()}
    d, _ = TR._tensors(z, mc, dev, False)
    preds = []
    for fold in (0, 1, 3, 5):
        rd = Path(runs_dir) / f"v1e_{core}_f{fold}_s0"
        m, c = TR.load_model(rd / "ckpt.pt", dev)
        preds.append(TR.run_lap(m, d, TR.CONDS["cam"], dev))
    ticks = np.asarray(X["ticks"], np.int64)
    e = {"ts": ticks, "x": np.mean([p["x"] for p in preds], 0), "y": np.mean([p["y"] for p in preds], 0), "yaw": np.angle(np.mean([np.exp(1j * p["yaw"]) for p in preds], 0)),
         "logsig": np.log(np.mean([np.exp(p["logsig"]) for p in preds], 0)), "n": np.mean([p["n"] for p in preds], 0), "d": np.mean([p["d"] for p in preds], 0)}
    import v2_s31e_eval as EV
    lm = np.zeros(len(ticks), np.float32)
    for c_ in EV.LM_CAMS:
        lm = np.maximum(lm, X[f"{c_}_pres"])
    full = postprocess({k: mc[k] for k in mc}, e, lm, X["dt"], tr, FallbackCfg())
    return {"full": full, "raw": e, "folds": preds}


# ------------------------------------------------------------------------------------------------ configuration 6': S3.1f GRU with recycling on mc7 + borders + poles (release form)
def gruf_inputs(cfg: LapCfg, inp: dict, border: list, tel=None, st_raw=None):
    """-> (X, bp, tr): the model inputs (v2_s31f_prep.finish without GT) and the default state mc7BP of the lap. The border rows are those of the closed-loop mc7_bp run (replayed: the filter
    reproduces its closed-loop output, S3.1f prep `replay_maxdiff`); poles = this lap's S2.5 detections. No GT."""
    import v2_s13_lib as L13
    import v2_s31e_prep as E
    import v2_s31f_lib as F
    from camloc.v2.learned.features import RAW_FIELDS
    tr = L13.track()
    if st_raw is None:
        st_raw = load_tel(cfg.tel, RAW_FIELDS, cfg.tel_limit_ns, window=(cfg.start_ns - MARGIN_NS, cfg.end_ns))
    r = F.run_bp(inp, border=border)
    dflt = F.add_track_state(r, tr)
    X = E.model_inputs(inp["ticks"], inp["t0"], st_raw, inp["rows0"], inp["streams"], {"bd": None}, tr)
    X.update(F.border_channel(inp["ticks"], inp["t0"], r["border"]))
    pa = F.PoleAssoc(inp["name"], inp["pole_map"], False)
    Xy = np.column_stack([r["x"], r["y"], r["yaw"]])
    X.update(F.pole_channel(inp["ticks"], inp["t0"], inp["poles"], Xy, r["cov"], pa))
    return X, dflt, tr


def gruf_apply(X: dict, bp: dict, tr, runs_dir=None, core="gru", rec="rec", device="cpu") -> dict:
    """v2_s31f_eval.dev: mean over the 4 fold models of (core, rec), then the v1d run-time rule with mc7BP as the default."""
    import torch
    import v2_s31f_eval as EV
    import v2_s31f_train as TR
    from camloc.v2.learned.v1d import FallbackCfg, postprocess
    torch.set_num_threads(2)
    dev = torch.device(device)
    d, _ = TR._tensors({k: np.asarray(v) for k, v in X.items()}, bp, dev, False)
    preds = []
    for fold in (0, 1, 3, 5):
        rd = Path(runs_dir or ROOT / "artifacts/v2/S3.1f-runs") / EV.run_name(rec, core, fold)
        m, c = TR.load_model(rd / "ckpt.pt", dev)
        preds.append(TR.run_lap(m, d, TR.CONDS["all"], dev))
    ticks = np.asarray(X["ticks"], np.int64)
    e = {"ts": ticks, "x": np.mean([p["x"] for p in preds], 0), "y": np.mean([p["y"] for p in preds], 0), "yaw": np.angle(np.mean([np.exp(1j * p["yaw"]) for p in preds], 0)),
         "logsig": np.log(np.mean([np.exp(p["logsig"]) for p in preds], 0)), "n": np.mean([p["n"] for p in preds], 0), "d": np.mean([p["d"] for p in preds], 0)}
    lm = np.zeros(len(ticks), np.float32)
    for c_ in EV.LM_CAMS:
        lm = np.maximum(lm, X[f"{c_}_pres"])
    full = postprocess(bp, e, lm, X["dt"], tr, FallbackCfg())
    return {"full": full, "raw": e}
