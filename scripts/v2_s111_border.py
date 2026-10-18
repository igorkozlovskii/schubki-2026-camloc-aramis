#!/usr/bin/env python3
"""S1.11 step 3: causal six-camera border stream along dev lap 2 from a release option's own poses (2 Hz), night or SIMULATED daylight store.

    CONDA_OVERRIDE_CUDA=12.4 pixi run --manifest-path environments/vision/pixi.toml python scripts/v2_s111_border.py --cond sunny --net s32e --start artifacts/runs/S1.11-Ap-sunny/run/predictions.csv --out artifacts/v2/S1.11-border/Ap-sunny-s32e

Same loop as scripts/v2_s32d_causal.py (imported pieces read-only), differences: start pose = the A' (G1) prediction at the tick (a causal output; its own finite-difference
velocity propagates it back to each camera stamp), 2 Hz, the full-train road-attitude map (S3.2c full.npz, looked up at the start's own s), the all-lap network checkpoint
(s32a = S3.2a lap-1_seed0; s32e = S3.2e variant B lap-1_seed0, daylight-augmented), body masks S1.1 `full`. Fit parameters temp 20000 / sd_dpsi 1.5 deg = the modal nested
S3.2d selection (3 of 4 folds), fixed before any lap-2 border result. NO GT is read here (evaluation is v2_s111_eval.py). Takes /tmp/camloc-gpu.lock.
"""
import argparse, fcntl, json, math, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_s32d_lib as D          # noqa: E402
import v2_s32c_run as R3         # noqa: E402
import v2_s11_common as H        # noqa: E402
from camloc.geometry.transforms import wrap_angle   # noqa: E402
from camloc.v2.border_probe import MEAS_COLUMNS, BorderMeas, meas_row   # noqa: E402
from camloc.v2.border_seg_att import AttTracker    # noqa: E402
from camloc.v2.border_fast import FastMeasurer     # noqa: E402
from camloc.v2.road_attitude import RoadAttitudeMap   # noqa: E402
from camloc.v2.seg.infer import SegPredictor, decode_jpeg   # noqa: E402

ROOT = D.ROOT
CAMS = ["fl", "fr", "cl", "cr", "rl", "rr"]
MAX_AGE_NS = 150_000_000
STORE = {"night": ROOT / "artifacts/v2/S1.11-store-night", "sunny": ROOT / "artifacts/v2/S1.11-store-sunny"}
CKPT = {"s32a": ROOT / "artifacts/v2/S3.2a-train/runs/lap-1_seed0/ckpt_last.pt", "s32e": ROOT / "artifacts/v2/S3.2e-train/runs/B_lap-1_seed0/ckpt_last.pt"}
MAP = ROOT / "artifacts/v2/S3.2c-attitude-map/full.npz"
MASKS = ROOT / "artifacts/v2/S1.1-masks/full"
FIT = {"temp": 20000, "sd_dpsi_deg": 1.5}


def wrapd(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cond", choices=list(STORE), required=True)
    ap.add_argument("--net", choices=list(CKPT), required=True)
    ap.add_argument("--start", required=True, help="predictions.csv of the option whose poses start the fit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rate", type=int, default=2, choices=(10, 5, 2, 1))
    ap.add_argument("--max-ticks", type=int, default=0)
    a = ap.parse_args(argv)
    cx = D.Ctx()
    cx.h = H.Harness(store_dir=STORE[a.cond])     # same class; GT is loaded by the class but never called in this script
    cx._ctx = {}
    h = cx.h
    ctx = h.context(None, MASKS)
    cfg = R3.make_cfg(FIT)
    pr = pd.read_csv(a.start)
    ts, fx, fy, fyaw = (pr[k].to_numpy() for k in ("timestamp_ns", "x", "y", "yaw"))
    ts = ts.astype(np.int64)
    fs = np.empty(len(ts))
    hint = None
    for i in range(len(ts)):
        s0, n0, th0, _ = h.cline.project(fx[i], fy[i], s_hint=hint)
        fs[i] = s0
        hint = s0
    sel = np.arange(0, len(ts), 10 // a.rate)
    if a.max_ticks:
        sel = sel[:a.max_ticks]
    lock = open("/tmp/camloc-gpu.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    t_all = time.time()
    try:
        meas = FastMeasurer(ctx, cx.polys, cfg, device="cuda", graph=True, prune=1e-6)
        pred = SegPredictor.from_checkpoint(str(CKPT[a.net]), "cuda")
        ram = RoadAttitudeMap.load(str(MAP))
        pool = ThreadPoolExecutor(6)
        tracker = AttTracker(cfg, True, sd0=cfg.sd0_att_map)
        rows, last_valid = [], None
        for i in sel:
            t = int(ts[i])
            t_start = time.perf_counter()
            j = max(i - 1, 0)
            dt_f = max((ts[i] - ts[j]) * 1e-9, 1e-3) if i > 0 else None
            vel = np.zeros(4) if dt_f is None else np.array([fx[i] - fx[j], fy[i] - fy[j], wrapd(fyaw[i] - fyaw[j]), fs[i] - fs[j]]) / dt_f
            fr = {}
            for c in CAMS:
                idx, st = h.camera_frame(c, t, MAX_AGE_NS)
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
                    items[c] = {"prob": pp[q_], "pose0": (fx[i] - vel[0] * dtc, fy[i] - vel[1] * dtc, fyaw[i] - vel[2] * dtc), "s_init": fs[i] - vel[3] * dtc, "attitude": None}
                ref = "fl" if "fl" in items else cams[0]
                ref_st = fr[ref][1]
                dz_, dpi_, dro_ = ram.lookup(items[ref]["s_init"])
                td = np.array([dro_, dpi_, dz_])
                mm = meas.measure_att(items, ref=ref, att_prior=tracker.prior(t, td), fit_att=True)
                if mm.valid:
                    tracker.update(t, [mm.q["droll"], mm.q["dpitch"], mm.q["dz"]], [mm.q["sd_roll"], mm.q["sd_pitch"], mm.q["sd_z"]], td)
                    s0, n0, th0, _ = h.cline.project(fx[i], fy[i], s_hint=fs[i])
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
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
    wall = time.time() - t_all
    st = pd.DataFrame(rows)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from camloc.v2.meas import write_meas
    write_meas(st[MEAS_COLUMNS], out / "lap2_multi.parquet")
    st[["stamp_ns", "tick_i"]].to_csv(out / "ticks.csv", index=False)
    (out / "run.json").write_text(json.dumps({"cond": a.cond, "net": a.net, "ckpt": str(CKPT[a.net]), "map": str(MAP), "masks": str(MASKS), "fit": FIT, "rate_hz": a.rate, "start": a.start,
                                              "label": "simulated daylight" if a.cond == "sunny" else "night", "wall_s": wall, "n_ticks": len(st), "valid": int(st.valid.sum()),
                                              "service_ms_median": float(st.q_service_ms.median())}, indent=1))
    print({"ticks": len(st), "valid": int(st.valid.sum()), "wall_s": round(wall, 1), "service_ms_median": float(st.q_service_ms.median())})


if __name__ == "__main__":
    main()
