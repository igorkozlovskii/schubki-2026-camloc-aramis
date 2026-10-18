"""PRG-002 S3.1: build per-tick input arrays (src/camloc/v2/learned/features.py, no GT) and, separately, the GT targets (evaluator side, here only).

Laps: training laps 0, 1, 3, 5 only (dev 2/4 and audit 6/7 are never loaded). Per lap one npz under artifacts/v2/S3.1-data/:
  inputs (features.tick_inputs) + targets gt_* (from derived/gt.csv; sigma and pit/tunnel flags from state_estimation pos_accuracy /
  current_lane, used ONLY as the loss mask) + first pose / t0.
Channels: laps 1/3/5 fl landmark + coarse = the E04 rows G1 sees (S07a LOO streams via scripts/v2_s13_lib.meas_rows, avail = stamp, D-018);
lap 0 (no E04 stream) uses the S1.2 LOO fl landmark / coarse streams (stride 2, measured avail). fr landmark: S1.2 LOO streams (all laps).
Border: S3.2d causal 2 Hz streams (laps 1/3/5; absent on lap 0). Stress (lap 3, simulated, D-047): F1, F3, F4b (S1.3 STRESS specs), X1.

    pixi run python scripts/v2_s31_prep.py [--laps 0 1 3 5] [--stress]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v2_s13_lib as L13  # noqa: E402
from v2_s13_lib import LAPS, ROOT, data  # noqa: E402
from camloc.filtering.tel_runtime import load_store  # noqa: E402
from camloc.v2.learned.features import RAW_FIELDS, tick_inputs  # noqa: E402

OUT = ROOT / "artifacts/v2/S3.1-data"
S12 = ROOT / "artifacts/v2/S1.2-streams"
BORDER = ROOT / "artifacts/v2/S3.2d-streams/causal_2hz/e_fit"
LAP0_SKIP_S = 60.0     # pit exit + tunnel (GT sigma > 1 m at 42.6-54.8 s), as in S1.1/S1.4
TRAIN = (0, 1, 3, 5)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows_parquet(p: Path) -> list[dict]:
    d = pd.read_parquet(p)
    return d.to_dict("records")


def lap_store(lap: int):
    from s10b_lib import lap_windows
    return load_store(L13.TELDIR, columns=RAW_FIELDS, windows=lap_windows([lap]))


def lap_setup(lap: int):
    """-> ticks, t0, first pose (x, y, yaw), channel rows. Inference-side inputs only."""
    assert lap in TRAIN
    if lap == 0:
        f = pd.read_csv(ROOT / "artifacts/inputs/split-v1/frames.csv")
        g = f[(f.camera == "fl") & (f.lap == 0)].stamp_ns.to_numpy(np.int64)
        g = np.sort(g[(g >= LAPS[0][0] + int(LAP0_SKIP_S * 1e9)) & (g < LAPS[0][1])])
        t0, ticks = int(g[0]), g[1:]
        fp = None                                   # training-only lap: start from GT (+ perturbation) in the trainer
        fl = rows_parquet(S12 / "lap0_fl_landmark.parquet")
        co = rows_parquet(S12 / "lap0_fl_coarse.parquet")
        ch = {"fl_lm": fl, "fl_co": co}
    else:
        _, fp, t0 = data(lap)
        ticks = L13.lap_ticks(lap)
        rows = L13.meas_rows(lap)
        ch = {"fl_lm": [r for r in rows if r["kind"] == "landmark"], "fl_co": [r for r in rows if r["kind"] == "coarse"]}
        ch["bd"] = rows_parquet(BORDER / f"lap{lap}_multi.parquet")
    ch["fr_lm"] = rows_parquet(S12 / f"lap{lap}_fr_landmark.parquet")
    return ticks, int(t0), fp, ch


def gt_targets(lap: int, ticks: np.ndarray) -> dict:
    """Evaluator side: GT pose at each tick (organizer bracketing, max gap 50 ms), track coordinates, speed, sigma and the loss mask."""
    from s07_fusion_lib import gt_arrays
    tr = L13.track()
    g = gt_arrays()
    a, b = LAPS[lap]
    sel = (g["t"] >= a - 2_000_000_000) & (g["t"] < b + 2_000_000_000)
    tg, gx, gy, gyaw = g["t"][sel], g["x"][sel], g["y"][sel], np.unwrap(g["yaw"][sel])
    i = np.clip(np.searchsorted(tg, ticks), 1, len(tg) - 1)
    ok = ~((tg[i] - tg[i - 1]) > 50_000_000)
    f = (ticks - tg[i - 1]) / (tg[i] - tg[i - 1])
    x = gx[i - 1] + f * (gx[i] - gx[i - 1])
    y = gy[i - 1] + f * (gy[i] - gy[i - 1])
    yaw = gyaw[i - 1] + f * (gyaw[i] - gyaw[i - 1])
    snp = np.zeros((len(ticks), 3))
    hint = None
    for k in range(len(ticks)):
        snp[k] = tr.from_xy(float(x[k]), float(y[k]), float(yaw[k]), s_hint=hint)
        hint = snp[k, 0]
    tt = ticks * 1e-9
    vx = np.gradient(x, tt)
    vy = np.gradient(y, tt)
    v = np.hypot(vx, vy)
    se = pd.read_parquet(ROOT / "derived/state_estimation.parquet", columns=["timestamp_ns", "sn_map_state.current_lane", "pos_accuracy.0", "pos_accuracy.1"])
    se = se.sort_values("timestamp_ns")
    st = se.timestamp_ns.to_numpy(np.int64)
    j = np.clip(np.searchsorted(st, ticks, side="right") - 1, 0, len(st) - 1)
    sig = np.hypot(se["pos_accuracy.0"].to_numpy(), se["pos_accuracy.1"].to_numpy())[j]
    lane0 = se["sn_map_state.current_lane"].to_numpy()[j] == 0
    w_ok = ok & (sig <= 1.0) & ~lane0
    return {"gt_x": x, "gt_y": y, "gt_yaw": np.angle(np.exp(1j * yaw)), "gt_s": snp[:, 0], "gt_n": snp[:, 1], "gt_psi": snp[:, 2], "gt_v": v,
            "gt_sig": sig, "gt_ok": ok, "gt_w": w_ok}


def build(lap: int, scen: dict | None, out: Path) -> dict:
    t1 = time.time()
    tr = L13.track()
    ticks, t0, fp, ch = lap_setup(lap)
    store = lap_store(lap)
    fp_shift = 0.0
    if scen is not None:
        k, p = scen["kind"], scen["p"]
        if k == "fp":
            fp_shift = float(p["d"])
        elif k == "wz":
            store = L13.store_edit(store, lap, p["t0"], p["dur"], ["wz_radps"], lambda x: x + p["dw"])
        elif k == "cluster_fix":
            ch["fl_lm"] = L13.meas_cluster(ch["fl_lm"], lap, p["t0"], p["dur"], p["mag"])
        else:
            raise ValueError(k)
    cols = {f: store._c[f] for f in store.fields}
    X = tick_inputs(ticks, t0, store._t, cols, ch, tr)
    G = gt_targets(lap, np.asarray(ticks, np.int64))
    if fp is None:   # lap 0 (training only): GT at t0 is the nominal start
        g0 = gt_targets(lap, np.array([t0, t0 + 1], np.int64))
        fp = (float(g0["gt_x"][0]), float(g0["gt_y"][0]), float(g0["gt_yaw"][0]))
    fp = (fp[0] + fp_shift, fp[1], fp[2])
    s0, n0, p0 = tr.from_xy(*fp)
    lat = store.latest(t0)
    v2 = float(np.hypot(lat.row["vx_mps"], lat.row["vy_mps"])) if lat is not None else 25.0
    v1 = float(lat.row["v_mps"]) if lat is not None else 25.0
    meta = {"lap": lap, "t0": t0, "fp": list(fp), "fp_snp": [s0, n0, p0], "v_init_T2": v2, "v_init_T1": v1, "scenario": None if scen is None else scen["id"],
            "n_ticks": int(len(ticks)), "channels_present": {c: int(X[f"{c}_pres"].sum()) for c in ("fl_lm", "fl_co", "fr_lm", "bd")}}
    np.savez_compressed(out, **X, **{k: np.asarray(v) for k, v in G.items()}, meta=json.dumps(meta))
    meta["seconds"] = round(time.time() - t1, 1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", type=int, nargs="*", default=list(TRAIN))
    ap.add_argument("--stress", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man = {"step": "PRG-002 S3.1", "files": {}, "inputs": {}}
    for lap in a.laps:
        m = build(lap, None, OUT / f"lap{lap}.npz")
        print(json.dumps(m), flush=True)
        man["files"][f"lap{lap}.npz"] = m
    if a.stress:
        from v2_s21_lib import extra_scenarios
        specs = [s for s in L13.STRESS if s["id"].split("-")[0] in ("F1", "F3", "F4b")] + [s for s in extra_scenarios() if s["id"].startswith("X1")]
        for s in specs:
            tag = s["id"].split("-")[0]
            m = build(3, s, OUT / f"lap3_{tag}.npz")
            m["spec"] = s
            print(json.dumps(m), flush=True)
            man["files"][f"lap3_{tag}.npz"] = m
    for p in sorted(OUT.glob("*.npz")):
        man["files"].setdefault(p.name, {})["sha256"] = sha(p)
    for p in [*sorted(S12.glob("lap[0135]_fl_*.parquet")), *sorted(S12.glob("lap[0135]_fr_landmark.parquet")), *sorted(BORDER.glob("lap[135]_multi.parquet"))]:
        if not p.name.endswith("_eval.parquet"):
            man["inputs"][str(p.relative_to(ROOT))] = sha(p)
    mp = ROOT / "manifests/v2/S3.1-data.json"
    old = json.loads(mp.read_text()) if mp.is_file() else {}
    if old:
        for k, v in old.get("files", {}).items():
            man["files"].setdefault(k, v)
    mp.write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
