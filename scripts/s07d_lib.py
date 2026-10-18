"""Shared offline helpers for the S07d scripts (yaw-aware kinematic filter over E04 streams; train-lap tuning uses the numpy mirror of the organizer
metrics, dev laps are scored by the official scorer through s07d_run_kin2.py). Import-only reuse of the S07a/S07c helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s07_fusion_lib import LAPS, ROOT, STREAMS, first_pose_for, gt_arrays  # noqa: E402,F401
from s07c_faults import baseline as hold_last_baseline  # noqa: E402,F401  (hold-last of the stream: PnP else retrieval else previous = e04-pnp-rfb-v1)
from s07c_lib import e04_stream  # noqa: E402,F401
from camloc.filtering.kin2_filter import Kin2Params  # noqa: E402,F401
from camloc.filtering.kin2_runtime import Kin2Stream  # noqa: E402

REP = ROOT / "reports/experiments/S07d-kin2"

# ---- search space (caps declared in DECISION_RULE.md before the first tuner run)
CONT = {
    "q_acc": (0.5, 30.0), "fix_sig_xy": (0.15, 3.0), "fix_sig_yaw": (0.005, 0.1), "fix_w_inl": (0.0, 5.0),
    "gate_hard": (6.0, 21.1), "gate_soft": (2.0, 30.0), "gap_s": (1.0, 8.0), "gap_infl_m": (0.5, 30.0), "gap_infl_v": (0.0, 10.0),
    "q_yawacc": (0.05, 3.5),
    "ret_sig_lon": (2.0, 30.0), "ret_sig_lat": (1.0, 15.0), "ret_sig_yaw": (0.01, 0.3), "ret_gate": (4.0, 40.0), "ret_min_interval_s": (0.3, 4.0),
}
Y2 = {"yaw_gate": (3.84, 15.1), "w_max": (0.8, 2.0), "gap_infl_yaw": (0.005, 0.3), "fix_w_inl_yaw": (0.0, 5.0)}
Y3 = {"yaw_hold_s": (0.3, 8.0)}
INT = {"reanchor_n": (3, 8)}
VARIANTS = {   # name: flags on top of CTR + retrieval + persistent-rejection reset
    "y1": dict(split_gate=False, yaw_only=False, ret_use_yaw=True),
    "y2": dict(split_gate=True, yaw_only=False, ret_use_yaw=True),     # ADDENDUM1 (first pass had yaw_only=True, ret_use_yaw=False)
    "y3": dict(split_gate=True, yaw_only=False, ret_use_yaw=True),
}
W = {"pos": 1.0, "lat": 1.0, "yaw": 2.0, "max": 0.5}     # objective weights (DECISION_RULE.md)


def full_metrics(lap_start_ns, lap_end_ns, ts, X, window_s=None):
    """Mirror of the organizer metrics (same bracketing/interpolation as scoring.mirror) plus max error; optional leading window (s) restriction.
    Evaluation side only (GT)."""
    g = gt_arrays()
    sel = (g["t"] >= lap_start_ns) & (g["t"] < lap_end_ns)
    tg, gx0, gy0, gyaw0 = g["t"][sel], g["x"][sel], g["y"][sel], g["yaw"][sel]
    ts = np.asarray(ts, dtype=np.int64)
    keep = np.ones(len(ts), bool) if window_s is None else (ts - lap_start_ns) <= int(window_s * 1e9)
    ts, X = ts[keep], np.asarray(X)[keep]
    i = np.searchsorted(tg, ts)
    ok = (i > 0) & (i < len(tg))
    i = np.clip(i, 1, len(tg) - 1)
    ok &= ~((tg[i] - tg[i - 1]) > 50_000_000)
    ts, i, X = ts[ok], i[ok], X[ok]
    a = (ts - tg[i - 1]) / (tg[i] - tg[i - 1])
    yu = np.unwrap(gyaw0)
    gx = gx0[i - 1] + a * (gx0[i] - gx0[i - 1])
    gy = gy0[i - 1] + a * (gy0[i] - gy0[i - 1])
    gyaw = yu[i - 1] + a * (yu[i] - yu[i - 1])
    dx, dy = X[:, 0] - gx, X[:, 1] - gy
    lat = -np.sin(gyaw) * dx + np.cos(gyaw) * dy
    pos = np.hypot(dx, dy)
    dyaw = np.degrees(np.angle(np.exp(1j * (X[:, 2] - gyaw))))          # wrapped explicitly (heading wrap-around)
    rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan")
    return {"n": int(len(pos)), "pos_rmse_m": rms(pos), "lat_rmse_m": rms(lat), "yaw_rmse_deg": rms(dyaw), "pos_median_m": float(np.median(pos)),
            "pos_p95_m": float(np.percentile(pos, 95)), "pos_max_m": float(pos.max()), "within_0_25m": float(np.mean(pos < 0.25)),
            "within_0_5m": float(np.mean(pos < 0.5)), "within_1m": float(np.mean(pos < 1.0))}


def run_offline(params, e04, first_pose, t0_ns, latency_scale=1.0):
    ks = Kin2Stream(params, first_pose, t0_ns, e04, latency_scale)
    ts = [r["t_ns"] for r in e04]
    out = [ks.tick(t) for t in ts]
    return np.array(ts, dtype=np.int64), out


def xyz(out):
    return np.array([[o["x"], o["y"], o["yaw"]] for o in out])


_BASE = {}


def baseline_metrics(lap):
    if lap not in _BASE:
        ts, X = hold_last_baseline(e04_stream(f"train-lap{lap}"))
        _BASE[lap] = full_metrics(*LAPS[lap], ts, X)
    return _BASE[lap]


def objective_lap(m, lap):
    b = baseline_metrics(lap)
    return (W["pos"] * m["pos_rmse_m"] / b["pos_rmse_m"] + W["lat"] * m["lat_rmse_m"] / b["lat_rmse_m"]
            + W["yaw"] * m["yaw_rmse_deg"] / b["yaw_rmse_deg"] + W["max"] * m["pos_max_m"] / b["pos_max_m"])
