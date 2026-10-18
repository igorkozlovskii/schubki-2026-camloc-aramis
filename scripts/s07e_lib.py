"""Shared offline helpers for the S07e scripts: latency-0 re-test of the frozen kinematic filters (K3, Y1) over E04 streams. Import-only reuse of the S07a/S07c/S07d
helpers (train laps use the numpy mirror of the organizer metrics; dev laps are scored by the official scorer through s07e_run.py).
latency_scale = 0 means: the E04 row stamped t is used for the output at tick t (same as the baseline hold-last), never earlier."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s07d_lib import CONT as CONT_S07D, INT as INT_S07D, LAPS, ROOT, STREAMS, VARIANTS, first_pose_for, full_metrics, gt_arrays  # noqa: E402,F401
from s07c_lib import e04_stream  # noqa: E402,F401
from camloc.filtering import fusion_streams as fs  # noqa: E402
from camloc.filtering.kin2_filter import Kin2Params  # noqa: E402,F401
from camloc.filtering.kin2_runtime import Kin2Stream  # noqa: E402
from camloc.filtering.kin_filter import KinParams, wrap  # noqa: E402,F401
from camloc.filtering.kin_runtime import KinStream  # noqa: E402

REP = ROOT / "reports/experiments/S07e-kin3"
S07C, S07D = ROOT / "reports/experiments/S07c-kin", ROOT / "reports/experiments/S07d-kin2"

# ---- (iii) search space: S07d caps with the six reviewer-found binding boxes widened once (DECISION_RULE.md)
WIDENED = {"gate_soft": (1.0, 60.0), "gap_infl_m": (0.5, 100.0), "gap_infl_v": (0.0, 30.0), "ret_sig_lon": (2.0, 60.0), "gap_s": (1.0, 16.0)}
CONT = {**CONT_S07D, **WIDENED}
INT = {"reanchor_n": (3, 12)}
W = {"pos": 1.0, "lat": 1.0, "yaw": 2.0, "max": 0.5}

STRESS_BASE = {"ctl-reencode": 5.968, "exp-x0.5": 6.16, "haze-severe": 7.406, "blur-11": 11.44, "noise-jpeg-hi": 159.874, "burst-2s-fl": 10.968,
               "stride-2": 20.798, "calib-yaw+1": 6.126}
DEV_BASE = {2: dict(pos=6.126, lat=3.520, yaw=2.012, max=36.101, cov=1.0), 4: dict(pos=5.855, lat=1.455, yaw=1.412, max=37.43, cov=1.0)}


def kin_cfg_params(name: str) -> dict:
    """Frozen config filter params of S07c/S07d (read-only)."""
    import yaml
    f = {"k3": "e09k-k3", "y1": "e09k2-y1"}[name]
    return yaml.safe_load((ROOT / f"configs/models/{f}.yaml").read_text())["params"]["filter"]


def make_stream(kind: str, params: dict, first_pose, t0_ns, rows, latency_scale: float):
    """kind 'k3' -> S07c KinStream/KinParams; 'y1' -> S07d Kin2Stream/Kin2Params (both imported unchanged)."""
    if kind == "k3":
        return KinStream(KinParams(**params), first_pose, t0_ns, rows, latency_scale)
    return Kin2Stream(Kin2Params(**params), first_pose, t0_ns, rows, latency_scale)


def run(kind: str, params: dict, rows, first_pose, t0_ns, latency_scale: float = 0.0):
    s = make_stream(kind, params, first_pose, t0_ns, rows, latency_scale)
    ts = np.array([r["t_ns"] for r in rows], dtype=np.int64)
    out = [s.tick(int(t)) for t in ts]
    return ts, out


def xyz(out):
    return np.array([[o["x"], o["y"], o["yaw"]] for o in out])


def holdlast(rows, latency_scale: float = 0.0):
    """Hold-last baseline. latency_scale 0: row t is the output at tick t (= e04-pnp-rfb-v1). latency_scale 1: only rows with stamp + measured latency <= t have arrived;
    the output at t is the latest-ARRIVED PnP-else-retrieval pose (the treatment the filters were charged in S07c/S07d). Transparency only."""
    ts = np.array([r["t_ns"] for r in rows], dtype=np.int64)
    if latency_scale == 0.0:
        X, last = [], None
        for r in rows:
            if r["src"] in ("pnp", "retrieval"):
                last = (r["x_m"], r["y_m"], r["yaw_rad"])
            X.append(last if last is not None else (0.0, 0.0, 0.0))
        return ts, np.array(X)
    order = sorted(range(len(rows)), key=lambda i: (fs.avail_ns(rows[i], latency_scale), rows[i]["t_ns"]))
    p, last, X = 0, None, []
    for t in ts:
        while p < len(order) and fs.avail_ns(rows[order[p]], latency_scale) <= t:
            r = rows[order[p]]
            p += 1
            if r["src"] in ("pnp", "retrieval"):
                last = (r["x_m"], r["y_m"], r["yaw_rad"])
        X.append(last if last is not None else (0.0, 0.0, 0.0))
    return ts, np.array(X)


def per_tick(lap_start_ns, lap_end_ns, ts, X):
    """Aligned per-tick errors (same bracketing/interpolation as full_metrics, evaluation side only). Returns dict of arrays + kept index."""
    g = gt_arrays()
    sel = (g["t"] >= lap_start_ns) & (g["t"] < lap_end_ns)
    tg, gx0, gy0, gyaw0 = g["t"][sel], g["x"][sel], g["y"][sel], g["yaw"][sel]
    ts = np.asarray(ts, dtype=np.int64)
    X = np.asarray(X)
    i = np.searchsorted(tg, ts)
    ok = (i > 0) & (i < len(tg))
    i = np.clip(i, 1, len(tg) - 1)
    ok &= ~((tg[i] - tg[i - 1]) > 50_000_000)
    idx = np.nonzero(ok)[0]
    i, Xk = i[ok], X[ok]
    a = (ts[ok] - tg[i - 1]) / (tg[i] - tg[i - 1])
    yu = np.unwrap(gyaw0)
    gx = gx0[i - 1] + a * (gx0[i] - gx0[i - 1])
    gy = gy0[i - 1] + a * (gy0[i] - gy0[i - 1])
    gyaw = yu[i - 1] + a * (yu[i] - yu[i - 1])
    dx, dy = Xk[:, 0] - gx, Xk[:, 1] - gy
    return {"idx": idx, "pos": np.hypot(dx, dy), "lat": -np.sin(gyaw) * dx + np.cos(gyaw) * dy,
            "dyaw": np.degrees(np.angle(np.exp(1j * (Xk[:, 2] - gyaw))))}


def yaw_stats(d):
    a = np.abs(d)
    return {"n": int(len(a)), "rmse": float(np.sqrt(np.mean(d ** 2))) if len(d) else float("nan"), "median": float(np.median(a)) if len(a) else float("nan"),
            "p95": float(np.percentile(a, 95)) if len(a) else float("nan")}


def tick_types(out):
    """Per tick: 'accepted' (a PnP fix accepted at this tick), 'rejected' (PnP fix rejected / re-anchored), 'retrieval' (retrieval weak update used), else 'nofresh'."""
    t = []
    for o in out:
        t.append("accepted" if o["accepted"] else "rejected" if (o["rejected"] or o["reanchor"]) else "retrieval" if o["ret_used"] else "nofresh")
    return np.array(t)


_BASE = {}


def baseline_metrics(lap):
    if lap not in _BASE:
        ts, X = holdlast(e04_stream(f"train-lap{lap}"), 0.0)
        _BASE[lap] = full_metrics(*LAPS[lap], ts, X)
    return _BASE[lap]


def objective_lap(m, lap):
    b = baseline_metrics(lap)
    return (W["pos"] * m["pos_rmse_m"] / b["pos_rmse_m"] + W["lat"] * m["lat_rmse_m"] / b["lat_rmse_m"]
            + W["yaw"] * m["yaw_rmse_deg"] / b["yaw_rmse_deg"] + W["max"] * m["pos_max_m"] / b["pos_max_m"])


def jl(p):
    return json.loads(Path(p).read_text())
