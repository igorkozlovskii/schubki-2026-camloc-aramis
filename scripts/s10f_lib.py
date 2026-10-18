"""S10f helpers: nested V3 reference, per-tick evaluator-side errors, reanchor-event forensics. Import-only reuse of s10b/s10e libs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s10b_lib import (LAPS, REP as REP_B, ROOT, TRAIN_LAPS, data, full_metrics, jl, objective_lap, per_tick, tel_params, train_store, xyz, y1_fold, y1_frozen)  # noqa: E402,F401
from camloc.filtering.tel_runtime import TelStream  # noqa: E402
from camloc.filtering.kin2_filter import Kin2Params  # noqa: E402

REP = ROOT / "reports/experiments/S10f-tel3"


def v3_over(h=None):
    """tuned_over of S10b's V3: fold h (excludes lap h) or all-train (h None)."""
    return jl(REP_B / f"tuning/V3_{'all' if h is None else 'holdout' + str(h)}.json")["tuned_over"]


def run_stream(lap, tp, kin, store, lat=0.0, rows=None, cls=TelStream):
    r, fp, t0 = data(lap)
    rows = r if rows is None else rows
    s = cls(tp, Kin2Params(**kin), fp, t0, rows, store, lat)
    ts = np.array([x["t_ns"] for x in rows], dtype=np.int64)
    out = [s.tick(int(t)) for t in ts]
    return ts, out, s


def errors(lap, ts, out):
    """Evaluator-side per-tick position error aligned to ts (nan where GT missing)."""
    X = xyz(out)
    pt = per_tick(*LAPS[lap], ts, X)
    e = np.full(len(ts), np.nan)
    e[pt["idx"]] = pt["pos"]
    return e


def events(lap, ts, out, e):
    """Re-anchor ticks with true error before/after (evaluator side)."""
    ev = []
    for k, o in enumerate(out):
        if o.get("reanchor"):
            ev.append({"lap": lap, "k": k, "t_s": (int(ts[k]) - LAPS[lap][0]) * 1e-9, "err_before": float(e[k - 1]) if k else None, "err_after": float(e[k])})
    return ev


from camloc.filtering.kin2_filter import Kin2Params as _K2  # noqa: E402
from camloc.filtering.tel3_filter import Reanchor3  # noqa: E402
from camloc.filtering.tel3_runtime import Kin3Stream, Tel3Stream  # noqa: E402

LOSS_DX = 10.0


def run3(lap, tp, kin, store, rp, lat=0.0, rows=None, fp_shift=0.0, camera_only=False, return_stream=False):
    """V3-R (Tel3Stream) or Y1-R (camera_only: Kin3Stream, no telemetry) over a train-lap E04 stream; fp_shift displaces the first pose in x (loss test)."""
    r, fp, t0 = data(lap)
    rows = r if rows is None else rows
    fp = (fp[0] + fp_shift, fp[1], fp[2])
    if camera_only:
        s = Kin3Stream(_K2(**kin), fp, t0, rows, lat, rp)
    else:
        s = Tel3Stream(tp, _K2(**kin), fp, t0, rows, store, lat, rp=rp)
    ts = np.array([x["t_ns"] for x in rows], dtype=np.int64)
    out = [s.tick(int(t)) for t in ts]
    return (ts, out, s) if return_stream else (ts, out)


def recover_s(lap, ts, out):
    e = errors(lap, ts, out)
    ok = np.nonzero(e < 3.0)[0]
    return float(min((int(ts[ok[0]]) - int(ts[0])) * 1e-9, 60.0)) if len(ok) else 60.0


def pooled_stats(per_lap_err):
    """per_lap_err: {lap: per-tick error array (nan = unscored)} -> pooled RMSE, n>3 m, per-lap max."""
    allv = np.concatenate([e[~np.isnan(e)] for e in per_lap_err.values()])
    return {"pos_rmse_m": float(np.sqrt(np.mean(allv ** 2))), "n_gt3": int((allv > 3.0).sum()), "n": int(len(allv)),
            "max_per_lap": {int(l): float(np.nanmax(e)) for l, e in per_lap_err.items()}}
