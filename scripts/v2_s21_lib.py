"""PRG-002 S2.1 evaluator-side helpers: the S1.3 harness (scripts/v2_s13_lib.py, imported read-only) re-pointed at `camloc.v2.filter2`.

GT is read by the imported evaluator helpers only, never by src/camloc/v2. Train laps 1, 3, 5 only; dev laps 2/4 and audit laps 6/7 are never loaded.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v2_s13_lib import *  # noqa: E402,F401,F403
import v2_s13_lib as L13  # noqa: E402
from camloc.v2.filter2.params import FilterParams, TIER_FIELDS, load_config  # noqa: E402,F401
from camloc.v2.filter2.stream import TrackStream  # noqa: E402

CFG2 = ROOT / "configs/v2/filter2"
REP2 = ROOT / "reports/experiments/V2-S2.1-filter2"
ART2 = ROOT / "artifacts/v2"


def run_filter(params: FilterParams, lap: int, rows=None, store="auto", fp_shift: float = 0.0, timing: bool = False, ret_stream: bool = False):
    """Run filter2 over a train lap's cached stream. Returns (ts, outs[, per-tick seconds][, stream])."""
    _, fp, t0 = data(lap)
    fp = (fp[0] + fp_shift, fp[1], fp[2])
    rows = L13.meas_rows(lap) if rows is None else rows
    st = L13.tier_store(params.tier) if (isinstance(store, str) and store == "auto") else store
    ts = L13.lap_ticks(lap)
    s = TrackStream(params, L13.track(), fp, t0, rows, st)
    outs, dts = [], []
    for t in ts:
        t1 = time.perf_counter()
        outs.append(s.tick(int(t)))
        dts.append(time.perf_counter() - t1)
    r = [ts, outs]
    if timing:
        r.append(np.array(dts))
    if ret_stream:
        r.append(s)
    return tuple(r)


def run_v1(cfg_path, lap: int, rows=None, store="auto", fp_shift: float = 0.0):
    """S1.3 filter (frozen, read-only) with its fold config, for the side-by-side rows."""
    from camloc.v2.filter.params import load_config as lc1
    from camloc.v2.filter.stream import TrackStream as TS1
    p, _ = lc1(cfg_path)
    _, fp, t0 = data(lap)
    fp = (fp[0] + fp_shift, fp[1], fp[2])
    rows = L13.meas_rows(lap) if rows is None else rows
    st = L13.tier_store(p.tier) if (isinstance(store, str) and store == "auto") else store
    ts = L13.lap_ticks(lap)
    s = TS1(p, L13.track(), fp, t0, rows, st)
    return ts, [s.tick(int(t)) for t in ts]


def xyz(outs):
    return np.array([[o["x"], o["y"], o["yaw"]] for o in outs])


def loo_cfg(tier: str, lap) -> Path:
    return CFG2 / "loo" / f"{tier}_holdout{lap}.yaml"


def extra_scenarios():
    """The reviewer's F4b-style scenarios beyond the S1.3 stress set (simulated, D-047): 20 s single-source +15 m, 8 s +25 m."""
    return [{"id": "X1-fixes+15m x20s@40", "kind": "cluster_fix", "p": {"t0": 40, "dur": 20, "mag": 15.0}, "win": (40.0, 60.0)},
            {"id": "X2-fixes+25m x8s@30", "kind": "cluster_fix", "p": {"t0": 30, "dur": 8, "mag": 25.0}, "win": (30.0, 38.0)}]
