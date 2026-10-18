"""Shared helpers for the S10b scripts (telemetry-assisted filter E10). Import-only reuse of the S07a/S07d/S07e helpers. Train work loads telemetry rows of the TRAIN laps only
(windows), so dev rows are not in memory; dev/stress runs go through the runner (s10b_run.py)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s07e_lib import (CONT as _C_UNUSED, LAPS, ROOT, STREAMS, Kin2Params, baseline_metrics, e04_stream, first_pose_for, full_metrics, holdlast, jl,  # noqa: E402,F401
                      kin_cfg_params, objective_lap, per_tick, yaw_stats)
from camloc.filtering.tel_filter import TelParams  # noqa: E402
from camloc.filtering.tel_runtime import TelStream, load_store  # noqa: E402
from camloc.io.telemetry import TelemetryStore  # noqa: E402

REP = ROOT / "reports/experiments/S10b-tel"
TELDIR = ROOT / "artifacts/telemetry/S10a-laps0-5"
S07D = ROOT / "reports/experiments/S07d-kin2"
S07E = ROOT / "reports/experiments/S07e-kin3"
TRAIN_LAPS = (1, 3, 5)

# caps and defaults: DECISION_RULE.md (declared before the first run)
CONT = {"q_pos": (0.01, 0.5), "q_psi": (1e-4, 5e-3), "q_a": (1e-5, 2e-3), "q_b": (1e-6, 1e-3), "fix_sig_xy": (0.15, 3.0), "fix_sig_yaw": (0.005, 0.1), "fix_w_inl": (0.0, 5.0),
        "gate_hard": (6.0, 21.1), "gate_soft": (2.0, 30.0), "ret_sig_lon": (2.0, 30.0), "ret_sig_lat": (1.0, 15.0), "ret_sig_yaw": (0.01, 0.3), "ret_gate": (4.0, 40.0),
        "ret_min_interval_s": (0.3, 4.0)}
INT = {"reanchor_n": (3, 8)}
DEFAULTS = dict(q_pos=0.07, q_psi=7.5e-4, q_a=1e-4, q_b=1e-5, sig_a0=1.5e-3, sig_b0=1e-4, fix_sig_xy=0.6, fix_sig_yaw=0.01, fix_w_inl=0.0, gate_hard=16.3, gate_soft=7.8,
                reanchor_n=4, ret_sig_lon=10.0, ret_sig_lat=2.5, ret_sig_yaw=0.06, ret_gate=11.3, ret_min_interval_s=1.0)
VARIANTS = {"V1": dict(use_vy=True, online_nuisance=True, use_retrieval=True),
            "V2": dict(use_vy=True, online_nuisance=False, use_retrieval=True),
            "V3": dict(use_vy=True, online_nuisance=True, use_retrieval=False)}
SKIP = {"V2": {"q_a", "q_b"}, "V3": {"ret_sig_lon", "ret_sig_lat", "ret_sig_yaw", "ret_gate", "ret_min_interval_s"}}
UTC = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())  # noqa: E731


def tel_params(variant: str, over: dict | None = None) -> TelParams:
    return TelParams(**{**DEFAULTS, **VARIANTS[variant], **(over or {})})


def lap_windows(laps):
    return [(LAPS[l][0] - 3_000_000_000, LAPS[l][1]) for l in laps]


_STORE = {}


def train_store(laps=TRAIN_LAPS) -> TelemetryStore:
    k = tuple(laps)
    if k not in _STORE:
        _STORE[k] = load_store(TELDIR, windows=lap_windows(laps))
    return _STORE[k]


_D = {}


def data(lap):
    if lap not in _D:
        fp, t0 = first_pose_for(lap)
        _D[lap] = (e04_stream(f"train-lap{lap}"), fp, t0)
    return _D[lap]


def y1_frozen() -> dict:
    return kin_cfg_params("y1")


def y1_fold(h: int) -> dict:
    return jl(S07D / f"tuning_holdout/y1_holdout{h}.json")["params"]


def run_tel(lap, tp: TelParams, kin_params: dict, store, lat: float = 0.0, rows=None, fallback_age_s: float = 1.0):
    r, fp, t0 = data(lap)
    rows = r if rows is None else rows
    s = TelStream(tp, Kin2Params(**kin_params), fp, t0, rows, store, lat, fallback_age_s)
    ts = np.array([x["t_ns"] for x in rows], dtype=np.int64)
    out = [s.tick(int(t)) for t in ts]
    return ts, out, s


def xyz(out):
    return np.array([[o["x"], o["y"], o["yaw"]] for o in out])


def eval_laps(variant, over, laps, kin_params, store=None, detail=False):
    store = train_store() if store is None else store
    res = {}
    for lap in laps:
        ts, out, _ = run_tel(lap, tel_params(variant, over), kin_params, store)
        res[lap] = full_metrics(*LAPS[lap], ts, xyz(out))
    J = float(sum(objective_lap(res[l], l) for l in laps) / len(laps))
    return (J, res) if detail else J
