"""Shared offline helpers for the S07c scripts (kinematic filter over E04 streams). Reuses the S07a stream loading and the numpy mirror of the
organizer metrics (used on TRAIN laps for tuning only; dev laps are scored by the official scorer through s07c_run_kin.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s07_fusion_lib import LAPS, ROOT, STREAMS, first_pose_for, quick_metrics  # noqa: E402,F401
from camloc.filtering import fusion_streams as fs  # noqa: E402
from camloc.filtering.kin_filter import KinParams  # noqa: E402,F401
from camloc.filtering.kin_runtime import KinStream  # noqa: E402

VARIANTS = {  # name: (model, use_retrieval, use_reanchor)
    "k1": ("cv", False, True), "k2": ("ctr", False, True), "k3": ("ctr", True, True), "k4": ("ctr", True, False),
}
CONT = {
    "q_acc": (0.5, 30.0), "fix_sig_xy": (0.15, 3.0), "fix_sig_yaw": (0.005, 0.1), "fix_w_inl": (0.0, 5.0),
    "gate_hard": (6.0, 21.1), "gate_soft": (2.0, 30.0), "gap_s": (1.0, 8.0), "gap_infl_m": (0.5, 30.0), "gap_infl_v": (0.0, 10.0),
}
YAW = {"q_yawacc": (0.05, 5.0)}
RET = {"ret_sig_lon": (2.0, 30.0), "ret_sig_lat": (1.0, 15.0), "ret_sig_yaw": (0.01, 0.3), "ret_gate": (4.0, 40.0), "ret_min_interval_s": (0.3, 4.0)}
INT = {"reanchor_n": (3, 8)}


def e04_stream(lap_id: str, root: Path = STREAMS):
    return fs.read_stream(root / f"{lap_id}.e04.jsonl.gz")


def run_offline(params: KinParams, e04, first_pose, t0_ns, latency_scale=1.0):
    ks = KinStream(params, first_pose, t0_ns, e04, latency_scale)
    ts = [r["t_ns"] for r in e04]
    out = [ks.tick(t) for t in ts]
    return np.array(ts, dtype=np.int64), out


def xyz(out):
    return np.array([[o["x"], o["y"], o["yaw"]] for o in out])
