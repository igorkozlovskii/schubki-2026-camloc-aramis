"""Stream-driven yaw-aware kinematic filter (S07d): same causal availability rule as S07c (E04 rows with stamp + measured latency x scale <= t)."""
from __future__ import annotations

from .kin2_filter import Kin2Filter, Kin2Params
from .kin_runtime import KinStream


class Kin2Stream(KinStream):
    def __init__(self, params: Kin2Params, first_pose: tuple, t0_ns: int, e04_rows: list, latency_scale: float = 1.0):
        super().__init__(params, first_pose, t0_ns, e04_rows, latency_scale)
        self.filter = Kin2Filter(params, first_pose, t0_ns)
