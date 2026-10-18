"""S10f streams: TelStream / Kin2Stream with the S10f re-anchor logic (tel3_filter). Only the filter object is swapped; stream, availability and mode logic are the S10b/S07d code."""
from __future__ import annotations

from .kin2_filter import Kin2Params
from .kin2_runtime import Kin2Stream
from .tel3_filter import Kin3Filter, Reanchor3, Tel3Filter
from .tel_runtime import TelStream


class Tel3Stream(TelStream):
    def __init__(self, params, kin_params, first_pose, t0_ns, e04_rows, store, latency_scale=0.0, fallback_age_s=1.0, coast_age_s=0.1, rp: Reanchor3 | None = None):
        super().__init__(params, kin_params, first_pose, t0_ns, e04_rows, store, latency_scale, fallback_age_s, coast_age_s)
        self.rp = rp or Reanchor3()
        self.filter = Tel3Filter(params, first_pose, t0_ns, self.rp)


class Kin3Stream(Kin2Stream):
    def __init__(self, params: Kin2Params, first_pose, t0_ns, e04_rows, latency_scale=0.0, rp: Reanchor3 | None = None):
        super().__init__(params, first_pose, t0_ns, e04_rows, latency_scale)
        self.rp = rp or Reanchor3()
        self.filter = Kin3Filter(params, first_pose, t0_ns, self.rp)
