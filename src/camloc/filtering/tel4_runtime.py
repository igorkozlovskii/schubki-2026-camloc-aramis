"""S10g stream: TelStream with the Tel4Filter (only the filter object is swapped; stream, availability and Y1-fallback logic are the S10b code)."""
from __future__ import annotations

from .tel4_filter import Reanchor4, Tel4Filter
from .tel_runtime import TelStream


class Tel4Stream(TelStream):
    def __init__(self, params, kin_params, first_pose, t0_ns, e04_rows, store, latency_scale=0.0, fallback_age_s=1.0, coast_age_s=0.1, rp: Reanchor4 | None = None):
        super().__init__(params, kin_params, first_pose, t0_ns, e04_rows, store, latency_scale, fallback_age_s, coast_age_s)
        self.rp = rp or Reanchor4()
        self.filter = Tel4Filter(params, first_pose, t0_ns, self.rp)
