"""Stream-driven kinematic filter: feeds KinFilter tick by tick from a cached E04 stream with the causal availability rule (S07c).

At tick t only E04 rows with stamp + measured latency x scale <= t are visible (camloc.filtering.fusion_streams.avail_ns). Only the
E04 stream is read: no E05, no pixels, no GT, no telemetry.
"""
from __future__ import annotations

from typing import Optional

from .fusion_streams import avail_ns
from .kin_filter import KinFilter, KinParams, Meas


class KinStream:
    def __init__(self, params: KinParams, first_pose: tuple, t0_ns: int, e04_rows: list, latency_scale: float = 1.0):
        self.p = params
        self.filter = KinFilter(params, first_pose, t0_ns)
        self.e04 = sorted(e04_rows, key=lambda r: (avail_ns(r, latency_scale), r["t_ns"]))
        self.lat = latency_scale
        self._ptr = 0
        self._tick_index: dict[int, int] = {}
        self.k = 0
        self.consumed_min: Optional[int] = None
        self.consumed_max: Optional[int] = None

    def tick(self, t_ns: int) -> dict:
        t_ns = int(t_ns)
        k = self.k
        self._tick_index[t_ns] = k
        lo = hi = None
        arrived = []
        while self._ptr < len(self.e04) and avail_ns(self.e04[self._ptr], self.lat) <= t_ns:
            row = self.e04[self._ptr]
            self._ptr += 1
            key = self._tick_index.get(row["t_ns"])
            if key is None:                       # stamp not on this run's clock grid, or a future tick: unusable
                continue
            lo = row["t_ns"] if lo is None else min(lo, row["t_ns"])
            hi = row["t_ns"] if hi is None else max(hi, row["t_ns"])
            if row["src"] in ("pnp", "retrieval"):
                arrived.append(Meas(key, k, row["src"], (row["x_m"], row["y_m"], row["yaw_rad"]), row.get("n_inl")))
        info = self.filter.step(t_ns, arrived)
        info["consumed_min_ns"], info["consumed_max_ns"] = lo, hi
        self.k += 1
        return info
