"""Stream-driven fusion: feeds FusionFilter tick by tick from cached E05/E04 streams with the causal availability rule.

`StreamFusion.tick(t_ns)` uses only stream rows with stamp (E05) / stamp + latency x scale (E04) <= t_ns, and records the earliest and
latest stamp it consumed so the model can report `consumed_min_ns/consumed_max_ns` (<= t). The same class drives offline tuning and
the protocol model (camloc.models.fusion), so the tuned code path is the shipped one.
"""
from __future__ import annotations

import math
from typing import Optional

from .fusion_filter import FusionFilter, FusionParams, Inc, Meas
from .fusion_streams import avail_ns


class StreamFusion:
    def __init__(self, params: FusionParams, first_pose: tuple, t0_ns: int, e05_rows: list, e04_rows: Optional[list],
                 latency_scale: float = 1.0, inc_yaw_rot_rad: float = 0.0, use_e04: bool = True):
        self.p = params
        self.filter = FusionFilter(params, first_pose, t0_ns)
        self.e05 = {r["t_ns"]: r for r in e05_rows}
        self.e04 = sorted(e04_rows or [], key=lambda r: (avail_ns(r, latency_scale), r["t_ns"])) if use_e04 else []
        self.lat = latency_scale
        self.rot = inc_yaw_rot_rad
        self._ptr = 0
        self._tick_index: dict[int, int] = {}
        self.k = 0
        self.consumed_min: Optional[int] = None
        self.consumed_max: Optional[int] = None

    def tick(self, t_ns: int) -> dict:
        t_ns = int(t_ns)
        k = self.k
        self._tick_index[t_ns] = k
        r = self.e05.get(t_ns)
        if r is not None and r["stream"] != "e05_motion":
            raise ValueError("wrong stream kind for E05 slot")
        if r is not None and r["valid"] and k > 0:
            dx, dy = r["dx_m"], r["dy_m"]
            if self.rot:
                c, s = math.cos(self.rot), math.sin(self.rot)
                dx, dy = c * dx - s * dy, s * dx + c * dy
            inc = Inc(dx, dy, r["dyaw_rad"], True, r["quality"])
        else:
            inc = Inc(0.0, 0.0, 0.0, False, "invalid")
        lo = hi = None
        if r is not None:
            lo = hi = r["t_ns"]
        arrived = []
        while self._ptr < len(self.e04) and avail_ns(self.e04[self._ptr], self.lat) <= t_ns:
            row = self.e04[self._ptr]
            self._ptr += 1
            key = self._tick_index.get(row["t_ns"])
            if key is None:                       # stamp not on the clock grid of this run (or a future tick): unusable
                continue
            lo = row["t_ns"] if lo is None else min(lo, row["t_ns"])
            hi = row["t_ns"] if hi is None else max(hi, row["t_ns"])
            if row["src"] == "pnp":
                arrived.append(Meas(key, k, "pnp", (row["x_m"], row["y_m"], row["yaw_rad"]), row.get("d_kf_m"), row.get("n_inl"), row.get("n_kf_inl")))
            elif row["src"] == "retrieval" and self.p.use_retrieval:
                arrived.append(Meas(key, k, "retrieval", (row["x_m"], row["y_m"], row["yaw_rad"]), row.get("d_kf_m"), row.get("n_inl")))
        info = self.filter.step(t_ns, inc, arrived)
        info["inc_valid"] = inc.valid
        info["consumed_min_ns"], info["consumed_max_ns"] = lo, hi
        self.k += 1
        return info
