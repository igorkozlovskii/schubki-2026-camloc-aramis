"""Stream-driven driver of the track filter: per tick it reads telemetry rows with stamp <= t (TelemetryStore.rows_between / latest) and `meas-v1` rows with
avail_ns <= t, nothing later. The first pose is consumed once (constructor). Works on cached measurement streams (offline) and is wrapped by `model.TrackFilterModel`
for the `camloc` runner (S2.1 wires it live; it is NOT registered in the release config).
"""
from __future__ import annotations

import math

from camloc.io.telemetry import TelemetryStore

from .core import TrackFilter
from .params import FilterParams
from .track import TrackFrame


def tel_input(p: FilterParams, row: dict, stamp: int):
    """Tier-specific control input (vlong, vlat, wz, ax, ay, stamp) from a telemetry row dict (only the tier's fields are present). vlat None = estimated state (T1)."""
    ax = row.get("ax_mps2", math.nan)
    if p.tier == "T2":
        return (row["vx_mps"], row["vy_mps"], row["wz_radps"], ax, math.nan, stamp)
    if p.t1_slip:
        return (row["v_mps"], None, row["wz_radps"], ax, row["ay_mps2"], stamp)
    return (row["v_mps"], 0.0, row["wz_radps"], ax, math.nan, stamp)


class TrackStream:
    def __init__(self, params: FilterParams, track: TrackFrame, first_pose: tuple, t0_ns: int, meas_rows: list[dict], store: TelemetryStore | None):
        self.p = params
        self.store = store if params.tier != "T0" else None
        if self.store is not None:
            missing = [f for f in params.fields if f not in self.store.fields]
            if missing:
                raise ValueError(f"telemetry store lacks {missing}")
        self.rows = meas_rows            # sorted by (avail_ns, stamp_ns, camera, kind)
        self._ptr = 0
        self.fb_ns = int(params.fallback_age_s * 1e9)
        self.t_prev = int(t0_ns)
        self._u_held = None
        v_init = None
        if self.store is not None:
            lat = self.store.latest(int(t0_ns))
            if lat is not None and (int(t0_ns) - lat.stamp_ns) <= self.fb_ns:
                self._u_held = tel_input(params, lat.row, lat.stamp_ns)
                v_init = math.hypot(self._u_held[0], self._u_held[1] or 0.0)
        self._first_pose_used = 0
        self._first_pose_used += 1       # consumed here, exactly once
        self.filter = TrackFilter(params, track, first_pose, t0_ns, v_init)
        self.mode_counts = {"telemetry": 0, "fallback": 0}

    def tick(self, t_ns: int) -> dict:
        t_ns = int(t_ns)
        p = self.p
        meas = []
        while self._ptr < len(self.rows) and int(self.rows[self._ptr]["avail_ns"]) <= t_ns:
            r = self.rows[self._ptr]
            self._ptr += 1
            if bool(r.get("valid", True)):
                meas.append(r)
        mode = "fallback"
        samples: list = []
        u_start = None
        tel_stamps: list[int] = []
        if self.store is not None:
            lat = self.store.latest(t_ns)
            if lat is not None and lat.age_ns <= self.fb_ns:
                mode = "telemetry"
        if mode == "telemetry":
            st, cols = self.store.rows_between(self.t_prev, t_ns)
            for i in range(len(st)):
                row = {f: float(cols[f][i]) for f in p.fields}
                samples.append((int(st[i]), tel_input(p, row, int(st[i]))))
            u_start = self._u_held
            if u_start is None and samples:
                u_start = samples[0][1]
            if samples:
                self._u_held = samples[-1][1]
            elif u_start is None:
                lat = self.store.latest(t_ns)
                u_start = tel_input(p, lat.row, lat.stamp_ns)
                self._u_held = u_start
            tel_stamps = [s for s, _ in samples]
        else:
            self._u_held = None
        out = self.filter.step(t_ns, u_start, samples, meas)
        self.t_prev = t_ns
        self.mode_counts[mode] += 1
        stamps = out.pop("consumed_stamps", []) + tel_stamps
        out["consumed_min_ns"] = min(stamps) if stamps else None
        out["consumed_max_ns"] = max(stamps) if stamps else None
        out["mode"] = mode
        out["n_samples"] = len(samples)
        return out
