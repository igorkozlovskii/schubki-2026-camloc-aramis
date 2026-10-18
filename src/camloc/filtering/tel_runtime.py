"""Stream-driven telemetry-assisted filter (S10b): per tick it reads telemetry rows with stamp <= tick (TelemetryStore.rows_between / latest; nothing later) and the E04
rows with stamp + measured latency x latency_scale <= tick (S07c availability rule; scale 0 in the baseline treatment, D-018).

Modes per tick: 'telemetry' (latest telemetry row <= fallback_age_s old; rows older than coast_age_s are only held = coasting) or 'fallback' (store absent, no row yet, or
the latest row older than fallback_age_s): the output is then EXACTLY the frozen Y1 filter (Kin2Stream, run in parallel on the same E04 rows) and the telemetry state is
reseeded from it. After a fallback stretch the first telemetry interval is back-filled with the first row's velocity (the row is stamped <= the output tick).
Reads only the fields in NEEDED (a subset of telemetry-v1 ALLOWED_FIELDS); the store itself rejects any forbidden/unknown column.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from camloc.io.telemetry import TelemetryStore
from .fusion_streams import avail_ns
from .kin2_filter import Kin2Params
from .kin2_runtime import Kin2Stream
from .kin_filter import Meas
from .tel_filter import TelFilter, TelParams, TickRec

NEEDED = ("v_mps", "vx_mps", "vy_mps", "wz_radps")


class TelStream:
    def __init__(self, params: TelParams, kin_params: Kin2Params, first_pose: tuple, t0_ns: int, e04_rows: list, store: Optional[TelemetryStore],
                 latency_scale: float = 0.0, fallback_age_s: float = 1.0, coast_age_s: float = 0.1):
        if store is not None:
            missing = [f for f in NEEDED if f not in store.fields]
            if missing:
                raise ValueError(f"telemetry store lacks {missing}")
        self.p = params
        self.store = store
        self.kin = Kin2Stream(kin_params, first_pose, t0_ns, e04_rows, latency_scale)
        self.filter = TelFilter(params, first_pose, t0_ns)
        self.e04 = sorted(e04_rows, key=lambda r: (avail_ns(r, latency_scale), r["t_ns"]))
        self.lat = latency_scale
        self.fb_ns = int(fallback_age_s * 1e9)
        self.coast_ns = int(coast_age_s * 1e9)
        self._ptr = 0
        self._tick_index: dict[int, int] = {}
        self.k = 0
        self.t_prev = int(t0_ns)
        self._u_held: Optional[tuple] = None
        self._prev_mode = "fallback"
        if store is not None:
            lat = store.latest(int(t0_ns))
            if lat is not None and (int(t0_ns) - lat.stamp_ns) <= self.fb_ns:
                self._u_held = tuple(lat.row[f] for f in NEEDED)
                self._prev_mode = "telemetry"
        self.mode_counts = {"telemetry": 0, "fallback": 0}

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
            if key is None:
                continue
            lo = row["t_ns"] if lo is None else min(lo, row["t_ns"])
            hi = row["t_ns"] if hi is None else max(hi, row["t_ns"])
            if row["src"] in ("pnp", "retrieval"):
                arrived.append(Meas(key, k, row["src"], (row["x_m"], row["y_m"], row["yaw_rad"]), row.get("n_inl")))
        kin = self.kin.tick(t_ns)
        # ---- telemetry availability at this tick (rows with stamp <= t only)
        age_ns = None
        mode = "fallback"
        samples: list = []
        if self.store is not None:
            lat = self.store.latest(t_ns)
            if lat is not None:
                age_ns = lat.age_ns
                if age_ns <= self.fb_ns:
                    mode = "telemetry"
        if mode == "telemetry":
            st, cols = self.store.rows_between(self.t_prev, t_ns)
            samples = [(int(st[i]), tuple(float(cols[f][i]) for f in NEEDED)) for i in range(len(st))]
            u_start = self._u_held
            if u_start is None and samples:
                u_start = samples[0][1]
            if samples:
                self._u_held = samples[-1][1]
            elif u_start is None:
                lat = self.store.latest(t_ns)
                u_start = tuple(lat.row[f] for f in NEEDED)
                self._u_held = u_start
            rec = TickRec(t_ns, self.t_prev, u_start, samples)
        else:
            self._u_held = None
            P3 = np.array(kin["cov"])
            rec = TickRec(t_ns, self.t_prev, None, [], (kin["x"], kin["y"], kin["yaw"], P3))
        info = self.filter.step(rec, arrived)
        self.t_prev = t_ns
        self.k += 1
        self._prev_mode = mode
        self.mode_counts[mode] += 1
        out = dict(info) if mode == "telemetry" else {**info, "x": kin["x"], "y": kin["y"], "yaw": kin["yaw"], "cov": kin["cov"]}
        out.update({"mode": mode, "tel_age_ms": None if age_ns is None else age_ns * 1e-6, "tel_coast": bool(age_ns is not None and age_ns > self.coast_ns and mode == "telemetry"),
                    "kin_x": kin["x"], "kin_y": kin["y"], "kin_yaw": kin["yaw"], "consumed_min_ns": lo, "consumed_max_ns": hi, "n_samples": len(samples),
                    "kin_accepted": kin["accepted"], "kin_rejected": kin["rejected"], "kin_reanchor": kin["reanchor"], "kin_ret_used": kin["ret_used"]})
        if mode != "telemetry":
            for kk in ("accepted", "rejected", "reanchor", "ret_used"):
                out[kk] = kin[kk]
        return out


LAST_OK_NS = 1791919107438726051      # S10a export limit (laps 0-5); no audit-lap telemetry exists in the export


def load_store(directory, columns=NEEDED, windows=None, stale_ns: int = TelemetryStore.DEFAULT_STALE_NS) -> TelemetryStore:
    """TelemetryStore holding only `columns` (default-deny: a subset of the allowed fields) and, if `windows` [(t0, t1)] is given, only rows with t0 <= stamp <= t1
    (train-only work passes the train-lap windows, so dev rows are not even in memory). Refuses a file whose stamps exceed the S10a export range."""
    import json
    from pathlib import Path

    import pandas as pd
    d = Path(directory)
    df = pd.read_csv(d / "telemetry.csv", usecols=["stamp_ns", *columns], dtype={"stamp_ns": "int64"})
    if len(df) and int(df.stamp_ns.max()) > LAST_OK_NS:
        raise ValueError("telemetry rows beyond the laps-0-5 export range (audit laps) are not allowed")
    st = df.stamp_ns.to_numpy(dtype=np.int64)
    if windows is not None:
        keep = np.zeros(len(st), bool)
        for a, b in windows:
            keep |= (st >= a) & (st <= b)
        df, st = df[keep], st[keep]
    man = json.loads((d / "telemetry_manifest.json").read_text()) if (d / "telemetry_manifest.json").is_file() else {}
    return TelemetryStore(st, {c: df[c].to_numpy(dtype=np.float64) for c in columns}, man, stale_ns)
