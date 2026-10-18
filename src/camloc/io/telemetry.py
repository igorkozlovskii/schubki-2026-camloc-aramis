"""Causal access to the separately exported allowed car-state telemetry (contract `inputs-v2`, step S10a, D-024).

INFERENCE-SIDE module: it reads only `telemetry.csv` + `telemetry_manifest.json` written by
`camloc.io.export_telemetry`, which carry ONLY the fields in `ALLOWED_FIELDS`. It never touches a raw MCAP, GT, or
`state_estimation` pose/track fields, and is deliberately not imported by `camloc.io.__init__` (telemetry is a separate
capability from the camera-only path: `capabilities.telemetry: true` in inputs-v2).

Availability at test time is UNKNOWN: `TelemetryStore.open_or_none(dir)` returns None when the file is absent, and the
caller must then run camera-only and report the mode (`mode_for(store)`).

Causality: a row is addressed by its int64 ns header stamp; `rows_upto(t)` / `rows_between(t0, t1]` never return a row
with stamp > t. The store owns no arrays that expose later rows (the future part is not sliced into returned objects).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TELEMETRY_NAME = "telemetry.csv"
TELEMETRY_MANIFEST_NAME = "telemetry_manifest.json"
TELEMETRY_ID = "telemetry-v1"

# name -> (unit, frame, description). THE allowlist: the exporter writes exactly these fields (default-deny); S10a
# FIELDS.md holds the evidence per field. All are body-frame / scalar quantities that encode no absolute pose or heading.
ALLOWED_FIELDS: dict[str, tuple[str, str, str]] = {
    "v_mps": ("m/s", "body (speed magnitude)", "fused speed over ground; |v_body| (== hypot(vx, vy) bitwise)"),
    "v_raw_mps": ("m/s", "body (speed magnitude)", "raw (wheel-based) speed before fusion"),
    "vx_mps": ("m/s", "body, x forward", "longitudinal velocity in base_link"),
    "vy_mps": ("m/s", "body, y left", "lateral velocity in base_link (small, |vy| < 1 m/s)"),
    "wx_radps": ("rad/s", "body", "roll rate"),
    "wy_radps": ("rad/s", "body", "pitch rate"),
    "wz_radps": ("rad/s", "body, z up", "yaw rate (integrates to heading change; NOT the course angle)"),
    "ax_mps2": ("m/s^2", "body", "longitudinal acceleration"),
    "ay_mps2": ("m/s^2", "body", "lateral acceleration (~ v * wz)"),
    "az_mps2": ("m/s^2", "body", "vertical acceleration"),
    "ax_vel_mps2": ("m/s^2", "velocity-aligned", "longitudinal acceleration along the velocity vector"),
    "ay_vel_mps2": ("m/s^2", "velocity-aligned", "lateral acceleration across the velocity vector"),
    "beta_rad": ("rad", "body", "body slip angle (== vy/vx; relative, not a course angle)"),
    "dbeta_radps": ("rad/s", "body", "slip-angle rate"),
    "kappa_radpm": ("rad/m", "body", "path curvature (= wz / v)"),
    "omega_w_fl": ("rad/s", "wheel", "front-left wheel angular speed"),
    "omega_w_fr": ("rad/s", "wheel", "front-right wheel angular speed"),
    "omega_w_rl": ("rad/s", "wheel", "rear-left wheel angular speed"),
    "omega_w_rr": ("rad/s", "wheel", "rear-right wheel angular speed"),
    "lambda_fl_perc": ("%", "wheel", "longitudinal tyre slip front-left"),
    "lambda_fr_perc": ("%", "wheel", "longitudinal tyre slip front-right"),
    "lambda_rl_perc": ("%", "wheel", "longitudinal tyre slip rear-left"),
    "lambda_rr_perc": ("%", "wheel", "longitudinal tyre slip rear-right"),
    "alpha_fl_rad": ("rad", "wheel", "tyre slip angle front-left"),
    "alpha_fr_rad": ("rad", "wheel", "tyre slip angle front-right"),
    "alpha_rl_rad": ("rad", "wheel", "tyre slip angle rear-left"),
    "alpha_rr_rad": ("rad", "wheel", "tyre slip angle rear-right"),
    "diff_fr_alpha_rad": ("rad", "wheel", "front-rear slip-angle difference"),
    "delta_wheel_rad": ("rad", "body", "steering (road-wheel) angle"),
    "gas": ("0..1", "driver", "throttle pedal"),
    "brake": ("Pa", "driver", "brake pressure command (values up to ~5e6: Pa, not 0..1)"),
    "gear": ("-", "powertrain", "gear index"),
    "rpm": ("rpm", "powertrain", "engine speed"),
    "cba_actual_pressure_fl_pa": ("Pa", "wheel", "actual brake pressure front-left"),
    "cba_actual_pressure_fr_pa": ("Pa", "wheel", "actual brake pressure front-right"),
    "cba_actual_pressure_rl_pa": ("Pa", "wheel", "actual brake pressure rear-left"),
    "cba_actual_pressure_rr_pa": ("Pa", "wheel", "actual brake pressure rear-right"),
    "valid_wheelsspeeds_b": ("bool", "flag", "wheel-speed validity flag (0.5 % false on laps 0-5)"),
}
FIELD_NAMES: tuple[str, ...] = tuple(ALLOWED_FIELDS)
STAMP_COL = "stamp_ns"

# Field names the exporter must NEVER write. Used only for the by-name 'skipped' report and for the leak tests; the
# real guard is default-deny (anything not in ALLOWED_FIELDS is skipped).
FORBIDDEN_NAMES: frozenset[str] = frozenset({
    "se_status", "se_state", "x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad", "pos_accuracy", "vel_accuracy",
    "yaw_vel_rad", "sn_map_state", "pos_acc_x", "pos_acc_y", "pos_acc_z",
})
FORBIDDEN_MARKERS: tuple[str, ...] = ("sn_map", "pos_acc", "yaw_vel", "_accuracy")


class TelemetryError(RuntimeError):
    pass


def assert_no_forbidden(columns) -> None:
    """Raise if any column is a forbidden/GT field name (substring markers included)."""
    bad = [c for c in columns if c in FORBIDDEN_NAMES or any(m in c for m in FORBIDDEN_MARKERS)]
    if bad:
        raise TelemetryError(f"forbidden telemetry columns present: {bad}")


@dataclass(frozen=True)
class Latest:
    """Latest row with stamp <= t. `age_ns` = t - stamp; `stale` = age above the cutoff; `row` is a dict of floats."""
    stamp_ns: int
    age_ns: int
    stale: bool
    row: dict


class TelemetryStore:
    """Immutable telemetry table indexed by int64 ns stamps (strictly increasing)."""

    DEFAULT_STALE_NS = 100_000_000   # 100 ms ~ 10 missed samples at 105 Hz (S10a: p99.9 gap ~ 10 ms, max 75 ms in laps 0-5)

    def __init__(self, stamps: np.ndarray, columns: dict[str, np.ndarray], manifest: dict | None = None,
                 stale_ns: int = DEFAULT_STALE_NS):
        st = np.ascontiguousarray(stamps, dtype=np.int64)
        if st.ndim != 1 or (len(st) > 1 and not np.all(np.diff(st) > 0)):
            raise TelemetryError("telemetry stamps must be a strictly increasing int64 ns vector")
        assert_no_forbidden(columns)
        unknown = [c for c in columns if c not in ALLOWED_FIELDS]
        if unknown:
            raise TelemetryError(f"telemetry columns outside the allowlist: {unknown}")
        self._t = st
        self._c = {k: np.ascontiguousarray(v, dtype=np.float64) for k, v in columns.items()}
        for k, v in self._c.items():
            if len(v) != len(st):
                raise TelemetryError(f"column {k} length {len(v)} != {len(st)}")
        self.manifest = manifest or {}
        self.stale_ns = int(stale_ns)

    # ---- construction -------------------------------------------------------------------------------------------------
    @classmethod
    def open(cls, directory, stale_ns: int = DEFAULT_STALE_NS) -> "TelemetryStore":
        import pandas as pd
        d = Path(directory)
        p = d / TELEMETRY_NAME
        if not p.is_file():
            raise TelemetryError(f"no {TELEMETRY_NAME} in {d} (telemetry unavailable: run camera-only)")
        df = pd.read_csv(p, dtype={STAMP_COL: "int64"})
        man = {}
        mp = d / TELEMETRY_MANIFEST_NAME
        if mp.is_file():
            man = json.loads(mp.read_text())
        cols = {c: df[c].to_numpy(dtype=np.float64) for c in df.columns if c != STAMP_COL}
        return cls(df[STAMP_COL].to_numpy(dtype=np.int64), cols, man, stale_ns)

    @classmethod
    def open_or_none(cls, directory, stale_ns: int = DEFAULT_STALE_NS) -> "TelemetryStore | None":
        """None when there is no telemetry file: the caller keeps the camera-only fallback and reports mode 'camera-only'."""
        if directory is None or not (Path(directory) / TELEMETRY_NAME).is_file():
            return None
        return cls.open(directory, stale_ns)

    @staticmethod
    def mode_for(store: "TelemetryStore | None") -> str:
        return "camera-only" if store is None else "telemetry"

    # ---- access -------------------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._t)

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self._c)

    def _upto_idx(self, t_ns: int) -> int:
        """Number of rows with stamp <= t (exclusive end index)."""
        return int(np.searchsorted(self._t, np.int64(t_ns), side="right"))

    def rows_upto(self, t_ns: int, n: int | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """(stamps, columns) of every row with stamp <= t (the last `n` of them if given). Copies: no view of later rows."""
        j = self._upto_idx(t_ns)
        i = 0 if n is None else max(0, j - int(n))
        return self._t[i:j].copy(), {k: v[i:j].copy() for k, v in self._c.items()}

    def rows_between(self, t0_ns: int, t1_ns: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Rows with t0 < stamp <= t1 (never beyond t1)."""
        j = self._upto_idx(t1_ns)
        i = self._upto_idx(t0_ns)
        i = min(i, j)
        return self._t[i:j].copy(), {k: v[i:j].copy() for k, v in self._c.items()}

    def latest(self, t_ns: int) -> Latest | None:
        """Latest row with stamp <= t with its age and stale flag; None if there is no such row (missing)."""
        j = self._upto_idx(t_ns)
        if j == 0:
            return None
        s = int(self._t[j - 1])
        age = int(t_ns) - s
        return Latest(s, age, age > self.stale_ns, {k: float(v[j - 1]) for k, v in self._c.items()})

    def first_stamp(self) -> int | None:
        return int(self._t[0]) if len(self._t) else None

    def gaps(self, max_gap_ns: int | None = None) -> list[tuple[int, int]]:
        """(prev_stamp, next_stamp) pairs with a gap above `max_gap_ns` (default: stale cutoff). Evaluator-side convenience."""
        lim = self.stale_ns if max_gap_ns is None else max_gap_ns
        d = np.diff(self._t)
        k = np.nonzero(d > lim)[0]
        return [(int(self._t[i]), int(self._t[i + 1])) for i in k]
