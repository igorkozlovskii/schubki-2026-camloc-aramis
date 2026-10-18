"""S1.8 car attitude for the border measurement: override container, telemetry-based estimator (inference-clean).

S1.1 projected the supplied borders with z / pitch / roll taken from the borders themselves (`BorderMap.road_attitude`, i.e. the road's
slope and banking, not the car's suspension attitude). Igor's hypothesis (17 Oct) is that the car's own pitch (braking/acceleration) and
roll (cornering) are why the projected and the visible border differ. This module holds

  * `Attitude`: absolute or additive (dz, dpitch, droll) override applied inside `border_probe._Scene` (None entries keep S1.1's value);
  * `map_attitude`: the S1.1 (z_base, pitch, roll) at a pose, from the supplied borders only;
  * `telemetry_features`: causal features from ALLOWED telemetry fields only (leaky integrals of the body rates wx / wy, low-passed
    ax / ay / az, speed squared, brake, gas);
  * `AttitudeModel`: a ridge-linear model delta = W f (+ b) with coefficients read from a yaml file; `estimate` returns an
    `Attitude` of additive deltas on top of the map attitude.

Nothing here reads ground truth, `derived/`, `refline_*` or `sn_map_state`; the GT fit lives in scripts/v2_s18_fit_attitude.py, which writes
the yaml this module loads (a test greps the source and the import closure).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from camloc.geometry.borders import BASE_BELOW_ROAD_M, BorderMap

# telemetry fields the estimator may read (subset of camloc.io.telemetry.ALLOWED_FIELDS)
FIELDS = ("v_mps", "wx_radps", "wy_radps", "ax_mps2", "ay_mps2", "az_mps2", "brake", "gas")
BRAKE_SCALE = 5e6                  # Pa, "values up to ~5e6" (telemetry FIELDS)
TAUS_RATE = (0.5, 2.0)             # s, leak time constants of the integrated roll / pitch rates
TAU_ACC = 0.15                     # s, low-pass of the accelerations
TAU_PEDAL = 0.3
FEATURE_NAMES = ("ax_lp", "ay_lp", "az_lp", "v2", "brake", "gas",
                 "int_wy_0.5", "int_wy_2.0", "int_wx_0.5", "int_wx_2.0")
TARGETS = ("dz", "dpitch", "droll")


@dataclass(frozen=True)
class Attitude:
    """Override of the car's attitude in the border projection. Absolute fields replace S1.1's value; d* fields are added after."""
    z: float | None = None              # base_link height (m)
    pitch: float | None = None          # rad, > 0 nose down (T = Rz Ry Rx)
    roll: float | None = None           # rad
    dz: float = 0.0
    dpitch: float = 0.0
    droll: float = 0.0

    def apply(self, z_base, pitch, roll):
        z = z_base if self.z is None else self.z
        p = pitch if self.pitch is None else self.pitch
        r = roll if self.roll is None else self.roll
        return z + self.dz, p + self.dpitch, r + self.droll


def map_attitude(bmap: BorderMap, x: float, y: float, yaw: float):
    """(z_base, pitch, roll) exactly as S1.1's `_Scene` takes them from the supplied borders at pose (x, y, yaw)."""
    rs = bmap.road_height("main", (x, y), None)
    pitch, roll = bmap.road_attitude("main", x, y, yaw, rs.z_road)
    return rs.z_road - BASE_BELOW_ROAD_M, pitch, roll


# ------------------------------------------------------------------------------------------------ causal telemetry features
def _lowpass(t_s: np.ndarray, x: np.ndarray, tau: float, integrate: bool = False) -> np.ndarray:
    """Causal first-order recursion over irregular stamps. integrate=False: exponential low-pass (unit DC gain);
    integrate=True: leaky integral dy/dt = x - y / tau. Output at index j uses samples 0..j only."""
    y = np.empty(len(x))
    acc = float(x[0]) if not integrate else 0.0
    y[0] = acc
    for j in range(1, len(x)):
        dt = float(t_s[j] - t_s[j - 1])
        a = math.exp(-dt / tau)
        if integrate:
            acc = a * acc + tau * (1.0 - a) * float(x[j])
        else:
            acc = a * acc + (1.0 - a) * float(x[j])
        y[j] = acc
    return y


def telemetry_features(stamps_ns: np.ndarray, cols: dict) -> np.ndarray:
    """(N, len(FEATURE_NAMES)) features for every telemetry row, causal. Needs the columns in FIELDS."""
    t = (np.asarray(stamps_ns, np.int64) - int(stamps_ns[0])) * 1e-9
    f = [_lowpass(t, cols["ax_mps2"], TAU_ACC), _lowpass(t, cols["ay_mps2"], TAU_ACC), _lowpass(t, cols["az_mps2"], TAU_ACC),
         np.asarray(cols["v_mps"], float) ** 2 / 100.0,
         _lowpass(t, np.asarray(cols["brake"], float) / BRAKE_SCALE, TAU_PEDAL), _lowpass(t, cols["gas"], TAU_PEDAL)]
    for k in ("wy_radps", "wx_radps"):
        for tau in TAUS_RATE:
            f.append(_lowpass(t, cols[k], tau, integrate=True))
    return np.column_stack(f)


# ------------------------------------------------------------------------------------------------ model
@dataclass
class AttitudeModel:
    """delta[target] = bias + sum_k coef[target][k] * (feature_k - mean_k) / scale_k, targets (dz, dpitch, droll) w.r.t. the map attitude."""
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray          # (3, n_features)
    bias: np.ndarray          # (3,)
    meta: dict

    def delta(self, feat: np.ndarray) -> np.ndarray:
        z = (np.asarray(feat, float) - self.mean) / self.scale
        return self.bias + self.coef @ z

    def attitude(self, feat: np.ndarray) -> Attitude:
        d = self.delta(feat)
        return Attitude(dz=float(d[0]), dpitch=float(d[1]), droll=float(d[2]))

    def to_yaml(self, path) -> None:
        Path(path).write_text(yaml.safe_dump({"features": list(FEATURE_NAMES), "targets": list(TARGETS), "mean": self.mean.tolist(),
                                              "scale": self.scale.tolist(), "coef": self.coef.tolist(), "bias": self.bias.tolist(),
                                              "meta": self.meta}, sort_keys=False))

    @classmethod
    def from_yaml(cls, path) -> "AttitudeModel":
        d = yaml.safe_load(Path(path).read_text())
        if tuple(d["features"]) != FEATURE_NAMES or tuple(d["targets"]) != TARGETS:
            raise ValueError("attitude model features/targets differ from this module's")
        return cls(np.array(d["mean"]), np.array(d["scale"]), np.array(d["coef"]), np.array(d["bias"]), d.get("meta", {}))


def latest_feature(stamps_ns: np.ndarray, cols: dict, t_ns: int) -> np.ndarray | None:
    """Feature row at the latest telemetry stamp <= t_ns (None if there is none). Rows after t_ns are never read."""
    j = int(np.searchsorted(stamps_ns, np.int64(t_ns), side="right"))
    if j == 0:
        return None
    sub = {k: v[:j] for k, v in cols.items()}
    return telemetry_features(stamps_ns[:j], sub)[-1]
