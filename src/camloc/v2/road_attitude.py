"""S3.2c road-attitude map: the car's (dz, dpitch, droll) relative to the supplied-border attitude (`border_attitude.map_attitude`) as a
function of the along-track coordinate s, built offline from training-lap ground truth (D-017) and looked up at inference by the caller's
OWN s estimate. This module only stores and interpolates the table (periodic in s); it reads no ground truth. The builder, which does, is
scripts/v2_s32c_build_attitude_map.py (training laps 0/1/3/5, train-eligible frames only; fold k excludes lap k; never dev/audit laps)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class RoadAttitudeMap:
    def __init__(self, s: np.ndarray, delta: np.ndarray, length: float, meta: dict | None = None):
        self.s, self.delta, self.length, self.meta = np.asarray(s, float), np.asarray(delta, float), float(length), meta or {}   # delta (N, 3) = dz, dpitch, droll

    def lookup(self, s: float) -> np.ndarray:
        """(dz [m], dpitch [rad], droll [rad]) at s, linear and periodic."""
        x = float(s) % self.length
        return np.array([np.interp(x, self.s, self.delta[:, i], period=self.length) for i in range(3)])

    def save(self, path) -> None:
        np.savez(path, s=self.s, delta=self.delta, length=self.length, meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path) -> "RoadAttitudeMap":
        z = np.load(Path(path), allow_pickle=False)
        return cls(z["s"], z["delta"], float(z["length"]), json.loads(str(z["meta"])))
