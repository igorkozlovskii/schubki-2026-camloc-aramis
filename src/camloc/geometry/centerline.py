"""Centerline of the supplied main-track borders: arc length s, heading and curvature (static map geometry only).

Built from `track_left` / `track_right` only (allowed static metadata): each left vertex is paired with the nearest right
vertex of the same elevation layer, the midpoints are smoothed and resampled at 1 m. `project` maps a planar position to
(s, n, theta, kappa) with a continuity window around a hint so that the two layers of the circuit (overhead crossing) are
not confused. The racing line reconstructions in `derived/refline_*` are NOT used (forbidden for inference).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


def _smooth_closed(p: np.ndarray, k: int) -> np.ndarray:
    out = np.empty_like(p)
    n = len(p)
    for i in range(p.shape[1]):
        ext = np.r_[p[-k:, i], p[:, i], p[:k, i]]
        c = np.cumsum(np.r_[0.0, ext])
        out[:, i] = (c[2 * k + 1:] - c[:-2 * k - 1])[:n] / (2 * k + 1)
    return out


class Centerline:
    def __init__(self, track_bounds: dict, step_m: float = 1.0, max_dz: float = 1.5):
        L = np.asarray(track_bounds["track_left"], float)
        R = np.asarray(track_bounds["track_right"], float)
        tree = cKDTree(R[:, :2])
        mids = []
        for p in L[:-1]:
            idx = tree.query_ball_point(p[:2], 30.0)
            if not idx:
                mids.append(None)
                continue
            q = R[idx]
            ok = np.abs(q[:, 2] - p[2]) < max_dz
            if not ok.any():
                mids.append(None)
                continue
            q = q[ok]
            j = int(np.argmin(np.hypot(q[:, 0] - p[0], q[:, 1] - p[1])))
            mids.append(0.5 * (p + q[j]))
        good = [m for m in mids if m is not None]
        M = np.asarray(good)
        # resample at step_m along the polyline (closed)
        seg = np.hypot(*np.diff(np.vstack([M, M[:1]])[:, :2], axis=0).T)
        s_raw = np.r_[0.0, np.cumsum(seg)]
        total = float(s_raw[-1])
        ss = np.arange(0.0, total, step_m)
        Mc = np.vstack([M, M[:1]])
        xyz = np.stack([np.interp(ss, s_raw, Mc[:, i]) for i in range(3)], axis=1)
        xyz = _smooth_closed(xyz, 6)
        self.xyz = xyz
        self.step = step_m
        self.length = len(xyz) * step_m
        d = np.diff(np.vstack([xyz, xyz[:1]])[:, :2], axis=0)
        self.theta = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        self.kappa = np.gradient(_smooth_closed(self.theta[:, None], 5)[:, 0], step_m)
        self.tree = cKDTree(xyz[:, :2])

    def project(self, x: float, y: float, s_hint: float | None = None, window_m: float = 150.0, z: float | None = None,
                max_dz: float = 3.0):
        """(s, n, theta, kappa) of the nearest centerline vertex (n > 0 = left of the travel direction). With `s_hint`
        only vertices within +-window_m of the hint (cyclic) are considered; with `z` vertices of another layer are skipped."""
        n_v = len(self.xyz)
        if s_hint is None:
            idx = np.array(self.tree.query_ball_point([x, y], 40.0), dtype=int)
            if len(idx) == 0:
                _, i = self.tree.query([x, y])
                idx = np.array([int(i)])
        else:
            i0 = int(round(s_hint / self.step)) % n_v
            w = int(window_m / self.step)
            idx = (np.arange(i0 - w, i0 + w + 1)) % n_v
        if z is not None:
            ok = np.abs(self.xyz[idx, 2] - z) < max_dz
            if ok.any():
                idx = idx[ok]
        d = np.hypot(self.xyz[idx, 0] - x, self.xyz[idx, 1] - y)
        j = int(idx[int(np.argmin(d))])
        th = float(self.theta[j])
        dx, dy = x - self.xyz[j, 0], y - self.xyz[j, 1]
        n = -dx * math.sin(th) + dy * math.cos(th)
        return j * self.step, float(n), th, float(self.kappa[j])

    def wrap_s(self, s: float) -> float:
        return s % self.length

    def ds(self, s_a: float, s_b: float) -> float:
        """Signed shortest arc difference s_b - s_a on the loop."""
        d = (s_b - s_a) % self.length
        return d - self.length if d > 0.5 * self.length else d
