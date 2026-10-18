"""`track-v1` frame: exact conversion between map pose (x, y, yaw) and track coordinates (s, n, dpsi).

Built on `camloc.geometry.centerline.Centerline` (centreline from `track_left` / `track_right` only; `derived/refline_*` is never used).
`Centerline.project` returns the nearest 1 m vertex, so its s is quantised and the along-track offset is dropped; this wrapper refines s with a
Newton iteration on the piecewise-linear centreline, which makes `to_xy(from_xy(p)) == p` to numerical precision (|n| * kappa < 1 required).

    c(s)      piecewise linear through the 1 m centreline vertices (closed loop)
    theta(s)  heading, linear between the vertex headings (circular difference across the +-pi seam)
    kappa(s)  curvature, linear between vertex values
    x, y = c(s) + n * N(s),  N = (-sin theta, cos theta) (left),  yaw = theta(s) + dpsi
"""
from __future__ import annotations

import math

import numpy as np

from camloc.geometry.centerline import Centerline

TWO_PI = 2.0 * math.pi


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class TrackFrame:
    def __init__(self, track_bounds: dict | None = None, centerline: Centerline | None = None):
        self.cl = centerline if centerline is not None else Centerline(track_bounds)
        self.step = float(self.cl.step)
        self.n_v = len(self.cl.xyz)
        self.length = float(self.cl.length)
        # python lists: scalar access is faster than numpy indexing in the per-substep hot path
        self._x = self.cl.xyz[:, 0].tolist()
        self._y = self.cl.xyz[:, 1].tolist()
        # Centerline.theta[i] is the heading of the SEGMENT i -> i+1; the heading at vertex i is the mean of the two adjacent segment headings (using the
        # segment heading as the vertex heading would shift theta(s) by half a vertex step, i.e. a systematic heading offset of kappa x 0.5 m).
        seg = self.cl.theta
        nv0 = len(seg)
        thv = np.array([seg[i - 1] + 0.5 * wrap(seg[i] - seg[i - 1]) for i in range(nv0)])
        self._thv = thv
        self._th = thv.tolist()
        # chord length between consecutive vertices / step: the box smoothing in Centerline shortens the polyline in bends (down to 0.93, 3.3 m cumulative), so s
        # (the vertex index x step, the track-v1 coordinate) is NOT metric arclength; the filter divides arclength rates by lam(s).
        nv = self.n_v
        xy = self.cl.xyz[:, :2]
        chord = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)
        self._lam = (chord / self.step).tolist()
        # heading difference to the next vertex (wrapped) so that interpolation is safe across the loop closure
        self._dth = [wrap(self._th[(i + 1) % self.n_v] - self._th[i]) for i in range(self.n_v)]
        # curvature per metre of arclength, constant on each segment and EXACTLY consistent with the linear theta(s): kappa_i = dtheta_i / chord_i. (A smoothed
        # curvature has the right total turning but is locally inconsistent with theta(s) in tight bends, which shows up as heading errors of several degrees.)
        self.kappa_arr = np.array(self._dth) / chord
        self._k = self.kappa_arr.tolist()

    # ---- evaluation at s ------------------------------------------------------------------------------------------
    def _idx(self, s: float) -> tuple[int, float]:
        u = (s % self.length) / self.step
        i = int(u)
        if i >= self.n_v:
            i = self.n_v - 1
        return i, u - i

    def theta(self, s: float) -> float:
        i, f = self._idx(s)
        return self._th[i] + f * self._dth[i]

    def kappa(self, s: float) -> float:
        return self._k[self._idx(s)[0]]

    def lam(self, s: float) -> float:
        """|dc/ds|: metres of centreline per unit of s (1 on straights, down to 0.93 in bends)."""
        i, f = self._idx(s)
        j = (i + 1) % self.n_v
        return self._lam[i] + f * (self._lam[j] - self._lam[i])

    def point(self, s: float) -> tuple[float, float]:
        i, f = self._idx(s)
        j = (i + 1) % self.n_v
        return self._x[i] + f * (self._x[j] - self._x[i]), self._y[i] + f * (self._y[j] - self._y[i])

    def to_xy(self, s: float, n: float, dpsi: float) -> tuple[float, float, float]:
        cx, cy = self.point(s)
        th = self.theta(s)
        return cx - n * math.sin(th), cy + n * math.cos(th), wrap(th + dpsi)

    def from_xy(self, x: float, y: float, yaw: float, s_hint: float | None = None, window_m: float = 150.0, iters: int = 40) -> tuple[float, float, float]:
        """(s, n, dpsi) of a map pose. `s_hint` limits the search to +-window_m (two circuit layers cross; unhinted = nearest vertex within 40 m)."""
        s0, _, _, _ = self.cl.project(x, y, s_hint=s_hint, window_m=window_m)
        s = float(s0)
        for _ in range(iters):
            cx, cy = self.point(s)
            th = self.theta(s)
            along = (x - cx) * math.cos(th) + (y - cy) * math.sin(th)
            s += along
            if abs(along) < 1e-11:
                break
        cx, cy = self.point(s)
        th = self.theta(s)
        n = -(x - cx) * math.sin(th) + (y - cy) * math.cos(th)
        return s % self.length, n, wrap(yaw - th)

    def ds(self, s_a: float, s_b: float) -> float:
        """Signed shortest arc difference s_b - s_a on the loop."""
        d = (s_b - s_a) % self.length
        return d - self.length if d > 0.5 * self.length else d
