"""Supplied 3D track borders: candidate selection, layer (elevation) gating, nominal attitude from the borders.

Inputs are the four static polylines of `track_bounds` (allowed static metadata; map frame, metres). Nothing here reads GT
or telemetry. The borders carry no occlusion or layer information; the elevation caveats of DEVELOPMENT_PLAN.md section 6
are handled conservatively:

* every hypothesis group ('main' = track_left+track_right, 'pit' = pitlane_left+pitlane_right) has its own road height,
  tracked causally with a continuity limit (`z_prev`); a layer jump larger than the allowed slope reports `layer_ok=False`
  and the caller abstains;
* candidate border points must lie within `dz_gate` of the predicted road height (plus a slope allowance with range), so
  overhead crossings and tunnel ceilings of the other layer are excluded;
* road pitch (grade) and roll (banking) are *fixed approximations* computed from the borders around the predicted pose;
  the model adds a bounded pitch nuisance on top. No GT z/roll/pitch is used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

GROUPS = {"main": ("track_left", "track_right"), "pit": ("pitlane_left", "pitlane_right")}
BASE_BELOW_ROAD_M = 0.047   # base_link sits about 0.047 m below the road z of the borders (nominal design constant)


@dataclass
class Strip:
    name: str
    xyz: np.ndarray
    tree: cKDTree
    closed: bool


@dataclass
class RoadState:
    z_road: float
    ok: bool
    n_layers: int
    reason: str = ""


@dataclass
class BorderRuns:
    """Candidate border points of one hypothesis group: runs of consecutive polyline indices."""
    group: str
    strip: list = field(default_factory=list)     # strip name per point
    index: np.ndarray = None                      # polyline index per point
    xyz: np.ndarray = None                        # (N, 3)
    run_id: np.ndarray = None                     # consecutive-run label per point (same strip, index step 1)

    def __len__(self):
        return 0 if self.xyz is None else len(self.xyz)


class BorderMap:
    def __init__(self, track_bounds: dict):
        self.strips: dict[str, Strip] = {}
        for name in ("track_left", "track_right", "pitlane_left", "pitlane_right"):
            xyz = np.asarray(track_bounds[name], float).reshape(-1, 3)
            closed = bool(np.linalg.norm(xyz[0] - xyz[-1]) < 1e-6)
            self.strips[name] = Strip(name, xyz, cKDTree(xyz[:, :2]), closed)

    # ------------------------------------------------------------------ road height per group
    def _group_pts(self, group: str, xy: np.ndarray, radius: float):
        """z values and indices of group points within `radius` of xy (both strips)."""
        zs, which = [], []
        for name in GROUPS[group]:
            s = self.strips[name]
            idx = s.tree.query_ball_point(xy, radius)
            if idx:
                zs.append(s.xyz[idx, 2])
                which += [name] * len(idx)
        return (np.concatenate(zs) if zs else np.zeros(0)), which

    def road_height(self, group: str, xy, z_prev: Optional[float], radius: float = 10.0, max_dz: float = 1.2,
                    gap: float = 1.5) -> RoadState:
        """Road height of the group near xy. Clusters the border z within `radius` (gap > `gap` m splits layers) and picks the
        cluster closest to `z_prev` (nearest cluster to the nearest border point when z_prev is None). `ok` is False when no
        border is near, or when the best cluster is further than `max_dz` from z_prev."""
        xy = np.asarray(xy, float)
        z, _ = self._group_pts(group, xy, radius)
        if len(z) == 0:
            return RoadState(z_prev if z_prev is not None else 0.0, False, 0, "no_border_near")
        zs = np.sort(z)
        cut = np.where(np.diff(zs) > gap)[0]
        clusters = np.split(zs, cut + 1)
        cz = np.array([float(np.median(c)) for c in clusters])
        if z_prev is None:
            # reference = z of the nearest border point of the group
            best_d, ref = 1e18, cz[0]
            for name in GROUPS[group]:
                s = self.strips[name]
                d, i = s.tree.query(xy)
                if d < best_d:
                    best_d, ref = d, float(s.xyz[i, 2])
            k = int(np.argmin(np.abs(cz - ref)))
            return RoadState(float(cz[k]), True, len(cz))
        k = int(np.argmin(np.abs(cz - z_prev)))
        ok = abs(cz[k] - z_prev) <= max_dz
        return RoadState(float(cz[k]), ok, len(cz), "" if ok else "layer_jump")

    def nearest_group_distance(self, group: str, xy) -> float:
        xy = np.asarray(xy, float)
        return float(min(self.strips[n].tree.query(xy)[0] for n in GROUPS[group]))

    # ------------------------------------------------------------------ nominal attitude from the borders
    def road_attitude(self, group: str, x: float, y: float, yaw: float, z_road: float, ahead_m: float = 8.0):
        """(pitch, roll) fixed approximations from the borders around the pose, in the T = Rz Ry Rx convention
        (pitch > 0 nose down, roll > 0 left side up). Returns (0, 0) when the borders do not support an estimate."""
        c, s = math.cos(yaw), math.sin(yaw)
        zs = []
        for d in (-ahead_m, ahead_m):
            p = np.array([x + d * c, y + d * s])
            vals = []
            for name in GROUPS[group]:
                st = self.strips[name]
                dist, i = st.tree.query(p)
                if dist < 12.0 and abs(st.xyz[i, 2] - z_road) < 2.5:
                    vals.append(st.xyz[i, 2])
            zs.append(float(np.mean(vals)) if vals else None)
        pitch = 0.0
        if zs[0] is not None and zs[1] is not None:
            pitch = -math.atan2(zs[1] - zs[0], 2 * ahead_m)
        roll = 0.0
        left, right = self.strips[GROUPS[group][0]], self.strips[GROUPS[group][1]]
        dl, il = left.tree.query([x, y])
        dr, ir = right.tree.query([x, y])
        if dl < 15 and dr < 15 and abs(left.xyz[il, 2] - z_road) < 2.5 and abs(right.xyz[ir, 2] - z_road) < 2.5:
            width = float(np.linalg.norm(left.xyz[il, :2] - right.xyz[ir, :2]))
            if width > 4.0:
                roll = math.atan2(float(left.xyz[il, 2] - right.xyz[ir, 2]), width)
        return float(np.clip(pitch, -0.08, 0.08)), float(np.clip(roll, -0.08, 0.08))

    # ------------------------------------------------------------------ candidates
    def candidates(self, group: str, x: float, y: float, yaw: float, z_road: float, max_range: float = 70.0,
                   dz_gate: float = 1.5, slope_allow: float = 0.06, thin: int = 1) -> BorderRuns:
        """Border points of the group within `max_range` of (x, y) whose z is within dz_gate + slope_allow * range of the
        predicted road height z_road. `thin` keeps every `thin`-th polyline index (0.5 m spacing in the supplied data)."""
        out_strip, out_idx, out_xyz = [], [], []
        for name in GROUPS[group]:
            s = self.strips[name]
            idx = np.array(sorted(s.tree.query_ball_point([x, y], max_range)), dtype=int)
            if len(idx) == 0:
                continue
            if thin > 1:
                idx = idx[idx % thin == 0]
            pts = s.xyz[idx]
            rng = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
            keep = np.abs(pts[:, 2] - z_road) <= dz_gate + slope_allow * rng
            idx, pts = idx[keep], pts[keep]
            out_strip += [name] * len(idx)
            out_idx.append(idx)
            out_xyz.append(pts)
        if not out_idx:
            return BorderRuns(group)
        index = np.concatenate(out_idx)
        xyz = np.concatenate(out_xyz)
        # run labels: new run when the strip changes or the polyline index jumps (more than `thin` steps)
        run = np.zeros(len(index), dtype=int)
        r = 0
        for i in range(1, len(index)):
            if out_strip[i] != out_strip[i - 1] or abs(int(index[i]) - int(index[i - 1])) > thin:
                r += 1
            run[i] = r
        return BorderRuns(group, out_strip, index, xyz, run)


    # ------------------------------------------------------------------ corridor (inside-the-borders) test
    def corridor(self, group: str, x: float, y: float, z_road: float, max_dz: float = 2.0, k: int = 12):
        """Position relative to the border corridor of the group at (x, y): t in [0, 1] is inside (0 = right border, 1 = left
        border), width of the corridor, the travel-direction heading of the left strip there, and the two border points.
        Only border points within `max_dz` of the road height are used (layer gate). None when no border pair is near."""
        L, R = self.strips[GROUPS[group][0]], self.strips[GROUPS[group][1]]
        out = []
        for st in (L, R):
            d, idx = st.tree.query([x, y], k=k)
            idx = np.atleast_1d(idx)
            d = np.atleast_1d(d)
            ok = [(float(dd), int(i)) for dd, i in zip(d, idx) if i < len(st.xyz) and abs(st.xyz[i, 2] - z_road) <= max_dz]
            if not ok:
                return None
            out.append(min(ok))
        (dl, il), (dr, ir) = out
        pL, pR = L.xyz[il, :2], R.xyz[ir, :2]
        w = pL - pR
        ww = float(w @ w)
        if ww < 1.0 or math.sqrt(ww) > 60.0:
            return None
        t = float((np.array([x, y]) - pR) @ w / ww)
        i0, i1 = max(il - 6, 0), min(il + 6, len(L.xyz) - 1)
        dv = L.xyz[i1, :2] - L.xyz[i0, :2]
        theta = math.atan2(float(dv[1]), float(dv[0]))
        return {"t": t, "width": math.sqrt(ww), "theta": theta, "pL": pL, "pR": pR, "d_left": dl, "d_right": dr}
