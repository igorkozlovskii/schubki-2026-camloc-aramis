"""Training-only keyframe selection, GT pose interpolation (OFFLINE map builder / evaluator side only) and coverage gaps.

This module reads ground-truth training poses and therefore must never be imported by the inference path
(`camloc.models.*`, `camloc.runner`). A test asserts that. Frame eligibility comes from the frozen S01 frame table
(`train_eligible`, never `~embargo`); dev/audit frames cannot enter (`select_keyframes` asserts role == train).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_LAPS = (0, 1, 3, 5)


@dataclass
class GtPoses:
    """6-DoF base_link poses in map: x y z roll pitch yaw per GT stamp (derived/gt_tf.parquet), plus sigma_xy."""
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    roll: np.ndarray
    pitch: np.ndarray
    yaw_unwrapped: np.ndarray
    sigma_xy: np.ndarray            # nearest-sample pos_accuracy (state_estimation), offline quality filter only

    def at(self, t_ns) -> dict:
        t_ns = np.atleast_1d(np.asarray(t_ns, np.int64))
        f = lambda a: np.interp(t_ns.astype(np.float64) - self.t[0], self.t.astype(np.float64) - self.t[0], a)
        yaw = f(self.yaw_unwrapped)
        j = np.clip(np.searchsorted(self.t, t_ns), 1, len(self.t) - 1)
        # quality: worst sigma of the two bracketing GT samples
        sig = np.maximum(self.sigma_xy[j - 1], self.sigma_xy[j])
        return {"x": f(self.x), "y": f(self.y), "z": f(self.z), "roll": f(self.roll), "pitch": f(self.pitch),
                "yaw": (yaw + np.pi) % (2 * np.pi) - np.pi, "sigma_xy": sig}


def load_gt_poses(root: Path) -> GtPoses:
    root = Path(root)
    g = pd.read_parquet(root / "derived/gt_tf.parquet").sort_values("timestamp_ns").reset_index(drop=True)
    s = pd.read_parquet(root / "derived/state_estimation.parquet").sort_values("timestamp_ns")
    sig = np.hypot(s["pos_accuracy.0"].to_numpy(), s["pos_accuracy.1"].to_numpy())
    # nearest state_estimation sample per GT stamp
    st = s["timestamp_ns"].to_numpy()
    gt_t = g["timestamp_ns"].to_numpy()
    j = np.clip(np.searchsorted(st, gt_t), 1, len(st) - 1)
    near = np.where(np.abs(st[j - 1] - gt_t) <= np.abs(st[j] - gt_t), j - 1, j)
    return GtPoses(gt_t.astype(np.int64), g.x.to_numpy(float), g.y.to_numpy(float), g.z.to_numpy(float), g.roll.to_numpy(float),
                   g.pitch.to_numpy(float), np.unwrap(g.yaw.to_numpy(float)), sig[near])


def rot_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) Ry(pitch) Rx(roll) (same convention as geometry.camera)."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def T_map_base(x, y, z, roll, pitch, yaw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot_zyx(roll, pitch, yaw)
    T[:3, 3] = (x, y, z)
    return T


def select_keyframes(frames: pd.DataFrame, gt: GtPoses, laps=TRAIN_LAPS, camera: str = "fl", hz: float = 2.0,
                     min_dist_m: float = 4.0, max_dist_m: float = 30.0, max_sigma_m: float = 1.0):
    """Select keyframes from train-eligible frames of `laps` (never dev/audit/embargo).

    Rule (density by distance/speed): walk the camera's frames in time order; take a frame when
    (elapsed >= 1/hz and travelled >= min_dist) or travelled >= max_dist since the previous keyframe of the same lap.
    Frames whose GT sigma_xy exceeds `max_sigma_m` are not eligible (reported in `filtered`).
    Returns (keyframes DataFrame, filtered DataFrame with reason).
    """
    f = frames[(frames.camera == camera)].copy()
    assert set(f[f.lap.isin(laps)].role.unique()) <= {"train"}, "only train-role frames may be keyframe candidates"
    bad_laps = set(laps) - set(TRAIN_LAPS)
    assert not bad_laps, f"laps {bad_laps} are not training laps"
    cand = f[f.lap.isin(laps)]
    rows, filt = [], []
    for lap, d in cand.groupby("lap"):
        d = d.sort_values("stamp_ns")
        ok = d.train_eligible.to_numpy()
        for r in d[~d.train_eligible].itertuples():
            filt.append({"stamp_ns": r.stamp_ns, "lap": lap, "reason": "not_train_eligible(embargo/outside_gt)"})
        d = d[ok]
        p = gt.at(d.stamp_ns.to_numpy())
        last = None
        for i, r in enumerate(d.itertuples()):
            if p["sigma_xy"][i] > max_sigma_m:
                filt.append({"stamp_ns": r.stamp_ns, "lap": lap, "reason": f"gt_sigma_xy>{max_sigma_m}"})
                continue
            if last is None:
                take = True
            else:
                el = (r.stamp_ns - last[0]) / 1e9
                dist = math.hypot(p["x"][i] - last[1], p["y"][i] - last[2])
                take = (el >= 1.0 / hz and dist >= min_dist_m) or dist >= max_dist_m
            if take:
                rows.append({"stamp_ns": int(r.stamp_ns), "lap": int(lap), "image_row": int(r.image_row), "x": p["x"][i], "y": p["y"][i],
                             "z": p["z"][i], "roll": p["roll"][i], "pitch": p["pitch"][i], "yaw": p["yaw"][i],
                             "sigma_xy": p["sigma_xy"][i]})
                last = (r.stamp_ns, p["x"][i], p["y"][i])
    kf = pd.DataFrame(rows).sort_values(["lap", "stamp_ns"]).reset_index(drop=True)
    kf.insert(0, "kf_id", np.arange(len(kf)))
    return kf, pd.DataFrame(filt)


def angdiff(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi)


def coverage_along(path_xy: np.ndarray, path_yaw: np.ndarray, kf_xy: np.ndarray, kf_yaw: np.ndarray, radius_m: float, yaw_tol: float):
    """Boolean per path sample: is there a keyframe within radius and heading tolerance (evaluator-side diagnostic)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(kf_xy)
    out = np.zeros(len(path_xy), bool)
    nn = tree.query_ball_point(path_xy, r=radius_m)
    for i, idx in enumerate(nn):
        if idx and (angdiff(kf_yaw[idx], path_yaw[i]) <= yaw_tol).any():
            out[i] = True
    return out


def gap_intervals(arclen: np.ndarray, covered: np.ndarray, min_gap_m: float = 20.0):
    """Maximal uncovered runs along the arclength with length >= min_gap_m: list of (s_start, s_end, length)."""
    gaps, start = [], None
    for i, c in enumerate(covered):
        if not c and start is None:
            start = i
        if (c or i == len(covered) - 1) and start is not None:
            end = i if c else i
            L = arclen[end] - arclen[start]
            if L >= min_gap_m:
                gaps.append((float(arclen[start]), float(arclen[end]), float(L)))
            start = None
    return gaps
