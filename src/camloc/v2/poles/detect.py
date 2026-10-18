"""GT-free detector of near-vertical line segments (S2.5 stage 1, option a).

Pipeline per image (half resolution, as decoded with IMREAD_REDUCED_*_2):
  gray -> CLAHE -> OpenCV LSD -> keep segments of length >= min_len_px whose image direction points at the vertical vanishing
  point of the camera (level-car extrinsics, angle <= max_vp_deg, measured in undistorted normalised coordinates) -> one row per
  segment with its base-frame azimuth (mean of the two endpoint rays), endpoint elevations and pixels.

Masking (car body) is applied by the caller (`apply_mask`) because masks are fold-specific; the detector itself never reads masks,
poses or ground truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

COLUMNS = ("u1", "v1", "u2", "v2", "len_px", "width_px", "vp_deg", "az", "el_top", "el_bot", "contrast", "pol")


@dataclass(frozen=True)
class DetConfig:
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    min_len_px: float = 15.0       # half-resolution pixels
    max_vp_deg: float = 3.0        # angle between the segment and the direction to the vertical vanishing point
    reduce: int = 2                # image decimation of the input (pixel coordinates are converted to full resolution)
    min_contrast: float = 6.0      # mean |gray difference| across the segment (CLAHE image), grey levels


_LSD = {}


def _lsd():
    if "d" not in _LSD:
        _LSD["d"] = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    return _LSD["d"]


def _clahe(cfg):
    k = (cfg.clahe_clip, cfg.clahe_tile)
    if k not in _LSD:
        _LSD[k] = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_tile, cfg.clahe_tile))
    return _LSD[k]


def base_rays(cam, uv_full: np.ndarray) -> np.ndarray:
    """Full-resolution pixels (N, 2) -> unit rays (N, 3) in the base frame (level car)."""
    xn = cam.undistort_normalized(uv_full)
    rc = np.column_stack([xn, np.ones(len(xn))])
    rb = rc @ cam.T_base_cam.R.T
    return rb / np.linalg.norm(rb, axis=1, keepdims=True)


def vertical_vp(cam) -> np.ndarray:
    """Base +z axis in the optical frame (homogeneous vanishing point in normalised coordinates)."""
    return cam.T_base_cam.R.T @ np.array([0.0, 0.0, 1.0])


def _contrast(gray, seg):
    """Signed mean difference (image-right side minus image-left side) across each segment, 3 px off the line, grey levels.
    Image right = clockwise seen from above for an upright camera, so the sign (edge polarity) is a property of the 3D edge."""
    h, w = gray.shape
    out = np.zeros(len(seg))
    for k, (x1, y1, x2, y2) in enumerate(seg):
        L = math.hypot(x2 - x1, y2 - y1)
        if L < 1:
            continue
        nx, ny = -(y2 - y1) / L, (x2 - x1) / L
        t = np.linspace(0.1, 0.9, max(4, int(L // 3)))
        px, py = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        a = gray[np.clip((py + 3 * ny).astype(int), 0, h - 1), np.clip((px + 3 * nx).astype(int), 0, w - 1)].astype(float)
        b = gray[np.clip((py - 3 * ny).astype(int), 0, h - 1), np.clip((px - 3 * nx).astype(int), 0, w - 1)].astype(float)
        out[k] = float(np.mean(a - b)) * (1.0 if nx >= 0 else -1.0)
    return out


def detect_vertical(img: np.ndarray, cam, cfg: DetConfig = DetConfig()) -> np.ndarray:
    """img: half-resolution BGR or gray. Returns a structured float array (N, len(COLUMNS)); pixels u, v are HALF resolution."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = _clahe(cfg).apply(gray)
    lines, widths, _, _ = _lsd().detect(g)
    if lines is None or len(lines) == 0:
        return np.zeros((0, len(COLUMNS)))
    seg = lines.reshape(-1, 4).astype(float)
    wid = widths.reshape(-1) if widths is not None else np.ones(len(seg))
    L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    keep = L >= cfg.min_len_px
    seg, wid, L = seg[keep], wid[keep], L[keep]
    if len(seg) == 0:
        return np.zeros((0, len(COLUMNS)))
    f = float(cfg.reduce)
    p1 = seg[:, :2] * f + 0.5 * (f - 1)
    p2 = seg[:, 2:] * f + 0.5 * (f - 1)
    n1 = cam.undistort_normalized(p1)
    n2 = cam.undistort_normalized(p2)
    d = vertical_vp(cam)
    mid = 0.5 * (n1 + n2)
    tvp = d[:2][None, :] - d[2] * mid                # image direction from the midpoint towards the vanishing point
    sv = n2 - n1
    cosang = np.abs((tvp * sv).sum(1)) / (np.linalg.norm(tvp, axis=1) * np.linalg.norm(sv, axis=1) + 1e-12)
    vp_deg = np.degrees(np.arccos(np.clip(cosang, 0, 1)))
    ok = vp_deg <= cfg.max_vp_deg
    seg, wid, L, vp_deg, p1, p2 = seg[ok], wid[ok], L[ok], vp_deg[ok], p1[ok], p2[ok]
    if len(seg) == 0:
        return np.zeros((0, len(COLUMNS)))
    scon = _contrast(g, seg)
    con = np.abs(scon)
    ok = con >= cfg.min_contrast
    seg, wid, L, vp_deg, p1, p2, con, scon = seg[ok], wid[ok], L[ok], vp_deg[ok], p1[ok], p2[ok], con[ok], scon[ok]
    if len(seg) == 0:
        return np.zeros((0, len(COLUMNS)))
    r1, r2 = base_rays(cam, p1), base_rays(cam, p2)
    h1 = np.arctan2(r1[:, 1], r1[:, 0])
    h2 = np.arctan2(r2[:, 1], r2[:, 0])
    az = np.arctan2(np.sin(h1) + np.sin(h2), np.cos(h1) + np.cos(h2))
    e1 = np.arctan2(r1[:, 2], np.hypot(r1[:, 0], r1[:, 1]))
    e2 = np.arctan2(r2[:, 2], np.hypot(r2[:, 0], r2[:, 1]))
    return np.column_stack([seg, L, wid, vp_deg, az, np.maximum(e1, e2), np.minimum(e1, e2), con, np.sign(scon)])


def mask_keep(det: np.ndarray, usable: np.ndarray | None) -> np.ndarray:
    """Bool per segment (columns u1 v1 u2 v2 first): both endpoints and the midpoint on usable pixels (True = usable)."""
    if usable is None or len(det) == 0:
        return np.ones(len(det), bool)
    h, w = usable.shape
    ok = np.ones(len(det), bool)
    for t in (0.0, 0.5, 1.0):
        u = det[:, 0] + t * (det[:, 2] - det[:, 0])
        v = det[:, 1] + t * (det[:, 3] - det[:, 1])
        ok &= usable[np.clip(v.astype(int), 0, h - 1), np.clip(u.astype(int), 0, w - 1)]
    return ok


def apply_mask(det: np.ndarray, usable: np.ndarray | None) -> np.ndarray:
    """Drop segments with an endpoint or the midpoint on the car body."""
    return det[mask_keep(det, usable)]


def _rot_pr(pitch, roll):
    """(N, 3, 3) Ry(pitch) @ Rx(roll): level-base vectors -> yaw-only (gravity-aligned) frame."""
    pitch, roll = np.atleast_1d(pitch).astype(float), np.atleast_1d(roll).astype(float)
    cp, sp, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    R = np.zeros((len(pitch), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = cp, sp * sr, sp * cr
    R[:, 1, 1], R[:, 1, 2] = cr, -sr
    R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = -sp, cp * sr, cp * cr
    return R


def segment_bearings(cam, seg_half: np.ndarray, pitch=0.0, roll=0.0, reduce: int = 2):
    """Azimuth of the vertical plane through each segment, in the car's yaw-only frame, given the base attitude (pitch, roll).

    seg_half: (N, 4) half-resolution u1 v1 u2 v2. The plane through the camera centre and the segment has normal n = r1 x r2
    (rays rotated into the gravity-aligned frame); its horizontal trace (-n_y, n_x), signed towards the rays, is the bearing.
    Returns (az, el_top, el_bot) in rad. With pitch = roll = 0 this is the level-car model."""
    seg = np.asarray(seg_half, float).reshape(-1, 4)
    if len(seg) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    f = float(reduce)
    r1 = base_rays(cam, seg[:, :2] * f + 0.5 * (f - 1))
    r2 = base_rays(cam, seg[:, 2:] * f + 0.5 * (f - 1))
    R = _rot_pr(np.broadcast_to(pitch, len(seg)), np.broadcast_to(roll, len(seg)))
    r1 = np.einsum("nij,nj->ni", R, r1)
    r2 = np.einsum("nij,nj->ni", R, r2)
    n = np.cross(r1, r2)
    hx, hy = -n[:, 1], n[:, 0]
    s = np.sign(hx * (r1[:, 0] + r2[:, 0]) + hy * (r1[:, 1] + r2[:, 1]))
    s[s == 0] = 1.0
    az = np.arctan2(s * hy, s * hx)
    e1 = np.arctan2(r1[:, 2], np.hypot(r1[:, 0], r1[:, 1]))
    e2 = np.arctan2(r2[:, 2], np.hypot(r2[:, 0], r2[:, 1]))
    return az, np.maximum(e1, e2), np.minimum(e1, e2)
