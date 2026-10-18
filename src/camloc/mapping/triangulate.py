"""Two-view geometry with known (training GT) poses: Sampson filtering, triangulation, extrinsic-rotation fit.

Pure numpy/OpenCV/scipy; no network, no torch. Offline builder side (it is fed training GT poses by the builder).
Conventions: `T_map_base` 4x4 (base_link in map), `T_base_cam` 4x4 (optical camera in base_link),
`T_map_cam = T_map_base @ T_base_cam`; normalized coordinates are undistorted (x/z, y/z) of the optical frame.
"""
from __future__ import annotations

import math

import numpy as np


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def undistort_norm(K: np.ndarray, D: np.ndarray, uv: np.ndarray) -> np.ndarray:
    import cv2
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    if len(uv) == 0:
        return np.zeros((0, 2))
    return cv2.undistortPoints(np.asarray(uv, np.float64).reshape(-1, 1, 2), np.asarray(K, np.float64), np.asarray(D, np.float64)).reshape(-1, 2)


def rel_pose(T_map_cam_a: np.ndarray, T_map_cam_b: np.ndarray) -> np.ndarray:
    """T_b_a: maps points in camera a coordinates into camera b coordinates."""
    return np.linalg.inv(T_map_cam_b) @ T_map_cam_a


def sampson_px(xa: np.ndarray, xb: np.ndarray, T_b_a: np.ndarray, focal: float) -> np.ndarray:
    """Sampson distance (pixels, via focal) of normalized correspondences xa<->xb under the pose T_b_a."""
    E = skew(T_b_a[:3, 3]) @ T_b_a[:3, :3]
    ha = np.c_[xa, np.ones(len(xa))]
    hb = np.c_[xb, np.ones(len(xb))]
    Ea = ha @ E.T            # E x_a
    Etb = hb @ E             # E^T x_b
    num = np.einsum("ij,ij->i", hb, Ea) ** 2
    den = Ea[:, 0] ** 2 + Ea[:, 1] ** 2 + Etb[:, 0] ** 2 + Etb[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-18)) * focal


def triangulate(xa: np.ndarray, xb: np.ndarray, T_b_a: np.ndarray):
    """Linear (DLT) triangulation in camera a; returns X_a (N,3), depth_a, depth_b."""
    import cv2
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = T_b_a[:3, :]
    Xh = cv2.triangulatePoints(P1, P2, xa.T.astype(np.float64), xb.T.astype(np.float64))
    X = (Xh[:3] / Xh[3]).T
    Xb = X @ T_b_a[:3, :3].T + T_b_a[:3, 3]
    return X, X[:, 2], Xb[:, 2]


def reproj_px(X_a: np.ndarray, T_b_a: np.ndarray, xa: np.ndarray, xb: np.ndarray, focal: float):
    """Pixel reprojection errors (focal-scaled normalized distance) in both views."""
    za = X_a[:, 2]
    ea = np.hypot(*(X_a[:, :2] / za[:, None] - xa).T) * focal
    Xb = X_a @ T_b_a[:3, :3].T + T_b_a[:3, 3]
    eb = np.hypot(*(Xb[:, :2] / Xb[:, 2:3] - xb).T) * focal
    return ea, eb


def tri_angle_deg(X_a: np.ndarray, T_b_a: np.ndarray) -> np.ndarray:
    """Angle at the point between the rays to the two camera centres."""
    ca = np.zeros(3)
    cb = np.linalg.inv(T_b_a)[:3, 3]            # camera b centre in a
    ra, rb = ca - X_a, cb - X_a
    cosang = np.einsum("ij,ij->i", ra, rb) / np.maximum(np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def triangulate_pair(xa, xb, T_map_cam_a, T_map_cam_b, focal, *, max_sampson_px=4.0, max_reproj_px=4.0, min_angle_deg=1.5,
                     min_depth=2.0, max_depth=250.0, min_baseline_m=2.0):
    """Filter + triangulate matches of one covisible pair. Returns dict with boolean `keep`, X_map (N,3), angle (N,), and
    per-stage rejection counts. xa, xb are matched normalized coordinates (N,2)."""
    T_b_a = rel_pose(T_map_cam_a, T_map_cam_b)
    base = float(np.linalg.norm(T_b_a[:3, 3]))
    n = len(xa)
    rej = {"n": n, "baseline_m": base}
    out = {"keep": np.zeros(n, bool), "X_map": np.full((n, 3), np.nan), "angle": np.zeros(n), "reject": rej}
    if n == 0 or base < min_baseline_m:
        rej["baseline_too_small"] = n
        return out
    s = sampson_px(xa, xb, T_b_a, focal)
    ok = s <= max_sampson_px
    rej["sampson"] = int((~ok).sum())
    if ok.sum() < 3:
        return out
    X, da, db = triangulate(xa[ok], xb[ok], T_b_a)
    cheir = (da > min_depth) & (db > min_depth) & (da < max_depth) & np.isfinite(X).all(1)
    ang = tri_angle_deg(X, T_b_a)
    ea, eb = reproj_px(X, T_b_a, xa[ok], xb[ok], focal)
    good = cheir & (ang >= min_angle_deg) & (np.maximum(ea, eb) <= max_reproj_px)
    rej["cheirality_or_depth"] = int((~cheir).sum())
    rej["parallax"] = int((cheir & (ang < min_angle_deg)).sum())
    rej["reprojection"] = int((cheir & (ang >= min_angle_deg) & (np.maximum(ea, eb) > max_reproj_px)).sum())
    idx = np.flatnonzero(ok)[good]
    Xm = X[good] @ T_map_cam_a[:3, :3].T + T_map_cam_a[:3, 3]
    out["keep"][idx] = True
    out["X_map"][idx] = Xm
    out["angle"][idx] = ang[good]
    out["T_b_a"] = T_b_a
    return out


# --------------------------------------------------------------------------------------- extrinsic rotation fit
def rotvec_to_R(v) -> np.ndarray:
    import cv2
    return cv2.Rodrigues(np.asarray(v, np.float64).reshape(3, 1))[0]


def fit_extrinsic_rotation(samples, T_base_cam_nom: np.ndarray, focal: float, max_nfev: int = 60, loss_scale_px: float = 2.0):
    """Fit a camera-frame rotation correction d so that T_base_cam = T_nom with R = R_nom @ Exp(d), minimizing robust
    Sampson errors over training pairs. `samples` = list of (xa (N,2), xb (N,2), T_map_base_a, T_map_base_b).
    Returns (d (3,), T_base_cam_fit, info)."""
    from scipy.optimize import least_squares

    def T_cam(d):
        T = T_base_cam_nom.copy()
        T[:3, :3] = T_base_cam_nom[:3, :3] @ rotvec_to_R(d)
        return T

    def resid(d):
        Tc = T_cam(d)
        r = []
        for xa, xb, Ta, Tb in samples:
            r.append(sampson_px(xa, xb, rel_pose(Ta @ Tc, Tb @ Tc), focal))
        return np.concatenate(r)

    r0 = resid(np.zeros(3))
    sol = least_squares(resid, np.zeros(3), loss="soft_l1", f_scale=loss_scale_px, max_nfev=max_nfev, x_scale=0.01)
    r1 = resid(sol.x)
    info = {"d_rotvec_rad": sol.x.tolist(), "d_deg": float(np.degrees(np.linalg.norm(sol.x))), "n_corr": int(len(r0)),
            "median_sampson_px_before": float(np.median(r0)), "median_sampson_px_after": float(np.median(r1)),
            "p90_before": float(np.percentile(r0, 90)), "p90_after": float(np.percentile(r1, 90)), "nfev": int(sol.nfev)}
    return sol.x, T_cam(sol.x), info
