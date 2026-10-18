"""Camera projection with analytic Jacobians (plumb_bob distortion, OpenCV convention).

Frame conventions (src/camloc/geometry/transforms.py, DEVELOPMENT_PLAN.md section 2.1): all frames are right-handed;
`map`/`base_link` have x forward, y left, z up; camera optical frames have z forward, x right, y down.
T_map_camera = T_map_base @ T_base_camera. The "right-hand -> left-hand note" of section 2.1 means: image pixel
coordinates (u right, v down) are a left-handed 2D system seen from the optical +z axis; projection therefore uses the
optical-frame convention above and never a base-frame y directly.

The pose parameterization used by the border update is

    T_map_base = T(x, y, z) Rz(yaw) Ry(pitch) Rx(roll)

with (x, y, yaw) the reported pose and (z, pitch, roll) *nuisance* terms (z, roll fixed from the supplied borders, pitch
a bounded nuisance with a prior). `project_points_pose` returns the Jacobian of the pixels w.r.t. (x, y, yaw, pitch).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .transforms import Transform


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def _drx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[0, 0, 0], [0, -s, -c], [0, c, -s]], float)


def _dry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[-s, 0, c], [0, 0, 0], [-c, 0, -s]], float)


def _drz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[-s, -c, 0], [c, -s, 0], [0, 0, 0]], float)


@dataclass(frozen=True)
class CameraModel:
    """Pinhole + plumb_bob (k1 k2 p1 p2 k3) with the extrinsics T_base_cam (base_link <- optical frame)."""
    name: str
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    T_base_cam: Transform
    z_min: float = 0.5  # points closer than this along the optical axis are never projected

    @staticmethod
    def from_static(name: str, cs) -> "CameraModel":
        """From a `camloc.contracts.CameraStatic`."""
        return CameraModel(name, int(cs.width), int(cs.height), np.array(cs.K, float).reshape(3, 3),
                           np.array(cs.D, float).ravel()[:5], cs.T_base_cam)

    # ---------------------------------------------------------------- projection of camera-frame points
    def distort(self, x: np.ndarray, y: np.ndarray):
        """Normalized undistorted (x, y) -> distorted normalized (xd, yd) and the 2x2 Jacobian d(xd,yd)/d(x,y)."""
        k1, k2, p1, p2, k3 = (self.D.tolist() + [0.0] * 5)[:5]
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        drad_dr2 = k1 + 2 * k2 * r2 + 3 * k3 * r2 * r2
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        dxd_dx = rad + x * drad_dr2 * 2 * x + 2 * p1 * y + p2 * (2 * x + 4 * x)
        dxd_dy = x * drad_dr2 * 2 * y + 2 * p1 * x + p2 * 2 * y
        dyd_dx = y * drad_dr2 * 2 * x + p1 * 2 * x + 2 * p2 * y
        dyd_dy = rad + y * drad_dr2 * 2 * y + p1 * (2 * y + 4 * y) + 2 * p2 * x
        J = np.stack([np.stack([dxd_dx, dxd_dy], -1), np.stack([dyd_dx, dyd_dy], -1)], -2)
        return xd, yd, J

    def project_cam(self, pc: np.ndarray):
        """Camera-frame points (N, 3) -> pixels (N, 2), in-front mask, and dUV/dPc (N, 2, 3).

        `valid` = z > z_min and the (distorted) pixel lies inside the image (with the plumb_bob monotonic-range guard:
        points whose normalized radius is beyond 1.15 x the image-corner radius are rejected, because the polynomial
        folds back far outside the image)."""
        pc = np.asarray(pc, float).reshape(-1, 3)
        z = pc[:, 2]
        front = z > self.z_min
        zs = np.where(front, z, 1.0)
        x, y = pc[:, 0] / zs, pc[:, 1] / zs
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        corner_r = max(math.hypot((0 - cx) / fx, (0 - cy) / fy), math.hypot((self.width - cx) / fx, (self.height - cy) / fy),
                       math.hypot((0 - cx) / fx, (self.height - cy) / fy), math.hypot((self.width - cx) / fx, (0 - cy) / fy))
        in_range = (x * x + y * y) < (1.15 * corner_r) ** 2
        xd, yd, Jd = self.distort(x, y)
        u = fx * xd + cx
        v = fy * yd + cy
        # d(x,y)/dPc
        dxy = np.zeros((len(pc), 2, 3))
        dxy[:, 0, 0] = 1 / zs
        dxy[:, 0, 2] = -pc[:, 0] / zs ** 2
        dxy[:, 1, 1] = 1 / zs
        dxy[:, 1, 2] = -pc[:, 1] / zs ** 2
        dd = Jd @ dxy
        J = np.stack([fx * dd[:, 0, :], fy * dd[:, 1, :]], axis=1)
        uv = np.stack([u, v], axis=1)
        inside = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return uv, front & in_range & inside, J

    # ---------------------------------------------------------------- rays
    def undistort_normalized(self, uv: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.undistortPoints(np.asarray(uv, np.float64).reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)

    # ---------------------------------------------------------------- pose chain
    def project_points_pose(self, p_map: np.ndarray, x: float, y: float, yaw: float, z: float = 0.0, pitch: float = 0.0,
                            roll: float = 0.0, with_jac: bool = True):
        """Project map points through the pose (x, y, yaw | z, pitch, roll).

        Returns uv (N,2), valid (N,), depth (N,), and J (N,2,4) = d uv / d (x, y, yaw, pitch) when with_jac."""
        p_map = np.asarray(p_map, float).reshape(-1, 3)
        R = _rz(yaw) @ _ry(pitch) @ _rx(roll)
        t = np.array([x, y, z])
        pb = (p_map - t) @ R                    # R^T (p - t), row-vector form
        Rbc, tbc = self.T_base_cam.R, self.T_base_cam.t
        pc = (pb - tbc) @ Rbc                   # Rbc^T (pb - tbc)
        uv, valid, Jc = self.project_cam(pc)
        if not with_jac:
            return uv, valid, pc[:, 2], None
        d = p_map - t
        # derivatives of pb = R^T d
        dpb_dx = -R.T[:, 0]
        dpb_dy = -R.T[:, 1]
        Rz, Ry, Rx = _rz(yaw), _ry(pitch), _rx(roll)
        dR_dyaw = _drz(yaw) @ Ry @ Rx
        dR_dpitch = Rz @ _dry(pitch) @ Rx
        dpb_dyaw = d @ dR_dyaw                  # (dR/dyaw)^T d
        dpb_dpitch = d @ dR_dpitch
        N = len(p_map)
        dpb = np.zeros((N, 3, 4))
        dpb[:, :, 0] = dpb_dx
        dpb[:, :, 1] = dpb_dy
        dpb[:, :, 2] = dpb_dyaw
        dpb[:, :, 3] = dpb_dpitch
        dpc = np.einsum("ji,njk->nik", Rbc, dpb)  # Rbc^T applied to every column
        J = np.einsum("nij,njk->nik", Jc, dpc)
        return uv, valid, pc[:, 2], J
