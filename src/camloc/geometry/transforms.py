"""Transform conventions (DEVELOPMENT_PLAN.md section 2.1).

`Transform(T_a_b)` is a 4x4 homogeneous matrix that maps coordinates expressed in frame b into frame a:
    p_a = T_a_b @ p_b            and            T_map_camera = T_map_base @ T_base_camera.
All frames are right-handed. `map` and `base_link`: x forward, y left, z up (REP 103).
Camera optical frames `constructor0/camera_*`: z forward, x right, y down.
Quaternions are (x, y, z, w) like ROS and `derived/calib.json::T_base_cam.q_xyzw`. Yaw is the rotation about +z
of base_link in map, radians. Only x, y, yaw are ever reported; z, roll, pitch of a camera pose are intermediate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return float(math.atan2(math.sin(a), math.cos(a)))


def _quat_to_rot(q_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in q_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("zero or non-finite quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass(frozen=True)
class Transform:
    """T_target_source as a 4x4 matrix; `frames=(target, source)` is a label for readability and checks."""
    matrix: np.ndarray
    target: str = ""
    source: str = ""

    def __post_init__(self):
        m = np.asarray(self.matrix, dtype=np.float64)
        if m.shape != (4, 4) or not np.isfinite(m).all():
            raise ValueError("transform must be a finite 4x4 matrix")
        if not np.allclose(m[3], [0, 0, 0, 1]):
            raise ValueError("last row must be [0 0 0 1]")
        object.__setattr__(self, "matrix", m)

    @staticmethod
    def from_quat_xyzw_t(q_xyzw, t, target: str = "", source: str = "") -> "Transform":
        m = np.eye(4)
        m[:3, :3] = _quat_to_rot(q_xyzw)
        m[:3, 3] = np.asarray(t, dtype=np.float64)
        return Transform(m, target, source)

    @property
    def R(self) -> np.ndarray:
        return self.matrix[:3, :3]

    @property
    def t(self) -> np.ndarray:
        return self.matrix[:3, 3]

    def inverse(self) -> "Transform":
        m = np.eye(4)
        m[:3, :3] = self.R.T
        m[:3, 3] = -self.R.T @ self.t
        return Transform(m, self.source, self.target)

    def __matmul__(self, other: "Transform") -> "Transform":
        """T_a_b @ T_b_c = T_a_c. Frame labels are checked when both are given."""
        if self.source and other.target and self.source != other.target:
            raise ValueError(f"frame mismatch: {self.target}<-{self.source} @ {other.target}<-{other.source}")
        return Transform(self.matrix @ other.matrix, self.target, other.source)

    def apply(self, p) -> np.ndarray:
        """Map points (3,) or (N,3) from the source frame into the target frame."""
        p = np.asarray(p, dtype=np.float64)
        return p @ self.R.T + self.t


def pose2d_to_transform(x: float, y: float, yaw: float, target: str = "map", source: str = "base_link") -> Transform:
    """T_map_base for a planar pose (z = 0, roll = pitch = 0)."""
    c, s = math.cos(yaw), math.sin(yaw)
    m = np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    return Transform(m, target, source)


def transform_to_pose2d(T_map_base: Transform) -> tuple[float, float, float]:
    """(x, y, yaw) of base_link in map: translation x,y and the heading of the base x axis projected on the map plane."""
    R = T_map_base.R
    return float(T_map_base.t[0]), float(T_map_base.t[1]), float(math.atan2(R[1, 0], R[0, 0]))


def base_pose_from_camera_pose(T_map_camera: Transform, T_base_camera: Transform) -> tuple[float, float, float]:
    """A camera PnP pose converted back to base_link (translation included) before scoring:
    T_map_base = T_map_camera @ inverse(T_base_camera)."""
    return transform_to_pose2d(T_map_camera @ T_base_camera.inverse())


def camera_pose_from_base_pose(x: float, y: float, yaw: float, T_base_camera: Transform) -> Transform:
    """T_map_camera = T_map_base @ T_base_camera."""
    return pose2d_to_transform(x, y, yaw) @ T_base_camera


def load_t_base_camera(calib: dict, camera: str) -> Transform:
    """Static extrinsics from derived/calib.json: cameras[cam]['T_base_cam'] = {t, q_xyzw} (base_link <- camera optical)."""
    e = calib["cameras"][camera]["T_base_cam"]
    return Transform.from_quat_xyzw_t(e["q_xyzw"], e["t"], target="base_link", source=f"camera_{camera}")
