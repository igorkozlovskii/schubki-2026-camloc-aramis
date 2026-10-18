"""Typed observation, pose, capability and protocol contracts (DEVELOPMENT_PLAN.md sections 2.2, 5.1, 5.2).

A model sees only: the first pose (once), static metadata, and `FrameBundle`s whose frames are stamped <= the
bundle timestamp. Nothing here can carry ground truth, telemetry or evaluator data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import numpy as np

CAMERAS = ("fl", "fr", "cl", "cr", "rl", "rr", "r")
CLOCK_CAMERA = "fl"
STALE_CUTOFF_NS = 150_000_000

FIRST_POSE_SCHEMA = "first-pose-v1"
FIRST_POSE_KEYS = {"schema", "timestamp_ns", "x", "y", "yaw", "provenance"}  # anything else is refused


class ContractError(RuntimeError):
    """A contract was violated (forbidden field, non-causal data, second initialization, bad result ...)."""


@dataclass(frozen=True)
class FirstPose:
    """The organizer-provided initial pose only. No height, pitch, roll or velocity."""
    timestamp_ns: int
    x: float
    y: float
    yaw: float

    def __post_init__(self):
        if not isinstance(self.timestamp_ns, (int, np.integer)) or isinstance(self.timestamp_ns, bool):
            raise ContractError("first pose timestamp_ns must be an integer")
        if not all(math.isfinite(v) for v in (self.x, self.y, self.yaw)):
            raise ContractError("first pose x,y,yaw must be finite")

    @staticmethod
    def from_json_dict(d: dict) -> "FirstPose":
        extra = set(d) - FIRST_POSE_KEYS
        if extra:
            raise ContractError(f"first-pose file has unknown/forbidden keys: {sorted(extra)}")
        if d.get("schema") != FIRST_POSE_SCHEMA:
            raise ContractError(f"first-pose schema must be {FIRST_POSE_SCHEMA!r}")
        missing = {"timestamp_ns", "x", "y", "yaw"} - set(d)
        if missing:
            raise ContractError(f"first-pose file misses {sorted(missing)}")
        return FirstPose(int(d["timestamp_ns"]), float(d["x"]), float(d["y"]), float(d["yaw"]))


@dataclass
class CameraFrame:
    """One camera at one tick. `status`: 'ok' (present, age <= stale cutoff), 'stale' (older than the cutoff:
    metadata only, pixels refused), 'missing' (no image with stamp <= t). Pixels are loaded lazily."""
    camera: str
    status: str
    stamp_ns: Optional[int] = None
    age_ns: Optional[int] = None
    _loader: Optional[Callable[[], bytes]] = field(default=None, repr=False, compare=False)

    @property
    def available(self) -> bool:
        return self.status == "ok"

    def jpeg(self) -> bytes:
        if self.status != "ok" or self._loader is None:
            raise ContractError(f"camera {self.camera} is {self.status}: pixels are not available")
        return self._loader()

    def image(self, reduce: int = 1) -> np.ndarray:
        """Decode to an RGB uint8 array (H, W, 3); reduce in {1, 2, 4, 8} uses OpenCV reduced decoding."""
        import cv2
        flags = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4,
                 8: cv2.IMREAD_REDUCED_COLOR_8}
        bgr = cv2.imdecode(np.frombuffer(self.jpeg(), np.uint8), flags[reduce])
        if bgr is None:
            raise ContractError(f"camera {self.camera}: JPEG decode failed")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@dataclass
class FrameBundle:
    """Output-clock tick t (int64 ns, exposure stamp of the clock camera) with the latest frame per camera <= t."""
    timestamp_ns: int
    tick_index: int
    frames: dict[str, CameraFrame]
    stale_cutoff_ns: int = STALE_CUTOFF_NS

    def __post_init__(self):
        for c, f in self.frames.items():
            if f.stamp_ns is not None and f.stamp_ns > self.timestamp_ns:
                raise ContractError(f"future image exposed: camera {c} stamp {f.stamp_ns} > t {self.timestamp_ns}")

    def available(self) -> list[str]:
        return [c for c, f in self.frames.items() if f.available]

    def ages_ns(self) -> dict[str, Optional[int]]:
        return {c: f.age_ns for c, f in self.frames.items()}

    def statuses(self) -> dict[str, str]:
        return {c: f.status for c, f in self.frames.items()}


@dataclass(frozen=True)
class CameraStatic:
    width: int
    height: int
    K: tuple
    D: tuple
    P: tuple
    distortion_model: str
    frame_id: str
    T_base_cam: Any  # camloc.geometry.Transform (base_link <- camera optical frame)


@dataclass(frozen=True)
class StaticMetadata:
    """Allowed static metadata: intrinsics/extrinsics and the supplied 3D borders. Hashes are for provenance."""
    cameras: dict[str, CameraStatic]
    track_bounds: dict[str, list]
    map_id: str
    calibration_sha256: str
    track_bounds_sha256: str
    store_id: str


@dataclass(frozen=True)
class Capabilities:
    """What a model declares; the runner refuses a model that asks for more than the contract allows."""
    cameras: tuple = ()
    clock_camera: str = CLOCK_CAMERA
    calibration: bool = False
    map_id: Optional[str] = None
    telemetry_fields: tuple = ()          # camera-only: must be empty (D-001)
    device: str = "cpu"
    assets: tuple = ()
    scale_status: str = "metric"          # metric | arbitrary | none
    online_mapping: str = "disabled"      # disabled | session_local
    telemetry: bool = False               # S10c, additive, default off: the model reads separately exported telemetry-v1 fields (D-024)

    def validate(self, store_cameras=CAMERAS, allow_telemetry=False):
        if (self.telemetry_fields or self.telemetry) and not allow_telemetry:
            raise ContractError(f"telemetry fields are disabled pending Q1: {self.telemetry_fields}")
        if self.telemetry_fields and not self.telemetry:
            raise ContractError("telemetry_fields declared without telemetry: true")
        if self.telemetry:
            from camloc.io.telemetry import ALLOWED_FIELDS, assert_no_forbidden
            assert_no_forbidden(self.telemetry_fields)
            outside = [f for f in self.telemetry_fields if f not in ALLOWED_FIELDS]
            if outside:
                raise ContractError(f"telemetry fields outside the telemetry-v1 allowlist: {outside}")
        bad = set(self.cameras) - set(store_cameras)
        if bad:
            raise ContractError(f"model needs cameras the store does not have: {sorted(bad)}")
        if self.clock_camera not in store_cameras:
            raise ContractError(f"clock camera {self.clock_camera!r} is not in the store")
        if self.online_mapping not in ("disabled", "session_local"):
            raise ContractError(f"online_mapping must be disabled|session_local, got {self.online_mapping!r}")
        if self.scale_status not in ("metric", "arbitrary", "none"):
            raise ContractError(f"scale_status must be metric|arbitrary|none, got {self.scale_status!r}")


@dataclass
class StepResult:
    """Pose of constructor0/base_link in map at the bundle stamp. Detailed diagnostics are separate (sidecar)."""
    x: float
    y: float
    yaw: float
    source_flags: tuple = ()
    tracking_state: str = "tracking"
    covariance: Optional[list] = None               # 3x3 over (x, y, yaw), optional
    consumed_min_ns: Optional[int] = None           # earliest input stamp used for this pose
    consumed_max_ns: Optional[int] = None           # latest input stamp used (must be <= t)
    residuals: dict = field(default_factory=dict)

    def validate(self, t_ns: int):
        for k in ("x", "y", "yaw"):
            v = getattr(self, k)
            if not isinstance(v, (int, float, np.floating, np.integer)) or not math.isfinite(float(v)):
                raise ContractError(f"result {k} must be finite, got {v!r}")
        if self.consumed_max_ns is not None and self.consumed_max_ns > t_ns:
            raise ContractError(f"non-causal: consumed_max_ns {self.consumed_max_ns} > t {t_ns}")
        if self.consumed_min_ns is not None and self.consumed_max_ns is not None and self.consumed_min_ns > self.consumed_max_ns:
            raise ContractError("consumed_min_ns > consumed_max_ns")
        if self.covariance is not None:
            c = np.asarray(self.covariance, dtype=float)
            if c.shape != (3, 3) or not np.isfinite(c).all():
                raise ContractError("covariance must be a finite 3x3 matrix")


@runtime_checkable
class Model(Protocol):
    """create_model(config, assets) -> Model; initialize(first_pose, static_metadata); step(bundle); close()."""
    capabilities: Capabilities

    def initialize(self, first_pose: FirstPose, static_metadata: StaticMetadata) -> None: ...
    def step(self, bundle: FrameBundle) -> StepResult: ...
    def close(self) -> None: ...
