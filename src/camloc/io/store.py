"""Safe camera/static-only input store (cameras-v1) and its reader.

Layout of a store directory (nothing else is accepted - an unknown file or index column makes the loader refuse):

    jpeg_<cam>.bin       concatenated JPEG payloads, byte-for-byte from the MCAP (no re-encode), one file per camera
    index.csv            camera,stamp_ns,log_time_ns,offset,length,sha256   (stamp_ns = header.stamp = exposure)
    calib.json           verbatim copy of derived/calib.json (intrinsics, extrinsics, tf_static)
    track_bounds.json    verbatim copy of derived/track_bounds.json (supplied 3D borders)
    store_manifest.json  source hash, per-file sha256, topic inventory (allowed / disabled / forbidden), counts

The reader opens only this directory. It never touches the MCAP, derived/ or any evaluator file.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from camloc.contracts import CAMERAS, CameraStatic, ContractError, StaticMetadata
from camloc.geometry import load_t_base_camera
from camloc.util import sha256_bytes, sha256_file

STORE_ID = "cameras-v1"
MANIFEST_NAME = "store_manifest.json"
INDEX_NAME = "index.csv"
CALIB_NAME = "calib.json"
BOUNDS_NAME = "track_bounds.json"
INDEX_COLUMNS = ["camera", "stamp_ns", "log_time_ns", "offset", "length", "sha256"]
ALLOWED_STORE_FILES = (
    {MANIFEST_NAME, INDEX_NAME, CALIB_NAME, BOUNDS_NAME} | {f"jpeg_{c}.bin" for c in CAMERAS}
)
FORBIDDEN_MARKERS = ("gt", "state_estimation", "sn_map_state", "refline", "yaw_vel", "telemetry", "tf.", "evaluate")


class StoreError(ContractError):
    """The store is not a valid cameras-v1 store (unknown file, column, hash mismatch ...)."""


def pack_name(cam: str) -> str:
    return f"jpeg_{cam}.bin"


def check_layout(root: Path, cameras=CAMERAS) -> None:
    """Refuse a directory that holds anything beyond the allowlist (e.g. a forbidden-looking sidecar)."""
    names = {p.name for p in root.iterdir()}
    extra = names - ALLOWED_STORE_FILES
    if extra:
        raise StoreError(f"store has files outside the allowlist: {sorted(extra)}")
    missing = ({MANIFEST_NAME, INDEX_NAME, CALIB_NAME, BOUNDS_NAME} | {pack_name(c) for c in cameras}) - names
    if missing:
        raise StoreError(f"store misses files: {sorted(missing)}")


class CameraStore:
    """Read-only random access to JPEG payloads by (camera, index position in stamp order)."""

    def __init__(self, root, verify: bool = True):
        self.root = Path(root)
        self.verify = verify
        if not self.root.is_dir():
            raise StoreError(f"store directory not found: {self.root}")
        check_layout(self.root)
        man_path = self.root / MANIFEST_NAME
        self.manifest = json.loads(man_path.read_text())
        self.manifest_sha256 = sha256_file(man_path)
        if self.manifest.get("store_id") != STORE_ID:
            raise StoreError(f"unexpected store_id {self.manifest.get('store_id')!r}")
        if self.manifest.get("forbidden_records") != 0:
            raise StoreError("manifest does not certify zero forbidden records")
        extra_keys = [k for k in self.manifest if any(m in k.lower() for m in ("gt_", "state_est", "telemetry"))]
        if extra_keys:
            raise StoreError(f"manifest carries forbidden-looking keys: {extra_keys}")
        files = self.manifest["files"]
        for name in (INDEX_NAME, CALIB_NAME, BOUNDS_NAME):  # small files: full hash at open
            if sha256_file(self.root / name) != files[name]["sha256"]:
                raise StoreError(f"hash mismatch for {name}")
        for c in CAMERAS:  # packs: size at open (full hash is the audit's job)
            if (self.root / pack_name(c)).stat().st_size != files[pack_name(c)]["bytes"]:
                raise StoreError(f"size mismatch for {pack_name(c)}")
        self._load_index()
        self._fds: dict[str, int] = {}

    # ------------------------------------------------------------------ index
    def _load_index(self):
        with open(self.root / INDEX_NAME, newline="") as f:
            r = csv.reader(f)
            header = next(r)
            if header != INDEX_COLUMNS:
                raise StoreError(f"index columns must be exactly {INDEX_COLUMNS}, got {header}")
            rows = list(r)
        by = {c: [] for c in CAMERAS}
        for row in rows:
            if row[0] not in by:
                raise StoreError(f"index row for unknown camera {row[0]!r}")
            by[row[0]].append(row)
        self.stamps: dict[str, np.ndarray] = {}
        self.log_times: dict[str, np.ndarray] = {}
        self.offsets: dict[str, np.ndarray] = {}
        self.lengths: dict[str, np.ndarray] = {}
        self.shas: dict[str, list[str]] = {}
        for c, rs in by.items():
            self.stamps[c] = np.array([int(x[1]) for x in rs], dtype=np.int64)
            self.log_times[c] = np.array([int(x[2]) for x in rs], dtype=np.int64)
            self.offsets[c] = np.array([int(x[3]) for x in rs], dtype=np.int64)
            self.lengths[c] = np.array([int(x[4]) for x in rs], dtype=np.int64)
            self.shas[c] = [x[5] for x in rs]
            if len(rs) and (np.diff(self.stamps[c]) <= 0).any():
                raise StoreError(f"camera {c}: stamps are not strictly increasing in the index")
            if len(rs) and int((self.offsets[c] + self.lengths[c]).max()) > (self.root / pack_name(c)).stat().st_size:
                raise StoreError(f"camera {c}: index points beyond the pack")
        self.n_rows = sum(len(v) for v in self.stamps.values())
        if self.n_rows != self.manifest["index_rows"]:
            raise StoreError("index row count differs from manifest")

    def cameras(self) -> list[str]:
        return [c for c in CAMERAS if len(self.stamps[c])]

    # ------------------------------------------------------------------ payloads
    def _fd(self, cam: str) -> int:
        fd = self._fds.get(cam)
        if fd is None:
            fd = os.open(self.root / pack_name(cam), os.O_RDONLY)
            self._fds[cam] = fd
        return fd

    def jpeg(self, cam: str, i: int) -> bytes:
        """JPEG bytes of the i-th frame (stamp order) of a camera; sha256-verified against the index when verify."""
        off, n = int(self.offsets[cam][i]), int(self.lengths[cam][i])
        b = os.pread(self._fd(cam), n, off)
        if len(b) != n:
            raise StoreError(f"short read for {cam}[{i}]")
        if self.verify and sha256_bytes(b) != self.shas[cam][i]:
            raise StoreError(f"payload hash mismatch for {cam}[{i}] stamp {int(self.stamps[cam][i])}")
        return b

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ------------------------------------------------------------------ static
    def calib(self) -> dict:
        return json.loads((self.root / CALIB_NAME).read_text())

    def track_bounds(self) -> dict:
        return json.loads((self.root / BOUNDS_NAME).read_text())


def load_static_metadata(store: CameraStore) -> StaticMetadata:
    calib = store.calib()
    cams = {}
    for c, e in calib["cameras"].items():
        cams[c] = CameraStatic(width=e["width"], height=e["height"], K=tuple(e["K"]), D=tuple(e["D"]), P=tuple(e["P"]),
                               distortion_model=e["distortion_model"], frame_id=e["frame_id"],
                               T_base_cam=load_t_base_camera(calib, c))
    return StaticMetadata(cameras=cams, track_bounds=store.track_bounds(), map_id="yas_marina_north/constructor0/track_bounds",
                          calibration_sha256=store.manifest["files"][CALIB_NAME]["sha256"],
                          track_bounds_sha256=store.manifest["files"][BOUNDS_NAME]["sha256"],
                          store_id=store.manifest["store_id"])
