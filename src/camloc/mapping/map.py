"""Training-only sparse reference map (E04): keyframes, global descriptors, and keypoints with metric 3D landmarks.

Directory layout (artifacts/maps/S04-<id>/):
    map.json         provenance manifest (section 2.3 keys), hashes of the data files
    keyframes.csv    kf_id,stamp_ns,lap,x,y,z,roll,pitch,yaw   (TRAIN GT poses, written by the offline builder)
    global.npy       (K, D) float32 global descriptors
    local.npz        kp_off (K+1,) int64, kpts (N,2) float32 full-image pixels, desc (N,128) float16,
                     xyz (N,3) float32 map frame, angle_deg (N,) float16 (triangulation angle), nobs (N,) uint8

Only keypoints that carry a landmark are stored. The inference path reads this module only (no GT file access):
the poses inside keyframes.csv are training poses of the map, not query ground truth.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from camloc.util import sha256_file

MAP_SCHEMA = "camloc-sparse-map-v1"
FILES = ("keyframes.csv", "global.npy", "local.npz")


@dataclass
class SparseMap:
    root: Path
    manifest: dict
    kf: pd.DataFrame
    global_desc: np.ndarray
    local: dict | None = None
    _cache: dict = field(default_factory=dict)

    @property
    def map_id(self) -> str:
        return self.manifest["map_id"]

    @property
    def kf_xy(self) -> np.ndarray:
        return self.kf[["x", "y"]].to_numpy(float)

    @property
    def kf_yaw(self) -> np.ndarray:
        return self.kf["yaw"].to_numpy(float)

    def kf_local(self, j: int):
        """(kpts, desc, xyz) of keyframe row j."""
        a, b = int(self.local["kp_off"][j]), int(self.local["kp_off"][j + 1])
        return self.local["kpts"][a:b], self.local["desc"][a:b], self.local["xyz"][a:b]


def save_map(root: Path, kf: pd.DataFrame, global_desc: np.ndarray, kp_off, kpts, desc, xyz, angle_deg, nobs, manifest: dict) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    kf.to_csv(root / "keyframes.csv", index=False, float_format="%.9g")
    np.save(root / "global.npy", np.asarray(global_desc, np.float32))
    np.savez(root / "local.npz", kp_off=np.asarray(kp_off, np.int64), kpts=np.asarray(kpts, np.float32), desc=np.asarray(desc, np.float16),
             xyz=np.asarray(xyz, np.float32), angle_deg=np.asarray(angle_deg, np.float16), nobs=np.asarray(nobs, np.uint8))
    manifest = dict(manifest)
    manifest["schema"] = MAP_SCHEMA
    manifest["files"] = {f: {"sha256": sha256_file(root / f), "bytes": (root / f).stat().st_size} for f in FILES}
    manifest["bytes_total"] = sum(v["bytes"] for v in manifest["files"].values())
    manifest["n_keyframes"] = int(len(kf))
    manifest["n_landmark_keypoints"] = int(len(kpts))
    (root / "map.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def load_map(root, need_local: bool = True, verify: bool = False) -> SparseMap:
    root = Path(root)
    man = json.loads((root / "map.json").read_text())
    if man.get("schema") != MAP_SCHEMA:
        raise ValueError(f"unknown map schema {man.get('schema')!r}")
    if verify:
        for f, v in man["files"].items():
            if sha256_file(root / f) != v["sha256"]:
                raise ValueError(f"map file {f} hash mismatch")
    kf = pd.read_csv(root / "keyframes.csv")
    g = np.load(root / "global.npy")
    loc = None
    if need_local:
        z = np.load(root / "local.npz")
        loc = {k: z[k] for k in z.files}
    return SparseMap(root, man, kf, g, loc)
