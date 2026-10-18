"""Camera-only calibration fingerprint (S08c). Standard library + camloc.util only.

A model that reads camera images depends on the cameras' intrinsics (K, D, P, size, distortion model), their frame ids and their
extrinsics `T_base_cam` (base_link -> camera, from the camera entries of /tf_static). Lidar, radar, GNSS, IMU and other sensor
extrinsics in /tf_static are irrelevant to it and must not trigger a refusal. The fingerprint hashes exactly the camera part of
`calib.json`; the whole-file sha256 is kept as a second, informational fingerprint.
"""
from __future__ import annotations

import json
from pathlib import Path

from camloc.util import canonical_hash, sha256_file

CAMERA_FIELDS = ("width", "height", "distortion_model", "K", "D", "P", "frame_id", "T_base_cam")
# fingerprints of the supplied recording's calibration (derived/calib.json, sha256 3bdea36b...); reproduced by test
TRAIN_CAMERA_FINGERPRINT = "5404445319c8f2e3a4b53d50fc42c23a6d464c71db259d78fba84b2a53e07c3c"
TRAIN_PER_CAMERA = {
    "cl": "cebe69e378754e163c1b699c62538d65bcec5d26f3d7c8174307a644069d6c35",
    "cr": "99777f7ea0636a9ea5ffdcdf45168f090a622a7dee4e4331bc0e3fab4d2abbb2",
    "fl": "38d73bb9fec506b854b816555118639f3bd66e93b7a33dc7a281801d1265fa4d",
    "fr": "23f024c764a314eeee0967b1bece943ac32fc4d860df882cdd45e57541b3355d",
    "r": "ca648126277937e818cdd1de57d2ffa81602e9e1a8001db4a96ccc7b07a7a47a",
    "rl": "fe65ff88abe3307a8e52979f873615feea5ee3e6da110d15da954360f197c7ea",
    "rr": "37907b8468f3340a93583ab2e506720b494d7329e6f2942d60e9a0ffa655f659",
}


def per_camera(calib: dict) -> dict:
    """camera -> sha256 over (size, model, K, D, P, frame id, T_base_cam). Missing fields hash as null (a different value)."""
    return {c: canonical_hash({k: e.get(k) for k in CAMERA_FIELDS}) for c, e in sorted(calib.get("cameras", {}).items())}


def camera_fingerprint(calib: dict) -> str:
    return canonical_hash(per_camera(calib))


def fingerprints(calib_path) -> dict:
    p = Path(calib_path)
    calib = json.loads(p.read_text())
    pc = per_camera(calib)
    return {"camera_fingerprint": canonical_hash(pc), "per_camera": pc, "full_sha256": sha256_file(p)}


def compare(reference_per_camera: dict, got_per_camera: dict, required=("fl",)) -> dict:
    """Policy: a camera present on both sides must be identical; every `required` camera (the ones the model reads) must be present
    and identical. A camera missing on one side that nobody requires is a note, not a refusal.
    Returns {status: 'match'|'note'|'mismatch', differing, missing_required, missing_in_input, extra_in_input}."""
    ref, got = reference_per_camera, got_per_camera
    differing = sorted(c for c in ref if c in got and ref[c] != got[c])
    missing_required = sorted(c for c in required if c not in got or c not in ref)
    missing_in_input = sorted(c for c in ref if c not in got)
    extra_in_input = sorted(c for c in got if c not in ref)
    if differing or missing_required:
        status = "mismatch"
    elif missing_in_input or extra_in_input:
        status = "note"
    else:
        status = "match"
    return {"status": status, "differing": differing, "missing_required": missing_required,
            "missing_in_input": missing_in_input, "extra_in_input": extra_in_input}
