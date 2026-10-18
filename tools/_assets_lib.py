"""Shared helpers of build_assets.py / verify_assets.py (PRG-003 R02b). Digest convention = the research freeze (scripts/final_run.py dir_digest)."""
from __future__ import annotations

import hashlib
from pathlib import Path

BUNDLE = "final-v1"
ASSET_MANIFEST_SCHEMA = "assets-final-v1"
TAR_TOP = "assets-final-v1"                       # top-level directory inside the tarball
BOUNDS_SHA = "28cf2bfd9ce74c2e5f1ededd5d36866221c341609b27a15e3b5bae94b33de0ff"

# Files the frozen run reads that are NOT hashed in manifests/final/final-v1.json (found by running F1's driver and by the R02b audit).
# (research-relative source, asset-root-relative destination, note)
SUPPLEMENTAL = [
    ("artifacts/external/S04-lightglue/weights/aliked_lightglue_v0.1_arxiv.pth", None,
     "LightGlue ALIKED matcher checkpoint loaded by camloc.models.retrieval.Matcher; absent from final-v1.json (gap), hash first recorded here"),
    ("configs/models/e09k2-y1.yaml", None, "filter parameter file read through scripts/s07e_lib.py; absent from final-v1.json"),
    ("reports/experiments/S10b-tel/tuning/V3_all.json", None, "tuned filter parameters read by scripts/s10b_lib.py; absent from final-v1.json"),
    ("reports/experiments/S10g-tel4/tuning/g1_all.json", None, "tuned re-anchor parameters read by scripts/s10g_lib.py; absent from final-v1.json"),
    ("derived/track_bounds.json", None, "read by scripts/v2_s13_lib.py (TrackFrame); identical to assets/final/track_bounds.json, sha256 checked"),
]
LICENCE = {
    "maps": "licenses/license-table.md (reference map rows: own derived data from the supplied recording; organizer data terms apply)",
    "poles": "licenses/license-table.md (own derived data from the supplied recording; organizer data terms apply)",
    "attitude_map": "own derived data from the supplied recording; organizer data terms apply (licences/ has no separate row)",
    "body_masks": "own derived data from the supplied recording; organizer data terms apply (licenses/ has no separate row)",
    "border_seg_B_all_laps": "own trained checkpoint on the supplied recording; component-licence reconciliation still open (PRG-003 R02a remaining work); see receipt",
    "aliked_n16_weights": "licenses/license-table.md (ALIKED-n16 checkpoint, BSD-3-Clause)",
    "aliked_lightglue": "licenses/license-table.md (LightGlue ALIKED matcher checkpoint, Apache-2.0)",
    "global_descriptor_dinov2_small": "licenses/license-table.md (DINOv2 ViT-S/14, Apache-2.0)",
    "reference_calibration": "licenses/license-table.md (organizer-supplied recording metadata)",
    "constants": "own derived constants (numbers); no third-party licence",
    "assets": "own derived track geometry from the organizer-supplied recording; organizer data terms apply",
    "supplemental": "own derived parameters / see note",
}


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_digest(p: Path) -> dict:
    """Same digest as the research freeze: sha256 over '<rel>\\0<file sha>\\n' of the sorted files (a single file has an empty rel)."""
    p = Path(p)
    files = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file() and "__pycache__" not in q.parts)
    h, tot = hashlib.sha256(), 0
    for q in files:
        rel = "" if p.is_file() else str(q.relative_to(p))
        h.update(f"{rel}\0{sha256_file(q)}\n".encode())
        tot += q.stat().st_size
    return {"files": len(files), "bytes": tot, "digest_sha256": h.hexdigest()}
