"""Bundle manifest: deterministic sha256 listing of a release directory, tamper detection and a content audit."""
from __future__ import annotations

import json
import os
from pathlib import Path

from camloc.util import canonical_hash, sha256_file

MANIFEST_NAME = "bundle_manifest.json"
SCHEMA = "release-bundle-v1"
# path components (lower-case) that must never appear in a release bundle: GT, evaluator, dev/audit data, caches, the store
FORBIDDEN_PARTS = ("gt.csv", "gt_tf", "state_estimation", "sn_map_state", "refline", "evaluate.py", "derived", "audit",
                   "jpeg_", "images.parquet", "filtered_training_rows", "split-v1", "cameras-v1", ".mcap",
                   "__pycache__", ".pyc", "dpvo", "orb_slam", "orbslam")
FORBIDDEN_NAMES = ("frames.csv", "gt.csv", "images.csv", "index.csv")   # exact file names
FORBIDDEN_SUFFIXES = (".mcap", ".bin", ".parquet", ".pyc")


class BundleError(Exception):
    pass


def file_entry(path: Path, role: str, extra: dict | None = None) -> dict:
    e = {"sha256": sha256_file(path), "bytes": path.stat().st_size, "role": role}
    if extra:
        e.update(extra)
    return e


def _walk(root: Path) -> list[str]:
    out = []
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for f in sorted(fn):
            out.append(Path(dp, f).relative_to(root).as_posix())
    return sorted(out)


def content_audit(paths) -> list[str]:
    """Return the list of paths that look like forbidden content (GT/dev/audit/training-cache/store/other-method files)."""
    bad = []
    for p in paths:
        if p == MANIFEST_NAME:
            continue
        low = p.lower()
        parts = low.split("/")
        if any(any(f in part for f in FORBIDDEN_PARTS) for part in parts) or low.endswith(FORBIDDEN_SUFFIXES) or parts[-1] in FORBIDDEN_NAMES:
            # allowed exception: the DINOv2/LightGlue directory names contain none of the markers; no exception needed
            bad.append(p)
    return bad


def build_manifest(root: Path, entries: dict[str, dict], meta: dict) -> dict:
    """entries: relative posix path -> {sha256, bytes, role, ...}. The manifest is canonical: sorted keys, no timestamps in the hashed part."""
    paths = sorted(entries)
    bad = content_audit(paths)
    if bad:
        raise BundleError(f"forbidden content in bundle: {bad}")
    man = {"schema": SCHEMA, "files": {p: entries[p] for p in paths}, "totals": {"files": len(paths), "bytes": sum(entries[p]["bytes"] for p in paths)}}
    man.update({k: v for k, v in meta.items() if k not in ("provenance",)})
    man["bundle_hash"] = canonical_hash({k: v for k, v in man.items()})
    if "provenance" in meta:
        man["provenance"] = meta["provenance"]   # not part of the hash (timestamps, commands)
    return man


def write_manifest(root: Path, man: dict) -> Path:
    p = Path(root) / MANIFEST_NAME
    p.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    return p


def verify_bundle(root, check_hash: bool = True) -> dict:
    """Strict verification: manifest present and self-consistent, every listed file exists with the listed size and sha256,
    no unlisted file, no forbidden content. Returns a report; raises BundleError on the first problem class."""
    root = Path(root)
    mp = root / MANIFEST_NAME
    if not mp.is_file():
        raise BundleError(f"bundle manifest not found: {mp}")
    man = json.loads(mp.read_text())
    if man.get("schema") != SCHEMA:
        raise BundleError(f"unknown bundle schema {man.get('schema')!r}")
    hashed = {k: v for k, v in man.items() if k not in ("bundle_hash", "provenance")}
    if canonical_hash(hashed) != man.get("bundle_hash"):
        raise BundleError("bundle_hash does not match the manifest content (manifest edited)")
    listed = man["files"]
    problems: list[str] = []
    for rel, e in listed.items():
        p = root / rel
        if not p.is_file():
            problems.append(f"missing: {rel}")
            continue
        if p.stat().st_size != e["bytes"]:
            problems.append(f"size mismatch: {rel}")
            continue
        if check_hash and sha256_file(p) != e["sha256"]:
            problems.append(f"sha256 mismatch: {rel}")
    on_disk = set(_walk(root)) - {MANIFEST_NAME}
    extra = sorted(on_disk - set(listed))
    if extra:
        problems.append(f"unlisted files: {extra[:10]}{' ...' if len(extra) > 10 else ''}")
    bad = content_audit(sorted(on_disk | set(listed)))
    if bad:
        problems.append(f"forbidden content: {bad[:10]}")
    if problems:
        raise BundleError("; ".join(problems))
    return {"bundle_hash": man["bundle_hash"], "files": len(listed), "bytes": man["totals"]["bytes"], "hashes_checked": check_hash,
            "release_id": man.get("release_id")}
