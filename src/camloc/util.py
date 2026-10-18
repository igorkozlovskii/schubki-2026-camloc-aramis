"""Small hashing/IO helpers. Same semantics as scripts/freeze_contracts.py (parity-tested), kept here so the
inference process does not import scripts that mention evaluator paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_hash(obj) -> str:
    """sha256 of canonical JSON (sorted keys, no spaces)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def code_hash(package_dir) -> str:
    """sha256 over the sorted (relative path, content hash) list of all .py files of a package directory."""
    root = Path(package_dir)
    items = [(str(p.relative_to(root)), sha256_file(p)) for p in sorted(root.rglob("*.py"))]
    return canonical_hash(items)
