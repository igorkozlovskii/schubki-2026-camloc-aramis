#!/usr/bin/env python3
"""Verify a release bundle: every file's sha256/size against bundle_manifest.json, no unlisted file, no forbidden content, size report.

    python3 -I scripts/verify_release.py --bundle <release-dir> [--report out.json]

Exit 0 only if everything matches. Standard library only: runs in a plain Python >= 3.10 with no packages installed
(no PyYAML, numpy, torch, pixi):  python3 -I scripts/verify_release.py --bundle <release-dir>
It imports camloc.release.bundle and camloc.util, both stdlib-only (camloc.release.__init__ loads PyYAML lazily).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # run from a bundle's code/ it must not write __pycache__ into the bundle it verifies

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent / "src", HERE.parent):   # repo checkout (src/) or bundle code/ (camloc/ next to scripts/)
    if (cand / "camloc").is_dir():
        sys.path.insert(0, str(cand))
        break

from camloc.release.bundle import BundleError, verify_bundle  # noqa: E402


class PinFailure(Exception):
    pass


def check_pins_if_possible(bundle: Path) -> str:
    """S08e: re-check the filter parameter pins of the bundle's release_config.yaml when this is a repository checkout with PyYAML and numpy (the pins read the
    checkout's params files); the stdlib-only path (python3 -I, bundle code/ copy) reports 'skipped' and verifies hashes only."""
    if not (HERE.parent / "src/camloc").is_dir():
        return "skipped (not a repository checkout)"
    try:
        import numpy  # noqa: F401
        import yaml  # noqa: F401
    except ImportError:
        return "skipped (PyYAML/numpy not importable: hashes only)"
    from camloc.release.config import ReleaseConfigError, load_release_config
    try:
        cfg = load_release_config(bundle / "release_config.yaml")
    except ReleaseConfigError as e:
        raise PinFailure(f"release config / filter pin: {e}") from None
    return "checked: " + (", ".join(sorted(cfg.get("filter_pins", {}))) or "no filter composite in this release")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--size-only", action="store_true", help="skip sha256 (sizes and file lists only)")
    a = ap.parse_args(argv)
    try:
        rep = verify_bundle(a.bundle, check_hash=not a.size_only)
        man = json.loads((a.bundle / "bundle_manifest.json").read_text())
        rep["target_bytes"] = 10 * 10**9
        rep["within_10gb_target"] = rep["bytes"] <= rep["target_bytes"]
        rep["by_role"] = {}
        for e in man["files"].values():
            r = rep["by_role"].setdefault(e["role"], {"files": 0, "bytes": 0})
            r["files"] += 1
            r["bytes"] += e["bytes"]
        rep["filter_pins"] = check_pins_if_possible(a.bundle)
        rep["status"] = "ok"
    except (BundleError, PinFailure) as e:
        rep = {"status": "FAILED", "problem": str(e)}
    if a.report:
        a.report.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    return 0 if rep["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
