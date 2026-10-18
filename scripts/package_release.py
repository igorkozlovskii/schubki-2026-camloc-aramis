#!/usr/bin/env python3
"""Build the compact offline release bundle from a release config (copies only what the config names; sha256 manifest, sizes, licenses).

    pixi run python scripts/package_release.py --release-config configs/release/release-rfb-v1.yaml --out artifacts/releases/S08a-release-rfb-v1 [--link]

Refuses a non-empty --out. Never includes the camera store, GT, dev/audit data, training caches or other methods' weights. Prints the
size report against the 10 GB target.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from camloc.release.bundle import BundleError, verify_bundle  # noqa: E402
from camloc.release.package import build_bundle  # noqa: E402

TARGET_BYTES = 10 * 10**9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=ROOT)
    ap.add_argument("--link", action="store_true", help="hard-link instead of copying (same filesystem; no extra storage)")
    ap.add_argument("--size-report", type=Path, default=None)
    a = ap.parse_args(argv)
    cmd = "package_release.py " + " ".join(argv if argv is not None else sys.argv[1:])
    try:
        man = build_bundle(a.release_config, a.repo_root, a.out, link=a.link, created_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), command=cmd)
        verify_bundle(a.out)
    except BundleError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    by_role: dict = {}
    for p, e in man["files"].items():
        r = by_role.setdefault(e["role"], {"files": 0, "bytes": 0})
        r["files"] += 1
        r["bytes"] += e["bytes"]
    rep = {"bundle": str(a.out), "bundle_hash": man["bundle_hash"], "files": man["totals"]["files"], "bytes": man["totals"]["bytes"],
           "target_bytes": TARGET_BYTES, "within_target": man["totals"]["bytes"] <= TARGET_BYTES, "by_role": by_role,
           "largest": sorted(((e["bytes"], p) for p, e in man["files"].items()), reverse=True)[:8]}
    if a.size_report:
        a.size_report.parent.mkdir(parents=True, exist_ok=True)
        a.size_report.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
