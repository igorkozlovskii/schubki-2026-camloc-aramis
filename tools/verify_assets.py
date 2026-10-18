#!/usr/bin/env python3
"""PRG-003 R02b -- verify the final-v1 asset bundle.

    python tools/verify_assets.py --assets <dir>                       # an unpacked / built bundle
    python tools/verify_assets.py --tar <tarball> --work <empty dir>   # unpack into a FRESH dir under --work, then verify
Checks, with exit 1 on any failure: (1) every file of manifests/assets-final-v1.json exists with the recorded size and sha256; (2) no extra file in the bundle;
(3) every group that final-v1.json hashes (maps, networks, constants, assets, e04 config) reproduces the freeze digest -- the freeze manifest is read from
--freeze (default: this release's manifests/final-v1-freeze.json, a verbatim copy of the research manifests/final/final-v1.json); (4) with --tar, the tarball sha256 equals the
manifest's. How to fetch: see docs/ASSETS.md (the release URL is filled in by R04).
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _assets_lib as A  # noqa: E402


def main(argv=None) -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--assets", type=Path)
    ap.add_argument("--tar", type=Path)
    ap.add_argument("--work", type=Path, help="directory in which --tar is unpacked into a fresh sub-directory")
    ap.add_argument("--manifest", type=Path, default=here / "manifests" / "assets-final-v1.json")
    ap.add_argument("--freeze", type=Path, default=here / "manifests" / "final-v1-freeze.json")
    a = ap.parse_args(argv)
    man = json.loads(a.manifest.read_text())
    fz = json.loads(a.freeze.read_text())
    bad = []
    if a.tar:
        if not a.work:
            ap.error("--tar needs --work")
        if man.get("tar") and A.sha256_file(a.tar) != man["tar"]["sha256"]:
            bad.append(f"tarball sha256 {A.sha256_file(a.tar)[:16]} != manifest {man['tar']['sha256'][:16]}")
        a.work.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="unpack-", dir=a.work))
        with tarfile.open(a.tar) as tf:
            for m in tf.getmembers():
                if m.name.startswith("/") or ".." in Path(m.name).parts:
                    raise SystemExit(f"REFUSED: unsafe tar member {m.name}")
            tf.extractall(tmp, filter="data")
        root = tmp / A.TAR_TOP
        print("unpacked to", root)
    elif a.assets:
        root = a.assets.resolve()
    else:
        ap.error("need --assets or --tar")
    have = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    want = {f["path"] for f in man["files"]}
    for f in man["files"]:
        p = root / f["path"]
        if not p.is_file():
            bad.append(f"missing {f['path']}")
        elif p.stat().st_size != f["bytes"] or A.sha256_file(p) != f["sha256"]:
            bad.append(f"mismatch {f['path']}")
    for x in sorted(have - want):
        bad.append(f"extra file not in manifest: {x}")
    n_groups = 0
    exp = {}
    for k, m in fz["maps"].items():
        exp[m["path"]] = m["digest_sha256"]
    for k, m in fz["networks"].items():
        exp[m["path"]] = m["digest_sha256"]
    for sect in ("constants", "assets"):
        for f, h in fz[sect].items():
            exp[f] = ("file", h)
    exp["configs/models/e04-pnp-rfb-v1.yaml"] = ("file", fz["code"]["configs/models/e04-pnp-rfb-v1.yaml"])
    for path, e in exp.items():
        p = root / path
        if not p.exists():
            bad.append(f"frozen asset missing: {path}")
            continue
        n_groups += 1
        if isinstance(e, tuple):
            if A.sha256_file(p) != e[1]:
                bad.append(f"frozen file hash differs from final-v1.json: {path}")
        elif A.dir_digest(p)["digest_sha256"] != e:
            bad.append(f"frozen digest differs from final-v1.json: {path}")
    if A.sha256_file(root / "derived/track_bounds.json") != A.BOUNDS_SHA:
        bad.append("derived/track_bounds.json differs from the packaged bounds")
    if bad:
        print("FAIL")
        for b in bad[:50]:
            print(" ", b)
        return 1
    print(f"OK: {len(want)} files hash-verified, {n_groups} frozen groups equal manifests/final/final-v1.json, no extra files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
