#!/usr/bin/env python3
"""PRG-003 R02b -- build the versioned asset bundle of the frozen final model (final-v1).

    python tools/build_assets.py --source-root <research checkout> --dest assets/final-v1 [--tar dist/camloc-aramis-assets-final-v1.tar]

Copies every asset listed in the research freeze manifest (manifests/final/final-v1.json: maps, poles, attitude map, body masks, border-segmentation
checkpoint, ALIKED weights, DINOv2-small, reference calibration, constants, track bounds), verifying each sha256 / directory digest against that manifest
BEFORE and AFTER copying; any mismatch refuses (exit 1, nothing partial is kept as final). Adds the few supplemental files the run reads that the freeze
manifest does not hash (listed in tools/_assets_lib.py), with their hashes recorded here. Layout under --dest mirrors the research layout
(artifacts/, manifests/v2/, configs/models/, reports/experiments/, derived/, assets/final/).
Writes manifests/assets-final-v1.json (path, bytes, sha256, original research path, licence pointer) and, with --tar, one deterministic tarball.
Read-only on the source root. Refuses an existing non-empty --dest.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _assets_lib as A  # noqa: E402

MAX_PART = int(1.9e9)


def plan(src: Path, fz: dict):
    """-> list of dicts {group, key, src_rel, dest_rel, kind(file|dir), expect(sha|digest|None), frozen}"""
    rows = []
    for k, m in fz["maps"].items():
        rows.append(dict(group=k if k in A.LICENCE else "maps", key=k, src_rel=m["path"], dest_rel=m["path"], kind="dir", expect=m["digest_sha256"], expect_n=m["files"], frozen=True))
    for k, m in fz["networks"].items():
        rows.append(dict(group=k, key=k, src_rel=m["path"], dest_rel=m["path"], kind="dir", expect=m["digest_sha256"], expect_n=m["files"], frozen=True))
    for f, h in fz["constants"].items():
        rows.append(dict(group="constants", key=f, src_rel=f, dest_rel=f, kind="file", expect=h, frozen=True))
    for f, h in fz["assets"].items():
        rows.append(dict(group="assets", key=f, src_rel=f, dest_rel=f, kind="file", expect=h, frozen=True))
    for f in ("configs/models/e04-pnp-rfb-v1.yaml",):          # hashed in freeze 'code'
        rows.append(dict(group="supplemental", key=f, src_rel=f, dest_rel=f, kind="file", expect=fz["code"][f], frozen=True))
    for s_rel, d_rel, note in A.SUPPLEMENTAL:
        g = "aliked_lightglue" if "aliked_lightglue" in s_rel else "supplemental"
        rows.append(dict(group=g, key=s_rel, src_rel=s_rel, dest_rel=d_rel or s_rel, kind="file", expect=A.BOUNDS_SHA if s_rel == "derived/track_bounds.json" else None, frozen=False, note=note))
    return rows


def copy_tree(sp: Path, dp: Path):
    if sp.is_file():
        dp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sp, dp)
        return
    for q in sorted(sp.rglob("*")):
        if q.is_file() and "__pycache__" not in q.parts:
            t = dp / q.relative_to(sp)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(q, t)


def check(path: Path, r: dict, side: str):
    if r["kind"] == "dir" or r["group"] in ("border_seg_B_all_laps", "aliked_n16_weights", "reference_calibration"):
        d = A.dir_digest(path)
        if d["digest_sha256"] != r["expect"] or (r.get("expect_n") and d["files"] != r["expect_n"]):
            raise SystemExit(f"REFUSED: {side} digest mismatch for {r['key']}: have {d['digest_sha256'][:16]} ({d['files']} files), freeze {r['expect'][:16]}")
        return d
    h = A.sha256_file(path)
    if r["expect"] and h != r["expect"]:
        raise SystemExit(f"REFUSED: {side} sha256 mismatch for {r['src_rel']}: have {h[:16]}, expected {r['expect'][:16]}")
    return {"files": 1, "bytes": path.stat().st_size, "sha256": h}


def main(argv=None) -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-root", required=True, type=Path)
    ap.add_argument("--dest", type=Path, default=here / "assets" / A.BUNDLE)
    ap.add_argument("--manifest", type=Path, default=here / "manifests" / "assets-final-v1.json")
    ap.add_argument("--tar", type=Path, default=None, help="also pack the bundle into this deterministic tar (one part < 1.9 GB; refuses otherwise)")
    ap.add_argument("--source-head", default=None)
    a = ap.parse_args(argv)
    src, dest = a.source_root.resolve(), a.dest.resolve()
    if src == dest or src in dest.parents:
        ap.error("--dest must not lie inside the research checkout")
    fz = json.loads((src / "manifests/final/final-v1.json").read_text())
    rows = plan(src, fz)
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"REFUSED: {dest} exists and is not empty")
    # 1. verify the source against the freeze before copying anything
    for r in rows:
        sp = src / r["src_rel"]
        if not sp.exists():
            raise SystemExit(f"REFUSED: source asset missing: {r['src_rel']}")
        r["src_check"] = check(sp, r, "source")
    # 2. copy, re-verify the copy, record
    out_files = []
    for r in rows:
        sp, dp = src / r["src_rel"], dest / r["dest_rel"]
        copy_tree(sp, dp)
        d = check(dp, r, "copy")
        files = [dp] if dp.is_file() else sorted(q for q in dp.rglob("*") if q.is_file())
        for q in files:
            out_files.append({"path": q.relative_to(dest).as_posix(), "bytes": q.stat().st_size, "sha256": A.sha256_file(q),
                              "original_path": (r["src_rel"] if dp.is_file() else f"{r['src_rel']}/{q.relative_to(dp).as_posix()}"),
                              "group": r["group"], "freeze_key": r["key"], "in_freeze_manifest": r["frozen"],
                              "licence": A.LICENCE.get(r["group"], A.LICENCE["supplemental"]), **({"note": r["note"]} if r.get("note") else {})})
        r["group_digest"] = d
    out_files.sort(key=lambda x: x["path"])
    assert len({f["path"] for f in out_files}) == len(out_files), "duplicate destination paths"
    groups = [{"group": r["group"], "key": r["key"], "path": r["dest_rel"], "kind": r["kind"], "in_freeze_manifest": r["frozen"],
               "files": r["group_digest"]["files"], "bytes": r["group_digest"].get("bytes"), "digest_sha256": r["group_digest"].get("digest_sha256") or r["group_digest"].get("sha256"),
               "freeze_expect": r["expect"]} for r in rows]
    man = {"schema": A.ASSET_MANIFEST_SCHEMA, "bundle": A.BUNDLE, "source_root_head": a.source_head, "freeze_manifest": "manifests/final/final-v1.json",
           "freeze_manifest_sha256": A.sha256_file(src / "manifests/final/final-v1.json"), "file_count": len(out_files),
           "total_bytes": sum(f["bytes"] for f in out_files), "groups": groups, "files": out_files, "tar": None}
    # schema validation
    for f in out_files:
        assert len(f["sha256"]) == 64 and f["bytes"] >= 0 and f["path"] and f["original_path"] and f["licence"], f
    if a.tar:
        tp = a.tar.resolve()
        tp.parent.mkdir(parents=True, exist_ok=True)
        if tp.exists():
            raise SystemExit(f"REFUSED: {tp} exists")
        with tarfile.open(tp, "w", format=tarfile.PAX_FORMAT) as tf:
            def add(p: Path, arc: str):
                ti = tf.gettarinfo(str(p), arc)
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ""
                ti.mtime = 0
                ti.mode = 0o644
                with open(p, "rb") as fh:
                    tf.addfile(ti, fh)
            for f in out_files:
                add(dest / f["path"], f"{A.TAR_TOP}/{f['path']}")
        size = tp.stat().st_size
        if size >= MAX_PART:
            raise SystemExit(f"REFUSED: tar {size} bytes >= 1.9 GB; split is not implemented (not needed so far)")
        man["tar"] = {"name": tp.name, "bytes": size, "sha256": A.sha256_file(tp), "parts": 1}
        print(f"tar {tp} {size} bytes sha256 {man['tar']['sha256']}")
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(man, indent=1) + "\n")
    print(f"assets: {len(out_files)} files, {man['total_bytes']} bytes -> {dest}; manifest {a.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
