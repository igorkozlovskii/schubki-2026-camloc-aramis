"""Build a release bundle directory from a release config (packaging side; reads the repo, writes only the new bundle dir)."""
from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
from pathlib import Path

from camloc.util import canonical_hash, sha256_file

from .bundle import MANIFEST_NAME, BundleError, build_manifest, file_entry, write_manifest
from .codeclosure import import_closure
from .config import load_release_config


def _git(root: Path, *args) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:   # noqa: BLE001
        return ""


def _expand(src: Path, include, exclude) -> list[str]:
    out = []
    for pat in include:
        hits = sorted(p.relative_to(src).as_posix() for p in src.glob(pat) if p.is_file())
        if not hits:
            raise BundleError(f"asset source {src}: pattern {pat!r} matches nothing (strict: no silent skip)")
        out += hits
    out = sorted(set(out))
    return [p for p in out if not any(fnmatch.fnmatch(p, e) for e in (exclude or []))]


def _put(src: Path, dst: Path, link: bool, drop: str | None = None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if drop is not None:
        lines = [ln for ln in src.read_text().splitlines(keepends=True) if drop not in ln.lower()]
        dst.write_text("".join(lines))
    elif link:
        os.link(src, dst)
    else:
        shutil.copyfile(src, dst)


def build_bundle(config_path, repo_root, out_dir, link: bool = False, created_utc: str | None = None, command: str = "") -> dict:
    out = Path(out_dir)
    pre_existing = out.exists()
    try:
        return _build(config_path, repo_root, out_dir, link, created_utc, command)
    except Exception:
        if not pre_existing and out.exists():
            shutil.rmtree(out)   # a failed build leaves nothing behind
        raise


def _build(config_path, repo_root, out_dir, link: bool = False, created_utc: str | None = None, command: str = "") -> dict:
    root, out = Path(repo_root), Path(out_dir)
    cfg = load_release_config(config_path)
    if out.exists() and any(out.iterdir()):
        raise BundleError(f"REFUSED: {out} exists and is not empty (bundles are never overwritten)")
    out.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict] = {}
    assets_meta = []

    def add(rel_dst: str, src: Path, role: str, extra=None, drop=None):
        _put(src, out / rel_dst, link, drop)
        entries[rel_dst] = file_entry(out / rel_dst, role, extra)

    for a in cfg["assets"]:
        src = root / a["src"]
        if not src.exists():
            raise BundleError(f"asset {a['id']}: source {a['src']} not found (strict failure; nothing is downloaded)")
        for rel in _expand(src, a["include"], a.get("exclude")):
            patch = (a.get("patch") or {}).get(rel)
            extra = {"asset": a["id"]}
            if patch:
                extra["derived_from_sha256"] = sha256_file(src / rel)
                extra["patch"] = f"lines containing {patch['drop_lines_containing']!r} removed"
            add(f"{a['dest']}/{rel}", src / rel, a["role"], extra, drop=(patch or {}).get("drop_lines_containing"))
        assets_meta.append({k: a[k] for k in ("id", "role", "src", "dest", "license", "source")})

    # code snapshot: exactly the import closure of the start modules + listed scripts
    code = cfg["code"]
    files = import_closure(root / "src", code["start_modules"])
    code_entries = {}
    for rel in files:
        add(f"code/{rel}", root / "src" / rel, "code")
        code_entries[f"src/{rel}"] = entries[f"code/{rel}"]["sha256"]
    for rel in code.get("extra_files", []):
        add(f"code/{rel}", root / rel, "code")
        code_entries[rel] = entries[f"code/{rel}"]["sha256"]
    dirty = [ln for ln in _git(root, "status", "--porcelain", "--", *[f"src/{f}" for f in files], *code.get("extra_files", [])).splitlines()]
    for x in cfg.get("extra_files", []):
        add(x["dest"], root / x["src"], "doc" if x["dest"].endswith((".md", ".json")) else "environment")
    add("release_config.yaml", Path(config_path), "config")

    meta = {"release_id": cfg["release_id"], "model_id": cfg["model"]["model_id"], "assets": assets_meta,
            "code": {"modules": sorted(files), "extra_files": code.get("extra_files", []), "code_hash": canonical_hash(sorted(code_entries.items()))},
            "dev_recording_sha256": cfg["dev_recording_sha256"], "calibration_sha256": cfg["runtime"]["calibration_sha256"],
            "provenance": {"created_utc": created_utc, "git_head": _git(root, "rev-parse", "HEAD"), "git_closure_uncommitted": dirty,
                           "command": command, "release_config_sha256": sha256_file(config_path), "linked": link}}
    man = build_manifest(out, entries, meta)
    write_manifest(out, man)
    return man
