#!/usr/bin/env python3
"""PRG-003 R02a — narrow, rerunnable source exporter for the D-086 final model.

Copies ONLY the runtime dependency closure of the single final model
(mc7 + borders + poles + road-attitude + segmentation, no learned/GRU layer),
the five generic run/export/package/verify CLIs, the fixed configs they pin, and
the *necessary* licensed LightGlue subset into a standalone destination tree.

Design rules (see research/programs/PRG-003-release/briefs/R02a.md, section A):

* No hard-coded user paths. Source root, destination and selection come from
  arguments (or sane relative defaults resolved from this file's location).
* The camloc module closure is computed with the project's own AST helper
  ``camloc.release.codeclosure.import_closure`` — the same tool the release
  packager uses — so the export tracks real imports, not a hand-kept list.
* Package layout and public entry-point names are preserved verbatim.
* Three identities are kept per copied file: the ORIGINAL source sha256, the
  COPIED (resulting) sha256, and an adaptation reason when they differ. The only
  adapted file in this step is ``lightglue/__init__.py`` (the documented
  SuperPoint import drop). Everything else is byte-identical (orig == copied).
* Conflict policy: an existing destination file that is byte-identical to what
  would be written is accepted (idempotent rerun); a differing existing file is
  REFUSED with a non-zero exit. Nothing is silently overwritten.
* No weights, raw data, ``.git``, caches, notebooks, research driver scripts or
  binary asset trees are ever copied. No network access, no model construction.

Output: writes/refreshes ``<dest>/manifests/source-export.json`` (schema
``source-export-v1``) recording the exact selection, per-file identities and the
LightGlue adaptation. Exit code is 0 on success, non-zero on any refused
conflict or schema failure.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "source-export-v1"

# --- The final-model (D-086) start modules -----------------------------------
# mc7 release front end + generic runner + packaging side (same start set the
# mc7-v1 release config pins under code.start_modules / code.extra_files) ...
MC7_START = [
    "camloc.runner",
    "camloc.release.run",
    "camloc.release.package",
    "camloc.release.validate",
    "camloc.models.e04t",
    "camloc.models.e04y",
    "camloc.models.pnp_fb",
    "camloc.models.pnp",
    "camloc.models.mc7",
    "camloc.filtering.tel4_runtime",
    "camloc.filtering.tel4_filter",
    "camloc.io.export_generic",
    "camloc.io.export_telemetry",
]
# ... plus the D-086 border / pole / road-attitude / segmentation library stack
# (the live-measurement modules the research driver glues together; the driver
# scripts themselves are NOT exported — their logic is R02b reference material).
D086_START = [
    "camloc.geometry.borders",
    "camloc.geometry.residuals",
    "camloc.geometry.camera",
    "camloc.geometry.centerline",
    "camloc.v2.border_fast",
    "camloc.v2.border_probe",
    "camloc.v2.border_seg",
    "camloc.v2.border_seg_att",
    "camloc.v2.road_attitude",
    "camloc.v2.poles",
    "camloc.v2.poles.detect",
    "camloc.v2.seg",
    "camloc.v2.seg.infer",
    "camloc.v2.seg.model",
    "camloc.v2.seg.config",
    "camloc.v2.seg.labels",
    "camloc.v2.meas",
    "camloc.v2.landmark_probe",
    "camloc.v2.body_masks",
]

# --- R02b: frozen driver scripts (configs/final/final-v1.yaml freeze.code) --------
# The listed research driver scripts and their script-level import closure are live release
# code (PRG-002 D-091 freeze, PRG-003 R02b brief). Lazy imports that only sit in functions the
# final run never calls (S3.1 GRU training, S2.8 telemetry-border filter, S1.11 research
# evaluators, ...) are explicitly excluded; everything else the closure reaches is copied.
DEFAULT_FREEZE_CONFIG = "configs/final/final-v1.yaml"
FREEZE_MANIFEST = ("manifests/final/final-v1.json", "manifests/final-v1-freeze.json")
PORTED = {"scripts/final_run.py"}        # hand-ported release driver (not byte-copied): recorded, not planned
SCRIPT_EXCLUDE = {
    "v2_s28_lib", "v2_s26_lib", "v2_s26_filter", "v2_s111_run",
    "v2_s31_eval", "v2_s31c_eval", "v2_s31d_eval", "v2_s31e_eval", "v2_s31f_eval",
    "v2_s31d_prep", "v2_s31e_prep", "v2_s31f_lib",
    "v2_s31c_train", "v2_s31e_train", "v2_s31f_train",
}
# camloc modules imported only inside driver functions the final run never calls (learned GRU post-processing in v2_s42_lib);
# `camloc.v2.learned.features` (pure numpy, imported at import time by v2_s31_prep / v2_s31c_prep) is kept.
CAMLOC_EXCLUDE = {"camloc.v2.learned.v1d"}
# scripts that use ROOT only for `src` or their own export defaults stay byte-identical
NO_ROOT_ADAPT = {"export_telemetry", "export_test_input", "run_inference", "v2_s43_vmax", "v2_s41_telemetry"}
ROOT_LINE = 'ROOT = Path(__file__).resolve().parents[1]'
ROOT_LINE_NEW = 'ROOT = Path(__import__("os").environ.get("CAMLOC_ASSET_ROOT") or Path(__file__).resolve().parents[1])'
ROOT_REASON = ("ROOT (data/asset root) taken from env CAMLOC_ASSET_ROOT when set (the release driver sets it to the "
               "asset bundle root, which mirrors the research artifacts/ manifests/ configs/ layout); falls back to the "
               "script's parent otherwise. One-line path adaptation, no scientific change.")
CONFIG_ASSET_RE = re.compile(r"(?<![\w/.$])(artifacts/|manifests/v2/|assets/final/)")
CONFIG_REASON = ("asset paths remapped to the release asset root: artifacts/, manifests/v2/, assets/final/ prefixed '$ASSETS/' "
                 "(the driver resolves '$ASSETS/' against --assets / CAMLOC_ASSETS / assets/final-v1); no value changed")

# JSON schemas the exported code opens at run time (grep of scripts/ and src/ for 'schemas/' and '*.schema.json')
SCHEMA_FILES = ["store_manifest_generic", "telemetry_manifest", "telemetry_row", "fusion_stream", "meas_v1_row", "prediction"]

# Generic CLIs (preserve names) — scripts/ relative to source root.
GENERIC_CLIS = [
    "scripts/run_release.py",
    "scripts/export_test_input.py",
    "scripts/export_telemetry.py",
    "scripts/package_release.py",
    "scripts/verify_release.py",
]

# Fixed configs/manifests the runner / packaging pin (filter params + constants).
CONFIG_FILES = [
    ("configs/models/e10g-g1.yaml", "configs/models/e10g-g1.yaml"),
    ("manifests/v2/S2.9-constants.json", "manifests/S2.9-constants.json"),
]

# LightGlue necessary licensed subset. camloc imports only ALIKED, LightGlue and
# three functions from lightglue.lightglue; aliked.py needs .utils. SuperPoint
# (restrictive Magic Leap licence) and the other unused backends / examples /
# tests / weights / .git are excluded.
LG_SRC_SUBDIR = "artifacts/external/S04-lightglue/LightGlue"
LG_DEST_SUBDIR = "external/S04-lightglue/LightGlue"
LG_KEEP = ["LICENSE", "lightglue/aliked.py", "lightglue/lightglue.py", "lightglue/utils.py"]
LG_INIT = "lightglue/__init__.py"
# Backends dropped from the adapted __init__ because their source is not shipped.
LG_DROP_IMPORT_TOKENS = ("superpoint", "disk", "dog_hardnet", "sift")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def adapt_lightglue_init(text: str) -> str:
    """Drop import lines for backends whose source is not shipped; keep ALIKED,
    LightGlue, match_pair. Mirrors the documented release-mc7-v1 patch, extended
    to the other unused backends so no dangling import remains at import time."""
    out = []
    for line in text.splitlines(keepends=True):
        low = line.strip().lower()
        if low.startswith("from .") and any(tok in low for tok in LG_DROP_IMPORT_TOKENS):
            continue
        out.append(line)
    return "".join(out)


class Planner:
    def __init__(self, src_root: Path, dest: Path):
        self.src_root = src_root
        self.dest = dest
        # list of dicts: {source_path, dest_path, role, orig_sha256,
        #                 copied_sha256, adaptation, payload(bytes)}
        self.plan: list[dict] = []
        self.freeze_info: dict | None = None

    def add_verbatim(self, rel_src: str, rel_dest: str, role: str) -> None:
        sp = self.src_root / rel_src
        if not sp.is_file():
            raise FileNotFoundError(f"source file missing: {rel_src}")
        data = sp.read_bytes()
        h = sha256_bytes(data)
        self.plan.append({
            "source_path": rel_src,
            "dest_path": rel_dest,
            "role": role,
            "orig_sha256": h,
            "copied_sha256": h,
            "adaptation": None,
            "_payload": data,
        })

    def add_transformed(self, rel_src: str, rel_dest: str, role: str,
                        new_data: bytes, reason: str) -> None:
        sp = self.src_root / rel_src
        orig = sha256_file(sp)
        self.plan.append({
            "source_path": rel_src,
            "dest_path": rel_dest,
            "role": role,
            "orig_sha256": orig,
            "copied_sha256": sha256_bytes(new_data),
            "adaptation": reason,
            "_payload": new_data,
        })



def _imports_of(path: Path) -> tuple[set[str], set[str]]:
    """(top-level script-style module names, camloc.* module names) imported anywhere in a script."""
    tree = ast.parse(path.read_text())
    names, camloc = set(), set()
    for n in ast.walk(tree):
        mods = []
        if isinstance(n, ast.Import):
            mods = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            mods = [n.module]
            if n.module.startswith("camloc"):
                mods += [f"{n.module}.{a.name}" for a in n.names]
        for m in mods:
            (camloc if m.startswith("camloc") else names).add(m.split(".")[0] if not m.startswith("camloc") else m)
    return names, camloc


def script_closure(src_root: Path, start: list[str]) -> tuple[list[str], set[str], list[str]]:
    """-> (sorted script module names incl. start, camloc modules they import, excluded modules actually reached)."""
    seen, camloc, excluded, todo = set(), set(), set(), list(start)
    sd = src_root / "scripts"
    while todo:
        m = todo.pop()
        if m in seen or not (sd / f"{m}.py").is_file():
            continue
        seen.add(m)
        names, cl = _imports_of(sd / f"{m}.py")
        camloc |= cl
        for x in names:
            if (sd / f"{x}.py").is_file():
                if x in SCRIPT_EXCLUDE:
                    excluded.add(x)
                else:
                    todo.append(x)
    return sorted(seen), camloc, sorted(excluded)


def freeze_selection(src_root: Path, freeze_config: str) -> dict:
    import yaml
    cfg = yaml.safe_load((src_root / freeze_config).read_text())
    code = cfg["freeze"]["code"]
    scripts = [Path(c).stem for c in code if c.startswith("scripts/") and c not in PORTED]
    cl_mods = [c[len("src/"):-3].replace("/", ".") for c in code if c.startswith("src/")]
    others = [c for c in code if not c.startswith(("scripts/", "src/"))]
    return {"scripts": scripts, "camloc": cl_mods, "others": others, "code": code}

def build_plan(src_root: Path, dest: Path, freeze_config: str | None = DEFAULT_FREEZE_CONFIG) -> Planner:
    sys.path.insert(0, str(src_root / "src"))
    from camloc.release.codeclosure import import_closure

    pl = Planner(src_root, dest)

    # 1. camloc runtime closure (final front end + D-086 library stack).
    fz = None
    extra_start: list[str] = []
    script_mods: list[str] = []
    if freeze_config:
        fz = freeze_selection(src_root, freeze_config)
        script_mods, script_camloc, excluded = script_closure(src_root, fz["scripts"])
        extra_start = sorted((set(fz["camloc"]) | script_camloc) - CAMLOC_EXCLUDE)
        pl.freeze_info = {"config": freeze_config, "freeze_code": fz["code"], "script_modules": script_mods,
                          "script_excluded_reached": excluded, "camloc_extra_start": extra_start}
    closure = import_closure(str(src_root / "src"), MC7_START + D086_START + extra_start)
    # Guard: no learned-layer module may enter the closure (excluded by D-086).
    learned = [m for m in closure if m.startswith("camloc/v2/learned") and m not in ("camloc/v2/learned/__init__.py", "camloc/v2/learned/features.py")]
    if learned:
        raise RuntimeError(f"learned-layer modules unexpectedly in closure: {learned}")
    for rel in closure:
        pl.add_verbatim(f"src/{rel}", f"src/{rel}", "runtime")

    # 2. generic CLIs.
    for rel in GENERIC_CLIS:
        pl.add_verbatim(rel, rel, "cli")

    # 2b. frozen driver scripts + script closure (R02b), ROOT-line path adaptation only
    done = {i["dest_path"] for i in pl.plan}
    for m in script_mods:
        rel = f"scripts/{m}.py"
        if rel in done:
            continue
        text = (src_root / rel).read_text()
        if m not in NO_ROOT_ADAPT and ROOT_LINE in text:
            pl.add_transformed(rel, rel, "driver", text.replace(ROOT_LINE, ROOT_LINE_NEW, 1).encode(), ROOT_REASON)
        else:
            pl.add_verbatim(rel, rel, "driver")
    if fz:
        for rel in fz["others"]:
            pl.add_verbatim(rel, rel, "config")
        # release config: asset paths remapped
        cfg_text = (src_root / freeze_config).read_text()
        pl.add_transformed(freeze_config, freeze_config, "config", CONFIG_ASSET_RE.sub(r"$ASSETS/\1", cfg_text).encode(), CONFIG_REASON)
        pl.add_verbatim(FREEZE_MANIFEST[0], FREEZE_MANIFEST[1], "manifest")
        for n in SCHEMA_FILES:
            pl.add_verbatim(f"schemas/{n}.schema.json", f"schemas/{n}.schema.json", "schema")

    # 3. fixed configs / constants.
    for rel_src, rel_dest in CONFIG_FILES:
        pl.add_verbatim(rel_src, rel_dest, "config")

    # 4. LightGlue necessary licensed subset.
    for rel in LG_KEEP:
        pl.add_verbatim(f"{LG_SRC_SUBDIR}/{rel}", f"{LG_DEST_SUBDIR}/{rel}", "third_party")
    init_src = src_root / LG_SRC_SUBDIR / LG_INIT
    adapted = adapt_lightglue_init(init_src.read_text()).encode()
    pl.add_transformed(
        f"{LG_SRC_SUBDIR}/{LG_INIT}", f"{LG_DEST_SUBDIR}/{LG_INIT}", "third_party",
        adapted,
        "dropped imports of unshipped backends (superpoint restrictive licence, "
        "plus unused disk/dog_hardnet/sift); kept ALIKED, LightGlue, match_pair",
    )
    return pl


def apply_plan(pl: Planner, *, write: bool) -> dict:
    """Apply (or dry-run) the plan. Refuse differing existing files."""
    conflicts = []
    accepted_existing = []
    written = []
    for item in pl.plan:
        target = pl.dest / item["dest_path"]
        payload = item["_payload"]
        if target.exists():
            existing = sha256_file(target)
            if existing == item["copied_sha256"]:
                accepted_existing.append(item["dest_path"])
                continue
            conflicts.append({
                "dest_path": item["dest_path"],
                "existing_sha256": existing,
                "would_write_sha256": item["copied_sha256"],
            })
            continue
        if write:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        written.append(item["dest_path"])
    if conflicts:
        raise SystemExit(
            "REFUSED: differing existing destination files (not byte-identical):\n"
            + "\n".join(f"  {c['dest_path']} (have {c['existing_sha256'][:12]}, "
                        f"want {c['would_write_sha256'][:12]})" for c in conflicts)
        )
    total_bytes = sum(len(i["_payload"]) for i in pl.plan)
    return {
        "written": written,
        "accepted_byte_identical": accepted_existing,
        "file_count": len(pl.plan),
        "total_bytes": total_bytes,
    }


def write_manifest(pl: Planner, src_head: str | None, applied: dict) -> Path:
    rows = []
    for i in pl.plan:
        rows.append({k: i[k] for k in (
            "source_path", "dest_path", "role", "orig_sha256", "copied_sha256", "adaptation")})
    manifest = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_root_head": src_head,
        "file_count": applied["file_count"],
        "total_bytes": applied["total_bytes"],
        "roles": _role_counts(rows),
        "freeze": pl.freeze_info,
        "ported": _ported_records(pl),
        "files": sorted(rows, key=lambda r: r["dest_path"]),
    }
    validate_manifest(manifest)
    out = pl.dest / "manifests" / "source-export.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def _ported_records(pl: Planner) -> list[dict]:
    out = []
    for rel in sorted(PORTED):
        sp, dp = pl.src_root / rel, pl.dest / rel
        frozen = json.loads((pl.src_root / FREEZE_MANIFEST[0]).read_text())["code"].get(rel)
        now = sha256_file(sp)
        out.append({"source_path": rel, "dest_path": rel,
                    "orig_sha256": frozen or now,            # the F1-frozen research file that was ported (freeze hash), not a later working-tree edit
                    "research_working_tree_sha256": now, "research_drifted_from_freeze": bool(frozen and frozen != now),
                    "copied_sha256": sha256_file(dp) if dp.is_file() else None,
                    "adaptation": "hand-ported release driver: pixi/VISION -> sys.executable, gpu_slot/flock removed, "
                                  "artifacts/ resolved via asset root, parity/freeze subcommands dropped (see docs/R02B_ADAPTATIONS.md)"})
    return out


def _role_counts(rows: list[dict]) -> dict:
    c: dict[str, int] = {}
    for r in rows:
        c[r["role"]] = c.get(r["role"], 0) + 1
    return dict(sorted(c.items()))


def validate_manifest(m: dict) -> None:
    """Schema check: fail loudly on any malformed row (the reviewer reruns)."""
    assert m.get("schema") == SCHEMA, "bad schema tag"
    assert isinstance(m.get("files"), list) and m["files"], "no files recorded"
    assert m["file_count"] == len(m["files"]), "file_count disagrees with files"
    seen_dest = set()
    for r in m["files"]:
        for k in ("source_path", "dest_path", "role", "orig_sha256", "copied_sha256"):
            assert k in r, f"row missing {k}: {r}"
        assert r["dest_path"] not in seen_dest, f"duplicate dest {r['dest_path']}"
        seen_dest.add(r["dest_path"])
        for hk in ("orig_sha256", "copied_sha256"):
            h = r[hk]
            assert isinstance(h, str) and len(h) == 64 and all(
                ch in "0123456789abcdef" for ch in h), f"bad {hk} in {r['dest_path']}"
        adapted = r["orig_sha256"] != r["copied_sha256"]
        assert bool(r.get("adaptation")) == adapted, (
            f"adaptation reason must be present iff transformed: {r['dest_path']}")
        # no research driver scripts, raw data, weights, .git
        bad = (".git", "artifacts/maps", ".pth", ".pt", ".npz", ".npy", ".mcap")
        assert not any(b in r["dest_path"] for b in bad), (
            f"forbidden content in export: {r['dest_path']}")


def main(argv=None) -> int:
    here = Path(__file__).resolve()
    default_dest = here.parents[1]  # the sibling release root (tools/..)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-root", required=True, type=Path,
                    help="research checkout root (the source of truth; read-only)")
    ap.add_argument("--dest", type=Path, default=default_dest,
                    help="destination release root (default: this tool's parent)")
    ap.add_argument("--source-head", default=None,
                    help="optional parent git HEAD to record in the manifest")
    ap.add_argument("--freeze-config", default=DEFAULT_FREEZE_CONFIG,
                    help="research-relative freeze config whose freeze.code lists the live release scripts/modules "
                         "(R02b); pass '' for the R02a-only export")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and validate only; write nothing")
    args = ap.parse_args(argv)

    src_root = args.source_root.resolve()
    dest = args.dest.resolve()
    if not (src_root / "src" / "camloc").is_dir():
        ap.error(f"--source-root {src_root} has no src/camloc")
    if src_root == dest:
        ap.error("source-root and dest must differ")

    pl = build_plan(src_root, dest, args.freeze_config or None)
    applied = apply_plan(pl, write=not args.dry_run)
    if args.dry_run:
        # still validate the manifest shape without writing it
        rows = [{k: i[k] for k in ("source_path", "dest_path", "role",
                 "orig_sha256", "copied_sha256", "adaptation")} for i in pl.plan]
        validate_manifest({"schema": SCHEMA, "file_count": len(rows), "files": rows})
        print(f"[dry-run] {applied['file_count']} files, {applied['total_bytes']} bytes; "
              f"{len(applied['written'])} to write, "
              f"{len(applied['accepted_byte_identical'])} already byte-identical")
        return 0
    out = write_manifest(pl, args.source_head, applied)
    print(f"exported {applied['file_count']} files ({applied['total_bytes']} bytes): "
          f"{len(applied['written'])} written, "
          f"{len(applied['accepted_byte_identical'])} byte-identical reuse")
    print(f"manifest: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
