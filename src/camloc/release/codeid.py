"""Identity of the code that is actually executing a release run (stdlib only).

There is exactly one execution mode: the checkout's `src/` (repo mode). The bundle carries a code snapshot as provenance; the run
compares the checkout against it by hash and refuses a mismatch. `executing_mode()` derives the mode from where `camloc` was
really imported from, so a run record can never claim a mode that differs from the code that ran (tested).
"""
from __future__ import annotations

import sys
from pathlib import Path

from camloc.util import canonical_hash, sha256_file


def camloc_root() -> Path:
    """Directory that contains the imported `camloc` package (the `src/` of the checkout in repo mode)."""
    import camloc
    return Path(camloc.__file__).resolve().parents[1]


def executing_mode(bundle) -> str:
    """'bundle' if the imported camloc lives inside the bundle directory, else 'repo'."""
    root, b = camloc_root(), Path(bundle).resolve()
    return "bundle" if (root == b or b in root.parents) else "repo"


def repo_counterpart(rel_in_bundle_code: str, src_root: Path) -> Path:
    """code/camloc/x.py -> <src>/camloc/x.py ; code/scripts/x.py -> <repo>/scripts/x.py."""
    rel = rel_in_bundle_code[len("code/"):]
    return (src_root / rel) if rel.startswith("camloc/") else (src_root.parent / rel)


def check_code(man: dict, src_root: Path) -> dict:
    """Compare every code file of the bundle manifest with the file of the executing checkout. tree_hash uses the same keying as
    the manifest's code.code_hash (src/<rel> for modules, <rel> for scripts), so tree_hash == snapshot_hash iff nothing differs."""
    bad, items = [], []
    for rel, e in man["files"].items():
        if e["role"] != "code":
            continue
        f = repo_counterpart(rel, src_root)
        sha = sha256_file(f) if f.is_file() else None
        r = rel[len("code/"):]
        items.append(((f"src/{r}" if r.startswith("camloc/") else r), sha))
        if sha != e["sha256"]:
            bad.append(rel)
    return {"code_files_checked": len(items), "mismatches": bad, "tree_hash": canonical_hash(sorted(items)),
            "snapshot_hash": man["code"]["code_hash"]}


def loaded_camloc_modules(src_root: Path) -> dict:
    """camloc.* modules currently imported, split into those under src_root and those elsewhere (a different camloc on the path)."""
    inside, outside = {}, {}
    for name, m in sorted(sys.modules.items()):
        if name != "camloc" and not name.startswith("camloc."):
            continue
        f = getattr(m, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        (inside if src_root in p.parents else outside)[name] = p
    return {"inside": inside, "outside": outside}


def unpinned_modules(man: dict, src_root: Path) -> list[str]:
    """Imported camloc modules whose file is not a code file of the bundle snapshot (executed but not hash-pinned)."""
    pinned = {str(repo_counterpart(rel, src_root).resolve()) for rel, e in man["files"].items() if e["role"] == "code"}
    mods = loaded_camloc_modules(src_root)
    return sorted([n for n, p in mods["inside"].items() if str(p) not in pinned] + [f"{n} (outside src)" for n in mods["outside"]])
