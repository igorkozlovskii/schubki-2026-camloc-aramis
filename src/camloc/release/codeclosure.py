"""Static import closure of camloc modules (AST, includes function-level imports). Used to snapshot exactly the code a release runs."""
from __future__ import annotations

import ast
from pathlib import Path


def _mod_file(src: Path, mod: str):
    base = src / Path(*mod.split("."))
    if (base.with_suffix(".py")).is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def import_closure(src_root, start_modules) -> list[str]:
    """Return sorted repo-src-relative posix paths of every camloc .py file reachable from start_modules (plus parent package __init__s)."""
    src = Path(src_root)
    seen: dict[str, Path] = {}
    todo = list(start_modules)
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        f = _mod_file(src, mod)
        if f is None:
            continue
        seen[mod] = f
        parts = mod.split(".")
        for i in range(1, len(parts)):          # parent packages execute their __init__
            todo.append(".".join(parts[:i]))
        is_pkg = f.name == "__init__.py"
        pkg = mod if is_pkg else ".".join(parts[:-1])
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith("camloc"):
                        todo.append(a.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg.split(".")
                    base = base[: len(base) - (node.level - 1)]
                    target = ".".join(base + ([node.module] if node.module else []))
                else:
                    target = node.module or ""
                if not target.startswith("camloc"):
                    continue
                todo.append(target)
                for a in node.names:
                    todo.append(f"{target}.{a.name}")   # `from camloc import runner` / `from .store import X` (X not a module -> ignored)
    return sorted(f.relative_to(src).as_posix() for f in seen.values())
