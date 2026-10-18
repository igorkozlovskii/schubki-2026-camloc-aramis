#!/usr/bin/env python3
"""PRG-003 R02a — standalone dependency / boundary probe.

Imports the whole runtime third-party stack and the selected final-model camloc
modules in ONE process and prints each package's resolved file origin, proving
they come from THIS sibling environment and tree (not a parent checkout). It
constructs no model, loads no checkpoint and runs no GPU workload.

Run from any working directory with the sibling's locked environment, e.g.:

  CONDA_OVERRIDE_CUDA=12.4 pixi run --manifest-path <sibling>/pixi.toml \
      python <sibling>/tools/dep_probe.py

Exit code 0 = every probed import resolved from inside the sibling env/tree.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LG = ROOT / "external" / "S04-lightglue" / "LightGlue"
ENV = ROOT / ".pixi"

for p in (str(SRC), str(LG)):
    if p not in sys.path:
        sys.path.insert(0, p)

THIRD_PARTY = [
    "numpy", "pandas", "pyarrow", "scipy", "yaml", "jsonschema",
    "torch", "torchvision", "cv2", "kornia", "transformers", "safetensors",
    "mcap", "mcap_ros2", "lightglue",
]
CAMLOC = [
    "camloc", "camloc.models.mc7", "camloc.release.run", "camloc.release.package",
    "camloc.geometry.borders", "camloc.v2.border_seg_att", "camloc.v2.poles.detect",
    "camloc.v2.seg.infer", "camloc.v2.road_attitude", "camloc.v2.landmark_probe",
    "camloc.io.export_telemetry",
]


def origin(mod) -> str:
    return getattr(mod, "__file__", "<builtin>") or "<namespace>"


def in_any(path: str, *roots: Path) -> bool:
    rp = Path(path).resolve()
    return any(str(rp).startswith(str(r.resolve())) for r in roots)


def main() -> int:
    bad = []
    print(f"python {sys.version.split()[0]}  env={ENV}")
    print("--- third-party runtime stack ---")
    for name in THIRD_PARTY:
        m = importlib.import_module(name)
        o = origin(m)
        ok = name == "lightglue" and in_any(o, LG) or in_any(o, ENV)
        print(f"  {'OK ' if ok else 'BAD'} {name:16s} {o}")
        if not ok:
            bad.append((name, o))
    print("--- selected final-model camloc modules ---")
    for name in CAMLOC:
        m = importlib.import_module(name)
        o = origin(m)
        ok = in_any(o, SRC)
        print(f"  {'OK ' if ok else 'BAD'} {name:28s} {o}")
        if not ok:
            bad.append((name, o))
    import torch
    print(f"--- torch {torch.__version__} cuda-build {torch.version.cuda} "
          f"(no device init) ---")
    if bad:
        print("FAIL: imports resolved outside the sibling:")
        for n, o in bad:
            print(f"  {n}: {o}")
        return 1
    print("PASS: all probed imports resolved inside the sibling environment/tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
