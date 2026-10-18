# Runtime foundation

**Status: historical foundation notes.** This page describes the source foundation and the single
environment that the final model runs on. The complete run (frozen driver, asset bundle, end-to-end run on the
test bag) is described in `README.md`, `docs/ASSETS.md` and `docs/R02B_ADAPTATIONS.md`; the commands below were
verified for the foundation and still apply.

## What the foundation contains

1. **A standalone source foundation** for the single final model (research
   decision D-086: mc7 + borders + poles + speed-aware / dynamic v_max, **no**
   learned/GRU layer):
   - `src/camloc/…` — the runtime import closure of the final front end plus the
     border / pole / road-attitude / segmentation library stack (67 modules).
   - `scripts/` — the five generic CLIs: `run_release.py`, `export_test_input.py`,
     `export_telemetry.py`, `package_release.py`, `verify_release.py`.
   - `configs/models/e10g-g1.yaml`, `manifests/S2.9-constants.json` — fixed
     filter / constant parameters the runner and packaging pin.
   - `external/S04-lightglue/LightGlue/` — the necessary licensed LightGlue
     subset (see "Third-party subset" below).
   - `manifests/source-export.json` — the schema-checked export manifest
     (`source-export-v1`) recording every file's original and copied sha256 and
     any adaptation reason.

2. **One dedicated, pinned Pixi environment** (`pixi.toml` + `pixi.lock` at the
   repo root) containing BOTH the model/vision stack and the MCAP reader/exporter
   stack. Streamlit, JupyterLab and research-only tooling are excluded.

### Added after the foundation

- The frozen final driver `scripts/final_run.py` (Pixi task `camloc-run`) and its script closure, with dynamic
  v_max as part of the frozen modules.
- The asset bundle (maps, pole map, masks, segmentation weights, LightGlue/ALIKED/DINOv2 weights, calibration),
  delivered as a release asset and verified by `tools/verify_assets.py` (`docs/ASSETS.md`).
- An end-to-end run on the test bag and its parity and resource measurements (`README.md`, `docs/RESULTS.md`).

No older runner option (A′ `release-e04t-fb-v2`, `release-mc7-v1`) is shipped as
a release choice. Per D-086 and PRG-003 D-007, exactly one final model is the
target; A′ is historical packaging evidence only and is **not** exported here.

## The one environment

`pixi.toml` (root) reuses the proven model pins from the research vision
environment (`environments/vision/pixi.toml`) verbatim — Python 3.11,
PyTorch 2.3.1 / pytorch-cuda 12.1, torchvision 0.18, numpy 1.26, pandas < 3,
scipy, pyarrow, pyyaml, jsonschema, pytest; PyPI kornia,
opencv-python-headless < 4.11, transformers < 5, safetensors — and adds the MCAP
reader/exporter dependencies `mcap` and `mcap-ros2-support` (PyPI) that the two
export CLIs need. This single environment replaces the research project's
model / exporter env split.

Import paths are set by `[activation]` and point **only inside this tree**:
`PYTHONPATH = $PIXI_PROJECT_ROOT/src:$PIXI_PROJECT_ROOT/external/S04-lightglue/LightGlue`,
with `PYTHONNOUSERSITE=1` and `HF_HUB_OFFLINE=1` (no weight fetch on import).
No parent manifest, lock or `.pixi/` is referenced or copied.

### Install and verify (verified commands)

`CONDA_OVERRIDE_CUDA=12.4` is a workaround for this machine's CUDA detection; it
is not an organizer requirement and no absolute CUDA/compiler/cache path is
encoded as one.

```bash
# 1. Install the locked environment (no parent writes)
CONDA_OVERRIDE_CUDA=12.4 pixi install --locked \
  --manifest-path /ABSOLUTE/PATH/TO/schubki_2026_release/pixi.toml

# 2. Dependency/boundary probe (prints each import's file origin; no model built)
CONDA_OVERRIDE_CUDA=12.4 pixi run --manifest-path .../pixi.toml \
  python tools/dep_probe.py

# 3. Runtime-foundation tests (import / path / closure checks)
CONDA_OVERRIDE_CUDA=12.4 pixi run --manifest-path .../pixi.toml \
  python -m pytest tests/test_runtime_foundation.py -q -p no:cacheprovider

# 4. Generic CLIs resolve from any working directory
CONDA_OVERRIDE_CUDA=12.4 pixi run --manifest-path .../pixi.toml \
  python scripts/run_release.py --help     # and the other four CLIs
```

## Re-running the export

`tools/export_sources.py` is the rerunnable, narrow exporter. It takes an
explicit source root and destination (no hard-coded user paths), computes the
camloc closure with the project's own AST helper
(`camloc.release.codeclosure.import_closure`), and writes
`manifests/source-export.json`.

```bash
python tools/export_sources.py \
  --source-root /ABSOLUTE/PATH/TO/schubki_2026 \
  --dest        /ABSOLUTE/PATH/TO/schubki_2026_release \
  --source-head <research git HEAD>
```

- A rerun on byte-identical source is safe: already-present identical files are
  accepted and nothing is rewritten.
- A destination file that differs from what would be written is **refused**
  (non-zero exit); nothing is silently overwritten.

## Third-party subset (LightGlue / ALIKED)

Only the files the pipeline actually needs are shipped:
`LICENSE`, `lightglue/aliked.py`, `lightglue/lightglue.py`, `lightglue/utils.py`
(verbatim) and an **adapted** `lightglue/__init__.py`. The adaptation drops the
import lines for backends whose source is not shipped — `superpoint`
(restrictive Magic Leap licence), `disk`, `dog_hardnet`, `sift` — keeping only
`ALIKED`, `LightGlue` and `match_pair`, which is all camloc imports. The
original and resulting sha256 of the adapted `__init__.py` are recorded in
`manifests/source-export.json` and in `licenses/`. No LightGlue weights, `.git`,
examples, notebooks or tests are copied. No weight is fetched on import.

## Boundary guarantees of the foundation

- `camloc` and all selected final-model modules import from this sibling's
  `src/` (asserted against the file origin), never from a research checkout.
- The vendored `lightglue` imports as a top-level package from
  `external/S04-lightglue/LightGlue`.
- The AST import closure of the final-model start modules equals the exported
  `src/camloc` runtime set in `manifests/source-export.json` exactly — no missing
  and no stray module — and contains **zero** learned-layer modules.
- No research driver script (`v2_s*`) is importable and no loaded module file
  resolves into a research checkout path.
- The full third-party runtime stack (torch/torchvision, numpy/pandas/pyarrow/
  scipy, OpenCV, kornia, transformers, safetensors, yaml/jsonschema, mcap)
  imports in one process from this environment.
