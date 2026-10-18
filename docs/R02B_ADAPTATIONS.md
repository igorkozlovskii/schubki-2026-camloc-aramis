# Frozen driver port: what differs from the research files (final-v1 R02b; final-v2 R05 section at the end)

No parameter, threshold, constant, map, weight or algorithm differs from the D-091 freeze (`configs/final/final-v1.yaml`, `manifests/final-v1-freeze.json`).
Every adapted file's original and copied sha256 are in `manifests/source-export.json` (`adaptation` is non-null exactly when they differ).

| File | Change |
|---|---|
| `scripts/v2_s42_lib.py`, `s07_fusion_lib.py`, `v2_s11_common.py`, `s07c_faults.py` | one line: `ROOT` is `$CAMLOC_ASSET_ROOT` when set, else the script's parent (all other drivers inherit `ROOT` from these) |
| `configs/final/final-v1.yaml` | asset paths prefixed `$ASSETS/` (the driver resolves them against the asset root); undoing the prefix gives the research config byte for byte (tested) |
| `external/.../lightglue/__init__.py` | unshipped backend imports dropped |
| `scripts/final_run.py` (hand port, research sha256 `01558e66...`) | `pixi run ... scripts/X` and `VISION` replaced by `sys.executable scripts/X`; `scripts/gpu_slot.sh`, `/tmp/camloc-gpu*.lock` and `GpuSlot` removed (probe processes run one after the other as before; optional `--wait-vram-mb`); frozen assets resolved through the asset root; probe configs get absolute `map_dir` and `ext_dir`; `--split configs/splits/release-v1.yaml` passed to the probe worker; optional path audit (`CAMLOC_AUDIT_DIR`); `parity` and `freeze` subcommands (need research stores) removed |
| `configs/splits/release-v1.yaml` (new) | empty lap table: the research split's only run-time use is the audit-lap guard; the release carries no research lap or audit intervals |

Kept: GT guard (`install_gt_guard`), the rl/rr S4.1 rule and its refusal of any other calibration difference, the camera-only refusal, `--vmax dyn|off`, the CSV schema.

Not shipped from the closure of the listed scripts: `v2_s28_lib`, `v2_s26_*`, `v2_s111_run`, `v2_s31{d,e,f}_*`, `v2_s31c_train`, `camloc.v2.learned.v1d` (S3.1/S2.8 research paths and GRU post-processing that the final run never calls).
`camloc.v2.learned.features` (numpy only) is shipped because two S3.1 prep scripts import it at import time.

## Environment
`pixi.toml` pins python 3.11.6, pyarrow 12.0.1, numpy 1.26.4, pandas 2.3.3, scipy 1.17.1, kornia 0.8.3, opencv-python-headless 4.10.0.84, transformers 4.57.6, safetensors 0.8.0 (the versions in the research vision lock
that ran F1) and `pytorch ==2.3.1`, `pytorch-cuda ==12.1`. The pyarrow conda build differs from the research lock (same version). `CUDA_HOME` is not needed: the measured stages compile no torch extension.
If a stage ever JIT-compiles one, set `TORCH_EXTENSIONS_DIR` (default `<out>/work/torch_extensions`) and `CUDA_HOME` to your CUDA toolkit.

## final-v2 (PRG-003 R05)

`configs/final/final-v2.yaml` is the default; `final-v1.yaml` stays runnable (`--config configs/final/final-v1.yaml`, before the subcommand).
The research keeps two drivers (`scripts/final_run.py` sha `01558e66...` for v1, `scripts/final_run_v2.py` sha `3865924f...` for v2, F2 additive `--config`/fast-stream changes). The release has ONE merged `scripts/final_run.py`:
the final-v2 branches (`is_fast`, `fast_model_config`, `fast_streams`, the `FINAL_PROBE_CAM=fast` probe-worker branch, `--config`) are copied verbatim and are active only when the config has a `six_cams` block; the v1 path is the R02b driver unchanged.
Adaptations of the v2 additions (paths only; the E parameters are read unchanged from `configs/v2/s122/generated/E-lap1.yaml`, hashed in `manifests/final-v2-freeze.json`):

| Where | Change |
|---|---|
| `fast_model_config` | `map_dirs` and `ext_dir` are absolute paths under the asset root (research: paths relative to the checkout and the default `artifacts/external`) |
| `stage_probes` | the config path rewrite for `map_dir`/`ext_dir` is skipped for the FastMultiCamModel config (already absolute) |
| `cmd_parity_v2`, `cmd_freeze`, the `parity-v2` / `freeze` subcommands | removed (research stores), as `parity`/`freeze` in R02b |
| `scripts/final_run_v2.py` | not shipped separately; `model.driver` in `final-v2.yaml` still names it (config kept byte-for-byte apart from `$ASSETS/`) |
| `src/camloc/v2/fastmatch/`, `configs/v2/s122/generated/E-lap1.yaml`, `configs/final/final-v2.yaml`, `manifests/final-v2-freeze.json` | exported by `tools/export_sources.py` (now takes both freeze configs; `source-export.json` records both ported drivers) |

No new asset: the final-v2 freeze manifest has the same maps/networks/constants/assets as final-v1, so `assets-final-v1` and its tarball are reused. Known research caveat (F2 receipt): the E streams do not pass `validate_meas_v1` (no s/n/dpsi/sigma columns); neither did the S1.22/S1.22b streams; the driver checks non-empty, monotone, finite and on the tick grid instead.

Update after this section was written: research re-froze `scripts/final_run_v2.py` at commit 32eeeaf (sha256 `da202707...`; the change is the docstring path and `SELF` for the probe-worker command only). The shipped merged driver already behaves like the re-frozen one.

Three small differences on the v1 path of the merged driver, with no numeric effect: `log["config"]` is written to `run_summary.json`; `--config` is `resolve()`d; `stage_probes` calls `cfg_load()` for v1.
