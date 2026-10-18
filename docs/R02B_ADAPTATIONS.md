# Frozen driver port: what differs from the research files 

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
