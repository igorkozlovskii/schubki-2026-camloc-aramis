# Camera-based race-car localization — organizer release

**Status: the frozen final model (D-091 final-v1) runs from a raw MCAP and a supplied first pose to
`pred.csv` with only this directory's code, its pinned environment and the asset bundle.** One end-to-end run
on the shared organizer test bag completed (pipeline check only). Reproduction against the research run is
**not bitwise**: measured per-frame probe latency feeds the causal availability times of the measurements, so
output differs slightly between runs on a loaded machine (`docs/R02B_ADAPTATIONS.md`, receipt of R02b for the
numbers). Packaging, scoring and the quick start below are still being finalised.

## What this will be

A compact, standalone repository that reproduces our causal localization
pipeline for the Yas Marina challenge: from the raw onboard-camera MCAP plus one
supplied first pose to a submission CSV, on a single-GPU workstation. It is a
delivery snapshot of a selected candidate from the research project — it does
not depend on the research checkout at run time.

The scientific owner's current final model (research decision D-086,
2026-10-18) is **mc7 + borders + poles + speed-aware (dynamic) v_max**, without
a learned/GRU layer. PRG-003 ships exactly this one model (D-007); the older
option A′ is historical packaging evidence and is **not** shipped as a release
choice. The source closure (`camloc` modules, the frozen driver scripts and their import closure, pinned configs, the necessary LightGlue subset), one pinned Pixi environment and the asset bundle are in this directory. See `docs/RUNTIME_FOUNDATION.md`, `docs/R02B_ADAPTATIONS.md`, `docs/ASSETS.md` and `RELEASE_PLAN.md`.

## Supplied-pose input / output contract

- **Input:** the organizer's raw `*.mcap` (7 cameras, calibration, track
  borders) **plus one supplied first pose**, delivered as a separate small file
  `{timestamp_ns, x, y, yaw}` (JSON or one-row CSV). The first pose is consumed
  exactly once; no later ground-truth pose, track-state or future observation
  enters inference. Permitted telemetry fields may be used by the telemetry
  option and are documented with the exact allowlist.
- **Output:** `pred.csv` with header `timestamp_ns,x,y,yaw` (metres, radians, map
  frame) at the 10 Hz camera rate, plus per-tick diagnostics and a run record.
- **Causality:** a pose at time *t* uses only data stamped ≤ *t*.

## Environment and run flow

```bash
CONDA_OVERRIDE_CUDA=12.4 pixi install --locked          # CUDA override only works around driver detection on one machine
# asset bundle: see docs/ASSETS.md (fetch, unpack to assets/final-v1, verify)
pixi run python tools/verify_assets.py --assets assets/final-v1
pixi run camloc-run --mcap <bag.mcap> --first-pose <first_pose.csv|json> --out <new dir> [--assets <dir>] [--wait-vram-mb 9000]
```

`--out` must be a new directory; `pred.csv` (`timestamp_ns,x,y,yaw`) and `run_summary.json` are written there. Measured on the test bag (722 ticks, 72 s of data, RTX 4090 shared with other jobs):
about 9 GB GPU memory, 1.9 GB RAM, 7-9 minutes on a busy machine (about 3.6 minutes when the machine was quiet in the research run). Asset and licence details: `docs/ASSETS.md`, `licenses/`.

## Not included

Raw organizer recordings, ground truth, training caches, credentials and
machine-local paths are never part of this delivery. Trained weights and maps
are delivered as a verified asset bundle (or fetch-and-verify links), separate
from the source.

## Provenance

This repository is prepared by program PRG-003 in the research checkout. The
internal inventory that backs the eventual export lives there
(`reports/release/PRG003-R01/`) and is not part of the organizer delivery.
