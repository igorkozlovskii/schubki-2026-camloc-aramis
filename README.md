# Camera-based race-car localization on Yas Marina (SchubKI 2026)


**Status.** The frozen final model runs from a raw MCAP and a supplied first pose to `pred.csv` using only this
directory's code, its pinned environment and the asset bundle. It was run end to end on the organizers' shared
test bag (72 s, no ground truth: a pipeline check, not a score).

## What it does

The program estimates the position and heading of the race car on Yas Marina from the seven
onboard cameras, the supplied calibration, a prebuilt landmark map of the track, the track
borders and a restricted set of car-state telemetry. It starts from one supplied pose and
then emits causal poses at 10 Hz: the pose at time t uses only data stamped at or before t.
Each of the seven cameras is matched against its own landmark map built from the four training
laps (0, 1, 3, 5); this seven-camera landmark matching is called "mc7" in the documents. Closed-loop
track-border fits and pole bearings correct the result, and a telemetry-driven track-coordinate
filter fuses everything. The submitted system is the configuration frozen as `final-v2` (release v2.0.0; `final-v1`, release v1.0.0, stays
runnable), without a learned sequence layer. `final-v2` differs from `final-v1` only in how the six non-front
cameras are matched: the S1.22 "E" matcher (`camloc.v2.fastmatch`, one batched-feature model run) replaces the six
per-camera probes; maps, borders, poles, filter, constants and the asset bundle are the same.

## Inputs and contract

| Input | What is used |
|---|---|
| Raw organizer MCAP | the seven camera image topics `/constructor0/sensor/camera_{fl,fr,cl,cr,r,rl,rr}/compressed_image` (JPEG, 10 Hz, header stamp = exposure time), their `camera_info`, the camera extrinsics (`/tf_static`), and the telemetry topics named below |
| First pose | one pose `timestamp_ns, x, y, yaw`, supplied separately for each run; consumed once at the start |
| Track bounds | packaged with the release (`track_bounds.json`, hashed) |

- **First pose.** The organizers supply the first pose of each run. The organizers have not
  specified how it is delivered; this program reads it from a file (`--first-pose`, CSV
  `timestamp_ns,x,y,yaw` or JSON) and consumes it once. The first pose is also passed to the
  camera probes' prior at start; nothing else carries a pose. There is no relocalization and no
  later reset from any ground-truth value.
- **Telemetry (fallback order).** (1) The allowlisted car-state fields of
  `/constructor0/state_estimation` (38 allowlisted fields, listed in `docs/RESULTS.md`; the
  exporter is default-deny). The filter itself reads only four of them: `v_mps`, `vx_mps`,
  `vy_mps`, `wz_radps`. The other 34 allowlisted fields are written to the exported telemetry file but are not loaded by the filter. (2) If that topic is absent, the
  `/constructor0/validation/twist` and `/accel` pair. (3) Camera-only operation is not
  implemented in this release: with neither source present the run is refused.
- **Forbidden fields, never read:** `x_m`, `y_m`, `z_m`, `yaw_rad`, `roll_rad`, `pitch_rad`,
  `yaw_vel_rad`, `sn_map_state`, `se_state`, `se_status`, the position and velocity accuracy
  fields. Ground truth, full-run reference lines and future observations never enter inference.
- **Causality.** Camera frames, telemetry rows and border fits are used only after their stamp;
  a border fit becomes available after its measured service time.
- **Calibration.** The run reads the calibration from the input file. The rear-left and
  rear-right extrinsics are corrected by a fixed rule when they differ from training; any other
  difference is refused with a message.

## Output

`pred.csv` with the header `timestamp_ns,x,y,yaw`: exposure timestamp in nanoseconds, map-frame
x and y in metres, yaw in radians, six decimals, one row per 10 Hz tick. At least 90 % of the
one-second intervals of a run need a pose for the run to be ranked. Per-tick diagnostics and a
run record are written beside it.

## Requirements

| Item | Value |
|---|---|
| GPU | one CUDA GPU; peak about 9 GB of GPU memory in the measured run (whole-GPU nvidia-smi sampling every 2 s minus the baseline, so it includes other processes' variation) |
| System memory | peak resident set 1.9 GB (largest process) |
| Disk | asset tarball 0.91 GB (913,162,240 B), unpacked about 0.9 GB; Pixi environment about 7 GB; working space per run about 0.4 GB (store export 399 MB) plus 37 KB of output |
| Reference machine for timing | one RTX 4090 shared with other jobs (load average 11 to 14); a quiet-machine time of this release driver was not measured |

## Install

The environment is one pinned Pixi project (`pixi.toml`, `pixi.lock`).

```bash
pixi install --locked
```

`CONDA_OVERRIDE_CUDA=12.4` in front of the command is needed only on a machine whose CUDA detection fails
(it was needed on the machine where this was developed); it is not otherwise required. Python 3.11.6, PyTorch
2.3.1 (CUDA 12.1 build), no network access is needed at run time (`HF_HUB_OFFLINE=1`).

## Assets

The trained and derived assets (per-camera landmark maps, the pole map, the border network,
the ALIKED and LightGlue checkpoints, DINOv2-small, the reference calibration) are delivered as
one versioned asset of the repository's GitHub release.

- Download: `camloc-aramis-assets-final-v1.tar` from
  <https://github.com/igorkozlovskii/schubki-2026-camloc-aramis/releases/tag/v1.0.0>
- Size: 913,162,240 bytes (76 files, 913,103,441 bytes unpacked); SHA-256:
  `2f3fffc60b5dcecd211f55912c0779df6fa9c1c4c47c80e16f434199f6d981f9`

```bash
gh release download v1.0.0 -R igorkozlovskii/schubki-2026-camloc-aramis -p camloc-aramis-assets-final-v1.tar
sha256sum camloc-aramis-assets-final-v1.tar            # compare with the value above
pixi run python tools/verify_assets.py --tar camloc-aramis-assets-final-v1.tar --work /tmp/unpack-check
mkdir -p assets && tar -xf camloc-aramis-assets-final-v1.tar -C assets && mv assets/assets-final-v1 assets/final-v1
pixi run python tools/verify_assets.py --assets assets/final-v1
```

The verifier hashes every file, compares the frozen groups with `manifests/final-v1-freeze.json` and refuses extra
files. Details: [docs/ASSETS.md](docs/ASSETS.md).

## Run

```bash
pixi run camloc-run --mcap <bag.mcap> --first-pose <first_pose.csv|json> --out <new dir> [--assets assets/final-v1] [--wait-vram-mb 9000]
```

`camloc-run` is the Pixi task for `python scripts/final_run.py --config configs/final/final-v2.yaml run` (final-v2, the default).
Final-v1 is still runnable: `pixi run camloc-run-v1 ...` or `python scripts/final_run.py --config configs/final/final-v1.yaml run ...`. Arguments: the raw MCAP, the first-pose file,
and a new output directory (the run refuses an existing one). `--assets` defaults to `assets/final-v1` (or the
environment variable `CAMLOC_ASSETS`). Outputs in the directory: `pred.csv`, `run_summary.json`, per-tick
diagnostics.

**Measured on the test bag** (722 ticks, 72.1 s of data, one RTX 4090 shared with other jobs): two runs took 431 s
and 543 s wall clock (about 0.6 and 0.75 s per output tick; 384 s and 498 s of it image probing), with about 9 GB
GPU memory and 1.9 GB RAM. The research run of the same model on a quiet machine took 215 s (298 ms per tick); a
quiet-machine time of the release driver is not measured. The program is offline in structure, not real-time.

**Reproducibility.** Two identical runs are not bitwise equal. The measured per-frame latency of the image
probes decides when each camera measurement becomes available to the causal filter, so a loaded machine changes
the output slightly. On the test bag the two release runs differ from the research output by 14.0 mm and 9.5 mm
pooled position RMS (median 1.3 mm, 95th percentile 18 mm, maximum 0.22 m and 0.085 m); the research driver
rerun on the same inputs differs from its own earlier output by 8.4 mm RMS (median 1.3 mm). The six camera
measurement streams of the release runs are identical to the research run's; only the times at which they become
available differ. The release therefore reproduces the research driver within its run-to-run envelope (8-14 mm
RMS, median about 1 mm, maximum up to 0.22 m on this bag), not bitwise.

## Results

Position RMSE in metres at the per-tick pose level. The audit rows were scored lap by lap with
the unchanged organizer scorer (coverage 100 %); the pooled value is computed from the two laps'
per-tick errors. The dev and held-out rows come from a re-implementation of the organizer scoring
formulas (same ground-truth interpolation and 50 ms gap rule). On earlier development runs that
were also scored with the organizer script, the two agreed in position RMSE (equal in one check,
within 0.005 m in another), and our scoring wrapper refuses any run on which they differ by more
than 0.0006 in any printed metric. The dev and held-out rows of this table were not each scored
with the organizer script. Pooled values over several laps are tick-pooled and
are not an organizer metric. "Held-out" means leave-one-lap-out: the maps and networks used for
that lap were built without it. Dev, held-out and audit sets are different populations and are
not averaged together; details, sets and caveats are in [docs/RESULTS.md](docs/RESULTS.md).

| Evaluation set | Lighting | First pose | Position RMSE (m) |
|---|---|---|---|
| Held-out laps 1, 3, 5 (leave-one-lap-out maps), pooled | night | GT first pose | 0.326 |
| Dev lap 2 | night | GT first pose | 0.276 |
| Audit laps 6 and 7, scored once, pooled (2,419 ticks) | night | GT first pose | 0.420 |
| Dev lap 4, relit with the v3 generator | simulated daylight | GT first pose | 0.590 |
| Dev lap 2, sunnier variant | simulated daylight | GT first pose | 0.699 |
| Held-out lap 1, relit with the v3 generator | simulated daylight | GT first pose | 1.104 |

The audit row is the estimate for night conditions of the training session; the hidden run may differ in speed and time of day. Daylight rows use images relit by a
synthetic generator and say how the method degrades, not how it performs on real daylight.
The shared second recording (higher speed, different time of day) was used only to check that
the pipeline runs; it has no ground truth and no score. Processing time per frame is in `docs/RESULTS.md`; the release run measured 597 ms
and 752 ms per output tick (total wall time over 722 ticks) on a shared RTX 4090. The shipped test run has no ground truth, so
there is no scorer output for it; the table above comes from the research pipeline.

## Licences and declared pre-trained models

| Item | Use | Licence |
|---|---|---|
| LightGlue (matcher) and its ALIKED extractor code | local feature matching | Apache-2.0; ALIKED part BSD-3-Clause |
| ALIKED-n16 checkpoint | keypoints | BSD-3-Clause |
| LightGlue-ALIKED checkpoint | matching | Apache-2.0 |
| DINOv2 ViT-S/14 (facebook/dinov2-small) | global retrieval descriptor | Apache-2.0 |
| ResNet-18 encoder (torchvision) | initialization of the border network | BSD-3-Clause code; ImageNet-trained weights, ImageNet terms |
| Track-border segmentation network | border fits | trained by us on laps 0, 1, 3, 5 from the ResNet-18 encoder above (torchvision IMAGENET1K_V1; ImageNet terms, research use); additional training frames were generated offline with Stable Diffusion 1.5 + ControlNet-canny (not shipped; generator licences: CreativeML OpenRAIL-M, OpenRAIL) |
| Landmark maps, pole map, attitude map | localization | derived from the supplied recording; organizer data terms apply |

We trained on no external dataset. Pre-trained components (ResNet-18 encoder on ImageNet, LightGlue on MegaDepth/Oxford-Paris, DINOv2 on LVD-142M, ALIKED) were trained by their authors on external data; the table lists them. Raw organizer recordings and ground truth are not
part of this repository. Python packages are recreated from the lockfile and not redistributed;
the full table is `licenses/license-table.md`; additional declared items are in `licenses/ADDITIONS.md`.

## Repository layout

```
pixi.toml, pixi.lock     one pinned environment (task camloc-run)
src/camloc/              runtime library
scripts/                 frozen driver (final_run.py) and its script closure; export CLIs
configs/final/           frozen configurations final-v2.yaml (default) and final-v1.yaml (asset paths prefixed $ASSETS/)
configs/v2/s122/         E-matcher parameters read by final-v2
configs/models, splits/  model parameters; empty lap table for organizer data
schemas/                 data schemas used by export and validation
external/S04-lightglue/  vendored LightGlue subset (licence files included)
tools/                   asset build/verify, source export, dependency probe
manifests/               asset manifest, freeze manifest, source-export manifest
licenses/                licence table, ADDITIONS.md, external-resources.json
docs/                    ASSETS.md, RESULTS.md, ADAPTATIONS notes, RUNTIME_FOUNDATION.md
tests/                   45 tests
assets/, dist/, runs/    generated locally (assets bundle, tarball, run outputs); not in Git
```

## Provenance

The model is research decision D-086 (mc7 with borders, poles and speed-aware v_max, no
learned layer; D-074 and D-083 give its parts), frozen as `configs/final/final-v1.yaml` by decision D-091
(SHA-256 `eaeedd442f9544cbbcda45097bb454aa17bdf1047edc8f0b3a34c6b65b7c1db2`). Maps, networks
and checkpoints are fitted on laps 0, 1, 3 and 5 only. The research checkout is not part of
this repository. Port details and deviations from the research driver: `docs/R02B_ADAPTATIONS.md`.
