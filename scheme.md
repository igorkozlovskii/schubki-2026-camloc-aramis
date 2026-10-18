# Map building → Inference · final-v2

[scheme.drawio](scheme.drawio): map building at the top, inference below. Dashed arrows connect stored map data to its consumers; side notes explain keyframes, preprocessing and the E matcher.

## Keyframe

A **keyframe** is a selected training image with a known camera pose, representing one view of the track. Its embedding and local features are stored in the map. Selection uses elapsed time, travelled distance and training-pose quality.

## Map building and preparation

Training laps **0, 1, 3, 5** → keyframes, separately for each camera:

- Camera-specific crop → DINOv2 → global embeddings (`global.npy`).
- Crop → ALIKED → LightGlue between training frames → geometric checks and triangulation using known poses → local keypoints, descriptors and 3D landmarks (`local.npz`).
- Keyframe IDs and training poses → `keyframes.csv`.

DINOv2 resizes and normalizes the crop; ALIKED keypoints are restored to full-image pixel coordinates. Camera calibration supplies the geometry.

**Final-v2 reuses the final-v1 maps and asset bundle.** At startup, embeddings and keyframe positions form a retrieval index. In the E matcher, selected keyframes' features are uploaded to a GPU cache on first use and reused; 3D coordinates remain linked to those features. The frozen configuration uses `mode: serial`, so the optional padded, batched map layout is not built.

## Inference

The driver runs two paths:

- **fl:** the original E04 probe, every tick (~10 Hz).
- **fr, cl, cr, r, rl, rr:** per-camera fixes from one E-matcher run, every other tick (~5 Hz).

Both follow the same geometry:

1. Decode and crop each new frame.
2. DINOv2 query embedding + that camera's map index → up to six candidate keyframes, restricted by its own pose prior.
3. ALIKED query features + cached keyframe features → LightGlue matches → 2D query points linked to 3D landmarks.
4. PnP + quality gates → accepted vehicle pose.

The E matcher shares networks across cameras but runs DINOv2, ALIKED and LightGlue **serially**. It visits candidate keyframes in retrieval order and stops matching once at least **400 valid 2D–3D correspondences** have accumulated, or candidates are exhausted. PnP solves run in a thread pool (`pnp_threads: 8`). The model also computes an internal fl result; fusion uses only its six non-fl outputs and keeps the separate E04 fl stream.

Each camera has its own map and prior: the initial pose, then its last accepted PnP fix. Retrieval becomes global after 5 s without a fix or when the prior window has no candidates.

Telemetry → motion prediction. Accepted visual poses + border and pole corrections + prediction → EKF → final `x, y, yaw` at 10 Hz. The fl retrieval fallback can provide a weak correction; other cameras contribute only accepted PnP fixes. Border fitting and pole association use the current filter estimate and their own maps.

Sources: [release driver](scripts/final_run.py), [final-v2 config](configs/final/final-v2.yaml), [E parameters](configs/v2/s122/generated/E-lap1.yaml), [E matcher and cache](src/camloc/v2/fastmatch/model.py), [keyframe selection](src/camloc/mapping/keyframes.py), [map format](src/camloc/mapping/map.py).
