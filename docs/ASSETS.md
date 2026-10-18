# Asset bundle `final-v1`

The frozen model needs 76 files (about 913 MB): reference maps for 7 cameras, the pole map, road-attitude map, body masks, the border-segmentation
checkpoint, ALIKED and LightGlue weights, DINOv2-small, the reference calibration, two constants files, the track bounds and a few small parameter files.
Everything is listed with size, sha256, original research path and licence pointer in `manifests/assets-final-v1.json`; it is **not** in Git.

| Item | Where |
|---|---|
| Tarball | `dist/camloc-aramis-assets-final-v1.tar` (one part, 913,162,240 B, sha256 in `manifests/assets-final-v1.json` -> `tar`) |
| Download URL | placeholder: filled in at release time (R04) |
| Layout | unpacks to `assets-final-v1/` (mirrors the research layout: `artifacts/`, `manifests/v2/`, `configs/models/`, `reports/experiments/`, `derived/`, `assets/final/`) |

```bash
# 1. fetch dist/camloc-aramis-assets-final-v1.tar (URL: see release page), then
mkdir -p assets && tar -xf camloc-aramis-assets-final-v1.tar -C assets && mv assets/assets-final-v1 assets/final-v1
# 2. verify (hashes of every file, equality with the research freeze manifest, no extra files); or verify the tarball directly:
pixi run python tools/verify_assets.py --assets assets/final-v1
pixi run python tools/verify_assets.py --tar camloc-aramis-assets-final-v1.tar --work /tmp/unpack-check
```

`scripts/final_run.py run --assets <dir>` (or env `CAMLOC_ASSETS`) points the driver at any unpacked copy; the default is `assets/final-v1`.

Rebuilding from the research checkout (needs it; not for organizers): `python tools/build_assets.py --source-root <research> --dest assets/final-v1 --tar dist/camloc-aramis-assets-final-v1.tar`.
It refuses on any hash/digest mismatch with the research freeze (`manifests/final-v1-freeze.json`).

Files outside the research freeze manifest (flagged `in_freeze_manifest: false`): the LightGlue matcher checkpoint `aliked_lightglue_v0.1_arxiv.pth` (loaded at run time but not
hashed by the freeze), `configs/models/e09k2-y1.yaml`, two tuned-parameter JSONs under `reports/experiments/`, and a copy of the track bounds at `derived/track_bounds.json`.
