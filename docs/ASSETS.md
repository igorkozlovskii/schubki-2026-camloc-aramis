# Asset bundle `final-v1` (also used by final-v2)

final-v2 (release v2.0.0) adds no asset: `manifests/final-v2-freeze.json` hashes the same maps, networks, constants and assets as `manifests/final-v1-freeze.json` (checked equal), and the verifier passes against either (`tools/verify_assets.py --freeze manifests/final-v2-freeze.json`). The v1 tarball is reused unchanged. The only extra files final-v2 needs are code and the small E-parameter file `configs/v2/s122/generated/E-lap1.yaml`, which are in the repository.

The frozen model needs 76 files (about 913 MB): reference maps for 7 cameras, the pole map, road-attitude map, body masks, the border-segmentation
checkpoint, ALIKED and LightGlue weights, DINOv2-small, the reference calibration, two constants files, the track bounds and a few small parameter files.
Everything is listed with size, sha256, original research path and licence pointer in `manifests/assets-final-v1.json`; it is **not** in Git.

| Item | Where |
|---|---|
| Tarball | `dist/camloc-aramis-assets-final-v1.tar` (one part, 913,162,240 B, sha256 in `manifests/assets-final-v1.json` -> `tar`) |
| Download URL | <https://github.com/igorkozlovskii/schubki-2026-camloc-aramis/releases/tag/v1.0.0> (`gh release download v1.0.0 -R igorkozlovskii/schubki-2026-camloc-aramis`); tarball SHA-256 `2f3fffc60b5dcecd211f55912c0779df6fa9c1c4c47c80e16f434199f6d981f9` |
| Layout | unpacks to `assets-final-v1/` (mirrors the research layout: `artifacts/`, `manifests/v2/`, `configs/models/`, `reports/experiments/`, `derived/`, `assets/final/`) |

```bash
# 1. fetch dist/camloc-aramis-assets-final-v1.tar (URL above), then
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
