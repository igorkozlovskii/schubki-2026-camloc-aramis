# Licence table (release-rfb-v1, S08a, revised S08c)

Generated from `external-resources.json`. Items marked NO are not part of the release.

| Item | Version | In release | Licence | Redistribution |
|---|---|---|---|---|
| LightGlue (matcher) + bundled ALIKED extractor code | git eb42fee2d71449efb0aa5c10549752b5d75384d8 | yes | Apache-2.0 (LightGlue); lightglue/aliked.py BSD-3-Clause header (ALIKED, Zhao Xiaoming) | inside the bundle (permissive licences); lightglue/superpoint.py is NOT redistributed (res |
| ALIKED-n16 checkpoint aliked-n16.pth | ALIKED repo HEAD 683d7c65197395c0b3f01ebe76e1084a27e73a65 (m | yes | BSD-3-Clause (repository LICENSE; no separate weights licence file) | inside the bundle; alternative: fetch from the URL and verify the sha256 |
| LightGlue ALIKED matcher checkpoint aliked_lightglue.pth | release tag v0.1_arxiv | yes | Apache-2.0 (LightGlue README: pre-trained weights Apache-2.0) | inside the bundle; alternative: fetch and verify |
| DINOv2 ViT-S/14 (facebook/dinov2-small), config + preprocessor + model.safetensors | HF revision ed25f3a31f01632728cabb09d1542f84ab7b0056 | yes | Apache-2.0 (HF model card; facebookresearch/dinov2) | inside the bundle; alternative: fetch the pinned revision |
| Reference map S04-full-v1 (training laps 0,1,3,5; 1086 keyframes, 259,266 landmarks) | map.json code_hash 9022b765...fb30; map id S04-full-v1 | yes | own derived data from the supplied recording (organizer data terms apply) | inside the bundle |
| Camera calibration calib.json (verbatim copy of the recording's) | sha256-pinned | yes | organizer-supplied recording metadata | inside the bundle (reference copy; the run uses the input store's copy and refuses a misma |
| python | 3.11.6 | yes | Python-2.0 | not redistributed: recreated from the lockfile (bundle carries environment/pixi.toml and p |
| pytorch | 2.3.1 (py3.11_cuda12.1_cudnn8.9.2_0) | yes | BSD-3-Clause | not redistributed: recreated from the lockfile (bundle carries environment/pixi.toml and p |
| pytorch-cuda | 12.1 | yes | per package metadata | not redistributed: recreated from the lockfile (bundle carries environment/pixi.toml and p |
| torchvision | 0.18.1 | yes | BSD | not redistributed: recreated from the lockfile (bundle carries environment/pixi.toml and p |
| numpy | 1.26.4 | yes | BSD-3-Clause | not redistributed: recreated from the lockfile (bundle carries environment/pixi.toml and p |
| scipy | 1.17.1 | yes | BSD-3-Clause | not redistributed: recreated from the lockfile (bundle carries environment/pixi.toml and p |
| pandas | 2.3.3 | yes | BSD-3-Clause | not redistributed: recreated from the lockfile (bundle carries environment/pixi.toml and p |
| pyyaml | 6.0.3 | yes | MIT | not redistributed: recreated from the lockfile (bundle carries environment/pixi.toml and p |
| pyarrow | 12.0.1 | yes | Apache-2.0 | not redistributed: recreated from the lockfile (bundle carries environment/pixi.toml and p |
| jsonschema | 4.26.0 | yes | MIT | not redistributed: recreated from the lockfile (bundle carries environment/pixi.toml and p |
| kornia | 0.8.3 | yes | Apache-2.0 (upstream; PyPI metadata field empty in the lock) | not redistributed: recreated from the lockfile (bundle carries environment/pixi.toml and p |
| kornia_rs | 0.2.0 | yes | Apache-2.0 (upstream; not re-verified) | not redistributed: recreated from the lockfile (bundle carries environment/pixi.toml and p |
| transformers | 4.57.6 | yes | Apache-2.0 (upstream; not re-verified) | not redistributed: recreated from the lockfile (bundle carries environment/pixi.toml and p |
| safetensors | 0.8.0 | yes | Apache-2.0 (upstream; not re-verified) | not redistributed: recreated from the lockfile (bundle carries environment/pixi.toml and p |
| tokenizers | 0.22.2 | yes | Apache-2.0 (upstream; not re-verified) | not redistributed: recreated from the lockfile (bundle carries environment/pixi.toml and p |
| huggingface_hub | 0.36.2 | yes | Apache-2.0 (upstream; not re-verified; used only as a transformers dependency, HF_HUB_OFFLINE=1) | not redistributed: recreated from the lockfile (bundle carries environment/pixi.toml and p |
| opencv-python-headless | 4.10.0.84 | yes | Apache-2.0 for OpenCV 4.x, MIT for the PyPI packaging; wheel bundles third-party libraries (e.g. FFmpeg under  | not redistributed: recreated from the lockfile (bundle carries environment/pixi.toml and p |
| environments/vision/pixi.lock (the full 287-package closure) | lock sha256 | yes | n/a | in the bundle (environment/) |
| DPVO (princeton-vl) | git 0ac95b656d1fda91c271d2a106460d19ad966fc7 | NO | MIT (code) | NOT PART OF THE RELEASE |
| DPVO checkpoint dpvo.pth (inside models.zip) | models.zip sha256 89f43de2c92676ddcf7f49e8ae3f8940b7af73f549 | NO | UNSTATED (O-007): not in the repository nor in the download; code is MIT; trained on TartanAir | NOT PART OF THE RELEASE; if a later champion needs it: find the licence or ask the organiz |
| ORB-SLAM3 | git 4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4 (HEAD at 2026-1 | NO | GPL-3.0 | NOT PART OF THE RELEASE; conveying a bundle that links it would carry GPL-3.0 obligations  |

## Scope of this table and licence of the full lock (S08c)

- The table lists **17 of the 287** locked packages individually (the direct imports plus python and pytorch-cuda), the bundled
  third-party code and weights, and the lock itself. The other 270 packages are transitive and covered only by the lockfile hash.
- The release is **recreated from the lock on the target machine, not shipped**: no package of the environment is redistributed.
- The 287-package vision environment contains GPL/LGPL components (e.g. pyqt, readline, fftw, gettext-tools; libgcc/libstdc++ with the GCC
  exception), one AGPL-3.0-only package (libdb), NVIDIA End User License Agreement components (cuda-version, libcurand, cuda-opencl) and
  Intel-proprietary mkl. Counted from the `license` fields of `pixi.lock` (258 of 287 carry one); see `full_lock_license_note` in
  `external-resources.json`. If a delivery ever ships the environment itself (conda-pack, container), a per-package review is required first.
- DPVO (weights licence unstated, O-007) and ORB-SLAM3 (GPL, not acquired) stay `part_of_release=false`.
