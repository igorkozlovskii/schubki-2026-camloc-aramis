# Additional declared resources

These rows extend `license-table.md` (ResNet-18 initialization of the border network, generative models used only to make training frames, and training-data notes for the pre-trained components). `external-resources.json` is unchanged.

| Item | Version | In release | Licence | Redistribution |
|---|---|---|---|---|
| ResNet-18 encoder weights (torchvision) | IMAGENET1K_V1, sha256 f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec | initial weights of the border network (the trained border network ships inside the asset bundle) | code BSD-3-Clause; weights trained on ImageNet, ImageNet terms (research use; accepted by Igor, PRG-001 D-009) | the pre-trained file itself is not shipped; the fine-tuned border network is |
| Stable Diffusion 1.5 (stable-diffusion-v1-5/stable-diffusion-v1-5, rev 451f4fe16113bff5a5d2269ed5ad43b0592e9a14) + ControlNet-canny v1.1p (lllyasviel/control_v11p_sd15_canny, rev 115a470d547982438f70198e353a921996e2e819) | manifests/artifacts/relight-sample-v1-20261017.json | NO (training-time only) | CreativeML OpenRAIL-M (SD 1.5); OpenRAIL (ControlNet) | frames generated offline for border-network training are not shipped |
| LightGlue / DINOv2 / ALIKED training data | n/a | yes (weights) | trained by their authors on MegaDepth/Oxford-Paris (LightGlue), LVD-142M (DINOv2); declared for completeness | as in the existing table |
