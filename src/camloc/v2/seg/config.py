"""Training configuration (YAML-backed dataclass; stdlib + yaml only so a remote run directory can use it)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AugConfig:
    """Photometric / blur augmentation hooks (D-047: night, daylight-like relighting later through `hook`)."""
    enabled: bool = True
    gain: tuple = (0.6, 1.6)            # global brightness multiplier
    gain_p: float = 0.8
    gamma: tuple = (0.7, 1.5)
    gamma_p: float = 0.5
    contrast: tuple = (0.7, 1.3)
    contrast_p: float = 0.5
    channel_gain: tuple = (0.85, 1.15)  # per-channel white-balance jitter
    channel_gain_p: float = 0.5
    blur_sigma: tuple = (0.5, 2.5)      # gaussian blur at model resolution
    blur_p: float = 0.3
    motion_blur_len: tuple = (3, 9)     # horizontal motion blur kernel (px)
    motion_blur_p: float = 0.2
    noise_std: tuple = (0.0, 8.0)       # additive gaussian noise on 0..255
    noise_p: float = 0.3
    hflip_p: float = 0.0                # off by default: left/right cameras are different sensors


@dataclass
class TrainConfig:
    model: str = "unet_r18"             # unet_r18 | segformer_b0
    pretrained: str = ""                # path to a state_dict for the encoder ("" = random init)
    width: int = 384
    height: int = 192
    batch_size: int = 32
    epochs: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dice_weight: float = 0.5
    boundary_ignore_px: int = 2         # label-resolution band around the road boundary excluded from the loss
    amp: bool = True
    num_workers: int = 8
    seed: int = 0
    train_laps: tuple = (0, 1, 3, 5)
    holdout_lap: int = -1               # -1: train on all four laps (no validation lap)
    cameras: tuple = ("fl", "fr", "cl", "cr", "rl", "rr", "r")
    train_stride: int = 1               # keep every n-th frame of the index (per camera and lap)
    val_stride: int = 4
    max_train_frames: int = 0           # 0 = all
    max_steps: int = 0                  # 0 = epochs * steps per epoch
    deterministic: bool = False         # cudnn deterministic (slower)
    aug: AugConfig = field(default_factory=AugConfig)

    @staticmethod
    def from_yaml(path) -> "TrainConfig":
        import yaml
        d = yaml.safe_load(Path(path).read_text()) or {}
        return TrainConfig.from_dict(d)

    @staticmethod
    def from_dict(d: dict) -> "TrainConfig":
        d = dict(d)
        aug = AugConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in (d.pop("aug", {}) or {}).items()})
        d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
        return TrainConfig(aug=aug, **d)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
