"""Inference-clean paved-surface probability for one tick of one camera image.

Inputs are image pixels only: no GT, no pose, no labels, no map. This module imports nothing from `labels`, `build`,
`calib_hook`, `camloc.geometry`, `camloc.mapping` (a test enforces it).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from . import LABEL_H, LABEL_W
from .config import TrainConfig
from .model import build_model


def to_model_input(rgb: np.ndarray, width: int = LABEL_W, height: int = LABEL_H) -> np.ndarray:
    """RGB uint8 image of any size -> RGB uint8 (height, width, 3), area-resampled (the training pipeline's resize)."""
    if rgb.shape[:2] == (height, width):
        return rgb
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def decode_jpeg(jpeg: bytes, width: int = LABEL_W, height: int = LABEL_H) -> np.ndarray:
    """JPEG bytes (full resolution) -> RGB uint8 model input, using the same reduced decode + area resize as the cache."""
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
    return to_model_input(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), width, height)


class SegPredictor:
    def __init__(self, model: torch.nn.Module, device="cuda", width: int = LABEL_W, height: int = LABEL_H, amp: bool = True):
        self.model = model.eval().to(device)
        self.device, self.width, self.height, self.amp = device, width, height, amp and str(device).startswith("cuda")

    @classmethod
    def from_checkpoint(cls, path, device="cuda") -> "SegPredictor":
        ck = torch.load(path, map_location="cpu")
        cfg = TrainConfig.from_dict(ck["config"])
        m = build_model(cfg.model, "")
        m.load_state_dict(ck["model"])
        return cls(m, device, cfg.width, cfg.height, cfg.amp)

    @torch.no_grad()
    def predict_prob(self, rgb: np.ndarray, out_size=None) -> np.ndarray:
        """Per-pixel probability of "paved surface inside the map borders", float32 (height, width) in [0, 1]
        (or `out_size` = (W, H) when given). `rgb` is one uint8 RGB image of any size."""
        x = torch.from_numpy(to_model_input(rgb, self.width, self.height)).permute(2, 0, 1)[None].float().to(self.device)
        with torch.autocast("cuda", enabled=self.amp):
            p = torch.sigmoid(self.model(x).float())[0, 0]
        p = p.cpu().numpy().astype(np.float32)
        if out_size is not None:
            p = cv2.resize(p, tuple(out_size), interpolation=cv2.INTER_LINEAR)
        return p

    def predict_prob_jpeg(self, jpeg: bytes, out_size=None) -> np.ndarray:
        return self.predict_prob(decode_jpeg(jpeg, self.width, self.height), out_size)
