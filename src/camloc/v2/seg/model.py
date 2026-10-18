"""Small segmentation models for the paved-surface mask. Input: uint8-range RGB float tensor (B, 3, H, W) in 0..255 (the
wrapper normalises); output: logits (B, 1, H, W) at input resolution.

  unet_r18    ResNet-18 encoder (optionally ImageNet-pretrained from a local state_dict) + light U-Net decoder, ~14 M parameters.
  segformer_b0  SegFormer-B0 architecture (transformers, random init; the NVIDIA MiT-b0 weights are under a non-commercial
              licence and were deliberately NOT acquired, see the S1.4 report).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class _Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255.0)

    def forward(self, x):
        return (x - self.mean) / self.std


def _conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class _Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.c1 = _conv(cin + cskip, cout)
        self.c2 = _conv(cout, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.c2(self.c1(torch.cat([x, skip], 1)))


class UNetR18(nn.Module):
    def __init__(self, pretrained: str = ""):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet18(weights=None)
        if pretrained:
            sd = torch.load(pretrained, map_location="cpu")
            r.load_state_dict(sd)
        self.norm = _Norm()
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)      # /2, 64
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4   # /4 64, /8 128, /16 256, /32 512
        self.u3 = _Up(512, 256, 128)
        self.u2 = _Up(128, 128, 64)
        self.u1 = _Up(64, 64, 48)
        self.u0 = _Up(48, 64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        x = self.norm(x)
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0))
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        s4 = self.l4(s3)
        y = self.u3(s4, s3)
        y = self.u2(y, s2)
        y = self.u1(y, s1)
        y = self.u0(y, s0)
        y = self.head(y)
        return F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)


class SegFormerB0(nn.Module):
    def __init__(self, pretrained: str = ""):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        cfg = SegformerConfig(num_labels=1, depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256], num_attention_heads=[1, 2, 5, 8],
                              decoder_hidden_size=256, sr_ratios=[8, 4, 2, 1])
        self.m = SegformerForSemanticSegmentation(cfg)
        if pretrained:
            self.m.load_state_dict(torch.load(pretrained, map_location="cpu"))
        self.norm = _Norm()

    def forward(self, x):
        H, W = x.shape[-2:]
        y = self.m(pixel_values=self.norm(x)).logits
        return F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)


def build_model(name: str, pretrained: str = "") -> nn.Module:
    if name == "unet_r18":
        return UNetR18(pretrained)
    if name == "segformer_b0":
        return SegFormerB0(pretrained)
    raise ValueError(f"unknown model {name!r}")


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
