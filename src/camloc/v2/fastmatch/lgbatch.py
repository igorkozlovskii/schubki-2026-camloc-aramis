"""Batched LightGlue with per-sample padding masks (S1.22).

The stock `LightGlue.forward` prunes points with `torch.where(mask)[1]`, which is only valid for batch size 1, and it cannot pad variable-length
map keyframes. This module re-runs the same layers (weights unchanged) on a padded batch: every pair has the same query length M and a keyframe
length padded to N with a validity mask. Point pruning is off (it changes the result of a pair and is not batchable); early stopping is optional and
evaluated over the whole batch (ratio of confident valid tokens). With depth_confidence <= 0 and a net built with width_confidence=-1 the matches equal
`LightGlue.forward` pair by pair (tests/v2/s122/test_lgbatch.py).
"""
from __future__ import annotations

import torch

from lightglue.lightglue import filter_matches, normalize_keypoints, sigmoid_log_double_softmax


class BatchedLG:
    def __init__(self, net, image_size=(1506, 728), depth_confidence: float = -1.0, n_layers: int | None = None, mp: bool = False):
        self.net = net
        self.size = image_size
        self.depth = float(depth_confidence)
        self.n_layers = int(n_layers or net.conf.n_layers)
        self.mp = mp
        self.last_stop = None

    @torch.no_grad()
    def match(self, k0, d0, k1, d1, valid1, valid0=None):
        """k0 [B,M,2], d0 [B,M,D] (queries); k1 [B,N,2], d1 [B,N,D] (keyframes, zero padded), valid1 [B,N] bool. -> (m0 [B,M] long: index into the keyframe or -1, score0 [B,M])."""
        net = self.net
        b, m, _ = k0.shape
        n = k1.shape[1]
        dev = k0.device
        if valid0 is None:
            valid0 = torch.ones(b, m, dtype=torch.bool, device=dev)
        size = torch.tensor([self.size], device=dev, dtype=torch.float32).expand(b, 2)
        with torch.autocast(enabled=self.mp, device_type="cuda"):
            kp0 = normalize_keypoints(k0, size).clone()
            kp1 = normalize_keypoints(k1, size).clone()
            if self.mp:
                d0, d1 = d0.half(), d1.half()
            x0, x1 = net.input_proj(d0), net.input_proj(d1)
            e0, e1 = net.posenc(kp0), net.posenc(kp1)
            mk0, mk1 = valid0[:, None, :, None], valid1[:, None, :, None]
            nv = (valid0.sum() + valid1.sum()).float()
            L = min(self.n_layers, net.conf.n_layers)
            used = L
            for i in range(L):
                x0, x1 = net.transformers[i](x0, x1, e0, e1, mask0=mk0, mask1=mk1)
                if i == L - 1:
                    continue
                if self.depth > 0:
                    t0, t1 = net.token_confidence[i](x0, x1)
                    thr = net.confidence_thresholds[i]
                    unconf = (((t0 < thr) & valid0).sum() + ((t1 < thr) & valid1).sum()).float()
                    if (1.0 - unconf / nv) > self.depth:
                        used = i + 1
                        break
            self.last_stop = used
            la = net.log_assignment[used - 1]
            mdesc0, mdesc1 = la.final_proj(x0), la.final_proj(x1)
            dd = mdesc0.shape[-1]
            mdesc0, mdesc1 = mdesc0 / dd ** 0.25, mdesc1 / dd ** 0.25
            sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
            z0, z1 = la.matchability(x0), la.matchability(x1)
            pad = ~(valid0[:, :, None] & valid1[:, None, :])
            sim = sim.masked_fill(pad, -1e4)
            scores = sigmoid_log_double_softmax(sim, z0, z1)
            scores[:, :m, :n] = scores[:, :m, :n].masked_fill(pad, float("-inf"))
            m0, _m1, ms0, _ms1 = filter_matches(scores, net.conf.filter_threshold)
        return m0, ms0
