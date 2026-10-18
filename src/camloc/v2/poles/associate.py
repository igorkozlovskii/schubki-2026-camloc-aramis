"""Prediction-gated association of vertical-segment detections to map landmarks (GT-free; S2.5 stage 3/4).

Given a predicted pose (x, y, yaw) and the detections of one camera frame (base-frame azimuths), every map landmark inside the
camera's horizontal field of view and range window is projected to a predicted bearing. Gate per landmark (declared):

    |beta_meas - beta_pred| <= atan(gate_m / r) + gate0_deg            (fixed-gate mode, stage 3)
    nu^2 / (H P H^T + sig^2) <= gate_chi2                             (filter mode, stage 4; P from the filter)

Isolation: a landmark is skipped when any detection lies outside its gate but within iso_mult x gate (clutter near the prediction).
Height and polarity (optional): candidates must agree with the landmark's mid height and edge polarity (image-right minus left). Ambiguity rejection (the look-alike protection): a detection whose gate holds >= 2 landmarks is discarded; a landmark's
measurement is the median bearing of its remaining (unambiguous) detections, and it is dropped when those detections spread
by more than `max_spread_deg`. One measurement per landmark and frame.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .geometry import predict_bearing, wrap


@dataclass(frozen=True)
class AssocConfig:
    gate_m: float = 1.0
    gate0_deg: float = 0.2
    iso_mult: float = 2.5             # no other detection may lie in (gate, iso_mult x gate) of the prediction (isolation test)
    dz_gate_m: float = 2.5            # |z of the detection midpoint at the predicted range - landmark zmid| (when heights are given)
    r_min: float = 3.0
    r_max: float = 60.0
    fov_margin_deg: float = -2.0      # negative = stay inside the image border
    max_spread_deg: float = 1.0
    sig_floor_deg: float = 0.15

    def as_dict(self):
        return asdict(self)


def cam_fov(cam):
    """(optical-axis azimuth in the base frame, half horizontal FOV) in rad (pinhole approximation)."""
    z = cam.T_base_cam.R[:, 2]
    half = max(math.atan(cam.K[0, 2] / cam.K[0, 0]), math.atan((cam.width - cam.K[0, 2]) / cam.K[0, 0]))
    return math.atan2(z[1], z[0]), half


def visible(pose, cam, lm_xy, cfg: AssocConfig):
    """indices of landmarks in the FOV / range window, their predicted bearings and ranges."""
    ax, half = cam_fov(cam)
    b, r = predict_bearing(pose[0], pose[1], pose[2], cam.T_base_cam.t, lm_xy[:, 0], lm_xy[:, 1])
    ok = (r >= cfg.r_min) & (r <= cfg.r_max) & (np.abs(wrap(b - ax)) <= half + math.radians(cfg.fov_margin_deg))
    idx = np.nonzero(ok)[0]
    return idx, b[idx], r[idx]


def associate(det_az: np.ndarray, pose, cam, lm_xy: np.ndarray, cfg: AssocConfig, gate_fn=None, det_el=None, lm_z=None, cam_z=None, det_pol=None, lm_pol=None):
    """-> list of dict(lm, beta, beta_pred, r, n_det, spread_deg). `gate_fn(lm_index, r) -> half-width (rad)` overrides the fixed gate.
    Height test (optional): det_el = elevation of each detection's midpoint (rad, gravity frame), lm_z = landmark zmid (m), cam_z =
    camera height (m, map frame); a detection is a candidate only when cam_z + r tan(det_el) is within dz_gate_m of lm_z."""
    det_az = np.asarray(det_az, float)
    if len(det_az) == 0 or len(lm_xy) == 0:
        return []
    idx, bp, r = visible(pose, cam, lm_xy, cfg)
    if len(idx) == 0:
        return []
    if gate_fn is None:
        g = np.arctan(cfg.gate_m / r) + math.radians(cfg.gate0_deg)
    else:
        g = np.array([gate_fn(int(i), float(rr)) for i, rr in zip(idx, r)])
    D = np.abs(wrap(det_az[:, None] - bp[None, :]))          # (n_det, n_vis)
    inside = D <= g[None, :]
    if det_el is not None and lm_z is not None and cam_z is not None:
        zd = cam_z + r[None, :] * np.tan(np.asarray(det_el, float)[:, None])
        hok = np.abs(zd - np.asarray(lm_z, float)[idx][None, :]) <= cfg.dz_gate_m
    else:
        hok = np.ones_like(inside)
    if det_pol is not None and lm_pol is not None:
        hok = hok & (np.asarray(det_pol)[:, None] == np.asarray(lm_pol)[idx][None, :])
    annulus = (D > g[None, :]) & (D <= cfg.iso_mult * g[None, :])
    isolated = ~annulus.any(0)                                # per visible landmark
    inside &= hok
    n_in = inside.sum(1)
    out = []
    uniq = inside & (n_in[:, None] == 1)
    for j in np.nonzero(uniq.any(0) & isolated)[0]:
        dd = det_az[uniq[:, j]]
        ref = bp[j]
        rel = wrap(dd - ref)
        spread = math.degrees(float(rel.max() - rel.min())) if len(rel) > 1 else 0.0
        if spread > cfg.max_spread_deg:
            continue
        beta = float(wrap(ref + np.median(rel)))
        out.append({"lm": int(idx[j]), "beta": beta, "beta_pred": float(bp[j]), "r": float(r[j]), "n_det": int(len(dd)), "spread_deg": spread})
    return out


def bearing_sigma(lm_row, r: float, cfg: AssocConfig) -> float:
    """Declared 1-sigma of a bearing (rad): landmark ray scatter and position sigma at range r, floored."""
    s = math.hypot(math.radians(float(lm_row["sig_theta_deg"])), float(lm_row["sigma_m"]) / max(r, 1.0))
    return max(s, math.radians(cfg.sig_floor_deg))
