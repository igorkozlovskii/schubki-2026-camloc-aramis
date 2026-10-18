"""Shared harness of the S1.8 scripts (GT side: evaluators and offline builders only). Builds on the S1.1 harness (read-only import)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_s11_common as H  # noqa: E402
from camloc.v2 import border_attitude as BA  # noqa: E402

ROOT = H.ROOT
TELDIR = ROOT / "artifacts/telemetry/S10a-laps0-5"
ATT_CFG = ROOT / "configs/v2/s18/attitude_{fold}.yaml"
CAMERAS = H.CAMERAS
TRAIN = H.TRAIN


def load_telemetry():
    """(stamps, cols, causal features) of the train-lap telemetry file (allowed fields only)."""
    tel = pd.read_csv(TELDIR / "telemetry.csv", usecols=["stamp_ns"] + list(BA.FIELDS))
    st = tel.stamp_ns.to_numpy()
    cols = {k: tel[k].to_numpy(float) for k in BA.FIELDS}
    return st, cols, BA.telemetry_features(st, cols)


def feature_at(st, F, t_ns):
    j = int(np.searchsorted(st, np.int64(t_ns), side="right")) - 1
    return F[j] if j >= 0 else None


def gt_attitude(g):
    return Attitude_abs(g["z"], g["pitch"], g["roll"])


def Attitude_abs(z=None, pitch=None, roll=None):
    return BA.Attitude(z=z, pitch=pitch, roll=roll)


def effect_size(h, ctx, cam, g, which="gt"):
    """Oracle diagnostic at the GT pose: how far the border projected with S1.1's attitude must be moved (dn along the centreline
    normal, dpsi) to coincide with the border projected with the GT attitude. Returns dict (dn_only_m, dn_joint_m, dpsi_joint_deg,
    rms_px_before = rms normal-direction displacement, n_pts) or None."""
    from camloc.v2 import border_probe as bp
    sc = bp._Scene(ctx, cam, (g["x"], g["y"], g["yaw"]), g["s"])
    if len(sc.runs) == 0:
        return None
    cfg = ctx.cfg
    usable = ctx.masks[cam]
    P = sc.runs.xyz
    z, pit, rol = sc.z_base, sc.pitch0, sc.roll0          # S1.1 values; `which` swaps components for the GT ones
    if which in ("gt", "gtz"):
        z = g["z"]
    if which in ("gt", "gtpitch"):
        pit = g["pitch"]
    if which in ("gt", "gtroll"):
        rol = g["roll"]
    uvg, okg, dg, _ = sc.cm.project_points_pose(P, g["x"], g["y"], g["yaw"], z, pit, rol, with_jac=False)
    uv0, ok0, d0, _ = sc.project(P, 0.0, 0.0, 0.0)
    ok = okg & ok0 & (dg > cfg.min_depth_m) & (dg < cfg.max_depth_m)
    ok = bp._usable_at(uvg, usable, ok)
    if ok.sum() < cfg.min_points:
        return None
    Pk = P[ok]
    tgt = uvg[ok]
    from camloc.geometry.residuals import image_tangents
    tau, tok = image_tangents(tgt, sc.runs.run_id[ok])
    Pk, tgt, tau = Pk[tok], tgt[tok], tau[tok]
    if len(Pk) < cfg.min_points:
        return None
    nrm = np.stack([-tau[:, 1], tau[:, 0]], 1)          # image normal of the GT-attitude border: the only direction a border measurement sees

    def res(v, joint):
        dn, dp = (v[0], v[1]) if joint else (v[0], 0.0)
        uv, o, _, _ = sc.project(Pk, dn, dp, 0.0)
        return np.where(o, ((uv - tgt) * nrm).sum(1), 0.0)

    out = {"n_pts": int(ok.sum()), "rms_px_before": float(np.sqrt(np.mean(res(np.zeros(2), True) ** 2)))}
    for joint in (False, True):
        v = np.zeros(2 if joint else 1)
        for _ in range(4):
            r0 = res(v, joint)
            eps = np.array([1e-3, 1e-4])[:len(v)]
            J = np.stack([(res(v + np.eye(len(v))[i] * eps[i], joint) - r0) / eps[i] for i in range(len(v))], 1)
            v = v - np.linalg.lstsq(J, r0, rcond=None)[0]
            v = np.clip(v, [-6.0, -0.2][:len(v)], [6.0, 0.2][:len(v)])
        if joint:
            out["dn_joint_m"], out["dpsi_joint_deg"] = float(v[0]), float(math.degrees(v[1]))
            out["rms_px_after_joint"] = float(np.sqrt(np.mean(res(v, True) ** 2)))
        else:
            out["dn_only_m"] = float(v[0])
            out["rms_px_after_dn"] = float(np.sqrt(np.mean(res(v, False) ** 2)))
    return out
