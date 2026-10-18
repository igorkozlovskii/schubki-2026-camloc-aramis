"""S1.2 landmark-fix probe: the accepted E04 (retrieval -> ALIKED/LightGlue -> PnP) and E03 (retrieval pose) pipelines run on ONE
camera at a fixed tick stride, plus the pure converter from a run's diagnostics to `meas-v1` stream rows.

MEASUREMENT PATH: imports no ground-truth module (no camloc.mapping.keyframes / triangulate_map, no derived/, no evaluators).
Maps are training-only artifacts built offline by scripts/v2_s12_build_map.py. Telemetry: none. `online_mapping: disabled`.
The E04/E03 classes are imported unchanged; this module only (1) re-points a camera's frame to the pipeline's clock slot, as
`camloc.models.pnp_multi` does for fr, and (2) skips ticks to implement the stride.

`init`: the first pose of the episode is consumed once (runner contract) and later priors are the pipeline's own accepted fixes;
no GT-perturbed or filter-supplied pose enters, so streams are written with init = "none".
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from camloc.contracts import Capabilities, ContractError, FirstPose, FrameBundle, StaticMetadata, StepResult
from camloc.models.pnp import DEFAULTS, E04PnpModel
from camloc.models.retrieval import ROOT, E03RetrievalModel, GlobalDescriptor, LocalFeatures

PIPELINES = ("e04", "e03")
SKIP_FLAG = "stride_skip"
MEAS_V1_COLUMNS = ["stamp_ns", "avail_ns", "lap", "camera", "kind", "valid", "s", "n", "dpsi", "x", "y", "yaw",
                   "sig_s", "sig_n", "sig_dpsi", "sig_xy", "sig_yaw", "init"]


class CamProbeModel:
    """One camera `camera`, pipeline `e04` (PnP fix) or `e03` (retrieval pose, fl), every `stride`-th tick of the lap."""

    def __init__(self, config: dict | None = None, assets: dict | None = None):
        p = {"camera": "fl", "stride": 2, "pipeline": "e04", **dict(config or {})}
        a = dict(assets or {})
        if p["pipeline"] not in PIPELINES:
            raise ContractError(f"unknown pipeline {p['pipeline']!r}")
        if int(p["stride"]) < 1:
            raise ContractError("stride must be >= 1")
        self.p, self.cam, self.stride = p, p["camera"], int(p["stride"])
        base = {k: v for k, v in p.items() if k in DEFAULTS}
        if p["pipeline"] == "e04":
            self.inner = E04PnpModel(base, {"map_dir": a["map_dir"], "ext_dir": a.get("ext_dir")})
            man = self.inner.map.manifest
            crop = tuple(man.get("image_crop") or (12, 0, 1494, 540))
            ext = a.get("ext_dir")
            # per-camera crop recorded in the map manifest (same pattern as pnp_multi for fr)
            self.inner.fe = LocalFeatures(ext, max_kp=p.get("max_kp", DEFAULTS["max_kp"]), det_thr=p.get("det_thr", DEFAULTS["det_thr"]), crop=crop)
            self.inner.gd = GlobalDescriptor(ext, pool=man["global_descriptor"]["pool"], crop=crop)
        else:
            if self.cam != "fl":
                raise ContractError("the coarse (E03) pipeline is defined for fl only")
            self.inner = E03RetrievalModel(base, {"map_dir": a["map_dir"], "ext_dir": a.get("ext_dir")})
        man = self.inner.map.manifest
        if man.get("camera", "fl") != self.cam:
            raise ContractError(f"map camera {man.get('camera')!r} != probe camera {self.cam!r}")
        self.capabilities = Capabilities(cameras=(self.cam,), clock_camera="fl", calibration=True, map_id=self.inner.map.map_id, telemetry_fields=(),
                                         device="cuda", assets=(str(a["map_dir"]),), scale_status="metric", online_mapping="disabled")
        self.i = 0
        self.n_init = 0

    def initialize(self, first_pose: FirstPose, static_metadata: StaticMetadata) -> None:
        self.n_init += 1
        if self.n_init > 1:
            raise ContractError("probe initialized twice")
        cams = dict(static_metadata.cameras)
        if self.cam not in cams:
            raise ContractError(f"camera {self.cam} missing in the static metadata")
        self.inner.initialize(first_pose, dataclasses.replace(static_metadata, cameras={"fl": cams[self.cam]}))

    def step(self, bundle: FrameBundle) -> StepResult:
        i = self.i
        self.i += 1
        pr = self.inner.prior
        if i % self.stride:
            return StepResult(pr.x, pr.y, pr.yaw, source_flags=(SKIP_FLAG,), tracking_state="held", residuals={"reason": SKIP_FLAG})
        frame = bundle.frames.get(self.cam)
        frames = {} if frame is None else {"fl": frame}
        res = self.inner.step(dataclasses.replace(bundle, frames=frames))
        return dataclasses.replace(res, residuals={**res.residuals, "cam": self.cam})

    def close(self) -> None:
        self.inner.close()


# ------------------------------------------------------------------------------------------------ diagnostics -> meas-v1
def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def rows_from_run(diag_rows: list[dict], pred: "pd.DataFrame", camera: str, kind: str, lap: int, centerline, kf_xy: np.ndarray | None,
                  kf_ids: np.ndarray | None, latency_scale: float = 1.0):
    """Convert a run's `step` diagnostics (+ predictions.csv DataFrame timestamp_ns,x,y,yaw) into meas-v1 rows (list of dicts).

    kind 'landmark': valid = the E04 PnP fix passed the gates; x, y, yaw only for valid rows (the gate-rejected PnP pose and the labeled
    retrieval-fallback pose are kept in q_rej_* / q_fb_*). kind 'coarse': the retrieval pose, valid when a candidate was found.
    Rows exist only for attempted (non-skipped) ticks. No ground truth is used. Track coordinates: `track-v1` centerline from the
    supplied borders; the s hint is the last valid measurement of this stream within 3 s.
    """
    out = []
    s_hint, t_hint = None, None
    kf_rank_of = None if kf_ids is None else {int(k): j for j, k in enumerate(kf_ids)}
    for d in diag_rows:
        if SKIP_FLAG in d["flags"]:
            continue
        t, r = int(d["t"]), d["residuals"]
        flags = d["flags"]
        i = int(d["i"])
        assert int(pred.timestamp_ns.iloc[i]) == t, "predictions.csv row order differs from diagnostics"
        px, py, pyaw = float(pred.x.iloc[i]), float(pred.y.iloc[i]), float(pred.yaw.iloc[i])
        service_ns = int(d["latency_ns"])
        row = {"stamp_ns": t, "avail_ns": t + int(round(service_ns * latency_scale)), "lap": lap, "camera": camera, "kind": kind,
               "valid": False, "s": np.nan, "n": np.nan, "dpsi": np.nan, "x": np.nan, "y": np.nan, "yaw": np.nan,
               "sig_s": np.nan, "sig_n": np.nan, "sig_dpsi": np.nan, "sig_xy": np.nan, "sig_yaw": np.nan, "init": "none",
               "q_reason": r.get("reason", ""), "q_flags": "|".join(flags), "q_mode": r.get("mode", ""), "q_service_ms": service_ns / 1e6,
               "q_cam_age_ms": (d["ages_ns"].get(camera) or 0) / 1e6 if d.get("ages_ns") else np.nan, "q_status": (d.get("status") or {}).get(camera, "")}
        if kind == "landmark":
            fix = "pnp_fix" in flags
            row.update({"valid": bool(fix), "q_n_cand": r.get("n_cand", np.nan), "q_sim_top": r.get("sim_top", np.nan),
                        "q_n_corr": r.get("n_corr", np.nan), "q_n_inl": r.get("n_inl", np.nan), "q_inlier_ratio": r.get("inlier_ratio", np.nan),
                        "q_rms_px": r.get("rms_px", np.nan), "q_d_kf_m": r.get("d_kf_m", np.nan), "q_z_dev_m": r.get("z_dev_m", np.nan),
                        "q_yaw_dev_deg": r.get("yaw_dev_deg", np.nan), "q_n_kf_inl": r.get("n_kf_inl", np.nan),
                        "q_dt_since_fix_s": r.get("dt_since_fix_s", np.nan), "q_vram_peak_mb": r.get("vram_peak_mb", np.nan),
                        "q_rej_x": np.nan, "q_rej_y": np.nan, "q_rej_yaw": np.nan, "q_fb_x": np.nan, "q_fb_y": np.nan, "q_fb_yaw": np.nan,
                        "q_rank_support": np.nan})
            for k, v in (r.get("t_ms") or {}).items():
                row[f"q_t_{k}_ms"] = v
            if "rejected_pose" in r:
                row["q_rej_x"], row["q_rej_y"], row["q_rej_yaw"] = r["rejected_pose"]
            if "retrieval_fallback" in flags:
                row["q_fb_x"], row["q_fb_y"], row["q_fb_yaw"] = px, py, pyaw
            pose = (px, py, pyaw) if fix else (row["q_rej_x"], row["q_rej_y"], row["q_rej_yaw"])
            if kf_xy is not None and r.get("kf_ids") and np.isfinite(pose[0]):
                ids = r["kf_ids"]                                  # candidate kf_ids in retrieval-score order
                dd = [math.hypot(kf_xy[kf_rank_of[k], 0] - pose[0], kf_xy[kf_rank_of[k], 1] - pose[1]) for k in ids]
                row["q_rank_support"] = int(np.argmin(dd))         # rank of the candidate keyframe nearest to the PnP pose
        else:
            fix = bool({"retrieval_top1"} & set(flags))
            row.update({"valid": bool(fix), "q_sim_top": r.get("sim", np.nan), "q_kf_id": r.get("kf_id", -1)})
        if row["valid"]:
            row["x"], row["y"], row["yaw"] = px, py, pyaw
            hint = s_hint if (s_hint is not None and t - t_hint < 3e9) else None
            s, n, th, _ = centerline.project(px, py, s_hint=hint)
            row["s"], row["n"], row["dpsi"] = float(s), float(n), float(_wrap(pyaw - th))
            s_hint, t_hint = s, t
        out.append(row)
    return out


# ------------------------------------------------------------------------------------------------ schema validation
CAMERAS = ("fl", "fr", "cl", "cr", "r", "rl", "rr", "multi")
KINDS = ("border", "landmark", "coarse", "vo")
INITS = ("gt_perturbed", "pred_stream", "none")


def validate_meas_v1(df) -> list[str]:
    """Schema/consistency check of a meas-v1 stream DataFrame; returns a list of problems (empty = ok)."""
    bad = []
    miss = [c for c in MEAS_V1_COLUMNS if c not in df.columns]
    if miss:
        return [f"missing columns {miss}"]
    for c in ("stamp_ns", "avail_ns"):
        if str(df[c].dtype) != "int64":
            bad.append(f"{c} dtype {df[c].dtype} != int64")
    if (df.avail_ns < df.stamp_ns).any():
        bad.append("avail_ns < stamp_ns")
    if not df.stamp_ns.is_monotonic_increasing or df.stamp_ns.duplicated().any():
        bad.append("stamp_ns not strictly increasing")
    if str(df["valid"].dtype) != "bool":
        bad.append("valid is not bool")
    for c, allowed in (("camera", CAMERAS), ("kind", KINDS), ("init", INITS)):
        if not df[c].isin(allowed).all():
            bad.append(f"{c} outside {allowed}")
    pose = ["s", "n", "dpsi", "x", "y", "yaw"]
    v = df[df.valid.astype(bool)]
    nv = df[~df.valid.astype(bool)]
    if not np.isfinite(v[pose].to_numpy(float)).all():
        bad.append("valid row with non-finite pose/track coordinates")
    if nv[pose].notna().any().any():
        bad.append("invalid row carries a measurement (must be NaN)")
    return bad
