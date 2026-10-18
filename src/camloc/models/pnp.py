"""E04 causal absolute localization: retrieval -> ALIKED/LightGlue matching -> PnP+RANSAC against the training-only map.

Inference path: imports no ground-truth module. The map (artifacts/maps/S04-*) holds TRAINING poses/landmarks only
(provenance and lineage are in map.json). Telemetry: none. `online_mapping: disabled`.

Per tick (clock camera fl):
  1. first pose / last accepted fix defines a spatial prior whose radius grows with the time since it (`vmax`);
     after `lost_s` seconds without a fix retrieval is global (re-initialization);
  2. DINOv2-GeM retrieval, top-K keyframes inside the prior; ALIKED features of the query; LightGlue to each keyframe;
  3. 2D-3D correspondences from the keyframes' landmarks, PnP RANSAC + LM refinement on undistorted normalized points;
  4. camera pose -> base_link with the map's fitted extrinsic (including translation), gates, else labeled hold-last.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from camloc.contracts import Capabilities, ContractError, FirstPose, FrameBundle, StaticMetadata, StepResult
from camloc.mapping.map import load_map
from camloc.models.retrieval import (ROOT, GlobalDescriptor, LocalFeatures, Matcher, PriorState, RetrievalIndex, angdiff,
                                     prior_radius)

DEFAULTS = {
    "top_k": 6, "lost_s": 5.0, "prior_base_m": 40.0, "vmax_mps": 45.0, "heading_tol_deg": None,
    "max_kp": 2048, "det_thr": 0.0, "ransac_px": 3.0, "ransac_iters": 3000,
    "min_inliers": 25, "min_inlier_ratio": 0.15, "max_rms_px": 2.5, "max_kf_dist_m": 60.0, "max_z_dev_m": 4.0, "max_yaw_dev_deg": 45.0,
    "min_corr": 12, "fallback": "hold",   # hold | retrieval (labeled retrieval top-1 keyframe pose when no PnP fix)
}


# ---------------------------------------------------------------------------------------------- calibration check (S08e)
# E04 used to compare the WHOLE calib.json sha256 with its map's, which refuses a harmless difference (camera_info arrival order, non-camera extrinsics).
# It now compares what it actually reads: the camera it uses (size, model, K, D, P, frame id, T_base_cam) against the training calibration the map was built with.
# KNOWN_CALIBRATIONS maps a map's `calibration_sha256` (whole-file hash of the training calib.json) to per-camera digests of the model-side view; a map with an
# unknown calibration keeps the strict whole-file comparison. The digest hashes values rounded to 9 decimals (CameraStatic carries the extrinsic as a matrix).
KNOWN_CALIBRATIONS = {
    "3bdea36b925c2145d4ef1a87c36be159ac4974b6278196a5daf04fed7ee1036b": {   # whole-file hash of the supplied training calibration
        "fl": "cb0caa254d29441276ae2a9414471ce08a6bae38dfd3a060144414769b0ff727",
        "fr": "454b340993f923630fa84216fa3391a3d3c6515a79d247ca7cbc615f52073d23",
    },
}


def camera_static_digest(cam) -> str:
    """sha256 of the model-side view of one camera (CameraStatic): size, distortion model, K, D, P, frame id, T_base_cam (floats rounded to 9 decimals)."""
    r = lambda seq: [round(float(v), 9) + 0.0 for v in seq]     # noqa: E731  (+0.0 turns -0.0 into 0.0)
    d = {"width": int(cam.width), "height": int(cam.height), "distortion_model": str(cam.distortion_model), "K": r(cam.K), "D": r(cam.D), "P": r(cam.P),
         "frame_id": str(cam.frame_id), "T_base_cam": r(np.asarray(cam.T_base_cam.matrix, float).ravel())}
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_calibration(static_metadata: StaticMetadata, map_manifest: dict) -> dict:
    """Accept when the whole calib.json hash equals the map's, or when the camera this map was built for (manifest `camera`, default fl; the model reads it as
    cameras['fl'] of the static metadata) equals the training camera by its digest. Otherwise ContractError."""
    want = map_manifest["calibration_sha256"]
    if static_metadata.calibration_sha256 == want:
        return {"mode": "whole_file_sha256", "calibration_sha256": want}
    ref = KNOWN_CALIBRATIONS.get(want, {}).get(map_manifest.get("camera", "fl"))
    cam = static_metadata.cameras.get("fl")
    if ref is not None and cam is not None:
        got = camera_static_digest(cam)
        if got == ref:
            return {"mode": "camera_fingerprint", "camera": map_manifest.get("camera", "fl"), "digest": got, "calibration_sha256": static_metadata.calibration_sha256,
                    "map_calibration_sha256": want}
        raise ContractError("map was built with a different calibration: the camera this map reads differs from the training camera "
                            f"(digest {got[:12]} != {ref[:12]})")
    raise ContractError("map was built with a different calibration")


def pnp_pose(obj: np.ndarray, uvn: np.ndarray, focal: float, ransac_px: float, iters: int, seed: int = 0):
    """RANSAC PnP on normalized image points (K = I). Returns dict(R_cw, t_cw, inliers, rms_px) or None."""
    import cv2
    if len(obj) < 6:
        return None
    cv2.setRNGSeed(seed)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj.astype(np.float64), uvn.astype(np.float64), np.eye(3), None, iterationsCount=iters,
                                             reprojectionError=ransac_px / focal, confidence=0.9999, flags=cv2.SOLVEPNP_AP3P)
    if not ok or inl is None or len(inl) < 6:
        return None
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl].astype(np.float64), uvn[inl].astype(np.float64), np.eye(3), None, rvec, tvec)
    R = cv2.Rodrigues(rvec)[0]
    pc = obj @ R.T + tvec.ravel()
    z = pc[:, 2]
    front = z > 0.1
    proj = pc[:, :2] / np.where(front, z, 1.0)[:, None]
    err = np.hypot(*(proj - uvn).T) * focal
    inl_mask = front & (err <= ransac_px)
    if inl_mask.sum() < 6:
        return None
    return {"R_cw": R, "t_cw": tvec.ravel(), "inlier_mask": inl_mask, "rms_px": float(np.sqrt(np.mean(err[inl_mask] ** 2)))}


def base_pose_from_pnp(R_cw: np.ndarray, t_cw: np.ndarray, T_base_cam: np.ndarray):
    """world->camera (R_cw, t_cw) -> base_link pose in map, including the camera translation (DEVELOPMENT_PLAN.md 2.1).
    Returns x, y, z, yaw."""
    T_cw = np.eye(4)
    T_cw[:3, :3], T_cw[:3, 3] = R_cw, t_cw
    T_map_cam = np.linalg.inv(T_cw)
    T_map_base = T_map_cam @ np.linalg.inv(T_base_cam)
    f = T_map_base[:3, 0]
    return float(T_map_base[0, 3]), float(T_map_base[1, 3]), float(T_map_base[2, 3]), float(math.atan2(f[1], f[0]))


class E04PnpModel:
    def __init__(self, config: dict | None = None, assets: dict | None = None):
        import torch
        # runner.create_model passes (config["params"], assets)
        a = dict(assets or {})
        self.p = {**DEFAULTS, **dict(config or {})}
        self.map = load_map(ROOT / a["map_dir"], need_local=True)
        man = self.map.manifest
        self.capabilities = Capabilities(cameras=("fl",), clock_camera="fl", calibration=True, map_id=self.map.map_id, telemetry_fields=(),
                                         device="cuda", assets=(str(a["map_dir"]),), scale_status="metric", online_mapping="disabled")
        ext = a.get("ext_dir")
        self.fe = LocalFeatures(ext, max_kp=self.p["max_kp"], det_thr=self.p["det_thr"])
        self.mt = Matcher(ext)
        self.gd = GlobalDescriptor(ext, pool=man["global_descriptor"]["pool"])
        self.index = RetrievalIndex(self.map.global_desc, self.map.kf_xy, self.map.kf_yaw)
        self.T_base_cam = np.array(man["T_base_cam_fitted"], float)
        self.torch = torch
        self._kf_dev: dict = {}
        self.prior: PriorState | None = None
        self.K = None
        self.n_init = 0
        self.n_fix = 0
        self.n_steps = 0

    # ------------------------------------------------------------------ protocol
    def initialize(self, first_pose: FirstPose, static_metadata: StaticMetadata) -> None:
        self.n_init += 1
        if self.n_init > 1:
            raise ContractError("E04 initialized twice")
        cam = static_metadata.cameras["fl"]
        self.K = np.array(cam.K, float).reshape(3, 3)
        self.D = np.array(cam.D, float)
        self.calibration_check = check_calibration(static_metadata, self.map.manifest)
        self.prior = PriorState(first_pose.x, first_pose.y, first_pose.yaw, first_pose.timestamp_ns)

    def _kf_local(self, j: int):
        d = self._kf_dev.get(j)
        if d is None:
            kp, desc, xyz = self.map.kf_local(j)
            d = {"dev": self.mt.to_dev({"kpts": kp, "desc": desc}), "xyz": xyz.astype(np.float64), "n": len(kp)}
            self._kf_dev[j] = d
        return d

    def _hold(self, flags, state, f, resid, top1=None):
        pr = self.prior
        if self.p["fallback"] == "retrieval" and top1 is not None:
            j = int(top1)
            return StepResult(float(self.map.kf_xy[j, 0]), float(self.map.kf_xy[j, 1]), float(self.map.kf_yaw[j]),
                              source_flags=("retrieval_fallback",) + tuple(flags), tracking_state=state,
                              consumed_min_ns=None if f is None else f.stamp_ns, consumed_max_ns=None if f is None else f.stamp_ns, residuals=resid)
        return StepResult(pr.x, pr.y, pr.yaw, source_flags=("hold_last",) + tuple(flags), tracking_state=state,
                          consumed_min_ns=None if f is None else f.stamp_ns, consumed_max_ns=None if f is None else f.stamp_ns, residuals=resid)

    def step(self, bundle: FrameBundle) -> StepResult:
        from camloc.mapping.triangulate import undistort_norm
        pr = self.prior
        if pr is None:
            raise ContractError("E04 step before initialize")
        self.n_steps += 1
        f = bundle.frames.get("fl")
        if f is None or not f.available:
            return self._hold(("no_image",), "held", None, {"reason": "no_image"})
        p, torch = self.p, self.torch
        t = bundle.timestamp_ns
        dt = (t - pr.t_ns) / 1e9
        tm = {}
        t0 = time.perf_counter()
        rgb = f.image()
        tm["decode"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        q_glob = self.gd.describe(rgb)
        tm["global"] = time.perf_counter() - t0
        lost = dt > p["lost_s"]
        rad = None if lost else prior_radius(dt, p["prior_base_m"], p["vmax_mps"])
        idx, sims = self.index.query(q_glob, k=p["top_k"], center_xy=None if lost else (pr.x, pr.y), radius_m=rad)
        mode = "restricted"
        if idx.size == 0:
            idx, sims = self.index.query(q_glob, k=p["top_k"])
            mode = "global"
        elif lost:
            mode = "global"
        t0 = time.perf_counter()
        qf = self.fe.extract(rgb)
        qdev = self.mt.to_dev(qf)
        tm["local"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        obj, uv, src = [], [], []
        for j in idx:
            kl = self._kf_local(int(j))
            if kl["n"] < 5:
                continue
            m, _ = self.mt.match(qdev, kl["dev"])
            if len(m):
                X = kl["xyz"][m[:, 1]]
                v = np.isfinite(X).all(1)            # context-only keypoints carry NaN
                obj.append(X[v])
                uv.append(qf["kpts"][m[v, 0]])
                src.append(np.full(int(v.sum()), int(j)))
        tm["match"] = time.perf_counter() - t0
        resid = {"vram_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1), "mode": mode, "n_cand": int(len(idx)), "sim_top": float(sims[0]) if len(sims) else None, "dt_since_fix_s": dt,
                 "kf_ids": [int(self.map.kf.kf_id.iloc[int(j)]) for j in idx], "t_ms": {k: round(v * 1e3, 1) for k, v in tm.items()}}
        if not obj:
            resid["reason"] = "no_correspondences"
            return self._hold(("no_corr",), "lost" if lost else "held", f, resid, top1=idx[0] if len(idx) else None)
        obj, uv, src = np.concatenate(obj), np.concatenate(uv), np.concatenate(src)
        resid["n_corr"] = int(len(obj))
        if len(obj) < p["min_corr"]:
            resid["reason"] = "few_correspondences"
            return self._hold(("few_corr",), "lost" if lost else "held", f, resid, top1=idx[0] if len(idx) else None)
        t0 = time.perf_counter()
        uvn = undistort_norm(self.K, self.D, uv)
        sol = pnp_pose(obj, uvn, float(self.K[0, 0]), p["ransac_px"], p["ransac_iters"])
        tm["pnp"] = time.perf_counter() - t0
        resid["t_ms"]["pnp"] = round(tm["pnp"] * 1e3, 1)
        if sol is None:
            resid["reason"] = "pnp_failed"
            return self._hold(("pnp_failed",), "lost" if lost else "held", f, resid, top1=idx[0] if len(idx) else None)
        n_inl = int(sol["inlier_mask"].sum())
        ratio = n_inl / len(obj)
        x, y, z, yaw = base_pose_from_pnp(sol["R_cw"], sol["t_cw"], self.T_base_cam)
        # nearest candidate keyframe diagnostics
        kxy = self.map.kf_xy[idx]
        j_best = int(idx[int(np.argmin(np.hypot(kxy[:, 0] - x, kxy[:, 1] - y)))])
        d_kf = float(np.min(np.hypot(kxy[:, 0] - x, kxy[:, 1] - y)))
        z_dev = abs(z - float(self.map.kf.z.iloc[j_best]))
        yaw_dev = float(math.degrees(angdiff(yaw, self.map.kf_yaw[j_best])))
        resid.update({"n_inl": n_inl, "inlier_ratio": ratio, "rms_px": sol["rms_px"], "d_kf_m": d_kf, "z_dev_m": z_dev, "yaw_dev_deg": yaw_dev,
                      "n_kf_inl": int(len(np.unique(src[sol["inlier_mask"]])))})
        reasons = []
        if n_inl < p["min_inliers"]:
            reasons.append("inliers")
        if ratio < p["min_inlier_ratio"]:
            reasons.append("inlier_ratio")
        if sol["rms_px"] > p["max_rms_px"]:
            reasons.append("rms")
        if d_kf > p["max_kf_dist_m"]:
            reasons.append("kf_distance")
        if z_dev > p["max_z_dev_m"]:
            reasons.append("z")
        if yaw_dev > p["max_yaw_dev_deg"]:
            reasons.append("yaw")
        if not lost and math.hypot(x - pr.x, y - pr.y) > rad:
            reasons.append("prior")
        if reasons:
            resid["reason"] = "gate:" + ",".join(reasons)
            resid["rejected_pose"] = [x, y, yaw]
            return self._hold(("gate_reject",), "lost" if lost else "held", f, resid, top1=idx[0] if len(idx) else None)
        self.prior = PriorState(x, y, yaw, t, True)
        self.n_fix += 1
        return StepResult(x, y, yaw, source_flags=("pnp_fix", mode), tracking_state="tracking", consumed_min_ns=f.stamp_ns, consumed_max_ns=f.stamp_ns,
                          residuals=resid)

    def close(self) -> None:
        self.prior = None
        self._kf_dev.clear()
