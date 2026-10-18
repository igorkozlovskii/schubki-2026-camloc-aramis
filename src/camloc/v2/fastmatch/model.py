"""S1.22 multi-camera landmark-fix model: seven independent E04 chains (own prior per camera, as the S1.2 streams), with switchable acceleration.

Every `stride`-th tick each camera with an available frame runs retrieval -> ALIKED -> LightGlue -> PnP + gates exactly as `camloc.models.pnp.E04PnpModel`
(the control variant `serial` is that code path, camera by camera). Switches (config params):

  mode               "serial"  : one camera at a time, stock Matcher.match per candidate keyframe (pruning + early stop as in production)
                     "batched" : DINOv2 in one batch; ALIKED batched per image size (or padded to one size); LightGlue on all (camera, candidate) pairs in one padded batch
                                 (BatchedLG: no point pruning, optional batch-level early stop); PnP in a thread pool
  kp_fl / kp_other   ALIKED max keypoints of the query for fl / the other cameras (map keyframes unchanged)
  topk_fl / topk_other  candidate keyframes per camera
  lg_depth           batch-level LightGlue early-stop confidence (-1 = all 9 layers), lg_layers = hard layer cap
  lg_mp              LightGlue under fp16 autocast
  aliked_pad         "none" | "group" | "pad" (zero pad all crops to the largest size; one ALIKED call)
  cams_active        policy: "all" (default) | "fl_always_others_every:N" etc. is evaluated offline, not here
Inference path: imports no ground-truth module; maps are the training-only LOO maps.
"""
from __future__ import annotations

import dataclasses
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from camloc.contracts import Capabilities, ContractError, FirstPose, FrameBundle, StaticMetadata, StepResult
from camloc.mapping.map import load_map
from camloc.models.pnp import DEFAULTS, base_pose_from_pnp, check_calibration, pnp_pose
from camloc.models.retrieval import ROOT, GlobalDescriptor, LocalFeatures, Matcher, PriorState, RetrievalIndex, angdiff, prior_radius

CAMS = ("fl", "fr", "cl", "cr", "r", "rl", "rr")
SKIP_FLAG = "stride_skip"


class _Cam:
    def __init__(self, name: str, map_dir: str, preload: bool):
        self.name = name
        self.map = load_map(ROOT / map_dir, need_local=True)
        man = self.map.manifest
        self.man = man
        self.crop = tuple(man.get("image_crop") or (12, 0, 1494, 540))
        self.pool = man["global_descriptor"]["pool"]
        self.index = RetrievalIndex(self.map.global_desc, self.map.kf_xy, self.map.kf_yaw)
        self.T_base_cam = np.array(man["T_base_cam_fitted"], float)
        self.prior: PriorState | None = None
        self.K = self.D = None
        self.kf_dev: dict = {}
        self.pad = None
        self.preload = preload

    def init(self, first_pose, sm: StaticMetadata, cam_static):
        self.K = np.array(cam_static.K, float).reshape(3, 3)
        self.D = np.array(cam_static.D, float)
        check_calibration(dataclasses.replace(sm, cameras={"fl": cam_static}), self.man)
        self.prior = PriorState(first_pose.x, first_pose.y, first_pose.yaw, first_pose.timestamp_ns)

    def kf_np(self, j):
        a, b = int(self.map.local["kp_off"][j]), int(self.map.local["kp_off"][j + 1])
        return a, b

    def build_pad(self, torch, dev, nmax_cap: int | None = None):
        """All keyframes padded to one length on the GPU: kpts [K,N,2], desc [K,N,128] (fp16), n [K]. Used by the batched path."""
        off = np.asarray(self.map.local["kp_off"], np.int64)
        n = np.diff(off)
        N = int(n.max()) if nmax_cap is None else min(int(n.max()), nmax_cap)
        K = len(n)
        kp = np.ones((K, N, 2), np.float32)
        ds = np.zeros((K, N, self.map.local["desc"].shape[1]), np.float16)
        for j in range(K):
            c = min(int(n[j]), N)
            kp[j, :c] = self.map.local["kpts"][off[j]:off[j] + c]
            ds[j, :c] = self.map.local["desc"][off[j]:off[j] + c]
        self.pad = {"kp": torch.from_numpy(kp).to(dev), "desc": torch.from_numpy(ds).to(dev), "n": torch.from_numpy(np.minimum(n, N)).to(dev), "n_np": np.minimum(n, N), "off": off}


class FastMultiCamModel:
    def __init__(self, config: dict | None = None, assets: dict | None = None):
        import torch
        self.torch = torch
        p = {"cameras": list(CAMS), "stride": 2, "mode": "serial", "kp_fl": 2048, "kp_other": 2048, "topk_fl": 6, "topk_other": 6, "lg_depth": 0.95, "lg_width": 0.99,
             "lg_layers": 9, "lg_mp": False, "aliked_pad": "group", "pnp_threads": 0, "decode_threads": 0, "limit_ticks": None, "min_kp_other_map": None, **dict(config or {})}
        a = dict(assets or {})
        self.p = p
        self.e04 = {**DEFAULTS, **{k: v for k, v in p.items() if k in DEFAULTS}}
        self.names = list(p["cameras"])
        self.stride = int(p["stride"])
        self.batched = p["mode"] == "batched"
        ext = a.get("ext_dir")
        self.cams = {c: _Cam(c, a["map_dirs"][c], self.batched) for c in self.names}
        # shared networks: DINOv2 (one), ALIKED (fl and other keypoint budgets), LightGlue (stock matcher; batched wrapper reuses its net)
        c0 = self.cams[self.names[0]]
        self.gd = GlobalDescriptor(ext, pool=c0.pool, crop=c0.crop)
        self.fe_fl = LocalFeatures(ext, max_kp=int(p["kp_fl"]), det_thr=self.e04["det_thr"], crop=c0.crop)
        self.fe_ot = self.fe_fl if int(p["kp_other"]) == int(p["kp_fl"]) else LocalFeatures(ext, max_kp=int(p["kp_other"]), det_thr=self.e04["det_thr"], crop=c0.crop)
        self.mt = Matcher(ext)
        self.mt_ot = self.mt
        if p["lg_width"] != 0.99 or p["lg_depth"] != 0.95:
            from lightglue import LightGlue
            # serial path with different adaptivity: a second LightGlue with the same weights
            self.mt_ot = Matcher(ext, filter_threshold=0.1)
            self.mt_ot.net.conf.depth_confidence = float(p["lg_depth"])
            self.mt_ot.net.conf.width_confidence = float(p["lg_width"])
        self.blg = None
        if self.batched:
            from camloc.v2.fastmatch.lgbatch import BatchedLG
            self.blg = BatchedLG(self.mt.net, depth_confidence=float(p["lg_depth"]), n_layers=int(p["lg_layers"]), mp=bool(p["lg_mp"]))
            for cm in self.cams.values():
                cm.build_pad(torch, "cuda")
        self.pool_pnp = ThreadPoolExecutor(int(p["pnp_threads"])) if int(p["pnp_threads"]) > 0 else None
        self.pool_dec = ThreadPoolExecutor(int(p["decode_threads"])) if int(p["decode_threads"]) > 0 else None
        self.capabilities = Capabilities(cameras=tuple(self.names), clock_camera="fl", calibration=True, map_id="+".join(c.map.map_id for c in self.cams.values()), telemetry_fields=(),
                                         device="cuda", assets=tuple(str(v) for v in a["map_dirs"].values()), scale_status="metric", online_mapping="disabled")
        self.i = 0
        self.n_init = 0
        self.n_ticks = 0

    # ------------------------------------------------------------------ protocol
    def initialize(self, first_pose: FirstPose, static_metadata: StaticMetadata) -> None:
        self.n_init += 1
        if self.n_init > 1:
            raise ContractError("initialized twice")
        cams = dict(static_metadata.cameras)
        for c, cm in self.cams.items():
            if c not in cams:
                raise ContractError(f"camera {c} missing in the static metadata")
            cm.init(first_pose, static_metadata, cams[c])

    def close(self) -> None:
        if self.pool_pnp:
            self.pool_pnp.shutdown()
        if self.pool_dec:
            self.pool_dec.shutdown()

    def _sync(self):
        self.torch.cuda.synchronize()
        return time.perf_counter()

    # ------------------------------------------------------------------ stages
    def _describe_batch(self, crops):
        """DINOv2-GeM for a list of cropped RGB arrays -> [B,384] float32. Same preprocessing as GlobalDescriptor.describe, one forward."""
        import cv2
        torch, gd = self.torch, self.gd
        ims = [cv2.resize(im, (gd.size[1], gd.size[0]), interpolation=cv2.INTER_AREA) for im in crops]
        t = torch.from_numpy(np.stack(ims)).to(gd.device).permute(0, 3, 1, 2).float() / 255.0
        t = ((t - gd.mean) / gd.std).half()
        with torch.no_grad():
            h = gd.model(pixel_values=t).last_hidden_state.float()
        cls, pt = h[:, 0], h[:, 1:]
        gem = pt.clamp(min=1e-6).pow(3).mean(1).pow(1 / 3) if gd.pool == "gem" else pt.mean(1)
        v = cls if gd.pool == "cls" else gem
        return torch.nn.functional.normalize(v, dim=-1).cpu().numpy().astype(np.float32)

    def _aliked_batch(self, items, fe):
        """items: list of (cam, crop image). -> {cam: (kpts_gpu [m,2] full-image px, desc_gpu [m,128], kpts_np)}; batched per image size (aliked_pad=group) or padded (pad)."""
        torch = self.torch
        out = {}
        groups = {}
        mode = self.p["aliked_pad"]
        if mode == "pad":
            H = max(im.shape[0] for _, im in items)
            W = max(im.shape[1] for _, im in items)
            groups[(H, W)] = [(c, im) for c, im in items]
        else:
            for c, im in items:
                groups.setdefault(im.shape[:2], []).append((c, im))
        for (H, W), lst in groups.items():
            arr = np.zeros((len(lst), H, W, 3), np.uint8)
            for k, (_, im) in enumerate(lst):
                arr[k, :im.shape[0], :im.shape[1]] = im
            t = torch.from_numpy(arr).to("cuda").permute(0, 3, 1, 2).float() / 255.0
            with torch.no_grad():
                o = fe.net({"image": t})
            for k, (c, im) in enumerate(lst):
                x0, y0 = self.cams[c].crop[0], self.cams[c].crop[1]
                kp = o["keypoints"][k].float() + torch.tensor([x0, y0], device="cuda", dtype=torch.float32)
                out[c] = (kp, o["descriptors"][k])
        return out

    def _correspondences(self, c, kpq_np, sel, matches_per_kf):
        """matches_per_kf: list of (kf row j, query idx array, kf-local idx array) -> obj, uv, src arrays."""
        cm = self.cams[c]
        obj, uv, src = [], [], []
        xyz_all = cm.map.local["xyz"]
        for j, qi, ki in matches_per_kf:
            a = int(cm.map.local["kp_off"][j])
            X = xyz_all[a + ki].astype(np.float64)
            v = np.isfinite(X).all(1)
            obj.append(X[v]); uv.append(kpq_np[qi[v]]); src.append(np.full(int(v.sum()), int(j)))
        return obj, uv, src

    def _solve(self, c, obj, uv, src, idx, sims, mode, rad, lost, dt, tm):
        """PnP + gates of E04PnpModel.step. -> per-camera record (dict)."""
        from camloc.mapping.triangulate import undistort_norm
        cm, p, pr = self.cams[c], self.e04, self.cams[c].prior
        rec = {"mode": mode, "n_cand": int(len(idx)), "sim_top": float(sims[0]) if len(sims) else None, "dt_since_fix_s": dt, "valid": False,
               "kf_ids": [int(cm.map.kf.kf_id.iloc[int(j)]) for j in idx]}
        if not obj or sum(len(o) for o in obj) == 0:
            rec["reason"] = "no_correspondences"
            return rec
        obj, uv, src = np.concatenate(obj), np.concatenate(uv), np.concatenate(src)
        rec["n_corr"] = int(len(obj))
        if len(obj) < p["min_corr"]:
            rec["reason"] = "few_correspondences"
            return rec
        t0 = time.perf_counter()
        uvn = undistort_norm(cm.K, cm.D, uv)
        sol = pnp_pose(obj, uvn, float(cm.K[0, 0]), p["ransac_px"], p["ransac_iters"])
        rec["pnp_ms"] = (time.perf_counter() - t0) * 1e3
        if sol is None:
            rec["reason"] = "pnp_failed"
            return rec
        n_inl = int(sol["inlier_mask"].sum())
        ratio = n_inl / len(obj)
        x, y, z, yaw = base_pose_from_pnp(sol["R_cw"], sol["t_cw"], cm.T_base_cam)
        kxy = cm.map.kf_xy[idx]
        j_best = int(idx[int(np.argmin(np.hypot(kxy[:, 0] - x, kxy[:, 1] - y)))])
        d_kf = float(np.min(np.hypot(kxy[:, 0] - x, kxy[:, 1] - y)))
        z_dev = abs(z - float(cm.map.kf.z.iloc[j_best]))
        yaw_dev = float(math.degrees(angdiff(yaw, cm.map.kf_yaw[j_best])))
        rec.update({"n_inl": n_inl, "inlier_ratio": ratio, "rms_px": sol["rms_px"], "d_kf_m": d_kf, "z_dev_m": z_dev, "yaw_dev_deg": yaw_dev, "n_kf_inl": int(len(np.unique(src[sol["inlier_mask"]])))})
        reasons = []
        if n_inl < p["min_inliers"]: reasons.append("inliers")
        if ratio < p["min_inlier_ratio"]: reasons.append("inlier_ratio")
        if sol["rms_px"] > p["max_rms_px"]: reasons.append("rms")
        if d_kf > p["max_kf_dist_m"]: reasons.append("kf_distance")
        if z_dev > p["max_z_dev_m"]: reasons.append("z")
        if yaw_dev > p["max_yaw_dev_deg"]: reasons.append("yaw")
        if not lost and math.hypot(x - pr.x, y - pr.y) > rad: reasons.append("prior")
        if reasons:
            rec["reason"] = "gate:" + ",".join(reasons)
            rec["rejected_pose"] = [x, y, yaw]
            return rec
        rec.update(valid=True, x=x, y=y, yaw=yaw)
        return rec

    # ------------------------------------------------------------------ step
    def step(self, bundle: FrameBundle) -> StepResult:
        i = self.i
        self.i += 1
        fl = self.cams["fl"]
        pr0 = fl.prior
        if pr0 is None:
            raise ContractError("step before initialize")
        lim = self.p.get("limit_ticks")
        if i % self.stride or (lim is not None and self.n_ticks >= int(lim)):
            return StepResult(pr0.x, pr0.y, pr0.yaw, source_flags=(SKIP_FLAG,), tracking_state="held", residuals={"reason": SKIP_FLAG})
        self.n_ticks += 1
        t = bundle.timestamp_ns
        torch = self.torch
        tm = {}
        T0 = self._sync()
        act = [c for c in self.names if (bundle.frames.get(c) is not None and bundle.frames[c].available)]
        # ---- decode
        if self.pool_dec:
            rgbs = dict(zip(act, self.pool_dec.map(lambda c: bundle.frames[c].image(), act)))
        else:
            rgbs = {c: bundle.frames[c].image() for c in act}
        T1 = time.perf_counter()
        tm["decode"] = T1 - T0
        st = {}
        for c in act:
            cm = self.cams[c]
            dt = (t - cm.prior.t_ns) / 1e9
            lost = dt > self.e04["lost_s"]
            st[c] = {"dt": dt, "lost": lost, "rad": None if lost else prior_radius(dt, self.e04["prior_base_m"], self.e04["vmax_mps"])}
        crops = {c: rgbs[c][self.cams[c].crop[1]:self.cams[c].crop[3], self.cams[c].crop[0]:self.cams[c].crop[2]] for c in act}
        # ---- global descriptor + retrieval
        if self.batched:
            Q = self._describe_batch([crops[c] for c in act])
            qs = dict(zip(act, Q))
        else:
            qs = {}
            for c in act:
                self.gd.crop = self.cams[c].crop
                qs[c] = self.gd.describe(rgbs[c])
        T2 = self._sync()
        tm["global"] = T2 - T1
        cand = {}
        for c in act:
            cm, s = self.cams[c], st[c]
            k = int(self.p["topk_fl"] if c == "fl" else self.p["topk_other"])
            idx, sims = cm.index.query(qs[c], k=k, center_xy=None if s["lost"] else (cm.prior.x, cm.prior.y), radius_m=s["rad"])
            mode = "restricted"
            if idx.size == 0:
                idx, sims = cm.index.query(qs[c], k=k)
                mode = "global"
            elif s["lost"]:
                mode = "global"
            cand[c] = (idx, sims, mode)
        T3 = time.perf_counter()
        tm["retrieve"] = T3 - T2
        recs = {}
        if self.batched:
            recs = self._batched(act, crops, cand, st, tm)
        else:
            recs = self._serial(act, rgbs, cand, st, tm)
        T4 = self._sync()
        # ---- update priors, record
        out = {}
        for c in act:
            r = recs[c]
            if r["valid"]:
                self.cams[c].prior = PriorState(r["x"], r["y"], r["yaw"], t, True)
            out[c] = r
        total = T4 - T0
        tm["total"] = total
        res = {"cams": out, "t_ms": {k: round(v * 1e3, 2) for k, v in tm.items()}, "n_cam": len(act)}
        if self.blg is not None:
            res["lg_stop"] = self.blg.last_stop
        fl_r = out.get("fl")
        if fl_r is not None and fl_r["valid"]:
            return StepResult(fl_r["x"], fl_r["y"], fl_r["yaw"], source_flags=("pnp_fix",), tracking_state="tracking", consumed_min_ns=t, consumed_max_ns=t, residuals=res)
        p = self.cams["fl"].prior
        return StepResult(p.x, p.y, p.yaw, source_flags=("hold_last",), tracking_state="held", consumed_min_ns=t, consumed_max_ns=t, residuals=res)

    # ------------------------------------------------------------------ serial (control)
    def _serial(self, act, rgbs, cand, st, tm):
        tl = tmm = 0.0
        data = {}
        stop_corr = int(self.p.get("stop_corr", 0) or 0)
        for c in act:
            cm = self.cams[c]
            idx, sims, mode = cand[c]
            fe = self.fe_fl if c == "fl" else self.fe_ot
            mt = self.mt if c == "fl" else self.mt_ot
            t0 = self._sync()
            fe.crop = cm.crop
            qf = fe.extract(rgbs[c])
            qdev = mt.to_dev(qf)
            t1 = self._sync()
            obj, uv, src = [], [], []
            used = []
            for j in idx:
                d = cm.kf_dev.get(int(j))
                if d is None:
                    kp, desc, xyz = cm.map.kf_local(int(j))
                    d = {"dev": mt.to_dev({"kpts": kp, "desc": desc}), "xyz": xyz.astype(np.float64), "n": len(kp)}
                    cm.kf_dev[int(j)] = d
                if d["n"] < 5:
                    continue
                m, _ = mt.match(qdev, d["dev"])
                used.append(int(j))
                if len(m):
                    X = d["xyz"][m[:, 1]]
                    v = np.isfinite(X).all(1)
                    obj.append(X[v]); uv.append(qf["kpts"][m[v, 0]]); src.append(np.full(int(v.sum()), int(j)))
                if stop_corr and sum(len(o) for o in obj) >= stop_corr:
                    break
            t2 = self._sync()
            tl += t1 - t0; tmm += t2 - t1
            data[c] = (obj, uv, src, len(used))
        tm["local"], tm["match"] = tl, tmm
        t2 = time.perf_counter()

        def solve(c):
            idx, sims, mode = cand[c]
            s = st[c]
            t = time.perf_counter()
            rec = self._solve(c, *data[c][:3], idx, sims, mode, s["rad"], s["lost"], s["dt"], tm)
            rec["n_kf_matched"] = data[c][3]
            rec["t_ms"] = {"pnp": (time.perf_counter() - t) * 1e3}
            return rec
        recs = dict(zip(act, self.pool_pnp.map(solve, act))) if self.pool_pnp else {c: solve(c) for c in act}
        tm["pnp"] = time.perf_counter() - t2
        return recs

    # ------------------------------------------------------------------ batched
    def _batched(self, act, crops, cand, st, tm):
        torch = self.torch
        t0 = self._sync()
        feats = {}
        fl_items = [(c, crops[c]) for c in act if c == "fl"]
        ot_items = [(c, crops[c]) for c in act if c != "fl"]
        if fl_items:
            feats.update(self._aliked_batch(fl_items, self.fe_fl))
        if ot_items:
            feats.update(self._aliked_batch(ot_items, self.fe_ot))
        t1 = self._sync()
        tm["local"] = t1 - t0
        # ---- LightGlue: all (camera, candidate) pairs, grouped by query length (kp_fl may differ from kp_other)
        pairs = []   # (cam, kf row j)
        for c in act:
            for j in cand[c][0]:
                if self.cams[c].pad["n_np"][int(j)] >= 5:
                    pairs.append((c, int(j)))
        res = {}
        if pairs:
            groups = {}
            for k, (c, j) in enumerate(pairs):
                groups.setdefault(feats[c][0].shape[0], []).append(k)
            for m, ks in groups.items():
                qk = torch.stack([feats[pairs[k][0]][0] for k in ks])
                qd = torch.stack([feats[pairs[k][0]][1] for k in ks]).float()
                nmax = int(max(self.cams[pairs[k][0]].pad["n_np"][pairs[k][1]] for k in ks))
                def _fit(t, ones=False):
                    t = t[:nmax]
                    if t.shape[0] < nmax:
                        t = torch.cat([t, (torch.ones if ones else torch.zeros)((nmax - t.shape[0], t.shape[1]), device=t.device, dtype=t.dtype)])
                    return t
                kk = torch.stack([_fit(self.cams[pairs[k][0]].pad["kp"][pairs[k][1]], True) for k in ks])
                kd = torch.stack([_fit(self.cams[pairs[k][0]].pad["desc"][pairs[k][1]]) for k in ks]).float()
                nn_ = torch.tensor([int(self.cams[pairs[k][0]].pad["n_np"][pairs[k][1]]) for k in ks], device="cuda")
                valid1 = torch.arange(nmax, device="cuda")[None] < nn_[:, None]
                m0, _ = self.blg.match(qk, qd, kk, kd, valid1)
                m0 = m0.cpu().numpy()
                for r, k in enumerate(ks):
                    res[k] = m0[r]
        t2 = self._sync()
        tm["match"] = t2 - t1
        kpnp = {c: feats[c][0].cpu().numpy() for c in act}
        per = {c: ([], [], []) for c in act}
        for k, (c, j) in enumerate(pairs):
            m0 = res[k]
            qi = np.flatnonzero(m0 >= 0)
            if len(qi) == 0:
                continue
            ki = m0[qi]
            o, u, s_ = self._correspondences(c, kpnp[c], None, [(j, qi, ki)])
            per[c][0].extend(o); per[c][1].extend(u); per[c][2].extend(s_)

        def solve(c):
            idx, sims, mode = cand[c]
            s = st[c]
            t = time.perf_counter()
            rec = self._solve(c, *per[c], idx, sims, mode, s["rad"], s["lost"], s["dt"], tm)
            rec["t_ms"] = {"pnp": (time.perf_counter() - t) * 1e3}
            return rec
        if self.pool_pnp:
            recs = dict(zip(act, self.pool_pnp.map(solve, act)))
        else:
            recs = {c: solve(c) for c in act}
        tm["pnp"] = time.perf_counter() - t2
        return recs
