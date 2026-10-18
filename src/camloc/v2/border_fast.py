"""S3.2d: GPU implementation of the S3.2c `e_fit` border measurement (same objective, priors, abstain logic; batched on the GPU).

Objective (identical to `border_seg_att`): J(theta) = -S(theta) / T + 0.5 * sum_free ((theta_i - mu_i) / sd_i)^2 with
S(theta) = (1 / ss^2) * sum over cameras of the log-odds evidence summed over the road polygon projected at theta = (dn, dpsi, droll, dpitch, dz).
What changes is the machinery, not the model:
  * the 5-plane clipping, pinhole + plumb-bob projection and the label-grid mapping of the corridor polygons run in torch float64, for all
    candidate thetas x cameras x polygons at once (same clip planes, z_min and 1.1 x view limits as `border_seg.SegScene._project`);
  * the polygon fill is not rasterised: a scan-line integral over a row-prefix-sum table of the evidence image gives sum_E exactly in x
    (sub-pixel, linear in the prefix) and at the pixel-row centres in y (rows of the ss x label grid). `edge_pad` (pixels) widens each span the
    way cv2.fillPoly's outline pixels do (declared, tested against the CPU score);
  * the search is batched: the (dn, dpsi) grid is one batch; the pattern search polls the 10 axis neighbours of the 3 distinct starts together
    (plus the combined improving step) instead of one coordinate at a time; the Newton / Laplace stage evaluates its 51 points as one batch.
Everything else (scene setup: corridor polygons, ignore masks, map attitude; the abstain reasons; Laplace covariance; the causal `AttTracker`)
is the S3.2c code, imported read-only. No ground truth: `s_init` is the caller's own estimate; the oracle attitude is only an optional
`items[cam]["attitude"]` passed by an evaluator.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from camloc.geometry.transforms import wrap_angle
from camloc.v2.border_probe import BorderMeas
from camloc.v2.border_seg import evidence_image
from camloc.v2.border_seg_att import DEG, AttFitConfig, AttMeasurer, AttScene, _scales
from camloc.v2.seg import LABEL_H, LABEL_W
from camloc.v2.seg import labels as L

EXTRA = 4            # vertices a clip may add per plane before the buffer saturates (overflow is counted, see FastBatch.overflow)


# ------------------------------------------------------------------------------------------------ small torch helpers
def _clip_batch(P: torch.Tensor, cnt: torch.Tensor, normal: torch.Tensor, off: torch.Tensor):
    """Sutherland-Hodgman against normal . p - off > 0 for N padded polygons. P (N, m, 3) float64, cnt (N,), normal (N, 3), off (N,).
    Returns (P' (N, m + EXTRA, 3), cnt' (N,), overflow count)."""
    N, m, _ = P.shape
    dev = P.device
    idx = torch.arange(m, device=dev)[None, :]
    valid = idx < cnt[:, None]
    nxt = torch.where(idx + 1 >= cnt[:, None], torch.zeros_like(idx), idx + 1)
    d = (P * normal[:, None, :]).sum(-1) - off[:, None]
    Pn = torch.gather(P, 1, nxt[..., None].expand(N, m, 3))
    dn = torch.gather(d, 1, nxt)
    ins, insn = d > 0, dn > 0
    cross = valid & (ins != insn)
    den = d - dn
    t = d / torch.where(den.abs() < 1e-12, torch.ones_like(den), den)
    X = P + (Pn - P) * t[..., None]
    pts = torch.stack([X, Pn], 2).reshape(N, 2 * m, 3)
    keep = torch.stack([cross, valid & insn], 2).reshape(N, 2 * m)
    pos = torch.cumsum(keep.to(torch.int32), 1) - 1
    tot = keep.sum(1)
    W = m + EXTRA
    pos = torch.where(keep, pos.clamp(max=W - 1), torch.full_like(pos, W))
    out = torch.zeros(N, W + 1, 3, dtype=P.dtype, device=dev)
    out.scatter_(1, pos.long()[..., None].expand(N, 2 * m, 3), pts)
    return out[:, :W], tot.clamp(max=W), (tot > W).sum()


class FastBatch:
    """Per-tick device tensors of the cameras that enter the fit, and the batched evaluation of S(theta)."""

    def __init__(self, fm: "FastMeasurer", scenes: list, Es: list):
        dev, C = fm.device, len(scenes)
        self.fm, self.C, self.dev, self.ss = fm, C, dev, fm.acfg.seg.ss
        cams = [sc.cam for sc in scenes]
        st = fm.static
        self.Rt = torch.stack([st[c]["RbcT"] for c in cams])           # (C, 3, 3) Rbc^T
        self.tbc = torch.stack([st[c]["tbc"] for c in cams])
        self.fxy = torch.stack([st[c]["intr"] for c in cams])         # (C, 4): fx fy cx cy
        self.D = torch.stack([st[c]["D"] for c in cams])              # (C, 5)
        self.lim = torch.stack([st[c]["lim"] for c in cams])         # (C, 2)
        self.sxy = torch.stack([st[c]["sxy"] for c in cams])         # (C, 2): label scale x, y  (label / full-res)
        Q = max(len(sc.road_world) for sc in scenes)
        M = max(len(w) for sc in scenes for w in sc.road_world)
        Pw = np.zeros((C, Q, M, 3))
        cnt = np.zeros((C, Q), np.int64)
        for c, sc in enumerate(scenes):
            for q, w in enumerate(sc.road_world):
                Pw[c, q, : len(w)] = w
                cnt[c, q] = len(w)
        self.Pw = torch.from_numpy(Pw).to(dev)
        self.cnt = torch.from_numpy(cnt).to(dev)
        self.Q, self.M = Q, M
        pose = np.array([[sc.pose0[0], sc.pose0[1], sc.z_base, sc.roll0, sc.pitch0, sc.pose0[2], sc.nrm[0], sc.nrm[1]] for sc in scenes])
        self.pose = torch.from_numpy(pose).to(dev)
        # evidence -> row-prefix table  Pc[c, r, i] = sum_{i' < i} E[c, r, i']   (r over the ss x label rows)
        E = torch.stack(Es) if isinstance(Es[0], torch.Tensor) else torch.from_numpy(np.stack(Es)).to(dev)
        self.R, self.W = E.shape[1], E.shape[2]
        self.Pc = torch.nn.functional.pad(torch.cumsum(E.to(torch.float32), 2), (1, 0)).contiguous()
        self.overflow = 0

    # ---- S(theta) for K candidates, returns (K,) float64
    def score(self, th: torch.Tensor, cap: int | None = None) -> torch.Tensor:
        if cap is None and getattr(self, 'runner', None) is not None:
            return self.runner.score(self, th)
        K, C, Q, M, dev = th.shape[0], self.C, self.Q, self.M, self.dev
        p = self.pose
        dn, dpsi, dr, dp, dz = (th[:, i][:, None] for i in range(5))
        roll, pitch, yaw = p[None, :, 3] + dr, p[None, :, 4] + dp, p[None, :, 5] + dpsi          # (K, C)
        cr, sr, cp, sp, cy, sy = torch.cos(roll), torch.sin(roll), torch.cos(pitch), torch.sin(pitch), torch.cos(yaw), torch.sin(yaw)
        Rm = torch.stack([torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], -1),
                          torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], -1),
                          torch.stack([-sp, cp * sr, cp * cr], -1)], -2)                           # (K, C, 3, 3) = Rz Ry Rx
        t = torch.stack([p[None, :, 0] + dn * p[None, :, 6], p[None, :, 1] + dn * p[None, :, 7], p[None, :, 2] + dz], -1)   # (K, C, 3)
        B = torch.einsum("cij,kclj->kcil", self.Rt, Rm)                                          # Rbc^T @ R^T  (Rm^T[j, l] = Rm[l, j])
        d = -torch.einsum("kcij,kcj->kci", B, t) - torch.einsum("cij,cj->ci", self.Rt, self.tbc)[None]
        Pcam = torch.einsum("kcij,cqmj->kcqmi", B, self.Pw) + d[:, :, None, None, :]           # (K, C, Q, M, 3)
        N = K * C * Q
        P = Pcam.reshape(N, M, 3)
        cnt = self.cnt[None].expand(K, C, Q).reshape(N)
        lx, ly = self.lim[:, 0], self.lim[:, 1]
        zero, one = torch.zeros_like(lx), torch.ones_like(lx)
        planes = [(torch.stack([zero, zero, one], -1), torch.full_like(lx, L_ZMIN)),
                  (torch.stack([-one, zero, lx], -1), zero), (torch.stack([one, zero, lx], -1), zero),
                  (torch.stack([zero, -one, ly], -1), zero), (torch.stack([zero, one, ly], -1), zero)]
        ovf = 0
        for nrm, off in planes:
            nn_ = nrm[None, :, None, :].expand(K, C, Q, 3).reshape(N, 3)
            oo = off[None, :, None].expand(K, C, Q).reshape(N)
            P, cnt, o = _clip_batch(P, cnt, nn_, oo)
            ovf = ovf + o
        self._ovf = ovf
        m = P.shape[1]
        P = P.reshape(K, C, Q, m, 3)
        cnt = cnt.reshape(K, C, Q)
        idx = torch.arange(m, device=dev)
        vmask = idx[None, None, None, :] < cnt[..., None]
        Z = torch.where(vmask, P[..., 2], torch.ones_like(P[..., 2]))
        x, y = P[..., 0] / Z, P[..., 1] / Z
        k1, k2, p1, p2, k3 = (self.D[None, :, None, None, i] for i in range(5))
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        fx, fy, cx, cy_ = (self.fxy[None, :, None, None, i] for i in range(4))
        sx, sy_ = self.sxy[None, :, None, None, 0], self.sxy[None, :, None, None, 1]
        ss = self.ss
        qx = ((fx * xd + cx) + 0.5) * sx * ss - 0.5
        qy = ((fy * yd + cy_) + 0.5) * sy_ * ss - 0.5
        qx, qy = qx.clamp(-2e4, 2e4), qy.clamp(-2e4, 2e4)
        nx_ = torch.where(idx[None, None, None, :] + 1 >= cnt[..., None], torch.zeros_like(idx)[None, None, None, :], idx[None, None, None, :] + 1)
        bx, by = torch.gather(qx, 3, nx_), torch.gather(qy, 3, nx_)
        ev = (idx[None, None, None, :] < cnt[..., None]) & (cnt[..., None] >= 3)
        area = ((qx * by - bx * qy) * ev).sum(-1)                                         # (K, C, Q) twice the signed area
        sgn = torch.sign(area)
        r0 = torch.ceil(torch.minimum(qy, by)).clamp(min=0)
        r1 = (torch.ceil(torch.maximum(qy, by)) - 1).clamp(max=self.R - 1)
        ne = ((r1 - r0 + 1).clamp(min=0) * ev).to(torch.int64).reshape(-1)
        out = torch.zeros(K, dtype=torch.float64, device=dev)
        if cap is None:                                   # dynamic: one host sync for the number of (edge, row) crossings
            tot = int(ne.sum())
            if tot == 0:
                return out
            eid = torch.repeat_interleave(torch.arange(ne.numel(), device=dev), ne, output_size=tot)
            start = torch.cumsum(ne, 0) - ne
            valid = torch.ones(tot, dtype=torch.bool, device=dev)
            ent = torch.arange(tot, device=dev)
        else:                                             # static capacity (CUDA graph): searchsorted over the cumulative counts
            cum = torch.cumsum(ne, 0)
            ent = torch.arange(cap, device=dev)
            eid = torch.searchsorted(cum, ent, right=True).clamp(max=ne.numel() - 1)
            start = cum - ne
            valid = ent < cum[-1]
            self._tot = cum[-1]
        row = (r0.reshape(-1)[eid] + (ent - start[eid]).to(torch.float64)).clamp(0, self.R - 1)
        ay_, by_ = qy.reshape(-1)[eid], by.reshape(-1)[eid]
        ax_, bx_ = qx.reshape(-1)[eid], bx.reshape(-1)[eid]
        den = by_ - ay_
        xs = ax_ + (row - ay_) * (bx_ - ax_) / torch.where(den == 0, torch.ones_like(den), den)
        dirn = torch.where(by_ > ay_, 1.0, -1.0).to(torch.float64)
        kcq = eid // m
        sg = sgn.reshape(-1)[kcq]
        c = (kcq // Q) % C
        k = kcq // (C * Q)
        u = (xs + dirn * sg * self.fm.edge_pad + 0.5).clamp(0, self.W)
        i0 = u.floor().clamp(max=self.W - 1)
        fr = (u - i0).to(torch.float32)
        base = (c * self.R + row.long()) * (self.W + 1) + i0.long()
        F = torch.take(self.Pc, base) * (1 - fr) + torch.take(self.Pc, base + 1) * fr
        w = torch.where(valid, dirn * sg * F.to(torch.float64), torch.zeros_like(xs))
        out.index_add_(0, k, w)
        return out / float(ss * ss)


L_ZMIN = L.LabelConfig().z_min


class GraphRunner:
    """CUDA-graph replay of `FastBatch.score` for the usual tick (6 cameras, <= Q polygons, M <= Mpad): static shapes, entry capacity `cap` with an overflow
    check (a tick that exceeds it falls back to the eager path). The graph runs the same torch ops as the eager path, so values agree to float64 summation order."""

    BUCKETS = (4, 64, 160)

    def __init__(self, fm, C=6, Q=2, Mpad=512):
        self.fm, self.C, self.Q, self.Mpad, self.dev = fm, C, Q, Mpad, fm.device
        self.graphs = {}
        self.shadow = None
        self.cap = {}
        self.fallbacks = 0

    def _alloc(self, fb):
        d, f64 = self.dev, torch.float64
        sh = FastBatch.__new__(FastBatch)
        sh.fm, sh.C, sh.Q, sh.M, sh.dev, sh.ss, sh.R, sh.W = self.fm, self.C, self.Q, self.Mpad, d, fb.ss, fb.R, fb.W
        sh.runner = None
        sh.Rt, sh.tbc, sh.fxy, sh.D, sh.lim, sh.sxy = (torch.zeros_like(x) for x in (fb.Rt, fb.tbc, fb.fxy, fb.D, fb.lim, fb.sxy))
        sh.Pw = torch.zeros(self.C, self.Q, self.Mpad, 3, dtype=f64, device=d)
        sh.cnt = torch.zeros(self.C, self.Q, dtype=torch.int64, device=d)
        sh.pose = torch.zeros_like(fb.pose)
        sh.Pc = torch.zeros_like(fb.Pc)
        self.shadow = sh

    def usable(self, fb):
        return fb.C == self.C and fb.Q <= self.Q and fb.M <= self.Mpad and fb.dev.type == "cuda"

    def bind(self, fb):
        fb.runner = self if self.usable(fb) else None
        return fb

    def _load(self, fb):
        sh = self.shadow
        if sh is None:
            self._alloc(fb)
            sh = self.shadow
        for n in ("Rt", "tbc", "fxy", "D", "lim", "sxy", "pose", "Pc"):
            getattr(sh, n).copy_(getattr(fb, n))
        sh.Pw.zero_(); sh.cnt.zero_()
        sh.Pw[:, : fb.Q, : fb.M].copy_(fb.Pw); sh.cnt[:, : fb.Q].copy_(fb.cnt)

    def score(self, fb, th):
        K = th.shape[0]
        Kb = next((b for b in self.BUCKETS if K <= b), None)
        if Kb is None:
            self.fallbacks += 1
            fb.runner = None
            try:
                return fb.score(th)
            finally:
                fb.runner = self
        self._load(fb)
        g = self.graphs.get(Kb)
        if g is None:
            th_in = torch.zeros(Kb, 5, dtype=torch.float64, device=self.dev)
            th_in[:K] = th
            # calibrate the entry capacity eagerly on this tick, then capture
            self.shadow.runner = None
            self.shadow.score(th_in)
            self.shadow.score(th_in, cap=1 << 16)
            need = int(self.shadow._tot)
            cap = max(1 << 16, int(2.0 * need))
            cap = (cap + 1023) // 1024 * 1024
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2):
                    self.shadow.score(th_in, cap=cap)
            torch.cuda.current_stream().wait_stream(st)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self.shadow.score(th_in, cap=cap)
                tot_t = self.shadow._tot
                ovf_t = self.shadow._ovf
            g = self.graphs[Kb] = {"graph": graph, "th": th_in, "out": out, "tot": tot_t, "ovf": ovf_t, "cap": cap}
        g["th"].zero_()
        g["th"][:K] = th
        g["graph"].replay()
        if int(g["tot"]) > g["cap"]:                       # crossing buffer overflow: eager fallback for this call
            self.fallbacks += 1
            fb.runner = None
            try:
                return fb.score(th)
            finally:
                fb.runner = self
        fb._ovf = g["ovf"].clone()
        return g["out"][:K].clone()


# ------------------------------------------------------------------------------------------------ measurer
class FastMeasurer(AttMeasurer):
    """Drop-in for `AttMeasurer.measure_att` on the GPU. `items[cam]["prob"]` may be a numpy map or a torch tensor (any float/uint8, 0..1 or 0..255
    given by `prob_scale`)."""

    def __init__(self, ctx, polys, cfg: AttFitConfig | None = None, device="cuda", edge_pad: float = 0.5, rounds: int = 2, pat_levels: int = 3, workers: int = 1, graph: bool = False, prune: float = 0.0):
        super().__init__(ctx, polys, cfg)
        self.device, self.edge_pad, self.rounds, self.pat_levels = torch.device(device), float(edge_pad), int(rounds), int(pat_levels)
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(workers) if workers > 1 else None
        self.prune = float(prune)      # > 0: after the first pattern level, starts whose J is more than `prune` nats above the best are dropped (speed)
        self.static = {}
        for cam, cm in ctx.cams.items():
            lim = L._view_limits(cm)
            self.static[cam] = {
                "RbcT": torch.tensor(cm.T_base_cam.R.T, dtype=torch.float64, device=self.device),
                "tbc": torch.tensor(cm.T_base_cam.t, dtype=torch.float64, device=self.device),
                "intr": torch.tensor([cm.K[0, 0], cm.K[1, 1], cm.K[0, 2], cm.K[1, 2]], dtype=torch.float64, device=self.device),
                "D": torch.tensor(([*cm.D.tolist(), 0, 0, 0, 0, 0])[:5], dtype=torch.float64, device=self.device),
                "lim": torch.tensor(lim, dtype=torch.float64, device=self.device),
                "sxy": torch.tensor([LABEL_W / cm.width, LABEL_H / cm.height], dtype=torch.float64, device=self.device)}
        self.last_timing = {}
        self.graph = GraphRunner(self) if (graph and self.device.type == 'cuda') else None

    # ---------------------------------------------------------------- evidence on the device
    def _evidence(self, sc: AttScene, prob):
        cfg = self.acfg.seg
        if isinstance(prob, torch.Tensor):
            p = prob.to(self.device).float()
            if p.max() > 1.5:
                p = p / 255.0
            p = p.clamp(cfg.p_clip, 1 - cfg.p_clip)
            e = torch.log(p / (1 - p))
            e = e.masked_fill(torch.from_numpy(sc.ignore).to(self.device), 0.0)
            return torch.nn.functional.interpolate(e[None, None], scale_factor=cfg.ss, mode="bilinear", align_corners=False)[0, 0]
        e = evidence_image(prob, sc.ignore, cfg.p_clip, cfg.ss)
        return torch.from_numpy(e).to(self.device)

    # ---------------------------------------------------------------- objective
    def _J(self, fb: FastBatch, th: torch.Tensor, mu, sd, free, T) -> torch.Tensor:
        th = torch.where(free[None], th, mu[None])
        z = (th - mu[None]) / sd[None]
        pri = 0.5 * (z * z * free[None]).sum(1)
        return -fb.score(th) / T + pri

    def _fit_att(self, items, ref, att_prior, fit_att):
        a, cfg, dev = self.acfg, self.acfg.seg, self.device
        tm = {}
        t0 = time.perf_counter()
        scenes, Es, reasons = {}, [], []

        def build(cam):
            it = items[cam]
            sc = AttScene(self, cam, it["pose0"], it.get("s_init"), it.get("attitude"))
            if not sc.has_road:
                return cam, sc, "no_border_near", 0
            n0 = int(sc.mask(0.0, 0.0).sum())
            if n0 < cfg.min_poly_px * cfg.ss ** 2:
                return cam, sc, "few_pixels", n0
            return cam, sc, "", n0

        built = list(self.pool.map(build, list(items))) if self.pool else [build(c) for c in items]
        npts = {}
        for cam, sc, why, n0 in built:
            if why:
                reasons.append(why)
                continue
            scenes[cam] = sc
            npts[cam] = n0
            Es.append(self._evidence(sc, items[cam]["prob"]))
        if not scenes:
            return None, None, {"reason": reasons[0] if reasons else "no_border_near", "n_cams": 0}
        sc_ref = scenes.get(ref) or next(iter(scenes.values()))
        sl = list(scenes.values())
        fb = FastBatch(self, sl, Es)
        fb.runner = None
        if self.graph is not None:
            self.graph.bind(fb)
        tm["setup_ms"] = 1e3 * (time.perf_counter() - t0)
        amu, asd = (np.zeros(3), np.array([a.sd0_att[0] * DEG, a.sd0_att[1] * DEG, a.sd0_att[2]])) if att_prior is None else att_prior
        mu_np = np.concatenate([[0.0, 0.0], np.asarray(amu, float)])
        sd_np = np.concatenate([[a.sd_dn, a.sd_dpsi_deg * DEG], np.asarray(asd, float)])
        free_np = np.array([True, True, fit_att, fit_att, fit_att])
        mu, sd, free = (torch.tensor(v, dtype=torch.float64, device=dev) for v in (mu_np, sd_np, free_np))
        free = free.bool()
        T = a.temp
        step0, hfd = _scales(a)
        n_evals = 0
        batches = []

        def J(th_np):
            nonlocal n_evals
            th = torch.as_tensor(np.atleast_2d(th_np), dtype=torch.float64, device=dev)
            n_evals += th.shape[0]
            batches.append(th.shape[0])
            return self._J(fb, th, mu, sd, free, T).cpu().numpy()

        # stage A: coarse (dn, dpsi) grid at the prior attitude
        t1 = time.perf_counter()
        dns = np.arange(-cfg.dn_range_m, cfg.dn_range_m + 1e-9, cfg.coarse_dn)
        dps = np.deg2rad(np.arange(-cfg.dpsi_range_deg, cfg.dpsi_range_deg + 1e-9, cfg.coarse_dpsi_deg))
        G = np.tile(mu_np, (len(dns) * len(dps), 1))
        G[:, 0], G[:, 1] = np.repeat(dns, len(dps)), np.tile(dps, len(dns))
        JG = J(G)
        order = np.argsort(JG, kind="stable")
        starts = []
        for i in order:
            th = G[i]
            if all(abs(th[0] - t2[0]) >= 0.5 or abs(th[1] - t2[1]) >= DEG for t2 in starts):
                starts.append(th.copy())
            if len(starts) >= max(a.n_starts, 1):
                break
        grid_best = starts[0].copy()
        tm["grid_ms"] = 1e3 * (time.perf_counter() - t1)
        # stage B: batched pattern search over the starts; each round polls the full 51-point stencil (axes + all pair diagonals) of every start at once
        t2_ = time.perf_counter()
        S = np.array(starts)
        fS = J(S)
        idx = np.flatnonzero(free_np)
        k = len(idx)
        sten = [np.zeros(5)]
        for ai in range(k):
            for sg in (1.0, -1.0):
                e = np.zeros(5); e[idx[ai]] = sg
                sten.append(e)
        for ai in range(k):
            for bj in range(ai + 1, k):
                for sa in (1.0, -1.0):
                    for sb in (1.0, -1.0):
                        e = np.zeros(5); e[idx[ai]] = sa; e[idx[bj]] = sb
                        sten.append(e)
        sten = np.array(sten)
        for lv in range(self.pat_levels):
            st = step0 / (2.0 ** lv)
            for _ in range(self.rounds):
                cand = (S[:, None, :] + sten[None] * st[None, None, :]).reshape(-1, 5)
                Jc = J(cand).reshape(len(S), len(sten))
                bb = Jc.argmin(1)
                imp = Jc[np.arange(len(S)), bb] < fS - 1e-12
                if not imp.any():
                    break
                for s_ in np.flatnonzero(imp):
                    S[s_] = S[s_] + sten[bb[s_]] * st
                    fS[s_] = Jc[s_, bb[s_]]
            if lv == 0 and self.prune > 0 and len(S) > 1:
                keep = fS <= fS.min() + self.prune
                S, fS = S[keep], fS[keep]
        b = int(np.argmin(fS))
        th, f = S[b].copy(), float(fS[b])
        tm["pattern_ms"] = 1e3 * (time.perf_counter() - t2_)
        # stage C: Newton + Laplace (51-point batches)
        t3 = time.perf_counter()
        k = len(idx)
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        h = hfd
        ok, H = False, None

        def newton_batch(c):
            pts = [c]
            for i in idx:
                e = np.zeros(5); e[i] = h[i]
                pts += [c + e, c - e]
            for ai, bj in pairs:
                ei = np.zeros(5); ei[idx[ai]] = h[idx[ai]]
                ej = np.zeros(5); ej[idx[bj]] = h[idx[bj]]
                pts += [c + ei + ej, c + ei - ej, c - ei + ej, c - ei - ej]
            v = J(np.array(pts))
            f0 = v[0]
            fp, fm = v[1:1 + 2 * k:2], v[2:2 + 2 * k:2]
            g = np.array([(fp[q] - fm[q]) / (2 * h[idx[q]]) for q in range(k)])
            Hh = np.zeros((k, k))
            for q in range(k):
                Hh[q, q] = (fp[q] + fm[q] - 2 * f0) / h[idx[q]] ** 2
            o = 1 + 2 * k
            for n_, (ai, bj) in enumerate(pairs):
                vv = v[o + 4 * n_:o + 4 * n_ + 4]
                Hh[ai, bj] = Hh[bj, ai] = (vv[0] - vv[1] - vv[2] + vv[3]) / (4 * h[idx[ai]] * h[idx[bj]])
            return f0, g, Hh

        refined = False
        for _ in range(max(a.newton_iters, 1) + 1):
            f0, g, Hh = newton_batch(th)
            H = Hh
            ok = bool(np.linalg.eigvalsh(Hh).min() > 1e-9)
            if not ok:
                f = f0
                if refined:
                    break
                # Hessian not positive definite at a coarsely converged point: refine with finer pattern levels once, then retry
                refined = True
                for lv in range(self.pat_levels, self.pat_levels + 2):
                    st = step0 / (2.0 ** lv)
                    for _r in range(self.rounds):
                        cand = th[None] + sten * st[None]
                        Jc = J(cand)
                        bb = int(Jc.argmin())
                        if Jc[bb] < f - 1e-12:
                            th, f = cand[bb].copy(), float(Jc[bb])
                        else:
                            break
                continue
            d = np.clip(-np.linalg.solve(Hh, g), -h[idx], h[idx])
            c = th.copy()
            c[idx] += d
            fc = float(J(c)[0])
            if fc <= f0:
                th2, f2 = c, fc
            else:
                th2, f2 = th, f0
            moved = f2 < f0 - 1e-9
            f = f2
            th = th2
            if not moved:
                break
        tm["newton_ms"] = 1e3 * (time.perf_counter() - t3)
        q = {"cost": float(f), "n_cams": len(sl), "support": float(len(sl)), "n_evals": float(n_evals),
             "n_points": float(sum(npts.values()) / cfg.ss ** 2),
             "dn_fit": float(th[0]), "dpsi_fit": float(th[1]), "droll": float(th[2]), "dpitch": float(th[3]), "dz": float(th[4]),
             "dn_grid": float(grid_best[0]), "dpsi_grid": float(grid_best[1]), "clip_overflow": float(int(fb._ovf)), "n_batches": float(len(batches)), "n_big_batches": float(sum(b > 100 for b in batches))}
        q.update({f"t_{k_}": v for k_, v in tm.items()})
        if not ok:
            q["reason"] = "flat"
            return sc_ref, None, q
        C = np.linalg.inv(H)
        sdv = np.sqrt(np.clip(np.diag(C), 0, None))
        full = np.full(5, np.nan)
        full[idx] = sdv
        q["sig_raw_n"], q["sig_raw_dpsi"] = float(full[0]), float(full[1])
        q["sd_roll"], q["sd_pitch"], q["sd_z"] = (float(v) for v in full[2:])
        return sc_ref, th, q
