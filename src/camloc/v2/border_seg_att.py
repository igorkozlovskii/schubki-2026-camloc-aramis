"""S3.2c: joint six-camera border fit of (dn, dpsi) together with the car's attitude (droll, dpitch, dz), no ground truth.

Builds on `border_seg` (imported, not edited): the same road-polygon log-likelihood of the network's paved-surface probability,
    S(theta) = (1 / ss^2) * sum_{pixels in polygon(theta)} [log p - log(1 - p)]   summed over the available cameras,
but with five shared parameters per tick, theta = (dn [m], dpsi [rad], droll [rad], dpitch [rad], dz [m]); the attitude entries are
additive to the map attitude of S1.1 (`Attitude` base, if any, comes first). The estimate is the MAP of
    J(theta) = -S(theta) / T + 0.5 * sum_{free i} ((theta_i - mu_i) / sd_i)^2 ,
T = `AttFitConfig.temp` (evidence temperature: neighbouring pixels and the network's errors are strongly correlated, so the raw
log-likelihood is over-confident; T is declared, not fitted per tick). Priors (causal only):
  * pose (dn, dpsi): N(0, sd) around the start pose (the start's own uncertainty; the S1.1 perturbation is uniform +-1 m / +-1.5 deg),
  * attitude: random walk from the previous valid tick's estimate (`AttTracker`), optionally on top of an allowed-telemetry
    attitude (S1.8 model, `use_tel`): the prior mean is tel_now + (last estimate - tel at that tick),
    map attitude (delta 0, wide sd) at the start of a run or after a gap.
Search: (dn, dpsi) coarse grid at the prior attitude (robust start, 13 x 11) -> coordinate pattern search over the free parameters with
halving steps -> one damped Newton step with central finite differences at a coarse step (smooths the piecewise-constant fill) -> the
same Hessian gives the Laplace covariance; sig_raw_n / sig_raw_dpsi are its marginal widths (calibrated out of fold downstream).
Abstain reasons: no_border_near, few_pixels (as S3.2b), flat (Hessian not positive definite), error:*. There is no grid-edge abstention:
the prior and the pattern search continue past the coarse grid.
No ground truth, no `derived/`, no `refline_*`, no `sn_map_state`, no GT s (`s_init` is the caller's own estimate), no GT attitude: the
oracle variant passes an absolute `Attitude` through `items[cam]["attitude"]` from the evaluator; this module never builds one.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from camloc.geometry.transforms import wrap_angle
from camloc.v2.border_probe import BorderMeas
from camloc.v2.border_seg import SegFitConfig, SegMeasurer, SegScene, evidence_image
from camloc.v2.seg import LABEL_H, LABEL_W
from camloc.v2.seg import labels as L

PARAMS = ("dn", "dpsi", "droll", "dpitch", "dz")
DEG = math.pi / 180.0


@dataclass
class AttFitConfig:
    seg: SegFitConfig = field(default_factory=SegFitConfig)
    temp: float = 3000.0                       # evidence temperature T (label-pixel log-likelihood units per prior nat)
    sd_dn: float = 0.6                         # m, pose prior around the start (uniform +-1 m has 0.58)
    sd_dpsi_deg: float = 0.9                   # deg (uniform +-1.5 deg has 0.87)
    sd0_att: tuple = (0.8, 0.3, 0.15)          # (droll deg, dpitch deg, dz m): start / gap attitude prior sd around the map attitude
    rw_att: tuple = (0.45, 0.12, 0.04)         # (deg, deg, m) random-walk sd per sqrt(second)
    sd0_att_map: tuple = (0.25, 0.08, 0.04)    # start / gap sd of the RESIDUAL on top of the road-attitude map (variant e): (deg, deg, m)
    sd_floor_att: tuple = (0.15, 0.05, 0.02)   # floor of the carried posterior sd (deg, deg, m): keeps the chain from locking
    reset_s: float = 3.0                       # gap after which the chain restarts from the start prior
    step0: tuple = (0.12, 0.3, 0.3, 0.12, 0.03)  # pattern-search steps (m, deg, deg, deg, m)
    n_levels: int = 5
    h_fd: tuple = (0.15, 0.4, 0.4, 0.15, 0.04)   # finite-difference step of the Newton / Hessian stage (m, deg, deg, deg, m)
    n_starts: int = 3                          # pattern searches from the best grid cells that are distinct (>= 0.5 m or >= 1 deg apart)
    newton_iters: int = 4                      # damped Newton / Hessian stages (repeated while J improves)


def _scales(cfg: AttFitConfig):
    """step / unit conversion vectors in the internal units (m, rad, rad, rad, m)."""
    u = np.array([1.0, DEG, DEG, DEG, 1.0])
    return np.array(cfg.step0) * u, np.array(cfg.h_fd) * u


# ------------------------------------------------------------------------------------------------ scene with the attitude as parameters
class AttScene(SegScene):
    def mask5(self, th) -> np.ndarray:
        dn, dpsi, dr, dp, dz = th
        canvas = np.zeros((LABEL_H * self.ss, LABEL_W * self.ss), np.uint8)
        if not self.has_road:
            return canvas
        x, y, yaw = self.pose0
        t = np.array([x + dn * self.nrm[0], y + dn * self.nrm[1], self.z_base + dz])
        R = L.pose_rotation(self.roll0 + dr, self.pitch0 + dp, yaw + dpsi)
        qs = [self._to_label(self._project(w, t, R), self.ss) for w in self.road_world]
        qs = [q for q in qs if len(q) >= 3]
        if qs:
            cv2.fillPoly(canvas, qs, 1, lineType=cv2.LINE_8, shift=4)
        return canvas


class _Objective:
    def __init__(self, scenes, Es, ss, T, mu, sd, free, prior_sd_free=True):
        self.sc, self.E, self.ss2, self.T = scenes, Es, float(ss * ss), T
        self.mu, self.sd, self.free = np.asarray(mu, float), np.asarray(sd, float), np.asarray(free, bool)
        self.cache, self.n = {}, 0

    def S(self, th) -> float:
        tot = 0.0
        for sc, E in zip(self.sc, self.E):
            m = sc.mask5(th)
            tot += cv2.mean(E, mask=m)[0] * cv2.countNonZero(m)
        return tot / self.ss2

    def __call__(self, th) -> float:
        th = np.where(self.free, th, self.mu)
        key = tuple(np.round(th, 9))
        v = self.cache.get(key)
        if v is None:
            z = (th - self.mu) / self.sd
            v = -self.S(th) / self.T + 0.5 * float((z[self.free] ** 2).sum())
            self.cache[key] = v
            self.n += 1
        return v


def _pattern(J, th, free, step0, levels):
    th = th.copy()
    f = J(th)
    idx = np.flatnonzero(free)
    for lv in range(levels):
        st = step0 / (2.0 ** lv)
        improved = True
        rounds = 0
        while improved and rounds < 6:
            improved, rounds = False, rounds + 1
            for i in idx:
                best, bv = None, f
                for sgn in (1.0, -1.0):
                    c = th.copy()
                    c[i] += sgn * st[i]
                    v = J(c)
                    if v < bv - 1e-12:
                        best, bv = c, v
                if best is not None:
                    th, f, improved = best, bv, True
    return th, f


def _newton_laplace(J, th, free, h):
    """Central-difference gradient and Hessian of J over the free parameters at th; returns (theta_new, J_new, H, ok)."""
    idx = np.flatnonzero(free)
    k = len(idx)
    f0 = J(th)
    g, H = np.zeros(k), np.zeros((k, k))
    fp, fm = np.zeros(k), np.zeros(k)
    for a, i in enumerate(idx):
        e = np.zeros(len(th)); e[i] = h[i]
        fp[a], fm[a] = J(th + e), J(th - e)
        g[a] = (fp[a] - fm[a]) / (2 * h[i])
        H[a, a] = (fp[a] + fm[a] - 2 * f0) / h[i] ** 2
    for a in range(k):
        for b in range(a + 1, k):
            ei = np.zeros(len(th)); ei[idx[a]] = h[idx[a]]
            ej = np.zeros(len(th)); ej[idx[b]] = h[idx[b]]
            H[a, b] = H[b, a] = (J(th + ei + ej) - J(th + ei - ej) - J(th - ei + ej) + J(th - ei - ej)) / (4 * h[idx[a]] * h[idx[b]])
    w = np.linalg.eigvalsh(H)
    ok = bool(w.min() > 1e-9)
    thn, fn = th, f0
    if ok:
        d = -np.linalg.solve(H, g)
        d = np.clip(d, -h[idx], h[idx])
        c = th.copy()
        c[idx] += d
        v = J(c)
        if v <= f0:
            thn, fn = c, v
    return thn, fn, H, ok, idx


# ------------------------------------------------------------------------------------------------ causal attitude chain
class AttTracker:
    """Carries the attitude estimate (droll, dpitch, dz relative to the map attitude) from tick to tick, causally.
    prior(t_ns, tel) -> (mu, sd) in internal units; update(t_ns, est, sd_post, tel) after a valid measurement."""

    def __init__(self, cfg: AttFitConfig, use_tel: bool, sd0=None):
        """use_tel: the `tel` argument is a base attitude (telemetry model or road-attitude map) the chain adds its residual to.
        sd0: start / gap sd (deg, deg, m) of the residual; default cfg.sd0_att."""
        self.cfg, self.use_tel = cfg, use_tel
        self.sd0 = tuple(cfg.sd0_att if sd0 is None else sd0)
        self.t = None
        self.resid = np.zeros(3)
        self.sd = self._sd0()

    def _sd0(self):
        return np.array([self.sd0[0] * DEG, self.sd0[1] * DEG, self.sd0[2]])

    def prior(self, t_ns: int, tel=None):
        c = self.cfg
        base = np.zeros(3) if (tel is None or not self.use_tel) else np.asarray(tel, float)
        if self.t is None or (t_ns - self.t) * 1e-9 > c.reset_s or (t_ns - self.t) <= 0:
            return base, self._sd0()
        dt = (t_ns - self.t) * 1e-9
        rw = np.array([c.rw_att[0] * DEG, c.rw_att[1] * DEG, c.rw_att[2]])
        sd = np.sqrt(np.minimum(self.sd, self._sd0()) ** 2 + rw ** 2 * dt)
        return base + self.resid, np.minimum(sd, self._sd0())

    def update(self, t_ns: int, est, sd_post, tel=None):
        c = self.cfg
        base = np.zeros(3) if (tel is None or not self.use_tel) else np.asarray(tel, float)
        self.t = int(t_ns)
        self.resid = np.asarray(est, float) - base
        fl = np.array([c.sd_floor_att[0] * DEG, c.sd_floor_att[1] * DEG, c.sd_floor_att[2]])
        self.sd = np.maximum(np.asarray(sd_post, float), fl)


# ------------------------------------------------------------------------------------------------ measurer
class AttMeasurer(SegMeasurer):
    def __init__(self, ctx, polys, cfg: AttFitConfig | None = None):
        cfg = cfg or AttFitConfig()
        super().__init__(ctx, polys, cfg.seg)
        self.acfg = cfg

    def measure_att(self, items: dict, ref: str | None = None, att_prior=None, fit_att: bool = True) -> BorderMeas:
        """items as `SegMeasurer.measure_joint` (camera -> prob, pose0, s_init, optional absolute/additive base `attitude`).
        att_prior = (mu (3,), sd (3,)) for (droll, dpitch, dz) in internal units, or None = start prior around the map attitude.
        fit_att=False freezes the attitude at the base (map attitude or the evaluator's oracle) and fits (dn, dpsi) only."""
        t0 = time.perf_counter()
        try:
            sc_ref, out, q = self._fit_att(items, ref, att_prior, fit_att)
        except Exception as e:      # abstain, never crash a stream
            return BorderMeas(False, f"error:{type(e).__name__}", q={"service_ms": 1e3 * (time.perf_counter() - t0)})
        q["service_ms"] = 1e3 * (time.perf_counter() - t0)
        if out is None:
            return BorderMeas(False, q.pop("reason"), q=q)
        dn, dp = out[0], out[1]
        x, y, yaw = sc_ref.pose0[0] + dn * sc_ref.nrm[0], sc_ref.pose0[1] + dn * sc_ref.nrm[1], sc_ref.pose0[2] + dp
        s, n, th, _ = self.ctx.cline.project(x, y, s_hint=sc_ref.s0)
        return BorderMeas(True, "", n=float(n), dpsi=wrap_angle(yaw - th), s=float(s), x=float(x), y=float(y), yaw=float(wrap_angle(yaw)), q=q)

    def _fit_att(self, items, ref, att_prior, fit_att):
        a, cfg = self.acfg, self.acfg.seg
        scenes, Es, reasons = {}, [], []
        for cam, it in items.items():
            sc = AttScene(self, cam, it["pose0"], it.get("s_init"), it.get("attitude"))
            if not sc.has_road:
                reasons.append("no_border_near")
                continue
            if int(sc.mask(0.0, 0.0).sum()) < cfg.min_poly_px * cfg.ss ** 2:
                reasons.append("few_pixels")
                continue
            scenes[cam] = sc
            Es.append(evidence_image(it["prob"], sc.ignore, cfg.p_clip, cfg.ss))
        if not scenes:
            return None, None, {"reason": reasons[0] if reasons else "no_border_near", "n_cams": 0}
        sc_ref = scenes.get(ref) or next(iter(scenes.values()))
        sl = list(scenes.values())
        amu, asd = (np.zeros(3), np.array([a.sd0_att[0] * DEG, a.sd0_att[1] * DEG, a.sd0_att[2]])) if att_prior is None else att_prior
        mu = np.concatenate([[0.0, 0.0], np.asarray(amu, float)])
        sd = np.concatenate([[a.sd_dn, a.sd_dpsi_deg * DEG], np.asarray(asd, float)])
        free = np.array([True, True, fit_att, fit_att, fit_att])
        J = _Objective(sl, Es, cfg.ss, a.temp, mu, sd, free)
        step0, hfd = _scales(a)
        # stage A: coarse (dn, dpsi) grid at the prior attitude
        dns = np.arange(-cfg.dn_range_m, cfg.dn_range_m + 1e-9, cfg.coarse_dn)
        dps = np.deg2rad(np.arange(-cfg.dpsi_range_deg, cfg.dpsi_range_deg + 1e-9, cfg.coarse_dpsi_deg))
        cand = []
        for dn in dns:
            for dp in dps:
                th = mu.copy(); th[0], th[1] = dn, dp
                cand.append((J(th), th))
        cand.sort(key=lambda c: c[0])
        starts = []
        for v, th in cand:
            if all(abs(th[0] - t2[0]) >= 0.5 or abs(th[1] - t2[1]) >= DEG for t2 in starts):
                starts.append(th)
            if len(starts) >= max(a.n_starts, 1):
                break
        grid_best = starts[0].copy()
        # stage B: pattern search from each start; stage C: Newton + Laplace at the best
        res = [_pattern(J, st, free, step0, a.n_levels) for st in starts]
        th, f = min(res, key=lambda r: r[1])
        for _ in range(max(a.newton_iters, 1)):
            th2, f2, H, ok, idx = _newton_laplace(J, th, free, hfd)
            moved = f2 < f - 1e-9
            th, f = th2, f2
            if not ok or not moved:
                break
        q = {"cost": float(f), "n_cams": len(sl), "support": float(len(sl)), "n_evals": float(J.n),
             "n_points": float(sum(int(sc.mask(0, 0).sum()) for sc in sl) / cfg.ss ** 2),
             "dn_fit": float(th[0]), "dpsi_fit": float(th[1]), "droll": float(th[2]), "dpitch": float(th[3]), "dz": float(th[4]),
             "dn_grid": float(grid_best[0]), "dpsi_grid": float(grid_best[1])}
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
