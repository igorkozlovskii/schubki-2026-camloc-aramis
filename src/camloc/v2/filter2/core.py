"""Track-coordinate filter `filter2` (PRG-002 S2.1): the S1.3 core (copied from `filter/core.py`, which stays frozen) with the reviewer's three changes.

State x = [s, n, dpsi, v, a, b, vy] (7), prediction in track coordinates and Module-1 telemetry motion exactly as in S1.3 (see `filter/core.py`).

Measurement handling (S2.1, D-013):
  Fix likelihood   `lik = student`: robust Student-t update, R_eff = R max(1, (nu + d2)/(nu + 3)); sigma per fix from the quality fields (`q_n_inl`, `q_rms_px`); chi-square gate `gate_hard`
                   decides only whether a fix is a candidate for a large correction. The coarse place may act as a tuned soft update (`use_coarse_update`).
  Independence     A fix and a coarse place from one camera image are ONE source (one stamp, one camera). Independent evidence is: fixes from different cameras that agree, or repeated
                   fixes of one camera (>= rep_k distinct stamps spanning >= rep_span_s) whose positions follow a constant-velocity fit on THEIR OWN stamps with plausible speed.
                   A single-camera repeated cluster is a constant-velocity-consistent set and can still be a persistent wrong source: this is the limit D-013 accepts, it is not
                   "never adopted". Coarse places (distinct images) rate a cluster only through their mean (sigma/sqrt(n)), and only if the cluster and the prediction are further
                   apart than `coarse_sep_k` of that sigma: they then support or veto it.
  Large correction  healthy state: cross-camera agreement, or repeated single-camera fixes AND coarse support; applied as an inflate-then-Kalman soft update (cross-covariances kept).
                    Fresh state (fewer than fresh_n_acc accepted fixes since start / correction): repeated fixes with the short fresh thresholds, hard reset (F3).
  Lost mode        entered on rejected-fix evidence (lost_after_s, lost_after_fresh_s while fresh) or a declared long gap (no accepted fix for lost_gap_s and sigma >= lost_sigma_m);
                   never on sigma alone. While lost: wide-gated, covariance-inflated Kalman updates (R x lost_r_infl) keep the state attached; exit needs the same independent
                   evidence as a large correction (no coarse support required, a coarse veto still applies), is a soft update on the fit (no `_set_pose`) and, in T0, also sets v from the fit.
Output every tick: pose, track coordinates, predicted position error (m), predicted heading error (deg), lost flag, covariance.
"""
from __future__ import annotations

import math
from bisect import bisect_right

import numpy as np

from .params import FilterParams
from ..filter.track import TrackFrame, wrap

_H = (1e-3, 1e-3, 1e-5, 1e-3, 1e-6, 1e-8, 1e-4)
NS = 7                      # state size: s, n, dpsi, v, a, b, vy (vy used by tier T1 only)
_MAX_SUB_S = 0.05


class TrackFilter:
    def __init__(self, p: FilterParams, track: TrackFrame, first_pose: tuple, t0_ns: int, v_init: float | None = None):
        self.p, self.tr = p, track
        self.tel = p.tier != "T0"
        s, n, psi = track.from_xy(*first_pose)
        a0, b0 = (p.a0, p.b0) if self.tel else (1.0, 0.0)
        v0 = float(v_init) if (self.tel and v_init is not None) else p.v0
        self.x = [s, n, psi, v0, a0, b0, 0.0]
        sa, sb = (p.init_sig_a, p.init_sig_b) if self.tel else (0.0, 0.0)
        sv = abs(v0) * p.init_sig_a if self.tel and v_init is not None else p.init_sig_v
        svy = p.init_sig_vy if (p.tier == "T1" and p.t1_slip) else 0.0
        self.P = np.diag([p.init_sig_s ** 2, p.init_sig_n ** 2, p.init_sig_psi ** 2, sv ** 2, sa ** 2, sb ** 2, svy ** 2])
        self.t_prev = int(t0_ns)
        self.t0_ns = int(t0_ns)
        self.O = 0.0
        self._Ot = [int(t0_ns)]
        self._Ov = [0.0]
        self.lost = False
        self.lost_t0: int | None = None
        self.rej_run_t0: int | None = None
        self.last_acc_t = int(t0_ns)    # last accepted fix / correction / start
        self.n_acc = 0                  # accepted fixes since start / correction
        self.cand: list[dict] = []      # gate-rejected landmark fixes (healthy state)
        self.cbuf: list[dict] = []      # coarse places (healthy state), for support / veto
        self._nis: list[float] = []         # chi-square of recent coarse places (bias monitor)
        self._prim: dict[int, float] = {}   # primary-camera fix stamp -> innovation d2 (on-demand second camera)
        self.lbuf: list[dict] = []      # landmark fixes + coarse places (lost state; every fix, accepted by the lost update or not)
        self.stats = {"accepted": 0, "rejected": 0, "adopted": 0, "lost_entered": 0, "lost_exited": 0, "border_acc": 0, "border_rej": 0, "vo_ignored": 0, "cam2_used": 0, "cam2_skipped": 0}
        self.events: list[dict] = []

    # ------------------------------------------------------------------ Module 1: prediction
    def _rates(self, s, n, psi, a, b, v, u, tmid, vy=0.0):
        tr, p = self.tr, self.p
        k = tr.kappa(s)
        den = 1.0 - n * k
        if den < 0.2:
            den = 0.2
        if u is not None:
            vl, vt, wz, ax, ay, st = u
            if vt is None:
                vt = vy                  # T1: estimated lateral velocity
            if p.ax_extrap_s > 0.0 and ax == ax:
                dtm = (tmid - st) * 1e-9
                vl = vl + ax * (0.0 if dtm < 0.0 else (p.ax_extrap_s if dtm > p.ax_extrap_s else dtm))
            vl, vt = vl * a, vt * a
            c, sn = math.cos(psi), math.sin(psi)
            u_arc = (vl * c - vt * sn) / den             # arclength rate along the centreline
            nd = vl * sn + vt * c
            pd = p.a_w * wz + b - k * u_arc
            return u_arc / tr.lam(s), nd, pd, vl
        c, sn = math.cos(psi), math.sin(psi)
        return v * c / den / tr.lam(s), v * sn, 0.0, v

    def _f(self, x, u, dt, tmid):
        s, n, psi, v, a, b, vy = x
        sd1, nd1, pd1, _ = self._rates(s, n, psi, a, b, v, u, tmid, vy)
        sd2, nd2, pd2, vnew = self._rates(s + 0.5 * dt * sd1, n + 0.5 * dt * nd1, psi + 0.5 * dt * pd1, a, b, v, u, tmid, vy)
        vy_new = vy
        if u is not None and u[1] is None:         # T1: d(vy)/dt = ay - omega vx - vy / tau
            vy_new = vy + dt * (u[4] - (self.p.a_w * u[2] + b) * vnew - vy / self.p.vy_tau_s)
        return [s + dt * sd2, n + dt * nd2, psi + dt * pd2, vnew, a, b, vy_new], sd2 * dt

    def _sub(self, dt: float, u, t_mid_ns: float) -> None:
        if dt <= 0.0:
            return
        nsub = max(1, int(math.ceil(dt / _MAX_SUB_S)))
        h = dt / nsub
        for i in range(nsub):
            tmid = t_mid_ns - 0.5 * dt * 1e9 + (i + 0.5) * h * 1e9
            x0 = self.x
            f0, dO = self._f(x0, u, h, tmid)
            F = np.empty((NS, NS))
            for j in range(NS):
                xj = list(x0)
                xj[j] += _H[j]
                fj, _ = self._f(xj, u, h, tmid)
                F[:, j] = (np.asarray(fj) - np.asarray(f0)) / _H[j]
            self.x = f0
            self.O += dO
            p = self.p
            kap = self.tr.kappa(x0[0])
            q = [p.q_s ** 2 * h, p.q_n ** 2 * h, p.q_psi ** 2 * h + kap * kap * p.q_s ** 2 * h, 0.0, 0.0, 0.0, 0.0]
            if u is None:
                q[3] = p.q_v ** 2 * h
            if self.tel:
                q[4], q[5] = p.q_a ** 2 * h, p.q_b ** 2 * h
            if p.tier == "T1" and p.t1_slip:
                q[6] = p.q_vy ** 2 * h
            Q = np.diag(q)
            # an along-track position error moves the car along a curved reference at fixed world heading: dpsi = -kappa ds, so the s noise enters dpsi with -kappa
            Q[0, 2] = Q[2, 0] = -kap * p.q_s ** 2 * h
            self.P = F @ self.P @ F.T + Q
            self.P = 0.5 * (self.P + self.P.T)

    def _clamp(self) -> None:
        p = self.p
        x = self.x
        x[0] = x[0] % self.tr.length
        x[2] = wrap(x[2])
        if self.tel:
            x[4] = min(max(x[4], p.a_lo), p.a_hi)
            x[5] = min(max(x[5], -p.b_lim), p.b_lim)

    def _O_at(self, t_ns: int) -> float:
        i = bisect_right(self._Ot, t_ns) - 1
        if i < 0:
            return self._Ov[0]
        if i >= len(self._Ot) - 1:
            return self._Ov[-1]
        f = (t_ns - self._Ot[i]) / (self._Ot[i + 1] - self._Ot[i])
        return self._Ov[i] + f * (self._Ov[i + 1] - self._Ov[i])

    # ------------------------------------------------------------------ measurement helpers
    def _project(self, m: dict) -> tuple[float, float, float]:
        tr = self.tr
        x, y, yaw = float(m["x"]), float(m["y"]), float(m["yaw"])
        if self.lost:
            return tr.from_xy(x, y, yaw, s_hint=None)
        h = tr.from_xy(x, y, yaw, s_hint=self.x[0], window_m=150.0)
        u = tr.from_xy(x, y, yaw, s_hint=None)
        return h if abs(h[1]) <= abs(u[1]) + 1.0 else u

    def _sig(self, m: dict, kind: str) -> tuple[float, float, float]:
        """Per-measurement sigmas (s, n, dpsi). Fixes: declared sigma, else the configured default inflated by the quality fields (few inliers, large reprojection rms)."""
        p = self.p

        def g(key, default):
            v = m.get(key)
            return default if v is None or not math.isfinite(float(v)) else float(v)
        if kind == "coarse":
            f = p.coarse_w_gate
            return g("sig_s", p.coarse_sig_s) * f, g("sig_n", p.coarse_sig_n) * f, g("sig_dpsi", p.coarse_sig_psi) * f
        fac = 1.0
        if p.fix_w_inl > 0.0:
            ni = m.get("q_n_inl")
            ni = p.fix_n_ref if ni is None or not math.isfinite(float(ni)) else float(ni)
            fac *= 1.0 + p.fix_w_inl * max(0.0, 1.0 - ni / p.fix_n_ref)
        if p.fix_w_rms > 0.0:
            rm = m.get("q_rms_px")
            rm = p.fix_rms_ref if rm is None or not math.isfinite(float(rm)) else float(rm)
            fac *= 1.0 + p.fix_w_rms * max(0.0, rm / p.fix_rms_ref - 1.0)
        sxy = g("sig_xy", math.nan)
        ss = g("sig_s", sxy if math.isfinite(sxy) else p.fix_sig_s * fac)
        sn = g("sig_n", sxy if math.isfinite(sxy) else p.fix_sig_n * fac)
        return ss, sn, g("sig_dpsi", g("sig_yaw", p.fix_sig_psi))

    def _R(self, sg, s: float):
        """Measurement covariance in (s, n, dpsi): dpsi_meas = yaw - theta(s_meas), so an along-track error e_s adds -kappa e_s to dpsi."""
        k = self.tr.kappa(s)
        R = np.diag([sg[0] ** 2, sg[1] ** 2, sg[2] ** 2 + k * k * sg[0] ** 2])
        R[0, 2] = R[2, 0] = -k * sg[0] ** 2
        return R

    def _kalman3(self, z, R) -> None:
        x, P = self.x, self.P
        nu = np.array([self.tr.ds(x[0], z[0]), z[1] - x[1], wrap(z[2] - x[2])])
        H = np.zeros((3, NS))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        S = P[:3, :3] + R
        K = P @ H.T @ np.linalg.inv(S)
        self.x = [xi + di for xi, di in zip(x, K @ nu)]
        I_KH = np.eye(NS) - K @ H
        self.P = I_KH @ P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._clamp()

    def _d2(self, z, R) -> float:
        x = self.x
        nu = np.array([self.tr.ds(x[0], z[0]), z[1] - x[1], wrap(z[2] - x[2])])
        return float(nu @ np.linalg.solve(self.P[:3, :3] + R, nu))

    def _robust_R(self, R, d2: float):
        """Fix likelihood: Student-t scale inflation (default) or the S1.3 Huber inflation above gate_soft."""
        p = self.p
        if p.lik == "student":
            return R * max(1.0, (p.t_nu + d2) / (p.t_nu + 3.0))
        return R * (d2 / p.gate_soft) if d2 > p.gate_soft else R

    def _set_pose(self, z, sig) -> None:
        """Hard set of (s, n, dpsi) with a fresh block covariance. Used ONLY by the fresh-state reset path (wrong first pose); v, a, b keep their values and variances."""
        self.x[0], self.x[1], self.x[2] = z[0], z[1], z[2]
        P = self.P.copy()
        P[:3, :] = 0.0
        P[:, :3] = 0.0
        P[:3, :3] = self._R(sig, z[0])
        self.P = P
        self._clamp()

    # ------------------------------------------------------------------ independent evidence (D-013)
    def _carry_now(self, zr, stamp: int):
        """Raw fix pose at its stamp moved to the present along the filter's own path (a short latency-sized shift, not used to compare fixes with each other)."""
        d = self.O - self._O_at(stamp) if stamp < self.t_prev else 0.0
        return ((zr[0] + d) % self.tr.length, zr[1], zr[2])

    def _close(self, a, b) -> bool:
        p = self.p
        return abs(self.tr.ds(a[0], b[0])) <= p.cons_tol_s and abs(a[1] - b[1]) <= p.cons_tol_n and abs(wrap(a[2] - b[2])) <= p.cons_tol_psi

    def _cv_fit(self, recs: list[dict]):
        """Constant-velocity fit of the along-track positions of one camera's fixes on THEIR OWN stamps (independent of the filter's speed). recs sorted by stamp."""
        tr, p = self.tr, self.p
        new = recs[-1]
        tau = np.array([(c["t"] - new["t"]) * 1e-9 for c in recs])
        xs = np.array([tr.ds(new["zr"][0], c["zr"][0]) for c in recs])
        A = np.column_stack([np.ones_like(tau), tau])
        coef = np.linalg.lstsq(A, xs, rcond=None)[0]
        rms = float(np.sqrt(np.mean((xs - A @ coef) ** 2)))
        ns = np.array([c["zr"][1] for c in recs])
        ps = np.array([wrap(c["zr"][2] - new["zr"][2]) for c in recs])
        span = float(-tau[0])
        ok = (rms <= p.cons_tol_s and float(np.ptp(ns)) <= 2.0 * p.cons_tol_n and float(np.ptp(ps)) <= 2.0 * p.cons_tol_psi and p.cv_v_lo <= coef[1] <= p.cv_v_hi)
        zr = ((new["zr"][0] + coef[0]) % tr.length, float(ns.mean()), wrap(new["zr"][2] + float(ps.mean())))
        return ok, {"zr": zr, "stamp": new["t"], "v_s": float(coef[1]), "rms": rms, "span": span, "k": len(recs)}

    def _evidence(self, fixes: list[dict], fresh: bool):
        """Independent evidence for a large correction from landmark fixes: cross-camera agreement, or repeated single-camera fixes consistent with a constant-velocity fit."""
        p = self.p
        by: dict[str, list[dict]] = {}
        for c in fixes:
            by.setdefault(c["cam"], {}).setdefault(c["t"], c)
        by = {cam: sorted(d.values(), key=lambda c: c["t"]) for cam, d in by.items()}
        if len(by) >= p.cons_min_cams:
            zs = {cam: self._carry_now(l[-1]["zr"], l[-1]["t"]) for cam, l in by.items()}
            cams = list(zs)
            best = None
            for i in range(len(cams)):
                grp = [cams[i]] + [c for c in cams if c != cams[i] and self._close(zs[c], zs[cams[i]])]
                if len(grp) >= p.cons_min_cams and (best is None or len(grp) > len(best)):
                    best = grp
            if best is not None:
                z = self._mean_pose([zs[c] for c in best])
                return {"kind": "cross", "z": z, "k": len(best), "cams": sorted(best), "stamps": {l[-1]["t"] for c, l in by.items() if c in best}, "v_s": None}
        k_need, span_need = (p.fresh_k, p.fresh_span_s) if fresh else (p.rep_k, p.rep_span_s)
        for cam, l in sorted(by.items(), key=lambda kv: -len(kv[1])):
            drop, step = 0, max(1, len(l) // 8)
            while len(l) - drop >= k_need:
                sub = l[drop:]
                if (sub[-1]["t"] - sub[0]["t"]) * 1e-9 < span_need:
                    break
                ok, f = self._fit_cached(sub)
                if ok:
                    return {"kind": "rep", "z": self._carry_now(f["zr"], f["stamp"]), "k": len(sub), "cams": [cam], "stamps": {c["t"] for c in sub}, "v_s": f["v_s"], "rms": f["rms"], "span": f["span"]}
                drop += step
        return None

    def _fit_cached(self, sub):
        return self._cv_fit(sub)

    def _mean_pose(self, zs: list[tuple]) -> tuple[float, float, float]:
        s0 = zs[-1][0]
        ds = [self.tr.ds(s0, z[0]) for z in zs]
        ps = [wrap(z[2] - zs[-1][2]) for z in zs]
        return (s0 + float(np.mean(ds))) % self.tr.length, float(np.mean([z[1] for z in zs])), wrap(zs[-1][2] + float(np.mean(ps)))

    def _coarse_vote(self, ev: dict, coarse: list[dict]) -> int:
        """+1 coarse places support the cluster, -1 veto it, 0 cannot tell. Only distinct images other than the cluster's count; the vote uses their MEAN (sigma / sqrt(n)) and only
        when the cluster and the prediction are further apart than coarse_sep_k of that sigma (the ~8 m coarse noise cannot tell nearer hypotheses apart)."""
        p, tr = self.p, self.tr
        d = {}
        for c in coarse:
            if c["t"] not in ev["stamps"]:
                d.setdefault(c["t"], c)
        if len(d) < p.coarse_n_min:
            return 0
        zs = [self._carry_now(c["zr"], c["t"]) for c in d.values()]
        zc = self._mean_pose(zs)
        n = len(zs)
        ss, sn = p.coarse_sig_s / math.sqrt(n), p.coarse_sig_n / math.sqrt(n)
        pred, cl = (self.x[0], self.x[1]), ev["z"]

        def dist2(a, b):
            return (tr.ds(b[0], a[0]) / ss) ** 2 + ((a[1] - b[1]) / sn) ** 2
        if math.sqrt(dist2(pred, cl)) < p.coarse_sep_k:
            return 0
        dd = dist2(zc, pred) - dist2(zc, cl)
        return 1 if dd >= p.coarse_dchi else (-1 if dd <= -p.coarse_dchi else 0)

    def _v_update(self, ev: dict) -> None:
        """T0 (speed is a state): set v from the constant-velocity fit of the fixes' own stamps (1-dof Kalman update, sigma from the fit)."""
        if self.tel or ev.get("v_s") is None:
            return
        s, n, ps = ev["z"]
        k = self.tr.kappa(s)
        den = max(1.0 - n * k, 0.2)
        vm = ev["v_s"] * den * self.tr.lam(s) / max(math.cos(ps), 0.2)
        sig = math.sqrt(self.p.exit_sig_v ** 2 + (math.sqrt(12.0 / max(ev["k"], 2)) * max(ev["rms"], self.p.fix_sig_s) / max(ev["span"], 0.3)) ** 2)
        P = self.P
        S = P[3, 3] + sig * sig
        K = P[:, 3] / S
        self.x = [xi + ki * (vm - self.x[3]) for xi, ki in zip(self.x, K)]
        I_KH = np.eye(NS)
        I_KH[:, 3] -= K
        self.P = I_KH @ P @ I_KH.T + np.outer(K, K) * sig * sig
        self.P = 0.5 * (self.P + self.P.T)
        self._clamp()

    def _correct(self, t_ns: int, ev: dict, how: str, reset_buffers: bool = True) -> None:
        """Apply a large correction: `hard` = fresh-state reset; `soft` = inflate the pose block and run a Kalman update on the evidence (cross-covariances kept)."""
        p = self.p
        sig = self._sig({}, "landmark")
        if how == "hard":
            self._set_pose(ev["z"], sig)
        else:
            P = self.P.copy()
            for i, w in enumerate((p.lost_widen_s, p.lost_widen_n, p.lost_widen_psi)):
                P[i, i] = max(P[i, i], w ** 2)
            self.P = P
            f = min(max(ev["k"] / p.exit_r_div, 1.0), 4.0)
            R = self._R(sig, ev["z"][0]) / f
            self._kalman3(ev["z"], R)
        self._v_update(ev)
        self.events.append({"t": t_ns, "ev": "lost_exit" if self.lost else "adopt", "how": how, "kind": ev["kind"], "n_fix": ev["k"], "cams": ev["cams"], "z": ev["z"]})
        if self.lost:
            self.stats["lost_exited"] += 1
        else:
            self.stats["adopted"] += 1
        self.lost = False
        self.lost_t0 = None
        self.cand, self.cbuf, self.lbuf = [], [], []
        self.rej_run_t0 = None
        self.n_acc = 0
        self.last_acc_t = t_ns
        self._nis = []

    # ------------------------------------------------------------------ Module 3: measurements
    def _prune(self, t_ns: int) -> None:
        w = int(self.p.cons_window_s * 1e9)
        self.cand = [c for c in self.cand if t_ns - c["t"] <= w]
        self.cbuf = [c for c in self.cbuf if t_ns - c["t"] <= w]
        wl = int(self.p.lost_window_s * 1e9)
        self.lbuf = [c for c in self.lbuf if t_ns - c["t"] <= wl]
        self._prim = {t: d for t, d in self._prim.items() if t_ns - t <= 5_000_000_000}

    def _try_healthy(self, t_ns: int) -> bool:
        p = self.p
        fresh = self.n_acc < p.fresh_n_acc
        ev = self._evidence(self.cand, fresh)
        if ev is None:
            return False
        vote = self._coarse_vote(ev, self.cbuf)
        if vote < 0:
            return False
        if fresh:
            self._correct(t_ns, ev, "hard")
            return True
        if ev["kind"] == "cross" or vote > 0:
            self._correct(t_ns, ev, "soft")
            return True
        return False

    def _try_exit(self, t_ns: int) -> bool:
        ev = self._evidence([c for c in self.lbuf if c["kind"] == "landmark"], False)
        if ev is None or self._coarse_vote(ev, [c for c in self.lbuf if c["kind"] == "coarse"]) < 0:
            return False
        self._correct(t_ns, ev, "soft")
        return True

    def _enter_lost(self, t_ns: int, why: str) -> None:
        p = self.p
        self.lost = True
        self.lost_t0 = t_ns
        self.stats["lost_entered"] += 1
        self.events.append({"t": t_ns, "ev": "lost_enter", "why": why})
        P = self.P.copy()
        for i, w in enumerate((p.lost_widen_s, p.lost_widen_n, p.lost_widen_psi)):
            P[i, i] = max(P[i, i], w ** 2)
        self.P = P
        self.lbuf = [dict(c) for c in self.cand] + [dict(c) for c in self.cbuf]
        self.cand, self.cbuf = [], []

    def _demanded(self, stamp: int) -> bool:
        """On-demand second camera: run it when lost, when the primary camera has no fix near this stamp, or when the primary fix was gate-rejected (it proposes a large correction)."""
        if self.lost:
            return True
        near = [d2 for t, d2 in self._prim.items() if abs(t - stamp) <= 60_000_000]
        return (not near) or max(near) > self.p.gate_hard

    def _border(self, m: dict, info: dict) -> None:
        p = self.p
        if not p.use_border or self.lost:
            return
        nz, pz = m.get("n"), m.get("dpsi")
        zn = float(nz)
        sn = float(m["sig_n"]) if m.get("sig_n") is not None and math.isfinite(float(m["sig_n"])) else p.border_sig_n
        has_psi = pz is not None and math.isfinite(float(pz))
        sp = float(m["sig_dpsi"]) if m.get("sig_dpsi") is not None and math.isfinite(float(m["sig_dpsi"])) else p.border_sig_psi
        idx = [1, 2] if has_psi else [1]
        z = np.array([zn, float(pz)][:len(idx)])
        Rm = np.diag([sn ** 2, sp ** 2][:len(idx)])
        H = np.zeros((len(idx), NS))
        for r, i in enumerate(idx):
            H[r, i] = 1.0
        nu = z - H @ np.asarray(self.x)
        if has_psi:
            nu[1] = wrap(nu[1])
        S = H @ self.P @ H.T + Rm
        d2 = float(nu @ np.linalg.solve(S, nu))
        info["d2"].append(d2)
        if d2 > (p.border_gate if has_psi else 10.83):
            self.stats["border_rej"] += 1
            return
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = [xi + di for xi, di in zip(self.x, K @ nu)]
        I_KH = np.eye(NS) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._clamp()
        self.stats["border_acc"] += 1

    def _consume(self, m: dict, t_ns: int, info: dict) -> None:
        p = self.p
        kind = m["kind"]
        stamp = int(m["stamp_ns"])
        info["consumed"].append(stamp)
        if kind == "vo":
            self.stats["vo_ignored"] += 1
            return
        if kind == "border":
            self._border(m, info)
            return
        if not bool(m.get("valid", True)):
            return
        if not all(m.get(k) is not None and math.isfinite(float(m[k])) for k in ("x", "y", "yaw")):
            return
        cam = str(m.get("camera", "fl"))
        if kind == "landmark" and p.on_demand_cam and cam == p.on_demand_cam:
            if self._demanded(stamp):
                self.stats["cam2_used"] += 1
            else:
                self.stats["cam2_skipped"] += 1
                return
        zr = self._project(m)
        z = list(zr)
        if p.latency_comp and stamp < t_ns:
            z[0] = (z[0] + (self.O - self._O_at(stamp))) % self.tr.length
        sg = self._sig(m, kind)
        if kind == "landmark" and p.on_demand_cam and cam == p.on_demand_cam and p.cam2_sig_scale != 1.0:
            sg = (sg[0] * p.cam2_sig_scale, sg[1] * p.cam2_sig_scale, sg[2] * p.cam2_sig_scale)
        rec = {"t": stamp, "kind": kind, "cam": str(m.get("camera", "fl")), "zr": tuple(zr)}
        R = self._R(sg, z[0])
        if self.lost:
            self.lbuf.append(rec)
            if kind == "coarse" and (p.use_coarse_update or p.coarse_lost_update):
                Rc = R * p.coarse_lost_r if p.coarse_lost_update else R
                d2 = self._d2(z, Rc)
                info["d2"].append(d2)
                if d2 <= p.lost_gate:
                    self._kalman3(z, self._robust_R(Rc, d2))
            if kind == "landmark":
                Rl = R * p.lost_r_infl
                d2 = self._d2(z, Rl)
                info["d2"].append(d2)
                if d2 <= p.lost_gate:
                    self._kalman3(z, self._robust_R(Rl, d2))
                self._try_exit(t_ns)
            return
        if kind == "coarse":
            self.cbuf.append(rec)
            if p.use_coarse_update or p.coarse_evidence or p.nis_k > 0:
                d2 = self._d2(z, R)
                info["d2"].append(d2)
                if p.nis_k > 0:
                    self._nis = (self._nis + [d2])[-p.nis_k:]
                if p.use_coarse_update and d2 <= p.coarse_gate:
                    self._kalman3(z, self._robust_R(R, d2))
                if p.coarse_evidence:
                    if d2 > p.coarse_lost_gate:
                        if self.rej_run_t0 is None:
                            self.rej_run_t0 = t_ns
                    elif d2 <= p.coarse_gate:
                        self.rej_run_t0 = None
            return
        d2 = self._d2(z, R)
        info["d2"].append(d2)
        if kind == "landmark" and cam == p.primary_cam:
            self._prim[stamp] = d2
        if d2 <= p.gate_hard:
            self._kalman3(z, self._robust_R(R, d2))
            self.stats["accepted"] += 1
            self.n_acc += 1
            self.last_acc_t = t_ns
            self._nis = []
            info["accepted"] += 1
            self.cand, self.cbuf = [], []
            self.rej_run_t0 = None
            return
        self.stats["rejected"] += 1
        info["rejected"] += 1
        if self.rej_run_t0 is None:
            self.rej_run_t0 = t_ns
        self.cand.append(rec)
        if self._try_healthy(t_ns):
            info["adopted"] += 1

    # ------------------------------------------------------------------ tick
    def step(self, t_ns: int, u_start, samples: list, meas: list[dict]) -> dict:
        """Advance to tick `t_ns`; same contract as `filter.core.TrackFilter.step` (telemetry rows with stamp <= t, measurements with avail_ns <= t)."""
        t_ns = int(t_ns)
        if t_ns <= self.t_prev:
            raise ValueError("non-increasing tick")
        t0 = self.t_prev
        u = u_start
        for stamp, un in samples:
            if stamp > t_ns:
                raise ValueError("telemetry row from the future")
            self._sub((stamp - t0) * 1e-9, u, 0.5 * (stamp + t0))
            t0, u = stamp, un
        self._sub((t_ns - t0) * 1e-9, u, 0.5 * (t_ns + t0))
        self._clamp()
        self.t_prev = t_ns
        self._Ot.append(t_ns)
        self._Ov.append(self.O)
        self._prune(t_ns)
        p = self.p
        if p.n_prior_sig > 0.0 and self.P[1, 1] > p.n_prior_sig ** 2:
            P = self.P
            Rn = p.n_prior_sig ** 2
            K = P[:, 1] / (P[1, 1] + Rn)                  # pseudo-observation of n at its own value: variance shrinks, mean does not move
            I_KH = np.eye(NS)
            I_KH[:, 1] -= K
            self.P = I_KH @ P @ I_KH.T + np.outer(K, K) * Rn
            self.P = 0.5 * (self.P + self.P.T)
        info = {"accepted": 0, "rejected": 0, "adopted": 0, "d2": [], "consumed": [], "lost_entered": 0}
        for m in meas:
            if int(m["avail_ns"]) > t_ns:
                raise ValueError("measurement from the future")
            was = self.lost
            self._consume(m, t_ns, info)
            info["lost_entered"] += int((not was) and self.lost)
        if not self.lost:
            sig_pos = math.sqrt(max(self.P[0, 0] + self.P[1, 1], 0.0))
            t_lost = p.lost_after_fresh_s if self.n_acc < p.fresh_n_acc else p.lost_after_s
            if self.rej_run_t0 is not None and (t_ns - self.rej_run_t0) * 1e-9 >= t_lost:
                self._enter_lost(t_ns, "rejected_for_T")
                info["lost_entered"] += 1
            elif p.nis_k > 0 and len(self._nis) >= p.nis_k and float(np.mean(self._nis)) >= 3.0 * p.nis_thr:
                self._enter_lost(t_ns, "nis")
                info["lost_entered"] += 1
            elif sig_pos >= p.lost_sigma_m and (t_ns - self.last_acc_t) * 1e-9 >= p.lost_gap_s:
                self._enter_lost(t_ns, "gap")
                info["lost_entered"] += 1
        return self.output(info)

    def output(self, info: dict | None = None) -> dict:
        x, P, tr = self.x, self.P, self.tr
        X, Y, YAW = tr.to_xy(x[0], x[1], x[2])
        cs = self.p.cal_scale
        sig_pos = math.sqrt(max(P[0, 0] + P[1, 1], 0.0)) * cs
        sig_head = math.degrees(math.sqrt(max(P[2, 2], 0.0))) * cs
        # covariance of (x, y, yaw): J = d(x, y, yaw)/d(s, n, dpsi)
        th = tr.theta(x[0])
        k = tr.kappa(x[0])
        T = (math.cos(th), math.sin(th))
        N = (-math.sin(th), math.cos(th))
        J = np.array([[T[0] * (1 - x[1] * k), N[0], 0.0], [T[1] * (1 - x[1] * k), N[1], 0.0], [k, 0.0, 1.0]])
        cov = J @ P[:3, :3] @ J.T * (cs * cs)
        out = {"x": X, "y": Y, "yaw": YAW, "s": x[0], "n": x[1], "dpsi": x[2], "v": x[3], "a": x[4], "b": x[5], "vy": x[6],
               "pred_pos_err_m": sig_pos, "pred_head_err_deg": sig_head, "lost": bool(self.lost), "cov": cov.tolist(),
               "sig_s": math.sqrt(max(P[0, 0], 0.0)), "sig_n": math.sqrt(max(P[1, 1], 0.0)), "sig_psi": math.sqrt(max(P[2, 2], 0.0))}
        if info is not None:
            cons = info["consumed"]
            out.update({"accepted": info["accepted"], "rejected": info["rejected"], "adopted": info["adopted"], "lost_entered": info["lost_entered"],
                        "d2": info["d2"], "consumed_stamps": cons})
        return out
