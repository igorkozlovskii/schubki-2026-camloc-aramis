"""Track-coordinate filter core v0 (PRG-002 S1.3): Module 1 (telemetry prediction), Module 3 (gate, update, consensus, lost mode), predicted error.

State x = [s, n, dpsi, v, a, b, vy] with covariance P (7x7):
  s, n, dpsi  track coordinates (`track-v1`): along the 1 m centreline, signed offset (left +), heading relative to the track;
  v           speed. In telemetry tiers it is a function of the telemetry (v = a * v_tel) and of the scale state; in T0 (and in telemetry
              fallback stretches) it is a random-walk state that the camera fixes correct through the s-v correlation;
  a, b        slow nuisance states: speed scale and yaw-rate bias (clamped; fixed in T0);
  vy          lateral (slip) velocity in the body frame: measured in T2 (telemetry vy), an estimated state driven by ay - omega vx in T1, unused in T0.

Prediction (brief 5.1, kappa(s) from the track frame, omega = a_w * yaw_rate + b, telemetry rows held from their own stamp, never earlier):
  ds/dt = v cos(dpsi) / (1 - n kappa),  dn/dt = v sin(dpsi),  d(dpsi)/dt = omega - kappa ds/dt
with a slip-aware body-velocity rotation when (vx, vy) are available (T2). Without telemetry: dpsi is constant (the car follows the track).

Update: 3-dof Kalman update of (s, n, dpsi) from `landmark` rows (and optionally `coarse`), 2-dof update of (n, dpsi) from `border` rows, with a chi-square
innovation gate. Measurements are consumed only with avail_ns <= t. A fix beyond the gate never acts alone:

  Consensus (healthy state): a gate-rejected fix is adopted only if >= cons_min_fixes rejected fixes within cons_window_s agree after being carried to the present
  by the filter's own path length AND they come from >= cons_min_sources distinct (camera, kind) sources (a second camera, or a coarse place within tolerance); a
  coarse place that disagrees with the cluster vetoes the adoption. One source repeating itself never adopts (the V3-R failure, brief 4.3).
  Lost mode: entered when gate-rejected fixes persist for lost_after_s without an accepted measurement, or the predicted position sigma exceeds lost_sigma_m. The
  covariance is widened; measurements are not gated but collected; lost mode ends when lost_k fixes of any single source agree (and, for the first
  lost_coarse_veto_s, agree with the coarse place if one is in the window) - the state is then set to their consensus in one step.

Output every tick: pose, track coordinates, predicted position error (m) = sqrt(P_ss + P_nn), predicted heading error (deg) = sqrt(P_psipsi), lost flag.
"""
from __future__ import annotations

import math
from bisect import bisect_right

import numpy as np

from .params import FilterParams
from .track import TrackFrame, wrap

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
        self.n_acc = 0                  # accepted fixes since start / adoption / lost exit
        self.cand: list[dict] = []      # gate-rejected landmark fixes (healthy state)
        self.cbuf: list[dict] = []      # coarse places (healthy state), for support / veto
        self.lbuf: list[dict] = []      # landmark + coarse (lost state)
        self.stats = {"accepted": 0, "rejected": 0, "adopted": 0, "lost_entered": 0, "lost_exited": 0, "border_acc": 0, "border_rej": 0, "vo_ignored": 0}
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
        p = self.p

        def g(key, default):
            v = m.get(key)
            return default if v is None or not math.isfinite(float(v)) else float(v)
        if kind == "coarse":
            return g("sig_s", p.coarse_sig_s), g("sig_n", p.coarse_sig_n), g("sig_dpsi", p.coarse_sig_psi)
        fac = 1.0
        if p.fix_w_inl > 0.0:
            ni = m.get("q_n_inl")
            ni = p.fix_n_ref if ni is None or not math.isfinite(float(ni)) else float(ni)
            fac = 1.0 + p.fix_w_inl * max(0.0, 1.0 - ni / p.fix_n_ref)
        sxy = g("sig_xy", math.nan)
        ss = g("sig_s", sxy if math.isfinite(sxy) else p.fix_sig_s * fac)
        sn = g("sig_n", sxy if math.isfinite(sxy) else p.fix_sig_n * fac)
        return ss, sn, g("sig_dpsi", g("sig_yaw", p.fix_sig_psi))

    def _R(self, sg, s: float):
        """Measurement covariance in (s, n, dpsi) from along-track / lateral / world-yaw sigmas: dpsi_meas = yaw - theta(s_meas), so an along-track error e_s adds -kappa e_s to dpsi."""
        k = self.tr.kappa(s)
        R = np.diag([sg[0] ** 2, sg[1] ** 2, sg[2] ** 2 + k * k * sg[0] ** 2])
        R[0, 2] = R[2, 0] = -k * sg[0] ** 2
        return R

    def _carry(self, c: dict) -> tuple[float, float, float]:
        return (c["z"][0] + (self.O - c["O"])) % self.tr.length, c["z"][1], c["z"][2]

    def _close(self, a, b, ts, tn, tp) -> bool:
        return abs(self.tr.ds(a[0], b[0])) <= ts and abs(a[1] - b[1]) <= tn and abs(wrap(a[2] - b[2])) <= tp

    def _dnorm(self, a, b) -> float:
        p = self.p
        return math.hypot(self.tr.ds(b[0], a[0]) / p.coarse_tol_s, (a[1] - b[1]) / p.coarse_tol_n)

    def _mean_pose(self, zs: list[tuple]) -> tuple[float, float, float]:
        s0 = zs[-1][0]
        ds = [self.tr.ds(s0, z[0]) for z in zs]
        n = [z[1] for z in zs]
        ps = [wrap(z[2] - zs[-1][2]) for z in zs]
        return (s0 + float(np.mean(ds))) % self.tr.length, float(np.mean(n)), wrap(zs[-1][2] + float(np.mean(ps)))

    def _kalman3(self, z, R) -> float:
        x, P = self.x, self.P
        nu = np.array([self.tr.ds(x[0], z[0]), z[1] - x[1], wrap(z[2] - x[2])])
        H = np.zeros((3, NS))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        S = P[:3, :3] + R
        K = P @ H.T @ np.linalg.inv(S)
        d = K @ nu
        self.x = [xi + di for xi, di in zip(x, d)]
        I_KH = np.eye(NS) - K @ H
        self.P = I_KH @ P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._clamp()
        return 0.0

    def _d2(self, z, R) -> float:
        x = self.x
        nu = np.array([self.tr.ds(x[0], z[0]), z[1] - x[1], wrap(z[2] - x[2])])
        return float(nu @ np.linalg.solve(self.P[:3, :3] + R, nu))

    def _set_pose(self, z, sig, s_for_kappa: float | None = None) -> None:
        """Hard set of (s, n, dpsi) with a fresh block covariance (consensus adoption / lost exit); v, a, b keep their values and variances."""
        self.x[0], self.x[1], self.x[2] = z[0], z[1], z[2]
        P = self.P.copy()
        P[:3, :] = 0.0
        P[:, :3] = 0.0
        P[:3, :3] = self._R(sig, z[0] if s_for_kappa is None else s_for_kappa)
        self.P = P
        self._clamp()

    # ------------------------------------------------------------------ Module 3: measurements
    def _prune(self, t_ns: int) -> None:
        w = int(self.p.cons_window_s * 1e9)
        self.cand = [c for c in self.cand if t_ns - c["t"] <= w]
        self.cbuf = [c for c in self.cbuf if t_ns - c["t"] <= w]
        wl = int(self.p.lost_window_s * 1e9)
        self.lbuf = [c for c in self.lbuf if t_ns - c["t"] <= wl]

    def _try_consensus(self, t_ns: int, newest: dict) -> bool:
        p = self.p
        ref = self._carry(newest)
        cl = [c for c in self.cand if self._close(self._carry(c), ref, p.cons_tol_s, p.cons_tol_n, p.cons_tol_psi)]
        if len(cl) < p.cons_min_fixes:
            return False
        zs = [self._carry(c) for c in cl]
        mean = self._mean_pose(zs)
        sources = {c["src"] for c in cl}
        veto = False
        pred = (self.x[0], self.x[1], self.x[2])
        for c in self.cbuf:
            zc = self._carry(c)
            d_cl, d_pr = self._dnorm(zc, mean), self._dnorm(zc, pred)
            # a coarse place counts only if it discriminates: clearly closer to the cluster than to the prediction (support) or the other way round (veto)
            if d_cl <= 1.0 and d_pr >= p.coarse_ratio * d_cl:
                sources.add(c["src"])
            elif d_pr * p.coarse_ratio <= d_cl:
                veto = True
        if veto or len(sources) < p.cons_min_sources:
            return False
        sig = self._sig({}, "landmark")
        self._set_pose(mean, sig)
        self.stats["adopted"] += 1
        self.events.append({"t": t_ns, "ev": "adopt", "n_fix": len(cl), "sources": sorted("/".join(s) for s in sources), "z": mean})
        self.cand, self.cbuf, self.lbuf = [], [], []
        self.rej_run_t0 = None
        self.n_acc = 0
        return True

    def _lost_exit(self, t_ns: int, newest: dict) -> bool:
        p = self.p
        ref = self._carry(newest)
        fixes = [c for c in self.lbuf if c["kind"] == "landmark"]
        cl = [c for c in fixes if c["src"] == newest["src"] and self._close(self._carry(c), ref, p.cons_tol_s, p.cons_tol_n, p.cons_tol_psi)]
        if len(cl) < p.lost_k:
            return False
        mean = self._mean_pose([self._carry(c) for c in cl])
        if self.lost_t0 is not None and (t_ns - self.lost_t0) * 1e-9 <= p.lost_coarse_veto_s:
            for c in self.lbuf:
                if c["kind"] == "coarse" and not self._close(self._carry(c), mean, p.coarse_tol_s, p.coarse_tol_n, 3.0):
                    return False
        self._set_pose(mean, self._sig({}, "landmark"))
        self.lost = False
        self.stats["lost_exited"] += 1
        self.events.append({"t": t_ns, "ev": "lost_exit", "n_fix": len(cl), "z": mean})
        self.lost_t0 = None
        self.cand, self.cbuf, self.lbuf = [], [], []
        self.rej_run_t0 = None
        self.n_acc = 0
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

    def _consume(self, m: dict, t_ns: int, info: dict) -> None:
        p = self.p
        kind = m["kind"]
        stamp = int(m["stamp_ns"])
        info["consumed"].append(stamp)
        if kind == "vo":
            self.stats["vo_ignored"] += 1
            return
        if kind == "border":
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
            lim = p.border_gate if has_psi else 10.83
            info["d2"].append(d2)
            if d2 > lim:
                self.stats["border_rej"] += 1
                return
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = [xi + di for xi, di in zip(self.x, K @ nu)]
            I_KH = np.eye(NS) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T
            self.P = 0.5 * (self.P + self.P.T)
            self._clamp()
            self.stats["border_acc"] += 1
            return
        # landmark / coarse: full pose measurement
        if not bool(m.get("valid", True)):
            return
        if not all(m.get(k) is not None and math.isfinite(float(m[k])) for k in ("x", "y", "yaw")):
            return
        z = list(self._project(m))
        if p.latency_comp and stamp < t_ns:
            z[0] = (z[0] + (self.O - self._O_at(stamp))) % self.tr.length
        sg = self._sig(m, kind)
        rec = {"t": stamp, "kind": kind, "src": (str(m.get("camera", "fl")), kind), "z": tuple(z), "O": self.O}
        R = self._R(sg, z[0])
        if self.lost:
            self.lbuf.append(rec)
            if kind == "landmark":
                self._lost_exit(t_ns, rec)
            return
        if kind == "coarse":
            self.cbuf.append(rec)
            if p.use_coarse_update:
                d2 = self._d2(z, R)
                info["d2"].append(d2)
                if d2 <= p.coarse_gate:
                    self._kalman3(z, R)
            return
        d2 = self._d2(z, R)
        info["d2"].append(d2)
        if d2 <= p.gate_hard:
            if d2 > p.gate_soft:
                R = R * (d2 / p.gate_soft)
            self._kalman3(z, R)
            self.stats["accepted"] += 1
            self.n_acc += 1
            info["accepted"] += 1
            self.cand, self.cbuf = [], []
            self.rej_run_t0 = None
            return
        self.stats["rejected"] += 1
        info["rejected"] += 1
        if self.rej_run_t0 is None:
            self.rej_run_t0 = t_ns
        self.cand.append(rec)
        if self._try_consensus(t_ns, rec):
            info["adopted"] += 1

    # ------------------------------------------------------------------ tick
    def step(self, t_ns: int, u_start, samples: list, meas: list[dict]) -> dict:
        """Advance to tick `t_ns`. `u_start` is the held telemetry input at the previous tick (None = no telemetry: T0 motion), `samples` the telemetry rows with
        t_prev < stamp <= t as [(stamp_ns, (vlong, vlat, wz, ax, stamp_ns))]. `meas` are meas-v1 rows (dicts) with avail_ns <= t, in consumption order."""
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
            elif sig_pos >= p.lost_sigma_m:
                self._enter_lost(t_ns, "sigma")
                info["lost_entered"] += 1
        return self.output(info)

    def output(self, info: dict | None = None) -> dict:
        x, P, tr = self.x, self.P, self.tr
        X, Y, YAW = tr.to_xy(x[0], x[1], x[2])
        sig_pos = math.sqrt(max(P[0, 0] + P[1, 1], 0.0))
        sig_head = math.degrees(math.sqrt(max(P[2, 2], 0.0)))
        # covariance of (x, y, yaw): J = d(x, y, yaw)/d(s, n, dpsi)
        th = tr.theta(x[0])
        k = tr.kappa(x[0])
        T = (math.cos(th), math.sin(th))
        N = (-math.sin(th), math.cos(th))
        J = np.array([[T[0] * (1 - x[1] * k), N[0], 0.0], [T[1] * (1 - x[1] * k), N[1], 0.0], [k, 0.0, 1.0]])
        cov = J @ P[:3, :3] @ J.T
        out = {"x": X, "y": Y, "yaw": YAW, "s": x[0], "n": x[1], "dpsi": x[2], "v": x[3], "a": x[4], "b": x[5], "vy": x[6],
               "pred_pos_err_m": sig_pos, "pred_head_err_deg": sig_head, "lost": bool(self.lost), "cov": cov.tolist(),
               "sig_s": math.sqrt(max(P[0, 0], 0.0)), "sig_n": math.sqrt(max(P[1, 1], 0.0)), "sig_psi": math.sqrt(max(P[2, 2], 0.0))}
        if info is not None:
            cons = info["consumed"]
            out.update({"accepted": info["accepted"], "rejected": info["rejected"], "adopted": info["adopted"], "lost_entered": info["lost_entered"],
                        "d2": info["d2"], "consumed_stamps": cons})
        return out
