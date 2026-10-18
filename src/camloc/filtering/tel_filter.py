"""S10b (E10): telemetry-assisted causal filter. State [x, y, psi, a, b]: map pose, speed scale a (clamped), yaw-rate bias b (clamped).

Motion model: dead reckoning at telemetry rate from the ALLOWED fields only (body vx, vy, yaw rate wz; or v along the heading), zero-order hold between rows (a row is
held from its own stamp on, never earlier). Absolute updates: E04 per-tick rows (PnP fix = 3-dof (x, y, psi) with chi-square hard gate, Huber inflation, inlier-dependent sigma;
retrieval pose = weak update) with the S07c/S07d persistent-rejection reset. Nuisance states are clamped (S07a scale-collapse lesson); no unclamped state exists.

Tick API as the S07c filter: delayed E04 rows (latency stress) are applied at the tick of their image and everything after is re-propagated from stored per-tick telemetry
records (nothing later than the current tick is used). A tick record may carry a `reseed` (camera-only fallback ticks: x, y, psi, P3 taken from the fallback filter).
Only imports `kin_filter.wrap` and `Meas` (S07c, read-only).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .kin_filter import Meas, wrap

A_LO, A_HI = 0.99, 1.01
B_LIM = 5e-4


@dataclass
class TelParams:
    # --- fixed constants (TRAIN-fitted, S10a DRIFT.md)
    a0: float = 1.0039
    a_w: float = 1.0052
    b0: float = 3.3e-5
    use_vy: bool = True            # body vx, vy (True) or v along the heading (False)
    online_nuisance: bool = True   # a, b as clamped states (True) or fixed at the S10a constants (False)
    use_retrieval: bool = True
    # --- process noise (per sqrt second)
    q_pos: float = 0.07
    q_psi: float = 7.5e-4
    q_a: float = 1e-4
    q_b: float = 1e-5
    sig_a0: float = 1.5e-3
    sig_b0: float = 1e-4
    # --- init
    init_sig_xy: float = 0.5
    init_sig_yaw: float = 0.02
    v_max: float = 40.0
    # --- PnP fix
    fix_sig_xy: float = 0.6
    fix_sig_yaw: float = 0.01
    fix_w_inl: float = 0.0
    fix_n_ref: float = 300.0
    gate_hard: float = 16.3
    gate_soft: float = 7.8
    use_gate: bool = True
    use_huber: bool = True
    # --- persistent-rejection reset
    use_reanchor: bool = True
    reanchor_n: int = 4
    reanchor_vmax_mult: float = 1.0
    # --- retrieval weak update
    ret_sig_lon: float = 10.0
    ret_sig_lat: float = 2.5
    ret_sig_yaw: float = 0.06
    ret_gate: float = 11.3
    ret_min_interval_s: float = 1.0
    max_lag_ticks: int = 60

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TickRec:
    t_ns: int
    t_prev_ns: int
    u_start: Optional[tuple]                    # held input (v, vx, vy, wz) at t_prev, None = no motion known
    samples: list = field(default_factory=list)  # [(stamp_ns, (v, vx, vy, wz))] with t_prev < stamp <= t
    reseed: Optional[tuple] = None              # (x, y, psi, P3) camera-only fallback tick


@dataclass
class TState:
    x: np.ndarray
    P: np.ndarray
    last_ret_t: Optional[int] = None
    rej: list = field(default_factory=list)
    n_reanchor: int = 0

    def copy(self) -> "TState":
        return TState(self.x.copy(), self.P.copy(), self.last_ret_t, list(self.rej), self.n_reanchor)


class TelFilter:
    def __init__(self, p: TelParams, first_pose: tuple, t0_ns: int):
        self.p = p
        self.t0_ns = int(t0_ns)
        x0 = np.array([first_pose[0], first_pose[1], first_pose[2], p.a0, p.b0])
        sa, sb = (p.sig_a0, p.sig_b0) if p.online_nuisance else (0.0, 0.0)
        P0 = np.diag([p.init_sig_xy ** 2, p.init_sig_xy ** 2, p.init_sig_yaw ** 2, sa ** 2, sb ** 2])
        self._s0 = TState(x0, P0)
        self.recs: list[TickRec] = []
        self.snap: list[TState] = []
        self.meas: dict[int, list[Meas]] = {}
        self.cur: Optional[TState] = None

    # ------------------------------------------------------------------ model
    def _sub(self, st: TState, dt: float, u: tuple) -> None:
        if dt <= 0.0:
            return
        p = self.p
        v, vx, vy, wz = u
        if not p.use_vy:
            vx, vy = v, 0.0
        a, b = st.x[3], st.x[4]
        om = p.a_w * wz + b
        psi = st.x[2]
        h = psi + 0.5 * om * dt
        c, s = math.cos(h), math.sin(h)
        ubx, uby = a * vx, a * vy
        wxs, wys = c * ubx - s * uby, s * ubx + c * uby          # world velocity
        F = np.eye(5)
        F[0, 2] = -wys * dt
        F[1, 2] = wxs * dt
        F[0, 3] = (c * vx - s * vy) * dt
        F[1, 3] = (s * vx + c * vy) * dt
        F[0, 4] = F[0, 2] * 0.5 * dt
        F[1, 4] = F[1, 2] * 0.5 * dt
        F[2, 4] = dt
        st.x = np.array([st.x[0] + wxs * dt, st.x[1] + wys * dt, wrap(psi + om * dt), a, b])
        qa = p.q_a ** 2 if p.online_nuisance else 0.0
        qb = p.q_b ** 2 if p.online_nuisance else 0.0
        Q = np.diag([p.q_pos ** 2 * dt, p.q_pos ** 2 * dt, p.q_psi ** 2 * dt, qa * dt, qb * dt])
        st.P = F @ st.P @ F.T + Q
        st.P = 0.5 * (st.P + st.P.T)

    def _clamp(self, st: TState) -> None:
        st.x[3] = min(max(st.x[3], A_LO), A_HI)
        st.x[4] = min(max(st.x[4], -B_LIM), B_LIM)

    def _propagate(self, st: TState, rec: TickRec) -> None:
        t = rec.t_prev_ns
        u = rec.u_start
        for stamp, un in rec.samples:
            if u is not None:
                self._sub(st, (stamp - t) * 1e-9, u)
            t, u = stamp, un
        if u is not None:
            self._sub(st, (rec.t_ns - t) * 1e-9, u)
        self._clamp(st)

    def _reanchor(self, st: TState, m: Meas, sig_xy: float) -> None:
        p = self.p
        x, y, yaw = m.z
        st.x[0], st.x[1], st.x[2] = x, y, yaw
        P = st.P.copy()
        P[:3, :] = 0.0
        P[:, :3] = 0.0
        P[0, 0] = P[1, 1] = sig_xy ** 2 + p.init_sig_xy ** 2
        P[2, 2] = p.fix_sig_yaw ** 2 + p.init_sig_yaw ** 2
        st.P = P
        st.rej = []
        st.n_reanchor += 1

    def _kalman(self, st: TState, nu: np.ndarray, S_extra_R: np.ndarray) -> float:
        H = np.zeros((3, 5))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        S = H @ st.P @ H.T + S_extra_R
        K = st.P @ H.T @ np.linalg.inv(S)
        st.x = st.x + K @ nu
        st.x[2] = wrap(st.x[2])
        self._clamp(st)
        I_KH = np.eye(5) - K @ H
        st.P = I_KH @ st.P @ I_KH.T + K @ S_extra_R @ K.T
        st.P = 0.5 * (st.P + st.P.T)
        return 0.0

    def _update(self, st: TState, m: Meas, t_key: int) -> tuple:
        p = self.p
        weak = m.kind == "retrieval"
        if weak:
            if not p.use_retrieval:
                return "ignored", float("nan")
            if st.last_ret_t is not None and (t_key - st.last_ret_t) * 1e-9 < p.ret_min_interval_s:
                return "ignored", float("nan")
        zx, zy, zyaw = m.z
        nu = np.array([zx - st.x[0], zy - st.x[1], wrap(zyaw - st.x[2])])
        if weak:
            c, s = math.cos(zyaw), math.sin(zyaw)
            Rot = np.array([[c, -s], [s, c]])
            R = np.zeros((3, 3))
            R[:2, :2] = Rot @ np.diag([p.ret_sig_lon ** 2, p.ret_sig_lat ** 2]) @ Rot.T
            R[2, 2] = p.ret_sig_yaw ** 2
            hard, soft, sig_xy = p.ret_gate, 1e9, math.sqrt(0.5 * (p.ret_sig_lon ** 2 + p.ret_sig_lat ** 2))
        else:
            fac = 1.0 + p.fix_w_inl * max(0.0, 1.0 - (m.n_inl if m.n_inl is not None else p.fix_n_ref) / p.fix_n_ref)
            sig_xy = p.fix_sig_xy * fac
            R = np.diag([sig_xy ** 2, sig_xy ** 2, p.fix_sig_yaw ** 2])
            hard, soft = p.gate_hard, p.gate_soft
        S = st.P[:3, :3] + R
        d2 = float(nu @ np.linalg.solve(S, nu))
        if p.use_gate and d2 > hard:
            if not weak:
                st.rej.append((float(nu[0]), float(nu[1]), t_key, zx, zy, zyaw))
                st.rej = st.rej[-max(p.reanchor_n, 2):]
                if p.use_reanchor and len(st.rej) >= p.reanchor_n:
                    a, b = st.rej[-2], st.rej[-1]
                    dtf = (b[2] - a[2]) * 1e-9
                    if dtf > 0 and math.hypot(b[3] - a[3], b[4] - a[4]) / dtf <= p.reanchor_vmax_mult * p.v_max:
                        self._reanchor(st, m, sig_xy)
                        return "reanchor", d2
            return "rejected", d2
        if p.use_huber and d2 > soft:
            R = R * (d2 / soft)
        self._kalman(st, nu, R)
        if weak:
            st.last_ret_t = t_key
        else:
            st.rej = []
        return "accepted", d2

    def _apply_reseed(self, st: TState, rs: tuple) -> None:
        x, y, psi, P3 = rs
        st.x[0], st.x[1], st.x[2] = x, y, psi
        P = st.P.copy()
        P[:3, :] = 0.0
        P[:, :3] = 0.0
        P[:3, :3] = np.asarray(P3)
        st.P = P
        st.rej = []

    # ------------------------------------------------------------------ tick API
    def step(self, rec: TickRec, arrived: list) -> dict:
        k = len(self.recs)
        self.recs.append(rec)
        jmin = k
        info = {"accepted": 0, "rejected": 0, "reanchor": 0, "ret_used": 0, "dropped_old": 0, "d2": [], "replay": 0}
        for m in arrived:
            if m.key > k:
                raise ValueError("measurement from the future")
            if k - m.key > self.p.max_lag_ticks:
                info["dropped_old"] += 1
                continue
            self.meas.setdefault(m.key, []).append(m)
            jmin = min(jmin, m.key)
        st = self._s0.copy() if jmin == 0 else self.snap[jmin - 1].copy()
        for j in range(jmin, k + 1):
            r = self.recs[j]
            self._propagate(st, r)
            if r.reseed is not None:
                self._apply_reseed(st, r.reseed)
            else:
                for m in sorted(self.meas.get(j, []), key=lambda mm: (mm.kind != "pnp", mm.arrive)):
                    if m.arrive > k:
                        continue
                    out, d2 = self._update(st, m, r.t_ns)
                    if any(m is a for a in arrived):
                        info["d2"].append(round(d2, 3) if d2 == d2 else d2)
                        if out == "accepted":
                            info["accepted" if m.kind == "pnp" else "ret_used"] += 1
                        elif out == "rejected":
                            info["rejected"] += 1
                        elif out == "reanchor":
                            info["reanchor"] += 1
            if j < len(self.snap):
                self.snap[j] = st.copy()
            else:
                self.snap.append(st.copy())
        info["replay"] = k - jmin
        self.cur = st
        info.update({"x": float(st.x[0]), "y": float(st.x[1]), "yaw": float(st.x[2]), "a": float(st.x[3]), "b": float(st.x[4]),
                     "cov": st.P[:3, :3].tolist(), "n_reanchor": st.n_reanchor})
        return info
