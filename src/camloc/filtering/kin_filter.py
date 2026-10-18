"""Causal kinematic filter over E04 camera-pose fixes only (S07c, E09k). No VO, no second environment.

State s = [x, y, yaw, v, w] in the map frame (base_link; v = speed along the heading, w = yaw rate; w is pinned to 0 for the CV model).
Motion: constant velocity (CV) or constant turn rate (CTR), mid-point heading integration, piecewise-constant acceleration noise
(q_acc along the heading, q_yawacc on the yaw rate) added every tick. Speed and yaw rate are estimated by the filter itself from its
accepted fixes; the only prior is a speed clamp [0, v_max] and a broad initial speed prior (from the training speed distribution).
Observations (E04 per-tick rows): PnP fix z=(x,y,yaw) with sigma growing as inliers drop (fix_w_inl); retrieval top-1 pose as a weak
measurement with along/across-track covariance and a rate limit.
Robustness: chi-square innovation gate (hard reject, soft Huber R-inflation), once-per-gap covariance inflation, and a predeclared
re-anchoring rule: reanchor_n consecutive rejected PnP fixes (persistent rejection, no innovation-spread test) reset position/yaw (and
speed from the last two fixes) to the latest fix, so the filter cannot lock onto a wrong state.
Delayed fixes (E04 arrives about one tick late): one post-update snapshot per tick; a fix is applied at the tick of its own image and
everything after it is re-propagated (nothing later than the current tick is used; output at tick a depends on data available at a).
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class KinParams:
    model: str = "ctr"             # cv | ctr
    # --- init
    init_sig_xy: float = 0.5
    init_sig_yaw: float = 0.02
    v0: float = 25.0               # a priori: train GT speed 10-90 % = 9-35 m/s
    init_sig_v: float = 10.0
    init_sig_w: float = 0.3
    v_max: float = 40.0
    w_max: float = 2.0
    # --- process
    q_acc: float = 3.0             # m/s^2 per tick
    q_yawacc: float = 0.5          # rad/s^2 per tick (CTR only)
    # --- PnP fix
    fix_sig_xy: float = 0.6
    fix_sig_yaw: float = 0.01
    fix_w_inl: float = 0.0         # sigma multiplier 1 + w * max(0, 1 - n_inl / fix_n_ref)
    fix_n_ref: float = 300.0
    gate_hard: float = 16.3        # chi2(3) 0.999; capped 21.1
    gate_soft: float = 7.8
    use_gate: bool = True
    use_huber: bool = True
    # --- gap inflation (once per gap)
    gap_s: float = 3.0
    gap_infl_m: float = 5.0
    gap_infl_v: float = 2.0
    gap_infl_yaw: float = 0.05
    # --- re-anchoring
    use_reanchor: bool = True
    reanchor_n: int = 4
    reanchor_vmax_mult: float = 1.0   # plausibility of consecutive rejected fixes: implied speed <= mult * v_max
    # --- retrieval weak update
    use_retrieval: bool = False
    ret_sig_lon: float = 10.0
    ret_sig_lat: float = 2.5
    ret_sig_yaw: float = 0.06
    ret_gate: float = 11.3
    ret_min_interval_s: float = 1.0
    # --- bookkeeping
    stale_fix_s: float = 5.0
    max_lag_ticks: int = 60

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Meas:
    key: int                       # tick of the image
    arrive: int
    kind: str                      # pnp | retrieval
    z: tuple
    n_inl: Optional[int] = None


@dataclass
class KState:
    x: np.ndarray
    P: np.ndarray
    last_fix_t: Optional[int] = None
    last_ret_t: Optional[int] = None
    infl_done: bool = False
    rej: list = field(default_factory=list)       # (innovation x, y, fix t_ns, fix x, fix y, fix yaw) of consecutive rejected PnP fixes
    n_reanchor: int = 0

    def copy(self) -> "KState":
        return KState(self.x.copy(), self.P.copy(), self.last_fix_t, self.last_ret_t, self.infl_done, list(self.rej), self.n_reanchor)


class KinFilter:
    def __init__(self, p: KinParams, first_pose: tuple, t0_ns: int):
        if p.model not in ("cv", "ctr"):
            raise ValueError("model must be cv or ctr")
        self.p = p
        x0 = np.array([first_pose[0], first_pose[1], first_pose[2], p.v0, 0.0])
        P0 = np.diag([p.init_sig_xy ** 2, p.init_sig_xy ** 2, p.init_sig_yaw ** 2, p.init_sig_v ** 2, p.init_sig_w ** 2 if p.model == "ctr" else 0.0])
        self._s0 = KState(x0, P0, last_fix_t=t0_ns)
        self.ticks: list[int] = []
        self.snap: list[KState] = []
        self.meas: dict[int, list[Meas]] = {}
        self.cur: Optional[KState] = None

    # ------------------------------------------------------------------ model
    def _predict(self, st: KState, dt: float) -> None:
        p = self.p
        x, y, yaw, v, w = st.x
        h = yaw + 0.5 * w * dt
        c, s = math.cos(h), math.sin(h)
        F = np.eye(5)
        F[0, 2] = -v * dt * s
        F[0, 3] = dt * c
        F[0, 4] = -v * dt * s * 0.5 * dt
        F[1, 2] = v * dt * c
        F[1, 3] = dt * s
        F[1, 4] = v * dt * c * 0.5 * dt
        F[2, 4] = dt
        st.x = np.array([x + v * dt * c, y + v * dt * s, wrap(yaw + w * dt), v, w])
        gA = np.array([c * dt * dt / 2, s * dt * dt / 2, 0.0, dt, 0.0])
        Q = (p.q_acc ** 2) * np.outer(gA, gA)
        if p.model == "ctr":
            gW = np.array([0.0, 0.0, dt * dt / 2, 0.0, dt])
            Q = Q + (p.q_yawacc ** 2) * np.outer(gW, gW)
        else:
            F[:, 4] = 0.0
            F[4, :] = 0.0
            st.x[4] = 0.0
        st.P = F @ st.P @ F.T + Q
        st.P = 0.5 * (st.P + st.P.T)

    def _clamp(self, st: KState) -> None:
        st.x[3] = min(max(st.x[3], 0.0), self.p.v_max)
        st.x[4] = min(max(st.x[4], -self.p.w_max), self.p.w_max) if self.p.model == "ctr" else 0.0

    def _inflate(self, st: KState) -> None:
        p = self.p
        add = np.diag([p.gap_infl_m ** 2, p.gap_infl_m ** 2, p.gap_infl_yaw ** 2, p.gap_infl_v ** 2, 0.0])
        st.P = st.P + add
        st.infl_done = True

    def _reanchor(self, st: KState, t_key: int, m: Meas, sig_xy: float) -> None:
        p = self.p
        x, y, yaw = m.z
        if len(st.rej) >= 2:
            a, b = st.rej[-2], st.rej[-1]
            dt = (b[2] - a[2]) * 1e-9
            if dt > 0:
                v = math.hypot(b[3] - a[3], b[4] - a[4]) / dt
                st.x[3] = min(max(v, 0.0), p.v_max)
        st.x[0], st.x[1], st.x[2] = x, y, yaw
        if p.model == "ctr":
            st.x[4] = 0.0
        P = np.zeros((5, 5))
        P[0, 0] = P[1, 1] = sig_xy ** 2 + p.init_sig_xy ** 2
        P[2, 2] = p.fix_sig_yaw ** 2 + p.init_sig_yaw ** 2
        P[3, 3] = 5.0 ** 2
        P[4, 4] = p.init_sig_w ** 2 if p.model == "ctr" else 0.0
        st.P = P
        st.rej = []
        st.last_fix_t = t_key
        st.infl_done = False
        st.n_reanchor += 1

    def _update(self, st: KState, m: Meas, t_key: int) -> tuple:
        p = self.p
        weak = m.kind == "retrieval"
        if weak:
            if not p.use_retrieval:
                return "ignored", float("nan")
            if st.last_ret_t is not None and (t_key - st.last_ret_t) * 1e-9 < p.ret_min_interval_s:
                return "ignored", float("nan")
        gap = (t_key - st.last_fix_t) * 1e-9 if st.last_fix_t is not None else 0.0
        if not weak and not st.infl_done and gap > p.gap_s:
            self._inflate(st)
        H = np.zeros((3, 5))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        zx, zy, zyaw = m.z
        nu = np.array([zx - st.x[0], zy - st.x[1], wrap(zyaw - st.x[2])])
        if weak:
            c, s = math.cos(zyaw), math.sin(zyaw)
            Rot = np.array([[c, -s], [s, c]])
            Rxy = Rot @ np.diag([p.ret_sig_lon ** 2, p.ret_sig_lat ** 2]) @ Rot.T
            R = np.zeros((3, 3))
            R[:2, :2] = Rxy
            R[2, 2] = p.ret_sig_yaw ** 2
            hard, soft, sig_xy = p.ret_gate, 1e9, math.sqrt(0.5 * (p.ret_sig_lon ** 2 + p.ret_sig_lat ** 2))
        else:
            fac = 1.0 + p.fix_w_inl * max(0.0, 1.0 - (m.n_inl if m.n_inl is not None else p.fix_n_ref) / p.fix_n_ref)
            sig_xy = p.fix_sig_xy * fac
            R = np.diag([sig_xy ** 2, sig_xy ** 2, p.fix_sig_yaw ** 2])
            hard, soft = p.gate_hard, p.gate_soft
        S = H @ st.P @ H.T + R
        d2 = float(nu @ np.linalg.solve(S, nu))
        if p.use_gate and d2 > hard:
            if not weak:
                st.rej.append((float(nu[0]), float(nu[1]), t_key, zx, zy, zyaw))
                st.rej = st.rej[-max(p.reanchor_n, 2):]
                if p.use_reanchor and len(st.rej) >= p.reanchor_n:
                    # persistent-rejection reset: reanchor_n consecutive rejected PnP fixes. The latest fix must be kinematically plausible
                    # relative to the previous rejected one (implied speed <= reanchor_vmax_mult * v_max); the prediction is NOT used.
                    a, b = st.rej[-2], st.rej[-1]
                    dtf = (b[2] - a[2]) * 1e-9
                    if dtf > 0 and math.hypot(b[3] - a[3], b[4] - a[4]) / dtf <= p.reanchor_vmax_mult * p.v_max:
                        self._reanchor(st, t_key, m, sig_xy)
                        return "reanchor", d2
            return "rejected", d2
        if p.use_huber and d2 > soft:
            R = R * (d2 / soft)
            S = H @ st.P @ H.T + R
        K = st.P @ H.T @ np.linalg.inv(S)
        st.x = st.x + K @ nu
        st.x[2] = wrap(st.x[2])
        self._clamp(st)
        I_KH = np.eye(5) - K @ H
        st.P = I_KH @ st.P @ I_KH.T + K @ R @ K.T
        st.P = 0.5 * (st.P + st.P.T)
        if weak:
            st.last_ret_t = t_key
        else:
            st.rej = []
            st.last_fix_t = t_key
            st.infl_done = False
        return "accepted", d2

    # ------------------------------------------------------------------ tick API
    def step(self, t_ns: int, arrived: list) -> dict:
        """Process tick k = len(ticks). `arrived`: Meas newly available at this tick (key <= k)."""
        k = len(self.ticks)
        self.ticks.append(int(t_ns))
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
            if j > 0:
                dt = (self.ticks[j] - self.ticks[j - 1]) * 1e-9
                self._predict(st, dt)
                self._clamp(st)
            for m in sorted(self.meas.get(j, []), key=lambda mm: (mm.kind != "pnp", mm.arrive)):
                if m.arrive > k:
                    continue
                out, d2 = self._update(st, m, self.ticks[j])
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
        gap = None if st.last_fix_t is None else (self.ticks[k] - st.last_fix_t) * 1e-9
        info.update({"x": float(st.x[0]), "y": float(st.x[1]), "yaw": float(st.x[2]), "v": float(st.x[3]), "w": float(st.x[4]),
                     "gap_s": gap, "cov": st.P[:3, :3].tolist(), "n_reanchor": st.n_reanchor,
                     "degraded": gap is None or gap > self.p.stale_fix_s})
        return info
