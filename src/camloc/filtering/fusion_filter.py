"""Causal, gated EKF that fuses E04 absolute fixes (and optionally weak E03 retrieval poses) with E05 relative VO motion (S07a, E08).

State  x = [X, Y, yaw, s]  (map frame, base_link; s = multiplicative scale nuisance on the VO translation, nominal 1, optional).
Propagation: body-frame increment (dx, dy, dyaw) of E05 between consecutive ticks, rotated by the current yaw, times s.
Update: absolute pose z = [x, y, yaw] with covariance R from a fix, after a robust innovation test:
    d2 = nu' S^-1 nu  (3 dof); d2 > gate_hard -> rejected; gate_soft < d2 <= gate_hard -> R inflated by d2/gate_soft (Huber-like).
Mode logic: VO invalid -> coast on the last valid increment for coast_max_s, then hold with a large process noise (labeled).
Covariance inflation: if the gap since the last accepted fix exceeds gap_s, P is inflated once before the next test (so a fix is not
rejected merely because the filter was left to drift). Consecutive rejected fixes whose innovations agree with each other
(reinit_n of them, spread <= reinit_tol_m) reset the state to the fix (the filter, not the fix, was wrong).
Delayed fixes (E04 arrives ~1 tick late): the filter keeps one snapshot per tick and re-plays from the fix's own tick when a fix
arrives, so a fix is applied where it belongs in time and everything after it is re-propagated with the stored VO increments. Nothing
with a stamp (or availability) later than the current tick is ever used; the output at tick a is a function of data available at a only.

Everything is a function of explicit parameters (FusionParams); no GT, telemetry or hidden state.
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
class FusionParams:
    # --- state / init
    use_scale: bool = True
    init_sig_xy: float = 0.5
    init_sig_yaw: float = 0.02
    init_sig_scale: float = 0.10
    # --- VO increment noise (per tick, body frame)
    q_along: float = 0.02          # fraction of |increment| along the motion direction
    q_lat: float = 0.01            # fraction of |increment| across
    q_floor_m: float = 0.02        # per tick, both axes
    q_yaw_sqrt_s: float = 0.002    # yaw random walk, rad / sqrt(s)
    q_scale_sqrt_s: float = 0.01   # scale random walk, 1 / sqrt(s)
    bridged_noise_mult: float = 3.0
    coast_max_s: float = 1.0
    coast_noise_mult: float = 3.0
    hold_q_pos_sqrt_s: float = 10.0
    # --- absolute fixes (PnP)
    fix_sig_xy: float = 0.6
    fix_sig_dkf: float = 0.0       # extra sigma per metre of distance to the nearest keyframe
    fix_w_inl: float = 0.0         # sigma multiplier (1 + w * max(0, 1 - n_inl / fix_n_ref)): few inliers -> less trusted
    fix_n_ref: float = 300.0
    fix_w_nkf: float = 0.0         # post-hoc S07a option (default 0 = frozen): sigma multiplier (1 + w * max(0, nkf_ref - n_kf_inl))
    fix_nkf_ref: float = 5.0
    weak_n_inl: float = 0.0        # post-hoc S07a option (0 = off = frozen): fixes with n_inl < weak_n_inl or n_kf_inl < weak_nkf use the tight gate weak_gate
    weak_nkf: float = 0.0
    weak_gate: float = 3.0
    fix_sig_yaw: float = 0.01
    gate_hard: float = 16.3        # chi2(3) 0.999
    gate_soft: float = 7.8         # chi2(3) 0.95
    use_gate: bool = True
    use_huber: bool = True
    reinit_n: int = 4
    reinit_tol_m: float = 3.0
    persistent_reset: bool = False  # post-hoc S07a option (default off = frozen behaviour): Huber-accepted (d2 > soft) fixes count towards the reinit streak instead of clearing it
    consec_reject_reset: int = 0   # post-hoc S07a option (0 = off = frozen): K consecutive rejected PnP fixes -> reinit to the latest fix and distrust VO
    distrust_s: float = 5.0
    distrust_mult: float = 10.0
    unlock_gate: bool = False      # post-hoc S07a option (default off = frozen behaviour): no gate until a PnP fix has been accepted (start) or after VO hold
    # --- gap inflation
    use_inflation: bool = True
    gap_s: float = 3.0
    gap_infl_m: float = 5.0
    gap_infl_yaw: float = 0.05
    # --- retrieval weak update
    use_retrieval: bool = False
    ret_sig_lon: float = 10.0      # retrieval keyframe pose error is mostly along-track (train LOO: lon 10.4 m, lat 2.5 m)
    ret_sig_lat: float = 2.5
    ret_sig_yaw: float = 0.06
    ret_gate: float = 11.3         # chi2(3) 0.99
    ret_min_interval_s: float = 1.0
    ret_lateral_only: bool = False  # post-hoc S07a option (default off = frozen behaviour): retrieval constrains lateral offset + yaw only
    # --- bookkeeping
    stale_fix_s: float = 5.0       # tracking_state 'degraded' when no accepted PnP fix for this long
    max_lag_ticks: int = 60        # a fix older than this at arrival is dropped (labeled)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Inc:
    dx: float
    dy: float
    dyaw: float
    valid: bool
    quality: str = "ok"            # ok | bridged | invalid


@dataclass
class Meas:
    """One absolute observation keyed by the tick of its image (`key`), revealed at tick `arrive`."""
    key: int
    arrive: int
    kind: str                      # pnp | retrieval
    z: tuple                       # x, y, yaw
    d_kf_m: Optional[float] = None
    n_inl: Optional[int] = None
    n_kf: Optional[int] = None


@dataclass
class State:
    x: np.ndarray
    P: np.ndarray
    t_ns: int = 0
    last_fix_t: Optional[int] = None     # tick stamp of the last accepted PnP fix
    last_ret_t: Optional[int] = None
    infl_done: bool = False
    rej: list = field(default_factory=list)   # innovations of consecutive rejected PnP fixes [(nu_x, nu_y)]
    last_inc: Optional[tuple] = None          # last valid (dx, dy, dyaw)
    coast_ticks: int = 0
    mode: str = "hold"                        # vo | coast | hold
    rej_run: int = 0                          # consecutive rejected PnP fixes since the last accepted one
    distrust_until: Optional[int] = None      # tick stamp until which VO process noise is multiplied
    locked: bool = False                      # a PnP fix has verified the estimate since start / since the last VO hold
    events: list = field(default_factory=list)

    def copy(self) -> "State":
        c = State(self.x.copy(), self.P.copy(), self.t_ns, self.last_fix_t, self.last_ret_t, self.infl_done, list(self.rej),
                  self.last_inc, self.coast_ticks, self.mode, [])
        c.locked = self.locked
        c.rej_run, c.distrust_until = self.rej_run, self.distrust_until
        return c


class FusionFilter:
    def __init__(self, params: FusionParams, first_pose: tuple, t0_ns: int):
        self.p = params
        x0 = np.array([first_pose[0], first_pose[1], first_pose[2], 1.0])
        P0 = np.diag([params.init_sig_xy ** 2, params.init_sig_xy ** 2, params.init_sig_yaw ** 2,
                      params.init_sig_scale ** 2 if params.use_scale else 1e-12])
        self.ticks: list[int] = []                 # tick stamps seen so far
        self.incs: list[Inc] = []
        self.meas: dict[int, list[Meas]] = {}      # key tick -> measurements (arrived)
        self.snap: list[State] = []                # state after processing tick k
        self._s0 = State(x0, P0, t_ns=int(t0_ns))
        self.out_events: list = []

    # ------------------------------------------------------------------ math
    def _predict(self, st: State, inc: Inc, dt: float) -> None:
        p = self.p
        x, y, psi, s = st.x
        if inc.valid:
            dx, dy, dpsi = inc.dx, inc.dy, inc.dyaw
            st.last_inc = (dx, dy, dpsi)
            st.coast_ticks = 0
            st.mode = "vo"
            mult = p.bridged_noise_mult if inc.quality == "bridged" else 1.0
            if st.distrust_until is not None and st.t_ns < st.distrust_until:
                mult *= p.distrust_mult
            hold = False
        elif st.last_inc is not None and st.coast_ticks * dt < p.coast_max_s:
            dx, dy, dpsi = st.last_inc
            st.coast_ticks += 1
            st.mode = "coast"
            mult = p.coast_noise_mult
            hold = False
        else:
            dx = dy = dpsi = 0.0
            st.mode = "hold"
            mult = 1.0
            hold = True
            st.locked = False
        c, sn = math.cos(psi), math.sin(psi)
        bx, by = (c * dx - sn * dy), (sn * dx + c * dy)       # increment in map axes (before scale)
        use_s = p.use_scale
        sc = s if use_s else 1.0
        st.x = np.array([x + sc * bx, y + sc * by, wrap(psi + dpsi), s])
        F = np.eye(4)
        F[0, 2] = sc * (-sn * dx - c * dy)
        F[1, 2] = sc * (c * dx - sn * dy)
        if use_s:
            F[0, 3], F[1, 3] = bx, by
        # process noise
        d = math.hypot(dx, dy)
        Q = np.zeros((4, 4))
        if hold:
            qh = (p.hold_q_pos_sqrt_s ** 2) * dt
            Q[0, 0] = Q[1, 1] = qh
            Q[2, 2] = (p.q_yaw_sqrt_s * 5) ** 2 * dt
        else:
            sa = (p.q_along * d) ** 2 + p.q_floor_m ** 2
            sl = (p.q_lat * d) ** 2 + p.q_floor_m ** 2
            if d > 1e-9:
                u = np.array([dx, dy]) / d
                Rb = np.array([[u[0], -u[1]], [u[1], u[0]]])
            else:
                Rb = np.eye(2)
            Qb = Rb @ np.diag([sa, sl]) @ Rb.T * (mult ** 2)
            Rm = np.array([[c, -sn], [sn, c]])
            Q[:2, :2] = Rm @ Qb @ Rm.T
            Q[2, 2] = (p.q_yaw_sqrt_s ** 2) * dt * (mult ** 2)
        if use_s:
            Q[3, 3] = (p.q_scale_sqrt_s ** 2) * dt
        else:
            Q[3, 3] = 0.0
        st.P = F @ st.P @ F.T + Q
        st.P = 0.5 * (st.P + st.P.T)

    def _update(self, st: State, m: Meas, t_key: int) -> tuple[str, float]:
        """Return (outcome, d2). Outcomes: accepted | rejected | reinit | skipped."""
        p = self.p
        weak = m.kind == "retrieval"
        if weak:
            if st.last_ret_t is not None and (t_key - st.last_ret_t) * 1e-9 < p.ret_min_interval_s:
                return "skipped", float("nan")
            sig_xy, sig_yaw, hard, soft = None, p.ret_sig_yaw, p.ret_gate, p.ret_gate
        else:
            sig_xy = math.hypot(p.fix_sig_xy, p.fix_sig_dkf * (m.d_kf_m or 0.0))
            if m.n_inl is not None and p.fix_w_inl > 0:
                sig_xy *= 1.0 + p.fix_w_inl * max(0.0, 1.0 - m.n_inl / p.fix_n_ref)
            if m.n_kf is not None and p.fix_w_nkf > 0:
                sig_xy *= 1.0 + p.fix_w_nkf * max(0.0, p.fix_nkf_ref - m.n_kf)
            sig_yaw, hard, soft = p.fix_sig_yaw, p.gate_hard, p.gate_soft
            if ((p.weak_n_inl > 0 and m.n_inl is not None and m.n_inl < p.weak_n_inl)
                    or (p.weak_nkf > 0 and m.n_kf is not None and m.n_kf < p.weak_nkf)):
                hard = soft = p.weak_gate
            # gap inflation: once per gap, before the first test after the gap
            if (p.use_inflation and not st.infl_done and st.last_fix_t is not None
                    and (t_key - st.last_fix_t) * 1e-9 > p.gap_s):
                st.P[0, 0] += p.gap_infl_m ** 2
                st.P[1, 1] += p.gap_infl_m ** 2
                st.P[2, 2] += p.gap_infl_yaw ** 2
                st.infl_done = True
                st.events.append(("inflate", t_key))
        H = np.zeros((3, 4))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        nu = np.array([m.z[0] - st.x[0], m.z[1] - st.x[1], wrap(m.z[2] - st.x[2])])
        if weak:
            cz, sz = math.cos(m.z[2]), math.sin(m.z[2])
            Rz = np.array([[cz, -sz], [sz, cz]])
            R = np.zeros((3, 3))
            R[:2, :2] = Rz @ np.diag([p.ret_sig_lon ** 2, p.ret_sig_lat ** 2]) @ Rz.T
            R[2, 2] = sig_yaw ** 2
            if p.ret_lateral_only:
                # rows: lateral offset in the keyframe heading frame, yaw
                nvec = np.array([-sz, cz])
                H = np.zeros((2, 4))
                H[0, 0], H[0, 1], H[1, 2] = nvec[0], nvec[1], 1.0
                nu = np.array([float(nvec @ nu[:2]), nu[2]])
                R = np.diag([p.ret_sig_lat ** 2, sig_yaw ** 2])
        else:
            R = np.diag([sig_xy ** 2, sig_xy ** 2, sig_yaw ** 2])
        S = H @ st.P @ H.T + R
        d2 = float(nu @ np.linalg.solve(S, nu))
        if p.use_gate and d2 > hard and not (p.unlock_gate and not weak and not st.locked):
            if not weak:
                st.rej.append((float(nu[0]), float(nu[1]), float(nu[2])))
                st.rej = st.rej[-p.reinit_n:]
                if len(st.rej) >= p.reinit_n:
                    a = np.array(st.rej)
                    spread = float(np.max(np.linalg.norm(a[:, :2] - a[:, :2].mean(0), axis=1)))
                    if spread <= p.reinit_tol_m:
                        st.x[0], st.x[1], st.x[2] = m.z
                        st.P[:3, :] = 0.0
                        st.P[:, :3] = 0.0
                        st.P[0, 0] = st.P[1, 1] = sig_xy ** 2 + p.init_sig_xy ** 2
                        st.P[2, 2] = sig_yaw ** 2 + p.init_sig_yaw ** 2
                        st.rej = []
                        st.last_fix_t = t_key
                        st.infl_done = False
                        st.events.append(("reinit", t_key))
                        return "reinit", d2
            if not weak:
                st.rej_run += 1
            if p.consec_reject_reset > 0 and not weak and st.rej_run >= p.consec_reject_reset:
                st.x[0], st.x[1], st.x[2] = m.z
                st.P[:3, :] = 0.0
                st.P[:, :3] = 0.0
                st.P[0, 0] = st.P[1, 1] = sig_xy ** 2 + p.init_sig_xy ** 2
                st.P[2, 2] = sig_yaw ** 2 + p.init_sig_yaw ** 2
                st.rej, st.rej_run = [], 0
                st.last_fix_t = t_key
                st.infl_done = False
                st.distrust_until = t_key + int(p.distrust_s * 1e9)
                st.events.append(("reset_distrust", t_key))
                return "reinit", d2
            return "rejected", d2
        if p.persistent_reset and not weak and d2 > soft:
            st.rej.append((float(nu[0]), float(nu[1]), float(nu[2])))
            st.rej = st.rej[-p.reinit_n:]
            if len(st.rej) >= p.reinit_n:
                a = np.array(st.rej)
                if float(np.max(np.linalg.norm(a[:, :2] - a[:, :2].mean(0), axis=1))) <= p.reinit_tol_m:
                    st.x[0], st.x[1], st.x[2] = m.z
                    st.P[:3, :] = 0.0
                    st.P[:, :3] = 0.0
                    st.P[0, 0] = st.P[1, 1] = sig_xy ** 2 + p.init_sig_xy ** 2
                    st.P[2, 2] = sig_yaw ** 2 + p.init_sig_yaw ** 2
                    st.rej = []
                    st.last_fix_t = t_key
                    st.infl_done = False
                    st.events.append(("reinit", t_key))
                    return "reinit", d2
        if p.use_huber and d2 > soft:
            R = R * (d2 / soft)
            S = H @ st.P @ H.T + R
        K = st.P @ H.T @ np.linalg.inv(S)
        st.x = st.x + K @ nu
        st.x[2] = wrap(st.x[2])
        I_KH = np.eye(4) - K @ H
        st.P = I_KH @ st.P @ I_KH.T + K @ R @ K.T
        st.P = 0.5 * (st.P + st.P.T)
        if weak:
            st.last_ret_t = t_key
        else:
            st.rej_run = 0
            st.locked = True
            st.last_fix_t = t_key
            st.infl_done = False
            if not (p.persistent_reset and d2 > soft):
                st.rej = []
        return "accepted", d2

    # ------------------------------------------------------------------ tick API
    def step(self, t_ns: int, inc: Inc, arrived: list[Meas]) -> dict:
        """Process tick k = len(ticks). `arrived`: measurements newly available at this tick (their .key <= k, .arrive == k).
        Returns the pose and bookkeeping for this tick."""
        k = len(self.ticks)
        self.ticks.append(int(t_ns))
        self.incs.append(inc)
        jmin = k
        info = {"accepted": 0, "rejected": 0, "reinit": 0, "ret_used": 0, "dropped_old": 0, "d2": [], "replay": 0}
        for m in arrived:
            if m.key > k:
                raise ValueError("measurement from the future")
            if k - m.key > self.p.max_lag_ticks:
                info["dropped_old"] += 1
                continue
            self.meas.setdefault(m.key, []).append(m)
            jmin = min(jmin, m.key)
        start = self._s0.copy() if jmin == 0 else self.snap[jmin - 1].copy()
        # tick 0 carries no motion; its own increment is applied as given (zero, invalid) with dt = 0.1 s nominal
        st = start
        for j in range(jmin, k + 1):
            dt = 0.1 if j == 0 else (self.ticks[j] - self.ticks[j - 1]) * 1e-9
            if j > 0:
                self._predict(st, self.incs[j], dt)
            st.t_ns = self.ticks[j]
            for m in sorted(self.meas.get(j, []), key=lambda mm: (mm.kind != "pnp", mm.arrive)):
                if m.arrive > k:
                    continue
                out, d2 = self._update(st, m, self.ticks[j])
                if any(m is a for a in arrived):          # count each measurement once, at its arrival tick
                    info["d2"].append(round(d2, 3) if d2 == d2 else d2)
                    if out == "accepted":
                        info["accepted" if m.kind == "pnp" else "ret_used"] += 1
                    elif out == "rejected":
                        info["rejected"] += 1
                    elif out == "reinit":
                        info["reinit"] += 1
            if j < len(self.snap):
                self.snap[j] = st.copy()
            else:
                self.snap.append(st.copy())
        info["replay"] = k - jmin
        self.cur = st
        self.out_events = list(st.events)
        gap = None if st.last_fix_t is None else (self.ticks[k] - st.last_fix_t) * 1e-9
        info.update({"x": float(st.x[0]), "y": float(st.x[1]), "yaw": float(st.x[2]), "scale": float(st.x[3]), "mode": st.mode,
                     "gap_s": gap, "cov": st.P[:3, :3].tolist(),
                     "degraded": gap is None or gap > self.p.stale_fix_s})
        return info
