"""S10f (E10 round 3): persistent-rejection reset with a consistency + persistence + error-budget test, for the telemetry filter (Tel3Filter) and the camera-only
filter (Kin3Filter, "Y1-R"). Subclasses of the frozen S10b TelFilter / S07d Kin2Filter that override ONLY the PnP update branch; `Reanchor3(mode="legacy")` is bitwise the old behaviour.

A candidate adoption exists when the old trigger fires (reanchor_n consecutive rejected PnP fixes + the old plausibility check on the last two). It is ADOPTED (the latest fix, as before) iff
  1. consistency: max_i |nu_i - mean(nu)| <= spread_x over the reanchor_n rejected fixes;
  2. persistence: t_last - t_first >= t_min;
  3. budget: |mean(nu)| <= b0 + b1 * T (T = time since the last accepted fix / re-anchor / start, from tick stamps) OR hard reset: T >= t_hard or sqrt(0.5 (Pxx+Pyy)) >= sig_hard.
Otherwise HOLD (no state change). R2 additionally inflates P_xx, P_yy by (g dt)^2 per held tick while a consistent+persistent over-budget candidate exists. R3: budget never passes (b0=b1=0).
Causal: uses only tick stamps and the fixes already in the reject buffer.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .kin2_filter import Kin2Filter, Kin2Params, Kin2State
from .kin_filter import Meas, wrap
from .tel_filter import TelFilter, TelParams, TState


@dataclass
class Reanchor3:
    mode: str = "legacy"        # legacy | r1 | r2 | r3
    spread_x: float = 3.0
    t_min: float = 1.0
    b0: float = 6.0
    b1: float = 0.2
    t_hard: float = 20.0
    g: float = 0.2
    sig_hard: float = 4.0

    def to_dict(self) -> dict:
        return asdict(self)


def decide(rej: list, T: float, sig_pos: float, rp: Reanchor3) -> tuple:
    """(adopt, reason, mean_nu, over_budget_consistent). rej: [(nux, nuy, t_ns, zx, zy, zyaw)] (the last reanchor_n rejected fixes)."""
    nu = np.array([[r[0], r[1]] for r in rej])
    mu = nu.mean(axis=0)
    spread = float(np.max(np.hypot(*(nu - mu).T)))
    dur = (rej[-1][2] - rej[0][2]) * 1e-9
    mean_nu = float(math.hypot(*mu))
    if spread > rp.spread_x:
        return False, "inconsistent", mean_nu, False
    if dur < rp.t_min:
        return False, "short", mean_nu, False
    hard = T >= rp.t_hard or sig_pos >= rp.sig_hard
    b0, b1 = (0.0, 0.0) if rp.mode == "r3" else (rp.b0, rp.b1)
    if hard:
        return True, "hard_reset", mean_nu, False
    if mean_nu <= b0 + b1 * T:
        return True, "within_budget", mean_nu, False
    return False, "over_budget", mean_nu, True


@dataclass
class TState3(TState):
    last_acc_t: Optional[int] = None

    def copy(self) -> "TState3":
        return TState3(self.x.copy(), self.P.copy(), self.last_ret_t, list(self.rej), self.n_reanchor, self.last_acc_t)


class Tel3Filter(TelFilter):
    def __init__(self, p: TelParams, first_pose: tuple, t0_ns: int, rp: Reanchor3):
        super().__init__(p, first_pose, t0_ns)
        self.rp = rp
        s = self._s0
        self._s0 = TState3(s.x, s.P, s.last_ret_t, list(s.rej), s.n_reanchor, int(t0_ns))
        self.events: dict = {}            # t_key -> dict (latest replay wins)

    def _apply_reseed(self, st, rs):
        super()._apply_reseed(st, rs)
        st.last_acc_t = None

    def _update(self, st: TState3, m: Meas, t_key: int) -> tuple:
        rp, p = self.rp, self.p
        if m.kind == "retrieval" or rp.mode == "legacy":
            out, d2 = super()._update(st, m, t_key)
            if m.kind == "pnp" and out in ("accepted", "reanchor"):
                st.last_acc_t = t_key
            return out, d2
        if st.last_acc_t is None:
            st.last_acc_t = t_key
        zx, zy, zyaw = m.z
        nu = np.array([zx - st.x[0], zy - st.x[1], wrap(zyaw - st.x[2])])
        fac = 1.0 + p.fix_w_inl * max(0.0, 1.0 - (m.n_inl if m.n_inl is not None else p.fix_n_ref) / p.fix_n_ref)
        sig_xy = p.fix_sig_xy * fac
        R = np.diag([sig_xy ** 2, sig_xy ** 2, p.fix_sig_yaw ** 2])
        S = st.P[:3, :3] + R
        d2 = float(nu @ np.linalg.solve(S, nu))
        if p.use_gate and d2 > p.gate_hard:
            st.rej.append((float(nu[0]), float(nu[1]), t_key, zx, zy, zyaw))
            st.rej = st.rej[-max(p.reanchor_n, 2):]
            if p.use_reanchor and len(st.rej) >= p.reanchor_n:
                a, b = st.rej[-2], st.rej[-1]
                dtf = (b[2] - a[2]) * 1e-9
                if dtf > 0 and math.hypot(b[3] - a[3], b[4] - a[4]) / dtf <= p.reanchor_vmax_mult * p.v_max:
                    T = (t_key - st.last_acc_t) * 1e-9
                    sp = math.sqrt(0.5 * (st.P[0, 0] + st.P[1, 1]))
                    adopt, why, mnu, over = decide(st.rej[-p.reanchor_n:], T, sp, rp)
                    if adopt:
                        self.events[t_key] = {"kind": "reanchor", "why": why, "mean_nu": mnu, "T": T, "d2": d2}
                        self._reanchor(st, m, sig_xy)
                        st.last_acc_t = t_key
                        return "reanchor", d2
                    self.events[t_key] = {"kind": "held", "why": why, "mean_nu": mnu, "T": T, "d2": d2}
                    if over and rp.mode == "r2":
                        dt = min(max((t_key - st.rej[-2][2]) * 1e-9, 0.0), 1.0)
                        st.P[0, 0] += (rp.g * dt) ** 2
                        st.P[1, 1] += (rp.g * dt) ** 2
            return "rejected", d2
        if p.use_huber and d2 > p.gate_soft:
            R = R * (d2 / p.gate_soft)
        self._kalman(st, nu, R)
        st.rej = []
        st.last_acc_t = t_key
        return "accepted", d2


class Kin3Filter(Kin2Filter):
    """Y1-R: frozen Y1 (Kin2Filter, split_gate False) with only the persistent-rejection reset replaced."""

    def __init__(self, p: Kin2Params, first_pose: tuple, t0_ns: int, rp: Reanchor3):
        super().__init__(p, first_pose, t0_ns)
        self.rp = rp
        self.events: dict = {}

    def _update(self, st: Kin2State, m: Meas, t_key: int) -> tuple:
        rp, p = self.rp, self.p
        if m.kind == "retrieval" or rp.mode == "legacy" or p.split_gate:
            return super()._update(st, m, t_key)
        gap = (t_key - st.last_fix_t) * 1e-9 if st.last_fix_t is not None else 0.0
        if not st.infl_done and gap > p.gap_s:
            self._inflate(st)
        H = np.zeros((3, 5))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        zx, zy, zyaw = m.z
        nu = np.array([zx - st.x[0], zy - st.x[1], wrap(zyaw - st.x[2])])
        fac = 1.0 + p.fix_w_inl * max(0.0, 1.0 - (m.n_inl if m.n_inl is not None else p.fix_n_ref) / p.fix_n_ref)
        sig_xy = p.fix_sig_xy * fac
        R = np.diag([sig_xy ** 2, sig_xy ** 2, p.fix_sig_yaw ** 2])
        S = H @ st.P @ H.T + R
        d2 = float(nu @ np.linalg.solve(S, nu))
        if p.use_gate and d2 > p.gate_hard:
            st.rej.append((float(nu[0]), float(nu[1]), t_key, zx, zy, zyaw))
            st.rej = st.rej[-max(p.reanchor_n, 2):]
            if p.use_reanchor and len(st.rej) >= p.reanchor_n:
                a, b = st.rej[-2], st.rej[-1]
                dtf = (b[2] - a[2]) * 1e-9
                if dtf > 0 and math.hypot(b[3] - a[3], b[4] - a[4]) / dtf <= p.reanchor_vmax_mult * p.v_max:
                    T = gap
                    sp = math.sqrt(0.5 * (st.P[0, 0] + st.P[1, 1]))
                    adopt, why, mnu, over = decide(st.rej[-p.reanchor_n:], T, sp, rp)
                    if adopt:
                        self.events[t_key] = {"kind": "reanchor", "why": why, "mean_nu": mnu, "T": T, "d2": d2}
                        self._reanchor(st, t_key, m, sig_xy)
                        return "reanchor", d2
                    self.events[t_key] = {"kind": "held", "why": why, "mean_nu": mnu, "T": T, "d2": d2}
                    if over and rp.mode == "r2":
                        dt = min(max((t_key - st.rej[-2][2]) * 1e-9, 0.0), 1.0)
                        st.P[0, 0] += (rp.g * dt) ** 2
                        st.P[1, 1] += (rp.g * dt) ** 2
            return "rejected", d2
        if p.use_huber and d2 > p.gate_soft:
            R = R * (d2 / p.gate_soft)
            S = H @ st.P @ H.T + R
        K = st.P @ H.T @ np.linalg.inv(S)
        st.x = st.x + K @ nu
        st.x[2] = wrap(st.x[2])
        self._clamp(st)
        I_KH = np.eye(5) - K @ H
        st.P = I_KH @ st.P @ I_KH.T + K @ R @ K.T
        st.P = 0.5 * (st.P + st.P.T)
        st.rej = []
        st.last_fix_t = t_key
        st.infl_done = False
        st.last_yaw_fix = (float(m.z[2]), t_key)
        return "accepted", d2
