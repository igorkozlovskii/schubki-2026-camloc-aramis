"""S07d (E09k2): yaw-aware extension of the S07c kinematic filter over E04 camera-pose fixes. Imports the frozen S07c filter and overrides only what the
yaw channel needs; with every new switch at its default the behavior is bit-identical to S07c KinFilter (tested).

New (all causal, none uses anything but the E04 rows delivered by the runtime):
- split_gate: PnP fix tested separately on position (2-dof marginal chi-square vs gate_hard) and on yaw (1-dof vs yaw_gate). Good position + implausible yaw ->
  position-only update; rejected position + plausible yaw (yaw_only) -> yaw-only update (the fix still counts toward the persistent-rejection reset).
- ret_use_yaw=False: the retrieval weak update acts on position only (S07d first pass: harmful on train, see ADDENDUM1).
- fix_w_inl_yaw: PnP yaw sigma grows as inliers drop (split_gate path).
- yaw_hold_s: OUTPUT yaw = raw yaw of the last accepted PnP fix once the last accepted yaw measurement is older than yaw_hold_s (state untouched).
Heading differences are wrapped explicitly (kin_filter.wrap).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .kin_filter import KinFilter, KinParams, KState, Meas, wrap


@dataclass
class Kin2Params(KinParams):
    split_gate: bool = False
    yaw_gate: float = 9.0          # chi2(1) ~0.997
    yaw_only: bool = False
    ret_use_yaw: bool = True
    yaw_hold_s: float = 1e9        # output hold of the last accepted fix yaw; 1e9 = off
    fix_w_inl_yaw: float = 0.0     # split_gate only: PnP yaw sigma multiplier 1 + w * max(0, 1 - n_inl / fix_n_ref)
    use_retrieval: bool = True
    model: str = "ctr"


@dataclass
class Kin2State(KState):
    last_yaw_fix: Optional[tuple] = None        # (yaw of the last accepted PnP fix, its tick time ns)

    def copy(self) -> "Kin2State":
        return Kin2State(self.x.copy(), self.P.copy(), self.last_fix_t, self.last_ret_t, self.infl_done, list(self.rej), self.n_reanchor, self.last_yaw_fix)


class Kin2Filter(KinFilter):
    def __init__(self, p: Kin2Params, first_pose: tuple, t0_ns: int):
        super().__init__(p, first_pose, t0_ns)
        s = self._s0
        self._s0 = Kin2State(s.x, s.P, s.last_fix_t, s.last_ret_t, s.infl_done, list(s.rej), s.n_reanchor, None)

    # ---------------------------------------------------------------- helpers
    def _kalman(self, st: Kin2State, rows: list, nu: np.ndarray, R: np.ndarray) -> None:
        H = np.zeros((len(rows), 5))
        for i, r in enumerate(rows):
            H[i, r] = 1.0
        S = H @ st.P @ H.T + R
        K = st.P @ H.T @ np.linalg.inv(S)
        st.x = st.x + K @ nu
        st.x[2] = wrap(st.x[2])
        self._clamp(st)
        I_KH = np.eye(5) - K @ H
        st.P = I_KH @ st.P @ I_KH.T + K @ R @ K.T
        st.P = 0.5 * (st.P + st.P.T)

    def _reanchor(self, st, t_key, m, sig_xy):
        super()._reanchor(st, t_key, m, sig_xy)
        st.last_yaw_fix = (float(m.z[2]), t_key)

    # ---------------------------------------------------------------- update
    def _update(self, st: Kin2State, m: Meas, t_key: int) -> tuple:
        p = self.p
        weak = m.kind == "retrieval"
        if (weak and not p.ret_use_yaw) or ((not weak) and p.split_gate):
            return self._update_split(st, m, t_key, weak)
        out, d2 = super()._update(st, m, t_key)
        if out == "accepted" and not weak:
            st.last_yaw_fix = (float(m.z[2]), t_key)
        return out, d2

    def _update_split(self, st: Kin2State, m: Meas, t_key: int, weak: bool) -> tuple:
        p = self.p
        if weak:
            if not p.use_retrieval:
                return "ignored", float("nan")
            if st.last_ret_t is not None and (t_key - st.last_ret_t) * 1e-9 < p.ret_min_interval_s:
                return "ignored", float("nan")
        gap = (t_key - st.last_fix_t) * 1e-9 if st.last_fix_t is not None else 0.0
        if not weak and not st.infl_done and gap > p.gap_s:
            self._inflate(st)
        zx, zy, zyaw = m.z
        nu = np.array([zx - st.x[0], zy - st.x[1], wrap(zyaw - st.x[2])])
        if weak:
            c, s = math.cos(zyaw), math.sin(zyaw)
            Rot = np.array([[c, -s], [s, c]])
            Rxy = Rot @ np.diag([p.ret_sig_lon ** 2, p.ret_sig_lat ** 2]) @ Rot.T
            Sxy = st.P[:2, :2] + Rxy
            d2 = float(nu[:2] @ np.linalg.solve(Sxy, nu[:2]))
            if p.use_gate and d2 > p.ret_gate:
                return "rejected", d2
            self._kalman(st, [0, 1], nu[:2], Rxy)
            st.last_ret_t = t_key
            return "accepted", d2
        fac = 1.0 + p.fix_w_inl * max(0.0, 1.0 - (m.n_inl if m.n_inl is not None else p.fix_n_ref) / p.fix_n_ref)
        sig_xy = p.fix_sig_xy * fac
        Rxy = np.diag([sig_xy ** 2, sig_xy ** 2])
        Sxy = st.P[:2, :2] + Rxy
        fac_y = 1.0 + p.fix_w_inl_yaw * max(0.0, 1.0 - (m.n_inl if m.n_inl is not None else p.fix_n_ref) / p.fix_n_ref)
        sig_yaw = p.fix_sig_yaw * fac_y
        Syaw = st.P[2, 2] + sig_yaw ** 2
        d2xy = float(nu[:2] @ np.linalg.solve(Sxy, nu[:2]))
        d2y = float(nu[2] ** 2 / Syaw)
        pos_ok = (not p.use_gate) or d2xy <= p.gate_hard
        yaw_ok = (not p.use_gate) or d2y <= p.yaw_gate
        if not pos_ok:
            st.rej.append((float(nu[0]), float(nu[1]), t_key, zx, zy, zyaw))
            st.rej = st.rej[-max(p.reanchor_n, 2):]
            if p.use_reanchor and len(st.rej) >= p.reanchor_n:
                a, b = st.rej[-2], st.rej[-1]
                dtf = (b[2] - a[2]) * 1e-9
                if dtf > 0 and math.hypot(b[3] - a[3], b[4] - a[4]) / dtf <= p.reanchor_vmax_mult * p.v_max:
                    self._reanchor(st, t_key, m, sig_xy)
                    return "reanchor", d2xy
            if p.yaw_only and yaw_ok:
                self._kalman(st, [2], nu[2:3], np.array([[sig_yaw ** 2]]))
                st.last_yaw_fix = (float(zyaw), t_key)
            return "rejected", d2xy
        rows = [0, 1] + ([2] if yaw_ok else [])
        R = np.diag([sig_xy ** 2, sig_xy ** 2] + ([sig_yaw ** 2] if yaw_ok else []))
        d2 = d2xy + (d2y if yaw_ok else 0.0)
        if p.use_huber and d2 > p.gate_soft:
            R = R * (d2 / p.gate_soft)
        self._kalman(st, rows, nu[rows], R)
        st.rej = []
        st.last_fix_t = t_key
        st.infl_done = False
        if yaw_ok:
            st.last_yaw_fix = (float(zyaw), t_key)
        return "accepted", d2

    # ---------------------------------------------------------------- tick API
    def step(self, t_ns: int, arrived: list) -> dict:
        info = super().step(t_ns, arrived)
        st = self.cur
        info["yaw_state"] = info["yaw"]
        info["yaw_held"] = False
        if st.last_yaw_fix is not None and (int(t_ns) - st.last_yaw_fix[1]) * 1e-9 > self.p.yaw_hold_s:
            info["yaw"] = float(st.last_yaw_fix[0])
            info["yaw_held"] = True
        return info
