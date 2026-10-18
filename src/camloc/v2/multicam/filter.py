"""G1 (Tel4Filter, imported unchanged) with multi-camera fixes: per-camera measurement scale and an agreement gate on adoptions.

Measurements are the PnP rows of several cameras. Row tags (cam, kappa_xy, kappa_yaw, conf) arrive through `tags[(t_ns, x_m, y_m)]`.
 * kappa scales the fix noise of that camera (sigma_xy * kappa_xy, sigma_yaw * kappa_yaw) in the gain and in the gate; kappa = 1 is G1's fl noise.
 * Agreement gate (`require_confirm`): a correction above G1's gate is the only place where a fix moves the state by more than the gate allows, i.e. the adoption
   (re-anchor: Path A hard reset / Path D gradual) of the Tel4Filter. With the gate on, an adoption needs the candidate fix to be CONFIRMED: another camera's fix of the
   same image tick lies within D metres (flag `conf`, computed from the same tick's rows only: causal). An unconfirmed candidate is held (state unchanged). After
   `escape_s` seconds without any accepted fix the gate is lifted (G1 behaviour), so the filter cannot be locked out when only one camera works.
Everything else is the code of camloc.filtering.tel4_filter (copied logic of Tel4Filter._update for the pnp branch with the two hooks; retrieval rows unchanged).
No ground truth.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from camloc.filtering.kin_filter import wrap
from camloc.filtering.tel4_filter import Tel4Filter, decide4


class MultiCamTel4Filter(Tel4Filter):
    def __init__(self, p, first_pose, t0_ns, rp, tags: dict, require_confirm: bool = False, escape_s: float = 30.0):
        super().__init__(p, first_pose, t0_ns, rp)
        self.tags = tags
        self.require_confirm = bool(require_confirm)
        self.escape_ns = int(escape_s * 1e9)
        self.log: dict = {}

    def _tag(self, t_key: int, z: tuple) -> dict:
        return self.tags.get((int(t_key), round(float(z[0]), 5), round(float(z[1]), 5)), {"cam": "fl", "kxy": 1.0, "kyaw": 1.0, "conf": False})

    def _update(self, st, m, t_key):
        rp, p = self.rp, self.p
        if m.kind != "pnp" or rp.mode == "legacy":
            return super()._update(st, m, t_key)
        tag = self._tag(t_key, m.z)
        if st.last_acc_t is None:
            st.last_acc_t = t_key
        zx, zy, zyaw = m.z
        nu = np.array([zx - st.x[0], zy - st.x[1], wrap(zyaw - st.x[2])])
        fac = 1.0 + p.fix_w_inl * max(0.0, 1.0 - (m.n_inl if m.n_inl is not None else p.fix_n_ref) / p.fix_n_ref)
        sig_xy = p.fix_sig_xy * fac * tag["kxy"]
        sig_yaw = p.fix_sig_yaw * tag["kyaw"]
        R = np.diag([sig_xy ** 2, sig_xy ** 2, sig_yaw ** 2])
        S = st.P[:3, :3] + R
        d2 = float(nu @ np.linalg.solve(S, nu))
        if p.use_gate and d2 > p.gate_hard:
            if st.run_first_nu is None:
                st.run_first_nu, st.run_first_T, st.run_first_t = float(math.hypot(nu[0], nu[1])), (t_key - st.last_acc_t) * 1e-9, t_key
            st.rej.append((float(nu[0]), float(nu[1]), t_key, zx, zy, zyaw))
            st.rej = st.rej[-max(p.reanchor_n, 2):]
            if p.use_reanchor and len(st.rej) >= p.reanchor_n:
                a, b = st.rej[-2], st.rej[-1]
                dtf = (b[2] - a[2]) * 1e-9
                if dtf > 0 and math.hypot(b[3] - a[3], b[4] - a[4]) / dtf <= p.reanchor_vmax_mult * p.v_max:
                    T = (t_key - st.last_acc_t) * 1e-9
                    sp = math.sqrt(0.5 * (st.P[0, 0] + st.P[1, 1]))
                    buf = st.rej[-p.reanchor_n:]
                    adopt, why, extra = decide4(T, sp, st.n_acc, st.run_first_nu, st.run_first_T, (t_key - st.run_first_t) * 1e-9, buf, rp)
                    mu = np.mean([[r[0], r[1]] for r in buf], axis=0)
                    ev = {"why": why, "mean_nu": float(math.hypot(*mu)), "T": T, "d2": d2, "n_acc": st.n_acc, "first_nu": st.run_first_nu, "first_T": st.run_first_T,
                          "cam": tag["cam"], "conf": bool(tag["conf"]), **extra}
                    if adopt and self.require_confirm and not tag["conf"] and (t_key - st.last_acc_t) < self.escape_ns:
                        self.events[t_key] = {"kind": "held", **{**ev, "why": "unconfirmed:" + why}}
                        return "rejected", d2
                    if adopt:
                        z, how = self._select(st, m, t_key, buf)
                        self.events[t_key] = {"kind": "reanchor", "sel": how, **ev}
                        self._reanchor(st, replace(m, z=z), sig_xy)
                        st.last_acc_t, st.n_acc, st.run_first_nu, st.run_first_T, st.run_first_t = t_key, 0, None, None, None
                        return "reanchor", d2
                    self.events[t_key] = {"kind": "held", **ev}
            return "rejected", d2
        if p.use_huber and d2 > p.gate_soft:
            R = R * (d2 / p.gate_soft)
        self._kalman(st, nu, R)
        st.rej = []
        st.last_acc_t = t_key
        st.n_acc += 1
        st.run_first_nu, st.run_first_T, st.run_first_t = None, None, None
        return "accepted", d2
