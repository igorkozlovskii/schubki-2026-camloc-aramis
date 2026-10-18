"""S10g (E10 round 4): telemetry-assisted filter with a hard reset that the fix-consistency test cannot block.

Subclass of the frozen S10b TelFilter that replaces ONLY the PnP update branch. A candidate adoption exists when the S10b trigger fires (len(rej) >= reanchor_n and the kinematic plausibility
check on the last two rejected fixes). Then (see reports/experiments/S10g-tel4/DECISION_RULE.md):
  Path A  hard reset: T >= cap_eff (t_fresh while the state has < n_sup accepted fixes since the last adoption/reseed/start, else t_cap) or position sigma >= sig_hard. Nothing blocks it.
  Path D  gradual divergence (G1, G2): the innovations of the buffered fixes are linear in time up to spread_d, the first rejected fix of the run had |nu| <= onset_m + onset_rate * T_first (amendment 1: the error grew
          through the gate instead of jumping; T_first = time since the last accepted fix), the run lasted >= t_min_d and the slope is <= slope_max: adopt the latest fix.
  else    HOLD (no state change).
Candidate on adoption: "latest" fix, or "median" (G2): every buffered fix carried to the current tick through the stored telemetry-propagated state history, componentwise median; if the carried fixes
disagree (> spread_sel) the history is not trusted (diverging state) and the latest fix is used. Causal: only the reject buffer, the stored states of earlier ticks and tick stamps are used.
`Reanchor4(mode="legacy")` is bitwise the frozen V3 filter.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Optional

import numpy as np

from .kin_filter import Meas, wrap
from .tel_filter import TelFilter, TelParams, TickRec, TState


@dataclass
class Reanchor4:
    mode: str = "legacy"            # legacy | g1 | g2 | g3
    t_cap: float = 8.0
    t_fresh: float = 2.0
    n_sup: int = 50
    sig_hard: float = 4.0
    spread_d: float = 2.0
    onset_m: float = 6.0
    onset_rate: float = 8.0
    t_min_d: float = 0.5
    slope_max: float = 20.0
    spread_sel: float = 3.0

    @property
    def use_d(self) -> bool:
        return self.mode in ("g1", "g2")

    @property
    def sel(self) -> str:
        return "median" if self.mode == "g2" else "latest"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TState4(TState):
    last_acc_t: Optional[int] = None
    n_acc: int = 0
    run_first_nu: Optional[float] = None     # |nu| of the first rejected fix of the current rejection run
    run_first_T: Optional[float] = None      # seconds from the last accepted fix / adoption / start to that fix
    run_first_t: Optional[int] = None

    def copy(self) -> "TState4":
        return TState4(self.x.copy(), self.P.copy(), self.last_ret_t, list(self.rej), self.n_reanchor, self.last_acc_t, self.n_acc, self.run_first_nu, self.run_first_T, self.run_first_t)


def trend_fit(rej: list) -> tuple:
    """Per-axis linear fit of nu over time on the buffer: (max residual norm [m], slope norm [m/s])."""
    nu = np.array([[r[0], r[1]] for r in rej])
    t = np.array([(r[2] - rej[0][2]) * 1e-9 for r in rej])
    tc = t - t.mean()
    den = float(tc @ tc)
    if den <= 0.0:
        return float(np.max(np.hypot(*(nu - nu.mean(axis=0)).T))), 0.0
    b = (tc @ nu) / den
    res = nu - nu.mean(axis=0) - np.outer(tc, b)
    return float(np.max(np.hypot(res[:, 0], res[:, 1]))), float(math.hypot(*b))


def decide4(T: float, sig_pos: float, n_acc: int, first_nu: Optional[float], first_T: Optional[float], run_dur: float, rej: list, rp: Reanchor4) -> tuple:
    """(adopt, reason, extra) for a candidate adoption. Path A first (independent of the fix cluster), then Path D."""
    cap = min(rp.t_fresh, rp.t_cap) if n_acc < rp.n_sup else rp.t_cap
    if T >= cap or sig_pos >= rp.sig_hard:
        return True, "hard_reset", {"cap_eff": cap}
    if rp.use_d and first_nu is not None:
        spread, slope = trend_fit(rej)
        gradual = first_nu <= rp.onset_m + rp.onset_rate * first_T
        if spread <= rp.spread_d and gradual and run_dur >= rp.t_min_d and slope <= rp.slope_max:
            return True, "gradual", {"trend_spread": spread, "slope": slope}
        why = "jump" if not gradual else ("inconsistent_trend" if spread > rp.spread_d else ("short" if run_dur < rp.t_min_d else "fast"))
        return False, "held_" + why, {"trend_spread": spread, "slope": slope}
    return False, "held_wait", {}


class Tel4Filter(TelFilter):
    def __init__(self, p: TelParams, first_pose: tuple, t0_ns: int, rp: Reanchor4):
        super().__init__(p, first_pose, t0_ns)
        self.rp = rp
        s = self._s0
        self._s0 = TState4(s.x, s.P, s.last_ret_t, list(s.rej), s.n_reanchor, int(t0_ns), 0, None, None, None)
        self.events: dict = {}          # t_key -> dict (latest replay wins)
        self._tidx: dict = {}           # tick stamp -> tick index

    def step(self, rec: TickRec, arrived: list) -> dict:
        self._tidx[rec.t_ns] = len(self.recs)
        return super().step(rec, arrived)

    def _apply_reseed(self, st, rs):
        super()._apply_reseed(st, rs)
        st.last_acc_t, st.n_acc, st.run_first_nu, st.run_first_T, st.run_first_t = None, 0, None, None, None

    def _select(self, st: TState4, m: Meas, t_key: int, buf: list) -> tuple:
        """(z, how): candidate pose to adopt."""
        if self.rp.sel == "latest":
            return m.z, "latest"
        carried, ok = [], True
        for r in buf:
            if r[2] == t_key:
                carried.append((r[3], r[4], r[5]))
                continue
            j = self._tidx.get(r[2])
            if j is None or j >= len(self.snap):
                ok = False
                break
            si, sn = self.snap[j].x, st.x
            c, s = math.cos(si[2]), math.sin(si[2])
            dx, dy = sn[0] - si[0], sn[1] - si[1]
            rx, ry = c * dx + s * dy, -s * dx + c * dy
            cz, sz = math.cos(r[5]), math.sin(r[5])
            carried.append((r[3] + cz * rx - sz * ry, r[4] + sz * rx + cz * ry, wrap(r[5] + wrap(sn[2] - si[2]))))
        if not ok:
            return m.z, "latest_nohist"
        C = np.array(carried)
        med = np.median(C[:, :2], axis=0)
        dist = np.hypot(C[:, 0] - med[0], C[:, 1] - med[1])
        if float(dist.max()) > self.rp.spread_sel:
            return m.z, "latest_fallback"
        k = int(np.argmin(dist))
        return (float(med[0]), float(med[1]), float(C[k, 2])), "median"

    def _update(self, st: TState4, m: Meas, t_key: int) -> tuple:
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
                    ev = {"why": why, "mean_nu": float(math.hypot(*mu)), "T": T, "d2": d2, "n_acc": st.n_acc, "first_nu": st.run_first_nu, "first_T": st.run_first_T, **extra}
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
