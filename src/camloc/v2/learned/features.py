"""Per-tick inputs of the S3.1 learned model, built causally from telemetry and meas-v1 streams. No GT, no reference lines, no map-state or pose fields.

Every array row k describes what is known at tick t_k from inputs stamped (telemetry) or available (measurements) in (t_{k-1}, t_k], plus
the latest telemetry row with stamp <= t_k. Nothing later is read: telemetry integrals use a zero-order hold of rows with stamp <= the upper
end of each sub-interval, measurements are filtered on avail_ns <= t_k.

Telemetry (tier fields only, inputs-v2 / telemetry-v1):
    rates2[k, j]  T2 sub-interval means of (vx + ax extrapolation, vy, wz) over M equal sub-steps of (t_{k-1}, t_k]
    rates1[k, j]  T1 means of (v + ax extrapolation, 0, wz)
    raw[k]        latest (vx, vy, wz, ax, v) at t_k (NaN -> 0), ok2 / ok1: the tier's latest row is at most `fallback_s` old (filter2 rule)
Measurements per channel (fl_lm landmark fix, fl_co coarse place, fr_lm second-camera fix, bd border):
    pres[k], z[k] = (s, n, dpsi) absolute in track-v1, age[k] = t_k - stamp (s), q[k] = channel quality fields (fixed transforms, no fitted scaling)
"""
from __future__ import annotations

import math

import numpy as np

CHANNELS = ("fl_lm", "fl_co", "fr_lm", "bd")
QDIM = {"fl_lm": 5, "fl_co": 2, "fr_lm": 5, "bd": 4}
M_SUB = 4
RAW_FIELDS = ("vx_mps", "vy_mps", "wz_radps", "ax_mps2", "v_mps")
TIER_RAW = {"T2": (0, 1, 2, 3), "T1": (2, 3, 4), "T0": ()}


class _ZOHIntegral:
    """Cumulative integral of a zero-order-hold signal (optionally plus ax * min(tau - stamp, e)): F(t) uses rows with stamp <= t only."""

    def __init__(self, ts: np.ndarray, u: np.ndarray, ax: np.ndarray | None = None, e_s: float = 0.0):
        self.ts = ts.astype(np.int64)
        self.u = np.nan_to_num(u.astype(np.float64))
        self.ax = None if ax is None else np.nan_to_num(ax.astype(np.float64))
        self.e = float(e_s)
        d = np.diff(self.ts) * 1e-9
        seg = self.u[:-1] * d
        if self.ax is not None and self.e > 0:
            seg = seg + self.ax[:-1] * self._g(d)
        self.C = np.r_[0.0, np.cumsum(seg)]

    def _g(self, d):
        e = self.e
        return np.where(d <= e, 0.5 * d * d, 0.5 * e * e + e * (d - e))

    def F(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, np.int64)
        i = np.searchsorted(self.ts, t, side="right") - 1
        ok = i >= 0
        i = np.clip(i, 0, len(self.ts) - 1)
        d = np.where(ok, (t - self.ts[i]) * 1e-9, 0.0)
        out = self.C[i] + self.u[i] * d
        if self.ax is not None and self.e > 0:
            out = out + self.ax[i] * self._g(d)
        return np.where(ok, out, np.nan)


def telemetry_inputs(ticks: np.ndarray, t0_ns: int, stamps: np.ndarray, cols: dict, ax_extrap_s: float = 0.02, fallback_s: float = 1.0) -> dict:
    """Telemetry part of the tick inputs. `cols` holds only allowed fields (RAW_FIELDS subset). Tier availability follows the fields present."""
    ticks = np.asarray(ticks, np.int64)
    N = len(ticks)
    prev = np.r_[np.int64(t0_ns), ticks[:-1]]
    dt = (ticks - prev) * 1e-9
    a = prev[:, None] + ((ticks - prev)[:, None] * np.arange(M_SUB)[None, :]) // M_SUB
    b = prev[:, None] + ((ticks - prev)[:, None] * np.arange(1, M_SUB + 1)[None, :]) // M_SUB
    out = {"dt": dt.astype(np.float32)}
    ts = np.asarray(stamps, np.int64)
    zero = np.zeros(len(ts))
    get = lambda f: cols[f] if f in cols else None  # noqa: E731
    ax = get("ax_mps2")
    # latest row at t_k (stamp <= t_k)
    j = np.searchsorted(ts, ticks, side="right") - 1
    have = j >= 0
    jj = np.clip(j, 0, max(len(ts) - 1, 0))
    age = np.where(have, (ticks - ts[jj]) * 1e-9, np.inf) if len(ts) else np.full(N, np.inf)
    raw = np.zeros((N, len(RAW_FIELDS)), np.float32)
    for c, f in enumerate(RAW_FIELDS):
        v = get(f)
        if v is not None and len(ts):
            raw[:, c] = np.where(have, np.nan_to_num(v[jj]), 0.0)
    out["raw"] = raw
    h = np.maximum((b - a) * 1e-9, 1e-9)

    def mean_rate(u, with_ax):
        I = _ZOHIntegral(ts, u, ax if (with_ax and ax is not None) else None, ax_extrap_s if with_ax else 0.0)
        aa = np.maximum(a, ts[0]) if len(ts) else a
        r = (I.F(b) - I.F(aa)) / h
        return np.nan_to_num(r)

    for tier, need in (("T2", ("vx_mps", "vy_mps", "wz_radps")), ("T1", ("v_mps", "wz_radps"))):
        if len(ts) == 0 or any(get(f) is None for f in need):
            out[f"rates{tier[1]}"] = np.zeros((N, M_SUB, 3), np.float32)
            out[f"ok{tier[1]}"] = np.zeros(N, np.float32)
            continue
        if tier == "T2":
            r = np.stack([mean_rate(get("vx_mps"), True), mean_rate(get("vy_mps"), False), mean_rate(get("wz_radps"), False)], -1)
        else:
            r = np.stack([mean_rate(get("v_mps"), True), np.zeros((N, M_SUB)), mean_rate(get("wz_radps"), False)], -1)
        ok = (age <= fallback_s).astype(np.float32)
        out[f"rates{tier[1]}"] = (r * ok[:, None, None]).astype(np.float32)
        out[f"ok{tier[1]}"] = ok
    _ = zero
    return out


def _q(chan: str, r: dict) -> list[float]:
    g = lambda k, d=0.0: (float(r[k]) if (k in r and r[k] is not None and r[k] == r[k]) else d)  # noqa: E731
    if chan in ("fl_lm", "fr_lm"):
        return [math.log1p(g("q_n_inl")) / 6.0, g("q_inlier_ratio"), min(g("q_rms_px", 2.0), 10.0) / 3.0, math.log1p(g("q_d_kf_m", 5.0)) / 3.0, g("q_sim_top", 0.9)]
    if chan == "fl_co":
        return [g("q_sim_top", 0.9), math.log1p(g("q_d_kf_m", 5.0)) / 3.0]
    if chan == "bd":
        return [math.log(max(g("sig_n", 0.5), 1e-3)) / 3.0, math.log(max(g("sig_dpsi", 0.02), 1e-5)) / 5.0, g("q_support", 0.0) / 10.0, max(min(g("q_cost", 0.0), 50.0), -50.0) / 20.0]
    raise ValueError(chan)


def channel_inputs(ticks: np.ndarray, t0_ns: int, rows: list[dict], chan: str, track) -> dict:
    """Latest valid row per tick interval (by avail_ns); (s, n, dpsi) recomputed from (x, y, yaw) with the row's s as hint so every source uses the same frame."""
    ticks = np.asarray(ticks, np.int64)
    N = len(ticks)
    pres = np.zeros(N, np.float32)
    z = np.zeros((N, 3), np.float32)
    age = np.zeros(N, np.float32)
    q = np.zeros((N, QDIM[chan]), np.float32)
    rr = sorted([r for r in rows if bool(r.get("valid", True))], key=lambda r: (int(r["avail_ns"]), int(r["stamp_ns"])))
    av = np.array([int(r["avail_ns"]) for r in rr], np.int64)
    prev = np.r_[np.int64(t0_ns), ticks[:-1]]
    hi = np.searchsorted(av, ticks, side="right") - 1        # last row with avail <= t_k
    lo = np.searchsorted(av, prev, side="right") - 1         # last row with avail <= t_{k-1}
    for k in np.nonzero(hi > lo)[0]:
        r = rr[hi[k]]
        hint = float(r["s"]) if r.get("s") is not None and r["s"] == r["s"] else None
        s, n, dpsi = track.from_xy(float(r["x"]), float(r["y"]), float(r["yaw"]), s_hint=hint)
        pres[k] = 1.0
        z[k] = (s, n, dpsi)
        age[k] = (ticks[k] - int(r["stamp_ns"])) * 1e-9
        q[k] = _q(chan, r)
    return {f"{chan}_pres": pres, f"{chan}_z": z, f"{chan}_age": age, f"{chan}_q": q}


def tick_inputs(ticks, t0_ns, stamps, cols, chan_rows: dict, track, ax_extrap_s: float = 0.02, fallback_s: float = 1.0) -> dict:
    """All per-tick inputs. chan_rows: {channel: meas-v1 rows or None (absent channel -> all-zero, pres 0)}."""
    out = telemetry_inputs(ticks, t0_ns, stamps, cols, ax_extrap_s, fallback_s)
    N = len(ticks)
    for c in CHANNELS:
        rows = chan_rows.get(c)
        if rows is None:
            out.update({f"{c}_pres": np.zeros(N, np.float32), f"{c}_z": np.zeros((N, 3), np.float32), f"{c}_age": np.zeros(N, np.float32),
                        f"{c}_q": np.zeros((N, QDIM[c]), np.float32)})
        else:
            out.update(channel_inputs(ticks, t0_ns, rows, c, track))
    out["ticks"] = np.asarray(ticks, np.int64)
    return out
