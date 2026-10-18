"""PRG-002 S2.3 contribution analysis: variants of the G1 stack (telemetry + PnP fixes) on train laps 1/3/5, S2.1 harness (read-only imports).

Variants (id -> what): see RUNS. Oracle variants (r5, r6) use GT only to build the input in this evaluator script; they are analysis bounds, NOT models.
GT never enters src/camloc. Train laps 1, 3, 5 only; identical ticks, E04 streams, first poses, telemetry store and G1 fold constants as the S2.1 G1 rows.
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v2_s21_lib as L  # noqa: E402
from s10g_lib import cfg_for, data, run_one, train_store  # noqa: E402
from camloc.filtering.kin2_filter import Kin2Params  # noqa: E402
from camloc.filtering.kin_filter import Meas  # noqa: E402
from camloc.filtering.tel4_filter import Tel4Filter  # noqa: E402
from camloc.filtering.tel4_runtime import Tel4Stream  # noqa: E402
from camloc.filtering.tel_filter import TickRec  # noqa: E402

ROOT = L.ROOT
ART = ROOT / "artifacts/v2/S2.3-runs"
BORDER = ROOT / "artifacts/v2/S3.2d-streams/causal_2hz/e_fit"
START = ROOT / "artifacts/v2/S1.3-eval"
LAPS3 = (1, 3, 5)
RHO = 0.94
ALPHA_INC = 0.2       # fraction of the S1.3 start error that the S3.2d fit corrects (slope 0.21, S3.2d)
GATE_BORDER = 9.0     # chi2(1) ~ 3 sigma

RUNS = {
    "r1": "telemetry only (dead reckoning from the first pose, no camera rows)",
    "r2": "PnP fixes alone per tick, hold last fix (no motion model)",
    "r3": "PnP + retrieval + Y1 constant-velocity filter, no telemetry (Kin2Stream; Y1 has no lost mode)",
    "r4": "G1 (telemetry + fixes)",
    "r5": "ORACLE gating: G1 fed only PnP fixes within 5 m of GT (not a model)",
    "r6": "ORACLE lateral: G1 + n from GT, sigma 0.1 m, 2 Hz, no latency (not a model)",
    "r7a": "G1 + causal S3.2d border stream 2 Hz, sigma x1, absolute n (double-counting)",
    "r7b": "G1 + causal S3.2d border stream 2 Hz, sigma x2, absolute n (double-counting)",
    "r7c": "G1 + border stream, increment form (dependence-aware)",
    "r7d": "G1 + border stream, absolute n, sigma / sqrt(1-0.94^2) (dependence-aware)",
    "r8a": "G1, PnP fixes thinned to every 5th", "r8b": "G1, PnP fixes thinned to every 10th", "r8c": "G1, PnP fixes thinned to every 20th",
}
ORACLE = {"r5", "r6"}


# ------------------------------------------------------------------ evaluator-side GT helpers
def gt_pose(lap: int, ts):
    """GT (x, y, yaw) interpolated at ts (same bracketing as v2_s13_lib.aligned_errors; nan where the 50 ms gap rule drops the stamp)."""
    g = L.gt_arrays()
    a, b = L.LAPS[lap]
    sel = (g["t"] >= a) & (g["t"] < b)
    tg, gx0, gy0, gyaw0 = g["t"][sel], g["x"][sel], g["y"][sel], g["yaw"][sel]
    ts = np.asarray(ts, np.int64)
    i = np.searchsorted(tg, ts)
    ok = (i > 0) & (i < len(tg))
    i = np.clip(i, 1, len(tg) - 1)
    ok &= ~((tg[i] - tg[i - 1]) > 50_000_000)
    f = (ts - tg[i - 1]) / (tg[i] - tg[i - 1])
    yu = np.unwrap(gyaw0)
    out = np.column_stack([gx0[i - 1] + f * (gx0[i] - gx0[i - 1]), gy0[i - 1] + f * (gy0[i] - gy0[i - 1]), yu[i - 1] + f * (yu[i] - yu[i - 1])])
    out[~ok] = np.nan
    return out


def n_of(xyz, s_hint=None):
    tr = L.track()
    s, n, _ = tr.from_xy(float(xyz[0]), float(xyz[1]), float(xyz[2]), s_hint=s_hint)
    return s, n


# ------------------------------------------------------------------ border inputs
def border_rows(lap: int, mode: str, sig_scale: float = 1.0):
    """list of dict(stamp, avail, s, z, sig, inc) per the variant. mode: abs | inc | rho | oracle. Uses the delivered stream (valid rows)."""
    import pandas as pd
    d = pd.read_parquet(BORDER / f"lap{lap}_multi.parquet")
    d = d[d.valid].sort_values("avail_ns")
    out = []
    if mode == "oracle":
        gp = gt_pose(lap, d.stamp_ns.to_numpy())
        for st, s, g in zip(d.stamp_ns, d.s, gp):
            if np.isnan(g[0]):
                continue
            out.append({"stamp": int(st), "avail": int(st), "s": float(n_of(g, float(s))[0]), "z": float(n_of(g, float(s))[1]), "sig": 0.1, "inc": False})
        return out
    if mode == "inc":
        z = np.load(START / f"ticks_v2-T2_lap{lap}.npz")
        ti = np.searchsorted(z["ts"], d.stamp_ns.to_numpy())
        for (st, av, s, n, sg), j in zip(zip(d.stamp_ns, d.avail_ns, d.s, d.n, d.sig_n), ti):
            if j >= len(z["ts"]) or z["ts"][j] != st:
                continue
            ns = n_of((z["x"][j], z["y"][j], z["yaw"][j]), float(s))[1]
            out.append({"stamp": int(st), "avail": int(av), "s": float(s), "z": float(n - ns), "ns": float(ns), "sig": float(inc_sigma(lap)), "inc": True})
        return out
    k = {"abs": sig_scale, "rho": 1.0 / math.sqrt(1.0 - RHO ** 2)}[mode]
    for st, av, s, n, sg in zip(d.stamp_ns, d.avail_ns, d.s, d.n, d.sig_n):
        out.append({"stamp": int(st), "avail": int(av), "s": float(s), "z": float(n), "sig": float(sg) * k * (sig_scale if mode == "rho" else 1.0), "inc": False})
    return out


_SIG_INC = {}


def inc_sigma(lap: int) -> float:
    """Out-of-fold spread of the increment's error: std of (n_fit - n_gt) pooled over the OTHER two laps (GT used on the evaluator side to calibrate sigma only)."""
    if lap not in _SIG_INC:
        import pandas as pd
        e = []
        for o in LAPS3:
            if o == lap:
                continue
            d = pd.read_parquet(BORDER / f"lap{o}_multi.parquet")
            d = d[d.valid]
            gp = gt_pose(o, d.stamp_ns.to_numpy())
            for s, n, g in zip(d.s, d.n, gp):
                if not np.isnan(g[0]):
                    e.append(n - n_of(g, float(s))[1])
        _SIG_INC[lap] = float(np.std(e))
    return _SIG_INC[lap]


# ------------------------------------------------------------------ filter with a lateral (n) update
class BorderG1Filter(Tel4Filter):
    """G1 (Tel4Filter, unchanged) + a 1-D lateral update h = n(x, y) on the track frame (linearised by finite differences), gated chi2 9.
    Border measurements arrive at their avail stamp (first tick >= avail) and are replayed to their image tick like E04 rows."""

    def __init__(self, p, first_pose, t0_ns, rp, border):
        super().__init__(p, first_pose, t0_ns, rp)
        self.border = sorted(border, key=lambda b: b["avail"])
        self._bp = 0
        self.nb = {"used": 0, "gated": 0}
        self.tr = L.track()

    def step(self, rec: TickRec, arrived: list) -> dict:
        k = len(self.recs)
        self._tidx[rec.t_ns] = k
        arrived = list(arrived)
        while self._bp < len(self.border) and self.border[self._bp]["avail"] <= rec.t_ns:
            b = self.border[self._bp]
            self._bp += 1
            key = self._tidx.get(b["stamp"])
            if key is not None:
                arrived.append(Meas(key, k, "border", (b["z"], b["sig"], b["s"], 1.0 if b["inc"] else 0.0, b.get("ns", 0.0))))
        return super().step(rec, arrived)

    def _update(self, st, m, t_key):
        if m.kind != "border":
            return super()._update(st, m, t_key)
        z, sig, sh, inc, ns = m.z
        x, y, yaw = st.x[0], st.x[1], st.x[2]
        n0 = self.tr.from_xy(x, y, yaw, s_hint=sh)[1]
        e = 0.05
        gx = (self.tr.from_xy(x + e, y, yaw, s_hint=sh)[1] - n0) / e
        gy = (self.tr.from_xy(x, y + e, yaw, s_hint=sh)[1] - n0) / e
        al = ALPHA_INC if inc else 1.0
        # increment form (round 2): z = n_fit - n_S13 is predicted from the state as alpha * (n_state - n_S13), H = alpha * grad n
        nu = z - al * (n0 - ns) if inc else z - n0
        H = np.zeros((1, 5))
        H[0, 0], H[0, 1] = al * gx, al * gy
        S = float((H @ st.P @ H.T)[0, 0]) + sig ** 2
        d2 = nu * nu / S
        if d2 > GATE_BORDER:
            self.nb["gated"] += 1
            return "rejected_border", d2
        K = (st.P @ H.T) / S
        st.x = st.x + (K[:, 0] * nu)
        st.x[2] = ((st.x[2] + math.pi) % (2 * math.pi)) - math.pi
        self._clamp(st)
        I_KH = np.eye(5) - K @ H
        st.P = I_KH @ st.P @ I_KH.T + (K * sig ** 2) @ K.T
        st.P = 0.5 * (st.P + st.P.T)
        self.nb["used"] += 1
        return "accepted_border", d2


def make_stream(lap, rows, store, border=None, fold=None, fallback_age_s=1.0):
    fold = lap if fold is None else fold
    tp, kin, rp = cfg_for("g1", fold)
    r0, fp, t0 = data(lap)
    s = Tel4Stream(tp, Kin2Params(**kin), fp, t0, rows, store, 0.0, fallback_age_s=fallback_age_s, rp=rp)
    if border is not None:
        s.filter = BorderG1Filter(tp, fp, t0, rp, border)
    return s, r0


def drive(s, r0):
    ts = np.array([x["t_ns"] for x in r0], dtype=np.int64)
    return ts, [s.tick(int(t)) for t in ts]


# ------------------------------------------------------------------ variants
def pnp_rows(rows):
    return [r for r in rows if r["src"] == "pnp"]


def hold_last_fix(lap, rows, fp):
    ts = np.array([x["t_ns"] for x in rows], dtype=np.int64)
    fx = sorted(pnp_rows(rows), key=lambda r: r["t_ns"])
    out, j, cur = [], 0, fp
    for t in ts:
        while j < len(fx) and fx[j]["t_ns"] <= t:
            cur = (fx[j]["x_m"], fx[j]["y_m"], fx[j]["yaw_rad"])
            j += 1
        out.append({"x": cur[0], "y": cur[1], "yaw": cur[2]})
    return ts, out


def run_variant(rid: str, lap: int):
    """-> dict(ts, X (n,3), fix_t (stamps of the PnP rows the variant could see), extra)."""
    rows, fp, t0 = data(lap)
    store = train_store()
    extra = {}
    if rid == "r2":
        ts, outs = hold_last_fix(lap, rows, fp)
        fix_t = np.array(sorted(r["t_ns"] for r in pnp_rows(rows)))
    elif rid == "r1":
        s, r0 = make_stream(lap, [], store, fallback_age_s=1e6)   # never fall back to the (empty) camera filter: telemetry gaps are held
        ts, outs = drive(s, r0)
        fix_t = np.array([], dtype=np.int64)
        extra = {"fallback_ticks": int(s.mode_counts["fallback"]), "telemetry_ticks": int(s.mode_counts["telemetry"])}
    elif rid == "r3":
        s, r0 = make_stream(lap, rows, None)
        ts, outs = drive(s, r0)
        fix_t = np.array(sorted(r["t_ns"] for r in pnp_rows(rows)))
    else:
        use = rows
        border = None
        if rid == "r5":
            ts0 = np.array([r["t_ns"] for r in rows], dtype=np.int64)
            gp = gt_pose(lap, ts0)
            keep = []
            for r, g in zip(rows, gp):
                if r["src"] != "pnp":
                    keep.append(r)
                elif not np.isnan(g[0]) and math.hypot(r["x_m"] - g[0], r["y_m"] - g[1]) <= 5.0:
                    keep.append(r)
            use = keep
            extra = {"pnp_dropped": int(len(pnp_rows(rows)) - len(pnp_rows(use))), "pnp_total": len(pnp_rows(rows))}
        elif rid.startswith("r8"):
            k = {"r8a": 5, "r8b": 10, "r8c": 20}[rid]
            c, keep = 0, []
            for r in rows:
                if r["src"] == "pnp":
                    if c % k == 0:
                        keep.append(r)
                    c += 1
                else:
                    keep.append(r)
            use = keep
            extra = {"pnp_kept": len(pnp_rows(use)), "pnp_total": c}
        elif rid == "r6":
            border = border_rows(lap, "oracle")
        elif rid == "r7a":
            border = border_rows(lap, "abs", 1.0)
        elif rid == "r7b":
            border = border_rows(lap, "abs", 2.0)
        elif rid == "r7c":
            border = border_rows(lap, "inc")
        elif rid == "r7d":
            border = border_rows(lap, "rho")
        elif rid != "r4":
            raise ValueError(rid)
        s, r0 = make_stream(lap, use, store, border)
        ts, outs = drive(s, r0)
        fix_t = np.array(sorted(r["t_ns"] for r in pnp_rows(use)))
        if border is not None:
            extra.update({"border_n": len(border), **s.filter.nb})
    X = np.array([[o["x"], o["y"], o["yaw"]] for o in outs])
    return {"ts": ts, "X": X, "fix_t": fix_t, "extra": extra}


def time_since_fix(ts, fix_t, t0):
    """seconds since the latest PnP row stamped <= tick (since the lap start before the first)."""
    if len(fix_t) == 0:
        return (ts - t0) * 1e-9
    j = np.searchsorted(fix_t, ts, side="right") - 1
    last = np.where(j >= 0, fix_t[np.clip(j, 0, None)], t0)
    return (ts - last) * 1e-9
