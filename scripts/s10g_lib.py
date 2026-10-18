"""S10g helpers: fault injectors (reviewer builders), scenario metrics, config loaders and a generic runner for V3 legacy / V3-R / G1-G3. Import-only reuse of s10b/s10f libs (train work only)."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s10f_lib import (LAPS, ROOT, TRAIN_LAPS, Reanchor3, data, errors, full_metrics, jl, objective_lap, tel_params, train_store, v3_over, xyz, y1_fold, y1_frozen)  # noqa: E402,F401
from s10f_report import rp_of  # noqa: E402
from camloc.filtering.kin2_filter import Kin2Params  # noqa: E402
from camloc.filtering.tel3_runtime import Tel3Stream  # noqa: E402
from camloc.filtering.tel4_filter import Reanchor4  # noqa: E402
from camloc.filtering.tel4_runtime import Tel4Stream  # noqa: E402
from camloc.io.telemetry import TelemetryStore  # noqa: E402

REP = ROOT / "reports/experiments/S10g-tel4"
VARIANTS = ("g1", "g2", "g3")
# DECISION_RULE.md item 2: (lo, hi, start)
CAPS = {"t_cap": (5.0, 10.0, 8.0), "t_fresh": (0.8, 4.0, 2.0), "n_sup": (10, 150, 50), "spread_d": (1.0, 4.0, 2.0), "onset_m": (3.0, 12.0, 6.0), "onset_rate": (2.0, 20.0, 8.0), "t_min_d": (0.4, 3.0, 0.5)}
USE = {"g1": ("t_cap", "t_fresh", "n_sup", "spread_d", "onset_m", "onset_rate", "t_min_d"), "g2": ("t_cap", "t_fresh", "n_sup", "spread_d", "onset_m", "onset_rate", "t_min_d"), "g3": ("t_cap", "t_fresh", "n_sup")}
W_BAD = 0.01
WIN_AFTER = 30.0


def make_rp4(variant, over=None):
    d = {k: CAPS[k][2] for k in CAPS}
    d.update(over or {})
    d["n_sup"] = int(round(d["n_sup"]))
    if d["t_fresh"] > d["t_cap"]:
        d["t_fresh"] = d["t_cap"]
    return Reanchor4(mode=variant, **d)


def rp4_of(variant, h):
    return Reanchor4(**jl(REP / f"tuning/{variant}_{'all' if h is None else 'holdout' + str(h)}.json")["reanchor4"])


def cfg_for(kind, fold, rp=None):
    tp = tel_params("V3", v3_over(fold))
    kin = y1_fold(fold) if fold is not None else y1_frozen()
    if rp is None:
        rp = Reanchor3("legacy") if kind == "v3" else rp_of("v3", "r3", fold) if kind == "v3r" else rp4_of(kind, fold)
    return tp, kin, rp


def run_one(kind, lap, fold, rows=None, store=None, fp_shift=0.0, rp=None, ticks=None):
    """kind: v3 (legacy) | v3r (S10f R3) | g1 | g2 | g3 (or any Reanchor4 passed as rp). rows = E04 rows the stream sees (default: the lap's); ticks = tick stamps (default: the lap's rows)."""
    tp, kin, rp = cfg_for(kind, fold, rp)
    r0, fp, t0 = data(lap)
    rows = r0 if rows is None else rows
    fp = (fp[0] + fp_shift, fp[1], fp[2])
    cls = Tel3Stream if isinstance(rp, Reanchor3) else Tel4Stream
    s = cls(tp, Kin2Params(**kin), fp, t0, rows, train_store() if store is None else store, 0.0, rp=rp)
    ts = np.array([x["t_ns"] for x in (r0 if ticks is None else ticks)], dtype=np.int64)
    out = [s.tick(int(t)) for t in ts]
    return ts, out, s


# ----------------------------------------------------------------------------- injectors
def edit_store(store, lap, t0s, dur, fields, fn):
    t = store._t
    c = {k: v.copy() for k, v in store._c.items()}
    a = LAPS[lap][0] + int(t0s * 1e9)
    m = (t > a) & (t <= a + int(dur * 1e9))
    for f in fields:
        c[f][m] = fn(c[f][m])
    return TelemetryStore(t.copy(), c)


def cluster_rows(rows, lap, t0s, dur, mag, ang=60.0):
    r = copy.deepcopy(rows)
    a = LAPS[lap][0] + int(t0s * 1e9)
    b = a + int(dur * 1e9)
    dx, dy = mag * math.cos(math.radians(ang)), mag * math.sin(math.radians(ang))
    for x in r:
        if a <= x["t_ns"] < b and x.get("x_m") is not None:
            x["x_m"] += dx
            x["y_m"] += dy
    return r


def blackout(rows, store, lap, t0s, dur=10.0):
    a = LAPS[lap][0] + int(t0s * 1e9)
    b = a + int(dur * 1e9)
    rr = [x for x in rows if not (a <= x["t_ns"] < b)]
    t = store._t
    keep = ~((t > a) & (t <= b))
    return rr, TelemetryStore(t[keep], {k: v[keep] for k, v in store._c.items()})


def apply(inj, lap, rows, store):
    """-> (rows_for_stream, store, fp_shift). Ticks always come from the full rows."""
    k, p = inj["kind"], inj["p"]
    if k == "nominal":
        return rows, store, 0.0
    if k == "fp":
        return rows, store, float(p["d"])
    if k == "wz":
        return rows, edit_store(store, lap, p["t0"], p["dur"], ["wz_radps"], lambda x: x + p["dw"]), 0.0
    if k == "speed":
        return rows, edit_store(store, lap, p["t0"], p["dur"], ["v_mps", "vx_mps", "vy_mps"], lambda x: x * p["f"]), 0.0
    if k == "cluster":
        return cluster_rows(rows, lap, p["t0"], p["dur"], p["mag"]), store, 0.0
    if k == "blackout":
        rr, st = blackout(rows, store, lap, p["t0"], p["dur"])
        return rr, st, 0.0
    raise ValueError(k)


def _inj(family, kind, t_start, t_end, **p):
    return {"family": family, "kind": kind, "p": p, "t_start": float(t_start), "t_end": float(t_end), "id": f"{family}:" + ",".join(f"{k}={v}" for k, v in p.items())}


def tuning_injections():
    return [_inj("TS1", "fp", 0, 0, d=15.0), _inj("TS2", "wz", 50, 58, t0=50, dur=8, dw=0.07), _inj("TS3", "speed", 55, 60, t0=55, dur=5, f=1.25),
            _inj("TS4", "cluster", 45, 50, t0=45, dur=5, mag=20.0), _inj("TS5", "cluster", 40, 52, t0=40, dur=12, mag=15.0)]


def suite_injections(lap):
    """Item-3 (a2) injections for one lap (DECISION_RULE.md)."""
    L = []
    for t0 in (30, 70):
        L.append(_inj("F1", "wz", t0, t0 + 10, t0=t0, dur=10, dw=0.1))
    for t0 in (35, 75):
        L.append(_inj("F2", "speed", t0, t0 + 6, t0=t0, dur=6, f=1.3))
    for d in (10.0, 40.0):
        L.append(_inj("F3", "fp", 0, 0, d=d))
    for t0 in (30, 70):
        for dur in (3, 6):
            for mag in (15.0, 25.0):
                L.append(_inj("F4a", "cluster", t0, t0 + dur, t0=t0, dur=dur, mag=mag))
        for dur in (10, 15):
            for mag in (8.0, 25.0):
                L.append(_inj("F4b", "cluster", t0, t0 + dur, t0=t0, dur=dur, mag=mag))
    for t0 in ((30, 60) + ((100,) if lap == 5 else ())):
        L.append(_inj("F5", "blackout", t0, t0 + 10, t0=t0, dur=10))
    return L


def scen_metrics(lap, ts, out, inj):
    e = errors(lap, ts, out)
    t = (ts - LAPS[lap][0]) * 1e-9
    dt = np.r_[np.diff(t), np.median(np.diff(t))]
    lo, hi = inj["t_start"], min(inj["t_end"] + WIN_AFTER, float(t[-1]))
    win = (t >= lo) & (t <= hi) & ~np.isnan(e)
    bad = (e > 3.0) & win
    after = win & (t >= inj["t_end"])
    ok = np.nonzero(after & (e < 3.0))[0]
    return {"n_bad": int(bad.sum()), "bad_s": float(dt[bad].sum()), "recovery_s": float(t[ok[0]] - inj["t_end"]) if len(ok) else WIN_AFTER,
            "over10_s": float(dt[after & (e > 10.0)].sum()), "max_m": float(np.nanmax(np.where(win, e, np.nan)))}


def run_inj(kind, lap, fold, inj, rp=None):
    rows, _, _ = data(lap)
    base = train_store()
    rr, st, sh = apply(inj, lap, rows, base)
    ts, out, s = run_one(kind, lap, fold, rows=rr, store=st, fp_shift=sh, rp=rp)
    return ts, out, s, scen_metrics(lap, ts, out, inj) if inj["kind"] != "nominal" else None
