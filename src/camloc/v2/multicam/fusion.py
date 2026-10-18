"""Pure helpers for S1.16 stage 1: confirmation flags from same-tick fixes, per-camera noise scales, and construction of the G1 input rows.

All inputs are inference-side quantities (the cameras' meas-v1 fix streams). The noise scales are CALIBRATED by the caller on other laps (evaluator side, the only place
that sees ground truth) and passed in as numbers.
"""
from __future__ import annotations

import copy

import numpy as np


def _opt(v):
    return None if v is None or v != v else float(v)


def confirm_flags(fixes: dict, D: float) -> dict:
    """fixes: {cam: {stamp_ns: (x, y)}}. -> {cam: {stamp_ns: bool}}; True when another camera has a fix of the SAME stamp within D metres. Same-stamp only (causal)."""
    out = {c: {} for c in fixes}
    stamps = set()
    for d in fixes.values():
        stamps |= set(d)
    for t in stamps:
        have = [(c, d[t]) for c, d in fixes.items() if t in d]
        for c, (x, y) in have:
            out[c][t] = any(c2 != c and np.hypot(x - x2, y - y2) <= D for c2, (x2, y2) in have)
    return out


def kappa_from_errors(errs: dict, ref: str = "fl", floor: float = 1.0) -> dict:
    """errs: {cam: (pos_err_array, yaw_err_deg_array)} from OTHER laps. kappa_c = max(floor, median_c / median_ref) for position and yaw."""
    mp = {c: float(np.nanmedian(e[0])) for c, e in errs.items()}
    my = {c: float(np.nanmedian(e[1])) for c, e in errs.items()}
    return {c: {"kxy": max(floor, mp[c] / mp[ref]), "kyaw": max(floor, my[c] / my[ref])} for c in errs}


def build_rows(rows0: list, streams: dict, cams: tuple, kappa: dict, conf_D: float | None, mode: str = "zero", extra: bool = True) -> tuple:
    """G1 input rows + tag dictionary.

    rows0: the production fl rows (E04 stream rows of the lap; fl PnP rows keep their own fix and latency). streams: {cam: DataFrame of meas-v1 landmark rows (S1.2, held-out map)}.
    extra: add the other cameras' valid fixes as pnp rows (arm b/c); False = only fl measurements (arm a; other cameras serve only for confirmation).
    conf_D: distance for the confirmation flags (None = no flags, all False).
    mode 'zero': every row has latency 0 (G1's harness convention); 'delivered': fl rows keep their own stream latency, extra cameras' rows latency = avail_ns - stamp_ns.
    Returns (rows, tags, n_extra_rows).
    """
    rows = copy.deepcopy(rows0)
    if mode == "zero":
        for r in rows:
            r["latency_ns"] = 0
    base = {int(r["t_ns"]): r for r in rows}
    fixes = {"fl": {int(r["t_ns"]): (r["x_m"], r["y_m"]) for r in rows if r["src"] == "pnp"}}
    ext = {}
    for c in cams:
        if c == "fl":
            continue
        df = streams[c]
        df = df[df.valid]
        ext[c] = df
        fixes[c] = {int(q.stamp_ns): (float(q.x), float(q.y)) for q in df.itertuples() if int(q.stamp_ns) in base}
    flags = confirm_flags(fixes, conf_D) if conf_D is not None else {c: {} for c in fixes}
    tags = {}
    k1 = kappa.get("fl", {"kxy": 1.0, "kyaw": 1.0})
    for r in rows:
        if r["src"] == "pnp":
            tags[(int(r["t_ns"]), round(float(r["x_m"]), 5), round(float(r["y_m"]), 5))] = {"cam": "fl", **k1, "conf": bool(flags["fl"].get(int(r["t_ns"]), False))}
    n_extra = 0
    if extra:
        new = []
        for c, df in ext.items():
            k = kappa[c]
            for q in df.itertuples():
                st = int(q.stamp_ns)
                b = base.get(st)
                if b is None:
                    continue
                r = {kk: v for kk, v in b.items() if kk not in ("x_m", "y_m", "yaw_rad")}
                r.update(src="pnp", x_m=float(q.x), y_m=float(q.y), yaw_rad=float(q.yaw), latency_ns=int(q.avail_ns - q.stamp_ns) if mode == "delivered" else 0,
                         n_inl=_opt(q.q_n_inl), inlier_ratio=_opt(q.q_inlier_ratio), rms_px=_opt(q.q_rms_px), d_kf_m=_opt(q.q_d_kf_m), sim_top=_opt(q.q_sim_top), reason=c)
                new.append(r)
                tags[(st, round(r["x_m"], 5), round(r["y_m"], 5))] = {"cam": c, **k, "conf": bool(flags[c].get(st, False))}
        rows = rows + new
        n_extra = len(new)
    return rows, tags, n_extra
