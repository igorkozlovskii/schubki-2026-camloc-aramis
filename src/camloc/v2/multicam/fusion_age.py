"""Frame-age-aware construction of the G1 input rows (S1.16 Amendment 4).

A non-clock camera's frame is the latest one stamped <= the tick; when its stamp lies a few microseconds after the tick it is the frame of the PREVIOUS tick (age ~100 ms,
2-2.5 m of travel at race speed). A fix computed from such a frame describes the pose at the image time (tick - age), not at the tick. The rows built here are keyed to the tick
nearest to the image time (the stream's own `q_cam_age_ms`, an inference-side quantity), and agreement flags compare fixes of the same image time only.
"""
from __future__ import annotations

import copy

import numpy as np

from camloc.v2.multicam.fusion import _opt, confirm_flags


def image_stamp(stamp_ns: int, age_ms: float) -> int:
    return int(stamp_ns - round(float(age_ms) * 1e6))


def build_rows_age(rows0: list, ticks: np.ndarray, streams: dict, cams: tuple, kappa: dict, conf_D: float | None, mode: str = "zero", extra: bool = True, tol_ns: int = 1_000_000) -> tuple:
    """-> (rows, tags, info). Extra-camera rows are keyed to the tick of their image; rows whose image time has no tick within tol_ns are dropped (counted in info)."""
    rows = copy.deepcopy(rows0)
    if mode == "zero":
        for r in rows:
            r["latency_ns"] = 0   # fl rows: stamped at their image (clock camera)
    base = {int(r["t_ns"]): r for r in rows}
    tarr = np.asarray(ticks, np.int64)
    fixes = {"fl": {int(r["t_ns"]): (r["x_m"], r["y_m"]) for r in rows if r["src"] == "pnp"}}
    keyed = {}
    info = {"dropped_no_tick": 0, "age_shifted": 0}
    for c in cams:
        if c == "fl":
            continue
        df = streams[c]
        df = df[df.valid]
        out = []
        for q in df.itertuples():
            img = image_stamp(int(q.stamp_ns), q.q_cam_age_ms if q.q_cam_age_ms == q.q_cam_age_ms else 0.0)
            j = int(np.searchsorted(tarr, img))
            cand = [k for k in (j - 1, j) if 0 <= k < len(tarr)]
            if not cand:
                info["dropped_no_tick"] += 1
                continue
            k = min(cand, key=lambda i: abs(int(tarr[i]) - img))
            if abs(int(tarr[k]) - img) > tol_ns or int(tarr[k]) not in base:
                info["dropped_no_tick"] += 1
                continue
            info["age_shifted"] += int(int(tarr[k]) != int(q.stamp_ns))
            out.append((int(tarr[k]), q, img))
        keyed[c] = out
        fixes[c] = {}
        for key, q, _img in out:
            fixes[c][key] = (float(q.x), float(q.y))
    flags = confirm_flags(fixes, conf_D) if conf_D is not None else {c: {} for c in fixes}
    tags = {}
    k1 = kappa.get("fl", {"kxy": 1.0, "kyaw": 1.0})
    for r in rows:
        if r["src"] == "pnp":
            tags[(int(r["t_ns"]), round(float(r["x_m"]), 5), round(float(r["y_m"]), 5))] = {"cam": "fl", **k1, "conf": bool(flags["fl"].get(int(r["t_ns"]), False))}
    n_extra = 0
    if extra:
        new = []
        for c, lst in keyed.items():
            k = kappa[c]
            for key, q, img in lst:
                b = base[key]
                r = {kk: v for kk, v in b.items() if kk not in ("x_m", "y_m", "yaw_rad")}
                lat = int(q.avail_ns - key) if mode == "delivered" else max(0, int(img) - int(key))   # Amendment 6: never usable before its own image stamp
                r.update(src="pnp", x_m=float(q.x), y_m=float(q.y), yaw_rad=float(q.yaw), latency_ns=lat, n_inl=_opt(q.q_n_inl), inlier_ratio=_opt(q.q_inlier_ratio), rms_px=_opt(q.q_rms_px),
                         d_kf_m=_opt(q.q_d_kf_m), sim_top=_opt(q.q_sim_top), reason=c)
                new.append(r)
                tags[(key, round(r["x_m"], 5), round(r["y_m"], 5))] = {"cam": c, **k, "conf": bool(flags[c].get(key, False))}
        rows = rows + new
        n_extra = len(new)
    info["n_extra_rows"] = n_extra
    return rows, tags, info
