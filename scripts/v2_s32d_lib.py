"""S3.2d shared code: the S3.2c `e_fit` protocol (cached S3.2b probabilities, GT-perturbed starts, eligible ticks, chunked chains) on the GPU fitter."""
from __future__ import annotations
import math, sys, time
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_s11_common as H  # noqa: E402
import v2_s32b_run as R  # noqa: E402
import v2_s32c_run as R3  # noqa: E402  (read-only: ticks_of, make_cfg)
from camloc.io import load_static_metadata  # noqa: E402
from camloc.v2 import border_attitude as BA  # noqa: E402
from camloc.v2.border_probe import MEAS_COLUMNS, meas_row  # noqa: E402
from camloc.v2.border_seg_att import AttTracker  # noqa: E402
from camloc.v2.border_fast import FastMeasurer  # noqa: E402
from camloc.v2.road_attitude import RoadAttitudeMap  # noqa: E402
from camloc.v2.seg import labels as L  # noqa: E402

METHOD = "seg_ll_att_gpu"
ROOT = H.ROOT


class Ctx:
    """Harness + road polygons, built once per process."""
    def __init__(self):
        cv2.setNumThreads(1)
        self.h = H.Harness()
        self.polys = L.build_road_polys(load_static_metadata(self.h.store).track_bounds)
        self._ctx = {}

    def context(self, lap):
        if lap not in self._ctx:
            self._ctx[lap] = self.h.context(None, ROOT / f"artifacts/v2/S1.1-masks/loo{lap}")
        return self._ctx[lap]


def run_lap(cx: Ctx, lap: int, probs: str, map_path: str, over: dict, chunk: int = 60, device="cuda", edge_pad=0.5, **fm_kw):
    """e_fit protocol on one lap. Returns (stream DataFrame meas-v1, eval DataFrame)."""
    h = cx.h
    cfg = R3.make_cfg(over)
    meas = FastMeasurer(cx.context(lap), cx.polys, cfg, device=device, edge_pad=edge_pad, **fm_kw)
    cams = [c for c in H.CAMERAS if c != "r"]
    P = {}
    for c in cams:
        z = np.load(Path(probs) / f"lap{lap}_{c}.npz")
        P[c] = {kk: z[kk] for kk in ("img_i", "img_stamp_ns", "prob")}
    TT = R3.ticks_of(lap, probs, h)
    ram = RoadAttitudeMap.load(map_path)
    rows, ev = [], []
    tracker = None
    for ti, T in enumerate(TT):
        if ti % chunk == 0:
            tracker = AttTracker(cfg, True, sd0=cfg.sd0_att_map)
        k, t, g = T["k"], T["t"], T["g"]
        dn, ds, dp = T["dn"], T["ds"], T["dp"]
        items = {}
        for c in [c for c in cams if P[c]["img_i"][k] >= 0]:
            stc = int(P[c]["img_stamp_ns"][k])
            gc = g if c == "fl" else h.gt_at(stc)
            items[c] = {"prob": P[c]["prob"][k].astype(np.float32) / 255.0, "pose0": R.apply_perturb(gc, dn, ds, dp), "s_init": gc["s"] + ds, "attitude": None}
        dz_, dpi_, dro_ = ram.lookup(items["fl"]["s_init"])         # the start's own s, never the GT s
        td = np.array([dro_, dpi_, dz_])
        prior = tracker.prior(t, td)
        m = meas.measure_att(items, ref="fl", att_prior=prior, fit_att=True)
        if m.valid:
            tracker.update(t, [m.q["droll"], m.q["dpitch"], m.q["dz"]], [m.q["sd_roll"], m.q["sd_pitch"], m.q["sd_z"]], td)
        r = meas_row(t, lap, "multi", METHOD, m, T["st"], dn, dp, f"loo{lap}")
        r["q_n_points"] = float(m.q.get("n_cams", np.nan))
        rows.append(r)
        en = m.n - g["n"] if m.valid else np.nan
        ed = ((m.dpsi - g["dpsi"] + np.pi) % (2 * np.pi) - np.pi) if m.valid else np.nan
        pose0 = R.apply_perturb(g, dn, ds, dp)
        s_i, n_i, th_i, _ = h.cline.project(*pose0[:2], s_hint=g["s"] + ds)
        ev.append({"stamp_ns": t, "camera": "multi", "lap": lap, "eligible": True, "excl_reason": "", "gt_s": g["s"], "gt_n": g["n"], "gt_dpsi": g["dpsi"],
                   "gt_sigma_xy": g["sigma_xy"], "speed": g["speed"], "init_n": n_i,
                   "init_dpsi": float(((pose0[2] - th_i + np.pi) % (2 * np.pi)) - np.pi), "init_err_n": n_i - g["n"], "init_ds": ds,
                   "err_n": en, "err_dpsi": ed, "err_s": (m.s - g["s"]) if m.valid else np.nan, "tick_i": ti,
                   "n_evals": m.q.get("n_evals", np.nan)})
    return pd.DataFrame(rows, columns=MEAS_COLUMNS), pd.DataFrame(ev)


def lossfn(ev: pd.DataFrame, st: pd.DataFrame):
    """Gate-1 weighted tuning loss on valid rows (policy all): median/0.25 + P95/0.75 + gross/0.01 (the three Gate 1 border lines, equal weight)."""
    v = st.valid.to_numpy()
    e = np.abs(ev.err_n.to_numpy()[v])
    if len(e) < 10 or v.mean() < 0.5:
        return float("inf"), {}
    d = {"median": float(np.median(e)), "p95": float(np.quantile(e, .95)), "gross": float((e > 1).mean()), "avail": float(v.mean())}
    return d["median"] / 0.25 + d["p95"] / 0.75 + d["gross"] / 0.01, d
