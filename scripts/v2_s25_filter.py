#!/usr/bin/env python3
"""S2.5 stage 4 (EVALUATOR): G1 + pole bearings, and the no-landmark mode, on held-out train laps 1/3/5.

Filter: S2.3's BorderG1Filter (G1 = Tel4Filter unchanged + optional S3.2d border n update, run 7d form) + a bearing-only EKF update
per associated landmark. The pole stream is the GT-free detection cache (fold mask, fold map camera offsets); association happens
INSIDE the filter at the state of the image tick (prediction-gated: chi2(1) gate on nu^2 / (H P H^T + sig^2), plus the isolation,
height, polarity and ambiguity rules of camloc.v2.poles.associate). Measurements arrive at avail = image stamp + 50 ms and are
replayed to their tick like the E04/border rows. GT is used only by the metrics.

Runs (RUNS below); results artifacts/v2/S2.5-filter/<run>_lap<k>.npz/.json.
    pixi run python scripts/v2_s25_filter.py --runs g1 g1p tb tbp tp --procs 15
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_s23_lib as S23  # noqa: E402
from camloc.filtering.kin_filter import Meas  # noqa: E402
from camloc.v2 import body_masks  # noqa: E402
from camloc.v2.border_attitude import map_attitude  # noqa: E402
from camloc.v2.poles.associate import AssocConfig, associate, bearing_sigma  # noqa: E402
from camloc.v2.poles.detect import mask_keep  # noqa: E402
from camloc.v2.poles.geometry import bearing_jacobian, predict_bearing, wrap  # noqa: E402

ROOT = S23.ROOT
DET = ROOT / "artifacts/v2/S2.5-det/night"
MAPS = ROOT / "artifacts/maps"
OUT = ROOT / "artifacts/v2/S2.5-filter"
CAMS = ("fl", "fr", "cl", "cr")
LATENCY_NS = 50_000_000
GATE_CHI2 = 9.0          # chi2(1), 3 sigma
SIG_INFLATE = 2.0        # default; bearings of one landmark over consecutive frames share its map error (variant suffix _k<k>)
GATE_MAX_M = 3.0         # the bearing gate never exceeds atan(3 m / r) (prevents association when the state is lost)

RUNS = {
    "g1": "G1 (telemetry + PnP fixes) -- reproduces S2.3 r4",
    "g1p": "G1 + pole bearings",
    "tb": "no-landmark: telemetry + S3.2d borders (7d form), no PnP / retrieval",
    "tbp": "no-landmark: telemetry + borders + pole bearings",
    "tp": "no-landmark: telemetry + pole bearings (no borders)",
}


def pole_stream(lap: int):
    """GT-free: per (tick, camera) frame the masked detections with the fold map's camera offset."""
    fold = f"loo{lap}"
    meta = json.loads((MAPS / f"V2-S2.5-poles-{fold}" / "map.json").read_text())
    off = meta.get("cam_az_offset_rad", {})
    masks = body_masks.load_masks(ROOT / f"artifacts/v2/S1.1-masks/{fold}")
    out = []
    for cam in CAMS:
        d = pd.read_parquet(DET / f"lap{lap}_{cam}.parquet")
        d = d[mask_keep(d[["u1", "v1", "u2", "v2"]].to_numpy(), masks[cam])]
        for (tick, img), q in d.groupby(["tick_ns", "img_ns"]):
            out.append({"stamp": int(tick), "avail": int(img) + LATENCY_NS, "cam": cam, "az": q.az.to_numpy() + off.get(cam, 0.0),
                        "el": 0.5 * (q.el_top.to_numpy() + q.el_bot.to_numpy()), "pol": q.pol.to_numpy()})
    return sorted(out, key=lambda r: r["avail"])


class PoleFilter(S23.BorderG1Filter):
    def __init__(self, p, first_pose, t0_ns, rp, border, poles, lm, cams, bmap, cfg=AssocConfig(), k_sig=SIG_INFLATE):
        super().__init__(p, first_pose, t0_ns, rp, border or [])
        self.k_sig = k_sig
        self.poles = poles
        self._pp = 0
        self.lm = lm
        self.lm_xy = lm[["x", "y"]].to_numpy()
        self.lm_z, self.lm_pol = lm.zmid.to_numpy(), lm.pol.to_numpy()
        self.cams, self.bmap, self.cfg = cams, bmap, cfg
        self._last = {}                               # id(frame) -> (n_assoc, n_used, n_gated) of the latest (re)play

    def step(self, rec, arrived):
        k = len(self.recs)
        self._tidx[rec.t_ns] = k
        arrived = list(arrived)
        while self._pp < len(self.poles) and self.poles[self._pp]["avail"] <= rec.t_ns:
            m = self.poles[self._pp]
            self._pp += 1
            key = self._tidx.get(m["stamp"])          # tick index of the image tick
            if key is not None:
                arrived.append(Meas(key, k, "pole", m))
        return super().step(rec, arrived)

    def _update(self, st, m, t_key):
        if m.kind != "pole":
            return super()._update(st, m, t_key)
        d = m.z
        cam = self.cams[d["cam"]]
        x, y, yaw = float(st.x[0]), float(st.x[1]), float(st.x[2])
        P3 = st.P[:3, :3]
        tbc = cam.T_base_cam.t

        def gate_fn(i, r):
            Hj = bearing_jacobian(x, y, yaw, tbc, self.lm_xy[i, 0], self.lm_xy[i, 1])
            sig = self.k_sig * bearing_sigma(self.lm.iloc[i], r, self.cfg)
            g = math.sqrt(GATE_CHI2 * (float(Hj @ P3 @ Hj) + sig ** 2))
            return min(g, math.atan(GATE_MAX_M / max(r, 1.0)))

        zb, _, _ = map_attitude(self.bmap, x, y, yaw)
        A = associate(d["az"], (x, y, yaw), cam, self.lm_xy, self.cfg, gate_fn=gate_fn, det_el=d["el"], lm_z=self.lm_z,
                      cam_z=zb + tbc[2], det_pol=d["pol"], lm_pol=self.lm_pol)
        used, gated = 0, 0
        for a in A:
            i = a["lm"]
            X, Y = self.lm_xy[i]
            x, y, yaw = float(st.x[0]), float(st.x[1]), float(st.x[2])
            bp, r = predict_bearing(x, y, yaw, tbc, X, Y)
            nu = float(wrap(a["beta"] - bp))
            H = np.zeros((1, 5))
            H[0, :3] = bearing_jacobian(x, y, yaw, tbc, X, Y)
            sig = self.k_sig * bearing_sigma(self.lm.iloc[i], float(r), self.cfg)
            S = float((H @ st.P @ H.T)[0, 0]) + sig ** 2
            if nu * nu / S > GATE_CHI2:
                gated += 1
                continue
            K = (st.P @ H.T) / S
            st.x = st.x + K[:, 0] * nu
            st.x[2] = ((st.x[2] + math.pi) % (2 * math.pi)) - math.pi
            self._clamp(st)
            I_KH = np.eye(5) - K @ H
            st.P = I_KH @ st.P @ I_KH.T + (K * sig ** 2) @ K.T
            st.P = 0.5 * (st.P + st.P.T)
            used += 1
        self._last[id(d)] = (len(A), used, gated)
        return ("accepted_pole" if used else "no_pole"), float(used)


def run(rid: str, lap: int):
    k_sig = SIG_INFLATE
    if "_k" in rid:
        rid, ks = rid.split("_k")
        k_sig = float(ks)
    rows, fp, t0 = S23.data(lap)
    store = S23.train_store()
    use_fix = rid in ("g1", "g1p")
    border = S23.border_rows(lap, "rho") if rid in ("tb", "tbp") else None
    poles = pole_stream(lap) if rid in ("g1p", "tbp", "tp") else []
    s, r0 = S23.make_stream(lap, rows if use_fix else [], store, border, fallback_age_s=1.0 if use_fix else 1e6)
    if poles or border is not None:
        import v2_s11_common as H
        h = H.Harness()
        lm = pd.read_parquet(MAPS / f"V2-S2.5-poles-loo{lap}" / "landmarks.parquet")
        tp, kin, rp = S23.cfg_for("g1", lap)
        s.filter = PoleFilter(tp, fp, t0, rp, border, poles, lm, h.cams, h.bmap, k_sig=k_sig)
    ts, outs = S23.drive(s, r0)
    X = np.array([[o["x"], o["y"], o["yaw"]] for o in outs])
    last = getattr(s.filter, "_last", {})
    extra = {"pole_frames": len(last), "pole_assoc": int(sum(v[0] for v in last.values())), "pole_used": int(sum(v[1] for v in last.values())),
             "pole_gated": int(sum(v[2] for v in last.values()))} if last else {}
    extra.update(getattr(s.filter, "nb", {}))
    return ts, X, extra


def job(a):
    rid, lap = a
    t = time.perf_counter()
    ts, X, extra = run(rid, lap)
    e = S23.L.aligned_errors(lap, ts, X)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{rid}_lap{lap}.npz", ts=ts, X=X, **{k: e[k] for k in ("pos", "lat", "lon", "dyaw_deg")})
    m = S23.L.lap_metrics(lap, ts, X)
    (OUT / f"{rid}_lap{lap}.json").write_text(json.dumps({"rid": rid, "lap": lap, "desc": RUNS[rid.split("_k")[0]] + (f" (sigma x{rid.split('_k')[1]})" if "_k" in rid else ""), "wall_s": time.perf_counter() - t,
                                                          "extra": extra, "metrics": m}, default=float))
    return rid, lap, round(time.perf_counter() - t, 1), round(m["pos_rmse_m"], 4), extra


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=list(RUNS))
    ap.add_argument("--laps", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--procs", type=int, default=15)
    a = ap.parse_args()
    jobs = [(r, l) for r in a.runs for l in a.laps]
    with Pool(min(a.procs, len(jobs))) as p:
        for x in p.imap_unordered(job, jobs):
            print(*x, flush=True)
