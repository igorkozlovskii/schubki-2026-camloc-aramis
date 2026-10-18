#!/usr/bin/env python3
"""S3.2c CPU step: joint six-camera border fit with the attitude estimated in the fit, S3.2b protocol, meas-v1 streams + GT error tables.

    pixi run python scripts/v2_s32c_run.py --probs artifacts/v2/S3.2b-probs/s0 --out artifacts/v2/S3.2c-streams/<tag> --variants a_map,b_fit,c_fit_tel,d_gt

Protocol = S3.2b `multi` (same perturbed starts: the rng draw sequence over all fl ticks is replayed, so every variant is paired with the S3.2b
`multi` rows), training laps, eligible ticks only (ineligible ticks are not fitted: the attitude chain sees a longer gap there). Each lap is split
into chunks of --chunk consecutive eligible ticks; every chunk starts from the start prior (chunks run in parallel, so a lap restarts the chain
every ~50 s: slightly pessimistic for the fitted-attitude variants).
Variants: a_map = attitude frozen at the map attitude (the S3.2b baseline attitude, with the new prior and search); b_fit = attitude fitted,
causal random-walk prior; c_fit_tel = fitted, prior on top of the S1.8 telemetry attitude (fold model); d_gt = GT attitude fixed (oracle,
evaluator-side input). GT is read here only for the start perturbation, eligibility, the oracle and the error tables; `camloc.v2.border_seg_att`
never sees it. Writes <out>/<variant>/lap<k>_multi.parquet (meas-v1) and lap<k>_multi_eval.parquet (+ attitude estimate and its error vs GT).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, dataclasses, json, math, sys, time
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_s11_common as H  # noqa: E402
import v2_s32b_run as R  # noqa: E402  (read-only: perturbation helpers, telemetry attitude)
from camloc.io import load_static_metadata  # noqa: E402
from camloc.v2 import border_attitude as BA  # noqa: E402
from camloc.v2.border_probe import MEAS_COLUMNS, meas_row  # noqa: E402
from camloc.v2.border_seg import SegFitConfig  # noqa: E402
from camloc.v2.road_attitude import RoadAttitudeMap  # noqa: E402
from camloc.v2.border_seg_att import AttFitConfig, AttMeasurer, AttTracker  # noqa: E402
from camloc.v2.seg import labels as L  # noqa: E402

METHOD = "seg_ll_att"
VARIANTS = ("a_map", "b_fit", "c_fit_tel", "d_gt", "e_fit", "e_frozen", "e_fit_sp3", "e_fit_sm3", "e_frozen_sp3", "e_frozen_sm3")
ATTMAP = H.ROOT / "artifacts/v2/S3.2c-attitude-map/loo{lap}.npz"      # D-017 road-attitude map, fold k excludes lap k
S_OFFSET = {"sp3": 3.0, "sm3": -3.0}
_G = {}


def _init():
    cv2.setNumThreads(1)
    h = H.Harness()
    meta = load_static_metadata(h.store)
    _G["h"], _G["polys"] = h, L.build_road_polys(meta.track_bounds)


def make_cfg(over: dict) -> AttFitConfig:
    over = dict(over)
    seg = SegFitConfig(**over.pop("seg", {}))
    return AttFitConfig(seg=seg, **{k: (tuple(v) if isinstance(v, list) else v) for k, v in over.items()})


def ticks_of(lap, probs, h):
    """eligible fl ticks of a lap with the replayed perturbation draws: list of dicts (k, g, dn, ds, dp)."""
    z = np.load(Path(probs) / f"lap{lap}_fl.npz")
    cams = [c for c in H.CAMERAS if c != "r"]
    imgs = {c: np.load(Path(probs) / f"lap{lap}_{c}.npz")["img_i"] for c in cams if c != "fl"}
    imgs["fl"] = z["img_i"]
    rng = np.random.default_rng(1000 * lap + 7)
    out = []
    for k, t in enumerate(z["tick_ns"]):
        if imgs["fl"][k] < 0:
            continue
        st = int(z["img_stamp_ns"][k])
        g = h.gt_at(st)
        ok, why = h.eligibility(lap, "fl", st, g)
        dn, ds, dp = R.draw_perturb(rng)
        if ok:
            out.append({"k": k, "t": int(t), "st": st, "g": g, "dn": dn, "ds": ds, "dp": dp})
    return out


def job(args):
    variant, lap, ci, chunk, probs, out, masks, over = args
    h, polys = _G["h"], _G["polys"]
    fold = f"loo{lap}"
    cfg = make_cfg(over)
    ctx = h.context(None, Path(masks.format(fold=fold)))
    meas = AttMeasurer(ctx, polys, cfg)
    cams = [c for c in H.CAMERAS if c != "r"]
    P = {c: np.load(Path(probs) / f"lap{lap}_{c}.npz") for c in cams}
    TT = ticks_of(lap, probs, h)
    TT = TT[ci * chunk:(ci + 1) * chunk]
    if not TT:
        return variant, lap, ci, 0, 0.0
    use_map = variant.startswith("e_")
    use_tel = variant == "c_fit_tel"
    tel = R.TelAttitude(fold) if use_tel else None
    ram = RoadAttitudeMap.load(str(ATTMAP).format(lap=lap)) if use_map else None
    s_off = S_OFFSET.get(variant.split("_")[-1], 0.0)
    tracker = AttTracker(cfg, use_tel or use_map, sd0=cfg.sd0_att_map if use_map else None)
    rows, ev = [], []
    t0 = time.time()
    for T in TT:
        k, t, g = T["k"], T["t"], T["g"]
        dn, ds, dp = T["dn"], T["ds"], T["dp"]
        have = [c for c in cams if P[c]["img_i"][k] >= 0]
        items = {}
        for c in have:
            stc = int(P[c]["img_stamp_ns"][k])
            gc = g if c == "fl" else h.gt_at(stc)
            pose0 = R.apply_perturb(gc, dn, ds, dp)
            base = BA.Attitude(z=gc["z"], pitch=gc["pitch"], roll=gc["roll"]) if variant == "d_gt" else None      # evaluator-side oracle
            items[c] = {"prob": P[c]["prob"][k].astype(np.float32) / 255.0, "pose0": pose0, "s_init": gc["s"] + ds, "attitude": base}
        fit_att = variant in ("b_fit", "c_fit_tel") or variant.startswith("e_fit")
        td = None
        prior = None
        if use_tel:
            a = tel.at(None, t)
            td = None if a is None else np.array([a.droll, a.dpitch, a.dz])
        if use_map:       # lookup at the START's own s (+ declared offset for the robustness variants), never the GT s
            dz_, dpi_, dro_ = ram.lookup(items["fl"]["s_init"] + s_off)
            td = np.array([dro_, dpi_, dz_])
        if fit_att:
            prior = tracker.prior(t, td)
        elif use_map:
            prior = (td, tracker._sd0())      # frozen at the map lookup
        m = meas.measure_att(items, ref="fl", att_prior=prior, fit_att=fit_att)
        if fit_att and m.valid:
            tracker.update(t, [m.q["droll"], m.q["dpitch"], m.q["dz"]], [m.q["sd_roll"], m.q["sd_pitch"], m.q["sd_z"]], td)
        r = meas_row(t, lap, "multi", METHOD, m, T["st"], dn, dp, fold)
        r["q_n_points"] = float(m.q.get("n_cams", np.nan))
        rows.append(r)
        en = m.n - g["n"] if m.valid else np.nan
        ed = ((m.dpsi - g["dpsi"] + np.pi) % (2 * np.pi) - np.pi) if m.valid else np.nan
        zb, pb, rb = BA.map_attitude(h.bmap, *R.apply_perturb(g, dn, ds, dp))          # map attitude at the start pose (error tables only)
        pose0 = R.apply_perturb(g, dn, ds, dp)
        s_i, n_i, th_i, _ = h.cline.project(*pose0[:2], s_hint=g["s"] + ds)
        e = {"stamp_ns": t, "camera": "multi", "lap": lap, "eligible": True, "excl_reason": "", "gt_s": g["s"], "gt_n": g["n"], "gt_dpsi": g["dpsi"],
             "gt_sigma_xy": g["sigma_xy"], "speed": g["speed"], "init_n": n_i,
             "init_dpsi": float(((pose0[2] - th_i + np.pi) % (2 * np.pi)) - np.pi), "init_err_n": n_i - g["n"], "init_ds": ds,
             "err_n": en, "err_dpsi": ed, "err_s": (m.s - g["s"]) if m.valid else np.nan, "tick_i": len(rows) - 1,
             "map_err_roll_deg": math.degrees(rb - g["roll"]), "map_err_pitch_deg": math.degrees(pb - g["pitch"]), "map_err_z_m": zb - g["z"],
             "n_evals": m.q.get("n_evals", np.nan), "sd_roll_deg": math.degrees(m.q.get("sd_roll", np.nan)),
             "sd_pitch_deg": math.degrees(m.q.get("sd_pitch", np.nan)), "sd_z_m": m.q.get("sd_z", np.nan)}
        if m.valid and (fit_att or use_map):
            e.update(droll_deg=math.degrees(m.q["droll"]), dpitch_deg=math.degrees(m.q["dpitch"]), dz_m=m.q["dz"],
                     att_err_roll_deg=math.degrees(rb + m.q["droll"] - g["roll"]), att_err_pitch_deg=math.degrees(pb + m.q["dpitch"] - g["pitch"]),
                     att_err_z_m=zb + m.q["dz"] - g["z"])
        ev.append(e)
    d = Path(out) / variant / "parts"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=MEAS_COLUMNS).to_parquet(d / f"lap{lap}_c{ci}.parquet", index=False)
    pd.DataFrame(ev).to_parquet(d / f"lap{lap}_c{ci}_eval.parquet", index=False)
    return variant, lap, ci, len(rows), time.time() - t0


def merge(out, variants, laps):
    for v in variants:
        d = Path(out) / v
        for l in laps:
            parts = [p for p in (d / "parts").glob(f"lap{l}_c[0-9]*.parquet") if not p.stem.endswith("_eval")]
            parts = sorted(parts, key=lambda p: int(p.stem.split("_c")[-1]))
            if not parts:
                continue
            pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_parquet(d / f"lap{l}_multi.parquet", index=False)
            pd.concat([pd.read_parquet(p.with_name(p.stem + "_eval.parquet")) for p in parts], ignore_index=True).to_parquet(d / f"lap{l}_multi_eval.parquet", index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--laps", default="0,1,3,5")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--masks", default=str(H.ROOT / "artifacts/v2/S1.1-masks/{fold}"))
    ap.add_argument("--chunk", type=int, default=100, help="consecutive eligible ticks per chain (restart at each chunk start)")
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--merge-only", action="store_true", help="only merge existing parts into per-lap files")
    ap.add_argument("--fit", default="{}", help="JSON overrides of AttFitConfig (seg: SegFitConfig overrides)")
    a = ap.parse_args(argv)
    laps = [int(x) for x in a.laps.split(",")]
    assert set(laps) <= set(H.TRAIN), "training laps only"
    variants = a.variants.split(",")
    assert set(variants) <= set(VARIANTS)
    over = json.loads(a.fit)
    make_cfg(over)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "run_args.json").write_text(json.dumps({"args": vars(a), "method": METHOD, "config": dataclasses.asdict(make_cfg(over))}, indent=1, default=str))
    if a.merge_only:
        merge(a.out, variants, laps)
        return
    h = H.Harness()
    nch = {l: int(math.ceil(len(ticks_of(l, a.probs, h)) / a.chunk)) for l in laps}
    jobs = [(v, l, ci, a.chunk, a.probs, a.out, a.masks, over) for ci in range(max(nch.values())) for l in laps if ci < nch[l] for v in variants]
    jobs.sort(key=lambda j: (-j[2] * 0, j[1] != 0))     # lap 0 (longest) first
    print(f"{len(jobs)} jobs", flush=True)
    with Pool(min(a.procs, len(jobs)), initializer=_init) as p:
        for v, lap, ci, n, w in p.imap_unordered(job, jobs):
            print(f"{v} lap{lap} chunk{ci}: {n} ticks, {w:.0f} s ({w / max(n, 1):.2f} s/tick)", flush=True)
    merge(a.out, variants, laps)


if __name__ == "__main__":
    main()
