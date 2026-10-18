#!/usr/bin/env python3
"""S3.2b step B (CPU): border measurements (n, dpsi) from stored probability maps, meas-v1 streams + GT error tables, S1.1 protocol.

    pixi run python scripts/v2_s32b_run.py --probs artifacts/v2/S3.2b-probs/<tag> --out artifacts/v2/S3.2b-streams/<tag> [--laps 3 --cams fr --variants perturbed]

Protocol = S1.1 (scripts/v2_s11_run_probe.py): training laps 0/1/3/5, every 5th train-eligible fl tick, each camera's latest frame <= tick, the
initial pose = GT perturbed by `border_probe.PERTURBATION` (seed 1000 * lap + camera index, same draw order as S1.1), fold k's masks.
Differences: (1) the along-track hint passed to the measurement is the perturbed start's own s (`g.s + ds`), not the GT s;
(2) the pose is corrected with the *published* calibration (the S1.4 labels were rendered with it), no fitted offsets.
Variants (directory names): perturbed = headline (map attitude, perturbed start); gtstart = exact GT start; perturbed_gtatt / gtstart_gtatt =
oracle: GT z / pitch / roll replace the map attitude; perturbed_telatt = attitude from allowed telemetry (S1.8 model, if its yaml exists).
`multi` = joint fit of all cameras that have a frame at the tick (one correction (dn, dpsi) in track coordinates, per-camera pose0 at its own
image stamp; reference = fl, age 0). GT is read here (perturbation, eligibility, error tables) and never inside `camloc.v2.border_seg`.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, math, sys, time
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_s11_common as H  # noqa: E402
from camloc.io import load_static_metadata  # noqa: E402
from camloc.v2 import border_attitude as BA  # noqa: E402
from camloc.v2.border_probe import MEAS_COLUMNS, meas_row  # noqa: E402
from camloc.v2.border_seg import SegFitConfig, SegMeasurer  # noqa: E402
from camloc.v2.seg import labels as L  # noqa: E402

METHOD = "seg_ll"
VARIANTS = ("perturbed", "gtstart", "perturbed_gtatt", "gtstart_gtatt", "perturbed_telatt")
ATT_YAML = H.ROOT / "configs/v2/s18/attitude_{fold}.yaml"
_G = {}


def _init():
    cv2.setNumThreads(1)
    h = H.Harness()
    meta = load_static_metadata(h.store)
    _G["h"], _G["polys"] = h, L.build_road_polys(meta.track_bounds)


def apply_perturb(g, dn, ds, dp):
    th = g["theta"]
    return (g["x"] + ds * math.cos(th) - dn * math.sin(th), g["y"] + ds * math.sin(th) + dn * math.cos(th), g["yaw"] + dp)


def draw_perturb(rng):
    from camloc.v2.border_probe import PERTURBATION as P
    dn = rng.uniform(-P["dn_m"], P["dn_m"])
    ds = rng.uniform(-P["ds_m"], P["ds_m"])
    dp = math.radians(rng.uniform(-P["dpsi_deg"], P["dpsi_deg"]))
    return dn, ds, dp          # same draw order as H.Harness.perturb


class TelAttitude:
    """S1.8 telemetry attitude model (allowed telemetry only), loaded lazily per fold."""
    def __init__(self, fold):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import v2_s18_common as C
        self.C = C
        self.model = BA.AttitudeModel.from_yaml(Path(str(ATT_YAML).format(fold=fold)))
        self.st, _, self.F = C.load_telemetry()

    def at(self, ctx_pose, t_ns):
        f = self.C.feature_at(self.st, self.F, t_ns)
        if f is None:
            return None
        return self.model.attitude(f)


def attitude_for(variant, g, tel, t_ns):
    if variant.endswith("_gtatt"):
        return BA.Attitude(z=g["z"], pitch=g["pitch"], roll=g["roll"])            # evaluator-side oracle
    if variant.endswith("_telatt"):
        d = tel.at(None, t_ns)
        return d                                                                  # additive deltas on the map attitude
    return None


def job(args):
    lap, cam, probs, out, variants, masks, every, fitcfg = args
    h, polys = _G["h"], _G["polys"]
    fold = f"loo{lap}"
    ctx = h.context(None, Path(masks.format(fold=fold)))
    meas = SegMeasurer(ctx, polys, SegFitConfig(**fitcfg))
    tel = TelAttitude(fold) if "perturbed_telatt" in variants else None
    joint = cam == "multi"
    cams_all = [c for c in H.CAMERAS if c != "r"]
    cams = cams_all if joint else [cam]
    P = {c: np.load(Path(probs) / f"lap{lap}_{c}.npz") for c in cams}
    z0 = P[cams[0]]
    ticks = z0["tick_ns"]
    rng = np.random.default_rng(1000 * lap + (H.CAMERAS.index(cam) if not joint else 7))
    rows = {v: [] for v in variants}
    ev = {v: [] for v in variants}
    t0 = time.time()
    for k, t in enumerate(ticks):
        # frames available at this tick (img_i < 0: none)
        have = [c for c in cams if P[c]["img_i"][k] >= 0]
        if (joint and "fl" not in have) or (not joint and cam not in have):
            continue
        ref = "fl" if joint else cam
        st_ref = int(P[ref]["img_stamp_ns"][k])
        g = h.gt_at(st_ref)
        ok, why = h.eligibility(lap, ref, st_ref, g)
        dn, ds, dp = draw_perturb(rng)
        s_i, n_i, th_i, _ = h.cline.project(*apply_perturb(g, dn, ds, dp)[:2], s_hint=g["s"] + ds)   # start's own (s, n): the filter's estimate
        for v in variants:
            gts = v.startswith("gtstart")
            items = {}
            for c in (have if joint else [cam]):
                stc = int(P[c]["img_stamp_ns"][k])
                gc = g if c == ref else h.gt_at(stc)
                pose0 = (gc["x"], gc["y"], gc["yaw"]) if gts else apply_perturb(gc, dn, ds, dp)
                items[c] = {"prob": P[c]["prob"][k].astype(np.float32) / 255.0, "pose0": pose0,
                            "s_init": gc["s"] + (0.0 if gts else ds), "attitude": attitude_for(v, gc, tel, stc)}
            m = meas.measure_joint(items, ref=ref)
            r = meas_row(t, lap, "multi" if joint else cam, METHOD, m, st_ref, 0.0 if gts else dn, 0.0 if gts else dp, fold)
            if joint:
                r["q_n_points"] = float(m.q.get("n_cams", np.nan))
            rows[v].append(r)
            en = m.n - g["n"] if m.valid else np.nan
            ed = ((m.dpsi - g["dpsi"] + np.pi) % (2 * np.pi) - np.pi) if m.valid else np.nan
            ev[v].append({"stamp_ns": int(t), "camera": r["camera"], "lap": lap, "eligible": bool(ok), "excl_reason": why, "gt_s": g["s"],
                          "gt_n": g["n"], "gt_dpsi": g["dpsi"], "gt_sigma_xy": g["sigma_xy"], "speed": g["speed"],
                          "init_n": n_i, "init_dpsi": float(((apply_perturb(g, dn, ds, dp)[2] - th_i + np.pi) % (2 * np.pi)) - np.pi) if not gts else g["dpsi"],
                          "init_err_n": (n_i - g["n"]) if not gts else 0.0, "init_ds": 0.0 if gts else ds, "err_n": en, "err_dpsi": ed,
                          "err_s": (m.s - g["s"]) if m.valid else np.nan})
    name = "multi" if joint else cam
    for v in variants:
        d = Path(out) / v
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows[v], columns=MEAS_COLUMNS).to_parquet(d / f"lap{lap}_{name}.parquet", index=False)
        pd.DataFrame(ev[v]).to_parquet(d / f"lap{lap}_{name}_eval.parquet", index=False)
    return lap, name, len(rows[variants[0]]), time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--laps", default="0,1,3,5")
    ap.add_argument("--cams", default="fl,fr,cl,cr,rl,rr,multi")
    ap.add_argument("--variants", default="perturbed,gtstart,perturbed_gtatt,gtstart_gtatt")
    ap.add_argument("--masks", default=str(H.ROOT / "artifacts/v2/S1.1-masks/{fold}"))
    ap.add_argument("--every", type=int, default=H.SAMPLE_EVERY)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--fit", default="{}", help="JSON overrides of SegFitConfig")
    a = ap.parse_args(argv)
    laps = [int(x) for x in a.laps.split(",")]
    assert set(laps) <= set(H.TRAIN), "training laps only"
    variants = a.variants.split(",")
    assert set(variants) <= set(VARIANTS)
    fitcfg = json.loads(a.fit)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "run_args.json").write_text(json.dumps({"args": vars(a), "method": METHOD}, indent=1))
    jobs = [(l, c, a.probs, a.out, variants, a.masks, a.every, fitcfg) for c in a.cams.split(",") for l in laps]
    with Pool(min(a.procs, len(jobs)), initializer=_init) as p:
        for lap, cam, n, w in p.imap_unordered(job, jobs):
            print(f"lap{lap} {cam}: {n} ticks, {w:.0f} s", flush=True)


if __name__ == "__main__":
    main()
