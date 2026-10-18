"""PRG-002 S3.1c data bank: G1 run causally on the same (optionally fault-injected) inputs the v1c model sees, plus the model's per-tick inputs.

Default env (no torch). G1 = Tel4Stream with the S10g G1 constants of a fold (`h1`/`h3`/`h5` = tuned without lap 1/3/5 = the fold's own nested-LOO
set; `all` = the frozen release constants, used by fold 0 whose held-out lap 0 is not in G1's tuning laps 1/3/5).

    pixi run python scripts/v2_s31c_prep.py train [--variants 7]     # laps 0,1,3,5 x variants (v0 clean, v1..v6 fault-injected) x G1 sets
    pixi run python scripts/v2_s31c_prep.py dev                      # dev laps 2/4: inputs + G1 `all` ONLY (no GT, no scoring); run after PROTOCOL.md

Per training lap and variant (artifacts/v2/S3.1c-bank/):
  lap{L}_v{V}.npz         model inputs (camloc.v2.learned.features.tick_inputs on the variant's rows/stores; fl from the E04 rows G1 sees, avail = stamp)
                          + GT targets (evaluator side, scripts/v2_s31_prep.gt_targets; loss target / mask only) + episode list
  g1_lap{L}_{set}_v{V}.npz G1 per tick: x, y, yaw, (s, n, dpsi) in track-v1 (s chained with the previous tick as hint), cov, flags
Lap 0 has no E04 stream: a pseudo-E04 stream is built from the S1.2 LOO fl streams (landmark valid -> pnp, else coarse valid -> retrieval, else none;
stamp = frame stamp, latency 0 like laps 1/3/5), ticks = lap-0 fl frames after 60 s (as S3.1), G1 started from GT at t0 (training lap only).
Fault episodes (simulated, D-047) are applied to the E04 rows (and the telemetry stores) BEFORE both G1 and the feature builder see them.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v2_s13_lib as L13  # noqa: E402
from v2_s13_lib import LAPS, ROOT  # noqa: E402
from camloc.filtering.kin2_filter import Kin2Params  # noqa: E402
from camloc.filtering.tel4_runtime import Tel4Stream  # noqa: E402
from camloc.filtering.tel_runtime import NEEDED, load_store  # noqa: E402
from camloc.io.telemetry import TelemetryStore  # noqa: E402
from camloc.v2.learned.features import RAW_FIELDS, tick_inputs  # noqa: E402
from camloc.v2.meas import from_fusion_e04, to_records  # noqa: E402

OUT = ROOT / "artifacts/v2/S3.1c-bank"
S12 = ROOT / "artifacts/v2/S1.2-streams"
BORDER = ROOT / "artifacts/v2/S3.2d-streams/causal_2hz/e_fit"
TRAIN = (0, 1, 3, 5)
DEV = (2, 4)
SETS = {"all": None, "h1": 1, "h3": 3, "h5": 5}
LAP0_SKIP_S = 60.0
N_VAR = 7                     # v0 clean + 6 fault-injected variants per training lap


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def lap_windows(laps):
    return [(LAPS[l][0] - 3_000_000_000, LAPS[l][1]) for l in laps]


# ------------------------------------------------------------------------------------------------ inputs per lap (inference side)
def e04_rows(lap: int):
    """-> (E04 rows, first pose or None, t0, ticks). Lap 0: pseudo-E04 from the S1.2 LOO fl streams."""
    if lap == 0:
        f = pd.read_csv(ROOT / "artifacts/inputs/split-v1/frames.csv")
        g = f[(f.camera == "fl") & (f.lap == 0)].stamp_ns.to_numpy(np.int64)
        g = np.sort(g[(g >= LAPS[0][0] + int(LAP0_SKIP_S * 1e9)) & (g < LAPS[0][1])])
        t0, ticks = int(g[0]), g[1:]
        lm = pd.read_parquet(S12 / "lap0_fl_landmark.parquet").set_index("stamp_ns")
        co = pd.read_parquet(S12 / "lap0_fl_coarse.parquet").set_index("stamp_ns")
        rows = []
        for i, t in enumerate(ticks):
            r = {"i": i, "t_ns": int(t), "latency_ns": 0, "stream": "e04_fix", "schema": "fusion-stream-v1", "run_id": "train-lap0:pseudo-e04-S1.2", "reason": None}
            src = "none"
            q = None
            if t in lm.index and bool(lm.loc[t, "valid"]):
                q, src = lm.loc[t], "pnp"
            elif t in co.index and bool(co.loc[t, "valid"]):
                q, src = co.loc[t], "retrieval"
            elif t in lm.index:
                q = lm.loc[t]
            r["src"] = src
            if q is not None:
                for a, b in (("n_inl", "q_n_inl"), ("inlier_ratio", "q_inlier_ratio"), ("rms_px", "q_rms_px"), ("d_kf_m", "q_d_kf_m"), ("sim_top", "q_sim_top")):
                    v = q.get(b)
                    r[a] = None if v is None or v != v else float(v)
                if src != "none":
                    r.update(x_m=float(q["x"]), y_m=float(q["y"]), yaw_rad=float(q["yaw"]))
            rows.append(r)
        return rows, None, t0, ticks
    if lap in DEV:
        from s07c_lib import e04_stream
        from s07_fusion_lib import first_pose_for
        rows = e04_stream(f"dev-lap{lap}")
        fp, t0 = first_pose_for(lap)
    else:
        rows, fp, t0 = L13.data(lap)
    rows = copy.deepcopy(rows)
    return rows, fp, int(t0), np.array([r["t_ns"] for r in rows], np.int64)


def side_channels(lap: int) -> dict:
    """fr landmark (S1.2 LOO streams) and border (S3.2d causal) rows; absent channels -> None (masked)."""
    ch = {}
    p = S12 / f"lap{lap}_fr_landmark.parquet"
    ch["fr_lm"] = pd.read_parquet(p).to_dict("records") if p.is_file() else None
    p = BORDER / f"lap{lap}_multi.parquet"
    ch["bd"] = pd.read_parquet(p).to_dict("records") if p.is_file() else None
    return ch


# ------------------------------------------------------------------------------------------------ fault episodes (simulated)
EP_TYPES = (("lat_coarse", 0.30), ("lat_fix", 0.30), ("lat_both", 0.10), ("cluster_fix", 0.15), ("wz_bias", 0.15))


def draw_episodes(lap: int, t0: int, t_end: int, rng) -> list[dict]:
    eps, t = [], t0 * 1e-9 + float(rng.uniform(5, 30))
    while t < t_end * 1e-9 - 5:
        k = rng.choice(len(EP_TYPES), p=[p for _, p in EP_TYPES])
        typ = EP_TYPES[k][0]
        if typ.startswith("lat"):
            e = {"type": typ, "dur": float(rng.uniform(5, 30)), "b": float(rng.choice([-1, 1]) * rng.uniform(2, 5))}
        elif typ == "cluster_fix":
            e = {"type": typ, "dur": float(rng.uniform(3, 15)), "mag": float(rng.uniform(5, 25)), "ang": float(rng.uniform(0, 360))}
        else:
            e = {"type": typ, "dur": float(rng.uniform(3, 10)), "dw": float(rng.choice([-1, 1]) * rng.uniform(0.02, 0.12))}
        e["t0_ns"] = int(t * 1e9)
        e["t1_ns"] = int((t + e["dur"]) * 1e9)
        eps.append(e)
        t += e["dur"] + float(rng.uniform(10, 40))
    return eps


def apply_episodes(rows, stores: list, eps, track):
    rows = copy.deepcopy(rows)
    for e in eps:
        a, b = e["t0_ns"], e["t1_ns"]
        if e["type"] == "wz_bias":
            for i, st in enumerate(stores):
                if st is None or "wz_radps" not in st._c:
                    continue
                c = {k: v.copy() for k, v in st._c.items()}
                m = (st._t > a) & (st._t <= b)
                c["wz_radps"][m] += e["dw"]
                stores[i] = TelemetryStore(st._t.copy(), c, getattr(st, "manifest", {}), getattr(st, "stale_ns", TelemetryStore.DEFAULT_STALE_NS))
            continue
        srcs = {"lat_coarse": ("retrieval",), "lat_fix": ("pnp",), "lat_both": ("pnp", "retrieval"), "cluster_fix": ("pnp",)}[e["type"]]
        for r in rows:
            if a <= r["t_ns"] < b and r.get("src") in srcs and r.get("x_m") is not None:
                if e["type"] == "cluster_fix":
                    r["x_m"] += e["mag"] * math.cos(math.radians(e["ang"]))
                    r["y_m"] += e["mag"] * math.sin(math.radians(e["ang"]))
                else:
                    s, n, dp = track.from_xy(r["x_m"], r["y_m"], r["yaw_rad"])
                    x, y, yaw = track.to_xy(s, n + e["b"], dp)
                    r["x_m"], r["y_m"], r["yaw_rad"] = float(x), float(y), float(yaw)
    return rows, stores


# ------------------------------------------------------------------------------------------------ G1
def run_g1(gset: str, rows, fp, t0, ticks, store, track) -> dict:
    from s10g_lib import cfg_for
    tp, kin, rp = cfg_for("g1", SETS[gset])
    s = Tel4Stream(tp, Kin2Params(**kin), fp, t0, rows, store, 0.0, rp=rp)
    outs = [s.tick(int(t)) for t in ticks]
    X = np.array([[o["x"], o["y"], o["yaw"]] for o in outs])
    snp = np.zeros((len(ticks), 3))
    hint = None
    for k in range(len(ticks)):
        snp[k] = track.from_xy(float(X[k, 0]), float(X[k, 1]), float(X[k, 2]), s_hint=hint)
        hint = snp[k, 0]
    cov = np.array([np.asarray(o["cov"], float)[:3, :3] for o in outs])
    fl = np.array([[o.get("accepted", 0), o.get("rejected", 0), o.get("reanchor", 0), o.get("ret_used", 0), o["mode"] == "telemetry"] for o in outs], np.float32)
    return {"ts": np.asarray(ticks, np.int64), "x": X[:, 0], "y": X[:, 1], "yaw": X[:, 2], "s": snp[:, 0], "n": snp[:, 1], "dpsi": snp[:, 2],
            "cov": cov.astype(np.float32), "flags": fl}


def g1_tick_arrays(g: dict) -> dict:
    """Model-side G1 channel: (s, n, dpsi), log position sigma, flags (all from G1's causal output at the tick)."""
    sig = np.sqrt(np.maximum(g["cov"][:, 0, 0] + g["cov"][:, 1, 1], 1e-6))
    return {"g1_snp": np.column_stack([g["s"], g["n"], g["dpsi"]]).astype(np.float32), "g1_logsig": np.log(sig).astype(np.float32),
            "g1_flags": np.minimum(g["flags"], 1.0).astype(np.float32)}


# ------------------------------------------------------------------------------------------------ build
def build_lap(lap: int, nvar: int, gsets, with_gt: bool, man: dict):
    tr = L13.track()
    rows0, fp, t0, ticks = e04_rows(lap)
    win = lap_windows([lap])
    st_g1 = load_store(L13.TELDIR, columns=NEEDED, windows=win)
    st_raw = load_store(L13.TELDIR, columns=RAW_FIELDS, windows=win)
    side = side_channels(lap)
    if fp is None:   # lap 0 (training only): GT at t0 is the nominal start
        from v2_s31_prep import gt_targets
        g0 = gt_targets(lap, np.array([t0, t0 + 1], np.int64))
        fp = (float(g0["gt_x"][0]), float(g0["gt_y"][0]), float(g0["gt_yaw"][0]))
    G = None
    if with_gt:
        from v2_s31_prep import gt_targets
        G = gt_targets(lap, np.asarray(ticks, np.int64))
    for v in range(nvar):
        t1 = time.time()
        rng = np.random.default_rng(1000 * lap + v)
        eps = [] if v == 0 else draw_episodes(lap, t0, int(ticks[-1]), rng)
        rows, (sg, sr) = apply_episodes(rows0, [st_g1, st_raw], eps, tr) if eps else (rows0, [st_g1, st_raw])
        meas = to_records(from_fusion_e04(rows, tr, lap, 0.0))
        ch = {"fl_lm": [r for r in meas if r["kind"] == "landmark"], "fl_co": [r for r in meas if r["kind"] == "coarse"], **side}
        X = tick_inputs(ticks, t0, sr._t, {f: sr._c[f] for f in sr.fields}, ch, tr)
        s0, n0, p0 = tr.from_xy(*fp)
        meta = {"lap": lap, "variant": v, "t0": t0, "fp": list(fp), "fp_snp": [s0, n0, p0], "n_ticks": int(len(ticks)), "episodes": eps,
                "channels_present": {c: int(X[f"{c}_pres"].sum()) for c in ("fl_lm", "fl_co", "fr_lm", "bd")},
                "first_pose": "GT at t0 (training lap 0)" if lap == 0 else "organizer first pose (S05)"}
        name = f"lap{lap}_v{v}.npz"
        extra = {k: np.asarray(val) for k, val in G.items()} if G is not None else {}
        np.savez_compressed(OUT / name, **X, **extra, meta=json.dumps(meta))
        man["files"][name] = {"sha256": sha(OUT / name), "episodes": len(eps), "gt": with_gt}
        for gs in gsets:
            g = run_g1(gs, rows, fp, t0, ticks, sg, tr)
            gn = f"g1_lap{lap}_{gs}_v{v}.npz"
            np.savez_compressed(OUT / gn, **g, **g1_tick_arrays(g))
            man["files"][gn] = {"sha256": sha(OUT / gn)}
        print(json.dumps({"lap": lap, "v": v, "eps": len(eps), "sec": round(time.time() - t1, 1)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("train", "dev"))
    ap.add_argument("--laps", type=int, nargs="*")
    ap.add_argument("--variants", type=int, default=N_VAR)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mp = ROOT / "manifests/v2/S3.1c-bank.json"
    man = json.loads(mp.read_text()) if mp.is_file() else {"step": "PRG-002 S3.1c", "files": {}, "inputs": {}}
    if a.what == "train":
        for lap in (a.laps or TRAIN):
            assert lap in TRAIN
            build_lap(lap, a.variants, list(SETS), True, man)
        for p in [*sorted(S12.glob("lap[0135]_f[lr]_*.parquet")), *sorted(BORDER.glob("lap[135]_multi.parquet"))]:
            if not p.name.endswith("_eval.parquet"):
                man["inputs"][str(p.relative_to(ROOT))] = sha(p)
    else:
        for lap in (a.laps or DEV):
            assert lap in DEV
            build_lap(lap, 1, ["all"], False, man)    # no GT is read for dev laps here
        for lap in DEV:
            p = ROOT / f"artifacts/streams/S07a-v1/dev-lap{lap}.e04.jsonl.gz"
            man["inputs"][str(p.relative_to(ROOT))] = sha(p)
    mp.write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
