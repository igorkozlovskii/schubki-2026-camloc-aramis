#!/usr/bin/env python3
"""S07c: train-only fault injection on the E04 LOO streams (no dev data). Faults: (w) 5 % of PnP fixes displaced 5-22 m (random direction, inlier counts unchanged);
(g) one 10.8 s window per lap in which all rows are dropped (src 'none'); (b) a burst of 8 consecutive wild PnP fixes displaced by the same vector (persistent wrong).
Compares the frozen/tuned variants with the baseline (hold-last of the stream: PnP else retrieval else previous) on mirror metrics.

    pixi run python scripts/s07c_faults.py --variants k2 k3 k4 --out reports/experiments/S07c-kin/tables/faults_train.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from s07c_lib import LAPS, KinParams, e04_stream, first_pose_for, quick_metrics, run_offline, xyz

ROOT = Path(__import__("os").environ.get("CAMLOC_ASSET_ROOT") or Path(__file__).resolve().parents[1])


def baseline(rows):
    X, last = [], None
    for r in rows:
        if r["src"] in ("pnp", "retrieval"):
            last = (r["x_m"], r["y_m"], r["yaw_rad"])
        X.append(last if last is not None else (0.0, 0.0, 0.0))
    return np.array([r["t_ns"] for r in rows], dtype=np.int64), np.array(X)


def inject(rows, kind, seed):
    rng = np.random.default_rng(seed)
    rows = copy.deepcopy(rows)
    pnp = [i for i, r in enumerate(rows) if r["src"] == "pnp"]
    if kind == "w":
        for i in rng.choice(pnp, size=int(0.05 * len(pnp)), replace=False):
            d, th = rng.uniform(5, 22), rng.uniform(0, 2 * math.pi)
            rows[i]["x_m"] += d * math.cos(th)
            rows[i]["y_m"] += d * math.sin(th)
    elif kind == "g":
        n = int(10.8 / 0.1)
        s = int(rng.integers(100, len(rows) - n - 50))
        for i in range(s, s + n):
            rows[i]["src"] = "none"
    elif kind == "b":
        s = int(rng.integers(100, len(rows) - 100))
        d, th = rng.uniform(8, 22), rng.uniform(0, 2 * math.pi)
        cnt = 0
        for i in range(s, len(rows)):
            if rows[i]["src"] == "pnp":
                rows[i]["x_m"] += d * math.cos(th)
                rows[i]["y_m"] += d * math.sin(th)
                cnt += 1
                if cnt == 8:
                    break
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["k2", "k3", "k4"])
    ap.add_argument("--tuning", type=Path, default=ROOT / "reports/experiments/S07c-kin/tuning")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    P = {v: KinParams(**json.loads((a.tuning / f"{v}.json").read_text())["params"]) for v in a.variants}
    res = {}
    for kind in ("clean", "w", "g", "b"):
        acc = {}
        for lap in (1, 3, 5):
            fp, t0 = first_pose_for(lap)
            for seed in ([0] if kind == "clean" else a.seeds):
                rows = e04_stream(f"train-lap{lap}") if kind == "clean" else inject(e04_stream(f"train-lap{lap}"), kind, seed * 10 + lap)
                ts, X = baseline(rows)
                acc.setdefault("baseline", []).append(quick_metrics(*LAPS[lap], ts, X)["pos_rmse_m"])
                for v, p in P.items():
                    ts, out = run_offline(p, rows, fp, t0)
                    acc.setdefault(v, []).append(quick_metrics(*LAPS[lap], ts, xyz(out))["pos_rmse_m"])
        res[kind] = {k: round(float(np.mean(x)), 3) for k, x in acc.items()}
        print(kind, res[kind], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
