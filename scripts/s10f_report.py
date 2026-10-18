#!/usr/bin/env python3
"""S10f tables on TRAIN streams (evaluator-side numpy mirror; GT only on the evaluation side).

    pixi run python scripts/s10f_report.py select       # R1/R2/R3 selection by nested J_nominal (DECISION_RULE.md)
    pixi run python scripts/s10f_report.py nested       # rule item (a) for V3-R and Y1-R + forensics of every re-anchor / held event (old and new logic)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from s10f_lib import (LAPS, REP, TRAIN_LAPS, Reanchor3, data, errors, events, jl, pooled_stats, run3, tel_params, train_store, v3_over, y1_fold)
from s07e_lib import run as kin_run

TAB = REP / "tables"
EV_WIN = (17.5, 19.5)


def rp_of(target, variant, h):
    return Reanchor3(**jl(REP / f"tuning/{target}_{variant}_{'all' if h is None else 'holdout' + str(h)}.json")["reanchor3"])


def sel_variant():
    return json.loads((TAB / "selection.json").read_text())["selected"]


def select():
    TAB.mkdir(parents=True, exist_ok=True)
    J = {v: [jl(REP / f"tuning/v3_{v}_holdout{h}.json")["heldout"]["J_nominal"] for h in TRAIN_LAPS] for v in ("r1", "r2", "r3")}
    mean = {v: float(np.mean(x)) for v, x in J.items()}
    best = min(mean, key=mean.get)
    order = ["r3", "r1", "r2"]                      # simplicity order
    sel = next(v for v in order if mean[v] <= mean[best] * 1.01)
    out = {"nested_J_nominal_per_lap": J, "mean": mean, "best": best, "selected": sel, "rule": "lowest mean nested J_nominal; ties within 1 % -> simpler (R3 < R1 < R2)"}
    (TAB / "selection.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))


def lap_run(kind, h, variant=None):
    """kind: v3 (legacy V3 nested), v3r (new logic nested fold), y1 (Y1-L0 nested), y1r."""
    rows, fp, t0 = data(h)
    kin = y1_fold(h)
    if kind == "v3":
        ts, out, s = run3(h, tel_params("V3", v3_over(h)), kin, train_store(), Reanchor3("legacy"), return_stream=True)
    elif kind == "v3r":
        ts, out, s = run3(h, tel_params("V3", v3_over(h)), kin, train_store(), rp_of("v3", variant, h), return_stream=True)
    elif kind == "y1":
        ts, out, s = run3(h, None, kin, None, Reanchor3("legacy"), camera_only=True, return_stream=True)
    else:
        ts, out, s = run3(h, None, kin, None, rp_of("y1", variant, h), camera_only=True, return_stream=True)
    return ts, out, s


def forensic_rows(kind, h, ts, out, s, e):
    rows = []
    ev = getattr(s.filter, "events", {}) if kind in ("v3r", "y1r") else {}
    tk = {int(t): k for k, t in enumerate(ts)}
    for k, o in enumerate(out):
        t_s = (int(ts[k]) - LAPS[h][0]) * 1e-9
        if o.get("reanchor"):
            ek = ev.get(int(ts[k]), {})
            rows.append({"kind": kind, "lap": h, "tick": k, "t_s": round(t_s, 2), "event": "reanchor", "why": ek.get("why", "legacy"), "mean_nu_m": ek.get("mean_nu"),
                         "err_before_m": float(e[k - 1]) if k else None, "err_after_m": float(e[k])})
    for t, v in sorted(ev.items()):
        if v["kind"] == "held":
            k = tk[t]
            rows.append({"kind": kind, "lap": h, "tick": k, "t_s": round((t - LAPS[h][0]) * 1e-9, 2), "event": "held", "why": v["why"], "mean_nu_m": v["mean_nu"], "T_s": v["T"],
                         "err_before_m": float(e[k - 1]) if k else None, "err_after_m": float(e[k])})
    return rows


def nested():
    TAB.mkdir(parents=True, exist_ok=True)
    var = sel_variant()
    res, forensics = {}, []
    for kind in ("v3", "v3r", "y1", "y1r"):
        errs, per = {}, {}
        for h in TRAIN_LAPS:
            ts, out, s = lap_run(kind, h, var)
            e = errors(h, ts, out)
            errs[h] = e
            win = (e[(((ts - LAPS[h][0]) * 1e-9) >= EV_WIN[0]) & (((ts - LAPS[h][0]) * 1e-9) <= EV_WIN[1])])
            ok = ~np.isnan(e)
            per[h] = {"pos_rmse_m": float(np.sqrt(np.mean(e[ok] ** 2))), "n_gt3": int((e[ok] > 3).sum()), "max_m": float(np.nanmax(e)), "event_window_max_m": float(np.nanmax(win)),
                      "n_reanchor": int(out[-1]["n_reanchor"]), "n_rejected": int(sum(o["rejected"] for o in out))}
            forensics += forensic_rows(kind, h, ts, out, s, e)
        res[kind] = {"pooled": pooled_stats(errs), "per_lap": per}
    V3, V3R, Y1, Y1R = (res[k] for k in ("v3", "v3r", "y1", "y1r"))
    a = {"variant": var}
    a["V3R"] = {"pooled_rmse_ratio_to_V3": V3R["pooled"]["pos_rmse_m"] / V3["pooled"]["pos_rmse_m"],
                "rmse_not_worse_5pct": bool(V3R["pooled"]["pos_rmse_m"] <= 1.05 * V3["pooled"]["pos_rmse_m"]),
                "n_gt3": [V3["pooled"]["n_gt3"], V3R["pooled"]["n_gt3"]], "n_gt3_reduced_50pct": bool(V3R["pooled"]["n_gt3"] <= 0.5 * V3["pooled"]["n_gt3"]),
                "max_not_worse_each_lap": {str(h): bool(V3R["per_lap"][h]["max_m"] <= V3["per_lap"][h]["max_m"] + 1e-12) for h in TRAIN_LAPS},
                "lap1_event_window_max_m": V3R["per_lap"][1]["event_window_max_m"], "lap1_event_lt_3m": bool(V3R["per_lap"][1]["event_window_max_m"] < 3.0)}
    a["V3R"]["PASS_a"] = bool(a["V3R"]["rmse_not_worse_5pct"] and a["V3R"]["n_gt3_reduced_50pct"] and all(a["V3R"]["max_not_worse_each_lap"].values()) and a["V3R"]["lap1_event_lt_3m"])
    a["Y1R"] = {"pooled_rmse_ratio_to_Y1L0": Y1R["pooled"]["pos_rmse_m"] / Y1["pooled"]["pos_rmse_m"], "improved_10pct": bool(Y1R["pooled"]["pos_rmse_m"] <= 0.90 * Y1["pooled"]["pos_rmse_m"]),
                "n_gt3": [Y1["pooled"]["n_gt3"], Y1R["pooled"]["n_gt3"]], "n_gt3_reduced_50pct": bool(Y1R["pooled"]["n_gt3"] <= 0.5 * Y1["pooled"]["n_gt3"]),
                "max_not_worse_each_lap": {str(h): bool(Y1R["per_lap"][h]["max_m"] <= Y1["per_lap"][h]["max_m"] + 1e-12) for h in TRAIN_LAPS},
                "lap1_event_window_max_m": Y1R["per_lap"][1]["event_window_max_m"], "lap1_event_lt_3m": bool(Y1R["per_lap"][1]["event_window_max_m"] < 3.0)}
    a["Y1R"]["PASS_a"] = bool(a["Y1R"]["improved_10pct"] and a["Y1R"]["n_gt3_reduced_50pct"] and all(a["Y1R"]["max_not_worse_each_lap"].values()) and a["Y1R"]["lap1_event_lt_3m"])
    out = {"results": {k: {"pooled": v["pooled"], "per_lap": {str(h): x for h, x in v["per_lap"].items()}} for k, v in res.items()}, "rule_a": a}
    (TAB / "nested.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    (TAB / "reanchor_forensics_train.json").write_text(json.dumps(forensics, indent=1, default=float) + "\n")
    print("| row | pooled RMSE | n>3 m | max per lap (1/3/5) | event window max (lap 1) | reanchors (1/3/5) |\n|---|---|---|---|---|---|")
    for k, nm in (("v3", "V3 nested (legacy)"), ("v3r", f"V3-R nested ({var})"), ("y1", "Y1-L0 nested"), ("y1r", f"Y1-R nested ({var})")):
        r = res[k]
        print(f"| {nm} | {r['pooled']['pos_rmse_m']:.3f} | {r['pooled']['n_gt3']} | " + " / ".join(f"{r['per_lap'][h]['max_m']:.2f}" for h in TRAIN_LAPS)
              + f" | {r['per_lap'][1]['event_window_max_m']:.2f} | " + " / ".join(str(r['per_lap'][h]['n_reanchor']) for h in TRAIN_LAPS) + " |")
    print(json.dumps(a))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "nested"])
    c = ap.parse_args().cmd
    {"select": select, "nested": nested}[c]()
