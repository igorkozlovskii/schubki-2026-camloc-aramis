"""PRG-002 S1.3 evaluator-side helpers (GT is read here, never in src/camloc/v2). Train laps 1, 3, 5 only (the laps that have E04 streams); no dev or audit rows.

Reuses, read-only, the PRG-001 helpers for the cached E04 streams, first poses, the train-lap telemetry windows and the G1 / V3-R baseline runner (s10g_lib).
Metrics: `camloc.evaluation.scoring.mirror` (mirror of the organizer evaluate.py: same GT bracketing/interpolation, max GT gap 50 ms) plus tail statistics.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from s10g_lib import LAPS, ROOT, TRAIN_LAPS, data, run_one, train_store  # noqa: E402,F401
from s07_fusion_lib import gt_arrays  # noqa: E402
from camloc.evaluation import scoring  # noqa: E402
from camloc.v2.filter.params import FilterParams, TIER_FIELDS, load_config  # noqa: E402
from camloc.v2.filter.stream import TrackStream  # noqa: E402
from camloc.v2.filter.track import TrackFrame  # noqa: E402
from camloc.v2.meas import from_fusion_e04, to_records  # noqa: E402
from camloc.filtering.tel_runtime import load_store  # noqa: E402

CFG_DIR = ROOT / "configs/v2/filter"
TELDIR = ROOT / "artifacts/telemetry/S10a-laps0-5"
TIERS = ("T2", "T1", "T0")
_TRACK = {}
_MEAS = {}
_STORES = {}


def track() -> TrackFrame:
    if "t" not in _TRACK:
        _TRACK["t"] = TrackFrame(json.loads((ROOT / "derived/track_bounds.json").read_text()))
    return _TRACK["t"]


def meas_rows(lap: int, latency_scale: float = 0.0) -> list[dict]:
    """E04 fusion-stream rows of a train lap converted to meas-v1 records (consumption order)."""
    k = (lap, latency_scale)
    if k not in _MEAS:
        rows, _, _ = data(lap)
        _MEAS[k] = to_records(from_fusion_e04(rows, track(), lap, latency_scale))
    return _MEAS[k]


def tier_store(tier: str):
    """Train-lap telemetry store restricted to the tier's fields (default-deny column subset)."""
    if tier == "T0":
        return None
    if tier not in _STORES:
        from s10b_lib import lap_windows
        _STORES[tier] = load_store(TELDIR, columns=TIER_FIELDS[tier], windows=lap_windows(TRAIN_LAPS))
    return _STORES[tier]


def lap_ticks(lap: int) -> np.ndarray:
    rows, _, _ = data(lap)
    return np.array([r["t_ns"] for r in rows], dtype=np.int64)


def run_filter(params: FilterParams, lap: int, rows: list[dict] | None = None, store="auto", fp_shift: float = 0.0, timing: bool = False):
    """Run the track filter over a train lap's cached stream. Returns (ts, outs[, per-tick seconds])."""
    _, fp, t0 = data(lap)
    fp = (fp[0] + fp_shift, fp[1], fp[2])
    rows = meas_rows(lap) if rows is None else rows
    st = tier_store(params.tier) if (isinstance(store, str) and store == "auto") else store
    ts = lap_ticks(lap)
    s = TrackStream(params, track(), fp, t0, rows, st)
    outs, dts = [], []
    for t in ts:
        t1 = time.perf_counter()
        outs.append(s.tick(int(t)))
        dts.append(time.perf_counter() - t1)
    return (ts, outs, np.array(dts)) if timing else (ts, outs)


# ---------------------------------------------------------------------------------------------------- metrics
def aligned_errors(lap: int, ts, X):
    """Per-tick GT-aligned error vectors (nan where the organizer bracketing drops the tick). Evaluator side."""
    g = gt_arrays()
    a, b = LAPS[lap]
    sel = (g["t"] >= a) & (g["t"] < b)
    tg, gx0, gy0, gyaw0 = g["t"][sel], g["x"][sel], g["y"][sel], g["yaw"][sel]
    ts = np.asarray(ts, np.int64)
    i = np.searchsorted(tg, ts)
    ok = (i > 0) & (i < len(tg))
    i = np.clip(i, 1, len(tg) - 1)
    ok &= ~((tg[i] - tg[i - 1]) > 50_000_000)
    f = (ts - tg[i - 1]) / (tg[i] - tg[i - 1])
    yu = np.unwrap(gyaw0)
    gx = gx0[i - 1] + f * (gx0[i] - gx0[i - 1])
    gy = gy0[i - 1] + f * (gy0[i] - gy0[i - 1])
    gyaw = yu[i - 1] + f * (yu[i] - yu[i - 1])
    X = np.asarray(X)
    dx, dy = X[:, 0] - gx, X[:, 1] - gy
    e = {"dx": np.where(ok, dx, np.nan), "dy": np.where(ok, dy, np.nan), "pos": np.where(ok, np.hypot(dx, dy), np.nan),
         "lat": np.where(ok, -np.sin(gyaw) * dx + np.cos(gyaw) * dy, np.nan),
         "lon": np.where(ok, np.cos(gyaw) * dx + np.sin(gyaw) * dy, np.nan),
         "dyaw_deg": np.where(ok, np.degrees(np.angle(np.exp(1j * (X[:, 2] - gyaw)))), np.nan)}
    return e


def lap_metrics(lap: int, ts, X) -> dict:
    """Organizer-formula numbers (scoring.mirror) + tail statistics + jump count (tick-to-tick change of the error VECTOR > 1 m, which removes the true motion)."""
    g = gt_arrays()
    a, b = LAPS[lap]
    sel = (g["t"] >= a) & (g["t"] < b)
    ps = np.column_stack([np.asarray(ts, np.int64), np.asarray(X)[:, 0], np.asarray(X)[:, 1], np.asarray(X)[:, 2]])
    m = scoring.mirror(g["x"][sel], g["y"][sel], g["yaw"][sel], g["t"][sel], ps, 50_000_000, 1_000_000_000)
    e = aligned_errors(lap, ts, X)
    pos = e["pos"][~np.isnan(e["pos"])]
    dv = np.hypot(np.diff(e["dx"]), np.diff(e["dy"]))
    dv = dv[~np.isnan(dv)]
    return {"n": m["n_evaluated"], "pos_rmse_m": m["pos_rmse_m"], "lat_rmse_m": m["lat_rmse_m"], "lon_rmse_m": m["lon_rmse_m"], "yaw_rmse_deg": m["yaw_rmse_deg"],
            "pos_median_m": m["pos_median_m"], "pos_p95_m": m["pos_p95_m"], "within_1m": m["within_1m"], "pos_max_m": float(pos.max()),
            "n_gt3m": int((pos > 3.0).sum()), "n_jumps_gt1m": int((dv > 1.0).sum()), "coverage": m["coverage"]}


def calibration(lap_errs: list[tuple[np.ndarray, np.ndarray, np.ndarray]], edges=(0.0, 0.25, 0.5, 1.0, 3.0, 1e9)):
    """Binned predicted position error vs actual. Input: [(pred_pos_err, actual_pos_err, pred_head_err_deg, actual_abs_dyaw_deg)] is flattened by the caller;
    here: list of (pred, actual, lost) arrays. Returns rows per bin of predicted error."""
    pred = np.concatenate([x[0] for x in lap_errs])
    act = np.concatenate([x[1] for x in lap_errs])
    ok = ~np.isnan(act)
    pred, act = pred[ok], act[ok]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pred >= lo) & (pred < hi)
        if m.sum() == 0:
            rows.append({"bin": f"{lo:g}-{hi:g}", "n": 0})
            continue
        rows.append({"bin": f"{lo:g}-{'inf' if hi > 1e8 else format(hi, 'g')}", "n": int(m.sum()), "pred_mean_m": float(pred[m].mean()), "actual_rmse_m": float(np.sqrt(np.mean(act[m] ** 2))),
                     "actual_median_m": float(np.median(act[m])), "actual_p95_m": float(np.percentile(act[m], 95)), "share_actual_le_pred": float(np.mean(act[m] <= pred[m]))})
    return rows


def objective(met: dict) -> float:
    """Tuning objective on training laps: position RMSE + 3 x share beyond 1 m + 0.01 x frames beyond 3 m (tail-weighted)."""
    return float(met["pos_rmse_m"] + 3.0 * (1.0 - met["within_1m"]) + 0.01 * met["n_gt3m"])


def jl(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------------------------------- simulated faults (labelled `simulated`, D-047)
def meas_cluster(rows: list[dict], lap: int, t0s: float, dur: float, mag: float, ang: float = 60.0) -> list[dict]:
    """Shift the landmark fixes in [t0s, t0s + dur) (s from lap start) by `mag` m in map direction `ang` deg: a repeating single-source wrong cluster."""
    a = LAPS[lap][0] + int(t0s * 1e9)
    b = a + int(dur * 1e9)
    dx, dy = mag * math.cos(math.radians(ang)), mag * math.sin(math.radians(ang))
    out = []
    for r in rows:
        if a <= r["stamp_ns"] < b and r["kind"] == "landmark" and r["valid"]:
            r = {**r, "x": r["x"] + dx, "y": r["y"] + dy}
        out.append(r)
    return out


def meas_blackout(rows: list[dict], lap: int, t0s: float, dur: float) -> list[dict]:
    a = LAPS[lap][0] + int(t0s * 1e9)
    b = a + int(dur * 1e9)
    return [r for r in rows if not (a <= r["stamp_ns"] < b)]


def store_edit(store, lap: int, t0s: float, dur: float, fields, fn):
    from s10g_lib import edit_store
    return edit_store(store, lap, t0s, dur, [f for f in fields if f in store.fields], fn)


def store_dropout(store, lap: int, t0s: float, dur: float):
    from camloc.io.telemetry import TelemetryStore
    a = LAPS[lap][0] + int(t0s * 1e9)
    keep = ~((store._t > a) & (store._t <= a + int(dur * 1e9)))
    return TelemetryStore(store._t[keep], {k: v[keep] for k, v in store._c.items()})


def scenario(tier: str, lap: int, name: str):
    """-> (meas rows, store, fp_shift, (t_start, t_end)) for a named simulated fault on a train lap."""
    rows, st = meas_rows(lap), tier_store(tier)
    if name == "nominal":
        return rows, st, 0.0, (0.0, 0.0)
    if name == "cluster5":      # 5 s single-source wrong cluster, 20 m
        return meas_cluster(rows, lap, 45, 5, 20.0), st, 0.0, (45.0, 50.0)
    if name == "cluster12":     # 12 s single-source wrong cluster, 15 m
        return meas_cluster(rows, lap, 40, 12, 15.0), st, 0.0, (40.0, 52.0)
    if name == "wzbias":        # yaw-rate bias +0.07 rad/s for 8 s
        return rows, (st if st is None else store_edit(st, lap, 50, 8, ["wz_radps"], lambda x: x + 0.07)), 0.0, (50.0, 58.0)
    if name == "speed":         # speed x1.25 for 5 s
        return rows, (st if st is None else store_edit(st, lap, 55, 5, ["v_mps", "vx_mps", "vy_mps"], lambda x: x * 1.25)), 0.0, (55.0, 60.0)
    if name == "cam_loss":      # all camera measurements lost for 10 s
        return meas_blackout(rows, lap, 60, 10), st, 0.0, (60.0, 70.0)
    if name == "tel_dropout":   # telemetry rows missing for 5 s
        return rows, (st if st is None else store_dropout(st, lap, 55, 5)), 0.0, (55.0, 60.0)
    if name == "first_pose":    # first pose displaced 15 m along x
        return rows, st, 15.0, (0.0, 0.0)
    raise ValueError(name)


# ---------------------------------------------------------------------------------------------------- stress specs shared by the filter and the G1 / V3-R baselines
STRESS = [   # id, kind, params, (t_start, t_end) in s from the lap start; simulated (D-047)
    {"id": "F1-wz+0.10x10s@30", "kind": "wz", "p": {"t0": 30, "dur": 10, "dw": 0.10}, "win": (30.0, 40.0)},
    {"id": "F2-speed1.3x6s@35", "kind": "speed", "p": {"t0": 35, "dur": 6, "f": 1.3}, "win": (35.0, 41.0)},
    {"id": "F3-first-pose+15m", "kind": "fp", "p": {"d": 15.0}, "win": (0.0, 0.0)},
    {"id": "F4a-fixes+25m x6s@30", "kind": "cluster_fix", "p": {"t0": 30, "dur": 6, "mag": 25.0}, "win": (30.0, 36.0)},
    {"id": "F4b-fixes+15m x12s@70", "kind": "cluster_fix", "p": {"t0": 70, "dur": 12, "mag": 15.0}, "win": (70.0, 82.0)},
    {"id": "F4c-fixes+coarse+25m x6s@30", "kind": "cluster_all", "p": {"t0": 30, "dur": 6, "mag": 25.0}, "win": (30.0, 36.0)},
    {"id": "F5-camera-loss 10s@60", "kind": "blackout", "p": {"t0": 60, "dur": 10}, "win": (60.0, 70.0)},
    {"id": "F6-telemetry-dropout 5s@55", "kind": "tel_dropout", "p": {"t0": 55, "dur": 5}, "win": (55.0, 60.0)},
]


def _cluster_fusion_rows(rows, lap, t0s, dur, mag, ang=60.0, only_pnp=False):
    import copy
    r = copy.deepcopy(rows)
    a = LAPS[lap][0] + int(t0s * 1e9)
    b = a + int(dur * 1e9)
    dx, dy = mag * math.cos(math.radians(ang)), mag * math.sin(math.radians(ang))
    for x in r:
        if a <= x["t_ns"] < b and x.get("x_m") is not None and (not only_pnp or x["src"] == "pnp"):
            x["x_m"] += dx
            x["y_m"] += dy
    return r


def apply_stress(tier: str, lap: int, spec: dict):
    """-> (meas rows, tier store, fp_shift) for the filter."""
    rows, st = meas_rows(lap), tier_store(tier)
    k, p = spec["kind"], spec["p"]
    if k == "fp":
        return rows, st, float(p["d"])
    if k == "wz":
        return rows, (st if st is None else store_edit(st, lap, p["t0"], p["dur"], ["wz_radps"], lambda x: x + p["dw"])), 0.0
    if k == "speed":
        return rows, (st if st is None else store_edit(st, lap, p["t0"], p["dur"], ["v_mps", "vx_mps", "vy_mps"], lambda x: x * p["f"])), 0.0
    if k == "cluster_fix":
        return meas_cluster(rows, lap, p["t0"], p["dur"], p["mag"]), st, 0.0
    if k == "cluster_all":
        a = LAPS[lap][0] + int(p["t0"] * 1e9)
        b = a + int(p["dur"] * 1e9)
        dx, dy = p["mag"] * math.cos(math.radians(60.0)), p["mag"] * math.sin(math.radians(60.0))
        out = [({**r, "x": r["x"] + dx, "y": r["y"] + dy} if (a <= r["stamp_ns"] < b and r["valid"]) else r) for r in rows]
        return out, st, 0.0
    if k == "blackout":
        return meas_blackout(rows, lap, p["t0"], p["dur"]), st, 0.0
    if k == "tel_dropout":
        return rows, (st if st is None else store_dropout(st, lap, p["t0"], p["dur"])), 0.0
    raise ValueError(k)


def run_baseline_stress(kind: str, lap: int, fold, spec: dict):
    """G1 / V3-R under the same injection. Returns (ts, outs). Uses the PRG-001 builders (s10g_lib), E04 fusion rows."""
    from s10g_lib import blackout, edit_store
    rows, _, _ = data(lap)
    store = train_store()
    k, p = spec["kind"], spec["p"]
    sh = 0.0
    if k == "fp":
        sh = float(p["d"])
    elif k == "wz":
        store = edit_store(store, lap, p["t0"], p["dur"], ["wz_radps"], lambda x: x + p["dw"])
    elif k == "speed":
        store = edit_store(store, lap, p["t0"], p["dur"], ["v_mps", "vx_mps", "vy_mps"], lambda x: x * p["f"])
    elif k == "cluster_fix":
        rows = _cluster_fusion_rows(rows, lap, p["t0"], p["dur"], p["mag"], only_pnp=True)
    elif k == "cluster_all":
        rows = _cluster_fusion_rows(rows, lap, p["t0"], p["dur"], p["mag"])
    elif k == "blackout":
        rows, store = blackout(rows, store, lap, p["t0"], p["dur"])
    elif k == "tel_dropout":
        store = store_dropout(store, lap, p["t0"], p["dur"])
    ticks = data(lap)[0]
    ts, out, _ = run_one(kind, lap, fold, rows=rows, store=store, fp_shift=sh, ticks=ticks)
    return ts, out


def stress_metrics(lap: int, ts, X, win, after_s: float = 30.0) -> dict:
    e = aligned_errors(lap, ts, X)["pos"]
    t = (np.asarray(ts) - LAPS[lap][0]) * 1e-9
    dt = np.r_[np.diff(t), np.median(np.diff(t))]
    lo, hi = win
    w = (t >= lo) & (t <= min(hi + after_s, float(t[-1]))) & ~np.isnan(e)
    after = w & (t >= hi)
    ok = np.nonzero(after & (e < 3.0))[0]
    return {"n_gt3m": int(((e > 3.0) & w).sum()), "max_m": float(np.nanmax(np.where(w, e, np.nan))),
            "recovery_s": float(t[ok[0]] - hi) if len(ok) else after_s, "over10_s": float(dt[after & (e > 10.0)].sum()),
            "rmse_lap_m": float(np.sqrt(np.nanmean(e ** 2)))}
