"""Wrapper around the UNCHANGED organizer scorer (tools/evaluate.py): one scorer run per evaluation interval.

EVALUATOR-SIDE: reads derived/gt.csv. Never imported by the inference path.

Views (configs/contracts/scoring.yaml interval_materialization):
  gt_view         original GT rows with timestamp in [start_ns, ts of the first GT row at/after end_ns] (text copied verbatim)
  prediction_view prediction rows with start_ns <= timestamp_ns < end_ns (text copied verbatim)
The organizer script is run as a subprocess on those two files; its raw stdout is kept and parsed. An independent numpy
mirror of the scorer formulas supplies discard reasons, sum of squared errors and a parity check (never the headline).
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from camloc.util import sha256_file

PRED_COLUMNS = ["timestamp_ns", "x", "y", "yaw"]
MIN_COVERAGE = 0.9


class ScoreRefused(RuntimeError):
    """The request itself is refused (audit role, scorer hash mismatch ...) - no row is produced."""


def load_contract(root: Path) -> dict:
    return yaml.safe_load((root / "configs/contracts/scoring.yaml").read_text())


def verified_scorer(root: Path, contract: dict) -> Path:
    p = root / contract["scorer"]["path"]
    got = sha256_file(p)
    if got != contract["scorer"]["sha256"]:
        raise ScoreRefused(f"organizer scorer hash {got} != scoring.yaml {contract['scorer']['sha256']}")
    return p


# ------------------------------------------------------------------ GT / prediction views
def read_gt_lines(path: Path) -> tuple[str, list[str], np.ndarray]:
    lines = Path(path).read_text().splitlines()
    header, body = lines[0], lines[1:]
    ts = np.array([int(l.split(",", 1)[0]) for l in body], dtype=np.int64)
    return header, body, ts


def gt_view(header: str, body: list[str], ts: np.ndarray, start_ns: int, end_ns: int) -> tuple[list[str], dict]:
    lo = int(np.searchsorted(ts, start_ns, side="left"))
    j = int(np.searchsorted(ts, end_ns, side="left"))  # first GT row at/after end_ns
    hi = min(j, len(ts) - 1)                           # last lap: end_ns is past the last row -> last row
    if lo >= hi:
        raise ScoreRefused("GT view is empty for the interval")
    sel = body[lo:hi + 1]
    return [header] + sel, {"gt_rows": len(sel), "gt_first_ns": int(ts[lo]), "gt_last_ns": int(ts[hi]),
                            "gt_first_row": lo, "gt_last_row": hi}


def validate_prediction_text(path: Path) -> tuple[list[str], Optional[list[str]], Optional[np.ndarray], list[str]]:
    """Returns (errors, body_lines, int64 stamps, header cols). Never raises on bad content - collects reasons."""
    errs: list[str] = []
    text = Path(path).read_text()
    lines = [l for l in text.splitlines() if l.strip() != ""]
    if not lines:
        return ["empty file"], None, None, []
    cols = [c.strip() for c in lines[0].split(",")]
    if cols != PRED_COLUMNS:
        return [f"columns must be exactly {PRED_COLUMNS}, got {cols}"], None, None, cols
    body = lines[1:]
    if not body:
        return ["no prediction rows"], None, None, cols
    ts = []
    vals = np.full((len(body), 3), np.nan)
    for i, l in enumerate(body):
        parts = l.split(",")
        if len(parts) != 4:
            errs.append(f"row {i}: expected 4 fields, got {len(parts)}")
            break
        try:
            ts.append(int(parts[0]))
        except ValueError:
            errs.append(f"row {i}: timestamp_ns is not an integer ({parts[0]!r})")
            break
        try:
            vals[i] = [float(p) for p in parts[1:]]
        except ValueError:
            errs.append(f"row {i}: non-numeric x/y/yaw")
            break
    if errs:
        return errs, None, None, cols
    ts = np.array(ts, dtype=np.int64)
    if not np.isfinite(vals).all():
        errs.append(f"non-finite x/y/yaw in {int((~np.isfinite(vals).all(axis=1)).sum())} rows (NaN/inf)")
    d = np.diff(ts)
    if (d == 0).any():
        errs.append(f"duplicate timestamps ({int((d == 0).sum())})")
    if (d < 0).any():
        errs.append(f"non-monotonic timestamps ({int((d < 0).sum())} decreases)")
    return errs, body, ts, cols


# ------------------------------------------------------------------ mirror of tools/evaluate.py (diagnostic)
def mirror(gt_x, gt_y, gt_yaw, tg: np.ndarray, ps: np.ndarray, max_gap: int, bin_ns: int) -> dict:
    ts, sx, sy, syaw = ps[:, 0].astype(np.int64), ps[:, 1], ps[:, 2], ps[:, 3]
    i = np.searchsorted(tg, ts)
    outside = ~((i > 0) & (i < len(tg)))
    i = np.clip(i, 1, len(tg) - 1)
    gap = ((tg[i] - tg[i - 1]) > max_gap) & ~outside
    ok = ~outside & ~gap
    n_sub = len(ts)
    ts, i, sx, sy, syaw = ts[ok], i[ok], sx[ok], sy[ok], syaw[ok]
    a = (ts - tg[i - 1]) / (tg[i] - tg[i - 1])
    yu = np.unwrap(gt_yaw)
    gx = gt_x[i - 1] + a * (gt_x[i] - gt_x[i - 1])
    gy = gt_y[i - 1] + a * (gt_y[i] - gt_y[i - 1])
    gyaw = yu[i - 1] + a * (yu[i] - yu[i - 1])
    dx, dy = sx - gx, sy - gy
    lon = np.cos(gyaw) * dx + np.sin(gyaw) * dy
    lat = -np.sin(gyaw) * dx + np.cos(gyaw) * dy
    pos = np.hypot(dx, dy)
    dyaw = np.degrees(np.angle(np.exp(1j * (syaw - gyaw))))
    bins = np.arange(tg[0], tg[-1], bin_ns)
    covered = np.unique(np.searchsorted(bins, ts, side="right") - 1)
    rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan")
    return {"n_submitted": n_sub, "n_evaluated": int(ok.sum()), "discard_outside_gt_bracket": int(outside.sum()),
            "discard_gt_gap_gt_max": int(gap.sum()), "n_bins": int(len(bins)), "n_covered_bins": int(len(covered)),
            "coverage": len(covered) / max(len(bins), 1),
            "pos_rmse_m": rms(pos), "pos_median_m": float(np.median(pos)) if len(pos) else float("nan"),
            "pos_p95_m": float(np.percentile(pos, 95)) if len(pos) else float("nan"),
            "lat_rmse_m": rms(lat), "lon_rmse_m": rms(lon), "yaw_rmse_deg": rms(dyaw),
            "within_0_25m": float(np.mean(pos < 0.25)) if len(pos) else float("nan"),
            "within_0_5m": float(np.mean(pos < 0.5)) if len(pos) else float("nan"),
            "within_1m": float(np.mean(pos < 1.0)) if len(pos) else float("nan"),
            "sum_sq_pos_err_m2": float(np.sum(pos ** 2))}


# ------------------------------------------------------------------ stdout parsing
_NUM = r"([-+]?(?:\d+\.?\d*|nan|inf))"
_PATTERNS = {
    "submitted_evaluated": re.compile(r"^poses submitted\s+(\d+)\s+evaluated\s+(\d+)", re.M),
    "coverage_pct": re.compile(r"^coverage \(1 s bins\)\s+" + _NUM + r" %", re.M),
    "pos_rmse_m": re.compile(r"^position RMSE\s+\[m\]\s+" + _NUM, re.M),
    "median_p95": re.compile(r"^position median/p95\s+" + _NUM + r" / " + _NUM, re.M),
    "lat_rmse_m": re.compile(r"^lateral RMSE\s+\[m\]\s+" + _NUM, re.M),
    "lon_rmse_m": re.compile(r"^longitudinal RMSE \[m\]\s+" + _NUM, re.M),
    "yaw_rmse_deg": re.compile(r"^yaw RMSE\s+\[deg\]\s+" + _NUM, re.M),
    "within_0_25m": re.compile(r"^within 0\.25 m\s+" + _NUM + r" %", re.M),
    "within_0_5m": re.compile(r"^within 0\.50 m\s+" + _NUM + r" %", re.M),
    "within_1m": re.compile(r"^within 1\.00 m\s+" + _NUM + r" %", re.M),
}


def parse_stdout(text: str) -> dict:
    out: dict = {}
    for k, rx in _PATTERNS.items():
        m = rx.search(text)
        if not m:
            raise ScoreRefused(f"cannot parse scorer stdout field {k!r}")
        out[k] = m.groups()
    r = {"n_submitted": int(out["submitted_evaluated"][0]), "n_evaluated": int(out["submitted_evaluated"][1]),
         "coverage_printed": float(out["coverage_pct"][0]) / 100.0,
         "pos_rmse_m": float(out["pos_rmse_m"][0]), "pos_median_m": float(out["median_p95"][0]),
         "pos_p95_m": float(out["median_p95"][1]), "lat_rmse_m": float(out["lat_rmse_m"][0]),
         "lon_rmse_m": float(out["lon_rmse_m"][0]), "yaw_rmse_deg": float(out["yaw_rmse_deg"][0]),
         "within_0_25m": float(out["within_0_25m"][0]) / 100.0, "within_0_5m": float(out["within_0_5m"][0]) / 100.0,
         "within_1m": float(out["within_1m"][0]) / 100.0, "not_ranked_line": "NOT RANKED" in text}
    return r


def call_scorer(scorer: Path, gt_csv: Path, pred_csv: Path, python: str = sys.executable) -> tuple[int, str, str]:
    p = subprocess.run([python, str(scorer), str(gt_csv), str(pred_csv)], capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


NULL_METRICS = ["pos_rmse_m", "pos_median_m", "pos_p95_m", "lat_rmse_m", "lon_rmse_m", "yaw_rmse_deg", "within_0_25m",
                "within_0_5m", "within_1m", "sum_sq_pos_err_m2"]


def score_interval(root: Path, pred_csv: Path, out_dir: Path, *, start_ns: int, end_ns: int, interval_id: str, role: str,
                   mode: str, experiment_id: str, split_id: str, split_hash: str, declared_span: Optional[tuple] = None,
                   condition: str = "all", seed: int = 0, gt_path: Optional[Path] = None) -> dict:
    """Score one interval. Returns {'row': schema-conformant metrics row, 'detail': extras}. Invalid inputs yield a
    row with status 'invalid', null metrics and a reason - never zeros, and the organizer scorer is not called."""
    contract = load_contract(root)
    scorer = verified_scorer(root, contract)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_gap = int(contract["scorer"]["max_gt_gap_ns"])
    bin_ns = int(contract["scorer"]["coverage_bin_ns"])
    min_cov = float(contract["scorer"]["min_coverage"])
    declared = declared_span or (start_ns, end_ns)

    row = {"experiment_id": experiment_id, "split_id": split_id, "split_hash": split_hash, "mode": mode,
           "interval_id": str(interval_id), "role": role, "interval_start_ns": int(start_ns), "interval_end_ns": int(end_ns),
           "condition": condition, "seed": int(seed), "status": "invalid", "n_submitted": 0, "n_evaluated": 0,
           "coverage": None, "ranked": None, "scorer_stdout_path": None, "null_reason": None}
    for k in NULL_METRICS:
        row[k] = None
    row["output_hz"] = None
    row["max_output_gap_s"] = None
    detail: dict = {"declared_span_ns": list(declared), "scorer": str(scorer), "scorer_sha256": contract["scorer"]["sha256"]}

    def invalid(reason: str) -> dict:
        row["null_reason"] = reason
        detail["rejected"] = reason
        (out_dir / "metrics.json").write_text(json.dumps({"row": row, "detail": detail}, indent=1) + "\n")
        return {"row": row, "detail": detail}

    errs, body, ts, cols = validate_prediction_text(Path(pred_csv))
    if errs:
        if body is not None and ts is not None:
            row["n_submitted"] = len(body)
        return invalid("; ".join(errs))
    row["n_submitted"] = int(len(ts))
    if not (declared[0] <= start_ns and end_ns <= declared[1]):
        return invalid(f"scored interval [{start_ns},{end_ns}) is not inside the run's declared span {list(declared)}")
    if int(ts[0]) < declared[0] or int(ts[-1]) >= declared[1]:
        return invalid(f"prediction span [{int(ts[0])},{int(ts[-1])}] is not inside the declared span [{declared[0]},{declared[1]})")
    sel = (ts >= start_ns) & (ts < end_ns)
    pred_view = [PRED_COLUMNS_HEADER] + [b for b, k in zip(body, sel) if k]
    if sel.sum() == 0:
        return invalid("no prediction rows inside the scored interval")
    ts_v = ts[sel]
    detail["pred_view_rows"] = int(sel.sum())
    detail["pred_rows_outside_interval_trimmed"] = int((~sel).sum())

    gpath = gt_path or (root / "derived/gt.csv")
    header, gbody, gts = read_gt_lines(gpath)
    try:
        gt_lines, gdet = gt_view(header, gbody, gts, start_ns, end_ns)
    except ScoreRefused as e:
        return invalid(str(e))
    detail.update(gdet)
    gt_csv, pred_csv_v = out_dir / "gt_view.csv", out_dir / "pred_view.csv"
    gt_csv.write_text("\n".join(gt_lines) + "\n")
    pred_csv_v.write_text("\n".join(pred_view) + "\n")
    detail["gt_view_sha256"], detail["pred_view_sha256"] = sha256_file(gt_csv), sha256_file(pred_csv_v)
    detail["effective_interval_ns"] = [gdet["gt_first_ns"], gdet["gt_last_ns"]]

    # mirror (diagnostic): discard reasons, bins, sum of squares
    gcols = np.array([[float(v) for v in l.split(",")[1:]] for l in gt_lines[1:]])  # x,y,z,yaw
    pv = np.array([[float(v) for v in b.split(",")] for b in pred_view[1:]])
    tg = np.array([int(l.split(",", 1)[0]) for l in gt_lines[1:]], dtype=np.int64)
    mir = mirror(gcols[:, 0], gcols[:, 1], gcols[:, 3], tg, pv, max_gap, bin_ns)
    detail["mirror"] = mir
    detail["bins"] = {"n_bins": mir["n_bins"], "n_covered": mir["n_covered_bins"]}
    detail["discard_reasons"] = {"outside_gt_bracket": mir["discard_outside_gt_bracket"], f"gt_gap_gt_{max_gap}ns": mir["discard_gt_gap_gt_max"]}
    if mir["n_evaluated"] == 0:
        return invalid("no prediction row has a GT bracket (all discarded by the scorer rules)")

    rc, so, se = call_scorer(scorer, gt_csv, pred_csv_v)
    (out_dir / "scorer_stdout.txt").write_text(so)
    (out_dir / "scorer_stderr.txt").write_text(se)
    row["scorer_stdout_path"] = str(out_dir / "scorer_stdout.txt")
    detail["scorer_returncode"] = rc
    if rc != 0:
        row["status"] = "failed"
        return invalid(f"organizer scorer exit {rc}: {se.strip().splitlines()[-1] if se.strip() else ''}")
    p = parse_stdout(so)
    # parity: printed numbers vs the mirror, to printed precision
    tol = {"pos_rmse_m": 6e-4, "pos_median_m": 6e-4, "pos_p95_m": 6e-4, "lat_rmse_m": 6e-4, "lon_rmse_m": 6e-4,
           "yaw_rmse_deg": 6e-4, "within_0_25m": 6e-4, "within_0_5m": 6e-4, "within_1m": 6e-4}
    diffs = {k: abs(p[k] - mir[k]) for k in tol}
    diffs["coverage"] = abs(p["coverage_printed"] - mir["coverage"])
    detail["parity_max_abs_diff_printed_vs_mirror"] = max(diffs.values())
    if p["n_submitted"] != mir["n_submitted"] or p["n_evaluated"] != mir["n_evaluated"] or \
            any(diffs[k] > tol[k] for k in tol) or diffs["coverage"] > 6e-4:
        raise ScoreRefused(f"parity failure between scorer stdout and mirror: {diffs}")
    row["n_submitted"], row["n_evaluated"] = p["n_submitted"], p["n_evaluated"]
    row["coverage"] = mir["coverage"]
    d = np.diff(ts_v)
    row["output_hz"] = float(1e9 / np.median(d)) if len(d) else None
    row["max_output_gap_s"] = float(d.max() / 1e9) if len(d) else None
    detail["not_ranked_printed"] = p["not_ranked_line"]
    if mir["coverage"] < min_cov:
        row["ranked"] = False
        return invalid(f"coverage {mir['coverage']:.4f} < {min_cov} (scorer: {'NOT RANKED' if p['not_ranked_line'] else 'ranked'}); "
                       f"metrics withheld, see scorer_stdout.txt")
    for k in ("pos_rmse_m", "pos_median_m", "pos_p95_m", "lat_rmse_m", "lon_rmse_m", "yaw_rmse_deg", "within_0_25m", "within_0_5m", "within_1m"):
        row[k] = p[k]
    row["sum_sq_pos_err_m2"] = mir["sum_sq_pos_err_m2"]
    row["status"], row["ranked"], row["null_reason"] = "valid", True, None
    (out_dir / "metrics.json").write_text(json.dumps({"row": row, "detail": detail}, indent=1) + "\n")
    return {"row": row, "detail": detail}


PRED_COLUMNS_HEADER = ",".join(PRED_COLUMNS)
