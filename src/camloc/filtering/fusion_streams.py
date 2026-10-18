"""Per-tick observation streams for the S07a fusion (E04 absolute fixes + E05 relative motion) and their causal availability rule.

A stream is derived from a finished run directory (predictions.csv + diagnostics.jsonl) of the source model, produced in that model's
own environment. Nothing here touches pixels, GT or telemetry. The fusion consumes a row only at the first tick whose stamp is >=
`avail_ns(row)`; E04 rows use their measured per-step latency (scaled by `latency_scale`), E05 rows are consumed in-line at their own tick
(their ~34 ms compute is part of the 10 Hz budget, not of the pose clock).
"""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

SCHEMA = "fusion-stream-v1"
E05_VALID_FLAGS = {"vo": "ok", "vo_scale_bridged": "bridged"}


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _read_pred(path: Path):
    ts, X = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            ts.append(int(row["timestamp_ns"]))
            X.append((float(row["x"]), float(row["y"]), float(row["yaw"])))
    return np.asarray(ts, dtype=np.int64), np.asarray(X, dtype=np.float64)


def _read_diag(path: Path) -> list[dict]:
    steps = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("event") == "step":
                steps.append(r)
    return steps


def build_e05_stream(run_dir: Path, run_id: str) -> list[dict]:
    """Body-frame increments between consecutive E05 output poses. valid only for flags vo / vo_scale_bridged (tracking)."""
    run_dir = Path(run_dir)
    ts, P = _read_pred(run_dir / "predictions.csv")
    steps = _read_diag(run_dir / "diagnostics.jsonl")
    if len(steps) != len(ts) or any(int(s["t"]) != int(t) for s, t in zip(steps, ts)):
        raise ValueError(f"{run_dir}: diagnostics and predictions disagree on ticks")
    rows = []
    for k in range(len(ts)):
        s = steps[k]
        flags = list(s.get("flags", []))
        q = next((E05_VALID_FLAGS[f] for f in flags if f in E05_VALID_FLAGS), None)
        if k == 0:
            dx = dy = dyaw = 0.0
            q, why = None, "first_tick"
        else:
            x0, y0, a0 = P[k - 1]
            ddx, ddy = P[k, 0] - x0, P[k, 1] - y0
            dx = math.cos(a0) * ddx + math.sin(a0) * ddy
            dy = -math.sin(a0) * ddx + math.cos(a0) * ddy
            dyaw = _wrap(P[k, 2] - a0)
            why = ",".join(flags) if q is None else ""
        if q is not None and s.get("tracking") != "tracking":
            q, why = None, f"tracking={s.get('tracking')}"
        row = {"schema": SCHEMA, "stream": "e05_motion", "run_id": run_id, "t_ns": int(ts[k]), "i": k,
               "dx_m": dx, "dy_m": dy, "dyaw_rad": dyaw, "valid": q is not None, "quality": q or "invalid",
               "latency_ns": int(s.get("latency_ns", 0))}
        if why:
            row["reason"] = why
        rows.append(row)
    return rows


def build_e04_stream(run_dir: Path, run_id: str) -> list[dict]:
    """Per tick: PnP fix (flag pnp_fix), else the labeled retrieval top-1 pose (retrieval_fallback), else none (hold_last of the source)."""
    run_dir = Path(run_dir)
    ts, P = _read_pred(run_dir / "predictions.csv")
    steps = _read_diag(run_dir / "diagnostics.jsonl")
    if len(steps) != len(ts) or any(int(s["t"]) != int(t) for s, t in zip(steps, ts)):
        raise ValueError(f"{run_dir}: diagnostics and predictions disagree on ticks")
    rows = []
    for k, s in enumerate(steps):
        flags = set(s.get("flags", []))
        res = s.get("residuals", {}) or {}
        src = "pnp" if "pnp_fix" in flags else ("retrieval" if "retrieval_fallback" in flags else "none")
        g = lambda key: (None if res.get(key) is None else res[key])  # noqa: E731
        rows.append({"schema": SCHEMA, "stream": "e04_fix", "run_id": run_id, "t_ns": int(ts[k]), "i": k, "src": src,
                     "x_m": float(P[k, 0]), "y_m": float(P[k, 1]), "yaw_rad": float(P[k, 2]),
                     "latency_ns": int(s.get("latency_ns", 0)),
                     "n_inl": None if g("n_inl") is None else int(res["n_inl"]),
                     "n_kf_inl": None if g("n_kf_inl") is None else int(res["n_kf_inl"]),
                     "inlier_ratio": g("inlier_ratio"), "rms_px": g("rms_px"), "d_kf_m": g("d_kf_m"),
                     "sim_top": g("sim_top"), "reason": res.get("reason") if "reason" in res else None})
    return rows


def write_stream(rows: Iterable[dict], path: Path) -> str:
    """JSONL (gzip when the name ends with .gz); returns the sha256 of the written bytes."""
    import hashlib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ("\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows) + "\n").encode()
    if path.suffix == ".gz":
        with open(path, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(data)
    else:
        path.write_bytes(data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_stream(path: Path) -> list[dict]:
    path = Path(path)
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def validate_rows(rows: list[dict], schema_path: Optional[Path] = None) -> None:
    """Validate against schemas/fusion_stream.schema.json and the table rules (strictly increasing t_ns, finite numbers)."""
    import jsonschema
    schema_path = schema_path or Path(__file__).resolve().parents[3] / "schemas" / "fusion_stream.schema.json"
    schema = json.loads(Path(schema_path).read_text())
    v = jsonschema.Draft202012Validator(schema)
    last = None
    for r in rows:
        errs = list(v.iter_errors(r))
        if errs:
            raise ValueError(f"row i={r.get('i')} violates the stream schema: {errs[0].message}")
        for k, val in r.items():
            if isinstance(val, float) and not math.isfinite(val):
                raise ValueError(f"row i={r['i']} field {k} is not finite")
        if last is not None and r["t_ns"] <= last:
            raise ValueError("t_ns must be strictly increasing")
        last = r["t_ns"]


def avail_ns(row: dict, latency_scale: float = 1.0) -> int:
    """First instant the row may be used. e05_motion: its own stamp. e04_fix: stamp + measured latency x scale."""
    if row["stream"] == "e05_motion":
        return int(row["t_ns"])
    return int(row["t_ns"] + max(0, int(round(row["latency_ns"] * latency_scale))))


def causal_rows(rows: list[dict], t_ns: int, latency_scale: float = 1.0) -> list[dict]:
    """Rows a consumer at tick `t_ns` is allowed to see (stamp + latency <= t_ns)."""
    return [r for r in rows if avail_ns(r, latency_scale) <= t_ns]
