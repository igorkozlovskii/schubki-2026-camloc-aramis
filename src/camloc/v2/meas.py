"""`meas-v1` measurement stream: loader, validator and converter from `fusion-stream-v1` `e04_fix` rows (PRG-002 PLAN.md section 5).

One row per (tick, camera, kind); the track filter consumes nothing else from images. Parquet file, columns as in the PLAN (plus producer `q_*`).
Rows hold model OUTPUTS only (landmark / retrieval poses); ground-truth error tables are separate `*_eval.parquet` files and never joined here.

Converter conventions (E04 -> meas-v1):
  src pnp        -> kind `landmark`, valid, camera `fl` (the E04 map is built on the front-left camera)
  src retrieval  -> kind `coarse`   (E03-style place-recognition pose; the stream labels it `retrieval_fallback`)
  src none       -> kind `landmark`, valid False, q_reason = `no_source`
  avail_ns = t_ns + round(latency_ns * latency_scale) (scale 0 = the D-018 baseline treatment: the row stamped t is usable at tick t)
  s, n, dpsi = unhinted nearest-vertex projection of (x, y, yaw) onto the track-v1 frame; `q_proj_amb` flags rows whose projection differs by > 20 m
  from the hinted projection (circuit layers cross); the filter always re-projects (x, y, yaw) with its own s hint.
  sig_*: NaN (the producer declares none); the filter uses its configured defaults per kind.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "meas_v1_row.schema.json"
CAMERAS = ("fl", "fr", "cl", "cr", "r", "rl", "rr", "multi")
KINDS = ("border", "landmark", "coarse", "vo")
INITS = ("gt_perturbed", "pred_stream", "none")
BASE_COLUMNS = ("stamp_ns", "avail_ns", "lap", "camera", "kind", "valid", "s", "n", "dpsi", "x", "y", "yaw",
                "sig_s", "sig_n", "sig_dpsi", "sig_xy", "sig_yaw", "init")
FLOAT_COLUMNS = ("s", "n", "dpsi", "x", "y", "yaw", "sig_s", "sig_n", "sig_dpsi", "sig_xy", "sig_yaw")
INT_COLUMNS = ("stamp_ns", "avail_ns", "lap")


class MeasError(ValueError):
    pass


def _row_for_schema(r: dict) -> dict:
    out = {}
    for k, v in r.items():
        if isinstance(v, (np.floating, float)):
            v = float(v) if math.isfinite(float(v)) else None
        elif isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        out[k] = v
    return out


def validate_meas(df) -> None:
    """Raise MeasError unless `df` satisfies the meas-v1 contract: columns, types, enums, avail >= stamp, valid rows carry a pose."""
    import jsonschema
    miss = [c for c in BASE_COLUMNS if c not in df.columns]
    if miss:
        raise MeasError(f"meas-v1: missing columns {miss}")
    extra = [c for c in df.columns if c not in BASE_COLUMNS and not c.startswith("q_")]
    if extra:
        raise MeasError(f"meas-v1: columns outside the contract (must start with q_): {extra}")
    schema = json.loads(SCHEMA_PATH.read_text())
    val = jsonschema.Draft202012Validator(schema)
    for i, r in enumerate(df.to_dict("records")):
        errs = list(val.iter_errors(_row_for_schema(r)))
        if errs:
            raise MeasError(f"meas-v1 row {i}: {errs[0].message}")
    if len(df) and (df["avail_ns"].to_numpy(np.int64) < df["stamp_ns"].to_numpy(np.int64)).any():
        raise MeasError("meas-v1: avail_ns < stamp_ns (a measurement cannot be available before its exposure)")
    v = df[df["valid"].astype(bool)]
    for kind, need in (("landmark", ("x", "y", "yaw")), ("coarse", ("x", "y", "yaw")), ("border", ("n",))):
        sub = v[v["kind"] == kind]
        for c in need:
            if len(sub) and not np.isfinite(sub[c].to_numpy(float)).all():
                raise MeasError(f"meas-v1: valid {kind} rows must carry finite {c}")


def load_meas(path, validate: bool = True):
    import pandas as pd
    df = pd.read_parquet(path)
    if validate:
        validate_meas(df)
    return df


def write_meas(df, path) -> None:
    validate_meas(df)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def from_fusion_e04(rows: list[dict], track, lap: int, latency_scale: float = 0.0, camera: str = "fl"):
    """fusion-stream-v1 `e04_fix` rows -> meas-v1 DataFrame (see module docstring). `track` is a `TrackFrame`."""
    import pandas as pd
    out = []
    s_prev = None
    for r in rows:
        if r.get("stream") != "e04_fix":
            raise MeasError(f"not an e04_fix row: {r.get('stream')}")
        src = r["src"]
        lat = int(r.get("latency_ns", 0))
        avail = int(r["t_ns"]) + max(0, int(round(lat * latency_scale)))
        d = {"stamp_ns": int(r["t_ns"]), "avail_ns": avail, "lap": int(lap), "camera": camera,
             "kind": "coarse" if src == "retrieval" else "landmark", "valid": src in ("pnp", "retrieval"),
             "s": math.nan, "n": math.nan, "dpsi": math.nan, "x": math.nan, "y": math.nan, "yaw": math.nan,
             "sig_s": math.nan, "sig_n": math.nan, "sig_dpsi": math.nan, "sig_xy": math.nan, "sig_yaw": math.nan,
             "init": "pred_stream", "q_src": src, "q_i": int(r["i"]), "q_latency_ns": lat,
             "q_n_inl": r.get("n_inl"), "q_inlier_ratio": r.get("inlier_ratio"), "q_rms_px": r.get("rms_px"),
             "q_d_kf_m": r.get("d_kf_m"), "q_sim_top": r.get("sim_top"), "q_reason": r.get("reason") if src != "none" else "no_source",
             "q_proj_amb": False}
        if d["valid"]:
            x, y, yaw = float(r["x_m"]), float(r["y_m"]), float(r["yaw_rad"])
            s, n, dp = track.from_xy(x, y, yaw)
            if s_prev is not None:
                sh, _, _ = track.from_xy(x, y, yaw, s_hint=s_prev, window_m=150.0)
                d["q_proj_amb"] = bool(abs(track.ds(s, sh)) > 20.0)
            d.update(x=x, y=y, yaw=yaw, s=s, n=n, dpsi=dp)
            s_prev = s
        out.append(d)
    df = pd.DataFrame(out)
    for c in ("q_n_inl", "q_inlier_ratio", "q_rms_px", "q_d_kf_m", "q_sim_top"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["q_reason"] = df["q_reason"].astype("object")
    for c in INT_COLUMNS:
        df[c] = df[c].astype("int64")
    return df


def to_records(df) -> list[dict]:
    """Rows sorted by (avail_ns, stamp_ns, camera, kind) as plain dicts (the filter's consumption order)."""
    d = df.sort_values(["avail_ns", "stamp_ns", "camera", "kind"], kind="mergesort")
    return d.to_dict("records")
