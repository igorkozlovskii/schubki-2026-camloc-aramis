"""Shared offline helpers for the S07a scripts: stream loading, quick evaluation (numpy mirror of the organizer scorer) and parameter IO."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("CAMLOC_ASSET_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "src"))

from camloc.evaluation import scoring  # noqa: E402
from camloc.filtering import fusion_streams as fs  # noqa: E402
from camloc.filtering.fusion_filter import FusionParams  # noqa: E402
from camloc.filtering.fusion_runtime import StreamFusion  # noqa: E402

STREAMS = ROOT / "artifacts/streams/S07a-v1"
LAPS = {0: (1791918157114146050, 1791918443953921626), 1: (1791918443953921626, 1791918611586947696),
        2: (1791918611586947696, 1791918735298153147), 3: (1791918735298153147, 1791918836730963974),
        4: (1791918836730963974, 1791918988712601923), 5: (1791918988712601923, 1791919109515726276)}


def load_lap_streams(lap_id: str, root: Path = STREAMS):
    e05 = fs.read_stream(root / f"{lap_id}.e05.jsonl.gz")
    p = root / f"{lap_id}.e04.jsonl.gz"
    return e05, (fs.read_stream(p) if p.exists() else None)


def first_pose_for(lap: int):
    d = json.loads((ROOT / f"artifacts/runs/S05-first-poses/lap{lap}.json").read_text())
    return (d["x"], d["y"], d["yaw"]), d["timestamp_ns"]


def run_offline(params: FusionParams, e05, e04, first_pose, t0_ns, latency_scale=1.0, rot=0.0, use_e04=True):
    sf = StreamFusion(params, first_pose, t0_ns, e05, e04, latency_scale, rot, use_e04)
    ts = [r["t_ns"] for r in e05]
    out = [sf.tick(t) for t in ts]
    return np.array(ts, dtype=np.int64), out


_GT = {}


def gt_arrays():
    if not _GT:
        import pandas as pd
        d = pd.read_csv(ROOT / "derived/gt.csv")
        _GT.update({"t": d.timestamp_ns.values.astype(np.int64), "x": d.x.values, "y": d.y.values, "yaw": d.yaw.values})
    return _GT


def quick_metrics(lap_start_ns: int, lap_end_ns: int, ts: np.ndarray, X: np.ndarray) -> dict:
    """Mirror of the organizer metrics on the lap's GT slice (evaluation side; used for TRAIN-lap tuning only)."""
    g = gt_arrays()
    sel = (g["t"] >= lap_start_ns) & (g["t"] < lap_end_ns)
    ps = np.column_stack([ts, X[:, 0], X[:, 1], X[:, 2]])
    return scoring.mirror(g["x"][sel], g["y"][sel], g["yaw"][sel], g["t"][sel], ps, 50_000_000, 1_000_000_000)
