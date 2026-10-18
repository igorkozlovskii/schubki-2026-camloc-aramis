"""PRG-002 S1.16 Amendment 4: stage 1 with frame-age-correct rows (extra-camera fixes keyed to their image time). Evaluator-side GT only in the calibration / diagnostic functions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v2_s116_lib as M  # noqa: E402
from v2_s13_lib import ROOT  # noqa: E402
from camloc.filtering.kin2_filter import Kin2Params  # noqa: E402
from camloc.filtering.tel4_runtime import Tel4Stream  # noqa: E402
from camloc.v2.multicam.filter import MultiCamTel4Filter  # noqa: E402
from camloc.v2.multicam.fusion import kappa_from_errors  # noqa: E402
from camloc.v2.multicam.fusion_age import build_rows_age, image_stamp  # noqa: E402
from s10g_lib import cfg_for  # noqa: E402

_GT = {}
_E = {}


def _gt():
    if "gt" not in _GT:
        from camloc.mapping.keyframes import load_gt_poses
        _GT["gt"] = load_gt_poses(ROOT)
    return _GT["gt"]


def img_errors(cam: str, lap: int, which: str = "image"):
    """(pos err m, yaw err deg, lon, lat) of the camera's valid eligible fixes vs GT at the IMAGE time (which='image') or at the tick (which='tick'). Evaluator side."""
    k = (cam, lap, which)
    if k not in _E:
        s = pd.read_parquet(M.S12 / f"lap{lap}_{cam}_landmark.parquet")
        e = pd.read_parquet(M.S12 / f"lap{lap}_{cam}_landmark_eval.parquet")
        m = (e.valid & e.eligible).to_numpy()
        s = s[m]
        age = np.nan_to_num(s.q_cam_age_ms.to_numpy(), nan=0.0)
        t = s.stamp_ns.to_numpy() - (age * 1e6).astype(np.int64) if which == "image" else s.stamp_ns.to_numpy()
        g = _gt().at(t)
        dx, dy = s.x.to_numpy() - g["x"], s.y.to_numpy() - g["y"]
        c, sn = np.cos(g["yaw"]), np.sin(g["yaw"])
        yaw = np.degrees(np.abs((s.yaw.to_numpy() - g["yaw"] + np.pi) % (2 * np.pi) - np.pi))
        _E[k] = (np.hypot(dx, dy), yaw, dx * c + dy * sn, -dx * sn + dy * c)
    return _E[k]


def calibrate_kappa_img(held: int, cams) -> dict:
    errs = {}
    for c in cams:
        parts = [img_errors(c, l) for l in M.CAL_LAPS if l != held]
        errs[c] = (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    return kappa_from_errors(errs)


def run_arm2(arm: str, lap: int, mode: str = "zero", fold: int | None = None, kappa: dict | None = None, data=None, streams=None):
    cams, extra, gate, dist = M.ARMS[arm]
    rows0, fp, t0, ticks, store = data if data is not None else M.setup(lap)
    st = streams if streams is not None else M.streams_for(lap, cams)
    if kappa is None:
        kappa = calibrate_kappa_img(lap if fold is None else fold, cams) if (extra and len(cams) > 1) else {c: {"kxy": 1.0, "kyaw": 1.0} for c in cams}
    rows, tags, info = build_rows_age(rows0, ticks, st, cams, kappa, dist if gate else None, mode, extra)
    tp, kin, rp = cfg_for("g1", fold)
    s = Tel4Stream(tp, Kin2Params(**kin), fp, t0, rows, store, 1.0, rp=rp)   # Amendment 6: latency_ns carries the causal availability in both modes
    s.filter = MultiCamTel4Filter(tp, fp, t0, rp, tags, require_confirm=gate, escape_s=M.ESCAPE_S)
    outs = [s.tick(int(t)) for t in ticks]
    X = np.array([[o["x"], o["y"], o["yaw"]] for o in outs])
    flags = np.array([[o.get("accepted", 0), o.get("rejected", 0), o.get("reanchor", 0), o["mode"] == "telemetry"] for o in outs], np.float32)
    return {"ts": ticks, "x": X[:, 0], "y": X[:, 1], "yaw": X[:, 2], "flags": flags, "events": {int(k): v for k, v in s.filter.events.items()}, "n_extra_rows": info["n_extra_rows"], "info": info, "kappa": kappa}
