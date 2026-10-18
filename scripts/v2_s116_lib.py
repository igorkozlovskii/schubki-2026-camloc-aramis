"""PRG-002 S1.16 stage 1 helpers (default env): G1 + multi-camera fixes on the S2.3 / S3.1d harness. GT is read only in the evaluator-side functions
`cam_errors` / `calibrate_kappa` (sigma calibration on OTHER laps) and in v2_s13_lib (scoring); src/camloc/v2/multicam never sees it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v2_s13_lib as L13  # noqa: E402
import v2_s31c_prep as P  # noqa: E402
from v2_s13_lib import ROOT  # noqa: E402
from s10g_lib import cfg_for  # noqa: E402
from camloc.filtering.kin2_filter import Kin2Params  # noqa: E402
from camloc.filtering.tel4_runtime import Tel4Stream  # noqa: E402
from camloc.filtering.tel_runtime import NEEDED, load_store  # noqa: E402
from camloc.v2.multicam.filter import MultiCamTel4Filter  # noqa: E402
from camloc.v2.multicam.fusion import build_rows, kappa_from_errors  # noqa: E402

S12 = ROOT / "artifacts/v2/S1.2-streams"
ALL7 = ("fl", "fr", "cl", "cr", "r", "rl", "rr")
FR = ("fl", "fr")
CONF_D = 2.0        # declared agreement distance (m), PROTOCOL.md
ESCAPE_S = 30.0     # declared: agreement gate lifted after 30 s without an accepted fix
HELD = (1, 3, 5)
CAL_LAPS = (0, 1, 3, 5)

# arm id -> (cameras seen, extra camera measurements, confirmation gate)
ARMS = {
    "G1": (("fl",), False, False, CONF_D),
    "A": (ALL7, False, True, CONF_D),           # agreement gate, fl measurements only
    "B": (ALL7, True, False, CONF_D),           # all-camera fixes with per-camera sigma
    "C": (ALL7, True, True, CONF_D),            # A + B
    "A_fr": (FR, False, True, CONF_D),
    "B_fr": (FR, True, False, CONF_D),
    "C_fr": (FR, True, True, CONF_D),
    # sensitivity to the declared agreement distance (reported, never used for selection)
    "A_D1": (ALL7, False, True, 1.0), "A_D5": (ALL7, False, True, 5.0),
    "C_D1": (ALL7, True, True, 1.0), "C_D5": (ALL7, True, True, 5.0),
}
MAIN_ARMS = ("G1", "A", "B", "C", "A_fr", "B_fr", "C_fr")


def setup(lap: int):
    rows0, fp, t0, ticks = P.e04_rows(lap)
    store = load_store(L13.TELDIR, columns=NEEDED, windows=P.lap_windows([lap]))
    return rows0, fp, int(t0), np.asarray(ticks, np.int64), store


def streams_for(lap: int, cams) -> dict:
    return {c: pd.read_parquet(S12 / f"lap{lap}_{c}_landmark.parquet") for c in cams if c != "fl"}


_ECACHE = {}


def cam_errors(cam: str, lap: int):
    """(pos error m, yaw error deg) of the camera's valid, eligible fixes on a training lap (evaluator-side parquet of S1.2; GT inside, used only to calibrate noise scales)."""
    k = (cam, lap)
    if k not in _ECACHE:
        e = pd.read_parquet(S12 / f"lap{lap}_{cam}_landmark_eval.parquet")
        e = e[e.valid & e.eligible]
        _ECACHE[k] = (e.err_pos.to_numpy(), e.err_yaw_deg.to_numpy())
    return _ECACHE[k]


def calibrate_kappa(held: int, cams) -> dict:
    """Nested LOO: noise scales for held-out lap `held` from the other training laps (0, 1, 3, 5 minus held)."""
    errs = {}
    for c in cams:
        parts = [cam_errors(c, l) for l in CAL_LAPS if l != held]
        errs[c] = (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    return kappa_from_errors(errs)


def run_arm(arm: str, lap: int, mode: str = "zero", fold: int | None = None, kappa: dict | None = None, data=None, streams=None, gset: str | None = None):
    """Run G1-with-multicam arm on a lap. fold = G1 constants fold (held-out training lap); dev laps pass gset='all' (frozen release constants)."""
    cams, extra, gate, dist = ARMS[arm]
    rows0, fp, t0, ticks, store = data if data is not None else setup(lap)
    st = streams if streams is not None else streams_for(lap, cams)
    if kappa is None:
        kappa = calibrate_kappa(lap if fold is None else fold, cams) if (extra and len(cams) > 1) else {c: {"kxy": 1.0, "kyaw": 1.0} for c in cams}
    rows, tags, n_extra = build_rows(rows0, st, cams, kappa, dist if gate else None, mode, extra)
    tp, kin, rp = cfg_for("g1", fold if gset is None else None)
    s = Tel4Stream(tp, Kin2Params(**kin), fp, t0, rows, store, 0.0 if mode == "zero" else 1.0, rp=rp)
    s.filter = MultiCamTel4Filter(tp, fp, t0, rp, tags, require_confirm=gate, escape_s=ESCAPE_S)
    outs = [s.tick(int(t)) for t in ticks]
    X = np.array([[o["x"], o["y"], o["yaw"]] for o in outs])
    flags = np.array([[o.get("accepted", 0), o.get("rejected", 0), o.get("reanchor", 0), o["mode"] == "telemetry"] for o in outs], np.float32)
    ev = {int(k): v for k, v in s.filter.events.items()}
    return {"ts": ticks, "x": X[:, 0], "y": X[:, 1], "yaw": X[:, 2], "flags": flags, "events": ev, "n_extra_rows": n_extra, "kappa": kappa}
