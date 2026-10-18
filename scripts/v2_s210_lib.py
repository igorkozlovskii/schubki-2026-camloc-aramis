"""PRG-002 S2.10 (CPU part): G1 + seven-camera fixes (mc7 = S1.16 arm C, Amendment-6 keying) + borders + pole bearings in ONE filter.

Arms (PROTOCOL.md):  mc7 = S1.16 arm C unchanged | mc7B = + border n | mc7P = + pole bearings | mc7BP = both.
The border measurement is made in a closed loop: the fit starts from the filter's own output at the tick (`drive_closed_loop`), not from another model's poses.
Inputs: held-out laps h1 / h3 / h5 (LOO maps, nets, pole maps) and dev lap 2 in three lightings (night, old simulated daylight 'sunny' = S1.10, new simulated daylight 'v3' = S1.15).
No GT here (the evaluator side is v2_s210_run.py -> v2_s13_lib); the only GT-derived inputs are the per-camera noise scales kappa of the OTHER laps (S1.16 calibration, evaluator side, numbers).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v2_s116_lib as M                      # noqa: E402
import v2_s116_lib2 as M2                    # noqa: E402
import v2_s23_lib as S23                     # noqa: E402
import v2_s25_filter as F25                  # noqa: E402
import v2_s13_lib as L13                     # noqa: E402
from v2_s13_lib import ROOT                  # noqa: E402
from camloc.filtering.kin2_filter import Kin2Params          # noqa: E402
from camloc.filtering.tel4_runtime import Tel4Stream         # noqa: E402
from camloc.v2.multicam.closedloop import MultiCamBase, PushBorder, drive_closed_loop   # noqa: E402
from camloc.v2.multicam.filter import MultiCamTel4Filter     # noqa: E402
from camloc.v2.multicam.fusion_age import build_rows_age     # noqa: E402
from s10g_lib import cfg_for                                 # noqa: E402

ARMS = ("mc7", "mc7B", "mc7P", "mc7BP")
CONSTS = ROOT / "manifests/v2/S2.8-constants.json"
K_POLE = 32.0                 # declared in PROTOCOL.md before any S2.10 run (S2.5 k >= 16, S2.8 daylight needs k >= 32)
EVERY = 5                     # 2 Hz border stream
HELD = (1, 3, 5)
INPUTS = ("h1", "h3", "h5", "night", "sunny", "v3")
SUNNY_STREAMS = ROOT / "artifacts/v2/S2.10-streams-sunny"
V3_STREAMS = ROOT / "artifacts/v2/S1.16-daylight-v3/streams"
DEV_STREAMS = ROOT / "artifacts/v2/S1.16-streams"
ROWS = {"sunny": ROOT / "artifacts/runs/S1.11-Ap-sunny/e04_rows.jsonl", "v3": ROOT / "artifacts/v2/S1.15-runs/S1.11-Ap-v3/e04_rows.jsonl"}
DET = {"night": ROOT / "artifacts/v2/S2.8-det/night", "sunny": ROOT / "artifacts/v2/S2.8-det/sunny", "v3": ROOT / "artifacts/v2/S1.15-det/v3"}
STORE = {"night": ROOT / "artifacts/v2/S1.11-store-night", "sunny": ROOT / "artifacts/v2/S1.11-store-sunny", "v3": ROOT / "artifacts/v2/S1.15-store-v3"}
MAPS = ROOT / "artifacts/maps"
LATENCY_NS = F25.LATENCY_NS


def consts() -> dict:
    return json.loads(CONSTS.read_text())


def border_sigma() -> float:
    c = consts()
    return float(c["border_sig0"] / math.sqrt(1.0 - c["rho"] ** 2))


# ------------------------------------------------------------------------------------------------ inputs
def load_input(name: str) -> dict:
    """GT-free inputs of one run: E04 fl rows, first pose, ticks, telemetry store, per-camera fix streams, noise scales (numbers), pole stream and map."""
    assert name in INPUTS, name
    if name.startswith("h"):
        lap = int(name[1:])
        assert lap in HELD
        rows0, fp, t0, ticks, store = M.setup(lap)
        streams = M.streams_for(lap, M.ALL7)
        fold = lap
        pole_map = MAPS / f"V2-S2.5-poles-loo{lap}"
        poles = F25.pole_stream(lap)
    else:
        lap = 2
        from s07_fusion_lib import first_pose_for
        from camloc.filtering.tel_runtime import NEEDED, load_store
        import v2_s31c_prep as P
        if name == "night":
            rows0, fp, t0, ticks, store = M.setup(2)
            streams = {c: pd.read_parquet(DEV_STREAMS / f"dev_lap2_{c}_landmark.parquet") for c in M.ALL7 if c != "fl"}
        else:
            rows0 = [json.loads(x) for x in open(ROWS[name])]
            fp, t0 = first_pose_for(2)
            ticks = np.array([r["t_ns"] for r in rows0], np.int64)
            store = load_store(L13.TELDIR, columns=NEEDED, windows=P.lap_windows([2]))
            d = V3_STREAMS if name == "v3" else SUNNY_STREAMS
            streams = {c: pd.read_parquet(d / f"lap2_{c}_landmark.parquet") for c in M.ALL7 if c != "fl"}
            t0 = int(t0)
        fold = None
        pole_map = MAPS / "V2-S2.5-poles-full"
        poles = pole_stream_dir(DET[name], pole_map, ROOT / "artifacts/v2/S1.1-masks/full")
    assert all(set(d.stamp_ns) <= set(map(int, ticks)) for d in streams.values()), "extra-camera stamps must lie on the tick grid"
    kappa = M2.calibrate_kappa_img(lap if fold is None else fold, M.ALL7)
    return {"name": name, "lap": lap, "fold": fold, "rows0": rows0, "fp": fp, "t0": int(t0), "ticks": np.asarray(ticks, np.int64), "store": store, "streams": streams,
            "kappa": kappa, "poles": poles, "pole_map": pole_map}


def pole_stream_dir(det_dir: Path, pole_map: Path, masks_dir: Path, cams=F25.CAMS) -> list:
    """The GT-free pole detection stream of lap 2 (same construction as F25.pole_stream: masked detections + map camera offsets, availability image stamp + 50 ms)."""
    from camloc.v2 import body_masks
    from camloc.v2.poles.detect import mask_keep
    off = json.loads((pole_map / "map.json").read_text()).get("cam_az_offset_rad", {})
    masks = body_masks.load_masks(masks_dir)
    out = []
    for cam in cams:
        d = pd.read_parquet(det_dir / f"lap2_{cam}.parquet")
        d = d[mask_keep(d[["u1", "v1", "u2", "v2"]].to_numpy(), masks[cam])]
        for (tick, img), q in d.groupby(["tick_ns", "img_ns"]):
            out.append({"stamp": int(tick), "avail": int(img) + LATENCY_NS, "cam": cam, "az": q.az.to_numpy() + off.get(cam, 0.0),
                        "el": 0.5 * (q.el_top.to_numpy() + q.el_bot.to_numpy()), "pol": q.pol.to_numpy()})
    return sorted(out, key=lambda r: r["avail"])


_STATIC = {}


def static_cams():
    if "v" not in _STATIC:
        from camloc.geometry.borders import BorderMap
        from camloc.geometry.camera import CameraModel
        from camloc.io import CameraStore, load_static_metadata
        st = CameraStore(STORE["night"], verify=False)
        meta = load_static_metadata(st)
        _STATIC["v"] = ({k: CameraModel.from_static(k, meta.cameras[k]) for k in F25.CAMS}, BorderMap(meta.track_bounds))
    return _STATIC["v"]


# ------------------------------------------------------------------------------------------------ filters
class MCBorder(PushBorder, S23.BorderG1Filter, MultiCamBase):
    """G1 + 7-camera fixes (+ agreement gate via set_multicam) + S2.3 border update; border rows may be pushed while the stream runs."""


class MCPole(PushBorder, F25.PoleFilter, MultiCamBase):
    """The same + S2.5 pole bearings (prediction-gated association inside the filter)."""


def make_filter(arm: str, inp: dict, tp, fp, t0, rp, tags: dict, border: list | None = None):
    gate, escape = True, M.ESCAPE_S
    if arm == "mc7":
        f = MultiCamTel4Filter(tp, fp, t0, rp, tags, require_confirm=gate, escape_s=escape)
        return f
    if arm == "mc7B":
        f = MCBorder(tp, fp, t0, rp, list(border or []))
    elif arm in ("mc7P", "mc7BP"):
        lm = pd.read_parquet(inp["pole_map"] / "landmarks.parquet")
        if inp["lap"] in HELD and inp["name"].startswith("h"):
            import v2_s11_common as H
            h = H.Harness()
            cams, bmap = h.cams, h.bmap
        else:
            cams, bmap = static_cams()
        f = MCPole(tp, fp, t0, rp, list(border or []), inp["poles"], lm, cams, bmap, k_sig=float(inp.get("k_pole", K_POLE)))
    else:
        raise ValueError(arm)
    f.set_multicam(tags, gate, escape)
    return f


def build(arm: str, inp: dict, mode: str = "zero", border: list | None = None):
    """-> (stream, info). Stage-1 construction of arm C (build_rows_age / run_arm2) with the filter of the arm."""
    cams, extra, gate, dist = M.ARMS["C"]
    rows, tags, info = build_rows_age(inp["rows0"], inp["ticks"], inp["streams"], cams, inp["kappa"], dist if gate else None, mode, extra)
    tp, kin, rp = cfg_for("g1", inp["fold"])
    s = Tel4Stream(tp, Kin2Params(**kin), inp["fp"], inp["t0"], rows, inp["store"], 1.0, rp=rp)   # latency_ns carries the causal availability (Amendment 6)
    s.filter = make_filter(arm, inp, tp, inp["fp"], inp["t0"], rp, tags, border)
    return s, info


def run(arm: str, inp: dict, mode: str = "zero", measure=None, border: list | None = None) -> dict:
    """Run one arm. measure = closed-loop border measurement function (needs a border-capable arm), border = a stored border list to replay instead."""
    s, info = build(arm, inp, mode, border=border if measure is None else None)
    ticks = inp["ticks"]
    if measure is not None:
        assert arm in ("mc7B", "mc7BP")
        outs = drive_closed_loop(s, ticks, measure, EVERY)
    else:
        outs = [s.tick(int(t)) for t in ticks]
    X = np.array([[o["x"], o["y"], o["yaw"]] for o in outs])
    flags = np.array([[o.get("accepted", 0), o.get("rejected", 0), o.get("reanchor", 0), o["mode"] == "telemetry"] for o in outs], np.float32)
    f = s.filter
    last = getattr(f, "_last", {})
    stats = {"border_used": getattr(f, "nb", {}).get("used"), "border_gated": getattr(f, "nb", {}).get("gated"), "border_rows": len(getattr(f, "border", []))}
    if last:
        stats.update(pole_frames=len(last), pole_assoc=int(sum(v[0] for v in last.values())), pole_used=int(sum(v[1] for v in last.values())), pole_gated=int(sum(v[2] for v in last.values())))
    return {"ts": ticks, "x": X[:, 0], "y": X[:, 1], "yaw": X[:, 2], "flags": flags, "events": {int(k): v for k, v in f.events.items()}, "n_extra_rows": info["n_extra_rows"],
            "info": info, "kappa": inp["kappa"], "border": list(getattr(f, "border", [])), "stats": stats}


def save(path: Path, r: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ts=r["ts"], x=r["x"], y=r["y"], yaw=r["yaw"], flags=r["flags"], events=json.dumps({str(k): v for k, v in r["events"].items()}, default=float),
                        kappa=json.dumps(r["kappa"]), n_extra=r["n_extra_rows"], info=json.dumps(r["info"]), stats=json.dumps(r["stats"], default=float))


def save_border(path: Path, border: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(border, default=float))


def load_border(path: Path) -> list:
    return json.loads(Path(path).read_text())
