"""PRG-002 S2.10 (GPU part, vision env): the S3.2d six-camera border fit started from the FILTER'S OWN output, one fit every 5th tick (2 Hz), inside the closed loop.

    flock /tmp/camloc-gpu.lock  CONDA_OVERRIDE_CUDA=12.4 TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_v2s210 pixi run --manifest-path environments/vision/pixi.toml python scripts/v2_s210_run.py ...

`LiveBorder.measure(i, t, outs)` is the `measure` hook of `camloc.v2.multicam.closedloop.drive_closed_loop`: it sees the filter outputs of ticks <= i and the camera frames stamped <= t_i, nothing else.
Same pipeline as scripts/v2_s32d_causal.py / v2_s111_border.py (cv2 reduced JPEG decode, segmentation net, `FastMeasurer.measure_att` with the road-attitude map looked up at the start's own s and the
causal `AttTracker`), differences: the start pose is the filter output at the tick (moved back to each frame stamp with the filter's own motion), and the row is the measurement of (start + correction) at the tick.
NO GT is read here (tested: tests/v2/s210).  The row's availability is the tick plus the measured end-to-end service time, as in S3.2d.
"""
from __future__ import annotations

import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v2_s11_common as H                      # noqa: E402
import v2_s32c_run as R3                       # noqa: E402
import v2_s32d_lib as D                        # noqa: E402
from camloc.geometry.transforms import wrap_angle                 # noqa: E402
from camloc.v2.border_fast import FastMeasurer                    # noqa: E402
from camloc.v2.border_seg_att import AttTracker                   # noqa: E402
from camloc.v2.multicam.closedloop import back_propagate          # noqa: E402
from camloc.v2.road_attitude import RoadAttitudeMap               # noqa: E402
from camloc.v2.seg.infer import SegPredictor, decode_jpeg         # noqa: E402

ROOT = D.ROOT
CAMS = ["fl", "fr", "cl", "cr", "rl", "rr"]
MAX_AGE_NS = 150_000_000
FIT = {"temp": 20000, "sd_dpsi_deg": 1.5}      # the modal nested S3.2d selection (3 of 4 folds), as S1.11 / S1.15
MASKS = ROOT / "artifacts/v2/S1.1-masks"
ATT = ROOT / "artifacts/v2/S3.2c-attitude-map"
NET_E = ROOT / "artifacts/v2/S3.2e-train/runs"   # S3.2e variant B (daylight-augmented): B_lap<k>_seed0 (LOO) / B_lap-1_seed0 (all laps)


def net_for(tag: str) -> dict:
    """tag 'loo<k>' (held-out lap k) or 'full' -> checkpoint, attitude map, masks."""
    if tag == "full":
        return {"ckpt": NET_E / "B_lap-1_seed0/ckpt_last.pt", "map": ATT / "full.npz", "masks": MASKS / "full"}
    k = int(tag[3:])
    return {"ckpt": NET_E / f"B_lap{k}_seed0/ckpt_last.pt", "map": ATT / f"loo{k}.npz", "masks": MASKS / f"loo{k}"}


class LiveBorder:
    def __init__(self, store_dir: Path | None, tag: str, sigma: float, ticks, fit: dict = FIT, ckpt=None, device: str = "cuda"):
        self.cx = D.Ctx()
        if store_dir is not None:
            self.cx.h = H.Harness(store_dir=store_dir)      # same class; GT is loaded by the class but never called in this module
            self.cx._ctx = {}
        self.h = self.cx.h
        n = net_for(tag)
        self.cfg = R3.make_cfg(fit)
        self.fit = dict(fit)
        self.meas = FastMeasurer(self.h.context(None, n["masks"]), self.cx.polys, self.cfg, device=device, graph=True, prune=1e-6)
        self.pred = SegPredictor.from_checkpoint(str(ckpt or n["ckpt"]), device)
        self.ram = RoadAttitudeMap.load(str(n["map"]))
        self.tracker = AttTracker(self.cfg, True, sd0=self.cfg.sd0_att_map)
        self.pool = ThreadPoolExecutor(6)
        self.device = device
        self.ticks = np.asarray(ticks, np.int64)         # the tick grid (stamps only); used for the spacing of consecutive filter outputs
        self.sigma = float(sigma)
        self.s_hint = None
        self.rows = []          # every attempt, valid or not (diagnostics; no GT)
        self.ckpt, self.map, self.tag = str(ckpt or n["ckpt"]), str(n["map"]), tag

    def _s(self, x, y, hint):
        return self.h.cline.project(x, y, s_hint=hint)[0]

    def fit_from(self, t: int, cur: tuple, prev: tuple | None, dt_prev_s: float | None) -> dict | None:
        """cur / prev = (x, y, yaw) of the start pose at t and at the previous tick (a causal pose source); -> border dict (stamp t) or None. Appends a diagnostics row to self.rows."""
        h, cl = self.h, self.h.cline
        t0 = time.perf_counter()
        s_cur = self._s(cur[0], cur[1], self.s_hint)
        self.s_hint = s_cur
        cur4 = (cur[0], cur[1], cur[2], s_cur)
        prev4 = None
        if prev is not None:
            s_prev = self._s(prev[0], prev[1], s_cur)
            prev4 = (prev[0], prev[1], prev[2], s_prev)
        fr = {}
        for c in CAMS:
            idx, st = h.camera_frame(c, t, MAX_AGE_NS)
            if idx is not None:
                fr[c] = (idx, st)
        diag = {"t": int(t), "n_cams": len(fr), "valid": False, "reason": "no_frame"}
        out = None
        if fr:
            cams = list(fr)
            imgs = list(self.pool.map(lambda c: decode_jpeg(h.store.jpeg(c, fr[c][0])), cams))
            x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().to(self.device)
            with torch.no_grad(), torch.autocast("cuda", enabled=self.pred.amp):
                pr = torch.sigmoid(self.pred.model(x).float())[:, 0]
            items = {}
            for q_, c in enumerate(cams):
                back = (t - fr[c][1]) * 1e-9
                if prev4 is None:
                    px, py, pyaw, ps = cur4
                else:
                    ds = cl.ds(prev4[3], cur4[3])
                    # finite-difference velocity of the filter's own output; s difference taken on the loop
                    vx, vy = (cur4[0] - prev4[0]) / dt_prev_s, (cur4[1] - prev4[1]) / dt_prev_s
                    vyaw, vs = float(wrap_angle(cur4[2] - prev4[2])) / dt_prev_s, ds / dt_prev_s
                    px, py, pyaw, ps = cur4[0] - vx * back, cur4[1] - vy * back, cur4[2] - vyaw * back, cur4[3] - vs * back
                items[c] = {"prob": pr[q_], "pose0": (px, py, pyaw), "s_init": ps % cl.length, "attitude": None}
            ref = "fl" if "fl" in items else cams[0]
            ref_st = fr[ref][1]
            dz_, dpi_, dro_ = self.ram.lookup(items[ref]["s_init"])
            td = np.array([dro_, dpi_, dz_])
            mm = self.meas.measure_att(items, ref=ref, att_prior=self.tracker.prior(t, td), fit_att=True)
            diag.update(reason=mm.reason if not mm.valid else "", ref_stamp=int(ref_st))
            if mm.valid:
                self.tracker.update(t, [mm.q["droll"], mm.q["dpitch"], mm.q["dz"]], [mm.q["sd_roll"], mm.q["sd_pitch"], mm.q["sd_z"]], td)
                s0, n0, th0, _ = cl.project(cur[0], cur[1], s_hint=s_cur)
                nx, ny = -math.sin(th0), math.cos(th0)
                x2, y2, yaw2 = cur[0] + mm.q["dn_fit"] * nx, cur[1] + mm.q["dn_fit"] * ny, cur[2] + mm.q["dpsi_fit"]
                s2, n2, th2, _ = cl.project(x2, y2, s_hint=s0)
                service = time.perf_counter() - t0
                diag.update(valid=True, n=float(n2), s=float(s2), n_start=float(n0), dn_fit=float(mm.q["dn_fit"]), dpsi_fit=float(mm.q["dpsi_fit"]), service_ms=1e3 * service)
                out = {"stamp": int(t), "avail": int(t) + int(service * 1e9), "s": float(s2), "z": float(n2), "sig": self.sigma, "inc": False, "n_start": float(n0), "service_ms": 1e3 * service}
        diag["service_ms"] = diag.get("service_ms", 1e3 * (time.perf_counter() - t0))
        self.rows.append(diag)
        return out

    def measure(self, i: int, t: int, outs: list) -> dict | None:
        """drive_closed_loop hook: the start is the filter's own output at tick i (outs[i]) and the one before (outs[i-1])."""
        assert len(outs) == i + 1 and int(self.ticks[i]) == int(t)
        o = outs[i]
        cur = (o["x"], o["y"], o["yaw"])
        if i > 0:
            p = outs[i - 1]
            return self.fit_from(t, cur, (p["x"], p["y"], p["yaw"]), (int(self.ticks[i]) - int(self.ticks[i - 1])) * 1e-9)
        return self.fit_from(t, cur, None, None)
