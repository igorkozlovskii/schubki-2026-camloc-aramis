"""Shared harness code of the S1.1 scripts (GT side: evaluators and offline builders only)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__import__("os").environ.get("CAMLOC_ASSET_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "src"))

from camloc.contracts import CAMERAS  # noqa: E402
from camloc.geometry.borders import BorderMap  # noqa: E402
from camloc.geometry.camera import CameraModel  # noqa: E402
from camloc.geometry.centerline import Centerline  # noqa: E402
from camloc.io import CameraStore, load_static_metadata  # noqa: E402
from camloc.mapping.keyframes import load_gt_poses  # noqa: E402
from camloc.v2.border_probe import ProbeConfig, ProbeContext, correct_camera  # noqa: E402
from camloc.v2 import body_masks  # noqa: E402

TRAIN = (0, 1, 3, 5)
STORE = ROOT / "artifacts/inputs/cameras-v1"
FRAMES = ROOT / "artifacts/inputs/split-v1/frames.csv"
LAP0_SKIP_S = 60.0          # lap 0: pit exit and tunnel (GT sigma > 1 m 42.6-54.8 s) are excluded from the eligible ticks
MIN_SPEED = 1.0             # m/s, "moving"
MAX_SIGMA = 1.0             # m, GT sigma_xy
SAMPLE_EVERY = 5


class Harness:
    def __init__(self, store_dir=STORE, frames_csv=FRAMES):
        self.store = CameraStore(store_dir, verify=False)
        meta = load_static_metadata(self.store)
        self.bmap = BorderMap(meta.track_bounds)
        self.cline = Centerline(meta.track_bounds)
        self.cams = {c: CameraModel.from_static(c, meta.cameras[c]) for c in CAMERAS}
        self.gt = load_gt_poses(ROOT)
        fr = pd.read_csv(frames_csv)
        self.frames = fr
        self.lap_start = {}
        lapsy = yaml.safe_load((ROOT / "configs/splits/laps-v1.yaml").read_text())["laps"]
        for l in lapsy:
            self.lap_start[l["lap"]] = l["start_ns"]
        self.elig = {c: fr[(fr.camera == c) & fr.train_eligible & fr.lap.isin(TRAIN)].set_index("stamp_ns").lap.to_dict() for c in CAMERAS}

    def ticks(self, lap, every=SAMPLE_EVERY):
        fl = self.frames[(self.frames.camera == "fl") & self.frames.train_eligible & (self.frames.lap == lap)].stamp_ns.sort_values().to_numpy()
        return fl[::every]

    def camera_frame(self, cam, tick_ns, max_age_ns=150_000_000):
        """Latest frame of `cam` with stamp <= tick (causal). Returns (index, stamp) or (None, None)."""
        st = self.store.stamps[cam]
        i = int(np.searchsorted(st, tick_ns, side="right")) - 1
        if i < 0 or tick_ns - st[i] > max_age_ns:
            return None, None
        return i, int(st[i])

    def gt_at(self, t_ns):
        p = self.gt.at(np.array([t_ns]))
        o = {k: float(v[0]) for k, v in p.items()}
        t2 = np.array([t_ns - 250_000_000, t_ns + 250_000_000])
        q = self.gt.at(t2)
        o["speed"] = float(math.hypot(q["x"][1] - q["x"][0], q["y"][1] - q["y"][0]) / 0.5)
        s, n, th, _ = self.cline.project(o["x"], o["y"], None, z=o["z"])
        o.update(s=s, n=n, theta=th, dpsi=float((o["yaw"] - th + math.pi) % (2 * math.pi) - math.pi))
        return o

    def in_pit(self, g):
        rs = self.bmap.road_height("pit", (g["x"], g["y"]), None)
        c = self.bmap.corridor("pit", g["x"], g["y"], g["z"])
        return bool(rs.ok and c is not None and -0.05 <= c["t"] <= 1.05 and self.bmap.nearest_group_distance("pit", (g["x"], g["y"])) < 10.0)

    def eligibility(self, lap, cam, img_stamp, g):
        """(eligible, reason) of a (camera frame, GT) pair for the Gate 1 metrics."""
        if self.elig[cam].get(img_stamp) != lap:
            return False, "not_train_eligible"
        if lap == 0 and (img_stamp - self.lap_start[0]) / 1e9 < LAP0_SKIP_S:
            return False, "lap0_pit_exit_tunnel"
        if g["sigma_xy"] > MAX_SIGMA:
            return False, "gt_sigma"
        if g["speed"] < MIN_SPEED:
            return False, "not_moving"
        if self.in_pit(g):
            return False, "pit_lane"
        return True, ""

    def perturb(self, g, rng):
        """Initial pose = GT shifted by (ds along, dn across) the centreline direction and rotated by dpsi."""
        from camloc.v2.border_probe import PERTURBATION as P
        dn = rng.uniform(-P["dn_m"], P["dn_m"])
        ds = rng.uniform(-P["ds_m"], P["ds_m"])
        dp = math.radians(rng.uniform(-P["dpsi_deg"], P["dpsi_deg"]))
        th = g["theta"]
        x = g["x"] + ds * math.cos(th) - dn * math.sin(th)
        y = g["y"] + ds * math.sin(th) + dn * math.cos(th)
        return (x, y, g["yaw"] + dp), dn, ds, dp

    def corrected_cams(self, offsets_yaml):
        """camera -> CameraModel with the fitted correction (yaml path or None = nominal)."""
        if offsets_yaml is None:
            return dict(self.cams)
        d = yaml.safe_load(Path(offsets_yaml).read_text())["cameras"]
        return {c: correct_camera(self.cams[c], math.radians(d[c]["dyaw_deg"]), math.radians(d[c]["dpitch_deg"]),
                                  math.radians(d[c]["droll_deg"])) for c in CAMERAS}

    def context(self, offsets_yaml, mask_dir, cfg=None):
        masks = body_masks.load_masks(mask_dir)
        return ProbeContext(self.bmap, self.cline, self.corrected_cams(offsets_yaml), masks, cfg or ProbeConfig())
