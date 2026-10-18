"""StressPlan: applies one manifest condition inside the S02 runner without changing its protocol.

The runner calls (only when a plan is passed; default off, behavior identical when absent):
  plan.bind(store, start_ns, end_ns, split, cameras)   once, refuses non-dev ranges
  plan.apply_static(static)                            calibration conditions perturb what the MODEL is told
  plan.make_bundle(store, pos, n, clock, stale_ns, cameras)   replaces io.bundles.make_bundle
  plan.meta()                                          provenance for run_meta.json

Causality is preserved by construction: every image is found by `searchsorted(stamps, look, 'right') - 1` with
look <= t, and reported stamps are clamped to <= t. Pixels are corrupted lazily after decode; clean originals are never
modified (the store is read-only).
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from camloc.contracts import CAMERAS, CameraFrame, ContractError, FrameBundle, StaticMetadata
from camloc.geometry import Transform
from camloc.util import canonical_hash

from .manifest import EVIDENCE_LEVEL, condition as get_condition, manifest_hash
from .ops import corrupt_jpeg, derive_seed


def _rot(axis: str, deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def perturb_extrinsics(T: Transform, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, dx=0.0, dy=0.0, dz=0.0) -> Transform:
    """T_base_cam' : the camera is rotated/translated in the BASE frame by the given amounts (the model is told the
    perturbed value while the images stay clean, i.e. the stored calibration is wrong)."""
    R = _rot("z", yaw_deg) @ _rot("y", pitch_deg) @ _rot("x", roll_deg)
    m = np.array(T.matrix, dtype=np.float64)
    m[:3, :3] = R @ m[:3, :3]
    m[:3, 3] = m[:3, 3] + np.array([dx, dy, dz], dtype=np.float64)
    return Transform(m, T.target, T.source)


def scale_focal(K: tuple, P: tuple, frac: float) -> tuple[tuple, tuple]:
    K, P = list(K), list(P)
    for i in (0, 4):
        K[i] *= 1.0 + frac
    for i in (0, 5):
        P[i] *= 1.0 + frac
    return tuple(K), tuple(P)


class StressPlan:
    def __init__(self, manifest: dict, cond_id: str):
        self.manifest = manifest
        self.cond = get_condition(manifest, cond_id)
        if self.cond["kind"] in ("eval_stratum", "startup"):
            raise ContractError(f"{cond_id}: kind {self.cond['kind']} is not applied inside the runner")
        self.id = cond_id
        self.kind = self.cond["kind"]
        self.spec = self.cond.get("spec", {})
        self.seed = int(manifest["master_seed"])
        self.quality = int(manifest.get("reencode_quality", 95))
        self.cam_filter = None
        self.bursts: dict[str, list[tuple[int, int]]] = {}
        self._bound = False

    # ------------------------------------------------------------------ binding
    def bind(self, store, start_ns: int, end_ns: int, split: dict, cameras=None):
        for L in split["laps"]:
            if start_ns < int(L["end_ns"]) and end_ns > int(L["start_ns"]) and L["role"] != "dev":
                raise ContractError(f"REFUSED: stress is applied to dev laps only; range reaches {L['role']} lap {L['lap']}")
        self.start_ns, self.end_ns = int(start_ns), int(end_ns)
        if self.kind == "dropout":
            d = self.spec
            self.drop_cams = list(d["cameras"])
            if d["mode"] == "burst":
                n, dur = int(d["n_bursts"]), int(round(float(d["burst_s"]) * 1e9))
                rng = np.random.default_rng(derive_seed(self.seed, self.id, "burst-placement"))
                span = self.end_ns - self.start_ns
                wins = []
                for k in range(n):  # one burst per equal segment, fractional position drawn from the seed
                    u = (k + rng.uniform(0.2, 0.7)) / n
                    a = self.start_ns + int(u * span)
                    wins.append((a, a + dur))
                self.bursts = {c: wins for c in self.drop_cams}
        self._bound = True

    # ------------------------------------------------------------------ static metadata
    def apply_static(self, static: StaticMetadata) -> StaticMetadata:
        if self.kind != "calib":
            return static
        s = self.spec
        cams = dict(static.cameras)
        for c in s["cameras"]:
            cs = cams[c]
            T = perturb_extrinsics(cs.T_base_cam, s.get("yaw_deg", 0.0), s.get("pitch_deg", 0.0), s.get("roll_deg", 0.0),
                                   s.get("dx_m", 0.0), s.get("dy_m", 0.0), s.get("dz_m", 0.0))
            K, P = (cs.K, cs.P)
            if s.get("focal_frac"):
                K, P = scale_focal(cs.K, cs.P, float(s["focal_frac"]))
            cams[c] = dataclasses.replace(cs, T_base_cam=T, K=K, P=P)
        return dataclasses.replace(static, cameras=cams)

    # ------------------------------------------------------------------ bundles
    def _in_burst(self, cam: str, t: int) -> Optional[int]:
        for a, b in self.bursts.get(cam, ()):
            if a <= t < b:
                return a
        return None

    def make_bundle(self, store, tick_pos: int, tick_no: int, clock: str, stale_ns: int, cameras=None) -> FrameBundle:
        if not self._bound:
            raise ContractError("StressPlan used before bind()")
        t = int(store.stamps[clock][tick_pos])
        stride = int(self.spec["stride"]) if self.kind == "stride" else 1
        t_look = int(store.stamps[clock][tick_pos - (tick_no % stride)]) if stride > 1 else t
        skew_ns = int(round(float(self.spec["skew_ms"]) * 1e6)) if self.kind == "skew" else 0
        frames: dict[str, CameraFrame] = {}
        for cam in (cameras or CAMERAS):
            s = store.stamps.get(cam)
            if s is None or len(s) == 0:
                frames[cam] = CameraFrame(cam, "missing")
                continue
            look = t_look
            if self.kind == "dropout":
                if self.spec["mode"] == "whole" and cam in self.drop_cams:
                    frames[cam] = CameraFrame(cam, "missing")
                    continue
                a = self._in_burst(cam, t)
                if a is not None:   # the stream stops: the newest image is the last one before the burst, so it ages
                    look = min(look, a)
            j = int(np.searchsorted(s, look, side="right")) - 1
            if j < 0:
                frames[cam] = CameraFrame(cam, "missing")
                continue
            true_stamp = int(s[j])
            stamp = min(true_stamp + skew_ns, t) if skew_ns else true_stamp   # reported stamp: never later than t
            age = t - stamp
            if age > stale_ns or (stride > 1 and tick_no % stride != 0):
                # stride replay: a skipped tick delivers no new image (metadata only, pixels refused), like a slower camera
                frames[cam] = CameraFrame(cam, "stale", stamp, age)
            else:
                frames[cam] = CameraFrame(cam, "ok", stamp, age, _loader=self._loader(store, cam, j, true_stamp))
        return FrameBundle(timestamp_ns=t, tick_index=tick_no, frames=frames, stale_cutoff_ns=stale_ns)

    def _loader(self, store, cam, j, true_stamp):
        if self.kind not in ("pixel", "control"):
            return lambda c=cam, k=j: store.jpeg(c, k)
        ops = self.spec.get("ops", [{"op": "identity"}])
        only = self.spec.get("cameras")
        if only is not None and cam not in only:
            return lambda c=cam, k=j: store.jpeg(c, k)
        return lambda c=cam, k=j: corrupt_jpeg(store.jpeg(c, k), ops, self.seed, self.id, c, true_stamp, self.quality)

    # ------------------------------------------------------------------ provenance
    def meta(self) -> dict:
        return {"manifest_id": self.manifest["manifest_id"], "manifest_hash": manifest_hash(self.manifest), "condition": self.id,
                "family": self.cond.get("family"), "kind": self.kind, "tier": self.cond.get("tier"),
                "spec_hash": canonical_hash(self.spec), "master_seed": self.seed, "evidence_level": EVIDENCE_LEVEL,
                "artificial": bool(self.cond.get("artificial", True)),
                "burst_windows_ns": {c: [list(w) for w in ws] for c, ws in self.bursts.items()},
                "range_ns": [getattr(self, "start_ns", None), getattr(self, "end_ns", None)]}
