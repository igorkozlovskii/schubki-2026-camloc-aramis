"""E04 with a fallback camera (S04e): the accepted E04 PnP pipeline on `fl`; the front-right camera `fr` answers only while `fl` is missing/stale.

Inference path: imports no ground-truth module. Maps are training-only (S04-*, S04d-fr-*). Telemetry: none. `online_mapping: disabled`.

Rule (declared in reports/experiments/S04e-fbcam/DECISION_RULE.md before any run):
  * `fl` not ok (stale or missing) for N consecutive ticks -> fallback mode ON (default N = 2);
  * in fallback mode `fr` answers on every tick whose `fr` frame is ok (else the unchanged `fl` path);
  * `fl` ok for M consecutive ticks -> fallback mode OFF (default M = 3), the `fl` prior takes the `fr` prior;
  * until the first ON the `fr` map, networks and model are never built and no `fr` pixel is read: the output is the unchanged E04 output, bit for bit.
"""
from __future__ import annotations

import dataclasses
import json
import time

from camloc.contracts import Capabilities, ContractError, FirstPose, FrameBundle, StaticMetadata, StepResult
from camloc.models.pnp import DEFAULTS, E04PnpModel
from camloc.models.retrieval import ROOT, GlobalDescriptor, LocalFeatures

FB_DEFAULTS = {"top_k_fr": 3, "stale_ticks": 2, "switch_back_ticks": 3}


class E04FallbackModel:
    def __init__(self, config: dict | None = None, assets: dict | None = None):
        p = {**DEFAULTS, **FB_DEFAULTS, **dict(config or {})}
        a = dict(assets or {})
        if p["stale_ticks"] < 1 or p["switch_back_ticks"] < 1:
            raise ContractError("stale_ticks and switch_back_ticks must be >= 1")
        self.p = p
        self._a = a
        base = {k: v for k, v in p.items() if k in DEFAULTS}
        self._base = base
        self.fl = E04PnpModel(base, {"map_dir": a["map_dir"], "ext_dir": a.get("ext_dir")})
        man_fr = json.loads((ROOT / a["map_dir_fr"] / "map.json").read_text())   # manifest only: no landmarks, no networks
        if man_fr["calibration_sha256"] != self.fl.map.manifest["calibration_sha256"]:
            raise ContractError("fr and fl maps were built with different calibrations")
        self.fr_map_id = man_fr["map_id"]
        self.fr: E04PnpModel | None = None
        self.capabilities = Capabilities(cameras=("fl", "fr"), clock_camera="fl", calibration=True,
                                         map_id=f"{self.fl.map.map_id}+{self.fr_map_id}", telemetry_fields=(), device="cuda",
                                         assets=(str(a["map_dir"]), str(a["map_dir_fr"])), scale_status="metric", online_mapping="disabled")
        self.n_init = 0
        self.bad = 0            # consecutive ticks with fl not ok
        self.good = 0           # consecutive ticks with fl ok
        self.active = False
        self.switches = 0
        self.fr_load_s: float | None = None
        self.n_fr_steps = 0
        self._init_args = None

    # ------------------------------------------------------------------ lazy fr
    def _build_fr(self) -> E04PnpModel:
        t0 = time.perf_counter()
        a, p = self._a, self.p
        frp = dict(self._base)
        frp["top_k"] = int(p["top_k_fr"])
        fr = E04PnpModel(frp, {"map_dir": a["map_dir_fr"], "ext_dir": a.get("ext_dir")})
        crop = tuple(fr.map.manifest.get("image_crop") or (12, 0, 1494, 540))
        ext = a.get("ext_dir")
        fr.fe = LocalFeatures(ext, max_kp=p["max_kp"], det_thr=p["det_thr"], crop=crop)
        fr.gd = GlobalDescriptor(ext, pool=fr.map.manifest["global_descriptor"]["pool"], crop=crop)
        fr.mt = self.fl.mt
        self.fr_load_s = time.perf_counter() - t0
        return fr

    def _ensure_fr(self) -> None:
        if self.fr is None:
            self.fr = self._build_fr()
            fp, sm = self._init_args
            cams = dict(sm.cameras)
            if "fr" not in cams:
                raise ContractError("E04 fallback needs camera fr in the static metadata")
            self.fr.initialize(fp, dataclasses.replace(sm, cameras={"fl": cams["fr"]}))

    # ------------------------------------------------------------------ protocol
    def initialize(self, first_pose: FirstPose, static_metadata: StaticMetadata) -> None:
        self.n_init += 1
        if self.n_init > 1:
            raise ContractError("E04 fallback initialized twice")
        if "fr" not in static_metadata.cameras:
            raise ContractError("E04 fallback needs camera fr in the static metadata")
        self._init_args = (first_pose, static_metadata)
        self.fl.initialize(first_pose, static_metadata)

    def _fb(self, src: str) -> dict:
        return {"src": src, "mode": "fallback" if self.active else "nominal", "switches": self.switches, "bad": self.bad, "good": self.good,
                "fr_loaded": self.fr is not None, "fr_load_s": self.fr_load_s}

    def step(self, bundle: FrameBundle) -> StepResult:
        f = bundle.frames.get("fl")
        fl_ok = f is not None and f.available
        if fl_ok:
            self.bad, self.good = 0, self.good + 1
        else:
            self.bad, self.good = self.bad + 1, 0
        if not self.active and self.bad >= self.p["stale_ticks"]:
            self._ensure_fr()
            self.fr.prior = self.fl.prior
            self.active = True
            self.switches += 1
        elif self.active and fl_ok and self.good >= self.p["switch_back_ticks"]:
            self.fl.prior = self.fr.prior
            self.active = False
            self.switches += 1
        if not self.active:
            res = self.fl.step(bundle)      # output untouched; only the diagnostics sidecar field is added
            return dataclasses.replace(res, residuals={**res.residuals, "fb": self._fb("fl")})
        frame = bundle.frames.get("fr")
        if frame is None or not frame.available:       # fr cannot answer: the unchanged fl path (fix if fl is ok, else its labeled hold)
            self.fl.prior = self.fr.prior
            res = self.fl.step(bundle)
            self.fr.prior = self.fl.prior
            return dataclasses.replace(res, source_flags=res.source_flags + ("fr_unavailable",), residuals={**res.residuals, "fb": self._fb("fl")})
        self.n_fr_steps += 1
        res = self.fr.step(dataclasses.replace(bundle, frames={"fl": frame}))
        if "pnp_fix" in res.source_flags:
            self.fl.prior = self.fr.prior
        return dataclasses.replace(res, source_flags=res.source_flags + ("cam_fr_fallback",), residuals={**res.residuals, "fb": self._fb("fr_fallback")})

    def close(self) -> None:
        self.fl.close()
        if self.fr is not None:
            self.fr.close()
