"""E04Y: E04 (map PnP + labeled retrieval fallback) -> frozen Y1 kinematic filter, as ONE live causal protocol model (S07f, release option A).

Per tick t:  E04PnpModel.step(bundle)  ->  row (src, x, y, yaw, n_inl)  ->  Kin2 filter (Y1, latency scale 0)  ->  emitted pose.
The row stamped t is used for the output at t (no earlier, no later); no future row exists. The filter never feeds back into E04 (E04's own prior
is its own last accepted fix), so the E04 outputs equal those of the stand-alone model and the filter equals the stream-based S07e runner
(it IS the S07e code path: `Kin2Stream` with latency_scale 0, the row appended just before its tick).

row src: `pnp` when E04 flagged `pnp_fix`; `retrieval` when `retrieval_fallback`; otherwise `none` (hold / no image / E04 failure): the filter then
coasts by its own frozen gap logic and never fabricates a fix.

Coast limit (S07g, D-032): the frozen gap logic coasts on its estimated velocity while no fix arrives. If the filter has been continuously `degraded`
for more than `params.coast_limit_s` seconds (default 12, about 2x gap_s; `null` or 0 disables), the model emits the LAST EMITTED pose unchanged with flag
`coast_limit_hold` and tracking_state `degraded`; the filter itself keeps running untouched and re-anchors normally when fixes return (the output is the
filter's again as soon as it is no longer degraded). `coast_limit_s` lives outside the pinned filter block and is not part of the filter hash.

config: class camloc.models.e04y:E04YModel
  assets: {map_dir: artifacts/maps/S04-full-v1, [ext_dir: ...]}          (passed to E04 unchanged)
  params: {e04: {E04 params}, filter: {Kin2Params fields}, latency_scale: 0, expected_filter_sha256: <hash>, coast_limit_s: 12.0}
The filter parameter block (Kin2Params with defaults filled + latency_scale) is pinned by `filter_params_sha256()`; a config whose block hashes differently,
or a code constant (FROZEN_FILTER_SHA256) that differs from the config pin, is refused. Frozen source: configs/models/e09k3-y1.yaml
(sha256 796a2390..., S07e FROZEN.json).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time

from camloc.contracts import Capabilities, ContractError, FirstPose, FrameBundle, StaticMetadata, StepResult
from camloc.filtering.coast_limit import DEFAULT_COAST_LIMIT_S, CoastLimiter, CoastLimitError
from camloc.filtering.kin2_filter import Kin2Params
from camloc.filtering.kin2_runtime import Kin2Stream

# sha256 of canonical JSON {"filter": asdict(Kin2Params(**frozen filter)), "latency_scale": 0}; computed from configs/models/e09k3-y1.yaml by scripts/s07f_hash.py
FROZEN_FILTER_SHA256 = "80ca16491564e94f074143b970e8800132bd81b9d0bb4a198d2ab8bb1d8e97e6"


def filter_params_sha256(filter_params: dict, latency_scale) -> str:
    p = Kin2Params(**(filter_params or {}))
    blob = json.dumps({"filter": dataclasses.asdict(p), "latency_scale": float(latency_scale)}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _flags_to_src(flags) -> str:
    fl = set(flags)
    if "pnp_fix" in fl:
        return "pnp"
    if "retrieval_fallback" in fl:
        return "retrieval"
    return "none"


class E04YModel:
    def __init__(self, params: dict | None = None, assets: dict | None = None):
        params = dict(params or {})
        self.kparams = Kin2Params(**params.get("filter", {}))
        self.latency_scale = float(params.get("latency_scale", 0.0))
        try:                                                             # outside the pinned filter block; None / 0 disables (camloc.filtering.coast_limit)
            self.limiter = CoastLimiter(params.get("coast_limit_s", DEFAULT_COAST_LIMIT_S), params.get("hold_policy"))
        except CoastLimitError as e:
            raise ContractError(str(e)) from None
        self.coast_limit_ns = self.limiter.limit_ns
        if self.latency_scale != 0.0:
            raise ContractError("E04Y runs the filter at latency scale 0 (frozen); a live model has no future rows to wait for")
        got = filter_params_sha256(params.get("filter", {}), self.latency_scale)
        pin = params.get("expected_filter_sha256")
        if not pin:
            raise ContractError("E04Y requires params.expected_filter_sha256 (the frozen Y1 parameter hash)")
        if got != pin:
            raise ContractError(f"filter parameters differ from the pinned hash: {got} != {pin}")
        if FROZEN_FILTER_SHA256 and got != FROZEN_FILTER_SHA256:
            raise ContractError(f"filter parameters differ from the frozen Y1 hash in code: {got} != {FROZEN_FILTER_SHA256}")
        self.e04_params = dict(params.get("e04", {}))
        self.e04 = self._make_e04(self.e04_params, assets)
        c = self.e04.capabilities
        self.capabilities = Capabilities(cameras=c.cameras, clock_camera=c.clock_camera, calibration=c.calibration, map_id=c.map_id, telemetry_fields=(),
                                         device=c.device, assets=c.assets, scale_status="metric", online_mapping="disabled")
        self._ks: Kin2Stream | None = None
        self._n_init = 0
        self._fp = None
        self._t0 = None
        self.n_e04_errors = 0

    @property
    def n_coast_hold(self) -> int:
        return self.limiter.n_hold

    def _make_e04(self, e04_params: dict, assets):
        from camloc.models.pnp import E04PnpModel
        return E04PnpModel(e04_params, assets)

    # ------------------------------------------------------------------ protocol
    def initialize(self, first_pose: FirstPose, static_metadata: StaticMetadata) -> None:
        self._n_init += 1
        if self._n_init > 1:
            raise ContractError("E04Y initialized twice")
        self.e04.initialize(first_pose, static_metadata)           # raises on a calibration mismatch
        self._fp = (first_pose.x, first_pose.y, first_pose.yaw)
        self._t0 = first_pose.timestamp_ns
        self._ks = Kin2Stream(self.kparams, self._fp, self._t0, [], 0.0)

    def step(self, bundle: FrameBundle) -> StepResult:
        if self._ks is None:
            raise ContractError("E04Y step before initialize")
        t = int(bundle.timestamp_ns)
        t0 = time.perf_counter()
        try:
            r = self.e04.step(bundle)
            flags, state, resid = tuple(r.source_flags), r.tracking_state, dict(r.residuals or {})
            ex, ey, eyaw = float(r.x), float(r.y), float(r.yaw)
            cmin, cmax = r.consumed_min_ns, r.consumed_max_ns
            r.validate(t)
            src = _flags_to_src(flags)
        except ContractError:
            raise
        except Exception as e:                                       # E04 failure: labeled, the filter coasts; never fabricates a fix
            self.n_e04_errors += 1
            flags, state, resid = ("e04_error",), "held", {"e04_error": f"{type(e).__name__}: {str(e)[:200]}"}
            ex = ey = eyaw = float("nan")
            cmin = cmax = None
            src = "none"
        t_e04 = time.perf_counter() - t0
        n_inl = resid.get("n_inl")
        row = {"stream": "e04_fix", "t_ns": t, "latency_ns": 0, "src": src, "x_m": ex, "y_m": ey, "yaw_rad": eyaw,
               "n_inl": None if n_inl is None else int(n_inl)}
        t0 = time.perf_counter()
        ks = self._ks
        ks.e04.append(row)                                           # visible at this tick only: avail_ns = t + 0
        info = ks.tick(t)
        del ks.e04[:ks._ptr]                                         # consumed rows are never needed again
        ks._ptr = 0
        t_kin = time.perf_counter() - t0
        cr = self.limiter.update(t, bool(info["degraded"]), (info["x"], info["y"], info["yaw"]))
        ox, oy, oyaw = cr.pose              # on a hold the filter keeps running; only the emitted pose is frozen
        hold, coast_s = cr.hold, cr.coast_s
        out = list(flags)
        out.append("e04y")
        if hold:
            out.append("coast_limit_hold")
        if info["accepted"]:
            out.append("fix_accepted")
        if info["rejected"]:
            out.append("fix_rejected")
        if info["reanchor"]:
            out.append("reanchor")
        if info["ret_used"]:
            out.append("retrieval_weak")
        if info["yaw_held"]:
            out.append("yaw_held_last_fix")
        if info["degraded"]:
            out.append("labeled_fallback_coast")
        resid["e04"] = {"src": src, "flags": list(flags), "state": state, "x": ex if ex == ex else None, "y": ey if ey == ey else None,
                        "yaw": eyaw if eyaw == eyaw else None}
        resid["kin"] = {k: info[k] for k in ("gap_s", "v", "w", "accepted", "rejected", "reanchor", "ret_used", "dropped_old", "replay", "d2", "n_reanchor")}
        resid["kin"]["coast_s"] = round(coast_s, 3)
        resid["t_e04_ms"] = round(t_e04 * 1e3, 3)
        resid["t_kin_ms"] = round(t_kin * 1e3, 3)
        lo, hi = info["consumed_min_ns"], info["consumed_max_ns"]
        cm = [v for v in (cmin, lo) if v is not None]
        cx = [v for v in (cmax, hi) if v is not None]
        return StepResult(ox, oy, oyaw, source_flags=tuple(out),
                          tracking_state="degraded" if info["degraded"] else "tracking", covariance=info["cov"],
                          consumed_min_ns=min(cm) if cm else None, consumed_max_ns=max(cx) if cx else None, residuals=resid)

    def close(self) -> None:
        self._ks = None
        self.e04.close()


class E04YFbModel(E04YModel):
    """Option A on the fallback-camera E04 (S08e): identical filter, coast limit and pins; the inner model is `E04FallbackModel` (fr answers only while fl is
    missing/stale, S04e). config: class camloc.models.e04y:E04YFbModel, assets {map_dir, map_dir_fr, [ext_dir]}, params.e04 may add top_k_fr / stale_ticks / switch_back_ticks."""

    def _make_e04(self, e04_params: dict, assets):
        from camloc.models.pnp_fb import E04FallbackModel
        return E04FallbackModel(e04_params, assets)
