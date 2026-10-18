"""E04T: E04 (map PnP + labeled retrieval fallback) -> telemetry-assisted filter (S10b V3 / E10), one live causal protocol model (S10c, release option A').

Per tick t:  E04PnpModel.step(bundle) -> row (src, x, y, yaw, n_inl) -> TelStream.tick(t) -> emitted pose.
TelStream consumes telemetry rows with stamp <= t only (TelemetryStore.rows_between / latest) and the E04 row of tick t at latency 0. The filter never feeds back into E04.
Telemetry absent (no `telemetry_dir` asset / no telemetry.csv), not yet started, older than `fallback_age_s` (1 s), or a store whose range does not cover the run:
mode `fallback` at that tick = EXACTLY the frozen Y1 filter (the S07f model E04Y), reported per tick as the source flag `mode_fallback` (else `mode_telemetry`).

Capability: `telemetry: true`, telemetry_fields = the telemetry-v1 subset actually read (v, vx, vy, wz). The runner refuses the model unless the run enables the capability
(RunSpec.allow_telemetry / --telemetry; default off). Only telemetry-v1 field names are ever read; a telemetry file whose header carries a forbidden/GT or unknown column is refused.

FILTER SELECTION IS CONFIG (S10c, orchestrator update): the stream-model class and its parameters are named in the model config, so switching V3 -> V3-R -> a tel4/e10g model
is a config change, no code edit.
config: class camloc.models.e04t:E04TModel
  assets: {map_dir: ..., [ext_dir: ...], [telemetry_dir: dir with telemetry.csv, injected by the runner]}
  params:
    e04: {...}
    filter:
      module: camloc.filtering.tel3_runtime        # importable module with the stream class
      class: Tel3Stream
      params_file: configs/models/e10f-v3r.yaml    # source of `params:` blocks (tel, kin, ...); relative to the repo root; or inline `params: {...}` instead
      extra: {rp: {module: camloc.filtering.tel3_filter, class: Reanchor3, from: reanchor3}}   # optional constructor kwargs built as Class(**params[from])
      expected_file_sha256: <sha256 of params_file>   # required with params_file
    latency_scale: 0, fallback_age_s: 1.0, expected_param_sha256: <hash>, telemetry_stamp_limit_ns: <int|null>
Stream-model interface assumed of the selected class (the S10b TelStream contract and nothing else): Cls(TelParams(**params.tel), Kin2Params(**params.kin), first_pose, t0_ns, e04_rows, store,
latency_scale, fallback_age_s, **extra).tick(t) -> dict with x, y, yaw, cov, mode, tel_age_ms, tel_coast, n_samples, consumed_min_ns/max_ns, accepted/rejected/reanchor/ret_used; it owns the
E04 row lists `.e04` and `.kin.e04` (appended per tick) and the pointers `._ptr`, `.kin._ptr`. A class that provides `Cls.from_params(params, first_pose, t0_ns, e04_rows, store,
latency_scale, fallback_age_s)` is built through it instead (use this for a filter whose constructor differs).
`expected_param_sha256` pins canonical JSON {filter: {module, class}, params blocks (tel, kin, extras: raw dicts), latency_scale, fallback_age_s}; the kin block must also hash to the
frozen Y1 block (camloc.models.e04y); a code constant FROZEN_PARAM_SHA256 pins it in code once a freeze exists (None until then: only the config pin applies).
Degraded / coast limit (D-032, S07g): in mode `fallback` the model behaves exactly as E04Y including its coast limit: the Y1 filter (running in parallel on the same rows) continuously
`degraded` (no accepted fix for > stale_fix_s) for more than `params.coast_limit_s` (default = E04Y's DEFAULT_COAST_LIMIT_S, null/0 disables) -> the LAST EMITTED pose is repeated with flag
`coast_limit_hold`. The limit is NOT applied in mode `telemetry`: dead reckoning on vx, vy, wz stays good for tens of seconds, freezing the pose would be worse; there the model only
reports `degraded` (no accepted fix for > stale_fix_s) with the same names as E04Y (tracking_state degraded, flag labeled_fallback_coast, residuals.kin.coast_s / residuals.tel.coast_s).
`telemetry_stamp_limit_ns`: refuse a telemetry file with a stamp above it (dev default = the S10a laps-0-5 export limit, so no audit-lap telemetry can enter a dev run); null = no limit
(release: the hidden recording has other stamps).
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import time
from pathlib import Path

import numpy as np

from camloc.contracts import Capabilities, ContractError, FirstPose, FrameBundle, StaticMetadata, StepResult
from camloc.filtering.coast_limit import CoastLimiter, CoastLimitError
from camloc.filtering.kin2_filter import Kin2Params
from camloc.filtering.tel_filter import TelParams
from camloc.filtering.tel_runtime import LAST_OK_NS, NEEDED
from camloc.io.telemetry import ALLOWED_FIELDS, TELEMETRY_MANIFEST_NAME, TELEMETRY_NAME, TelemetryError, TelemetryStore, assert_no_forbidden
from camloc.models.e04y import DEFAULT_COAST_LIMIT_S, FROZEN_FILTER_SHA256, _flags_to_src, filter_params_sha256

# sha256 of canonical JSON {"tel","kin","latency_scale","fallback_age_s"} for configs/models/e10-v3.yaml (S10b V3, not frozen: S10e decides); scripts/s10c_hash.py recomputes it.
# None = no code pin yet (the S10e freeze sets it). The config pin `expected_param_sha256` is mandatory either way.
FROZEN_PARAM_SHA256: str | None = None


REPO_ROOT = Path(__file__).resolve().parents[3]


def _canon(d) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def resolve_filter(fcfg: dict) -> dict:
    """Validate the `filter` block and load its parameter blocks: returns {module, class, blocks (params dict), extra, file_sha256}. Refuses an unpinned or changed params_file."""
    if not isinstance(fcfg, dict) or not fcfg.get("module") or not fcfg.get("class"):
        raise ContractError("E04T requires params.filter {module, class, params_file | params}")
    if not str(fcfg["module"]).startswith("camloc."):
        raise ContractError(f"filter module must live under camloc.*, got {fcfg['module']!r}")
    file_sha = None
    if fcfg.get("params_file"):
        import yaml
        pf = Path(fcfg["params_file"])
        pf = pf if pf.is_absolute() else REPO_ROOT / pf
        raw = pf.read_bytes()
        file_sha = hashlib.sha256(raw).hexdigest()
        if not fcfg.get("expected_file_sha256"):
            raise ContractError("filter.params_file requires filter.expected_file_sha256")
        if file_sha != fcfg["expected_file_sha256"]:
            raise ContractError(f"filter params_file {pf} differs from its pin: {file_sha} != {fcfg['expected_file_sha256']}")
        blocks = yaml.safe_load(raw)["params"]
    else:
        blocks = fcfg.get("params")
        if not isinstance(blocks, dict):
            raise ContractError("filter needs params_file or inline params")
    for k in ("tel", "kin"):
        if k not in blocks:
            raise ContractError(f"filter params lack the {k!r} block")
    extra = dict(fcfg.get("extra") or {})
    for name, spec in extra.items():
        if spec.get("from") not in blocks:
            raise ContractError(f"filter.extra.{name}: params block {spec.get('from')!r} missing")
        if not str(spec.get("module", "")).startswith("camloc."):
            raise ContractError("filter.extra modules must live under camloc.*")
    return {"module": fcfg["module"], "class": fcfg["class"], "blocks": blocks, "extra": extra, "file_sha256": file_sha}


def param_sha256(f: dict, latency_scale, fallback_age_s) -> str:
    """Canonical-JSON pin of the selected filter (module, class, raw parameter blocks incl. extras) + latency/fallback."""
    b = f["blocks"]
    d = {"filter": {"module": f["module"], "class": f["class"]}, "tel": b["tel"], "kin": b["kin"],
         "extra": {n: {"module": sp["module"], "class": sp["class"], "from": sp["from"], "block": b[sp["from"]]} for n, sp in sorted(f["extra"].items())},
         "latency_scale": float(latency_scale), "fallback_age_s": float(fallback_age_s)}
    return hashlib.sha256(_canon(d).encode()).hexdigest()


def build_stream(f: dict, first_pose, t0_ns, store, latency_scale, fallback_age_s):
    cls = getattr(importlib.import_module(f["module"]), f["class"])
    b = f["blocks"]
    if hasattr(cls, "from_params"):
        return cls.from_params(b, first_pose, t0_ns, [], store, latency_scale, fallback_age_s)
    kw = {n: getattr(importlib.import_module(sp["module"]), sp["class"])(**b[sp["from"]]) for n, sp in f["extra"].items()}
    return cls(TelParams(**b["tel"]), Kin2Params(**b["kin"]), first_pose, t0_ns, [], store, latency_scale, fallback_age_s, **kw)


def load_telemetry(directory, stamp_limit_ns=LAST_OK_NS, stale_ns: int = TelemetryStore.DEFAULT_STALE_NS) -> tuple[TelemetryStore, dict]:
    """Open `directory/telemetry.csv` default-deny: the HEADER is checked first (forbidden/GT names and columns outside telemetry-v1 are refused, even if unused);
    only the NEEDED columns are then loaded into the store. Returns (store, info)."""
    import pandas as pd
    d = Path(directory)
    p = d / TELEMETRY_NAME
    if not p.is_file():
        raise TelemetryError(f"no {TELEMETRY_NAME} in {d}")
    header = list(pd.read_csv(p, nrows=0).columns)
    if "stamp_ns" not in header:
        raise TelemetryError("telemetry.csv has no stamp_ns column")
    cols = [c for c in header if c != "stamp_ns"]
    assert_no_forbidden(cols)
    unknown = [c for c in cols if c not in ALLOWED_FIELDS]
    if unknown:
        raise TelemetryError(f"telemetry columns outside the telemetry-v1 allowlist: {unknown}")
    missing = [c for c in NEEDED if c not in cols]
    if missing:
        raise TelemetryError(f"telemetry.csv lacks required fields {missing}")
    df = pd.read_csv(p, usecols=["stamp_ns", *NEEDED], dtype={"stamp_ns": "int64"})
    st = df.stamp_ns.to_numpy(dtype=np.int64)
    if stamp_limit_ns is not None and len(st) and int(st.max()) > int(stamp_limit_ns):
        raise TelemetryError("telemetry rows beyond the allowed stamp limit (audit laps) are not allowed")
    man = json.loads((d / TELEMETRY_MANIFEST_NAME).read_text()) if (d / TELEMETRY_MANIFEST_NAME).is_file() else {}
    store = TelemetryStore(st, {c: df[c].to_numpy(dtype=np.float64) for c in NEEDED}, man, stale_ns)
    info = {"rows": int(len(st)), "first_stamp_ns": int(st[0]) if len(st) else None, "last_stamp_ns": int(st[-1]) if len(st) else None,
            "csv_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "fields_read": list(NEEDED)}
    return store, info


class E04TModel:
    def __init__(self, params: dict | None = None, assets: dict | None = None):
        params = dict(params or {})
        assets = dict(assets or {})
        tdir = assets.pop("telemetry_dir", None)
        self.filt = resolve_filter(params.get("filter"))
        kin = self.filt["blocks"]["kin"]
        self.latency_scale = float(params.get("latency_scale", 0.0))
        self.fallback_age_s = float(params.get("fallback_age_s", 1.0))
        if self.latency_scale != 0.0:
            raise ContractError("E04T runs at latency scale 0 (a live model has no future rows to wait for)")
        if filter_params_sha256(kin, 0.0) != FROZEN_FILTER_SHA256:
            raise ContractError("the camera-only fallback block (kin) must be the frozen Y1 block (S07f hash)")
        got = param_sha256(self.filt, self.latency_scale, self.fallback_age_s)
        pin = params.get("expected_param_sha256")
        if not pin:
            raise ContractError("E04T requires params.expected_param_sha256")
        if got != pin:
            raise ContractError(f"E04T parameters differ from the pinned hash: {got} != {pin}")
        if FROZEN_PARAM_SHA256 and got != FROZEN_PARAM_SHA256:
            raise ContractError(f"E04T parameters differ from the frozen hash in code: {got} != {FROZEN_PARAM_SHA256}")
        try:                                                             # outside the pinned block (as in E04Y); None / 0 disables (camloc.filtering.coast_limit)
            self.limiter = CoastLimiter(params.get("coast_limit_s", DEFAULT_COAST_LIMIT_S), params.get("hold_policy"))
        except CoastLimitError as e:
            raise ContractError(str(e)) from None
        self.coast_limit_ns = self.limiter.limit_ns
        self.stale_fix_s = float(kin.get("stale_fix_s", Kin2Params().stale_fix_s))
        self._tel_last_fix = None
        self.param_sha256 = got
        lim = params.get("telemetry_stamp_limit_ns", LAST_OK_NS)
        self.store, self._tinfo = None, {"available": False, "reason": "no telemetry_dir"}
        if tdir is not None and (Path(tdir) / TELEMETRY_NAME).is_file():
            self.store, info = load_telemetry(tdir, lim)
            self._tinfo = {"available": True, **info}
        elif tdir is not None:
            self._tinfo = {"available": False, "reason": f"no {TELEMETRY_NAME} in {tdir}"}
        self.e04_params = dict(params.get("e04", {}))
        self.e04 = self._make_e04(self.e04_params, assets)
        c = self.e04.capabilities
        self.capabilities = Capabilities(cameras=c.cameras, clock_camera=c.clock_camera, calibration=c.calibration, map_id=c.map_id, telemetry_fields=tuple(NEEDED),
                                         device=c.device, assets=c.assets + (("telemetry_store",) if self.store is not None else ()),
                                         scale_status="metric", online_mapping="disabled", telemetry=True)
        self._ts: TelStream | None = None
        self._n_init = 0
        self._fp = None
        self._t0 = None
        self.n_e04_errors = 0
        self.mode_counts = {"telemetry": 0, "fallback": 0}

    @property
    def n_coast_hold(self) -> int:
        return self.limiter.n_hold

    def _make_e04(self, e04_params: dict, assets):
        from camloc.models.pnp import E04PnpModel
        return E04PnpModel(e04_params, assets)

    def telemetry_info(self) -> dict:
        return {**self._tinfo, "param_sha256": self.param_sha256, "filter": f"{self.filt['module']}:{self.filt['class']}", "params_file_sha256": self.filt["file_sha256"], "mode_counts": dict(self.mode_counts), "n_e04_errors": self.n_e04_errors, "n_coast_hold": self.n_coast_hold}

    # ------------------------------------------------------------------ protocol
    def initialize(self, first_pose: FirstPose, static_metadata: StaticMetadata) -> None:
        self._n_init += 1
        if self._n_init > 1:
            raise ContractError("E04T initialized twice")
        self.e04.initialize(first_pose, static_metadata)
        self._fp = (first_pose.x, first_pose.y, first_pose.yaw)
        self._t0 = first_pose.timestamp_ns
        self._ts = build_stream(self.filt, self._fp, self._t0, self.store, 0.0, self.fallback_age_s)

    def step(self, bundle: FrameBundle) -> StepResult:
        if self._ts is None:
            raise ContractError("E04T step before initialize")
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
        ts = self._ts
        ts.e04.append(row)                                           # visible at this tick only: avail = t + 0
        ts.kin.e04.append(row)
        info = ts.tick(t)
        del ts.e04[:ts._ptr]                                         # consumed rows are never needed again
        ts._ptr = 0
        del ts.kin.e04[:ts.kin._ptr]
        ts.kin._ptr = 0
        t_tel = time.perf_counter() - t0
        mode = info["mode"]
        self.mode_counts[mode] += 1
        # ---- degraded / coast limit (see module docstring)
        lf = ts.kin.filter.cur.last_fix_t
        kgap = None if lf is None else (t - lf) * 1e-9
        kin_degraded = kgap is None or kgap > self.stale_fix_s                 # the Y1 definition (camera-only fallback filter)
        if info.get("accepted") or info.get("ret_used") or info.get("reanchor"):
            self._tel_last_fix = t
        tgap = None if self._tel_last_fix is None else (t - self._tel_last_fix) * 1e-9
        if mode == "fallback":
            degraded = kin_degraded
        else:
            degraded = (tgap is None or tgap > self.stale_fix_s)
        # the Y1 degraded run is tracked in both modes (reported as kin.coast_s); the hold itself only in mode `fallback`
        cr = self.limiter.update(t, kin_degraded, (info["x"], info["y"], info["yaw"]), pose_ok=not degraded, allow_hold=(mode == "fallback"))
        kin_coast_s, hold = cr.coast_s, cr.hold
        coast_s = kin_coast_s if mode == "fallback" else (0.0 if tgap is None else tgap)
        ox, oy, oyaw = cr.pose                                         # on a hold the filters keep running; only the emitted pose is frozen
        out = list(flags)
        out += ["e04t", "mode_" + mode]
        if hold:
            out.append("coast_limit_hold")
        if degraded:
            out.append("labeled_fallback_coast")
        for key, name in (("accepted", "fix_accepted"), ("rejected", "fix_rejected"), ("reanchor", "reanchor"), ("ret_used", "retrieval_weak")):
            if info.get(key):
                out.append(name)
        if info["tel_coast"]:
            out.append("telemetry_coast")
        resid["e04"] = {"src": src, "flags": list(flags), "state": state, "x": ex if ex == ex else None, "y": ey if ey == ey else None,
                        "yaw": eyaw if eyaw == eyaw else None}
        resid["kin"] = {"coast_s": round(kin_coast_s, 3), "gap_s": kgap, "degraded": kin_degraded}
        resid["tel"] = {"mode": mode, "degraded": degraded, "coast_s": round(coast_s, 3), "age_ms": info["tel_age_ms"], "n_samples": info["n_samples"], **{k: info.get(k) for k in ("accepted", "rejected", "reanchor", "ret_used", "n_reanchor")}}
        resid["t_e04_ms"] = round(t_e04 * 1e3, 3)
        resid["t_tel_ms"] = round(t_tel * 1e3, 3)
        lo, hi = info["consumed_min_ns"], info["consumed_max_ns"]
        cm = [v for v in (cmin, lo) if v is not None]
        cx = [v for v in (cmax, hi) if v is not None]
        return StepResult(ox, oy, oyaw, source_flags=tuple(out), tracking_state="degraded" if degraded else "tracking", covariance=info["cov"],
                          consumed_min_ns=min(cm) if cm else None, consumed_max_ns=max(cx) if cx else None, residuals=resid)

    def close(self) -> None:
        self._ts = None
        self.e04.close()


class E04TFbModel(E04TModel):
    """Option A' on the fallback-camera E04 (S08e): same telemetry filter, pins and coast limit; the inner model is `E04FallbackModel` (S04e).
    config: class camloc.models.e04t:E04TFbModel, assets {map_dir, map_dir_fr, [ext_dir]}."""

    def _make_e04(self, e04_params: dict, assets):
        from camloc.models.pnp_fb import E04FallbackModel
        return E04FallbackModel(e04_params, assets)
