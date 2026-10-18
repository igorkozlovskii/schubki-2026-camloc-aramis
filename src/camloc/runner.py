"""Online runner: causal scheduling, model protocol enforcement, append-only CSV and diagnostics sidecar.

Reads only: the camera store, one first-pose file, a model config, and the split interval table (lap start/end
stamps). It never opens the MCAP, `derived/`, GT or evaluator files.
"""
from __future__ import annotations

import importlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml

from camloc.contracts import (CAMERAS, CLOCK_CAMERA, STALE_CUTOFF_NS, ContractError, FirstPose, StepResult)
from camloc.io import CameraStore, iter_bundles, load_static_metadata
from camloc.util import canonical_hash, code_hash, sha256_file

CSV_HEADER = "timestamp_ns,x,y,yaw\n"
MODES = ("episode", "continuous")


# ------------------------------------------------------------------ config / split helpers
def load_yaml(path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_split(path) -> dict:
    s = load_yaml(path)
    if s.get("status") != "frozen":
        raise ContractError("split is not frozen")
    return s


def lap_interval(split: dict, lap: int) -> tuple[int, int, str]:
    for L in split["laps"]:
        if L["lap"] == lap:
            return int(L["start_ns"]), int(L["end_ns"]), L["role"]
    raise ContractError(f"lap {lap} not in split {split.get('split_id')}")


def audit_intervals(split: dict) -> list[tuple[int, int, int]]:
    return [(int(L["start_ns"]), int(L["end_ns"]), int(L["lap"])) for L in split["laps"] if L["role"] == "audit"]


def audit_guard(split: dict, start_ns: int, end_ns: int, cfg: dict) -> None:
    """Refuse any range that touches an audit lap (laps 6-7) unless the config explicitly opens the audit with a reason."""
    hit = [lap for (a, b, lap) in audit_intervals(split) if start_ns < b and end_ns > a]
    if not hit:
        return
    if cfg.get("audit_opened") is True and isinstance(cfg.get("audit_reason"), str) and cfg["audit_reason"].strip():
        return
    raise ContractError(f"REFUSED: range [{start_ns}, {end_ns}) reaches audit lap(s) {hit}; the config must carry "
                        f"audit_opened: true and a non-empty audit_reason")


def create_model(config: dict, assets: Optional[dict] = None):
    """Instantiate `config['class']` ('package.module:ClassName'); only camloc.* classes are loadable."""
    mod, _, cls = config["class"].partition(":")
    if not mod.startswith("camloc."):
        raise ContractError(f"model class must live under camloc.*, got {mod!r}")
    return getattr(importlib.import_module(mod), cls)(config.get("params", {}), assets or config.get("assets", {}))


def load_first_pose(path) -> tuple[FirstPose, dict, str]:
    raw = Path(path).read_bytes()
    d = json.loads(raw)
    return FirstPose.from_json_dict(d), d.get("provenance") or {}, __import__("hashlib").sha256(raw).hexdigest()


class _FirstPoseGate:
    """The first pose is consumed exactly once; a second read is a failure."""

    def __init__(self, pose: FirstPose):
        self._pose, self.count = pose, 0

    def consume(self) -> FirstPose:
        self.count += 1
        if self.count > 1:
            raise ContractError("first pose consumed twice")
        return self._pose


@dataclass
class RunSpec:
    store_dir: Path
    model_config: Path
    first_pose: Path
    out_dir: Path
    mode: str
    start_ns: int
    end_ns: int
    split_path: Path
    lap: Optional[int] = None
    clock: str = CLOCK_CAMERA
    stale_ns: int = STALE_CUTOFF_NS
    verify_payloads: bool = True
    command: str = ""
    stress: Optional[object] = None   # S06 hook: a camloc.stress.StressPlan; None (default) leaves behavior unchanged
    allow_telemetry: bool = False     # S10c hook: default off; True lets a model that declares `telemetry: true` run (telemetry_dir None = camera-only)
    telemetry_dir: Optional[Path] = None   # S10c hook: separate telemetry-v1 export directory, passed to the model as assets['telemetry_dir']


def _jl(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"


def run(spec: RunSpec, after_row: Optional[Callable[[int, Path], None]] = None, model=None, model_cfg: Optional[dict] = None) -> dict:
    """Run one model over [start_ns, end_ns) and write predictions.csv (append-only, flushed per row), diagnostics.jsonl
    and run_meta.json to out_dir. Returns the run metadata."""
    if spec.mode not in MODES:
        raise ContractError(f"mode must be one of {MODES}")
    if spec.end_ns <= spec.start_ns:
        raise ContractError("empty range")
    cfg = model_cfg if model_cfg is not None else load_yaml(spec.model_config)
    split = load_split(spec.split_path)
    audit_guard(split, spec.start_ns, spec.end_ns, cfg)
    if spec.mode == "episode":
        if spec.lap is None:
            raise ContractError("episode mode needs a lap")
        a, b, role = lap_interval(split, spec.lap)
        if (a, b) != (spec.start_ns, spec.end_ns):
            raise ContractError("episode range must equal the lap interval exactly")

    fp, prov, fp_sha = load_first_pose(spec.first_pose)
    if prov.get("mode") not in (None, spec.mode) or (spec.mode == "episode" and prov.get("lap") not in (None, spec.lap)):
        raise ContractError(f"first-pose provenance {prov.get('mode')}/{prov.get('lap')} does not match run {spec.mode}/{spec.lap}")
    if fp.timestamp_ns > spec.start_ns + (spec.end_ns - spec.start_ns):
        raise ContractError("first pose is stamped after the range")
    gate = _FirstPoseGate(fp)

    out = Path(spec.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, diag_path, meta_path = out / "predictions.csv", out / "diagnostics.jsonl", out / "run_meta.json"
    for p in (csv_path, diag_path, meta_path):
        if p.exists():
            raise ContractError(f"REFUSED: {p} exists (append-only outputs are never overwritten)")

    store = CameraStore(spec.store_dir, verify=spec.verify_payloads)
    static = load_static_metadata(store)
    own_model = model is None
    if own_model:
        assets = cfg.get("assets")
        if spec.telemetry_dir is not None:   # S10c hook
            assets = {**(assets or {}), "telemetry_dir": str(spec.telemetry_dir)}
        model = create_model(cfg, assets)
    caps = model.capabilities
    caps.validate(store_cameras=CAMERAS, allow_telemetry=bool(spec.allow_telemetry or spec.telemetry_dir is not None))
    if caps.clock_camera != spec.clock:
        raise ContractError(f"model clock camera {caps.clock_camera} != run clock {spec.clock}")
    want_cams = tuple(dict.fromkeys((spec.clock,) + tuple(caps.cameras)))  # least privilege: only declared cameras
    make = None
    if spec.stress is not None:   # S06 hook
        spec.stress.bind(store, spec.start_ns, spec.end_ns, split, want_cams)
        static = spec.stress.apply_static(static)
        make = spec.stress.make_bundle

    meta = {
        "schema_version": 1, "status": "failed", "mode": spec.mode, "lap": spec.lap,
        "start_ns": spec.start_ns, "end_ns": spec.end_ns, "clock": spec.clock, "stale_cutoff_ns": spec.stale_ns,
        "model_id": cfg.get("model_id"), "model_config_sha256": sha256_file(spec.model_config) if spec.model_config else None,
        "model_config_hash": canonical_hash(cfg),
        "capabilities": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(caps).items()},
        "store": {"path": str(spec.store_dir), "store_id": store.manifest["store_id"], "manifest_sha256": store.manifest_sha256,
                  "source_mcap_sha256": store.manifest["source"]["sha256"]},
        "first_pose": {"path": str(spec.first_pose), "sha256": fp_sha, "timestamp_ns": fp.timestamp_ns},
        "split_id": split["split_id"], "split_config_sha256": sha256_file(spec.split_path),
        "code_sha256": code_hash(Path(__file__).parent), "command": spec.command,
        "outputs": {"predictions": str(csv_path), "diagnostics": str(diag_path)},
    }
    if spec.stress is not None:   # S06 hook
        meta["stress"] = spec.stress.meta()
    if caps.telemetry:   # S10c hook: provenance of the telemetry input (only for models that declare it)
        meta["telemetry"] = {"dir": None if spec.telemetry_dir is None else str(spec.telemetry_dir), "info": getattr(model, "telemetry_info", lambda: None)()}
    lat = []
    n_rows = 0
    last_t = None
    t_wall0 = time.perf_counter()
    try:
        with open(csv_path, "x") as fc, open(diag_path, "x") as fd:
            fc.write(CSV_HEADER)
            fc.flush()
            model.initialize(gate.consume(), static)
            fd.write(_jl({"event": "init", "first_pose_consumed": gate.count, "first_pose_sha256": fp_sha,
                          "first_pose_timestamp_ns": fp.timestamp_ns, "model": cfg.get("model_id")}))
            for bundle in iter_bundles(store, spec.start_ns, spec.end_ns, spec.clock, spec.stale_ns,
                                       min_tick_ns=fp.timestamp_ns, cameras=want_cams, make=make):
                t = bundle.timestamp_ns
                if last_t is not None and t <= last_t:
                    raise ContractError(f"non-increasing tick {t} <= {last_t}")
                if any(f.stamp_ns is not None and f.stamp_ns > t for f in bundle.frames.values()):
                    raise ContractError("bundle contains a future image")
                t0 = time.perf_counter_ns()
                res = model.step(bundle)
                dt = time.perf_counter_ns() - t0
                if not isinstance(res, StepResult):
                    raise ContractError("step() must return StepResult")
                res.validate(t)
                fc.write(f"{t},{res.x:.6f},{res.y:.6f},{res.yaw:.6f}\n")
                fc.flush()
                fd.write(_jl({"event": "step", "i": n_rows, "t": t, "latency_ns": dt, "tracking": res.tracking_state,
                              "flags": list(res.source_flags), "consumed": [res.consumed_min_ns, res.consumed_max_ns],
                              "ages_ns": bundle.ages_ns(), "status": bundle.statuses(),
                              "cov": res.covariance, "residuals": res.residuals}))
                lat.append(dt)
                n_rows += 1
                last_t = t
                if after_row is not None:
                    after_row(n_rows, csv_path)
            if gate.count != 1:
                raise ContractError(f"first pose consumed {gate.count} times")
            fd.write(_jl({"event": "close", "rows": n_rows}))
        meta["status"] = "complete"
    finally:
        try:
            model.close()
        finally:
            store.close()
        a = np.array(lat, dtype=np.float64) / 1e6
        meta.update({
            "rows": n_rows, "first_tick_ns": None, "last_tick_ns": last_t, "first_pose_consumed": gate.count,
            "wall_s": round(time.perf_counter() - t_wall0, 3),
            "latency_ms": ({"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
                            "max": float(a.max())} if len(a) else None),
            "predictions_sha256": sha256_file(csv_path) if csv_path.exists() else None,
        })
        if csv_path.exists() and n_rows:
            with open(csv_path) as f:
                f.readline()
                meta["first_tick_ns"] = int(f.readline().split(",")[0])
        meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    return meta
