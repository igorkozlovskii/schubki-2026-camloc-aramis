"""Release config: model definition + assets (repo-relative sources, bundle-relative destinations). No absolute private paths."""
from __future__ import annotations

from pathlib import Path

REQUIRED = ("schema_version", "release_id", "model", "assets", "runtime", "dev_recording_sha256")


class ReleaseConfigError(Exception):
    pass


def load_release_config(path) -> dict:
    import yaml   # lazy: only the run/package side needs PyYAML
    cfg = yaml.safe_load(Path(path).read_text())
    miss = [k for k in REQUIRED if k not in cfg]
    if miss:
        raise ReleaseConfigError(f"release config misses {miss}")
    ids = [a["id"] for a in cfg["assets"]]
    if len(ids) != len(set(ids)):
        raise ReleaseConfigError("duplicate asset ids")
    for slot in ("model", "fallback_model"):    # fallback_model (S08e): the camera-only model run at tier T0 when a telemetry release has no usable telemetry
        m = cfg.get(slot)
        if m is None:
            continue
        for k in ("model_id", "class", "params", "asset_ids"):
            if k not in m:
                raise ReleaseConfigError(f"{slot} block misses {k}")
        if not str(m["class"]).startswith("camloc."):
            raise ReleaseConfigError(f"{slot} class must live under camloc.*")
        for name, aid in m["asset_ids"].items():
            if aid not in ids:
                raise ReleaseConfigError(f"{slot} asset {name} -> unknown asset id {aid}")
    if "fallback_model" in cfg and not cfg.get("telemetry"):
        raise ReleaseConfigError("fallback_model needs a `telemetry` block (it is the camera-only tier of a telemetry release)")
    tel = cfg.get("telemetry")
    if tel is not None and "fallback_model" not in cfg:
        raise ReleaseConfigError("a telemetry release needs a camera-only fallback_model (tier T0)")
    for a in cfg["assets"]:
        for k in ("id", "src", "dest", "role", "license"):
            if k not in a:
                raise ReleaseConfigError(f"asset {a.get('id')} misses {k}")
        for k in ("src", "dest"):
            if Path(a[k]).is_absolute() or ".." in Path(a[k]).parts:
                raise ReleaseConfigError(f"asset {a['id']}: {k} must be relative without '..' (no private absolute paths)")
    rt = cfg["runtime"]
    fp = rt.get("calibration_camera_fingerprint")
    if fp is not None and (not isinstance(fp, str) or len(fp) != 64):
        raise ReleaseConfigError("runtime.calibration_camera_fingerprint must be a 64-hex sha256")
    pin = cfg.get("split_pin")
    if pin is not None and not all(len(str(pin.get(k, ""))) == 64 for k in ("file_sha256", "split_hash")):
        raise ReleaseConfigError("split_pin needs file_sha256 and split_hash (64-hex each)")
    if cfg.get("audit_opened"):
        raise ReleaseConfigError("a release config never opens the audit")
    from . import pins   # filter parameter pins are enforced here: every packaging and every run goes through this loader
    try:
        pins.check_pins(cfg)
    except pins.PinError as e:
        raise ReleaseConfigError(str(e)) from None
    return cfg


def asset_by_id(cfg: dict, aid: str) -> dict:
    for a in cfg["assets"]:
        if a["id"] == aid:
            return a
    raise ReleaseConfigError(f"unknown asset {aid}")


def resolve_model_config(cfg: dict, bundle: Path, slot: str = "model") -> dict:
    """Runner-style model config with asset ids replaced by absolute bundle paths (absolute paths only in the per-run copy). slot: model | fallback_model."""
    m = cfg[slot]
    assets = {}
    for name, aid in m["asset_ids"].items():
        assets[name] = str((Path(bundle) / asset_by_id(cfg, aid)["dest"]).resolve())
    for name, rel in (m.get("asset_paths") or {}).items():
        assets[name] = str((Path(bundle) / rel).resolve())
    return {"schema_version": 1, "model_id": m["model_id"], "class": m["class"], "assets": assets, "params": dict(m["params"]),
            "audit_opened": False, "audit_reason": None}
