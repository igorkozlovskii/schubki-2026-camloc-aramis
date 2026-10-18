"""Stress manifest loader and hash (configs/stress/stress-v1.yaml). The hash covers the conditions, the master seed and
the re-encode quality; it excludes `manifest_hash` and `frozen_utc` themselves."""
from __future__ import annotations

from pathlib import Path

import yaml

from camloc.contracts import ContractError
from camloc.util import canonical_hash

KINDS = ("pixel", "dropout", "skew", "stride", "calib", "startup", "eval_stratum", "control")
EVIDENCE_LEVEL = "synthetic_stress"   # level 2 of DEVELOPMENT_PLAN.md section 9; levels 3 and 4 are absent


def hash_body(m: dict) -> dict:
    return {k: v for k, v in m.items() if k not in ("manifest_hash", "frozen_utc", "frozen_note")}


def manifest_hash(m: dict) -> str:
    return canonical_hash(hash_body(m))


def load_manifest(path, verify: bool = True) -> dict:
    m = yaml.safe_load(Path(path).read_text())
    if m.get("schema_version") != 1 or m.get("manifest_id") != "stress-v1":
        raise ContractError("not a stress-v1 manifest")
    ids = [c["id"] for c in m["conditions"]]
    if len(ids) != len(set(ids)):
        raise ContractError("duplicate condition ids")
    for c in m["conditions"]:
        if c["kind"] not in KINDS:
            raise ContractError(f"{c['id']}: unknown kind {c['kind']}")
        if c.get("tier") not in (1, 2):
            raise ContractError(f"{c['id']}: tier must be 1 or 2")
    if sum(1 for c in m["conditions"] if c["tier"] == 1 and c["kind"] != "eval_stratum") > 8:
        raise ContractError("tier 1 must hold at most 8 conditions")
    if verify and m.get("manifest_hash") != manifest_hash(m):
        raise ContractError(f"manifest hash mismatch: file says {m.get('manifest_hash')}, content is {manifest_hash(m)}")
    return m


def condition(m: dict, cond_id: str) -> dict:
    for c in m["conditions"]:
        if c["id"] == cond_id:
            return c
    raise ContractError(f"condition {cond_id!r} not in manifest {m['manifest_id']}")
