"""Prediction CSV validation: schemas/prediction.schema.json rows + table rules (header, int64 strictly increasing, finite)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def jsonschema_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("jsonschema")
    except Exception:   # noqa: BLE001
        return None


def validate_prediction_csv(path, schema_path=None, require_schema: bool = True) -> dict:
    """Raises ValueError with the first violation. Returns {rows, first, last, schema_validated, jsonschema_version}.
    No silent fallback: if a schema is given (or required) and jsonschema is not importable this raises, unless the caller passes
    require_schema=False, in which case the result says `schema_validated: False` (table rules still apply)."""
    path = Path(path)
    schema = json.loads(Path(schema_path).read_text()) if schema_path else None
    validator = None
    if schema is None and require_schema:
        raise ValueError("prediction schema file not found: a release run validates against schemas/prediction.schema.json")
    if schema is not None:
        try:
            import jsonschema
            validator = jsonschema.Draft202012Validator(schema)
        except ImportError:
            if require_schema:
                raise ValueError("jsonschema is not installed in this interpreter: the prediction CSV cannot be schema-validated "
                                 "(install it, or call with require_schema=False to record schema_validated: false)") from None
            validator = None
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header != ["timestamp_ns", "x", "y", "yaw"]:
            raise ValueError(f"header must be timestamp_ns,x,y,yaw, got {header}")
        n, last, first = 0, None, None
        for row in r:
            if len(row) != 4:
                raise ValueError(f"row {n + 1}: expected 4 fields")
            try:
                t = int(row[0])
                vals = [float(v) for v in row[1:]]
            except ValueError as e:
                raise ValueError(f"row {n + 1}: {e}")
            if not all(math.isfinite(v) for v in vals):
                raise ValueError(f"row {n + 1}: non-finite value")
            if last is not None and t <= last:
                raise ValueError(f"row {n + 1}: timestamp {t} not strictly increasing (prev {last})")
            if validator is not None:
                err = next(iter(validator.iter_errors({"timestamp_ns": t, "x": vals[0], "y": vals[1], "yaw": vals[2]})), None)
                if err is not None:
                    raise ValueError(f"row {n + 1}: schema: {err.message}")
            first = t if first is None else first
            last = t
            n += 1
    if n == 0:
        raise ValueError("no rows")
    return {"rows": n, "first_timestamp_ns": first, "last_timestamp_ns": last, "schema_validated": validator is not None,
            "schema_validated_with_jsonschema": validator is not None, "jsonschema_version": jsonschema_version() if validator is not None else None}


def validate_register(register: dict) -> list[str]:
    """Structural check of the external-resource register (works without jsonschema). Returns a list of problems (empty = ok)."""
    probs = []
    if register.get("schema") != "release-external-resources-v1":
        probs.append("wrong schema id")
    seen = set()
    for it in register.get("items", []):
        i = it.get("id")
        if i in seen:
            probs.append(f"duplicate id {i}")
        seen.add(i)
        for k in ("kind", "name", "part_of_release", "version", "license", "acquired_by", "redistribution", "limitations"):
            if k not in it or it[k] in (None, ""):
                probs.append(f"{i}: missing {k}")
        sha = it.get("sha256")
        if it.get("part_of_release") and it.get("kind") in ("weights", "data", "software") and not sha:
            probs.append(f"{i}: in-release item without sha256")
        if sha is not None and (len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
            probs.append(f"{i}: bad sha256")
        if not it.get("part_of_release") and "NOT PART OF THE RELEASE" not in it.get("redistribution", ""):
            probs.append(f"{i}: not-in-release item must say NOT PART OF THE RELEASE")
    if not register.get("items"):
        probs.append("no items")
    return probs
