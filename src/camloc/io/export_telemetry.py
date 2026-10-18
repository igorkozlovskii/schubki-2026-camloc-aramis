"""Separate MCAP pass that writes ONLY the allowed car-state telemetry (inputs-v2, S10a, D-024).

EVALUATOR/PACKAGING-SIDE (reads a raw MCAP). Reads exactly one topic, `/constructor0/state_estimation`, and copies the fields
named in `camloc.io.telemetry.ALLOWED_FIELDS` plus the header stamp. Default-deny: every other attribute of the message (pose,
pos/vel accuracy, status, yaw_vel_rad, sn_map_state, any unknown/new field) is skipped and only its NAME is reported. No value
of a skipped field is ever copied, hashed or summarized. Independent of `export_generic.py` (which stays camera-only).

Output directory (never overwritten; built as `<out>.partial` and renamed):
  telemetry.csv            stamp_ns,<allowed fields...>   float64 as repr() (round-trips bitwise); bool -> 0/1
  telemetry_manifest.json  source sha256, exported/skipped/missing field names, counts, rate stats, gaps, semantics
"""
from __future__ import annotations

import csv
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from camloc.util import sha256_file

from .telemetry import (ALLOWED_FIELDS, FIELD_NAMES, FORBIDDEN_NAMES, STAMP_COL, TELEMETRY_ID, TELEMETRY_MANIFEST_NAME,
                        TELEMETRY_NAME, TelemetryError, assert_no_forbidden)

TOPIC = "/constructor0/state_estimation"
SCHEMA_NAME = "sd_localization_msgs/msg/StateEstimation"
GAP_NS = 50_000_000          # gaps above this are listed in the manifest (nominal period 9.5 ms)
MAX_LISTED = 200


class TopicAbsent(TelemetryError):
    """The recording has no state_estimation topic: telemetry is unavailable, run camera-only (exit code 3)."""


def _stamp(h) -> int:
    return int(h.stamp.sec) * 10**9 + int(h.stamp.nanosec)


def _slot_names(msg) -> list[str]:
    s = getattr(msg, "__slots__", None)
    return list(s) if s else [k for k in vars(msg) if not k.startswith("_")]


def _scalar(v) -> float:
    if isinstance(v, (bool, np.bool_)):
        return 1.0 if v else 0.0
    return float(v)


def schema_dict() -> dict:
    """JSON schema of one telemetry.csv row (additionalProperties false: a leaked column fails validation)."""
    props = {STAMP_COL: {"type": "integer"}}
    props.update({k: {"type": "number"} for k in FIELD_NAMES})
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "telemetry_row.schema.json",
            "title": "telemetry-v1 row (allowed car-state fields only)", "type": "object", "properties": props,
            "required": [STAMP_COL], "additionalProperties": False}


def _iter_messages(path: Path, rng):
    from mcap.reader import SeekingReader
    from mcap_ros2.decoder import DecoderFactory
    with open(path, "rb") as f:
        rd = SeekingReader(f, decoder_factories=[DecoderFactory()])
        summ = rd.get_summary()
        if summ is None or not summ.channels:
            raise TelemetryError("MCAP has no index/summary (unindexed files are not supported by the telemetry pass)")
        chans = [c for c in summ.channels.values() if c.topic == TOPIC]
        if not chans:
            raise TopicAbsent(f"topic {TOPIC} absent from {path.name}: telemetry unavailable (camera-only mode)")
        for c in chans:
            sch = summ.schemas.get(c.schema_id)
            if sch is None or sch.name != SCHEMA_NAME:
                raise TelemetryError(f"{TOPIC} has schema {getattr(sch, 'name', None)!r}, expected {SCHEMA_NAME!r}; not decoded")
        s, e = rng if rng else (None, None)
        for _, ch, m, msg in rd.iter_decoded_messages(topics=[TOPIC], start_time=s, end_time=e):
            yield int(m.log_time), msg


def export_telemetry(mcap_path, out, rng=None, command: str = "", hash_source: bool = True) -> dict:
    """Export `mcap_path` into the telemetry directory `out`. `rng` = (a, b) on log_time, [a, b). Returns the manifest."""
    mcap_path, out = Path(mcap_path), Path(out)
    if out.exists():
        raise TelemetryError(f"refusing to overwrite existing {out}")
    if not mcap_path.is_file():
        raise TelemetryError(f"input MCAP not found: {mcap_path}")
    if rng is not None and not (rng[0] < rng[1]):
        raise TelemetryError(f"bad range {rng}")
    part = out.with_name(out.name + ".partial")
    if part.exists():
        shutil.rmtree(part)
    t0 = time.time()
    part.mkdir(parents=True)
    try:
        man = _export(mcap_path, part, rng, command, hash_source)
        (part / TELEMETRY_MANIFEST_NAME).write_text(json.dumps(man, indent=1, sort_keys=True, default=float) + "\n")
        part.rename(out)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    man["_timing_s"] = round(time.time() - t0, 1)
    return man


def _export(mcap_path: Path, part: Path, rng, command: str, hash_source: bool) -> dict:
    rows: list[tuple[int, int, list]] = []        # (stamp, arrival index, values)
    present: list[str] | None = None
    skipped: list[str] = []
    n_read = 0
    for _, msg in _iter_messages(mcap_path, rng):
        if present is None:
            names = _slot_names(msg)
            present = [f for f in FIELD_NAMES if f in names]
            skipped = sorted(n for n in names if n not in ALLOWED_FIELDS and n != "header")
        rows.append((_stamp(msg.header), n_read, [_scalar(getattr(msg, f)) for f in present]))
        n_read += 1
    if present is None:
        raise TopicAbsent(f"no {TOPIC} messages in the requested range")
    missing = [f for f in FIELD_NAMES if f not in present]
    assert_no_forbidden(present)
    order = sorted(rows, key=lambda r: (r[0], r[1]))                       # stable: equal stamps keep arrival order
    n_ooo = int(sum(1 for a, b in zip(rows, rows[1:]) if b[0] < a[0]))
    kept, dup_stamps, last = [], [], None
    for r in order:
        if last is not None and r[0] == last:
            dup_stamps.append(r[0])
            continue
        kept.append(r)
        last = r[0]
    with open(part / TELEMETRY_NAME, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([STAMP_COL] + present)
        for st, _, vals in kept:
            w.writerow([st] + [repr(v) for v in vals])
    st = np.array([r[0] for r in kept], dtype=np.int64)
    dt = np.diff(st) / 1e6 if len(st) > 1 else np.array([0.0])
    gaps = [{"after_stamp_ns": int(st[i]), "gap_ms": round(float(dt[i]), 3)} for i in np.nonzero(dt * 1e6 > GAP_NS)[0]]
    csv_sha = sha256_file(part / TELEMETRY_NAME)
    return {
        "schema_version": 1, "manifest_id": TELEMETRY_ID, "contract": "inputs-v2",
        "source": {"path": mcap_path.name, "bytes": mcap_path.stat().st_size, "sha256": sha256_file(mcap_path) if hash_source else None,
                   "topic": TOPIC, "schema": SCHEMA_NAME, "range_log_time_ns": list(rng) if rng else None},
        "file": {"name": TELEMETRY_NAME, "sha256": csv_sha, "bytes": (part / TELEMETRY_NAME).stat().st_size},
        "stamp_semantics": "int64 ns from header.stamp (same ROS clock as the image headers); NOT recorder log_time; strictly increasing",
        "units": {f: ALLOWED_FIELDS[f][0] for f in present},
        "fields": {"exported": present, "skipped_by_name": skipped, "allowed_but_missing_in_message": missing,
                   "forbidden_names_seen": sorted(n for n in skipped if n in FORBIDDEN_NAMES)},
        "counts": {"messages_read": n_read, "rows_written": len(kept), "duplicate_stamps_dropped": len(dup_stamps),
                   "out_of_order_arrivals": n_ooo, "duplicate_stamps_listed": dup_stamps[:MAX_LISTED]},
        "rate": {"median_dt_ms": round(float(np.median(dt)), 4), "rate_hz": round(1000.0 / float(np.median(dt)), 3) if np.median(dt) > 0 else None,
                 "max_dt_ms": round(float(dt.max()), 3), "n_gaps_over_50ms": len(gaps), "gaps_over_50ms": gaps[:MAX_LISTED]},
        "first_stamp_ns": int(st[0]), "last_stamp_ns": int(st[-1]),
        "creation_command": command, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "regenerable": True,
    }
