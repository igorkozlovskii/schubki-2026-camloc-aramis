#!/usr/bin/env python3
"""S4.1 telemetry adapter: organizer test bag /constructor0/validation/{twist,accel} -> a telemetry-v1 directory (contract inputs-v2, D-024 / D-067).

    pixi run python scripts/v2_s41_telemetry.py --mcap <bag.mcap> --out artifacts/inputs/hidden-20261013-1608-tel

Reads exactly TWO topics of the raw MCAP (geometry_msgs/msg/TwistStamped and AccelStamped, frame constructor0/base_link); no image, no pose, no other topic is decoded.
DECLARED MAPPING (body frame base_link, REP 103: x forward, y left, z up):
  vx_mps  <- twist.linear.x            vy_mps <- twist.linear.y            v_mps <- hypot(vx, vy)   (training: v_mps == hypot(vx, vy) bitwise)
  wx_radps, wy_radps, wz_radps <- twist.angular.x / y / z        (wz = yaw rate, left positive; no sign change, no frame change)
  ax_mps2, ay_mps2, az_mps2 <- accel.linear.x / y / z            (accel.angular is identically 0 and is not used)
  kappa_radpm <- wz / v (v > 0.5 m/s, else 0);  beta_rad <- atan2(vy, vx)  (training field == vy/vx, small angle)
Fields of telemetry-v1 that the bag does not carry (wheel speeds, slips, pedals, v_raw, steering ...) are NOT written; the models read only v_mps, vx_mps, vy_mps, wz_radps.
Stamp = header.stamp of the message (same clock as the image headers: checked in the manifest). Rows are sorted, duplicate stamps dropped and reported.
UNKNOWN (open question for Igor / organizers): the bag names the topics `validation/...`; their provenance (estimator or reference) is not stated. They carry body-frame rates only
(no pose, no map-frame heading), so they fit the allowed `inputs-v2` field classes, but the written organizer source for this substitution is pending.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from camloc.io.telemetry import ALLOWED_FIELDS, STAMP_COL, TELEMETRY_ID, TELEMETRY_MANIFEST_NAME, TELEMETRY_NAME, assert_no_forbidden  # noqa: E402
from camloc.util import sha256_file  # noqa: E402

TW, AC = "/constructor0/validation/twist", "/constructor0/validation/accel"
SCHEMA = {TW: "geometry_msgs/msg/TwistStamped", AC: "geometry_msgs/msg/AccelStamped"}
FRAME = "constructor0/base_link"
FIELDS = ("v_mps", "vx_mps", "vy_mps", "wx_radps", "wy_radps", "wz_radps", "ax_mps2", "ay_mps2", "az_mps2", "beta_rad", "kappa_radpm")
MAPPING = {"vx_mps": "twist.linear.x", "vy_mps": "twist.linear.y", "v_mps": "hypot(vx, vy)", "wx_radps": "twist.angular.x", "wy_radps": "twist.angular.y",
           "wz_radps": "twist.angular.z", "ax_mps2": "accel.linear.x", "ay_mps2": "accel.linear.y", "az_mps2": "accel.linear.z",
           "kappa_radpm": "wz / v (v > 0.5 m/s else 0)", "beta_rad": "atan2(vy, vx)"}
V_MIN_KAPPA = 0.5


def _stamp(h) -> int:
    return int(h.stamp.sec) * 10**9 + int(h.stamp.nanosec)


def adapt_rows(tw: list, ac: list) -> tuple[np.ndarray, np.ndarray, dict]:
    """tw: [(stamp_ns, vx, vy, vz, wx, wy, wz)], ac: [(stamp_ns, ax, ay, az)] -> (stamps int64, matrix (n, len(FIELDS)), report). Pure function (unit-tested).
    Rows are paired by identical stamp; unpaired rows are dropped and counted; duplicate stamps keep the first arrival; output strictly increasing."""
    rep = {"twist_in": len(tw), "accel_in": len(ac)}
    a_by = {}
    dup_a = 0
    for r in ac:
        if r[0] in a_by:
            dup_a += 1
            continue
        a_by[r[0]] = r
    seen, rows, dup_t, unpaired = set(), [], 0, 0
    for r in sorted(tw, key=lambda r: r[0]):
        if r[0] in seen:
            dup_t += 1
            continue
        seen.add(r[0])
        a = a_by.get(r[0])
        if a is None:
            unpaired += 1
            continue
        _, vx, vy, _vz, wx, wy, wz = r
        _, ax, ay, az = a
        v = math.hypot(vx, vy)
        rows.append((r[0], v, vx, vy, wx, wy, wz, ax, ay, az, math.atan2(vy, vx), wz / v if v > V_MIN_KAPPA else 0.0))
    rep.update(duplicate_twist_stamps=dup_t, duplicate_accel_stamps=dup_a, unpaired_twist=unpaired, rows=len(rows))
    if not rows:
        return np.zeros(0, np.int64), np.zeros((0, len(FIELDS))), rep
    arr = np.array(rows, dtype=np.float64)
    return np.array([r[0] for r in rows], dtype=np.int64), arr[:, 1:], rep


def read_topics(mcap: Path):
    from mcap.reader import SeekingReader
    from mcap_ros2.decoder import DecoderFactory
    tw, ac, frames = [], [], set()
    with open(mcap, "rb") as f:
        rd = SeekingReader(f, decoder_factories=[DecoderFactory()])
        s = rd.get_summary()
        have = {c.topic: s.schemas[c.schema_id].name for c in s.channels.values()}
        for t, sch in SCHEMA.items():
            if have.get(t) != sch:
                raise SystemExit(f"REFUSED: topic {t} has schema {have.get(t)!r}, expected {sch!r}")
        for _, ch, m, msg in rd.iter_decoded_messages(topics=[TW, AC]):
            frames.add(msg.header.frame_id)
            if ch.topic == TW:
                L, A = msg.twist.linear, msg.twist.angular
                tw.append((_stamp(msg.header), L.x, L.y, L.z, A.x, A.y, A.z))
            else:
                L = msg.accel.linear
                ac.append((_stamp(msg.header), L.x, L.y, L.z))
    return tw, ac, frames


def checks(st: np.ndarray, X: np.ndarray, cam_first_ns: int | None, cam_last_ns: int | None) -> dict:
    """Plausibility numbers from the adapted table only (GT-free)."""
    c = {k: X[:, i] for i, k in enumerate(FIELDS)}
    dt = np.diff(st) * 1e-9
    v, wz = c["v_mps"], c["wz_radps"]
    dist = float(np.sum(0.5 * (v[1:] + v[:-1]) * dt))
    yaw = float(np.sum(0.5 * (wz[1:] + wz[:-1]) * dt))
    ay_pred = v * wz
    m = v > 5.0
    slope = float(np.polyfit(ay_pred[m], c["ay_mps2"][m], 1)[0]) if m.sum() > 10 else None
    corr = float(np.corrcoef(ay_pred[m], c["ay_mps2"][m])[0, 1]) if m.sum() > 10 else None
    dv = np.gradient(v, st * 1e-9)
    corr_ax = float(np.corrcoef(dv[m], c["ax_mps2"][m])[0, 1]) if m.sum() > 10 else None
    return {"rows": int(len(st)), "first_stamp_ns": int(st[0]), "last_stamp_ns": int(st[-1]), "duration_s": float((st[-1] - st[0]) * 1e-9),
            "median_dt_ms": float(np.median(dt) * 1e3), "max_dt_ms": float(dt.max() * 1e3), "n_gaps_over_50ms": int((dt > 0.05).sum()),
            "speed_mps": {"min": float(v.min()), "mean": float(v.mean()), "max": float(v.max()), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))},
            "abs_vy_max_mps": float(np.abs(c["vy_mps"]).max()), "wz_radps": {"min": float(wz.min()), "max": float(wz.max())},
            "integrated_distance_m": dist, "integrated_yaw_rad": yaw, "integrated_yaw_deg": math.degrees(yaw),
            "sign_check_ay_vs_v_wz": {"slope": slope, "corr": corr, "note": "ay should be ~ +v*wz in a left-positive body frame (slope ~ +1); a negative slope would flag a sign flip"},
            "sign_check_dv_dt_vs_ax": {"corr": corr_ax},
            "stamp_vs_camera_clock": {"cam_first_ns": cam_first_ns, "cam_last_ns": cam_last_ns,
                                      "first_offset_ms": None if cam_first_ns is None else (int(st[0]) - cam_first_ns) * 1e-6,
                                      "last_offset_ms": None if cam_last_ns is None else (int(st[-1]) - cam_last_ns) * 1e-6}}


def write_dir(out: Path, st, X, rep, mcap: Path, chk: dict, command: str) -> dict:
    part = out.with_name(out.name + ".partial")
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists")
    if part.exists():
        shutil.rmtree(part)
    part.mkdir(parents=True)
    assert_no_forbidden(FIELDS)
    unknown = [f for f in FIELDS if f not in ALLOWED_FIELDS]
    assert not unknown, unknown
    with open(part / TELEMETRY_NAME, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([STAMP_COL, *FIELDS])
        for s, row in zip(st, X):
            w.writerow([int(s), *[repr(float(v)) for v in row]])
    dt = np.diff(st) * 1e-6
    gaps = [{"after_stamp_ns": int(st[i]), "gap_ms": round(float(dt[i]), 3)} for i in np.nonzero(dt > 50.0)[0]]
    man = {"schema_version": 1, "manifest_id": TELEMETRY_ID, "contract": "inputs-v2",
           "source": {"path": mcap.name, "bytes": mcap.stat().st_size, "sha256": sha256_file(mcap), "topic": f"{TW} + {AC}", "schema": " + ".join(SCHEMA.values()), "range_log_time_ns": None},
           "file": {"name": TELEMETRY_NAME, "sha256": sha256_file(part / TELEMETRY_NAME), "bytes": (part / TELEMETRY_NAME).stat().st_size},
           "stamp_semantics": "int64 ns from header.stamp (same ROS clock as the image headers); NOT recorder log_time; strictly increasing",
           "units": {f: ALLOWED_FIELDS[f][0] for f in FIELDS},
           "fields": {"exported": list(FIELDS), "skipped_by_name": [], "allowed_but_missing_in_message": [f for f in ALLOWED_FIELDS if f not in FIELDS], "forbidden_names_seen": []},
           "counts": {"messages_read": rep["twist_in"], "rows_written": int(len(st)), "duplicate_stamps_dropped": rep["duplicate_twist_stamps"], "out_of_order_arrivals": 0,
                      "duplicate_stamps_listed": []},
           "rate": {"median_dt_ms": round(float(np.median(dt)), 4), "rate_hz": round(1000.0 / float(np.median(dt)), 3), "max_dt_ms": round(float(dt.max()), 3),
                    "n_gaps_over_50ms": len(gaps), "gaps_over_50ms": gaps[:200]},
           "first_stamp_ns": int(st[0]), "last_stamp_ns": int(st[-1]), "creation_command": command, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "regenerable": True,
           "adapter": {"tool": "scripts/v2_s41_telemetry.py", "mapping": MAPPING, "frame": FRAME, "rows_report": rep, "checks": chk,
                       "provenance_note": "topics named validation/*; organizer source for using them as telemetry is pending (D-067); body-frame rates only"}}
    try:
        import jsonschema
        sch = json.loads((ROOT / "schemas/telemetry_manifest.schema.json").read_text())
        for k in ("topic", "schema"):       # the shared schema pins the state_estimation topic/message; this adapter's source is the validation pair (declared, not hidden)
            sch["properties"]["source"]["properties"][k].pop("const", None)
        jsonschema.validate(man, sch)
    except ImportError:
        pass
    (part / TELEMETRY_MANIFEST_NAME).write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    part.rename(out)
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcap", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--camera-store", type=Path, default=None, help="exported camera store (only its fl stamp range is read, for the clock-offset check)")
    a = ap.parse_args(argv)
    tw, ac, frames = read_topics(a.mcap)
    if frames != {FRAME}:
        raise SystemExit(f"REFUSED: unexpected frame ids {frames}")
    st, X, rep = adapt_rows(tw, ac)
    cf = cl = None
    if a.camera_store is not None:
        from camloc.io import CameraStore
        with CameraStore(a.camera_store, verify=False) as s:
            cf, cl = int(s.stamps["fl"][0]), int(s.stamps["fl"][-1])
    chk = checks(st, X, cf, cl)
    man = write_dir(a.out, st, X, rep, a.mcap, chk, "v2_s41_telemetry.py " + " ".join(argv if argv is not None else sys.argv[1:]))
    print(json.dumps({"rows": rep, "checks": chk, "csv_sha256": man["file"]["sha256"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
