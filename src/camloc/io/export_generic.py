"""Generic, allowlisting exporter: any raw MCAP of the task schema -> cameras-v1 safe store (test-day input path, S02b).

EVALUATOR-SIDE module (reads a raw MCAP); never imported by `camloc.io.__init__` or the inference path.

Differences to the recording-bound S02 exporter (`export.py`, untouched):
  * no dependence on recording-v1.json, derived/images.parquet, derived/calib.json or any GT-derived file;
  * default-deny topic policy: only the 7 camera `compressed_image` topics are exported as payloads, `camera_info`, `/tf_static`
    and `/constructor0/track_bounds` are read once for static metadata, every other topic is skipped BY TOPIC before
    deserialization and listed in the manifest (name, schema name, message count) - never written;
  * forbidden / disabled topics (`/tf`, `state_estimation`, ...) are refused even if requested (`requested_topics`);
  * cameras present/missing, counts, stamp ranges and duration are measured; nothing is fabricated;
  * calibration comes from the input (camera_info + tf_static) and is hash-compared with the training calibration;
  * duplicate stamps / out-of-order arrivals / non-jpeg payloads are reported (never silently dropped);
  * a truncated or unindexed MCAP is read as a stream up to the damage and flagged.

Failure policy (S08c): every failure while reading or writing - unexpected schema name on an allowed topic, undecodable message,
manifest that does not validate against `schemas/store_manifest_generic.schema.json`, any other exception - removes the
`<out>.partial` directory and surfaces as a clean `ExportError`/`StoreError` (CLI exit 1, no traceback). Only a hard kill
(SIGKILL, power loss) can leave `<out>.partial` behind; the next export then refuses to start until it is deleted by hand.

The output layout is exactly the S02 store layout, so `camloc.io.CameraStore` and all S02/S06 tooling work unchanged.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from camloc.contracts import CAMERAS
from camloc.util import sha256_bytes, sha256_file
from . import calibfp
from .store import (BOUNDS_NAME, CALIB_NAME, INDEX_COLUMNS, INDEX_NAME, MANIFEST_NAME, STORE_ID, CameraStore,
                    StoreError, check_layout, pack_name)

TOOL_VERSION = "export_generic-1"
NS = "constructor0"
IMG_TOPIC = {c: f"/{NS}/sensor/camera_{c}/compressed_image" for c in CAMERAS}
INFO_TOPIC = {c: f"/{NS}/sensor/camera_{c}/camera_info" for c in CAMERAS}
TF_STATIC = "/tf_static"
BOUNDS_TOPIC = f"/{NS}/track_bounds"
STATIC_TOPICS = tuple(INFO_TOPIC.values()) + (TF_STATIC, BOUNDS_TOPIC)
EXPECTED_SCHEMA = {**{t: "sensor_msgs/msg/CompressedImage" for t in IMG_TOPIC.values()},
                   **{t: "sensor_msgs/msg/CameraInfo" for t in INFO_TOPIC.values()},
                   TF_STATIC: "tf2_msgs/msg/TFMessage", BOUNDS_TOPIC: "visualization_msgs/msg/MarkerArray"}
ALLOWED_TOPICS = tuple(IMG_TOPIC.values()) + STATIC_TOPICS          # == recording-v1.json topics.allowed_topics (tested)
FORBIDDEN_TOPICS = ("/tf",)                                         # == recording-v1.json topics.forbidden_topics (tested)
DISABLED_TOPICS = (f"/{NS}/state_estimation",)                      # == recording-v1.json topics.disabled_pending_Q1 (tested)
FORBIDDEN_NAME_MARKERS = ("state_estimation", "sn_map_state", "yaw_vel", "telemetry", "ground_truth", "/gt")
# hashes of the supplied recording's static files (manifests/inputs/recording-v1.json); a differing input is reported, not reused
TRAIN_CALIB_SHA256 = "3bdea36b925c2145d4ef1a87c36be159ac4974b6278196a5daf04fed7ee1036b"
TRAIN_BOUNDS_SHA256 = "28cf2bfd9ce74c2e5f1ededd5d36866221c341609b27a15e3b5bae94b33de0ff"
CALIB_MISMATCH_MESSAGE = "calibration mismatch: fitted map extrinsics invalid"
CLOCK_FALLBACK_ORDER = ("fr", "cl", "cr", "rl", "rr", "r")          # documented fallback if `fl` is absent
BOUNDS_KEYS = ("track_left", "track_right", "pitlane_left", "pitlane_right")
MAX_LISTED = 200                                                    # per-category cap of itemised anomalies in the manifest


from .unlock import (AUDIT_LEDGER, AUDIT_TOKEN, DEV_RECORDING_PREFIX, DEV_RECORDING_SHA256, LAST_OK_NS, AuditRefused, check_audit_range,  # noqa: E402,F401
                        grant_audit)


class ExportError(StoreError):
    """The MCAP cannot be exported (no images, forbidden request, existing output ...)."""


def topic_class(topic: str) -> str:
    if topic in ALLOWED_TOPICS:
        return "allowed"
    if topic in DISABLED_TOPICS:
        return "disabled_pending_Q1"
    if topic in FORBIDDEN_TOPICS or any(m in topic.lower() for m in FORBIDDEN_NAME_MARKERS):
        return "forbidden"
    return "unknown"


def check_requested(topics) -> None:
    """Forbidden/disabled classes are refused even if requested; unknown topics have no place in the store layout."""
    bad = {t: topic_class(t) for t in topics if topic_class(t) != "allowed"}
    if bad:
        raise ExportError(f"REFUSED: requested topics are not exportable: {bad}")


def assert_schema(topic: str, schema_name) -> None:
    """An allowlisted topic must carry exactly the expected message type, otherwise it is not decoded (clean ExportError)."""
    exp = EXPECTED_SCHEMA.get(topic)
    if exp is not None and schema_name != exp:
        raise ExportError(f"REFUSED: allowed topic {topic} has schema {schema_name!r}, expected {exp!r} (not decoded)")


def _stamp(h) -> int:
    return int(h.stamp.sec) * 10**9 + int(h.stamp.nanosec)


def _decoder_for(channel, schema):
    from mcap_ros2.decoder import DecoderFactory
    return DecoderFactory().decoder_for(channel.message_encoding, schema)


class _Reader:
    """Yields (kind, channel, schema, log_time, decoded_msg_or_None) with the same policy in both read modes.
    Only allowlisted topics are ever decoded; for the rest only (channel, schema, count) are tracked."""

    def __init__(self, path: Path, rng):
        self.path, self.rng = path, rng
        self.mode = None            # 'indexed' | 'stream'
        self.truncated = False
        self.read_error = None
        self.channels: dict = {}    # topic -> {"schema_name", "messages"}

    # ---- indexed ----------------------------------------------------------------------------------------------
    def _try_indexed(self, f):
        from mcap.reader import SeekingReader
        from mcap_ros2.decoder import DecoderFactory
        rd = SeekingReader(f, decoder_factories=[DecoderFactory()])
        summary = rd.get_summary()
        if summary is None or summary.statistics is None or not summary.channels:
            return None
        for ch in summary.channels.values():
            sch = summary.schemas.get(ch.schema_id)
            e = self.channels.setdefault(ch.topic, {"schema_name": sch.name if sch else None, "messages": 0})
            e["messages"] += summary.statistics.channel_message_counts.get(ch.id, 0)
        return rd

    def static_and_images(self, f, want_images):
        """Generator of ('static'|'image', topic, log_time, msg). Static topics: first message of each, no range restriction."""
        rd = None
        try:
            rd = self._try_indexed(f)
        except Exception as e:   # noqa: BLE001  (no footer / bad magic: truncated or unindexed)
            self.read_error = f"indexed read unavailable: {type(e).__name__}: {e}"
            self.channels.clear()
        if rd is not None:
            self.mode = "indexed"
            for t, e in self.channels.items():
                assert_schema(t, e["schema_name"])      # before any message of this file is decoded
            present_static = [t for t in STATIC_TOPICS if t in self.channels]
            seen = set()
            if present_static:
                for _, ch, m, msg in rd.iter_decoded_messages(topics=present_static):
                    if ch.topic in seen:
                        continue
                    seen.add(ch.topic)
                    yield "static", ch.topic, int(m.log_time), msg
                    if len(seen) == len(present_static):
                        break
            topics = [t for t in want_images if t in self.channels]
            if topics:
                s, e = self.rng if self.rng else (None, None)
                for _, ch, m, msg in rd.iter_decoded_messages(topics=topics, start_time=s, end_time=e):
                    yield "image", ch.topic, int(m.log_time), msg
            return
        # ---- stream fallback: one linear pass, nothing outside the allowlist is deserialized ----------------------
        self.mode = "stream"
        from mcap.reader import NonSeekingReader
        f.seek(0)
        rd = NonSeekingReader(f)
        dec: dict = {}
        seen = set()
        try:
            for schema, ch, m in rd.iter_messages(log_time_order=False):
                e = self.channels.setdefault(ch.topic, {"schema_name": schema.name if schema else None, "messages": 0})
                e["messages"] += 1
                if ch.topic in EXPECTED_SCHEMA:
                    assert_schema(ch.topic, schema.name if schema else None)
                if ch.topic in STATIC_TOPICS:
                    if ch.topic in seen:
                        continue
                    seen.add(ch.topic)
                    d = dec.get(ch.id) or dec.setdefault(ch.id, _decoder_for(ch, schema))
                    yield "static", ch.topic, int(m.log_time), d(m.data)
                elif ch.topic in want_images:
                    if self.rng and not (self.rng[0] <= m.log_time < self.rng[1]):
                        continue
                    d = dec.get(ch.id) or dec.setdefault(ch.id, _decoder_for(ch, schema))
                    yield "image", ch.topic, int(m.log_time), d(m.data)
        except ExportError:
            raise
        except Exception as e:   # noqa: BLE001  (damaged tail: keep everything read so far)
            self.truncated = True
            self.read_error = f"stream read stopped: {type(e).__name__}: {e}"


def export_generic(mcap_path, out, command: str = "", rng=None, cameras=CAMERAS, requested_topics=(),
                   compare_calib=None, report_path=None, progress=sys.stderr, canonical_calib: bool = False) -> dict:
    """Export `mcap_path` into the store `out` (never overwritten). On ANY failure the `<out>.partial` directory (and a store that
    failed its final check) is removed and an `ExportError` (a `StoreError`) is raised; nothing the call created survives a failure.
    Details: see `_export`."""
    mcap_path, out = Path(mcap_path), Path(out)
    part = out.with_name(out.name + ".partial")
    pre_out, pre_part = out.exists(), part.exists()      # pre-existing paths are refused by `_export` and never touched here
    try:
        return _export(mcap_path, out, command, rng, cameras, requested_topics, compare_calib, report_path, progress, canonical_calib)
    except BaseException as e:
        for d, pre in ((part, pre_part), (out, pre_out)):
            if not pre and d.exists():
                shutil.rmtree(d, ignore_errors=True)
        if isinstance(e, (StoreError, KeyboardInterrupt, SystemExit)):
            raise
        raise ExportError(f"export failed: {type(e).__name__}: {e} (no output left behind)") from e


def _export(mcap_path, out, command, rng, cameras, requested_topics, compare_calib, report_path, progress, canonical_calib=False) -> dict:
    """Export `mcap_path` into the cameras-v1 store `out` (never overwritten). `rng` = (start_ns, end_ns) on log_time
    (reader-level: messages outside are not read from the chunks). Returns the manifest dict (+ `_timing_s`)."""
    t0 = time.time()
    mcap_path, out = Path(mcap_path), Path(out)
    check_requested(requested_topics)
    for t in requested_topics:
        if t not in tuple(IMG_TOPIC[c] for c in cameras) + STATIC_TOPICS:
            raise ExportError(f"REFUSED: topic {t} not in the exporter's selection")
    if not mcap_path.is_file():
        raise ExportError(f"input MCAP not found: {mcap_path}")
    if out.exists():
        raise ExportError(f"REFUSED: {out} exists (the exporter never overwrites)")
    part = out.with_name(out.name + ".partial")
    if part.exists():
        raise ExportError(f"REFUSED: leftover {part}; delete it explicitly")
    cameras = tuple(c for c in CAMERAS if c in cameras)
    want = [IMG_TOPIC[c] for c in cameras]
    topic_cam = {IMG_TOPIC[c]: c for c in CAMERAS}
    rng = (int(rng[0]), int(rng[1])) if rng else None
    if rng and rng[1] <= rng[0]:
        raise ExportError("empty --range-ns")

    mcap_sha = sha256_file(mcap_path)
    mcap_bytes = mcap_path.stat().st_size
    t_hash = time.time() - t0
    part.mkdir(parents=True)
    packs = {c: open(part / pack_name(c), "wb") for c in CAMERAS}
    pos = {c: 0 for c in CAMERAS}
    rows = []                                   # (camera, stamp_ns, log_time_ns, offset, length, sha256)
    seen_stamp = {c: set() for c in CAMERAS}
    last_stamp: dict = {}
    dup, ooo, badfmt, badframe = [], [], [], []
    n_counts = {"dup": 0, "ooo": 0, "badfmt": 0, "badframe": 0}
    info, tf_static, bounds = {}, {}, None
    exported_msgs = {t: 0 for t in want}
    rd = _Reader(mcap_path, rng)
    t1 = time.time()
    try:
        with open(mcap_path, "rb") as f:
            for kind, topic, log_time, msg in rd.static_and_images(f, want):
                if kind == "static":
                    if topic in INFO_TOPIC.values():
                        c = next(k for k, v in INFO_TOPIC.items() if v == topic)
                        info[c] = dict(width=msg.width, height=msg.height, distortion_model=msg.distortion_model,
                                       K=list(msg.k), D=list(msg.d), P=list(msg.p), frame_id=msg.header.frame_id)
                    elif topic == TF_STATIC:
                        for t in msg.transforms:
                            p, q = t.transform.translation, t.transform.rotation
                            tf_static[f"{t.header.frame_id}->{t.child_frame_id}"] = dict(
                                parent=t.header.frame_id, child=t.child_frame_id, t=[p.x, p.y, p.z], q_xyzw=[q.x, q.y, q.z, q.w])
                    else:
                        bounds = {mk.ns: [[p.x, p.y, p.z] for p in mk.points] for mk in msg.markers}
                    continue
                c = topic_cam[topic]
                stamp = _stamp(msg.header)
                b = bytes(msg.data)
                sha = sha256_bytes(b)
                rec = {"camera": c, "stamp_ns": stamp, "log_time_ns": log_time, "length": len(b), "sha256": sha}
                fmt = str(msg.format).lower()
                if not (fmt.startswith("jpeg") or fmt.startswith("jpg")):
                    n_counts["badfmt"] += 1
                    badfmt.append({**rec, "format": str(msg.format)}) if len(badfmt) < MAX_LISTED else None
                    continue
                if msg.header.frame_id != f"{NS}/camera_{c}":
                    n_counts["badframe"] += 1
                    badframe.append({**rec, "frame_id": msg.header.frame_id}) if len(badframe) < MAX_LISTED else None
                if stamp in seen_stamp[c]:
                    n_counts["dup"] += 1
                    dup.append({**rec, "kept": "first arrival"}) if len(dup) < MAX_LISTED else None
                    continue
                seen_stamp[c].add(stamp)
                if c in last_stamp and stamp < last_stamp[c]:
                    n_counts["ooo"] += 1
                    ooo.append({**rec, "previous_stamp_ns": last_stamp[c]}) if len(ooo) < MAX_LISTED else None
                last_stamp[c] = max(stamp, last_stamp.get(c, stamp))
                packs[c].write(b)
                rows.append((c, stamp, log_time, pos[c], len(b), sha))
                pos[c] += len(b)
                exported_msgs[topic] += 1
                if len(rows) % 20000 == 0 and progress:
                    print(f"  {len(rows)} payloads, {time.time() - t1:.0f} s", file=progress, flush=True)
    finally:
        for p in packs.values():
            p.close()
    t_read = time.time() - t1
    if not rows:
        _discard(part)
        raise ExportError("ERROR: zero camera images exported (no allowlisted image topic with a readable message"
                          + (f" in range {rng}" if rng else "") + (f"; {rd.read_error}" if rd.read_error else "") + ")")

    order = {c: i for i, c in enumerate(CAMERAS)}
    rows.sort(key=lambda r: (order[r[0]], r[1]))
    with open(part / INDEX_NAME, "w", newline="") as f:
        f.write(",".join(INDEX_COLUMNS) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")

    # ---- static metadata from the INPUT (same construction as scripts/build_index.py) -------------------------------------
    calib_cams = {}
    for c in info:
        e = tf_static.get(f"{NS}/base_link->{NS}/camera_{c}")
        if e is None:
            continue
        calib_cams[c] = {**info[c], "T_base_cam": dict(t=e["t"], q_xyzw=e["q_xyzw"])}
    uncalibrated = [c for c in CAMERAS if any(r[0] == c for r in rows) and c not in calib_cams]   # images without usable calibration
    (part / BOUNDS_NAME).write_text(json.dumps(bounds if bounds is not None else {}))
    bounds_sha = sha256_file(part / BOUNDS_NAME)
    # reference calibration: --compare-calib file if given (the release passes its bundle's calibration), else the training calibration
    if compare_calib:
        ref_full = sha256_file(compare_calib)
        ref_pc = calibfp.per_camera(json.loads(Path(compare_calib).read_text()))
        ref_src = f"--compare-calib {Path(compare_calib).name}"
    else:
        ref_full, ref_pc, ref_src = TRAIN_CALIB_SHA256, calibfp.TRAIN_PER_CAMERA, "training calibration (constants)"
    if canonical_calib and not compare_calib:
        raise ExportError("--canonical-calib needs --compare-calib (the reference calibration whose bytes are used)")
    input_text = json.dumps(dict(cameras=calib_cams, tf_static=tf_static), indent=1)
    input_calib_sha = sha256_bytes(input_text.encode())
    cam_fp = calibfp.per_camera({"cameras": calib_cams})
    cam_cmp = calibfp.compare(ref_pc, cam_fp, required=("fl",))
    # Explicit, reported normalisation (never silent): the S04 model compares the WHOLE calib.json hash with the map's. If the camera calibration is
    # identical to the reference (no differing camera) but the file bytes differ (camera_info arrival order, lidar/radar/gnss/imu extrinsics), the
    # reference bytes are written as calib.json. Camera data are identical by fingerprint, so a camera-only model sees the same values.
    normalized = bool(canonical_calib and cam_cmp["status"] != "mismatch" and input_calib_sha != ref_full)
    if normalized:
        (part / CALIB_NAME).write_bytes(Path(compare_calib).read_bytes())
    else:
        (part / CALIB_NAME).write_text(input_text)
    calib_sha = sha256_file(part / CALIB_NAME)
    full_match = calib_sha == ref_full
    mismatch = cam_cmp["status"] == "mismatch"          # camera-only: intrinsics, frame ids, camera extrinsics (S08c)
    non_camera_diff = (input_calib_sha != ref_full) and not mismatch

    counts = {c: sum(1 for r in rows if r[0] == c) for c in CAMERAS}
    present = [c for c in CAMERAS if counts[c]]
    missing = [c for c in CAMERAS if not counts[c]]
    per_cam = {}
    for c in CAMERAS:
        st = [r[1] for r in rows if r[0] == c]
        per_cam[c] = {"count": len(st), "first_stamp_ns": st[0] if st else None, "last_stamp_ns": st[-1] if st else None,
                      "duration_s": round((st[-1] - st[0]) / 1e9, 3) if len(st) > 1 else 0.0}
    clock = {"required": "fl", "available": "fl" in present}
    if "fl" in present:
        clock.update(used="fl", fallback=False, switch_reported=False)
    else:
        fb = next((c for c in CLOCK_FALLBACK_ORDER if c in present), None)
        clock.update(used=fb, fallback=True, switch_reported=True,
                     note=("fl absent: the output clock would have to be another actual camera's exposure stamps "
                           f"({fb}); the release runner does NOT switch silently: it refuses an input without fl")
                     )
    inventory = {}
    for t, e in sorted(rd.channels.items()):
        cls = topic_class(t)
        inventory[t] = {"messages": int(e["messages"]), "class": cls, "schema_name": e["schema_name"],
                        "exported_as_payload": t in exported_msgs and exported_msgs[t] > 0,
                        "exported_messages": int(exported_msgs.get(t, 0))}
    bad = [t for t, v in inventory.items() if v["exported_as_payload"] and v["class"] != "allowed"]
    if bad:
        raise ExportError(f"would export non-allowed topics: {bad}")
    unknown = {t: {"schema_name": v["schema_name"], "messages": v["messages"]} for t, v in inventory.items() if v["class"] == "unknown"}
    refused = {t: v["class"] for t, v in inventory.items() if v["class"] in ("forbidden", "disabled_pending_Q1")}

    cal_cmp = None
    if compare_calib:
        ref = json.loads(Path(compare_calib).read_text())
        cal_cmp = {}
        for c, e in calib_cams.items():
            r = ref.get("cameras", {}).get(c)
            if r is None:
                cal_cmp[c] = "not in reference"
                continue
            dk = max(abs(a - b) for a, b in zip(e["K"], r["K"]))
            dt = max(abs(a - b) for a, b in zip(e["T_base_cam"]["t"] + e["T_base_cam"]["q_xyzw"],
                                                r["T_base_cam"]["t"] + r["T_base_cam"]["q_xyzw"]))
            cal_cmp[c] = {"max_abs_dK": dk, "max_abs_dT_base_cam": dt}

    full = (set(cameras) == set(CAMERAS) and not missing and rng is None and not rd.truncated)
    man = {
        "schema_version": 1,
        "store_id": STORE_ID,
        "complete": bool(full),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "creation_command": command,
        "regenerable": True,
        "source": {"path": mcap_path.name, "bytes": mcap_bytes, "sha256": mcap_sha, "recording_manifest": None},
        "cameras": list(CAMERAS),
        "index_rows": len(rows),
        "counts_per_camera": counts,
        "jpeg_total_bytes": int(sum(r[4] for r in rows)),
        "payload_encoding": "byte-for-byte CompressedImage.data from the MCAP (format jpeg); no decode, no re-encode",
        "index_columns": INDEX_COLUMNS,
        "stamp_semantics": "stamp_ns = header.stamp (exposure); log_time_ns = bag arrival time (not used for causality)",
        "topic_inventory": inventory,
        # reader-compat certificate: the frozen S02 reader (store.py) refuses a manifest without forbidden_records == 0. It counts payloads
        # EXPORTED from non-allowlisted topics, which is 0 by construction (a guard raises otherwise); the informative field is skipped_by_topic.
        "forbidden_records": sum(v["exported_messages"] for v in inventory.values() if v["class"] != "allowed"),
        "skipped_by_topic": {t: {"class": v["class"], "schema_name": v["schema_name"], "messages": v["messages"]}
                             for t, v in inventory.items() if not v["exported_as_payload"] and t not in STATIC_TOPICS},
        "skipped_messages_total": int(sum(v["messages"] for t, v in inventory.items() if not v["exported_as_payload"] and t not in STATIC_TOPICS)),
        "forbidden_policy": "default deny: only camera compressed_image payloads plus static camera_info/tf_static/track_bounds metadata "
                            "are read; /tf, state_estimation, sn_map_state, GT, telemetry and every unknown topic are skipped by topic "
                            "before deserialization and never written",
        "generic": {
            "tool": TOOL_VERSION,
            "input_mode": "raw-mcap-generic",
            "read_mode": rd.mode,
            "read_error": rd.read_error,
            "truncated_input": bool(rd.truncated),
            "log_time_range_ns": list(rng) if rng else None,
            "cameras_requested": list(cameras),
            "cameras_present": present,
            "cameras_missing": missing,
            "per_camera": per_cam,
            "clock": clock,
            "calibration": {
                "source": "input camera_info + tf_static (never derived/calib.json)",
                "sha256": calib_sha, "training_sha256": TRAIN_CALIB_SHA256, "matches_training": bool(full_match),
                "input_derived_full_sha256": input_calib_sha,
                "normalized_to_reference": {"applied": normalized, "requested": bool(canonical_calib),
                                            "note": ("calib.json is the reference file's bytes: camera calibration identical by fingerprint, input file differed "
                                                     "(camera order / non-camera extrinsics); input-derived sha256 above" if normalized else "not applied")},
                "reference": ref_src, "reference_full_sha256": ref_full, "reference_camera_fingerprint": calibfp.canonical_hash(ref_pc),
                "full_sha256": calib_sha, "full_matches_training": bool(full_match),
                "camera_fingerprint": calibfp.canonical_hash(cam_fp), "training_camera_fingerprint": calibfp.TRAIN_CAMERA_FINGERPRINT,
                "camera_matches_training": calibfp.canonical_hash(cam_fp) == calibfp.canonical_hash(ref_pc),
                "per_camera_fingerprints": cam_fp, "camera_comparison": cam_cmp, "required_cameras": ["fl"],
                "non_camera_differences": bool(non_camera_diff),
                "calibration_mismatch": bool(mismatch),
                "message": (CALIB_MISMATCH_MESSAGE if mismatch else
                            "camera calibration identical to training; non-camera extrinsics (lidar/radar/gnss/imu/...) differ: ignored by a camera-only model"
                            if non_camera_diff else
                            "camera calibration identical to training, but not all cameras are present/calibrated: " + json.dumps(cam_cmp)
                            if cam_cmp["status"] == "note" else "identical to the training calibration"),
                "cameras_calibrated": sorted(calib_cams), "cameras_uncalibrated": sorted(set(uncalibrated)),
                "tf_static_transforms": len(tf_static),
                "comparison_to_reference": cal_cmp,
            },
            "track_bounds": {"present": bounds is not None, "keys": sorted(bounds) if bounds else [],
                             "sha256": bounds_sha, "training_sha256": TRAIN_BOUNDS_SHA256,
                             "matches_training": bounds_sha == TRAIN_BOUNDS_SHA256},
            "unknown_topics": unknown,
            "refused_topics_present": refused,
            "requested_topics": list(requested_topics),
            "anomalies": {
                "duplicate_stamps": {"count": n_counts["dup"], "policy": "first arrival kept, later duplicates not written (listed here)", "items": dup},
                "out_of_order_arrivals": {"count": n_counts["ooo"], "policy": "kept and sorted by stamp in the index", "items": ooo},
                "non_jpeg_payloads": {"count": n_counts["badfmt"], "policy": "not written (listed here)", "items": badfmt},
                "frame_id_mismatch": {"count": n_counts["badframe"], "policy": "kept, flagged", "items": badframe},
            },
            "stamp_range_ns": [min(r[1] for r in rows), max(r[1] for r in rows)],
            "export_command": command,
        },
    }
    files = {}
    # manifest hashes are filled after the other files exist; the manifest itself is not self-hashed (as in S02)
    man["files"] = {}
    for p in sorted(part.iterdir()):
        files[p.name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    man["files"] = files
    (part / MANIFEST_NAME).write_text(json.dumps(man, indent=1) + "\n")
    validate_manifest(man)
    check_layout(part)
    part.rename(out)
    man["_timing_s"] = {"mcap_hash": round(t_hash, 1), "read_write": round(t_read, 1), "total": round(time.time() - t0, 1)}
    if report_path is not None:
        Path(report_path).write_text(json.dumps({k: v for k, v in man["generic"].items()} | {"store": str(out), "index_rows": len(rows)}, indent=1) + "\n")
    # the S02 reader must accept what was just written
    with CameraStore(out, verify=False):
        pass
    return man


def _discard(d: Path) -> None:
    shutil.rmtree(d, ignore_errors=True)


def validate_manifest(man: dict) -> None:
    """Validate the manifest against schemas/store_manifest_generic.schema.json (mandatory: no schema or jsonschema -> ExportError)."""
    schema_path = Path(__file__).resolve().parents[3] / "schemas" / "store_manifest_generic.schema.json"
    try:
        import jsonschema
    except ImportError as e:
        raise ExportError("jsonschema is not installed in this interpreter; the exporter validates its manifest and will not skip that") from e
    if not schema_path.is_file():
        raise ExportError(f"manifest schema not found: {schema_path}")
    try:
        jsonschema.validate({k: v for k, v in man.items() if not k.startswith("_")}, json.loads(schema_path.read_text()))
    except jsonschema.ValidationError as e:
        raise ExportError(f"manifest does not validate against {schema_path.name}: {e.message} at {'/'.join(map(str, e.absolute_path))}") from e


def hash_only_counts(mcap_path) -> dict:
    """Cheap whole-file pass: sha256 + size + per-topic message counts from the MCAP summary. No message is read or decoded."""
    from mcap.reader import SeekingReader
    p = Path(mcap_path)
    out = {"sha256": sha256_file(p), "bytes": p.stat().st_size, "topics": {}}
    with open(p, "rb") as f:
        s = SeekingReader(f).get_summary()
        if s is not None:
            for ch in s.channels.values():
                out["topics"][ch.topic] = int(s.statistics.channel_message_counts.get(ch.id, 0))
    return out
