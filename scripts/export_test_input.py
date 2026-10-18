#!/usr/bin/env python3
"""Generic safe-input exporter for an unseen raw MCAP (test-day input path). EVALUATOR-SIDE: reads the raw MCAP.

    pixi run python scripts/export_test_input.py --mcap <test.mcap> --out <store-dir>
    pixi run python scripts/export_test_input.py --mcap <file> --out <dir> --range-ns START END     # log_time window (dev checks)
    pixi run python scripts/export_test_input.py --mcap <file> --hash-only                          # sha256 + summary counts, reads no message

Writes a cameras-v1 store (S02 layout, readable by camloc.io.CameraStore) with a generic manifest section and a sibling
`<out>.export_report.json` (anomalies, unknown topics, calibration status). Never overwrites. Exit 0 ok, 1 refused/error,
3 ok but the CAMERA calibration (intrinsics, frame ids, camera extrinsics; camera-only fingerprint) differs from the training
calibration (the release runner will refuse it: "calibration mismatch"). Differences in lidar/radar/gnss/imu extrinsics alone give
exit 0 with a note (`non_camera_differences`). On any failure the `<out>.partial` directory is removed; only a hard kill can leave
it behind, in which case delete it by hand before re-running.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from camloc.contracts import CAMERAS  # noqa: E402
from camloc.io import export_generic as eg  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcap", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="store directory to create (required unless --hash-only)")
    ap.add_argument("--range-ns", nargs=2, type=int, metavar=("START", "END"), default=None,
                    help="restrict to messages with log_time in [START, END) (reader level; dev recording: laps 0-5 only unless --audit-open + token)")
    ap.add_argument("--cameras", nargs="+", default=list(CAMERAS), choices=list(CAMERAS))
    ap.add_argument("--request-topic", nargs="*", default=[], metavar="TOPIC",
                    help="explicitly request extra topics; forbidden/disabled/unknown ones are refused")
    ap.add_argument("--compare-calib", type=Path, default=None, help="optional reference calib.json for a per-camera difference summary")
    ap.add_argument("--canonical-calib", action="store_true",
                    help="with --compare-calib: if the CAMERA calibration is identical to the reference but calib.json differs in bytes (camera_info order, "
                         "lidar/radar/gnss/imu extrinsics), write the reference calib.json bytes (reported in the manifest). Needed for models that compare "
                         "the whole calib.json hash (E04). Never applied when a camera differs.")
    ap.add_argument("--audit-open", action="store_true",
                    help="audit unlock for the dev recording's laps 6-7: needs the token file reports/release/AUDIT_OPEN.token (orchestrator, audit time); every use is ledgered")
    ap.add_argument("--hash-only", action="store_true")
    a = ap.parse_args(argv)
    cmd = "export_test_input.py " + " ".join(argv if argv is not None else sys.argv[1:])
    if a.hash_only:
        try:
            print(json.dumps(eg.hash_only_counts(a.mcap), indent=1))
        except Exception as e:   # noqa: BLE001
            print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        return 0
    if a.out is None:
        ap.error("--out is required")
    try:
        eg.check_audit_range(a.mcap, a.range_ns, kind="cameras", audit_open=a.audit_open, command=cmd)
    except (eg.AuditRefused, OSError) as e:
        print(str(e) if isinstance(e, eg.AuditRefused) else f"ERROR: {e}", file=sys.stderr)
        return 1
    report = a.out.with_name(a.out.name + ".export_report.json")
    if report.exists():
        print(f"REFUSED: {report} exists", file=sys.stderr)
        return 1
    try:
        man = eg.export_generic(a.mcap, a.out, command=cmd, rng=a.range_ns, cameras=a.cameras, requested_topics=a.request_topic,
                                compare_calib=a.compare_calib, report_path=report, canonical_calib=a.canonical_calib)
    except eg.StoreError as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 1
    except Exception as e:   # noqa: BLE001  (catch-all: a clean message, never a traceback; the partial directory is already removed)
        print(f"ERROR: unexpected {type(e).__name__}: {e} (export aborted; nothing left behind; rerun with the same command after fixing the cause)", file=sys.stderr)
        return 1
    g = man["generic"]
    print(json.dumps({"store": str(a.out), "rows": man["index_rows"], "counts": man["counts_per_camera"], "cameras_missing": g["cameras_missing"],
                      "clock": g["clock"], "truncated_input": g["truncated_input"], "read_mode": g["read_mode"],
                      "calibration_mismatch": g["calibration"]["calibration_mismatch"], "camera_fingerprint": g["calibration"]["camera_fingerprint"],
                      "non_camera_differences": g["calibration"]["non_camera_differences"], "calib_normalized_to_reference": g["calibration"]["normalized_to_reference"]["applied"], "calibration_message": g["calibration"]["message"],
                      "calibration_comparison": g["calibration"]["comparison_to_reference"], "unknown_topics": list(g["unknown_topics"]),
                      "anomalies": {k: v["count"] for k, v in g["anomalies"].items()}, "report": str(report),
                      "timing_s": man["_timing_s"]}, indent=1))
    return 3 if g["calibration"]["calibration_mismatch"] else 0


if __name__ == "__main__":
    sys.exit(main())
