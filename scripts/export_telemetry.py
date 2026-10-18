#!/usr/bin/env python3
"""Telemetry export: ONLY the allowed car-state fields of /constructor0/state_estimation (inputs-v2, S10a, D-024).

    pixi run python scripts/export_telemetry.py --input <file.mcap> --out <dir> [--range-ns START END]

EVALUATOR/PACKAGING-SIDE: reads a raw MCAP (the single topic /constructor0/state_estimation) and writes `<dir>/telemetry.csv`
+ `telemetry_manifest.json` (default-deny: pose, GT quality, yaw_vel_rad, sn_map_state and unknown fields are never written).
Never overwrites. Exit 0 ok; 1 error/refused; 3 topic absent (telemetry unavailable: run camera-only, nothing is written).
Dev recording (c0_2026-10-13_* or sha256 fe5a0b95...): a range ending at or before 1791919107438726051 (laps 0-5) is REQUIRED; audit laps 6-7 are refused
unless --audit-open is given and reports/release/AUDIT_OPEN.token exists (orchestrator, audit time); each grant is appended to reports/release/audit_ledger.jsonl.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from camloc.io import export_generic as eg  # noqa: E402
from camloc.io.export_telemetry import TopicAbsent, export_telemetry  # noqa: E402
from camloc.io.telemetry import TelemetryError  # noqa: E402

LAST_OK_NS = eg.LAST_OK_NS
DEV_RECORDING_PREFIX = eg.DEV_RECORDING_PREFIX


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--range-ns", nargs=2, type=int, metavar=("START", "END"), default=None, help="log_time window [START, END)")
    ap.add_argument("--audit-open", action="store_true",
                    help="audit unlock for the dev recording's laps 6-7: needs the token file reports/release/AUDIT_OPEN.token (orchestrator, audit time); every use is ledgered")
    ap.add_argument("--no-hash", action="store_true", help="skip the source sha256 (tests only)")
    a = ap.parse_args(argv)
    cmd = "export_telemetry.py " + " ".join(argv if argv is not None else sys.argv[1:])
    try:
        eg.check_audit_range(a.input, a.range_ns, kind="telemetry", audit_open=a.audit_open, command=cmd)
    except (eg.AuditRefused, OSError) as e:
        print(str(e) if isinstance(e, eg.AuditRefused) else f"ERROR: {e}", file=sys.stderr)
        return 1
    try:
        man = export_telemetry(a.input, a.out, rng=tuple(a.range_ns) if a.range_ns else None, command=cmd, hash_source=not a.no_hash)
    except TopicAbsent as e:
        print(f"NO TELEMETRY: {e}", file=sys.stderr)
        return 3
    except (TelemetryError, OSError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"ok: {man['counts']['rows_written']} rows, {len(man['fields']['exported'])} fields, {man['rate']['rate_hz']} Hz, "
          f"max gap {man['rate']['max_dt_ms']} ms, skipped by name {len(man['fields']['skipped_by_name'])} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
