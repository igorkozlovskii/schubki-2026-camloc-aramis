#!/usr/bin/env python3
"""Run a camera-only model over a lap interval or a continuous span. INFERENCE-SIDE: opens only the safe store,
one first-pose file, a model config and the split interval table. No MCAP, no derived/, no GT, no evaluator.

    pixi run python scripts/run_inference.py --mode episode --lap 2 --first-pose <json> --model-config configs/models/e00-constant.yaml --out artifacts/runs/S02-e00-lap2
    pixi run python scripts/run_inference.py --mode continuous --first-pose <json> --start-ns N --end-ns M ...

Episode: range = the lap interval of configs/splits/laps-v1.yaml. Continuous: default range = first-pose stamp .. start of the
first audit lap; a range that reaches lap 6 is refused unless the model config carries audit_opened: true and audit_reason.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from camloc import runner  # noqa: E402
from camloc.contracts import ContractError  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=list(runner.MODES), required=True)
    ap.add_argument("--lap", type=int, default=None, help="episode: lap number (interval from the split)")
    ap.add_argument("--start-ns", type=int, default=None)
    ap.add_argument("--end-ns", type=int, default=None)
    ap.add_argument("--store", type=Path, default=ROOT / "artifacts/inputs/cameras-v1")
    ap.add_argument("--split", type=Path, default=ROOT / "configs/splits/laps-v1.yaml")
    ap.add_argument("--model-config", type=Path, default=ROOT / "configs/models/e00-constant.yaml")
    ap.add_argument("--first-pose", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-verify-payloads", action="store_true", help="skip per-read sha256 of JPEG payloads")
    ap.add_argument("--stress", type=Path, default=None, help="S06: stress manifest (default off; needs --stress-condition)")
    ap.add_argument("--stress-condition", default=None, help="S06: condition id inside the stress manifest")
    ap.add_argument("--telemetry", default=None, metavar="DIR|none",
                    help="S10c: directory of the separately exported telemetry-v1 (telemetry.csv); 'none' lets a telemetry-capable model run camera-only. Default off: such a model is refused")
    a = ap.parse_args(argv)
    if (a.stress is None) != (a.stress_condition is None):
        ap.error("--stress and --stress-condition go together")
    cmd = "run_inference.py " + " ".join(argv if argv is not None else sys.argv[1:])
    try:
        split = runner.load_split(a.split)
        if a.mode == "episode":
            if a.lap is None:
                ap.error("--lap is required for episode mode")
            start, end, _role = runner.lap_interval(split, a.lap)
            if (a.start_ns, a.end_ns) != (None, None) and (a.start_ns, a.end_ns) != (start, end):
                ap.error("episode mode takes its range from the lap; do not pass a different --start-ns/--end-ns")
        else:
            fp, _, _ = runner.load_first_pose(a.first_pose)
            start = a.start_ns if a.start_ns is not None else fp.timestamp_ns
            end = a.end_ns if a.end_ns is not None else min(s for s, _e, _l in runner.audit_intervals(split))
        plan = None
        if a.stress is not None:
            from camloc.stress import StressPlan, load_manifest
            plan = StressPlan(load_manifest(a.stress), a.stress_condition)
        spec = runner.RunSpec(store_dir=a.store, model_config=a.model_config, first_pose=a.first_pose, out_dir=a.out,
                              mode=a.mode, start_ns=start, end_ns=end, split_path=a.split, lap=a.lap,
                              verify_payloads=not a.no_verify_payloads, command=cmd, stress=plan,
                              allow_telemetry=a.telemetry is not None, telemetry_dir=None if a.telemetry in (None, "none") else Path(a.telemetry))
        meta = runner.run(spec)
    except ContractError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(json.dumps({k: meta[k] for k in ("status", "mode", "lap", "rows", "wall_s", "latency_ms", "predictions_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
