#!/usr/bin/env python3
"""ONE-COMMAND offline camera-only inference for the hidden test run: config in, CSV out.

    CONDA_OVERRIDE_CUDA=12.4 TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_release \\
      pixi run --manifest-path environments/vision/pixi.toml python scripts/run_release.py \\
      --bundle <release-dir> --input <raw test.mcap | safe camera store dir> --first-pose <first_pose.json|row.csv> \\
      --config configs/release/release-rfb-v1.yaml --output pred.csv [--range START_NS END_NS] [--split laps-v1.yaml]

Writes pred.csv (timestamp_ns,x,y,yaw), pred.diagnostics.jsonl, pred.run_record.json and the evidence directory pred.run/.
No network (IP sockets and name resolution are blocked in-process; run it in a network-less namespace to prove it), no downloads, strict
failure on missing assets, all bundle files hash-verified, the first pose is consumed once, GT/state topics cannot enter (the S02
safe-store reader refuses anything outside its allowlist).
Telemetry releases (option A', config block `telemetry`): tier T2 = camera + the allowed telemetry-v1 fields (contract inputs-v2, separate export of
/constructor0/state_estimation); if the topic or a required field is missing, or the export fails, the run falls back to the release's camera-only model
(tier T0) with one stderr WARNING and `tier` in the run record. `--telemetry none` forces T0.
A raw *.mcap `--input` is first exported by the generic allowlisting exporter (scripts/export_test_input.py, a subprocess in an
interpreter that has mcap; its code is hash-pinned by the bundle) into a scratch store under --workdir, which is deleted afterwards
unless --keep-store. Dev recording input (by source sha256) requires --split, and for raw MCAP also --range (audit guard; audit laps
6-7 are never run before S08b). Code identity: this checkout's src/ is the only execution mode; it must equal the bundle's code
snapshot file by file (the run refuses otherwise) and the run record states mode, executing root and tree hash.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # a bundle must stay byte-identical: no __pycache__ inside bundle code or third-party code

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
try:
    import camloc  # noqa: F401
except ImportError:
    sys.exit("REFUSED: camloc is not importable: run scripts/run_release.py from a repository checkout (<repo>/src/camloc). The bundle's "
             "code/ directory is a hash-pinned provenance snapshot, not an execution mode.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True, help="release bundle directory (bundle_manifest.json inside)")
    ap.add_argument("--input", type=Path, required=True, help="safe camera store directory, or a raw *.mcap (exported to a scratch store first)")
    ap.add_argument("--workdir", type=Path, default=None, help="scratch directory for the raw-MCAP store (default <output>.work next to --output)")
    ap.add_argument("--keep-store", action="store_true", help="keep the scratch store exported from a raw MCAP (default: delete after the run)")
    ap.add_argument("--exporter-python", type=Path, default=None, help="python with mcap + mcap-ros2-support + jsonschema (default: current, else <repo>/.pixi/envs/default)")
    ap.add_argument("--first-pose", type=Path, required=True, help="first-pose JSON (timestamp_ns,x,y,yaw) or one-row CSV with header")
    ap.add_argument("--config", type=Path, required=True, help="release config yaml (must equal the bundle's release_config.yaml)")
    ap.add_argument("--output", type=Path, required=True, help="prediction CSV to create (never overwritten)")
    ap.add_argument("--range", nargs=2, type=int, metavar=("START_NS", "END_NS"), default=None)
    ap.add_argument("--split", type=Path, default=None, help="split yaml for the audit guard (mandatory for the dev recording)")
    ap.add_argument("--audit-open", action="store_true",
                    help="audit unlock for a STORE of the dev recording whose range reaches laps 6-7: needs reports/release/AUDIT_OPEN.token (orchestrator, audit time); one ledger line "
                         "(kind 'run') per use, single use per range; opens the audit for this run only. Refused for a raw MCAP input")
    ap.add_argument("--telemetry", default=None, metavar="DIR|none",
                    help="telemetry releases only: a telemetry-v1 directory (scripts/export_telemetry.py output), or 'none' to force the camera-only tier T0. Default: a raw "
                         "MCAP is exported (only /constructor0/state_estimation, only telemetry-v1 fields); a store input without this flag runs T0")
    ap.add_argument("--set", nargs="*", default=[], metavar="K=V", help="override model params, deep-merged (e.g. e04.top_k=3, coast_limit_s=6); an unknown key is refused. "
                                                                        "Benchmark/diagnostics only; recorded in the run record")
    a = ap.parse_args(argv)
    cmd = "run_release.py " + " ".join(argv if argv is not None else sys.argv[1:])
    from camloc.contracts import ContractError
    from camloc.release import BundleError, ReleaseConfigError
    from camloc.release.run import run_release, set_nested
    from camloc.util import sha256_file
    try:
        if sha256_file(a.config) != sha256_file(a.bundle / "release_config.yaml"):
            raise BundleError("--config differs from the bundle's release_config.yaml")
        sp = {}
        for kv in a.set:
            k, _, v = kv.partition("=")
            set_nested(sp, k, json.loads(v))
        rec = run_release(bundle=a.bundle, input_path=a.input, first_pose=a.first_pose, output=a.output, rng=a.range, split=a.split,
                          command=cmd, set_params=sp or None,
                          workdir=a.workdir, keep_store=a.keep_store, exporter_python=a.exporter_python, telemetry=a.telemetry,
                          audit_open=a.audit_open)
    except (ContractError, BundleError, ReleaseConfigError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(json.dumps({k: rec[k] for k in ("status", "release_id", "output", "latency_ms", "wall_s")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
