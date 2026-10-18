"""Audit unlock shared by the exporters (scripts/export_test_input.py, scripts/export_telemetry.py) and by run_release.py (S08b rounds 2-3).

The supplied dev recording holds the audit laps 6-7 (laps-v1). Every consumer refuses any range of that recording that ends after the last stamp of lap 5 (or has no
range) unless `--audit-open` is given AND the single-use token file exists. The orchestrator writes the token only at audit time; every grant appends one line to the
audit ledger, and a token cannot be used twice for the same (kind, range). Kept out of export_generic.py so that the release runner's import closure stays free of exporters.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from camloc.util import sha256_file

from .store import StoreError

DEV_RECORDING_SHA256 = "fe5a0b9565b1bc03dd1ad775645722645e135b2f19012cf7bc815a8bcb91d613"
DEV_RECORDING_PREFIX = "c0_2026-10-13_"
LAST_OK_NS = 1791919107438726051                                    # end of the allowed range (laps 0-5, 2 s before lap 6 starts)
_REPO = Path(__file__).resolve().parents[3]
AUDIT_TOKEN = _REPO / "reports/release/AUDIT_OPEN.token"            # module attributes so tests can point them at a tmpdir; CLIs have no flag for them
AUDIT_LEDGER = _REPO / "reports/release/audit_ledger.jsonl"


class AuditRefused(StoreError):
    """A dev-recording range that reaches the audit laps without a valid unlock."""


def check_audit_range(mcap_path, rng, *, kind: str, audit_open: bool = False, command: str = "", dev_sha: str = DEV_RECORDING_SHA256,
                      token: Path | None = None, ledger: Path | None = None) -> dict:
    """Return a dict describing the decision; raise AuditRefused. `kind` is 'cameras' or 'telemetry'. A recording is the dev recording when its name starts with
    DEV_RECORDING_PREFIX or its sha256 equals `dev_sha`. Allowed without unlock: rng given and rng[1] <= LAST_OK_NS."""
    p = Path(mcap_path)
    sha = None
    dev = p.name.startswith(DEV_RECORDING_PREFIX)
    if not dev:
        sha = sha256_file(p)
        dev = sha == dev_sha
    if not dev:
        return {"dev_recording": False, "audit_range": False, "granted": False}
    if rng is not None and int(rng[1]) <= LAST_OK_NS:
        return {"dev_recording": True, "audit_range": False, "granted": False}
    return grant_audit(kind, p.name, sha or sha256_file(p), rng, audit_open=audit_open, command=command, token=token, ledger=ledger)


def grant_audit(kind: str, input_name: str, input_sha256: str, rng, *, audit_open: bool, command: str = "", token: Path | None = None, ledger: Path | None = None,
                subject: str | None = None) -> dict:
    """The token-gated, ledgered unlock shared by the exporters (kind 'cameras' / 'telemetry') and run_release.py (kind 'run'). Raises AuditRefused unless `audit_open`
    and the token file exists and is non-empty and was not already used for the same (kind, range). The ledger line is written BEFORE the work starts: a later failure
    burns the token for that (kind, range, subject) and the orchestrator must write a new token to repeat it. `subject` (run_release: the release_id) lets several
    releases run the same range under one token; None for the exporters."""
    token = Path(token) if token is not None else AUDIT_TOKEN
    ledger = Path(ledger) if ledger is not None else AUDIT_LEDGER
    rng_s = None if rng is None else [int(rng[0]), int(rng[1])]
    msg = (f"REFUSED: the development recording needs a range ending at or before {LAST_OK_NS} (laps 0-5); laps 6-7 are the audit and are reserved. "
           f"The audit unlock needs --audit-open and the token file {token} (written by the orchestrator at audit time)")
    if not audit_open:
        raise AuditRefused(msg)
    if not token.is_file() or not token.read_bytes().strip():
        raise AuditRefused(msg + " [--audit-open given, but the token file is missing or empty]")
    token_sha = sha256_file(token)
    used = []
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if line.strip():
                used.append(json.loads(line))
    if any(u.get("token_sha256") == token_sha and u.get("kind") == kind and u.get("range_ns") == rng_s and u.get("subject") == subject for u in used):
        raise AuditRefused(f"REFUSED: this token was already used for a {kind} step of range {rng_s}{' (' + subject + ')' if subject else ''} (see {ledger}); the unlock is single-use per (kind, range, release); "
                           "after a failed step the orchestrator must write a new token")
    import getpass
    entry = {"time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "user": getpass.getuser(), "kind": kind, "command": command,
             "input": input_name, "input_sha256": input_sha256, "range_ns": rng_s, "subject": subject, "token_sha256": token_sha}
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return {"dev_recording": True, "audit_range": True, "granted": True, "ledger_entry": entry}
