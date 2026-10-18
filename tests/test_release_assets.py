"""PRG-003 R02b -- frozen final model packaging checks: config, driver, export manifest vs the freeze, asset manifest.

No GPU, no model construction, no asset bytes needed (the asset-bundle content is checked by tools/verify_assets.py; the tests that
need the unpacked bundle read CAMLOC_ASSETS and skip when it is absent).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FREEZE = json.loads((ROOT / "manifests/final-v1-freeze.json").read_text())
EXPORT = json.loads((ROOT / "manifests/source-export.json").read_text())
ASSETS = json.loads((ROOT / "manifests/assets-final-v1.json").read_text())
CFG = ROOT / "configs/final/final-v1.yaml"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_exactly_one_runnable_model_config():
    # models/ holds the e04 fl probe config (freeze.code) and the R02a pinned filter params; final/ holds the one release config; no other run config
    assert sorted(p.name for p in (ROOT / "configs/final").glob("*.yaml")) == ["final-v1.yaml"]
    assert sorted(p.name for p in (ROOT / "configs/models").glob("*.yaml")) == ["e04-pnp-rfb-v1.yaml", "e10g-g1.yaml"]


def test_release_config_is_the_freeze_config_with_only_asset_paths_remapped():
    cfg = yaml.safe_load(CFG.read_text())
    assert cfg["schema"] == "final-config-v1" and cfg["model"]["name"] == "mc7BP-dyn"
    assert cfg["vmax"]["floor"] == 40.0 and cfg["vmax"]["mult"] == 1.3 and cfg["poles"]["k_pole"] == 32.0
    row = [r for r in EXPORT["files"] if r["dest_path"] == "configs/final/final-v1.yaml"][0]
    assert row["adaptation"] and sha(CFG) == row["copied_sha256"]
    assert row["orig_sha256"] == FREEZE["config"]["sha256"]                    # the research config this was remapped from
    undo = re.sub(r"\$ASSETS/", "", CFG.read_text())
    assert hashlib.sha256(undo.encode()).hexdigest() == FREEZE["config"]["sha256"]   # remap is the only difference
    for key in ("maps", "networks", "constants", "assets"):
        for v in (cfg["freeze"][key].values() if isinstance(cfg["freeze"][key], dict) else cfg["freeze"][key]):
            assert str(v).startswith("$ASSETS/"), v


def test_freeze_code_all_exported_and_identities_recorded():
    rows = {r["dest_path"]: r for r in EXPORT["files"]}
    ported = {r["dest_path"]: r for r in EXPORT["ported"]}
    for rel, h in FREEZE["code"].items():
        if rel in ported:
            assert ported[rel]["orig_sha256"] == h and sha(ROOT / rel) == ported[rel]["copied_sha256"]
            continue
        assert rel in rows, f"freeze.code file not exported: {rel}"
        assert rows[rel]["orig_sha256"] == h, f"export source differs from the freeze hash: {rel}"
        assert sha(ROOT / rel) == rows[rel]["copied_sha256"]
        assert bool(rows[rel]["adaptation"]) == (rows[rel]["orig_sha256"] != rows[rel]["copied_sha256"])
    assert "src/camloc/v2/multicam/closedloop.py" in rows


def test_asset_manifest_covers_every_frozen_asset_with_equal_hashes():
    groups = {g["key"]: g for g in ASSETS["groups"]}
    for sect in ("maps", "networks"):
        for k, m in FREEZE[sect].items():
            g = groups[k]
            assert g["digest_sha256"] == m["digest_sha256"] and g["files"] == m["files"] and g["in_freeze_manifest"]
    for sect in ("constants", "assets"):
        for f, h in FREEZE[sect].items():
            assert groups[f]["digest_sha256"] == h
    assert ASSETS["freeze_manifest_sha256"]
    for f in ASSETS["files"]:
        assert len(f["sha256"]) == 64 and f["licence"] and f["original_path"]
    # nothing audit/GT/raw: no mcap, no gt, no parquet of the research streams
    for f in ASSETS["files"]:
        assert not re.search(r"(gt\.csv|\.mcap|state_estimation|refline_|S1\.2-streams|cameras-v1)", f["path"]), f["path"]


def test_driver_help_and_no_research_invocations():
    src = (ROOT / "scripts/final_run.py").read_text()
    for bad in ("pixi", "environments/vision", "gpu_slot", "camloc-gpu", "flock"):
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("#",)) and bad in l and "pixi run camloc-run" not in l)
        assert not code.replace("optional", ""), (bad, code[:200])
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    p = subprocess.run([sys.executable, str(ROOT / "scripts/final_run.py"), "run", "--help"], capture_output=True, text=True, cwd="/tmp", env=env)
    assert p.returncode == 0 and "--first-pose" in p.stdout and "--assets" in p.stdout


def test_camera_only_and_gt_guard_present():
    src = (ROOT / "scripts/final_run.py").read_text()
    assert "camera-only fallback is not implemented" in src and "install_gt_guard()" in src and "S4.1 rule" in src


def test_no_parity_or_freeze_subcommands():
    src = (ROOT / "scripts/final_run.py").read_text()
    assert "cmd_parity" not in src and "cmd_freeze" not in src


@pytest.mark.skipif(not os.environ.get("CAMLOC_ASSETS"), reason="CAMLOC_ASSETS (unpacked bundle) not set")
def test_unpacked_bundle_matches_manifest():
    root = Path(os.environ["CAMLOC_ASSETS"])
    for f in ASSETS["files"]:
        assert sha(root / f["path"]) == f["sha256"], f["path"]
