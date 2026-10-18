"""PRG-003 R02a — runtime-foundation boundary checks (import / path / closure).

Focused, fast checks that the exported source and the one dedicated Pixi
environment stand on their own. NO model is constructed, NO GPU workload is
initialized and NO checkpoint is loaded here (that is R02b/R03).

What these tests prove:

* ``camloc`` and the selected D-086 final-model dependency modules import from
  THIS sibling's ``src/`` — never from a parent research checkout — and the
  vendored ``lightglue`` subset imports as a top-level package.
* The third-party runtime stack (torch/torchvision, numpy/pandas/pyarrow/scipy,
  OpenCV, kornia, transformers, safetensors, yaml/jsonschema, mcap) imports in
  one process, with each package's file origin inside this environment.
* The AST import closure of the final-model start modules equals the exported
  ``src/camloc`` file set recorded in ``manifests/source-export.json`` — no
  missing and no stray runtime module.
* No research-only driver script (``v2_s*``) and no absolute research-checkout
  path is required at import time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LG = ROOT / "external" / "S04-lightglue" / "LightGlue"
EXPORT_MANIFEST = ROOT / "manifests" / "source-export.json"

# A research checkout must NOT be the import origin of anything we load.
FORBIDDEN_PATH_FRAGMENTS = ("schubki_2026/src", "/schubki_2026/scripts")


def _in_tree(module_file: str, *trees: Path) -> bool:
    mf = Path(module_file).resolve()
    return any(str(mf).startswith(str(t.resolve())) for t in trees)


@pytest.fixture(scope="session", autouse=True)
def _sibling_paths():
    # Exactly the activation PYTHONPATH, asserted to be sibling-local.
    for p in (SRC, LG):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    yield


def test_manifest_present_and_valid():
    m = json.loads(EXPORT_MANIFEST.read_text())
    assert m["schema"] == "source-export-v1"
    assert m["file_count"] == len(m["files"]) > 0
    # no forbidden content leaked into the recorded export
    for r in m["files"]:
        for bad in (".git", ".pth", ".pt", ".npz", ".mcap"):
            assert bad not in r["dest_path"], r["dest_path"]


def test_camloc_imports_from_sibling():
    import camloc
    assert _in_tree(camloc.__file__, SRC), camloc.__file__
    assert not _in_tree(camloc.__file__, Path("/home/igor/Projects/other/schubki_2026/src"))


@pytest.mark.parametrize("mod", [
    "camloc.models.mc7",
    "camloc.models.e04t",
    "camloc.models.e04y",
    "camloc.models.pnp",
    "camloc.models.pnp_fb",
    "camloc.models.retrieval",
    "camloc.release.run",
    "camloc.release.package",
    "camloc.release.validate",
    "camloc.v2.multicam.live",
    "camloc.v2.multicam.filter",
    "camloc.geometry.borders",
    "camloc.geometry.residuals",
    "camloc.geometry.camera",
    "camloc.geometry.centerline",
    "camloc.v2.border_fast",
    "camloc.v2.border_probe",
    "camloc.v2.border_seg",
    "camloc.v2.border_seg_att",
    "camloc.v2.road_attitude",
    "camloc.v2.poles.detect",
    "camloc.v2.seg.infer",
    "camloc.v2.seg.model",
    "camloc.v2.meas",
    "camloc.v2.landmark_probe",
    "camloc.v2.body_masks",
    "camloc.io.export_generic",
    "camloc.io.export_telemetry",
])
def test_final_model_modules_import_from_sibling(mod):
    __import__(mod)
    m = sys.modules[mod]
    assert _in_tree(m.__file__, SRC), f"{mod} loaded from {m.__file__}"


def test_vendored_lightglue_subset_imports():
    import lightglue
    assert _in_tree(lightglue.__file__, LG), lightglue.__file__
    # the adapted __init__ exposes exactly ALIKED, LightGlue, match_pair
    assert hasattr(lightglue, "ALIKED")
    assert hasattr(lightglue, "LightGlue")
    assert hasattr(lightglue, "match_pair")
    # the restrictive / unshipped backends are NOT present
    for gone in ("SuperPoint", "DISK", "DoGHardNet", "SIFT"):
        assert not hasattr(lightglue, gone), gone
    from lightglue.lightglue import filter_matches, normalize_keypoints, sigmoid_log_double_softmax  # noqa: F401


def test_superpoint_source_not_shipped():
    assert not (LG / "lightglue" / "superpoint.py").exists()
    for f in LG.rglob("*.pth"):
        pytest.fail(f"weight shipped in LightGlue tree: {f}")
    assert not (LG / ".git").exists()


def test_third_party_runtime_stack_imports_with_origin():
    import numpy, pandas, pyarrow, scipy, yaml, jsonschema  # noqa: F401
    import torch, torchvision  # noqa: F401
    import cv2, kornia, transformers, safetensors  # noqa: F401
    import mcap  # noqa: F401
    import mcap_ros2  # noqa: F401
    env_prefix = ROOT / ".pixi"
    for mod in (numpy, pandas, pyarrow, scipy, torch, torchvision, kornia,
                transformers, safetensors, mcap):
        f = getattr(mod, "__file__", None)
        assert f is not None, mod
        assert _in_tree(f, env_prefix), f"{mod.__name__} from {f}, not this env"


def test_torch_reports_cuda_build():
    import torch
    # No GPU forward pass; just the build tag (the pinned cu121 wheel/conda).
    assert torch.__version__.startswith("2.3.1"), torch.__version__
    assert "cu12" in torch.version.cuda.replace(".", "") or torch.version.cuda.startswith("12"), torch.version.cuda


def test_ast_closure_matches_exported_sources():
    from camloc.release.codeclosure import import_closure
    # Same final-model start set as tools/export_sources.py.
    sys.path.insert(0, str(ROOT / "tools"))
    import export_sources as ex
    m = json.loads(EXPORT_MANIFEST.read_text())
    # R02b: the start set also contains the frozen driver scripts' camloc imports and the freeze.code src modules (recorded in the manifest)
    closure = set(import_closure(str(SRC), ex.MC7_START + ex.D086_START + m["freeze"]["camloc_extra_start"]))
    assert "camloc/v2/multicam/closedloop.py" in closure                      # R02a review: freeze.code module missing from the closure
    # only the numpy feature module of the learned package (imported at import time by the S3.1 prep scripts); no GRU model
    assert not [x for x in closure if x.startswith("camloc/v2/learned") and x not in ("camloc/v2/learned/__init__.py", "camloc/v2/learned/features.py")]
    # manifest runtime src rows must equal the closure exactly
    manifested = {r["dest_path"][len("src/"):] for r in m["files"]
                  if r["role"] == "runtime" and r["dest_path"].startswith("src/")}
    assert manifested == closure, {
        "missing_from_export": sorted(closure - manifested),
        "stray_in_export": sorted(manifested - closure),
    }


def test_no_abs_research_path_at_import_time():
    # R02b: the frozen driver scripts are now live release code in scripts/ (byte-copied or ROOT-adapted); the unused S3.1 training/eval drivers are not.
    for drv in ("v2_s31e_train", "v2_s31f_train", "v2_s28_lib", "v2_s121_masks"):
        assert not (ROOT / "scripts" / f"{drv}.py").exists(), drv
    # No loaded camloc module file resolves into a research checkout.
    for name, mod in list(sys.modules.items()):
        if not name.startswith("camloc"):
            continue
        f = getattr(mod, "__file__", None) or ""
        for frag in FORBIDDEN_PATH_FRAGMENTS:
            assert frag not in f, f"{name} resolved into research checkout: {f}"
