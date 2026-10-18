"""Release runtime (S08a/S08c): bundle manifest/verification, release config, offline one-command runner.

Inference-side package: it imports no evaluator, GT or derived/ module. `camloc.release.bundle` and `camloc.release.codeid` are
stdlib-only (plain `python3 -I scripts/verify_release.py` needs nothing else); PyYAML is imported lazily by `config`/`run` only.
"""
from __future__ import annotations

from .bundle import (BundleError, FORBIDDEN_PARTS, build_manifest, file_entry, verify_bundle, write_manifest)  # noqa: F401


def __getattr__(name):   # lazy: config.py needs PyYAML, which the verify path must not require
    if name in ("ReleaseConfigError", "load_release_config", "resolve_model_config"):
        from . import config
        return getattr(config, name)
    raise AttributeError(name)
