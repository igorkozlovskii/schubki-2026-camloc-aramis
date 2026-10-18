"""S4.3: speed-aware v_max for the telemetry filters, as an in-process patch of TelFilter.step (shared code on disk is unchanged).
Variants: const40 (current, no patch) | fixed80 | dyn: v_max_eff(t) = max(40, 1.3 * max telemetry speed seen so far) (causal, GT-free).
Where v_max acts: the plausibility test of consecutive rejected fixes in the re-anchor logic (implied speed <= reanchor_vmax_mult * v_max)."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from camloc.filtering import tel_filter as TF  # noqa: E402

VARIANTS = ("const40", "fixed80", "dyn")
_ORIG = TF.TelFilter.step
STATE = {"variant": "const40", "log": []}


def _step(self, rec, arrived):
    var = STATE["variant"]
    if var != "const40":
        if var == "fixed80":
            vm = 80.0
        else:
            mx = getattr(self, "_s43_vmax_seen", 0.0)
            for _, u in rec.samples:
                mx = max(mx, abs(float(u[0])))
            if rec.u_start is not None:
                mx = max(mx, abs(float(rec.u_start[0])))
            self._s43_vmax_seen = mx
            vm = max(40.0, 1.3 * mx)
        if self.p.v_max != vm:
            self.p = dataclasses.replace(self.p, v_max=vm)
        self._s43_vmax_eff = vm
    return _ORIG(self, rec, arrived)


def install(variant: str):
    assert variant in VARIANTS
    STATE["variant"] = variant
    TF.TelFilter.step = _step if variant != "const40" else _ORIG


def install_kin(vmax: float = 80.0):
    """Camera-only filter (kin_filter) speed clamp [0, v_max] and its re-anchor plausibility: raise v_max (diagnostic for A on the test lap)."""
    from camloc.filtering import kin_filter as KF
    orig = KF.KinFilter.__init__

    def init(self, params, *a, **k):
        orig(self, dataclasses.replace(params, v_max=vmax), *a, **k)
    KF.KinFilter.__init__ = init
