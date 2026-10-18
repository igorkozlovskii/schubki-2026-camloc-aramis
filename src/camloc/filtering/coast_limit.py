"""Shared coast limiter (S07g hold logic factored out in S08e; used by E04Y and E04T). Standard library only.

The frozen filters coast on their estimated velocity while no fix arrives. When the caller reports a CONTINUOUS degraded run longer than `limit_s`
(strictly more; the run is measured from its first tick), the emitted pose is frozen: the filter itself keeps running and the hold ends with the first
tick that is no longer degraded. `limit_s` None or 0 disables the limiter.

hold_policy (which pose is repeated during a hold):
  * ``last_emitted`` (default, the S07g behaviour): the pose emitted at the tick before the limit was crossed, i.e. after the filter has already coasted `limit_s`;
  * ``last_non_degraded``: the last pose emitted at a tick that was not degraded (falls back to last_emitted while no such pose exists).

The caller passes `run_degraded` (decides the run and the hold) and optionally `pose_ok` (decides what counts as a non-degraded pose for the second policy;
default `not run_degraded`) and `allow_hold` (E04T: no hold in telemetry mode; the run is still tracked so the camera-only coast duration stays reported).
"""
from __future__ import annotations

from dataclasses import dataclass

POLICIES = ("last_emitted", "last_non_degraded")
DEFAULT_COAST_LIMIT_S = 12.0
DEFAULT_HOLD_POLICY = "last_emitted"


class CoastLimitError(ValueError):
    pass


def parse_limit_s(cl) -> int:
    """Validated limit in ns (0 = disabled). Accepts None or a number >= 0 (bool, NaN, negative, strings refused)."""
    if cl is not None and (not isinstance(cl, (int, float)) or isinstance(cl, bool) or cl < 0 or cl != cl):
        raise CoastLimitError(f"coast_limit_s must be null or a number >= 0, got {cl!r}")
    return int(round(float(cl) * 1e9)) if cl else 0


def parse_policy(p) -> str:
    if p is None:
        return DEFAULT_HOLD_POLICY
    if p not in POLICIES:
        raise CoastLimitError(f"hold_policy must be one of {POLICIES}, got {p!r}")
    return p


@dataclass(frozen=True)
class CoastResult:
    pose: tuple          # (x, y, yaw) to emit
    hold: bool
    coast_s: float       # seconds since the first tick of the current continuous degraded run (0.0 outside a run)


class CoastLimiter:
    def __init__(self, limit_s=DEFAULT_COAST_LIMIT_S, hold_policy=None):
        self.limit_ns = parse_limit_s(limit_s)
        self.policy = parse_policy(hold_policy)
        self.deg_start = None        # t_ns of the first tick of the current continuous degraded run
        self.last_out = None         # (x, y, yaw) of the last emitted, non-held pose
        self.last_ok = None          # (x, y, yaw) of the last emitted pose at a tick with pose_ok
        self.n_hold = 0

    def update(self, t_ns: int, run_degraded: bool, pose, *, pose_ok=None, allow_hold: bool = True) -> CoastResult:
        coast_s = 0.0
        hold = False
        if run_degraded:
            if self.deg_start is None:
                self.deg_start = t_ns
            coast_s = (t_ns - self.deg_start) / 1e9
            hold = bool(allow_hold) and bool(self.limit_ns) and (t_ns - self.deg_start) > self.limit_ns and self.last_out is not None
        else:
            self.deg_start = None
        if hold:
            self.n_hold += 1
            held = self.last_ok if (self.policy == "last_non_degraded" and self.last_ok is not None) else self.last_out
            return CoastResult(held, True, coast_s)
        pose = (pose[0], pose[1], pose[2])
        self.last_out = pose
        if (not run_degraded) if pose_ok is None else pose_ok:
            self.last_ok = pose
        return CoastResult(pose, False, coast_s)
