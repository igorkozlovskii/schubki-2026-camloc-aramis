"""Closed-loop helpers for S2.10 (mc7 + borders + poles): pieces that let an image measurement start from the filter's OWN state.

Additive to the S1.16 package: nothing here changes `filter.py`, `fusion.py` or `fusion_age.py`.

 * `MultiCamBase`    : MultiCamTel4Filter with the (p, first_pose, t0_ns, rp) constructor that the S2.3 `BorderG1Filter` / S2.5 `PoleFilter` hand to their base
                       class, so that `class X(PushBorder, BorderG1Filter, MultiCamBase)` is G1 + 7-camera fixes + borders (and poles with `PoleFilter`).
 * `PushBorder`      : a border measurement can be appended while the stream runs (closed loop). Rows must arrive strictly after the tick that produced them
                       (the fit needs time to run), which is what makes the loop causal; they are kept sorted by availability.
 * `drive_closed_loop`: ticks the stream, and after tick i hands the outputs of ticks <= i (and nothing else) to a measurement function; the row it returns is
                       appended for the next ticks.
 * `back_propagate`  : the filter output at tick t moved back to an earlier camera-frame stamp with the filter's own motion (finite difference of its two latest
                       outputs), the S3.2d causal-stream convention.
No ground truth, no files.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from camloc.v2.multicam.filter import MultiCamTel4Filter


class MultiCamBase(MultiCamTel4Filter):
    def __init__(self, p, first_pose, t0_ns, rp):
        super().__init__(p, first_pose, t0_ns, rp, tags={}, require_confirm=False)

    def set_multicam(self, tags: dict, require_confirm: bool, escape_s: float) -> None:
        self.tags = tags
        self.require_confirm = bool(require_confirm)
        self.escape_ns = int(escape_s * 1e9)


class PushBorder:
    """Mixin for classes that own `self.border` (list of dict(stamp, avail, s, z, sig, inc), sorted by avail) and consume it in `step`."""

    def push_border(self, b: dict, now_ns: int) -> None:
        """Insert a row produced at tick `now_ns`. It must become available strictly after that tick; rows are kept sorted by availability (stable), so a slow fit
        (service time above the 500 ms spacing of the fits) does not hold back a later, faster one. Everything already consumed has avail <= now, so the insertion point
        is always at or after the consumption pointer."""
        if b["avail"] <= now_ns:
            raise ValueError("a closed-loop measurement must become available after the tick that produced it")
        k = len(self.border)
        while k > 0 and self.border[k - 1]["avail"] > b["avail"]:
            k -= 1
        self.border.insert(k, b)


def back_propagate(cur: tuple, prev: tuple, dt_prev_s: float, back_s: float) -> tuple:
    """cur, prev: (x, y, yaw, s) of the filter output at tick t and at the previous tick; dt_prev_s their spacing; back_s = t - frame stamp (>= 0).
    -> (x, y, yaw, s) at the frame stamp, velocity = finite difference of the two outputs. With no previous output the velocity is zero."""
    if prev is None or dt_prev_s <= 0:
        return tuple(float(v) for v in cur)
    d = np.array([cur[0] - prev[0], cur[1] - prev[1], (cur[2] - prev[2] + math.pi) % (2 * math.pi) - math.pi, cur[3] - prev[3]]) / dt_prev_s
    return (float(cur[0] - d[0] * back_s), float(cur[1] - d[1] * back_s), float(cur[2] - d[2] * back_s), float(cur[3] - d[3] * back_s))


def drive_closed_loop(stream, ticks, measure: Callable[[int, int, list], dict | None], every: int = 5) -> list:
    """Tick `stream` over `ticks`; after every `every`-th tick i call measure(i, t_ns, outs[:i+1]) -> border dict or None and append it to stream.filter.

    `measure` receives a COPY of the outputs of ticks 0..i only (the filter's own estimates), so it cannot see anything after the tick.
    """
    outs = []
    for i, t in enumerate(ticks):
        outs.append(stream.tick(int(t)))
        if i % every == 0:
            b = measure(i, int(t), list(outs))
            if b is not None:
                stream.filter.push_border(b, int(t))
    return outs
