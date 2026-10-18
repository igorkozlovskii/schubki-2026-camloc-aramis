"""Causal multi-camera scheduler (DEVELOPMENT_PLAN.md section 5.2).

Output clock: exposure stamps of the clock camera (`fl`). For tick t each camera contributes its latest image with
stamp <= t, with its age. Older than the stale cutoff -> 'stale' (pixels refused); none <= t -> 'missing'.
An image stamped > t is never reachable: the search is `searchsorted(stamps, t, side='right') - 1`.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from camloc.contracts import CAMERAS, CLOCK_CAMERA, STALE_CUTOFF_NS, CameraFrame, FrameBundle
from .store import CameraStore


def tick_indices(store: CameraStore, start_ns: int, end_ns: int, clock: str = CLOCK_CAMERA) -> np.ndarray:
    """Positions (in the clock camera's index) of ticks with start_ns <= stamp < end_ns - random access by interval."""
    s = store.stamps[clock]
    lo = int(np.searchsorted(s, start_ns, side="left"))
    hi = int(np.searchsorted(s, end_ns, side="left"))
    return np.arange(lo, hi)


def make_bundle(store: CameraStore, tick_pos: int, tick_no: int, clock: str = CLOCK_CAMERA,
                stale_ns: int = STALE_CUTOFF_NS, cameras=None) -> FrameBundle:
    t = int(store.stamps[clock][tick_pos])
    frames: dict[str, CameraFrame] = {}
    for cam in (cameras or CAMERAS):
        s = store.stamps.get(cam)
        if s is None or len(s) == 0:
            frames[cam] = CameraFrame(cam, "missing")
            continue
        j = int(np.searchsorted(s, t, side="right")) - 1
        if j < 0:
            frames[cam] = CameraFrame(cam, "missing")
            continue
        age = t - int(s[j])
        assert age >= 0
        if age > stale_ns:
            frames[cam] = CameraFrame(cam, "stale", int(s[j]), age)
        else:
            frames[cam] = CameraFrame(cam, "ok", int(s[j]), age, _loader=(lambda c=cam, k=j: store.jpeg(c, k)))
    return FrameBundle(timestamp_ns=t, tick_index=tick_no, frames=frames, stale_cutoff_ns=stale_ns)


def iter_bundles(store: CameraStore, start_ns: int, end_ns: int, clock: str = CLOCK_CAMERA,
                 stale_ns: int = STALE_CUTOFF_NS, min_tick_ns: Optional[int] = None, cameras=None,
                 make=None) -> Iterator[FrameBundle]:
    """Yield bundles for clock ticks in [start_ns, end_ns). Ticks stamped before `min_tick_ns` (first pose) are skipped."""
    make = make or make_bundle   # S06 stress hook: an alternative bundle factory with make_bundle's signature (default unchanged)
    for n, pos in enumerate(tick_indices(store, start_ns, end_ns, clock)):
        if min_tick_ns is not None and int(store.stamps[clock][pos]) < min_tick_ns:
            continue
        yield make(store, int(pos), n, clock, stale_ns, cameras)
