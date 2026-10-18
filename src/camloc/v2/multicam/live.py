"""S2.9 live (causal, tick by tick) construction of the multi-camera G1 input rows.

The offline research builder (`fusion_age.build_rows_age`, S1.16 Amendment 6, mode `zero`) turns per-camera fix streams into G1 rows after the fact. This module does the
same job one tick at a time, from what exists at that tick: a camera fix computed from the latest frame stamped <= t is keyed to the tick nearest to that frame's exposure
stamp (tolerance 1 ms; a fix without such a tick is dropped and counted), its row is usable from the tick that computed it (latency = that tick - key >= exposure - key, so never before its own image stamp; see `build_rows_probe_tick` for the offline twin), and
the agreement flag (`conf`: another camera has a fix of the SAME key tick within D metres) is computed from the fixes known so far and UPDATED when a later-arriving fix of the
same key tick completes the picture (the shared tag dictionary is mutated in place; the filter's delayed-measurement replay reads it). Nothing here sees ground truth.
"""
from __future__ import annotations

import collections
import math

_DROP_FIELDS = ("x_m", "y_m", "yaw_rad")


def _opt(v):
    return None if v is None or v != v else float(v)


class LiveFixKeyer:
    """Keeps the tick grid, the fl base rows, the per-key fixes and the tag dictionary consumed by `MultiCamTel4Filter`.

    kappa: {camera: {"kxy", "kyaw"}} (the fl entry defaults to 1/1); conf_D: agreement distance in metres or None (no flags)."""

    def __init__(self, kappa: dict, conf_D: float | None = 2.0, tol_ns: int = 1_000_000, keep: int = 256):
        self.kappa = {c: dict(v) for c, v in kappa.items()}
        self.kappa.setdefault("fl", {"kxy": 1.0, "kyaw": 1.0})
        self.conf_D = conf_D
        self.tol_ns = int(tol_ns)
        self.keep = int(keep)
        self.tags: dict = {}
        self._ticks: collections.deque = collections.deque(maxlen=self.keep)
        self._base: collections.OrderedDict = collections.OrderedDict()
        self._fix: dict = {}                # key -> {cam: (x, y, tag)}
        self.info = {"dropped_no_tick": 0, "age_shifted": 0, "n_extra_rows": 0}

    # ---------------------------------------------------------------- registration
    def _register(self, key: int, cam: str, x: float, y: float, tag: dict) -> None:
        here = self._fix.setdefault(int(key), {})
        if self.conf_D is not None:
            for c2, (x2, y2, tag2) in here.items():
                if c2 != cam and math.hypot(x - x2, y - y2) <= self.conf_D:
                    tag["conf"] = True
                    tag2["conf"] = True
        here[cam] = (x, y, tag)

    def start_tick(self, t_ns: int) -> None:
        self._ticks.append(int(t_ns))

    def add_fl(self, row: dict) -> dict:
        """Register the clock camera's row of the current tick (every tick has one; only `pnp` rows carry a fix)."""
        key = int(row["t_ns"])
        self._base[key] = row
        while len(self._base) > self.keep:
            k0, _ = self._base.popitem(last=False)
            self._fix.pop(k0, None)
        if row["src"] == "pnp":
            tag = {"cam": "fl", **self.kappa["fl"], "conf": False}
            self.tags[(key, round(float(row["x_m"]), 5), round(float(row["y_m"]), 5))] = tag
            self._register(key, "fl", float(row["x_m"]), float(row["y_m"]), tag)
        return row

    def key_for(self, img_ns: int):
        """Nearest tick (<= current) to the exposure stamp within the tolerance, else None."""
        best = None
        for t in self._ticks:
            d = abs(t - img_ns)
            if d <= self.tol_ns and (best is None or d < best[0] or (d == best[0] and t < best[1])):
                best = (d, t)
        return None if best is None else best[1]

    def add_fix(self, cam: str, img_ns: int, x: float, y: float, yaw: float, diag: dict | None = None):
        """A valid fix of camera `cam` computed from a frame exposed at `img_ns` (<= the current tick). Returns the G1 row or None (dropped)."""
        key = self.key_for(int(img_ns))
        if key is None or key not in self._base:
            self.info["dropped_no_tick"] += 1
            return None
        diag = diag or {}
        self.info["age_shifted"] += int(key != self._ticks[-1])
        b = self._base[key]
        r = {k: v for k, v in b.items() if k not in _DROP_FIELDS}
        r.update(src="pnp", x_m=float(x), y_m=float(y), yaw_rad=float(yaw), latency_ns=max(0, int(self._ticks[-1]) - int(key)),
                 n_inl=_opt(diag.get("n_inl")), inlier_ratio=_opt(diag.get("inlier_ratio")), rms_px=_opt(diag.get("rms_px")), d_kf_m=_opt(diag.get("d_kf_m")),
                 sim_top=_opt(diag.get("sim_top")), reason=cam)
        tag = {"cam": cam, **self.kappa[cam], "conf": False}
        self.tags[(key, round(r["x_m"], 5), round(r["y_m"], 5))] = tag
        self._register(key, cam, r["x_m"], r["y_m"], tag)
        self.info["n_extra_rows"] += 1
        return r


def order_rows(rows: list) -> list:
    """The stream's arrival order: (key + latency, key), stable (fl row first, then cameras in configured order) = TelStream's own sort in the offline builder."""
    return sorted(rows, key=lambda r: (int(r["t_ns"]) + int(r.get("latency_ns") or 0), int(r["t_ns"])))


def build_rows_probe_tick(rows0: list, ticks, streams: dict, cams: tuple, kappa: dict, conf_D: float | None, tol_ns: int = 1_000_000) -> tuple:
    """Offline twin of the live model: `build_rows_age` (mode zero, extra rows on) with ONE difference in availability. A camera fix is computed at the tick whose bundle supplied its frame (the probe tick
    q.stamp_ns, the latest frame stamped <= that tick), so the live model can use it from that tick on: latency = max(0, probe tick - key). The research `zero` mode lets the pose of the
    image's own tick use it (latency = max(0, exposure - key)), i.e. zero compute time and a frame that was already on the bus one tick earlier; that is the idealisation the live model does
    not make. Everything else (keying, kappa, agreement flags, drop rule) is identical. -> (rows, tags, info)."""
    import copy

    import numpy as np

    rows = copy.deepcopy(rows0)
    for r in rows:
        r["latency_ns"] = 0
    base = {int(r["t_ns"]): r for r in rows}
    tarr = np.asarray(ticks, np.int64)
    fixes = {"fl": {int(r["t_ns"]): (r["x_m"], r["y_m"]) for r in rows if r["src"] == "pnp"}}
    keyed = {}
    info = {"dropped_no_tick": 0, "age_shifted": 0}
    for c in cams:
        if c == "fl":
            continue
        df = streams[c]
        df = df[df.valid]
        out = []
        for q in df.itertuples():
            img = int(q.stamp_ns) - round(float(q.q_cam_age_ms if q.q_cam_age_ms == q.q_cam_age_ms else 0.0) * 1e6)
            j = int(np.searchsorted(tarr, img))
            cand = [k for k in (j - 1, j) if 0 <= k < len(tarr)]
            if not cand:
                info["dropped_no_tick"] += 1
                continue
            k = min(cand, key=lambda i: abs(int(tarr[i]) - img))
            if abs(int(tarr[k]) - img) > tol_ns or int(tarr[k]) not in base:
                info["dropped_no_tick"] += 1
                continue
            info["age_shifted"] += int(int(tarr[k]) != int(q.stamp_ns))
            out.append((int(tarr[k]), q, img))
        keyed[c] = out
        fixes[c] = {key: (float(q.x), float(q.y)) for key, q, _img in out}
    stamps = set()
    for d in fixes.values():
        stamps |= set(d)
    flags = {c: {} for c in fixes}
    if conf_D is not None:
        for t in stamps:
            have = [(c, d[t]) for c, d in fixes.items() if t in d]
            for c, (x, y) in have:
                flags[c][t] = any(c2 != c and np.hypot(x - x2, y - y2) <= conf_D for c2, (x2, y2) in have)
    tags = {}
    k1 = kappa.get("fl", {"kxy": 1.0, "kyaw": 1.0})
    for r in rows:
        if r["src"] == "pnp":
            tags[(int(r["t_ns"]), round(float(r["x_m"]), 5), round(float(r["y_m"]), 5))] = {"cam": "fl", **k1, "conf": bool(flags["fl"].get(int(r["t_ns"]), False))}
    new = []
    for c, lst in keyed.items():
        for key, q, img in lst:
            r = {kk: v for kk, v in base[key].items() if kk not in _DROP_FIELDS}
            r.update(src="pnp", x_m=float(q.x), y_m=float(q.y), yaw_rad=float(q.yaw), latency_ns=max(0, int(q.stamp_ns) - key), n_inl=_opt(q.q_n_inl), inlier_ratio=_opt(q.q_inlier_ratio),
                     rms_px=_opt(q.q_rms_px), d_kf_m=_opt(q.q_d_kf_m), sim_top=_opt(q.q_sim_top), reason=c)
            new.append(r)
            tags[(key, round(r["x_m"], 5), round(r["y_m"], 5))] = {"cam": c, **kappa[c], "conf": bool(flags[c].get(key, False))}
    info["n_extra_rows"] = len(new)
    return rows + new, tags, info


def delivered_latency_ns(key_ns: int, probe_tick_ns: int, service_ms: float) -> int:
    """Delivered-latency rule (S2.9 PROTOCOL section 1): a camera fix computed from the bundle of tick `probe_tick_ns` and costing `service_ms` of wall time (parallel run) is usable
    from probe tick + service time, never before its own tick; the returned latency is relative to the row's key tick, so availability = key + latency."""
    return max(0, int(probe_tick_ns) + int(round(float(service_ms) * 1e6)) - int(key_ns))


def rows_from_live_log(steps: list, kappa: dict, conf_D: float | None = 2.0, rule: str = "probe_tick"):
    """Rebuild the G1 input rows of a live run from its diagnostics (`residuals.mc7`: the fl row and the accepted fixes of each tick). rule 'probe_tick' = what the live model did (outputs must
    equal the live run's up to the telemetry/filter being the same); rule 'delivered' = `delivered_latency_ns`. -> (rows, tags, ticks). No ground truth."""
    kk = LiveFixKeyer(kappa, conf_D)
    rows, ticks = [], []
    for s in steps:
        t = int(s["t"])
        mc = (s.get("residuals") or {}).get("mc7") or {}
        fr = mc.get("fl_row") or {"src": "none"}
        kk.start_tick(t)
        base = {"stream": "e04_fix", "t_ns": t, "latency_ns": 0, "src": fr["src"], "n_inl": fr.get("n_inl")}
        if fr["src"] != "none":
            base.update(x_m=fr["x_m"], y_m=fr["y_m"], yaw_rad=fr["yaw_rad"])
        rows.append(kk.add_fl(base))
        for f in mc.get("fixes") or []:
            r = kk.add_fix(f["cam"], f["img_ns"], f["x"], f["y"], f["yaw"], {"n_inl": f.get("n_inl")})
            if r is None:
                continue
            if rule == "delivered":
                r["latency_ns"] = delivered_latency_ns(r["t_ns"], t, f["ms"])
            rows.append(r)
        ticks.append(t)
    return rows, kk.tags, ticks
