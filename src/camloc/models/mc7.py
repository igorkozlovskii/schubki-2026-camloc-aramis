"""MC7: release candidate "mc7" (S2.9, D-062). A' (E04 on fl with fallback camera fr -> G1 telemetry filter, tier T2) plus image-time landmark fixes from up to six more cameras.

Per tick t (clock camera fl):
  * fl chain: exactly E04TFbModel (E04FallbackModel -> row -> Tel4Stream(G1)); nothing of it is changed.
  * every `stride`-th tick (tick index since initialize; default 2 = the S1.16 / S1.2 stride) each configured extra camera runs its own E04 pipeline (retrieval -> ALIKED/LightGlue -> PnP, the
    unchanged `E04PnpModel`, camera map, own prior chain from its own accepted fixes) on the latest frame stamped <= t. Cameras run in parallel (one worker thread per camera, `parallel: true`) while
    fl runs in the calling thread, or one after another (`parallel: false`); the outputs are identical, only the wall time differs.
  * a camera's accepted fix becomes a G1 row keyed to the tick nearest to its frame's exposure stamp (`camloc.v2.multicam.live.LiveFixKeyer`; tolerance 1 ms), usable from max(0, exposure - key) after the key,
    with the camera's measurement scale kappa (calibrated on training laps, in the config) and the agreement flag (another camera's fix of the same key tick within D metres). G1 is the
    S1.16 `MultiCamTel4Filter` (G1 `Tel4Filter` with kappa-scaled noise and the agreement gate on re-anchor adoptions, escape after `escape_s` without an accepted fix).
  * the pose of tick t uses data stamped <= t only (frames of the bundle; the runner checks `consumed_max_ns <= t`). The model is synchronous: a tick's output is emitted after all of its cameras
    have answered, so the wall time of a tick is reported (runner `latency_ms`, `residuals.mc7`), never backdated.
Capabilities: cameras = clock + extras, telemetry as E04T. Camera-only tier T0 is NOT this model: the release config's `fallback_model` is the frozen option A.
Extra-camera calibration: each map's camera is compared by digest (`mc7.camera_digests`, from the training calibration) instead of the whole-file calibration hash, so a test recording whose other
(non-camera) calibration entries differ still works; a camera whose digest differs is disabled with a stderr WARNING (the others keep running), never silently used.
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import multiprocessing as mp
import sys
import time

import numpy as np

from camloc.contracts import Capabilities, CameraFrame, ContractError, FirstPose, FrameBundle, StaticMetadata, StepResult
from camloc.models.e04t import E04TFbModel
from camloc.models.pnp import DEFAULTS, E04PnpModel, camera_static_digest
from camloc.models.retrieval import GlobalDescriptor, LocalFeatures, PriorState
from camloc.v2.multicam.filter import MultiCamTel4Filter
from camloc.v2.multicam.live import LiveFixKeyer, order_rows

EXTRA_CAMERAS = ("fr", "cl", "cr", "r", "rl", "rr")
CAM_ORDER = ("fl",) + EXTRA_CAMERAS


def mc7_param_sha256(mc: dict) -> str:
    """Canonical-JSON pin of the multi-camera block (cameras, stride, agreement distance, escape, kappa, camera digests); `parallel` is an execution detail and not pinned."""
    d = {"cameras": list(mc["cameras"]), "stride": int(mc["stride"]), "conf_D_m": float(mc["conf_D_m"]), "escape_s": float(mc["escape_s"]),
         "kappa": {c: {"kxy": float(mc["kappa"][c]["kxy"]), "kyaw": float(mc["kappa"][c]["kyaw"])} for c in sorted(mc["kappa"])},
         "camera_digests": {c: str(v) for c, v in sorted((mc.get("camera_digests") or {}).items())}}
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class _Cam:
    """One extra camera: its own E04 pipeline (map, crop, retrieval, matcher) and prior chain. Pipeline code is E04PnpModel, unchanged."""

    def __init__(self, name: str, map_dir: str, ext_dir, base: dict):
        self.name = name
        self.m = E04PnpModel(base, {"map_dir": map_dir, "ext_dir": ext_dir})
        man = self.m.map.manifest
        if man.get("camera", "fl") != name:
            raise ContractError(f"map {map_dir} was built for camera {man.get('camera')!r}, not {name!r}")
        crop = tuple(man.get("image_crop") or (12, 0, 1494, 540))
        self.m.fe = LocalFeatures(ext_dir, max_kp=base.get("max_kp", DEFAULTS["max_kp"]), det_thr=base.get("det_thr", DEFAULTS["det_thr"]), crop=crop)
        self.m.gd = GlobalDescriptor(ext_dir, pool=man["global_descriptor"]["pool"], crop=crop)
        self.enabled = True
        self.reason = None

    def initialize(self, first_pose: FirstPose, sm: StaticMetadata, want_digest) -> None:
        self.initialize_cam(first_pose, dict(sm.cameras).get(self.name), want_digest)

    def initialize_cam(self, first_pose: FirstPose, cam, want_digest) -> None:
        if cam is None:
            self.enabled, self.reason = False, "camera missing in the static metadata"
            return
        dig = camera_static_digest(cam)
        if want_digest is not None and dig != want_digest:
            self.enabled, self.reason = False, f"camera calibration digest {dig[:12]} != release {str(want_digest)[:12]}"
            return
        m = self.m                                        # = E04PnpModel.initialize with the digest check done above instead of its whole-file/known-camera check
        m.n_init += 1
        m.K = np.array(cam.K, float).reshape(3, 3)
        m.D = np.array(cam.D, float)
        m.calibration_check = {"mode": "camera_digest", "camera": self.name, "digest": dig}
        m.prior = PriorState(first_pose.x, first_pose.y, first_pose.yaw, first_pose.timestamp_ns)

    def close(self) -> None:
        self.m.close()

    def step(self, bundle: FrameBundle, frame) -> tuple:
        """-> (StepResult | None, skip reason | None, ms)."""
        if not self.enabled or frame is None or not frame.available:
            return None, "disabled" if not self.enabled else "no_image", 0.0
        t0 = time.perf_counter()
        try:
            r = self.m.step(dataclasses.replace(bundle, frames={"fl": frame}))
        except ContractError:
            raise
        except Exception as e:   # noqa: BLE001  a camera failure is labeled and costs that camera's fix only
            return None, f"error:{type(e).__name__}: {str(e)[:120]}", (time.perf_counter() - t0) * 1e3
        return r, None, (time.perf_counter() - t0) * 1e3


def _worker_main(conn, name, map_dir, ext_dir, base):
    """One process per extra camera (own CUDA context, own models, no shared Python state): receives the frame bytes of a tick, answers with the E04 StepResult."""
    try:
        cm = _Cam(name, map_dir, ext_dir, base)
        conn.send(("ready", cm.m.map.map_id))
    except BaseException as e:   # noqa: BLE001
        conn.send(("fail", f"{type(e).__name__}: {e}"))
        return
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        op = msg[0]
        if op == "close":
            cm.m.close()
            conn.send(("closed",))
            return
        if op == "init":
            _, fp, cam, digest = msg
            cm.initialize_cam(fp, cam, digest)
            conn.send(("init", cm.enabled, cm.reason))
        elif op == "step":
            _, t, i, stamp, age, status, jpeg, cutoff = msg
            frame = CameraFrame(camera=name, status=status, stamp_ns=stamp, age_ns=age, _loader=lambda b=jpeg: b)
            bundle = FrameBundle(timestamp_ns=t, tick_index=i, frames={"fl": frame}, stale_cutoff_ns=cutoff)
            try:
                r, skip, ms = cm.step(bundle, frame)
            except BaseException as e:   # noqa: BLE001
                conn.send(("res", None, f"error:{type(e).__name__}: {str(e)[:120]}", 0.0))
                continue
            conn.send(("res", r, skip, ms))


class _ProcCam:
    """Parent-side handle of a worker process."""

    def __init__(self, ctx, name, map_dir, ext_dir, base):
        self.name = name
        self.parent, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker_main, args=(child, name, map_dir, ext_dir, base), daemon=True, name=f"mc7-{name}")
        self.proc.start()
        child.close()
        self.enabled, self.reason, self.map_id, self.dead = True, None, None, False

    def wait_ready(self):
        msg = self.parent.recv()
        if msg[0] != "ready":
            raise ContractError(f"MC7 worker {self.name} failed to start: {msg[1]}")
        self.map_id = msg[1]

    def initialize(self, fp, cam, digest):
        self.parent.send(("init", fp, cam, digest))

    def init_done(self):
        _, self.enabled, self.reason = self.parent.recv()

    def submit(self, bundle, frame):
        self.parent.send(("step", int(bundle.timestamp_ns), int(bundle.tick_index), frame.stamp_ns, frame.age_ns, frame.status, frame.jpeg(), bundle.stale_cutoff_ns))

    def result(self):
        try:
            _, r, skip, ms = self.parent.recv()
        except (EOFError, OSError):
            self.dead, self.enabled, self.reason = True, False, "worker process died"
            return None, "error:worker died", 0.0
        return r, skip, ms

    def close(self):
        try:
            self.parent.send(("close",))
            self.parent.recv()
        except Exception:   # noqa: BLE001
            pass
        self.proc.join(timeout=10)
        if self.proc.is_alive():
            self.proc.terminate()


class MC7FbModel(E04TFbModel):
    """config: class camloc.models.mc7:MC7FbModel; params as E04TFbModel plus `mc7` {cameras, stride, conf_D_m, escape_s, parallel, kappa, camera_digests} and `expected_mc7_sha256`;
    assets as E04TFbModel plus map_dir_<camera> for every extra camera. `parallel`: "process" (default; one worker process per camera), "thread" or "serial" (identical outputs)."""

    def __init__(self, params: dict | None = None, assets: dict | None = None):
        params = dict(params or {})
        assets = dict(assets or {})
        mc = dict(params.pop("mc7", None) or {})
        pin = params.pop("expected_mc7_sha256", None)
        for k in ("cameras", "stride", "conf_D_m", "escape_s", "kappa"):
            if k not in mc:
                raise ContractError(f"MC7 requires params.mc7.{k}")
        cams = tuple(mc["cameras"])
        if not set(cams) <= set(EXTRA_CAMERAS) or len(set(cams)) != len(cams):
            raise ContractError(f"mc7.cameras must be distinct and within {EXTRA_CAMERAS}, got {cams}")
        cams = tuple(c for c in EXTRA_CAMERAS if c in cams)       # canonical order = the order the offline builder uses
        mc["cameras"] = list(cams)
        if int(mc["stride"]) < 1:
            raise ContractError("mc7.stride must be >= 1")
        miss = [c for c in cams if c not in mc["kappa"]]
        if miss:
            raise ContractError(f"mc7.kappa lacks cameras {miss}")
        if not pin:
            raise ContractError("MC7 requires params.expected_mc7_sha256")
        got = mc7_param_sha256(mc)
        if got != pin:
            raise ContractError(f"MC7 parameters differ from the pinned hash: {got} != {pin}")
        par = mc.get("parallel", "process")
        par = {True: "process", False: "serial"}.get(par, par)
        if par not in ("process", "thread", "serial"):
            raise ContractError(f"mc7.parallel must be process | thread | serial, got {par!r}")
        self.mc7_sha256, self.mc, self.cams_extra, self.stride, self.parallel = got, mc, cams, int(mc["stride"]), par
        map_dirs = {c: assets.get(f"map_dir_{c}") for c in cams}
        for c, v in map_dirs.items():
            if v is None:
                raise ContractError(f"MC7 assets lack map_dir_{c}")
        ext = assets.get("ext_dir")
        super().__init__(params, assets)
        base = {k: v for k, v in self.e04_params.items() if k in DEFAULTS}
        self.cams, map_ids = self._make_cams(map_dirs, ext, base)
        c0 = self.capabilities
        self.capabilities = Capabilities(cameras=tuple(dict.fromkeys(tuple(c0.cameras) + cams)), clock_camera=c0.clock_camera, calibration=c0.calibration,
                                         map_id="+".join([str(c0.map_id)] + map_ids), telemetry_fields=c0.telemetry_fields,
                                         device=c0.device, assets=tuple(c0.assets) + tuple(str(v) for v in map_dirs.values()), scale_status="metric",
                                         online_mapping="disabled", telemetry=c0.telemetry)
        self.keyer = LiveFixKeyer(mc["kappa"], float(mc["conf_D_m"]))
        self._i = 0
        self._pool = None
        self._handles: dict = {}
        self._orig_tick = None
        self._last_mc: dict = {}
        self.n_extra_fix = {c: 0 for c in cams}
        self.n_extra_runs = {c: 0 for c in cams}
        self.service_ms = {c: [] for c in cams}

    def _make_cams(self, map_dirs: dict, ext, base: dict) -> tuple:
        """-> ({camera: handle}, [map ids]); a seam for tests (fake cameras)."""
        if self.parallel == "process":
            ctx = mp.get_context("spawn")
            cams = {c: _ProcCam(ctx, c, map_dirs[c], ext, base) for c in self.cams_extra}
            for cm in cams.values():
                cm.wait_ready()
            return cams, [cm.map_id for cm in cams.values()]
        cams = {c: _Cam(c, map_dirs[c], ext, base) for c in self.cams_extra}
        return cams, [cm.m.map.map_id for cm in cams.values()]

    # ------------------------------------------------------------------ protocol
    def initialize(self, first_pose: FirstPose, static_metadata: StaticMetadata) -> None:
        super().initialize(first_pose, static_metadata)
        digests = self.mc.get("camera_digests") or {}
        cams = dict(static_metadata.cameras)
        if self.parallel == "process":
            for c, cm in self.cams.items():
                cm.initialize(first_pose, cams.get(c), digests.get(c))
            for cm in self.cams.values():
                cm.init_done()
        else:
            for c, cm in self.cams.items():
                cm.initialize(first_pose, static_metadata, digests.get(c))
        for c, cm in self.cams.items():
            if not cm.enabled:
                print(f"WARNING: MC7 camera {c} disabled ({cm.reason}); the other cameras keep running", file=sys.stderr)
        ts = self._ts
        ts.filter = MultiCamTel4Filter(ts.filter.p, self._fp, self._t0, ts.filter.rp, self.keyer.tags, require_confirm=True, escape_s=float(self.mc["escape_s"]))
        self._orig_tick = ts.tick
        ts.tick = self._tick_hook
        if self.parallel == "thread":
            self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(self.cams)), thread_name_prefix="mc7")

    def _submit(self, c: str, bundle: FrameBundle):
        """Start camera c on this tick's bundle (process: send the frame bytes; thread: submit; serial: nothing, it runs at collection time)."""
        cm = self.cams[c]
        f = bundle.frames.get(c)
        if self.parallel == "process":
            if cm.enabled and f is not None and f.available:
                cm.submit(bundle, f)
                return "sent"
            return None
        if self.parallel == "thread":
            return self._pool.submit(cm.step, bundle, f)
        return None

    def _collect(self, c: str, handle, bundle: FrameBundle):
        """-> (StepResult | None, skip reason | None, ms, frame)."""
        cm = self.cams[c]
        f = bundle.frames.get(c)
        if self.parallel == "process":
            if handle is None:
                return None, ("disabled" if not cm.enabled else "no_image"), 0.0, f
            r, skip, ms = cm.result()
        elif self.parallel == "thread":
            r, skip, ms = handle.result()
        else:
            r, skip, ms = cm.step(bundle, f)
        return r, skip, ms, f

    def step(self, bundle: FrameBundle) -> StepResult:
        t = int(bundle.timestamp_ns)
        run_extra = (self._i % self.stride == 0)
        self._i += 1
        self.keyer.start_tick(t)
        self._handles = {}
        self._tick_bundle = bundle
        self._run_extra = run_extra
        if run_extra and self.parallel != "serial":
            self._handles = {c: self._submit(c, bundle) for c in self.cams}
        res = super().step(bundle)
        mc = self._last_mc
        flags = res.source_flags + (("mc7_extra",) if mc.get("n_fix") else ())
        used = mc.pop("_stamps", [])
        cmin = [v for v in ([res.consumed_min_ns] + used) if v is not None]
        cmax = [v for v in ([res.consumed_max_ns] + used) if v is not None]
        return dataclasses.replace(res, source_flags=flags, consumed_min_ns=min(cmin) if cmin else None, consumed_max_ns=max(cmax) if cmax else None,
                                   residuals={**res.residuals, "mc7": mc})

    def _tick_hook(self, t: int) -> dict:
        ts = self._ts
        if len(ts.e04) != 1:
            raise ContractError(f"MC7 expected exactly the fl row at the tick, found {len(ts.e04)}")
        fl_row = self.keyer.add_fl(ts.e04[0])
        bundle = self._tick_bundle
        outs = []
        t_join0 = time.perf_counter()
        if self._run_extra:
            outs = [(c,) + self._collect(c, self._handles.get(c), bundle) for c in self.cams]
        t_join = time.perf_counter() - t_join0
        extra_rows, fixes, stamps, cam_ms, cam_t = [], [], [], {}, {}
        for c, r, skip, ms, f in outs:
            cam_ms[c] = round(ms, 1)
            if r is not None and r.residuals.get("t_ms"):
                cam_t[c] = r.residuals["t_ms"]
            if skip is None:
                self.n_extra_runs[c] += 1
                self.service_ms[c].append(ms)
            if r is None:
                continue
            stamps.append(int(f.stamp_ns))
            if "pnp_fix" in r.source_flags:
                self.n_extra_fix[c] += 1
                row = self.keyer.add_fix(c, int(f.stamp_ns), r.x, r.y, r.yaw, r.residuals)
                fixes.append({"cam": c, "img_ns": int(f.stamp_ns), "key_ns": None if row is None else int(row["t_ns"]), "x": r.x, "y": r.y, "yaw": r.yaw,
                               "n_inl": r.residuals.get("n_inl"), "ms": cam_ms[c]})
                if row is not None:
                    extra_rows.append(row)
        rows = order_rows([fl_row] + extra_rows)
        ts.e04[:] = rows
        ts.kin.e04[:] = list(rows)
        self._last_mc = {"ran": bool(self._run_extra), "cam_ms": cam_ms, "cam_t_ms": cam_t, "join_wait_ms": round(t_join * 1e3, 1), "n_fix": len(extra_rows), "fixes": fixes, "_stamps": stamps,
                         "fl_row": {k: fl_row.get(k) for k in ("src", "x_m", "y_m", "yaw_rad", "n_inl")},
                         "dropped_no_tick": self.keyer.info["dropped_no_tick"]}
        return self._orig_tick(t)

    def telemetry_info(self) -> dict:
        return {**super().telemetry_info(), "mc7_sha256": self.mc7_sha256, "mc7_extra_fix": dict(self.n_extra_fix), "mc7_extra_runs": dict(self.n_extra_runs),
                "mc7_keyer": dict(self.keyer.info), "mc7_disabled": {c: cm.reason for c, cm in self.cams.items() if not cm.enabled}}

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        for cm in self.cams.values():
            cm.close()
        super().close()
