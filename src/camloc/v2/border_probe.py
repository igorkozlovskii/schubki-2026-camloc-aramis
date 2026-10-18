"""S1.1 border measurement probe: non-learned (n, dpsi) measurements from the supplied 3D borders, three methods.

`measure_border` is the inference-clean entry point. Its inputs are the half-resolution BGR image of one camera, an initial
pose (x, y, yaw), the calibration (+ fitted correction), the supplied borders (`BorderMap`, `Centerline` = track-v1) and the
usable-pixel mask. It never touches ground truth, `app/camloc.py::load_all()`, `refline_*` or `sn_map_state` (a test greps this
module's source and imports). The probe harness (scripts/v2_s11_*.py) supplies the initial pose as GT perturbed by draws from
`PERTURBATION` and owns every GT read.

Methods (all return track coordinates n (m, + left), dpsi (rad) of the *corrected* pose, plus quality fields):
  m1_normal  E02-style normal search of the projected borders (masks, fitted calibration), robust Gauss-Newton on (dn, dpsi, pitch).
  m2_region  road-region boundary: Mahalanobis colour/texture statistics of the predicted road area (inside strips of the
             borders at the initial pose) against strips just outside the borders; grid search over (dn, dpsi, pitch).
  m3_chamfer truncated chamfer distance of the projected borders to Canny edges; same grid search.
Coordinates: pose hypotheses shift (x, y) along the centreline normal by dn and rotate yaw by dpsi at fixed other states;
pitch is a bounded nuisance, roll and z come from the borders (`BorderMap.road_attitude`).
"""
from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from camloc.geometry.borders import BASE_BELOW_ROAD_M, BorderMap
from camloc.geometry.camera import CameraModel
from camloc.geometry.centerline import Centerline
from camloc.geometry.residuals import build_edge_image, image_tangents, normal_search, thin_in_image
from camloc.geometry.transforms import Transform, wrap_angle

METHODS = ("m1_normal", "m2_region", "m3_chamfer")
REDUCE = 2
# declared initial-pose perturbation of the probe (uniform, independent), seed per (lap, camera) in the harness
PERTURBATION = {"dn_m": 1.0, "dpsi_deg": 1.5, "ds_m": 3.0, "dist": "uniform", "init": "gt_perturbed"}

MEAS_COLUMNS = ["stamp_ns", "avail_ns", "lap", "camera", "kind", "valid", "s", "n", "dpsi", "x", "y", "yaw", "sig_s", "sig_n",
                "sig_dpsi", "sig_xy", "sig_yaw", "init", "q_reason", "q_method", "q_img_stamp_ns", "q_service_ms", "q_cost",
                "q_contrast", "q_support", "q_n_points", "q_dn_init", "q_dpsi_init", "q_sig_raw_n", "q_sig_raw_dpsi", "q_fold"]


# ------------------------------------------------------------------------------------------------ calibration correction
def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def correct_camera(cm: CameraModel, dyaw_rad: float = 0.0, dpitch_rad: float = 0.0, droll_rad: float = 0.0) -> CameraModel:
    """Camera whose mount is rotated in base_link: R_bc' = Rz(dyaw) Ry(dpitch) Rx(droll) R_bc (translation unchanged).
    Positive dyaw turns the camera left, positive dpitch tilts it nose-down (road convention of `geometry.camera`)."""
    if dyaw_rad == 0.0 and dpitch_rad == 0.0 and droll_rad == 0.0:
        return cm
    m = cm.T_base_cam.matrix.copy()
    m[:3, :3] = _rz(dyaw_rad) @ _ry(dpitch_rad) @ _rx(droll_rad) @ m[:3, :3]
    return dataclasses.replace(cm, T_base_cam=Transform(m, cm.T_base_cam.target, cm.T_base_cam.source))


# ------------------------------------------------------------------------------------------------ context and results
@dataclass
class ProbeConfig:
    max_range_m: float = 60.0
    min_depth_m: float = 3.0
    max_depth_m: float = 40.0
    thin: int = 2                       # keep every 2nd 0.5 m border vertex (1 m spacing)
    trunc_px: float = 10.0              # truncated chamfer, half-resolution pixels
    out_penalty: float = 0.5            # share of TRUNC charged to border points that leave view / enter the mask under a hypothesis
    min_points: int = 20
    # grid search ranges around the initial pose (the probe's perturbation is |dn| <= 1 m, |dpsi| <= 1.5 deg)
    dn_range_m: float = 1.5
    dpsi_range_deg: float = 2.5
    dpitch_deg: tuple = (0.0,)       # pitch is not searched: it is degenerate with dn for near-horizontal borders (see REPORT)
    coarse_dn: float = 0.25
    coarse_dpsi_deg: float = 0.5
    fine_dn: float = 0.05
    fine_dpsi_deg: float = 0.1
    min_contrast: float = 0.05          # (median coarse cost - best cost) / reference scale; below: abstain
    # m1
    m1_band_px: float = 40.0
    m1_g_min: float = 2.0
    m1_sigma_px: float = 4.0
    m1_min_found: int = 12
    m1_min_support: float = 0.2
    m1_pitch_prior_deg: float = 0.25
    # m2
    m2_in_offsets: tuple = (0.5, 1.0, 1.5)
    m2_out_offsets: tuple = (0.4, 0.9, 1.4)
    m2_model_offsets: tuple = (0.5, 1.0, 1.5, 2.0, 2.5)
    m2_trunc: float = 4.0


@dataclass
class ProbeContext:
    bmap: BorderMap
    cline: Centerline
    cams: dict                           # camera -> CameraModel at full resolution, correction already applied
    masks: dict                          # camera -> bool (H/2, W/2), True = usable
    cfg: ProbeConfig = field(default_factory=ProbeConfig)


@dataclass
class BorderMeas:
    valid: bool
    reason: str = ""
    n: float = float("nan")
    dpsi: float = float("nan")
    s: float = float("nan")
    x: float = float("nan")
    y: float = float("nan")
    yaw: float = float("nan")
    q: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------ scene geometry
class _Scene:
    def __init__(self, ctx: ProbeContext, cam: str, pose0, s_hint=None, attitude=None):
        self.ctx, self.cam = ctx, cam
        self.cm = ctx.cams[cam]
        x, y, yaw = pose0
        self.pose0 = (float(x), float(y), float(yaw))
        s, n, th, _ = ctx.cline.project(x, y, s_hint=s_hint)
        self.s0, self.n0, self.theta = s, n, th
        self.nrm = np.array([-math.sin(th), math.cos(th)])
        bm = ctx.bmap
        rs = bm.road_height("main", (x, y), None)
        self.z_road = rs.z_road
        self.z_base = rs.z_road - BASE_BELOW_ROAD_M
        self.pitch0, self.roll0 = bm.road_attitude("main", x, y, yaw, rs.z_road)
        if attitude is not None:             # S1.8 additive override; None keeps the S1.1 behaviour bit for bit
            self.z_base, self.pitch0, self.roll0 = attitude.apply(self.z_base, self.pitch0, self.roll0)
        self.runs = bm.candidates("main", x, y, yaw, rs.z_road, max_range=ctx.cfg.max_range_m, dz_gate=1.5, thin=ctx.cfg.thin)

    def pose(self, dn, dpsi):
        x, y, yaw = self.pose0
        return x + dn * self.nrm[0], y + dn * self.nrm[1], yaw + dpsi

    def project(self, P, dn, dpsi, dpitch=0.0, jac=False):
        x, y, yaw = self.pose(dn, dpsi)
        return self.cm.project_points_pose(P, x, y, yaw, self.z_base, self.pitch0 + dpitch, self.roll0, with_jac=jac)


def _usable_at(uv, usable, valid):
    h, w = usable.shape
    xi = np.clip((uv[:, 0] / REDUCE).astype(int), 0, w - 1)
    yi = np.clip((uv[:, 1] / REDUCE).astype(int), 0, h - 1)
    return valid & usable[yi, xi]


# ------------------------------------------------------------------------------------------------ grid search machinery
def _grid_search(cost_fn, cfg: ProbeConfig):
    """Coarse grid (dn, dpsi, pitch) then a fine grid (dn, dpsi) around the coarse best. Returns best, cost, coarse array, info."""
    dns = np.arange(-cfg.dn_range_m, cfg.dn_range_m + 1e-9, cfg.coarse_dn)
    dps = np.deg2rad(np.arange(-cfg.dpsi_range_deg, cfg.dpsi_range_deg + 1e-9, cfg.coarse_dpsi_deg))
    pits = np.deg2rad(np.asarray(cfg.dpitch_deg))
    C = np.full((len(dns), len(dps), len(pits)), np.inf)
    for i, dn in enumerate(dns):
        for j, dp in enumerate(dps):
            for k, pt in enumerate(pits):
                C[i, j, k] = cost_fn(dn, dp, pt)
    i, j, k = np.unravel_index(int(np.argmin(C)), C.shape)
    dn0, dp0, pt0 = dns[i], dps[j], pits[k]
    fdn = np.arange(dn0 - cfg.coarse_dn, dn0 + cfg.coarse_dn + 1e-9, cfg.fine_dn)
    fdp = np.deg2rad(np.arange(np.rad2deg(dp0) - cfg.coarse_dpsi_deg, np.rad2deg(dp0) + cfg.coarse_dpsi_deg + 1e-9, cfg.fine_dpsi_deg))
    F = np.full((len(fdn), len(fdp)), np.inf)
    for a, dn in enumerate(fdn):
        for b, dp in enumerate(fdp):
            F[a, b] = cost_fn(dn, dp, pt0)
    a, b = np.unravel_index(int(np.argmin(F)), F.shape)
    best = (float(fdn[a]), float(fdp[b]), float(pt0))
    # curvature-based raw sigma from the fine grid: cost rises by `scale` at the width where the surface is 1 unit above its minimum
    return best, float(F[a, b]), C, {"i": int(i), "j": int(j), "shape": C.shape, "fine": F, "fdn": fdn, "fdp": fdp, "a": int(a), "b": int(b)}


def _contrast_and_edge(C, cost_best, info, cfg):
    finite = C[np.isfinite(C)]
    if len(finite) == 0:
        return 0.0, True
    contrast = float(np.median(finite) - cost_best)
    i, j = info["i"], info["j"]
    edge = i in (0, C.shape[0] - 1) or j in (0, C.shape[1] - 1)
    return contrast, bool(edge)


def _chamfer_cost_fn(sc: _Scene, P, dt, usable, cfg: ProbeConfig):
    tr = cfg.trunc_px
    h, w = dt.shape
    npts = len(P)

    def cost(dn, dp, pt):
        uv, ok, depth, _ = sc.project(P, dn, dp, pt)
        ok = ok & (depth > cfg.min_depth_m) & (depth < cfg.max_depth_m)
        ok = _usable_at(uv, usable, ok)
        if not ok.any():
            return tr
        xi = np.clip((uv[ok, 0] / REDUCE).astype(int), 0, w - 1)
        yi = np.clip((uv[ok, 1] / REDUCE).astype(int), 0, h - 1)
        d = np.minimum(dt[yi, xi], tr)
        return float((d.sum() + cfg.out_penalty * tr * (npts - ok.sum())) / npts)

    return cost


def chamfer_points(sc: _Scene, usable, cfg: ProbeConfig):
    """Border sample points (3D) visible, unmasked and in the depth range at the *initial* pose."""
    if len(sc.runs) == 0:
        return np.zeros((0, 3))
    uv, ok, depth, _ = sc.project(sc.runs.xyz, 0.0, 0.0, 0.0)
    ok = ok & (depth > cfg.min_depth_m) & (depth < cfg.max_depth_m)
    ok = _usable_at(uv, usable, ok)
    return sc.runs.xyz[ok]


def _edge_image(bgr, usable):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return build_edge_image(gray, reduce=REDUCE, usable_mask=usable, with_dt=True)


# ------------------------------------------------------------------------------------------------ method 3: chamfer
def _m3(sc, bgr, usable, cfg):
    P = chamfer_points(sc, usable, cfg)
    if len(P) < cfg.min_points:
        return None, "few_points", {"n_points": len(P)}
    ei = _edge_image(bgr, usable)
    if ei.dt is None:
        return None, "no_edges", {}
    cost_fn = _chamfer_cost_fn(sc, P, ei.dt, usable, cfg)
    best, c_best, C, info = _grid_search(cost_fn, cfg)
    contrast, edge = _contrast_and_edge(C, c_best, info, cfg)
    q = {"cost": c_best, "contrast": contrast / cfg.trunc_px, "n_points": len(P), "support": float(np.mean(np.isfinite(C)))}
    if edge:
        return None, "grid_edge", q
    if q["contrast"] < cfg.min_contrast:
        return None, "low_contrast", q
    dn, dp, _ = best
    q.update(_grid_sigma(info))
    return (dn, dp), "", q


def _grid_sigma(info):
    """Raw sigma (m, rad) from the width of the fine cost surface: extent where cost <= min + 0.5 * (max - min of the fine grid)."""
    F, fdn, fdp = info["fine"], info["fdn"], info["fdp"]
    lo, hi = float(F.min()), float(F.max())
    thr = lo + 0.5 * (hi - lo)
    a, b = info["a"], info["b"]
    wn = float(np.ptp(fdn[(F[:, b] <= thr)])) if (F[:, b] <= thr).any() else 0.0
    wp = float(np.ptp(fdp[(F[a, :] <= thr)])) if (F[a, :] <= thr).any() else 0.0
    return {"sig_raw_n": 0.5 * wn, "sig_raw_dpsi": 0.5 * wp}


# ------------------------------------------------------------------------------------------------ method 2: road region statistics
def _lab_features(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), 2.0)
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    tex = cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 3.0)
    return np.dstack([lab, tex])


def _strip_samples(sc: _Scene, offsets, sign):
    """3D points at ground offsets from the supplied border points; sign +1 = towards the road centre, -1 = outwards."""
    R = sc.runs
    xyz = R.xyz
    cl = sc.ctx.cline
    _, ci = cl.tree.query(xyz[:, :2])
    v = cl.xyz[ci, :2] - xyz[:, :2]
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
    out = []
    for o in offsets:
        p = xyz.copy()
        p[:, :2] += sign * o * v
        out.append(p)
    return np.stack(out, 1)               # (N, K, 3)


def _m2(sc, bgr, usable, cfg):
    if len(sc.runs) < cfg.min_points:
        return None, "few_points", {"n_points": len(sc.runs)}
    F = _lab_features(bgr)
    h, w = usable.shape
    Pin = _strip_samples(sc, cfg.m2_in_offsets, +1)
    Pout = _strip_samples(sc, cfg.m2_out_offsets, -1)
    Pmod = _strip_samples(sc, cfg.m2_model_offsets, +1)
    N, Ki, Ko = len(Pin), Pin.shape[1], Pout.shape[1]

    def sample(P, dn, dp, pt):
        uv, ok, depth, _ = sc.project(P.reshape(-1, 3), dn, dp, pt)
        ok = ok & (depth > cfg.min_depth_m) & (depth < cfg.max_depth_m)
        ok = _usable_at(uv, usable, ok)
        xi = np.clip((uv[:, 0] / REDUCE).astype(int), 0, w - 1)
        yi = np.clip((uv[:, 1] / REDUCE).astype(int), 0, h - 1)
        return F[yi, xi], ok

    fm, okm = sample(Pmod, 0.0, 0.0, 0.0)
    if okm.sum() < 30:
        return None, "few_points", {"n_points": int(okm.sum())}
    med = np.median(fm[okm], axis=0)
    mad = 1.4826 * np.median(np.abs(fm[okm] - med), axis=0)
    sd = np.maximum(mad, np.array([4.0, 2.0, 2.0, 1.5], np.float32))

    def mah(f):
        return np.minimum(np.sqrt((((f - med) / sd) ** 2).sum(-1)), cfg.m2_trunc)

    def cost(dn, dp, pt):
        fi, oi = sample(Pin, dn, dp, pt)
        fo, oo = sample(Pout, dn, dp, pt)
        di = np.where(oi, mah(fi), np.nan).reshape(N, Ki)
        do = np.where(oo, mah(fo), np.nan).reshape(N, Ko)
        # pair every border point's outside samples with its inside samples
        a = np.nanmean(di, axis=1) if np.isfinite(di).any() else np.full(N, np.nan)
        b = np.nanmean(do, axis=1) if np.isfinite(do).any() else np.full(N, np.nan)
        good = np.isfinite(a) & np.isfinite(b)
        if good.sum() < cfg.min_points:
            return 0.0
        return float(-(b[good] - a[good]).mean())

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        best, c_best, C, info = _grid_search(cost, cfg)
    contrast, edge = _contrast_and_edge(C, c_best, info, cfg)
    q = {"cost": c_best, "contrast": contrast / cfg.m2_trunc, "n_points": int(N), "support": float(np.mean(np.isfinite(C)))}
    if edge:
        return None, "grid_edge", q
    if q["contrast"] < cfg.min_contrast:
        return None, "low_contrast", q
    q.update(_grid_sigma(info))
    return (best[0], best[1]), "", q


# ------------------------------------------------------------------------------------------------ method 1: normal search + GN
def _m1(sc, bgr, usable, cfg):
    if len(sc.runs) < cfg.min_points:
        return None, "few_points", {"n_points": len(sc.runs)}
    ei = _edge_image(bgr, usable)
    th = sc.theta
    nx, ny = -math.sin(th), math.cos(th)
    dn, dp, pt = 0.0, 0.0, 0.0
    pri = np.diag([1 / 1.5 ** 2, 1 / math.radians(2.5) ** 2, 1 / math.radians(cfg.m1_pitch_prior_deg) ** 2])
    q = {}
    for it in range(3):
        uv, ok, depth, J = sc.project(sc.runs.xyz, dn, dp, pt, jac=True)
        ok = ok & (depth > cfg.min_depth_m) & (depth < cfg.max_depth_m)
        ok = _usable_at(uv, usable, ok)
        ii = np.where(ok)[0]
        if len(ii) < 4:
            return None, "few_points", {"n_points": int(len(ii))}
        tau, tok = image_tangents(uv[ii], sc.runs.run_id[ii])
        ii2 = ii[tok]
        keep = thin_in_image(uv[ii2], sc.runs.run_id[ii2], 8.0)
        sel = ii2[keep]
        tau = tau[tok][keep]
        if len(sel) < 4:
            return None, "few_points", {"n_points": int(len(sel))}
        R = normal_search(ei, uv[sel], J[sel], tau, band_px=cfg.m1_band_px, g_min=cfg.m1_g_min)
        q = {"n_points": int(len(sel)), "support": float(R.support), "n_found": int(R.n_found)}
        if R.n_found < cfg.m1_min_found or R.support < cfg.m1_min_support:
            return None, "low_support", q
        Jn = np.stack([nx * R.jn[:, 0] + ny * R.jn[:, 1], R.jn[:, 2], R.jn[:, 3]], 1)
        # (the Jacobian columns of normal_search are w.r.t. (x, y, yaw, pitch); dn moves (x, y) along the centreline normal)
        d = np.zeros(3)
        for _ in range(4):
            res = R.r0 + Jn @ d
            wgt = 1.0 / (1.0 + (res / cfg.m1_sigma_px) ** 2) * R.weight
            H = (Jn * wgt[:, None]).T @ Jn / cfg.m1_sigma_px ** 2 + pri
            g = (Jn * wgt[:, None]).T @ res / cfg.m1_sigma_px ** 2 + pri @ (np.array([dn, dp, pt]) + d)
            d = d - np.linalg.solve(H, g)
        dn, dp, pt = dn + d[0], dp + d[1], pt + d[2]
        dn, dp, pt = float(np.clip(dn, -cfg.dn_range_m, cfg.dn_range_m)), float(np.clip(dp, -math.radians(cfg.dpsi_range_deg), math.radians(cfg.dpsi_range_deg))), float(np.clip(pt, -math.radians(1.5), math.radians(1.5)))
    cov = np.linalg.inv(H)
    res = R.r0 + Jn @ d
    q.update({"cost": float(np.mean(np.log1p((res / cfg.m1_sigma_px) ** 2))), "contrast": float(R.support),
              "sig_raw_n": float(math.sqrt(max(cov[0, 0], 0))), "sig_raw_dpsi": float(math.sqrt(max(cov[1, 1], 0)))})
    if abs(dn) >= cfg.dn_range_m - 1e-6 or abs(dp) >= math.radians(cfg.dpsi_range_deg) - 1e-9:
        return None, "grid_edge", q
    return (dn, dp), "", q


_METHOD_FN = {"m1_normal": _m1, "m2_region": _m2, "m3_chamfer": _m3}


# ------------------------------------------------------------------------------------------------ inference-clean entry point
def measure_border(method: str, bgr_half: np.ndarray, cam: str, pose0, ctx: ProbeContext, s_hint=None) -> BorderMeas:
    """One border measurement from one camera image and an initial pose (x, y, yaw). No GT is read here."""
    return _measure(method, bgr_half, cam, pose0, ctx, s_hint, None)


def measure_border_att(method: str, bgr_half: np.ndarray, cam: str, pose0, ctx: ProbeContext, s_hint=None, attitude=None) -> BorderMeas:
    """S1.8: `measure_border` with the car's attitude overridden or shifted (`camloc.v2.border_attitude.Attitude`: z / pitch / roll that
    S1.1 took from the borders); attitude=None is identical to `measure_border`."""
    return _measure(method, bgr_half, cam, pose0, ctx, s_hint, attitude)


def _measure(method, bgr_half, cam, pose0, ctx, s_hint, attitude) -> BorderMeas:
    t0 = time.perf_counter()
    cfg = ctx.cfg
    usable = ctx.masks[cam]
    try:
        sc = _Scene(ctx, cam, pose0, s_hint, attitude)
        if len(sc.runs) == 0:
            return BorderMeas(False, "no_border_near", q={"service_ms": 1e3 * (time.perf_counter() - t0)})
        res, reason, q = _METHOD_FN[method](sc, bgr_half, usable, cfg)
    except Exception as e:  # abstain, never crash a stream
        return BorderMeas(False, f"error:{type(e).__name__}", q={"service_ms": 1e3 * (time.perf_counter() - t0)})
    q = dict(q)
    q["service_ms"] = 1e3 * (time.perf_counter() - t0)
    if res is None:
        return BorderMeas(False, reason, q=q)
    dn, dp = res
    x, y, yaw = sc.pose(dn, dp)
    s, n, theta, _ = ctx.cline.project(x, y, s_hint=sc.s0)
    return BorderMeas(True, "", n=float(n), dpsi=wrap_angle(yaw - theta), s=float(s), x=float(x), y=float(y), yaw=float(wrap_angle(yaw)), q=q)


# ------------------------------------------------------------------------------------------------ meas-v1 rows and metrics
def meas_row(stamp_ns, lap, camera, method, m: BorderMeas, img_stamp_ns, dn_init, dpsi_init, fold, latency_scale=1.0) -> dict:
    nan = float("nan")
    svc = m.q.get("service_ms", nan)
    return {"stamp_ns": int(stamp_ns), "avail_ns": int(stamp_ns + round(svc * 1e6 * latency_scale)) if svc == svc else int(stamp_ns),
            "lap": int(lap), "camera": camera, "kind": "border", "valid": bool(m.valid), "s": m.s, "n": m.n, "dpsi": m.dpsi,
            "x": m.x, "y": m.y, "yaw": m.yaw, "sig_s": nan, "sig_n": nan, "sig_dpsi": nan, "sig_xy": nan, "sig_yaw": nan,
            "init": "gt_perturbed", "q_reason": m.reason, "q_method": method, "q_img_stamp_ns": int(img_stamp_ns),
            "q_service_ms": svc, "q_cost": m.q.get("cost", nan), "q_contrast": m.q.get("contrast", nan),
            "q_support": m.q.get("support", nan), "q_n_points": float(m.q.get("n_points", nan)),
            "q_dn_init": dn_init, "q_dpsi_init": dpsi_init, "q_sig_raw_n": m.q.get("sig_raw_n", nan),
            "q_sig_raw_dpsi": m.q.get("sig_raw_dpsi", nan), "q_fold": str(fold)}


def gate1_metrics(df, sig_col_n="sig_n", sig_col_d="sig_dpsi") -> dict:
    """Gate 1 border metrics over rows of the *eligible* ticks (moving, GT sigma <= 1 m ...). `df` has the stream columns joined
    with err_n (m), err_dpsi (rad). availability = valid / eligible; the others are over valid rows."""
    nE = len(df)
    v = df[df.valid]
    out = {"n_eligible": int(nE), "n_valid": int(len(v)), "availability": float(len(v) / nE) if nE else float("nan")}
    if len(v) == 0:
        return out | {k: float("nan") for k in ("n_err_median", "n_err_p95", "gross_rate", "sig_coverage", "dpsi_err_median_deg")}
    e = v.err_n.abs()
    out["n_err_median"] = float(e.median())
    out["n_err_p95"] = float(e.quantile(0.95))
    out["gross_rate"] = float((e > 1.0).mean())
    sg = v[sig_col_n]
    out["sig_coverage"] = float((e <= 2 * sg).mean()) if sg.notna().all() else float("nan")
    out["dpsi_err_median_deg"] = float(np.rad2deg(v.err_dpsi.abs().median()))
    return out


def gate1_pass(m: dict) -> dict:
    """The PLAN.md section 5 rule. Returns per-criterion booleans, 'border_n' and 'border_dpsi' verdicts."""
    ok = {"availability": m["availability"] >= 0.5, "n_median": m["n_err_median"] <= 0.25, "n_p95": m["n_err_p95"] <= 0.75,
          "gross": m["gross_rate"] <= 0.01, "sigma": 0.85 <= m["sig_coverage"] <= 0.99}
    ok["border_n"] = bool(all(bool(v) for v in ok.values()))
    ok["dpsi"] = bool(ok["border_n"] and m["dpsi_err_median_deg"] <= 0.3)
    return {k: bool(v) for k, v in ok.items()}
