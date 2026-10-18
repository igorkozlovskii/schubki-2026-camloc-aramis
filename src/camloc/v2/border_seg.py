"""S3.2b: border measurement (n, dpsi) from the segmentation network's paved-surface probability.

Inputs of a measurement are the probability map(s) (float32, label grid 192 x 384, from `camloc.v2.seg.infer.SegPredictor`; this module
is torch-free), the camera name(s), an initial pose (x, y, yaw) and the initial along-track coordinate `s_init` of the *filter's own*
estimate, the calibration / masks / supplied borders (`ProbeContext`) and optionally a car attitude (`border_attitude.Attitude`).
No ground truth, no `derived/`, no `refline_*`, no `sn_map_state` and no GT s: the harness (scripts/v2_s32b_*.py) owns every GT read.

Method. The road polygon of the *map* (the paved corridor between `track_left` and `track_right`, same polygons, layer gate, far cut and
projection as the S1.4 label renderer) is projected at a hypothesis pose (shift dn along the centreline normal, rotation dpsi) and compared
with the network: the score is the Bernoulli log-likelihood of the polygon as the road label,
    S(dn, dpsi) = sum_{pixels in polygon} [log p - log(1 - p)]  (+ a hypothesis-independent constant),
with p clipped to [p_clip, 1 - p_clip] and zero evidence on car-body pixels and on pixels the label builder ignores (other layer, far).
Coarse grid over (dn, dpsi), fine grid around the coarse best, then the maximum of a local quadratic surface fit (the fill is piecewise
constant). Roll / pitch / z are the supplied-borders attitude of S1.1 unless an `Attitude` is given. The joint variant sums the score
surfaces of several cameras that share the pose hypothesis (one pose, several cameras).
Abstain reasons: no_border_near, few_pixels (polygon covers < min_poly_px label pixels), grid_edge, low_contrast, flat (not concave).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from camloc.geometry.borders import BASE_BELOW_ROAD_M
from camloc.geometry.transforms import wrap_angle
from camloc.v2.border_probe import BorderMeas, ProbeContext
from camloc.v2.seg import LABEL_H, LABEL_W
from camloc.v2.seg import labels as L   # geometry only: polygons + projection of the supplied map; nothing here reads a GT pose

SHIFT = 4
ONE = 1 << SHIFT


@dataclass
class SegFitConfig:
    dn_range_m: float = 1.5
    dpsi_range_deg: float = 2.5
    coarse_dn: float = 0.25
    coarse_dpsi_deg: float = 0.5
    fine_dn: float = 0.05
    fine_dpsi_deg: float = 0.1
    p_clip: float = 0.02
    min_poly_px: int = 400            # label pixels of road polygon (single camera); abstain below
    min_contrast: float = 0.0         # (S_best - S_median_coarse) / sum|E| over the evidence pixels
    refine_halfwidth: int = 3         # fine-grid cells around the best used for the quadratic surface fit
    ss: int = 2                       # evidence is evaluated at ss x the label grid (bilinear upsampling) so the score varies smoothly with the pose
    label: L.LabelConfig = field(default_factory=L.LabelConfig)


# ------------------------------------------------------------------------------------------------ evidence image
def evidence_image(prob: np.ndarray, ignore: np.ndarray, p_clip: float, ss: int = 1) -> np.ndarray:
    """log-odds of 'paved surface' (float32), 0 on ignored pixels. `prob` may have any size; it is resized to the label grid."""
    if prob.shape != (LABEL_H, LABEL_W):
        prob = cv2.resize(prob.astype(np.float32), (LABEL_W, LABEL_H), interpolation=cv2.INTER_LINEAR)
    p = np.clip(prob.astype(np.float32), p_clip, 1.0 - p_clip)
    e = np.log(p / (1.0 - p)).astype(np.float32)
    e[ignore] = 0.0
    if ss > 1:
        e = cv2.resize(e, (LABEL_W * ss, LABEL_H * ss), interpolation=cv2.INTER_LINEAR)
    return e


def _clip1(P: np.ndarray, normal: np.ndarray, off: float) -> np.ndarray:
    """Sutherland-Hodgman of one polygon (m, 3) against the half-space normal . p - off > 0."""
    d = P @ normal - off
    ins = d > 0
    if ins.all():
        return P
    if not ins.any():
        return P[:0]
    nx = np.roll(np.arange(len(P)), -1)
    Pn, dn, insn = P[nx], d[nx], ins[nx]
    cross = ins != insn
    den = np.where(cross, d - dn, 1.0)
    X = P + (Pn - P) * (d / den)[:, None]
    out = np.empty((2 * len(P), 3))
    keep = np.empty(2 * len(P), bool)
    out[0::2], out[1::2] = X, Pn
    keep[0::2], keep[1::2] = cross, insn
    return out[keep]


# ------------------------------------------------------------------------------------------------ scene (one camera, one pose0)
class SegScene:
    """Map road polygons around pose0, pre-clipped to the view at pose0 and held in world coordinates for fast re-projection."""

    def __init__(self, meas: "SegMeasurer", cam: str, pose0, s_init, attitude=None):
        ctx, cfg = meas.ctx, meas.cfg
        self.ss = cfg.ss
        self.cam, self.cm = cam, ctx.cams[cam]
        x, y, yaw = (float(v) for v in pose0)
        self.pose0 = (x, y, yaw)
        s, n, th, _ = ctx.cline.project(x, y, s_hint=s_init)
        self.s0, self.n0, self.theta = s, n, th
        self.nrm = np.array([-math.sin(th), math.cos(th)])
        bm = ctx.bmap
        rs = bm.road_height("main", (x, y), None)
        self.z_base = rs.z_road - BASE_BELOW_ROAD_M
        self.pitch0, self.roll0 = bm.road_attitude("main", x, y, yaw, rs.z_road)
        if attitude is not None:
            self.z_base, self.pitch0, self.roll0 = attitude.apply(self.z_base, self.pitch0, self.roll0)
        self.R0 = L.pose_rotation(self.roll0, self.pitch0, yaw)
        self.t0 = np.array([x, y, self.z_base])
        Rbc, tbc = self.cm.T_base_cam.R, self.cm.T_base_cam.t
        cam_z = float((self.t0 + self.R0 @ tbc)[2])
        self.sx, self.sy = LABEL_W / self.cm.width, LABEL_H / self.cm.height
        road, other = [], []
        lcfg = cfg.label
        for strip in meas.polys.strips:
            for verts, c in L.corridor_polygons(strip, self.t0, lcfg, cam_z):
                (road if c == 0 else other).append(verts)
        self.z_min = lcfg.z_min
        self.road_world = road               # raw corridor polygons (map frame); clipped to the view per hypothesis
        self.lim = L._view_limits(self.cm)
        self.has_road = len(road) > 0
        self.z_min = lcfg.z_min
        self.ignore = meas.body_ignore(cam).copy()
        if other:                                 # other layer / beyond far_m / pit surface: no evidence there (as in the training labels)
            canvas = np.zeros((LABEL_H, LABEL_W), np.uint8)
            qs = [self._to_label(L._project_polygon(self.cm, self.R0, self.t0, w, self.z_min), 1) for w in other]
            qs = [q for q in qs if len(q) >= 3]
            if qs:
                cv2.fillPoly(canvas, qs, 1, lineType=cv2.LINE_8, shift=SHIFT)
            self.ignore |= canvas.astype(bool)

    @property
    def V(self):
        return self.road_world if self.has_road else []

    def _project(self, w, t, R):
        """Map-frame polygon -> full-resolution pixel polygon under the pose (t, R), clipped to the near plane and the (1.1 x) image box
        exactly as the label renderer does (`labels._project_polygon`), with a single-polygon numpy clipper (3-4 x faster)."""
        Rbc, tbc = self.cm.T_base_cam.R, self.cm.T_base_cam.t
        P = ((w - t) @ R - tbc) @ Rbc
        lx, ly = self.lim
        for normal, off in (((0, 0, 1.0), self.z_min), ((-1.0, 0, lx), 0.0), ((1.0, 0, lx), 0.0), ((0, -1.0, ly), 0.0), ((0, 1.0, ly), 0.0)):
            P = _clip1(P, np.array(normal), off)
            if len(P) < 3:
                return np.zeros((0, 2))
        return L.project_unclipped(self.cm, P, 1e-6)[0]

    def _to_label(self, uv, ss):
        """full-resolution pixel polygon -> label-grid fixed-point coordinates (x ss)."""
        if len(uv) < 3:
            return np.zeros((0, 2), np.int32)
        q = np.stack([(uv[:, 0] + 0.5) * self.sx * ss - 0.5, (uv[:, 1] + 0.5) * self.sy * ss - 0.5], 1)
        return np.round(np.clip(q, -2e4, 2e4) * ONE).astype(np.int32)

    def mask(self, dn: float, dpsi: float) -> np.ndarray:
        """Road polygon (uint8 0/1, grid ss x label) at the hypothesis (dn, dpsi)."""
        canvas = np.zeros((LABEL_H * self.ss, LABEL_W * self.ss), np.uint8)
        if not self.has_road:
            return canvas
        x, y, yaw = self.pose0
        t = np.array([x + dn * self.nrm[0], y + dn * self.nrm[1], self.z_base])
        R = L.pose_rotation(self.roll0, self.pitch0, yaw + dpsi)
        qs = [self._to_label(self._project(w, t, R), self.ss) for w in self.road_world]
        qs = [q for q in qs if len(q) >= 3]
        if qs:
            cv2.fillPoly(canvas, qs, 1, lineType=cv2.LINE_8, shift=SHIFT)
        return canvas


# ------------------------------------------------------------------------------------------------ fitting
def _grid(cfg: SegFitConfig):
    dns = np.arange(-cfg.dn_range_m, cfg.dn_range_m + 1e-9, cfg.coarse_dn)
    dps = np.deg2rad(np.arange(-cfg.dpsi_range_deg, cfg.dpsi_range_deg + 1e-9, cfg.coarse_dpsi_deg))
    return dns, dps


def score_grid(scene: SegScene, E: np.ndarray, dns, dps) -> np.ndarray:
    S = np.zeros((len(dns), len(dps)))
    for i, dn in enumerate(dns):
        for j, dp in enumerate(dps):
            m = scene.mask(float(dn), float(dp))
            S[i, j] = float(cv2.mean(E, mask=m)[0] * cv2.countNonZero(m))
    return S


def _quad_peak(fdn, fdp, F, a, b, hw):
    """Maximum of a quadratic surface fitted to F around cell (a, b); returns (dn, dpsi, Hnn, Hpp, ok). Units: dn in m, dpsi in rad."""
    ia, ib = slice(max(a - hw, 0), a + hw + 1), slice(max(b - hw, 0), b + hw + 1)
    X, Y = np.meshgrid(fdn[ia], fdp[ib], indexing="ij")
    z = F[ia, ib].ravel()
    x0, y0 = fdn[a], fdp[b]
    u, v = (X.ravel() - x0), (Y.ravel() - y0) * 20.0          # scale dpsi (rad) so that both axes are O(0.1)
    A = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], 1)
    c, *_ = np.linalg.lstsq(A, z, rcond=None)
    Hm = np.array([[2 * c[3], c[4]], [c[4], 2 * c[5]]])
    if not (Hm[0, 0] < 0 and Hm[1, 1] < 0 and np.linalg.det(Hm) > 0):
        return float(x0), float(y0), float("nan"), float("nan"), False
    d = np.linalg.solve(Hm, -np.array([c[1], c[2]]))
    lim = np.array([3 * (fdn[1] - fdn[0]), 3 * (fdp[1] - fdp[0]) * 20.0])
    d = np.clip(d, -lim, lim)
    return float(x0 + d[0]), float(y0 + d[1] / 20.0), float(Hm[0, 0]), float(Hm[1, 1] * 400.0), True


class SegMeasurer:
    """Border measurement from probability maps. `polys` = `labels.build_road_polys(track_bounds)` (supplied borders only)."""

    def __init__(self, ctx: ProbeContext, polys: L.RoadPolys, cfg: SegFitConfig | None = None):
        self.ctx, self.polys, self.cfg = ctx, polys, cfg or SegFitConfig()
        self._body = {}

    def body_ignore(self, cam: str) -> np.ndarray:
        if cam not in self._body:
            usable = self.ctx.masks[cam].astype(np.uint8)
            bad = (1 - usable).astype(np.float32)
            self._body[cam] = cv2.resize(bad, (LABEL_W, LABEL_H), interpolation=cv2.INTER_AREA) >= 0.5
        return self._body[cam]

    # -------------------------------------------------------------------------------------------- single and joint
    def measure(self, cam: str, prob: np.ndarray, pose0, s_init=None, attitude=None) -> BorderMeas:
        """One camera. `s_init`: along-track coordinate of `pose0` from the caller's own estimate (never a GT value)."""
        return self.measure_joint({cam: {"prob": prob, "pose0": pose0, "s_init": s_init, "attitude": attitude}}, ref=cam)

    def measure_joint(self, items: dict, ref: str | None = None) -> BorderMeas:
        """One shared correction (dn, dpsi) in track coordinates for the cameras in `items`:
        camera -> {"prob", "pose0" (the caller's pose estimate at THIS camera's image stamp), "s_init", "attitude" (optional)}.
        Each camera's hypothesis shifts its own pose0 by dn along the centreline normal and rotates it by dpsi. The returned
        (n, dpsi, s, x, y, yaw) are the reference camera's pose0 (`ref`, default the first) with that correction applied."""
        t0 = time.perf_counter()
        try:
            sc_ref, out, q = self._fit(items, ref)
        except Exception as e:      # abstain, never crash a stream
            return BorderMeas(False, f"error:{type(e).__name__}", q={"service_ms": 1e3 * (time.perf_counter() - t0)})
        q["service_ms"] = 1e3 * (time.perf_counter() - t0)
        if out is None:
            return BorderMeas(False, q.pop("reason"), q=q)
        dn, dp = out
        q["dn_fit"], q["dpsi_fit"] = float(dn), float(dp)
        x, y, yaw = sc_ref.pose0[0] + dn * sc_ref.nrm[0], sc_ref.pose0[1] + dn * sc_ref.nrm[1], sc_ref.pose0[2] + dp
        s, n, th, _ = self.ctx.cline.project(x, y, s_hint=sc_ref.s0)
        return BorderMeas(True, "", n=float(n), dpsi=wrap_angle(yaw - th), s=float(s), x=float(x), y=float(y), yaw=float(wrap_angle(yaw)), q=q)

    def _fit(self, items, ref):
        cfg = self.cfg
        dns, dps = _grid(cfg)
        scenes, Es, reasons = {}, [], []
        for cam, it in items.items():
            sc = SegScene(self, cam, it["pose0"], it.get("s_init"), it.get("attitude"))
            if not sc.has_road:
                reasons.append("no_border_near")
                continue
            if int(sc.mask(0.0, 0.0).sum()) < cfg.min_poly_px * cfg.ss ** 2:
                reasons.append("few_pixels")
                continue
            scenes[cam] = sc
            Es.append(evidence_image(it["prob"], sc.ignore, cfg.p_clip, cfg.ss))
        if not scenes:
            return None, None, {"reason": reasons[0] if reasons else "no_border_near", "n_cams": 0}
        sc_ref = scenes.get(ref) or next(iter(scenes.values()))
        sl = list(scenes.values())
        C = sum(score_grid(sc, E, dns, dps) for sc, E in zip(sl, Es))
        i, j = np.unravel_index(int(np.argmax(C)), C.shape)
        scale = sum(float(np.abs(E).sum()) for E in Es)
        contrast = float((C[i, j] - np.median(C)) / max(scale, 1e-9))
        q = {"contrast": contrast, "support": float(len(sl)), "n_points": float(sum(int(sc.mask(0, 0).sum()) for sc in sl) / cfg.ss ** 2),
             "cost": -float(C[i, j]), "n_cams": len(sl)}
        if i in (0, len(dns) - 1) or j in (0, len(dps) - 1):
            q["reason"] = "grid_edge"
            return sc_ref, None, q
        if contrast < cfg.min_contrast:
            q["reason"] = "low_contrast"
            return sc_ref, None, q
        fdn = np.arange(dns[i] - cfg.coarse_dn, dns[i] + cfg.coarse_dn + 1e-9, cfg.fine_dn)
        fdp = np.deg2rad(np.arange(np.rad2deg(dps[j]) - cfg.coarse_dpsi_deg, np.rad2deg(dps[j]) + cfg.coarse_dpsi_deg + 1e-9, cfg.fine_dpsi_deg))
        F = sum(score_grid(sc, E, fdn, fdp) for sc, E in zip(sl, Es))
        a, b = np.unravel_index(int(np.argmax(F)), F.shape)
        dn, dp, hnn, hpp, ok = _quad_peak(fdn, fdp, F, int(a), int(b), cfg.refine_halfwidth)
        q["cost"] = -float(F[a, b])
        q["sig_raw_n"] = float(math.sqrt(-1.0 / hnn)) if ok else float("nan")      # unit-evidence curvature width (m); calibrated out of fold
        q["sig_raw_dpsi"] = float(math.sqrt(-1.0 / hpp)) if ok else float("nan")
        if not ok:
            q["reason"] = "flat"
            return sc_ref, None, q
        return sc_ref, (dn, dp), q
