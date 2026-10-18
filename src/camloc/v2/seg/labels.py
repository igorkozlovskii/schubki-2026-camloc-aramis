"""Paved-surface label rendering from the supplied map borders and a GT pose (OFFLINE BUILDER ONLY).

Label classes (uint8, label resolution LABEL_W x LABEL_H = 384 x 192, i.e. ~1/3.9 of the 1506 x 728 image):
    1   paved surface inside the main-track borders, on the car's own layer, within `far_m`
    0   other (not road, or road outside the borders)
    255 ignore: static car-body mask, the other layer (bridge deck / tunnel level / other elevation), the pit-lane
        surface, and main-track surface beyond `far_m` (sub-pixel strips; label noise grows with range)

Method. The road surface between `track_left` and `track_right` is cut into small quads along the polylines (each left
point is paired with the nearest right point in 3-D, z weighted, so overhead crossings are not paired). Per frame the
quads near the GT pose are projected (full plumb_bob distortion, GT z/roll/pitch, optional per-camera calibration
offsets) and filled with `cv2.fillPoly`. Quads with a vertex behind the camera or outside the valid distortion range
are clipped (Sutherland-Hodgman in the camera frame: near plane and image box) so a partly visible quad contributes its
visible part; quads are pre-cut to <= 1 m across so distorted edges stay close to straight. Layer gating follows
`geometry/borders.py`: a quad belongs to the own layer when |z_quad - z_road| <= dz_gate + slope_allow * range with
z_road = z_base + BASE_BELOW_ROAD_M. Occlusion: nothing is modelled beyond (i) the layer rule, (ii) the static
body mask, (iii) the far cut; walls, other cars and foliage that hide road inside the corridor are NOT in the labels
(declared in the report).

Offline builder: reads GT poses; never import this module from an inference path (a test enforces it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree

from camloc.geometry.borders import BASE_BELOW_ROAD_M
from camloc.geometry.camera import CameraModel

from . import CLASS_IGNORE, CLASS_OTHER, CLASS_ROAD, LABEL_H, LABEL_W

SHIFT = 4          # sub-pixel bits for fillPoly
ONE = 1 << SHIFT


@dataclass(frozen=True)
class LabelConfig:
    width: int = LABEL_W
    height: int = LABEL_H
    max_range_m: float = 150.0
    far_m: float = 80.0
    dz_gate: float = 1.5
    slope_allow: float = 0.06
    z_min: float = 0.3
    above_cam_range_m: float = 40.0   # beyond this range uphill road above the camera is legitimate and kept
    above_cam_margin_m: float = 0.03  # road higher than the camera centre minus this is not trusted (height error flips it)
    max_width_m: float = 40.0
    max_index_step: int = 10
    z_scale_pair: float = 3.0
    body_threshold: float = 0.3
    body_top_frac: float = 0.52
    body_dilate_px: int = 9          # at full resolution, before resizing to the label grid
    body_close_px: int = 15
    body_min_area: float = 1500.0    # full-res px^2
    body_max_top: float = 0.5        # contour centroid must lie below this fraction of the image height



# ------------------------------------------------------------------------------------------------ road strips
@dataclass
class Strip2:
    """One left/right border pair. Segment i joins left points i and i+1; its right-border span is jj[i]..jj[i+1]
    (cumulative, unwrapped index of the 3-D nearest right point), `ok[i]` False when the pairing is implausible."""
    group: int                 # 0 = main track, 1 = pit lane
    L: np.ndarray
    R: np.ndarray
    closed_l: bool
    closed_r: bool
    jj: np.ndarray             # (nL + 1,) cumulative right index
    ok: np.ndarray             # (nL,) bool
    tree: cKDTree

    @property
    def nL(self):
        return len(self.L)


@dataclass
class RoadPolys:
    strips: list


def _build_strip(group: int, L: np.ndarray, R: np.ndarray, cfg: LabelConfig) -> Strip2:
    cl = bool(np.linalg.norm(L[0] - L[-1]) < 1e-6)
    cr = bool(np.linalg.norm(R[0] - R[-1]) < 1e-6)
    if cl:
        L = L[:-1]
    if cr:
        R = R[:-1]
    nL, nR = len(L), len(R)
    tr = cKDTree(np.c_[R[:, 0], R[:, 1], cfg.z_scale_pair * R[:, 2]])
    j = tr.query(np.c_[L[:, 0], L[:, 1], cfg.z_scale_pair * L[:, 2]])[1]
    n_seg = nL if cl else nL - 1
    d = np.zeros(nL, int)
    ok = np.zeros(nL, bool)
    for i in range(n_seg):
        i1 = (i + 1) % nL
        dd = int(j[i1]) - int(j[i])
        if cr:
            dd = (dd + nR // 2) % nR - nR // 2
        d[i] = dd
        w0 = np.hypot(*(L[i, :2] - R[j[i], :2]))
        w1 = np.hypot(*(L[i1, :2] - R[j[i1], :2]))
        ok[i] = abs(dd) <= cfg.max_index_step and w0 <= cfg.max_width_m and w1 <= cfg.max_width_m
    jj = np.concatenate([[int(j[0])], int(j[0]) + np.cumsum(d)])
    return Strip2(group, L, R, cl, cr, jj, ok, cKDTree(L[:, :2]))


def build_road_polys(track_bounds: dict, cfg: LabelConfig = LabelConfig()) -> RoadPolys:
    out = []
    for g, (ln, rn) in enumerate((("track_left", "track_right"), ("pitlane_left", "pitlane_right"))):
        out.append(_build_strip(g, np.asarray(track_bounds[ln], float).reshape(-1, 3),
                                np.asarray(track_bounds[rn], float).reshape(-1, 3), cfg))
    return RoadPolys(out)


def _runs(cls: np.ndarray, cyclic: bool):
    """Maximal runs of equal class >= 0 as (start, length, class); cyclic when the strip is closed."""
    n = len(cls)
    if n == 0:
        return []
    start0 = 0
    if cyclic:
        brk = np.where(cls != np.roll(cls, 1))[0]
        if len(brk) == 0:
            return [(0, n, int(cls[0]))] if cls[0] >= 0 else []
        start0 = int(brk[0])
    order = (np.arange(n) + start0) % n if cyclic else np.arange(n)
    out, cur, ln = [], None, 0
    for k in order:
        c = int(cls[k])
        if c == cur and ln > 0:
            ln += 1
        else:
            if cur is not None and cur >= 0:
                out.append((st, ln, cur))
            cur, st, ln = c, int(k), 1
    if cur is not None and cur >= 0:
        out.append((st, ln, cur))
    return out


def corridor_polygons(strip: Strip2, pose_t: np.ndarray, cfg: LabelConfig, cam_z: float = np.inf):
    """3-D corridor polygons around the pose: list of (vertices (M, 3), class) with class 0 = own layer within far_m,
    1 = own layer beyond far_m, 2 = other layer / pit lane (ignore)."""
    nL = strip.nL
    near = np.zeros(nL, bool)
    near[strip.tree.query_ball_point(pose_t[:2], cfg.max_range_m)] = True
    nxt = (np.arange(nL) + 1) % nL
    segok = strip.ok & near & near[nxt]
    mid = 0.5 * (strip.L + strip.L[nxt])
    rng = np.hypot(mid[:, 0] - pose_t[0], mid[:, 1] - pose_t[1])
    z_road = pose_t[2] + BASE_BELOW_ROAD_M
    # layer test on both borders of the segment: left midpoint and the right point of its span
    jr = strip.jj[:-1] % len(strip.R) if strip.closed_r else np.clip(strip.jj[:-1], 0, len(strip.R) - 1)
    zR = strip.R[jr, 2]
    gate = cfg.dz_gate + cfg.slope_allow * rng
    own = (np.abs(mid[:, 2] - z_road) <= gate) & (np.abs(zR - z_road) <= gate + 0.5) & (strip.group == 0)
    # camera almost at road level (rear cameras ~0.12 m): a few cm of height error puts road above the camera centre -> ignore
    own &= (rng > cfg.above_cam_range_m) | (0.5 * (mid[:, 2] + zR) <= cam_z - cfg.above_cam_margin_m)
    cls = np.full(nL, -1, int)
    cls[segok & own & (rng <= cfg.far_m)] = 0
    cls[segok & own & (rng > cfg.far_m)] = 1
    cls[segok & ~own] = 2
    nR = len(strip.R)
    polys = []
    for st, ln, c in _runs(cls, strip.closed_l):
        li = (st + np.arange(ln + 1)) % nL
        if not strip.closed_l and st + ln > nL - 1:
            li = li[li >= st]
        ja, jb = int(strip.jj[st]), int(strip.jj[st] + (strip.jj[(st + ln)] - strip.jj[st] if st + ln <= nL else 0))
        if st + ln <= nL:
            jb = int(strip.jj[st + ln])
        else:                          # wrapped run: unwrapped cumulative index continues past the end
            jb = int(strip.jj[nL] + strip.jj[(st + ln) - nL] - strip.jj[0])
        step = 1 if jb >= ja else -1
        ri = np.arange(ja, jb + step, step)
        ri = ri % nR if strip.closed_r else np.clip(ri, 0, nR - 1)
        verts = np.concatenate([strip.L[li], strip.R[ri][::-1]], 0)
        polys.append((verts, c))
    return polys


# ------------------------------------------------------------------------------------------------ projection
_CORNER_R: dict = {}


def _corner_radius(cam: CameraModel) -> float:
    key = (cam.name, cam.width, cam.height, float(cam.K[0, 0]))
    if key not in _CORNER_R:
        fx, fy, cx, cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
        w, h = cam.width, cam.height
        _CORNER_R[key] = max(math.hypot((a - cx) / fx, (b - cy) / fy) for a in (0, w) for b in (0, h))
    return _CORNER_R[key]


def project_unclipped(cam: CameraModel, pc: np.ndarray, z_min: float):
    """Camera-frame points -> pixels (full resolution, distorted) and a mask (in front and inside the distortion range).
    Unlike `CameraModel.project_cam` the pixel need not lie inside the image (polygons are clipped by the rasteriser)."""
    z = pc[:, 2]
    front = z > z_min
    zs = np.where(front, z, 1.0)
    x, y = pc[:, 0] / zs, pc[:, 1] / zs
    ok = front & ((x * x + y * y) < (1.15 * _corner_radius(cam)) ** 2)
    xd, yd, _ = cam.distort(x, y)
    uv = np.stack([cam.K[0, 0] * xd + cam.K[0, 2], cam.K[1, 1] * yd + cam.K[1, 2]], 1)
    return uv, ok


def pose_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


_LIMITS: dict = {}


def _view_limits(cam: CameraModel, margin: float = 1.1):
    """Half-extents (x, y) of the undistorted normalized image plane covered by the image (from the undistorted border
    pixels), times `margin`. Wide lenses reach far beyond the pinhole value width / (2 fx)."""
    key = (cam.name, cam.width, cam.height, float(cam.K[0, 0]), tuple(np.round(cam.D, 6)))
    if key not in _LIMITS:
        u = np.r_[np.linspace(0, cam.width, 60), np.linspace(0, cam.width, 60), np.zeros(40), np.full(40, cam.width)]
        v = np.r_[np.zeros(60), np.full(60, cam.height), np.linspace(0, cam.height, 40), np.linspace(0, cam.height, 40)]
        n = cam.undistort_normalized(np.c_[u, v])
        _LIMITS[key] = (margin * float(np.abs(n[:, 0]).max()), margin * float(np.abs(n[:, 1]).max()))
    return _LIMITS[key]


def _clip_plane(P: np.ndarray, n: np.ndarray, normal: np.ndarray, offset: float = 0.0):
    """Sutherland-Hodgman against the half-space normal . p - offset > 0 for Q convex-ish polygons held as a padded array
    P (Q, m, 3) with vertex counts n (Q,). Returns (P', n') with m' = m + 1."""
    Q, m, _ = P.shape
    d = P @ normal - offset                                    # (Q, m)
    idx = np.arange(m)[None, :]
    valid = idx < n[:, None]
    nxt = (idx + 1) % np.maximum(n[:, None], 1)                # next vertex index within each polygon's own count
    Pn = np.take_along_axis(P, nxt[..., None].repeat(3, 2), axis=1)
    dn = np.take_along_axis(d, nxt, axis=1)
    ins, insn = d > 0, dn > 0
    cross = valid & (ins != insn)
    t = d / np.where(np.abs(d - dn) < 1e-12, 1.0, d - dn)
    X = P + (Pn - P) * t[..., None]
    emit_end = valid & insn
    slots = np.stack([cross, emit_end], 2).reshape(Q, 2 * m)  # per edge: crossing point first, then the end vertex
    pts = np.stack([X, Pn], 2).reshape(Q, 2 * m, 3)
    order = np.argsort(~slots, axis=1, kind="stable")[:, : m + 1]
    out = np.take_along_axis(pts, order[..., None].repeat(3, 2), axis=1)
    cnt = slots.sum(1)
    cnt = np.minimum(cnt, m + 1)
    return out, cnt


def _clip_to_view(pc: np.ndarray, cam: CameraModel, z_min: float):
    """Clip camera-frame quads (Q, 4, 3) to the near plane and the (1.15 x) normalized image box; returns (P, cnt)."""
    fx, fy, cx, cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
    lx = 1.15 * max(cx, cam.width - cx) / fx
    ly = 1.15 * max(cy, cam.height - cy) / fy
    P, n = pc, np.full(len(pc), 4)
    for normal, off in ((np.array([0, 0, 1.0]), z_min), (np.array([-1.0, 0, lx]), 0.0), (np.array([1.0, 0, lx]), 0.0),
                        (np.array([0, -1.0, ly]), 0.0), (np.array([0, 1.0, ly]), 0.0)):
        P, n = _clip_plane(P, n, normal, off)
    return P, n


def _project_polygon(cam, pose_R, pose_t, verts: np.ndarray, z_min: float):
    """Map-frame polygon (M, 3) -> clipped pixel polygon (K, 2) at full resolution (empty array when nothing is visible)."""
    Rbc, tbc = cam.T_base_cam.R, cam.T_base_cam.t
    pc = ((verts - pose_t) @ pose_R - tbc) @ Rbc
    P, n = pc[None], np.array([len(pc)])
    lx, ly = _view_limits(cam)
    for normal, off in ((np.array([0, 0, 1.0]), z_min), (np.array([-1.0, 0, lx]), 0.0), (np.array([1.0, 0, lx]), 0.0),
                        (np.array([0, -1.0, ly]), 0.0), (np.array([0, 1.0, ly]), 0.0)):
        P, n = _clip_plane(P, n, normal, off)
        if n[0] < 3:
            return np.zeros((0, 2))
    uv, _ = project_unclipped(cam, P[0, : n[0]], 1e-6)
    return uv


SS = 4   # supersampling of the polygon rasteriser (thin far-range strips leave 1-px cracks at 1x)


def _fill(mask: np.ndarray, polys: list, sx: float, sy: float):
    """Rasterise pixel polygons (full-res coordinates) into `mask` (H, W) by SS x supersampling; a label pixel is set when
    at least half of its area is covered."""
    if not polys:
        return
    H, W = mask.shape
    pts = []
    for q in polys:
        q = q.copy()
        q[:, 0] = (q[:, 0] + 0.5) * (sx * SS) - 0.5
        q[:, 1] = (q[:, 1] + 0.5) * (sy * SS) - 0.5
        pts.append(np.round(np.clip(q, -2e4, 2e4) * ONE).astype(np.int32))
    big = np.zeros((H * SS, W * SS), np.uint8)
    cv2.fillPoly(big, pts, 1, lineType=cv2.LINE_8, shift=SHIFT)
    mask[big.reshape(H, SS, W, SS).mean(axis=(1, 3)) >= 0.5] = 1


def render_label(cam: CameraModel, pose: dict, polys: RoadPolys, body_ignore: np.ndarray | None,
                 cfg: LabelConfig = LabelConfig()) -> np.ndarray:
    """One (H, W) uint8 label (0 / 1 / 255) for a camera at a GT pose {x,y,z,roll,pitch,yaw}.

    `cam` is the full-resolution camera model (with any calibration offset already applied); `body_ignore` is a bool mask
    (H, W) at label resolution or None."""
    H, W = cfg.height, cfg.width
    sx, sy = W / cam.width, H / cam.height
    R = pose_rotation(pose["roll"], pose["pitch"], pose["yaw"])
    t = np.array([pose["x"], pose["y"], pose["z"]])
    masks = [np.zeros((H, W), np.uint8) for _ in range(3)]
    buckets = [[], [], []]
    cam_z = float((t + R @ cam.T_base_cam.t)[2])
    for strip in polys.strips:
        for verts, c in corridor_polygons(strip, t, cfg, cam_z):
            q = _project_polygon(cam, R, t, verts, cfg.z_min)
            if len(q) >= 3:
                buckets[c].append(q)
    for m, b in zip(masks, buckets):
        _fill(m, b, sx, sy)
    label = np.zeros((H, W), np.uint8)
    label[masks[0] > 0] = CLASS_ROAD
    label[((masks[1] > 0) | (masks[2] > 0)) & (masks[0] == 0)] = CLASS_IGNORE
    # ground closer than z_min (clipped away) is neither road nor non-road: ignore everything below the lowest labelled row
    any_poly = (masks[0] | masks[1] | masks[2]) > 0
    has = any_poly.any(axis=0)
    last = H - 1 - np.argmax(any_poly[::-1], axis=0)
    below = (np.arange(H)[:, None] > last[None, :]) & has[None, :]
    label[below & (label == CLASS_OTHER)] = CLASS_IGNORE
    if body_ignore is not None:
        label[body_ignore] = CLASS_IGNORE
    return label


# ------------------------------------------------------------------------------------------------ static body masks
def body_mask_from_structure(frac: np.ndarray, cfg: LabelConfig = LabelConfig()) -> np.ndarray:
    """Filled car-body region (bool, full resolution) from a static-structure fraction map (outline-like: textured pixels
    that stay unchanged between frames). Threshold, close, keep large external contours whose centroid is in the lower
    `body_max_top` fraction of the image (the horizon band and lens-edge lines are static too but are not the body), fill."""
    h, w = frac.shape
    m = (frac.astype(np.float32) >= cfg.body_threshold).astype(np.uint8)
    m[: int(cfg.body_top_frac * h)] = 0     # horizon band / far static structure is not the body
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((cfg.body_close_px, cfg.body_close_px), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros((h, w), np.uint8)
    for c in cs:
        a = cv2.contourArea(c)
        if a < cfg.body_min_area:
            continue
        mm = cv2.moments(c)
        cy = mm["m01"] / max(mm["m00"], 1e-9)
        if cy < cfg.body_max_top * h:
            continue
        cv2.drawContours(out, [c], -1, 1, thickness=-1)
    return cv2.dilate(out, np.ones((cfg.body_dilate_px, cfg.body_dilate_px), np.uint8)) > 0


def load_body_masks(npz_path, cfg: LabelConfig = LabelConfig()) -> dict:
    """Bool ignore masks at label resolution from the S03 static-structure fraction maps (pixel-only asset, no GT)."""
    z = np.load(npz_path)
    out = {}
    for key in z.files:
        cam = key.replace("static_", "")
        m = body_mask_from_structure(z[key], cfg).astype(np.uint8)
        out[cam] = cv2.resize(m, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA) >= 0.5
    return out


def load_body_masks_png(mask_dir, cfg: LabelConfig = LabelConfig()) -> dict:
    """Bool ignore masks at label resolution from `<mask_dir>/<cam>.png` (S1.1 format: any size, >0 = car body)."""
    from pathlib import Path
    out = {}
    for f in sorted(Path(mask_dir).glob("*.png")):
        m = (cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        out[f.stem] = cv2.resize(m, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA) >= 0.5
    return out
