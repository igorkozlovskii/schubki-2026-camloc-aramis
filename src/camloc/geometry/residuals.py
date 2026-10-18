"""Image-edge residuals of projected 3D borders and their Jacobians (E02).

Two formulations over the same projected sample points:

* `normal_search`: for every sample, search the image gradient along the image normal of the projected polyline in a
  bounded band (+-W px), keep the strongest orientation-consistent edge; residual = signed offset (px) to that edge.
* `distance_transform`: residual = distance (px) from the projected sample to the nearest edge pixel of a Canny edge map,
  with the Jacobian taken from the gradient of the distance transform; an orientation gate compares the nearest edge's
  gradient direction with the polyline normal.

Both return the pre-update residual r0 (N,), the 1x4 row Jacobians jn (N,4) w.r.t. (x, y, yaw, pitch) in *full-resolution
pixels*, a quality weight in [0, 1], and gate bookkeeping. Residual sign convention: r(delta) = r0 + jn @ delta.
Everything works on a possibly reduced image (factor `reduce`); inputs and outputs are in full-resolution pixel units.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class EdgeImage:
    gx: np.ndarray
    gy: np.ndarray
    mag: np.ndarray
    reduce: int
    usable: Optional[np.ndarray] = None  # bool (h, w) at the reduced resolution, False = car body etc.
    dt: Optional[np.ndarray] = None
    dt_labels: Optional[np.ndarray] = None
    edge_xy: Optional[np.ndarray] = None
    n_edge_px: int = 0

    @property
    def shape(self):
        return self.mag.shape


def build_edge_image(gray: np.ndarray, reduce: int = 1, usable_mask: Optional[np.ndarray] = None, blur_sigma: float = 1.2,
                     with_dt: bool = True, canny_lo: float = 6.0, canny_hi: float = 18.0) -> EdgeImage:
    """Gradient field and (optionally) distance transform of a Canny edge map. `gray` is uint8 at the reduced resolution."""
    g = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    gf = g.astype(np.float32)
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = np.hypot(gx, gy)
    usable = None
    if usable_mask is not None:
        um = usable_mask
        if um.shape != gray.shape:
            um = cv2.resize(um.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        usable = um.astype(bool)
    ei = EdgeImage(gx, gy, mag, reduce, usable)
    if with_dt:
        e = cv2.Canny(g, canny_lo, canny_hi, L2gradient=True)
        if usable is not None:
            e[~usable] = 0
        ys, xs = np.nonzero(e)
        ei.n_edge_px = int(len(ys))
        if len(ys) > 0:
            dt, lab = cv2.distanceTransformWithLabels(255 - e, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
            ei.dt, ei.dt_labels = dt, lab
            ei.edge_xy = np.stack([xs, ys], axis=1)
    return ei


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) + img[y0 + 1, x0] * (1 - fx) * fy
            + img[y0 + 1, x0 + 1] * fx * fy)


def image_tangents(uv: np.ndarray, run_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit image tangents of consecutive samples within a run (central differences); ok=False where undefined."""
    n = len(uv)
    tau = np.zeros((n, 2))
    ok = np.zeros(n, bool)
    for r in np.unique(run_id):
        ii = np.where(run_id == r)[0]
        if len(ii) < 2:
            continue
        p = uv[ii]
        d = np.zeros_like(p)
        d[1:-1] = p[2:] - p[:-2]
        d[0] = p[1] - p[0]
        d[-1] = p[-1] - p[-2]
        nrm = np.linalg.norm(d, axis=1)
        good = nrm > 0.5
        tau[ii[good]] = d[good] / nrm[good, None]
        ok[ii[good]] = True
    return tau, ok


def thin_in_image(uv: np.ndarray, run_id: np.ndarray, min_px: float) -> np.ndarray:
    """Boolean keep-mask so that kept samples of a run are at least `min_px` apart along the run in the image."""
    keep = np.zeros(len(uv), bool)
    for r in np.unique(run_id):
        ii = np.where(run_id == r)[0]
        last = None
        for i in ii:
            if last is None or np.linalg.norm(uv[i] - uv[last]) >= min_px:
                keep[i] = True
                last = i
    return keep


@dataclass
class Residuals:
    r0: np.ndarray           # (M,)
    jn: np.ndarray           # (M, 4)
    weight: np.ndarray       # (M,) in [0, 1]
    n_samples: int           # samples offered (after visibility, mask and tangent checks)
    n_found: int             # samples with an edge passing the strength and orientation gates
    n_orient_rejected: int
    n_weak_rejected: int
    formulation: str
    idx: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # indices (into the offered samples) of kept rows

    @property
    def support(self) -> float:
        return self.n_found / max(1, self.n_samples)


def normal_search(edge: EdgeImage, uv: np.ndarray, J: np.ndarray, tau: np.ndarray, band_px: float = 24.0,
                  g_min: float = 3.0, orient_cos: float = 0.9, prior_sigma_frac: float = 0.6) -> Residuals:
    """Signed normal-search residuals. uv/J in full-resolution pixels; the search runs at the reduced resolution."""
    r = edge.reduce
    nrm = np.stack([-tau[:, 1], tau[:, 0]], axis=1)
    W = int(max(2, round(band_px / r)))
    offs = np.arange(-W, W + 1, dtype=np.float64)
    n = len(uv)
    px = (uv[:, 0] / r)[:, None] + offs[None, :] * nrm[:, 0:1]
    py = (uv[:, 1] / r)[:, None] + offs[None, :] * nrm[:, 1:2]
    inside = (px >= 1) & (px < edge.shape[1] - 2) & (py >= 1) & (py < edge.shape[0] - 2)
    gxs = _bilinear(edge.gx, px.ravel(), py.ravel()).reshape(n, -1)
    gys = _bilinear(edge.gy, px.ravel(), py.ravel()).reshape(n, -1)
    gn = gxs * nrm[:, 0:1] + gys * nrm[:, 1:2]
    mags = np.hypot(gxs, gys)
    score = np.abs(gn) * np.exp(-(offs[None, :] ** 2) / (2.0 * (prior_sigma_frac * W) ** 2)) * inside
    if edge.usable is not None:
        us = _bilinear(edge.usable.astype(np.float32), px.ravel(), py.ravel()).reshape(n, -1) > 0.99
        score = score * us
    k = np.argmax(score, axis=1)
    rows = np.arange(n)
    best = np.abs(gn[rows, k])
    cosang = best / (mags[rows, k] + 1e-9)
    weak = (best < g_min) | (k <= 1) | (k >= 2 * W - 1)   # a peak at the band limit is not a local maximum: not an observation
    orient_bad = (~weak) & (cosang < orient_cos)
    ok = ~weak & ~orient_bad
    # sub-pixel parabola refinement of the peak position
    kk = np.clip(k, 1, 2 * W - 1)
    a = np.abs(gn[rows, kk - 1])
    b = np.abs(gn[rows, kk])
    c = np.abs(gn[rows, kk + 1])
    den = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        sub = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / den, 0.0)
    sub = np.nan_to_num(sub)
    sub = np.clip(sub, -0.5, 0.5) * (k == kk)
    off = (offs[k] + sub) * r                     # full-resolution pixels
    r0 = -off                                     # r(delta) = -o* + n^T J delta
    jn = np.einsum("ni,nik->nk", nrm, J)
    wq = np.clip(best / 20.0, 0.1, 1.0)
    keep = np.where(ok)[0]
    return Residuals(r0[keep], jn[keep], wq[keep], n, int(ok.sum()), int(orient_bad.sum()), int(weak.sum()), "normal_search", keep)


def distance_transform(edge: EdgeImage, uv: np.ndarray, J: np.ndarray, tau: np.ndarray, max_px: float = 30.0,
                       orient_cos: float = 0.9) -> Residuals:
    """Distance-to-nearest-Canny-edge residuals with the DT gradient as the normal direction."""
    n = len(uv)
    if edge.dt is None or edge.n_edge_px == 0:
        return Residuals(np.zeros(0), np.zeros((0, 4)), np.zeros(0), n, 0, 0, n, "distance_transform")
    r = edge.reduce
    x, y = uv[:, 0] / r, uv[:, 1] / r
    h, w = edge.shape
    inb = (x >= 2) & (x < w - 3) & (y >= 2) & (y < h - 3)
    xi = np.clip(np.round(x).astype(int), 2, w - 3)
    yi = np.clip(np.round(y).astype(int), 2, h - 3)
    d = _bilinear(edge.dt, x, y)
    gdx = (_bilinear(edge.dt, x + 1, y) - _bilinear(edge.dt, x - 1, y)) * 0.5
    gdy = (_bilinear(edge.dt, x, y + 1) - _bilinear(edge.dt, x, y - 1)) * 0.5
    gnorm = np.hypot(gdx, gdy)
    lab = edge.dt_labels[yi, xi] - 1
    lab = np.clip(lab, 0, len(edge.edge_xy) - 1)
    exy = edge.edge_xy[lab]
    egx = edge.gx[exy[:, 1], exy[:, 0]]
    egy = edge.gy[exy[:, 1], exy[:, 0]]
    emag = np.hypot(egx, egy) + 1e-9
    nrm = np.stack([-tau[:, 1], tau[:, 0]], axis=1)
    cosang = np.abs(egx * nrm[:, 0] + egy * nrm[:, 1]) / emag
    far = d * r > max_px
    flat = gnorm < 0.3                                    # on a ridge/at the edge pixel: gradient undefined
    orient_bad = inb & ~far & (cosang < orient_cos)
    ok = inb & ~far & ~orient_bad
    # residual in full-res pixels; direction = DT gradient (away from the edge)
    gdir = np.where(gnorm[:, None] > 1e-6, np.stack([gdx, gdy], 1) / np.maximum(gnorm, 1e-6)[:, None], 0.0)
    r0 = d * r
    jn = np.einsum("ni,nik->nk", gdir, J)
    wq = np.clip(emag / 20.0, 0.1, 1.0)
    keep = np.where(ok & ~flat)[0]
    return Residuals(r0[keep], jn[keep], wq[keep], n, int((ok).sum()), int(orient_bad.sum()), int(far.sum()), "distance_transform", keep)


# ------------------------------------------------------------------ robust Gauss-Newton on the stacked residuals
def robust_weights(res: np.ndarray, scale: float, loss: str = "cauchy") -> np.ndarray:
    z = res / scale
    if loss == "cauchy":
        return 1.0 / (1.0 + z * z)
    if loss == "huber":
        a = np.abs(z)
        return np.where(a <= 1.0, 1.0, 1.0 / np.maximum(a, 1e-9))
    raise ValueError(loss)


def information_and_gradient(rs: list[Residuals], delta: np.ndarray, scale_px: float, loss: str = "cauchy"):
    """H = sum w J^T J / sigma^2 and g = sum w J^T r / sigma^2 at the linearization point delta, plus robust cost,
    inlier counts. `rs` are residual sets (one per camera/hypothesis); weights combine robust and quality terms."""
    H = np.zeros((4, 4))
    g = np.zeros(4)
    cost = 0.0
    n_in = 0
    n_all = 0
    for R in rs:
        if len(R.r0) == 0:
            continue
        res = R.r0 + R.jn @ delta
        w = robust_weights(res, scale_px, loss) * R.weight
        H += (R.jn * w[:, None]).T @ R.jn
        g += (R.jn * w[:, None]).T @ res
        cost += float(np.sum(np.log1p((res / scale_px) ** 2)))
        n_in += int(np.sum(np.abs(res) < 2.5 * scale_px))
        n_all += len(res)
    s2 = scale_px ** 2
    return H / s2, g / s2, cost, n_in, n_all
