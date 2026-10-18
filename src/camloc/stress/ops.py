"""Deterministic pixel corruption operations (DEVELOPMENT_PLAN.md section 9). Pure functions on RGB uint8 arrays.

Every random draw comes from a generator seeded by `derive_seed(...)`, so the same (manifest seed, condition, camera,
frame stamp) gives the same bytes regardless of which candidate asks, in which order, or how many frames it skips.
These are SYNTHETIC appearance changes; none of them is a physical weather or daylight model.
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np


def derive_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little")


def _f(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def gamma(img, value: float, **_):
    return _u8(255.0 * np.power(_f(img) / 255.0, float(value)))


def exposure(img, factor: float, **_):
    return _u8(_f(img) * float(factor))  # clipping at 255 is intended


def white_balance(img, r_gain: float, b_gain: float, **_):
    x = _f(img)
    x[..., 0] *= float(r_gain)
    x[..., 2] *= float(b_gain)
    return _u8(x)


def haze(img, alpha: float, airlight: float = 200.0, **_):
    """Airlight blend, strongest at the top of the frame (horizon side): alpha_y = alpha * (0.5 + 0.5 * (1 - y/H))."""
    h = img.shape[0]
    a = float(alpha) * (0.5 + 0.5 * (1.0 - np.arange(h, dtype=np.float32) / max(h - 1, 1)))
    a = a[:, None, None]
    return _u8(_f(img) * (1.0 - a) + float(airlight) * a)


def glare(img, strength: float, radius_frac: float, rng: np.random.Generator, **_):
    """One bounded Gaussian bright blob; centre drawn (per condition+camera) in the upper 60 % of the frame."""
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.05, 0.6) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    s = float(radius_frac) * w
    g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * s * s)))[..., None]
    return _u8(_f(img) + 255.0 * float(strength) * g)


def shadow(img, factor: float, width_frac: float, rng: np.random.Generator, **_):
    """Dark diagonal band (multiplicative `factor`), soft edges; geometry drawn per condition+camera."""
    h, w = img.shape[:2]
    c = rng.uniform(0.25, 0.75) * w
    slope = rng.uniform(-0.5, 0.5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.abs((xx - c) - slope * (yy - h / 2.0))
    half = 0.5 * float(width_frac) * w
    m = 1.0 / (1.0 + np.exp((d - half) / (0.02 * w)))
    return _u8(_f(img) * (1.0 - (1.0 - float(factor)) * m)[..., None])


def motion_blur(img, length_px: int, angle_deg: float = 0.0, **_):
    k = int(length_px)
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0
    m = cv2.getRotationMatrix2D(((k - 1) / 2.0, (k - 1) / 2.0), float(angle_deg), 1.0)
    ker = cv2.warpAffine(ker, m, (k, k))
    ker /= ker.sum()
    return cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_REPLICATE)


def gaussian_noise(img, sigma: float, rng: np.random.Generator, **_):
    return _u8(_f(img) + rng.normal(0.0, float(sigma), img.shape).astype(np.float32))


def jpeg_roundtrip(img, quality: int, **_):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    assert ok
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


OPS = {"gamma": gamma, "exposure": exposure, "white_balance": white_balance, "haze": haze, "glare": glare,
       "shadow": shadow, "motion_blur": motion_blur, "noise": gaussian_noise, "jpeg": jpeg_roundtrip,
       "identity": lambda img, **_: img}
GEOMETRY_OPS = {"glare", "shadow"}   # randomness fixed per (condition, camera), not per frame
FRAME_RANDOM_OPS = {"noise"}          # randomness fixed per (condition, camera, frame stamp)


def apply_ops(img_rgb: np.ndarray, ops: list[dict], master_seed: int, cond_id: str, camera: str, stamp_ns: int) -> np.ndarray:
    out = img_rgb
    for i, op in enumerate(ops):
        name = op["op"]
        if name not in OPS:
            raise ValueError(f"unknown stress op {name!r}")
        kw = {k: v for k, v in op.items() if k != "op"}
        if name in GEOMETRY_OPS:
            kw["rng"] = np.random.default_rng(derive_seed(master_seed, cond_id, camera, i, "geom"))
        elif name in FRAME_RANDOM_OPS:
            kw["rng"] = np.random.default_rng(derive_seed(master_seed, cond_id, camera, i, stamp_ns))
        out = OPS[name](out, **kw)
    return out


def corrupt_jpeg(jpeg: bytes, ops: list[dict], master_seed: int, cond_id: str, camera: str, stamp_ns: int, out_quality: int = 95) -> bytes:
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("JPEG decode failed")
    rgb = apply_ops(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ops, master_seed, cond_id, camera, stamp_ns)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(out_quality)])
    assert ok
    return buf.tobytes()
