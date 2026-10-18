"""Static car-body masks per camera (S1.1), fitted from TRAIN-lap images only (no GT is read).

A pixel belongs to the car body when it is textured and unchanged between consecutive same-camera frames (the scene moves, the
car does not) for a large fraction of frames, or when it is connected (hysteresis) to such pixels and has a low temporal
standard deviation. Masks live at half resolution (decode `IMREAD_REDUCED_*_2`, 753 x 364 for the 1506 x 728 cameras); 255 = car
body / excluded, 0 = usable. Each fit writes PNGs and a manifest with the sha256 of every mask and the frames used.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from camloc.util import sha256_file

REDUCE = 2


def decode_gray(jpeg: bytes, reduce: int = REDUCE) -> np.ndarray:
    flag = {1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2, 4: cv2.IMREAD_REDUCED_GRAYSCALE_4}[reduce]
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), flag)


def fit_mask(store, cam: str, stamps_eligible: np.ndarray, n_pairs: int = 200, seed: int = 0, hi: float = 0.40, lo: float = 0.20,
             std_max: float = 0.0, dilate_px: int = 3, min_gap_s: float = 20.0, diff_max: float = 6.0):
    """Return (mask uint8 {0,255}, diagnostics dict, used list[(cam, stamp_ns)]).

    `stamps_eligible`: sorted stamps (ns) of train-eligible frames of this camera, restricted by the caller to the fit laps."""
    s = np.sort(np.asarray(stamps_eligible, np.int64))
    rng = np.random.default_rng(seed)
    # pairs of frames at least `min_gap_s` apart: the scene (even distant scenery) differs, the car body does not
    ia = rng.integers(0, len(s), 6 * n_pairs)
    ib = rng.integers(0, len(s), 6 * n_pairs)
    good = np.abs(s[ia] - s[ib]) > int(min_gap_s * 1e9)
    pick = list(zip(ia[good][:n_pairs], ib[good][:n_pairs]))
    pick = np.arange(len(pick))
    pairs = list(zip(ia[good][:n_pairs], ib[good][:n_pairs]))
    idx_all = np.searchsorted(store.stamps[cam], s)
    acc = None
    n = 0
    mean = m2 = None
    used = []
    for p in pick:
        i0, i1 = int(idx_all[pairs[p][0]]), int(idx_all[pairs[p][1]])
        g = [cv2.GaussianBlur(decode_gray(store.jpeg(cam, i)).astype(np.float32), (0, 0), 1.5) for i in (i0, i1)]
        if acc is None:
            acc = np.zeros(g[0].shape, np.float32)
            mean = np.zeros_like(acc)
            m2 = np.zeros_like(acc)
        gx, gy = cv2.Sobel(g[0], cv2.CV_32F, 1, 0), cv2.Sobel(g[0], cv2.CV_32F, 0, 1)
        acc += ((np.hypot(gx, gy) > 6) & (np.abs(g[0] - g[1]) < diff_max)).astype(np.float32)
        n += 1
        d = g[0] - mean
        mean += d / n
        m2 += d * (g[0] - mean)
        used += [(cam, int(store.stamps[cam][i0])), (cam, int(store.stamps[cam][i1]))]
    frac = acc / n
    std = np.sqrt(m2 / max(n - 1, 1))
    seed_m = (frac >= hi).astype(np.uint8)
    grow = ((frac >= lo) | (std < std_max)).astype(np.uint8)
    nl, lab = cv2.connectedComponents(np.maximum(seed_m, grow), connectivity=8)
    keep_lab = np.unique(lab[seed_m > 0])
    keep_lab = keep_lab[keep_lab > 0]
    m = np.isin(lab, keep_lab).astype(np.uint8)
    # growing only through low-variance pixels can swallow dark sky/asphalt: the grown part must itself be low-variance or textured-static
    m = ((m > 0) & ((frac >= lo) | (std < std_max))).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)
    # components that reach the bottom image rows are body parts seen from above: everything below their upper contour is body
    nl2, lab2, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    H = m.shape[0]
    for c in range(1, nl2):
        if stats[c, cv2.CC_STAT_AREA] < 150 or stats[c, cv2.CC_STAT_TOP] + stats[c, cv2.CC_STAT_HEIGHT] < H - 3:
            continue
        sel = lab2 == c
        cols = np.where(sel.any(0))[0]
        for x in cols:
            y0 = int(np.argmax(sel[:, x]))
            m[y0:, x] = 1
    m[:, :3] = 1
    m[:, -3:] = 1
    m = cv2.dilate(m, k)
    mask = (m * 255).astype(np.uint8)
    diag = {"n_pairs": int(n), "mask_fraction": float((mask > 0).mean()), "hi": hi, "lo": lo, "std_max": std_max,
            "dilate_px": dilate_px, "seed": seed}
    return mask, diag, used


def save_masks(out_dir: Path, masks: dict, diags: dict, used: dict, extra: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for cam, m in masks.items():
        p = out_dir / f"{cam}.png"
        cv2.imwrite(str(p), m)
        files[cam] = {"path": p.name, "sha256": sha256_file(p), "shape": list(m.shape), **diags[cam],
                      "frames_used": len(used[cam])}
    man = {"schema": "body-masks-v1", "reduce": REDUCE, "convention": "255 = car body (excluded), 0 = usable", "files": files, **extra}
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1))
    (out_dir / "frames_used.json").write_text(json.dumps({c: [s for _, s in u] for c, u in used.items()}))
    return out_dir / "manifest.json"


def load_masks(mask_dir, cams=None, verify: bool = True) -> dict:
    """camera -> bool array (True = usable) at half resolution; sha256 checked against the manifest."""
    d = Path(mask_dir)
    man = json.loads((d / "manifest.json").read_text())
    out = {}
    for cam, e in man["files"].items():
        if cams is not None and cam not in cams:
            continue
        p = d / e["path"]
        if verify and sha256_file(p) != e["sha256"]:
            raise ValueError(f"mask hash mismatch: {p}")
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        out[cam] = m == 0
    return out
