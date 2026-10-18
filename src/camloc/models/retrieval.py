"""E03 building blocks (global retrieval) and the shared local-feature stack for E04: ALIKED + LightGlue + DINOv2.

Third-party code and weights live under `artifacts/external/S04-*` (manifests/artifacts/S04-acquisitions*.json record URL,
revision, sha256, licenses). Nothing here imports ground truth; the inference path (E03/E04 models) imports only this
module, `camloc.mapping.map` (reads the training-only map) and `camloc.geometry`.

The retrieval model `E03RetrievalModel` emits the pose of the best retrieved training keyframe (a candidate generator
used as a coarse pose baseline for the table 'dev pose improvement vs retrieval-only'); it is not a precise pose.
"""
from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from camloc.contracts import Capabilities, ContractError, FirstPose, FrameBundle, StaticMetadata, StepResult

ROOT = Path(__file__).resolve().parents[3]
EXT = ROOT / "artifacts/external"
# crop of the useful image area: drops the 12 px black border, the hood reflection and the dark patch (rows >= 540)
DEFAULT_CROP = (12, 0, 1494, 540)   # x0, y0, x1, y1


@contextlib.contextmanager
def _local_weights(files: dict):
    """torch.hub.load_state_dict_from_url -> local files keyed by URL basename (no network, no hub cache)."""
    import torch
    orig = torch.hub.load_state_dict_from_url

    def fake(url, *a, **k):
        base = url.rsplit("/", 1)[-1]
        if base not in files:
            raise ContractError(f"unexpected weight download requested: {url}")
        return torch.load(str(files[base]), map_location="cpu")

    torch.hub.load_state_dict_from_url = fake
    try:
        yield
    finally:
        torch.hub.load_state_dict_from_url = orig


def _wdir(ext_dir) -> Path:
    return Path(ext_dir or EXT) / "S04-lightglue/weights"


class LocalFeatures:
    """ALIKED-n16 keypoints+descriptors on the cropped front image; keypoint coords are in FULL-image pixels."""

    def __init__(self, ext_dir=None, max_kp: int = 2048, det_thr: float = 0.1, scale: float = 1.0, crop=DEFAULT_CROP, device="cuda"):
        import torch
        from lightglue import ALIKED
        w = _wdir(ext_dir)
        with _local_weights({"aliked-n16.pth": w / "aliked-n16.pth"}):
            self.net = ALIKED(model_name="aliked-n16", max_num_keypoints=max_kp, detection_threshold=det_thr).eval().to(device)
        self.torch, self.device, self.scale, self.crop = torch, device, scale, tuple(crop)

    def extract(self, rgb: np.ndarray) -> dict:
        torch = self.torch
        x0, y0, x1, y1 = self.crop
        im = rgb[y0:y1, x0:x1]
        if self.scale != 1.0:
            import cv2
            im = cv2.resize(im, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(np.ascontiguousarray(im)).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        with torch.no_grad():
            o = self.net({"image": t})
        kp = o["keypoints"][0].float().cpu().numpy() / self.scale + np.array([x0, y0], np.float32)
        return {"kpts": kp.astype(np.float32), "desc": o["descriptors"][0].half().cpu().numpy(),
                "scores": o["keypoint_scores"][0].float().cpu().numpy()}


class Matcher:
    """LightGlue (ALIKED weights) between two feature sets; returns (M,2) index pairs and scores."""

    def __init__(self, ext_dir=None, device="cuda", filter_threshold: float = 0.1, image_size=(1506, 728)):
        import torch
        from lightglue import LightGlue
        w = _wdir(ext_dir)
        with _local_weights({"aliked_lightglue.pth": w / "aliked_lightglue_v0.1_arxiv.pth"}):
            self.net = LightGlue(features="aliked", filter_threshold=filter_threshold).eval().to(device)
        self.torch, self.device, self.size = torch, device, image_size

    def to_dev(self, feats: dict) -> dict:
        t = self.torch
        return {"kpts": t.as_tensor(feats["kpts"], device=self.device).float()[None],
                "desc": t.as_tensor(feats["desc"], device=self.device).float()[None]}

    def match(self, a: dict, b: dict):
        """a, b from to_dev(). Returns numpy (M,2) int64 and (M,) float scores."""
        t = self.torch
        sz = t.tensor([self.size], device=self.device, dtype=t.float32)
        with t.no_grad():
            out = self.net({"image0": {"keypoints": a["kpts"], "descriptors": a["desc"], "image_size": sz},
                            "image1": {"keypoints": b["kpts"], "descriptors": b["desc"], "image_size": sz}})
        return out["matches"][0].cpu().numpy(), out["scores"][0].float().cpu().numpy()


class GlobalDescriptor:
    """DINOv2-small (Apache-2.0) global descriptor: GeM(p=3) over patch tokens, L2-normalized (384-d)."""

    def __init__(self, ext_dir=None, device="cuda", size_hw=(252, 672), crop=DEFAULT_CROP, pool: str = "gem"):
        import torch
        from transformers import Dinov2Model
        self.model = Dinov2Model.from_pretrained(str(Path(ext_dir or EXT) / "S04-dinov2-small"), local_files_only=True).eval().to(device).half()
        self.torch, self.device, self.size, self.crop, self.pool = torch, device, size_hw, tuple(crop), pool
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def describe(self, rgb: np.ndarray) -> np.ndarray:
        import cv2
        torch = self.torch
        x0, y0, x1, y1 = self.crop
        im = cv2.resize(rgb[y0:y1, x0:x1], (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(im).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        t = ((t - self.mean) / self.std).half()
        with torch.no_grad():
            h = self.model(pixel_values=t).last_hidden_state.float()
        cls, p = h[:, 0], h[:, 1:]
        gem = p.clamp(min=1e-6).pow(3).mean(1).pow(1 / 3) if self.pool == "gem" else p.mean(1)
        v = cls if self.pool == "cls" else gem
        v = torch.nn.functional.normalize(v, dim=-1)
        return v[0].cpu().numpy().astype(np.float32)


# ------------------------------------------------------------------------------------------------ retrieval index
def angdiff(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi)


class RetrievalIndex:
    """Cosine similarity over training-keyframe descriptors with an optional spatial/heading prior mask."""

    def __init__(self, desc: np.ndarray, xy: np.ndarray, yaw: np.ndarray):
        self.desc = np.asarray(desc, np.float32)
        self.xy = np.asarray(xy, float)
        self.yaw = np.asarray(yaw, float)

    def query(self, q: np.ndarray, k: int = 5, center_xy=None, radius_m=None, yaw=None, yaw_tol=None):
        sim = self.desc @ q.astype(np.float32)
        mask = np.ones(len(sim), bool)
        if center_xy is not None and radius_m is not None:
            mask &= np.hypot(self.xy[:, 0] - center_xy[0], self.xy[:, 1] - center_xy[1]) <= radius_m
        if yaw is not None and yaw_tol is not None:
            mask &= angdiff(self.yaw, yaw) <= yaw_tol
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return np.zeros(0, int), np.zeros(0, np.float32)
        order = idx[np.argsort(-sim[idx])[:k]]
        return order, sim[order]


# ------------------------------------------------------------------------------------------------ prior (shared)
@dataclass
class PriorState:
    """Last accepted fix (or the first pose) and its stamp; the spatial prior grows with the time since it."""
    x: float
    y: float
    yaw: float
    t_ns: int
    from_fix: bool = False


def prior_radius(dt_s: float, base_m: float, vmax_mps: float) -> float:
    return base_m + vmax_mps * max(dt_s, 0.0)


# ------------------------------------------------------------------------------------------------ E03 model
class E03RetrievalModel:
    """Pose = pose of the top-1 retrieved training keyframe (restricted by the spatial prior when not lost)."""

    def __init__(self, config: dict | None = None, assets: dict | None = None):
        from camloc.mapping.map import load_map
        p = dict(config or {})                 # runner.create_model passes (config["params"], assets)
        a = dict(assets or {})
        self.p = p
        self.map = load_map(ROOT / a["map_dir"], need_local=False)
        self.capabilities = Capabilities(cameras=("fl",), clock_camera="fl", calibration=True, map_id=self.map.map_id, telemetry_fields=(),
                                         device="cuda", assets=(str(a["map_dir"]),), scale_status="metric", online_mapping="disabled")
        self.gd = GlobalDescriptor(a.get("ext_dir"), pool=self.map.manifest["global_descriptor"]["pool"])
        self.index = RetrievalIndex(self.map.global_desc, self.map.kf_xy, self.map.kf_yaw)
        self.prior: PriorState | None = None
        self.n_init = 0
        self.stats = {"restricted": 0, "global": 0}

    def initialize(self, first_pose: FirstPose, static_metadata: StaticMetadata) -> None:
        self.n_init += 1
        if self.n_init > 1:
            raise ContractError("E03 initialized twice")
        self.prior = PriorState(first_pose.x, first_pose.y, first_pose.yaw, first_pose.timestamp_ns)

    def step(self, bundle: FrameBundle) -> StepResult:
        pr = self.prior
        if pr is None:
            raise ContractError("E03 step before initialize")
        f = bundle.frames.get("fl")
        if f is None or not f.available:
            return StepResult(pr.x, pr.y, pr.yaw, source_flags=("hold_last", "no_image"), tracking_state="held")
        t = bundle.timestamp_ns
        dt = (t - pr.t_ns) / 1e9
        q = self.gd.describe(f.image())
        lost = dt > self.p.get("lost_s", 5.0)
        rad = None if lost else prior_radius(dt, self.p.get("prior_base_m", 40.0), self.p.get("vmax_mps", 45.0))
        idx, sims = self.index.query(q, k=1, center_xy=(pr.x, pr.y) if rad else None, radius_m=rad)
        if idx.size == 0 and rad is not None:
            idx, sims = self.index.query(q, k=1)
        self.stats["global" if rad is None else "restricted"] += 1
        if idx.size == 0:
            return StepResult(pr.x, pr.y, pr.yaw, source_flags=("hold_last", "no_candidate"), tracking_state="held",
                              consumed_min_ns=f.stamp_ns, consumed_max_ns=f.stamp_ns)
        j = int(idx[0])
        x, y, yaw = float(self.map.kf_xy[j, 0]), float(self.map.kf_xy[j, 1]), float(self.map.kf_yaw[j])
        self.prior = PriorState(x, y, yaw, t, True)
        return StepResult(x, y, yaw, source_flags=("retrieval_top1",), tracking_state="tracking" if sims[0] > self.p.get("min_sim", 0.0) else "weak",
                          consumed_min_ns=f.stamp_ns, consumed_max_ns=f.stamp_ns, residuals={"sim": float(sims[0]), "kf_id": j})

    def close(self) -> None:
        self.prior = None
