"""Feed-forward visual geometry with Depth Anything 3 (stand-in for S4VY's VGGT-Omega backbone)."""

from __future__ import annotations

import gc
import types
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Geometry:
    images: np.ndarray  # (T, H, W, 3) uint8 RGB at the processing resolution
    depth: np.ndarray  # (T, H, W) z-depth, up to a global scale
    conf: np.ndarray  # (T, H, W) DA3 confidence
    K: np.ndarray  # (T, 3, 3) intrinsics at the processing resolution
    w2c: np.ndarray  # (T, 4, 4) world-to-camera

    def save(self, path: str) -> None:
        np.savez_compressed(path, images=self.images, depth=self.depth, conf=self.conf, K=self.K, w2c=self.w2c)

    @classmethod
    def load(cls, path: str) -> "Geometry":
        d = np.load(path)
        return cls(d["images"], d["depth"], d["conf"], d["K"], d["w2c"])

    @property
    def num_frames(self) -> int:
        return self.depth.shape[0]

    def unproject(self, t: int, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
        """World points for pixels (ys, xs) of frame t, same convention as DA3's GLB export."""
        z = self.depth[t, ys, xs]
        k = self.K[t]
        xc = (xs - k[0, 2]) * z / k[0, 0]
        yc = (ys - k[1, 2]) * z / k[1, 1]
        c2w = np.linalg.inv(self.w2c[t])
        pts = np.stack([xc, yc, z], axis=1)
        return (pts @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32)

    def valid_mask(self, conf_percentile: float) -> np.ndarray:
        """Pixels whose depth is finite and whose confidence is above the run-wide percentile."""
        thr = np.percentile(self.conf, conf_percentile)
        return np.isfinite(self.depth) & (self.depth > 0) & (self.conf >= thr)


def _fp32_forward(self, image, extrinsics=None, intrinsics=None, export_feat_layers=None, infer_gs=False,
                  use_ray_pose=False, ref_view_strategy="saddle_balanced"):
    # DA3 autocasts to float16; CPUs without native fp16 emulate it, which measured
    # 3x slower than plain fp32 here (27.2 s vs 9.2 s for 4 views, 0.02% depth difference).
    with torch.inference_mode():
        return self.model(image, extrinsics, intrinsics, export_feat_layers, infer_gs, use_ray_pose,
                          ref_view_strategy)


def estimate_geometry(frames_rgb: list[np.ndarray], model_id: str = "depth-anything/DA3-BASE",
                      process_res: int = 504) -> Geometry:
    from depth_anything_3.api import DepthAnything3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnything3.from_pretrained(model_id).to(device).eval()
    if device == "cpu":
        model.forward = types.MethodType(_fp32_forward, model)
    pred = model.inference(list(frames_rgb), process_res=process_res)
    n = pred.depth.shape[0]
    w2c = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    w2c[:, :3, :4] = pred.extrinsics[:, :3, :4]
    geom = Geometry(pred.processed_images, pred.depth.astype(np.float32), pred.conf.astype(np.float32),
                    pred.intrinsics.astype(np.float64), w2c)
    del model, pred
    gc.collect()
    return geom
