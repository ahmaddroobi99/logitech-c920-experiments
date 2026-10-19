"""Exhaustive and promptable 4D instance masks with SAM 2.1.

Stand-in for S4VY's space-time query decoder. S4VY decodes every instance in
one pass with persistent object queries. Here:
  1. SAM 2.1 automatic masks on a few keyframes spread over the clip,
  2. keyframe masks merged into objects by 3D overlap of their lifted points,
     so identity is anchored in geometry rather than in frame order,
  3. every object tracked through every frame by the SAM 2.1 video predictor,
     conditioned on all of its keyframe masks (forward, then backward for
     frames before an object's first keyframe).

SAM 2.1 runs at 512 px instead of 1024: DA3 frames are 504 px wide anyway,
and on CPU this cut tracking time 8.7x (162.6 s -> 18.7 s for 6 objects x 10 frames).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .geometry import Geometry

SAM2_REPO = "facebook/sam2.1-hiera-tiny"
SAM2_CKPT = "sam2.1_hiera_tiny.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_IMAGE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _ckpt() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(SAM2_REPO, SAM2_CKPT)


def _overrides() -> list[str]:
    return [f"++model.image_size={SAM2_IMAGE_SIZE}"]


def build_mask_generator(points_per_side: int = 16):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    model = build_sam2(SAM2_CFG, _ckpt(), device=DEVICE, hydra_overrides_extra=_overrides())
    amg = SAM2AutomaticMaskGenerator(model, points_per_side=points_per_side, points_per_batch=64,
                                     pred_iou_thresh=0.7, stability_score_thresh=0.85)
    s = SAM2_IMAGE_SIZE
    amg.predictor._bb_feat_sizes = [(s // 4,) * 2, (s // 8,) * 2, (s // 16,) * 2]  # hard-coded for 1024 upstream
    return amg


def _memory_to_fp32(module, args, kwargs):
    # SAM 2 stores its memory bank in bfloat16 and relies on CUDA autocast to mix it
    # with fp32 weights. On CPU without autocast, memory that holds no fp32 object
    # pointers (e.g. an object seeded only on later keyframes) stays bfloat16 and
    # crashes the fp32 projections, so cast it explicitly.
    kwargs = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in kwargs.items()}
    return args, kwargs


def build_video_predictor():
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(SAM2_CFG, _ckpt(), device=DEVICE, hydra_overrides_extra=_overrides())
    if DEVICE == "cpu":
        predictor.memory_attention.register_forward_pre_hook(_memory_to_fp32, with_kwargs=True)
    return predictor


@dataclass
class Seed:
    frame: int
    mask: np.ndarray  # (H, W) bool
    score: float


@dataclass
class Instance:
    seeds: list[Seed] = field(default_factory=list)
    voxels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    @property
    def first_frame(self) -> int:
        return min(s.frame for s in self.seeds)

    @property
    def area(self) -> int:
        return int(sum(s.mask.sum() for s in self.seeds))


def select_masks(amg_masks: list[dict], shape, min_area_frac: float = 0.002, max_area_frac: float = 0.8,
                 max_overlap: float = 0.3, whole_quality: float = 0.85) -> list[tuple[np.ndarray, float]]:
    """Greedy, near-non-overlapping subset of SAM's automatic masks, whole objects first.

    SAM proposes parts and wholes; S4VY's queries are object-level. Masks of high
    quality are taken largest-first so an object claims its pixels before its
    parts do; lower-quality masks only fill what is left.
    """
    area = shape[0] * shape[1]
    cands = [m for m in amg_masks if min_area_frac * area <= m["area"] <= max_area_frac * area]
    quality = lambda m: m["predicted_iou"] * m["stability_score"]  # noqa: E731
    cands.sort(key=lambda m: (quality(m) < whole_quality, -m["area"] if quality(m) >= whole_quality else -quality(m)))
    taken = np.zeros(shape, dtype=bool)
    out = []
    for m in cands:
        seg = m["segmentation"]
        if (seg & taken).sum() > max_overlap * seg.sum():
            continue
        seg = seg & ~taken
        if seg.sum() < min_area_frac * area:
            continue
        out.append((seg, float(m["predicted_iou"] * m["stability_score"])))
        taken |= seg
    return out


def _voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    q = np.floor(points / voxel).astype(np.int64)
    return np.unique((q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791))


def associate_in_3d(keyframe_masks: dict[int, list[tuple[np.ndarray, float]]], geom: Geometry, valid: np.ndarray,
                    voxel: float, assoc_iou: float = 0.25, max_points: int = 6000, seed: int = 0) -> list[Instance]:
    """Merge masks from different keyframes that cover the same 3D region into one instance."""
    rng = np.random.default_rng(seed)
    instances: list[Instance] = []
    for frame in sorted(keyframe_masks):
        for mask, score in keyframe_masks[frame]:
            ys, xs = np.nonzero(mask & valid[frame])
            if len(ys) == 0:
                continue
            if len(ys) > max_points:
                pick = rng.choice(len(ys), max_points, replace=False)
                ys, xs = ys[pick], xs[pick]
            keys = _voxel_keys(geom.unproject(frame, ys, xs), voxel)
            best, best_iou = None, 0.0
            for inst in instances:
                if any(s.frame == frame for s in inst.seeds):
                    continue  # two masks in one frame are two instances
                inter = len(np.intersect1d(keys, inst.voxels, assume_unique=True))
                iou = inter / (len(keys) + len(inst.voxels) - inter)
                if iou > best_iou:
                    best, best_iou = inst, iou
            if best is not None and best_iou >= assoc_iou:
                best.seeds.append(Seed(frame, mask, score))
                best.voxels = np.union1d(best.voxels, keys)
            else:
                instances.append(Instance([Seed(frame, mask, score)], keys))
    return instances


def track_instances(predictor, frames_dir: str, instances: list[Instance], shape) -> np.ndarray:
    """Per-object mask logits (N, T, H, W) for every instance in every frame."""
    with torch.inference_mode():
        state = predictor.init_state(frames_dir)
        n_frames = state["num_frames"]
        logits = np.full((len(instances), n_frames, *shape), -20.0, dtype=np.float16)
        for oid, inst in enumerate(instances, start=1):
            for s in inst.seeds:
                predictor.add_new_mask(state, s.frame, oid, s.mask)
        for f, ids, out in predictor.propagate_in_video(state, start_frame_idx=0):
            for k, oid in enumerate(ids):
                logits[oid - 1, f] = out[k, 0].float().cpu().numpy()
        late = {oid: inst.first_frame for oid, inst in enumerate(instances, start=1) if inst.first_frame > 0}
        if late:
            for f, ids, out in predictor.propagate_in_video(state, start_frame_idx=max(late.values()), reverse=True):
                for k, oid in enumerate(ids):
                    if f < late.get(oid, 0):
                        logits[oid - 1, f] = out[k, 0].float().cpu().numpy()
        predictor.reset_state(state)
    return logits


def click_to_mask(predictor, image_rgb: np.ndarray, point, granularity: str = "whole") -> np.ndarray:
    """Resolve a single click to one of SAM's three candidate masks.

    'best' takes SAM's top score. 'whole' takes the largest plausible candidate
    (score >= 0.25 and >= 35% of the best, under half the image), since S4VY's
    prompts select objects, not parts. Measured case: a click on a suitcase cut
    off by the frame edge scored its lid strip 0.82 and the whole suitcase 0.31.
    """
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image_predictor = SAM2ImagePredictor(predictor)
    s = SAM2_IMAGE_SIZE
    image_predictor._bb_feat_sizes = [(s // 4,) * 2, (s // 8,) * 2, (s // 16,) * 2]
    image_predictor.set_image(image_rgb)
    masks, scores, _ = image_predictor.predict(point_coords=np.asarray([point], dtype=np.float32),
                                               point_labels=np.array([1]), multimask_output=True)
    if granularity == "best":
        return masks[int(np.argmax(scores))].astype(bool)
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    ok = (scores >= 0.25) & (scores >= 0.35 * scores.max()) & (areas < 0.5 * masks[0].size)
    if not ok.any():
        return masks[int(np.argmax(scores))].astype(bool)
    return masks[int(np.argmax(np.where(ok, areas, -1)))].astype(bool)


def prompted_track(predictor, frames_dir: str, frame: int, point=None, box=None,
                   image_rgb: np.ndarray | None = None, granularity: str = "whole") -> np.ndarray:
    """Logits (T, H, W) of the single instance selected by a point or box on `frame`."""
    with torch.inference_mode():
        seed_mask = click_to_mask(predictor, image_rgb, point, granularity) if point is not None else None
        state = predictor.init_state(frames_dir)
        n_frames = state["num_frames"]
        if box is not None:
            predictor.add_new_points_or_box(state, frame, 1, box=np.asarray(box, dtype=np.float32))
        else:
            predictor.add_new_mask(state, frame, 1, seed_mask)
        logits = None
        for reverse in (False, True):
            for f, ids, out in predictor.propagate_in_video(state, start_frame_idx=frame, reverse=reverse):
                if logits is None:
                    logits = np.full((n_frames, *out.shape[-2:]), -20.0, dtype=np.float16)
                logits[f] = out[0, 0].float().cpu().numpy()
        predictor.reset_state(state)
    return logits


def id_maps_from_logits(logits: np.ndarray, min_pixels: int) -> tuple[np.ndarray, list[int]]:
    """Resolve overlaps (highest logit wins) into (T, H, W) instance-ID maps, 0 = none.

    Instances that end up with fewer than `min_pixels` in total are dropped and
    the rest renumbered 1..K. Returns the maps and, for each new ID, its old index.
    """
    best = logits.argmax(axis=0)
    ids = np.where(logits.max(axis=0) > 0, best + 1, 0).astype(np.int16)
    counts = np.bincount(ids.ravel(), minlength=logits.shape[0] + 1)
    keep = [i for i in range(1, logits.shape[0] + 1) if counts[i] >= min_pixels]
    remap = np.zeros(logits.shape[0] + 1, dtype=np.int16)
    for new, old in enumerate(keep, start=1):
        remap[old] = new
    return remap[ids], [old - 1 for old in keep]
