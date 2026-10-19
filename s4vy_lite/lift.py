"""Lift per-frame instance masks into 4D (3D points per instance per frame) and export them."""

from __future__ import annotations

import numpy as np

from .geometry import Geometry


def instance_stats(geom: Geometry, id_maps: np.ndarray, valid: np.ndarray, min_pixels: int = 30) -> list[dict]:
    """Per-instance 4D summary: visibility, 3D centroid track, extent and motion (scene-scale units)."""
    scene_scale = float(np.median(geom.depth[valid]))
    stats = []
    for iid in range(1, int(id_maps.max()) + 1):
        centroids, frames, areas, pts_all = [], [], [], []
        for t in range(geom.num_frames):
            m = id_maps[t] == iid
            areas.append(int(m.sum()))
            ys, xs = np.nonzero(m & valid[t])
            if len(ys) < min_pixels:
                continue
            pts = geom.unproject(t, ys, xs)
            centroids.append(np.median(pts, axis=0))
            frames.append(t)
            pts_all.append(pts[:: max(1, len(pts) // 2000)])
        if not frames:
            stats.append({"id": iid, "frames_visible": 0})
            continue
        c = np.array(centroids)
        p = np.concatenate(pts_all)
        lo, hi = np.percentile(p, 5, axis=0), np.percentile(p, 95, axis=0)
        stats.append({
            "id": iid,
            "frames_visible": len(frames),
            "first_frame": frames[0],
            "last_frame": frames[-1],
            "mean_pixel_area": float(np.mean([areas[t] for t in frames])),
            "centroid": c.mean(axis=0).round(4).tolist(),
            "extent": (hi - lo).round(4).tolist(),
            "size_rel_scene": float(np.linalg.norm(hi - lo) / scene_scale),
            "motion_rel_scene": float(np.linalg.norm(c - np.median(c, axis=0), axis=1).max() / scene_scale),
        })
    return stats


def fused_points(geom: Geometry, id_maps: np.ndarray, valid: np.ndarray, voxel: float, seed: int = 0):
    """All frames' confident points, voxel-downsampled per (instance, frame)."""
    xyz, rgb, ids, ts = [], [], [], []
    for t in range(geom.num_frames):
        ys, xs = np.nonzero(valid[t])
        p = geom.unproject(t, ys, xs)
        xyz.append(p)
        rgb.append(geom.images[t, ys, xs])
        ids.append(id_maps[t, ys, xs].astype(np.int32))
        ts.append(np.full(len(ys), t, dtype=np.int32))
    xyz, rgb, ids, ts = (np.concatenate(a) for a in (xyz, rgb, ids, ts))
    q = np.floor(xyz / voxel).astype(np.int64)
    key = (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791) ^ (ids.astype(np.int64) << 40)
    order = np.random.default_rng(seed).permutation(len(key))
    _, first = np.unique(key[order], return_index=True)
    keep = order[first]
    return xyz[keep], rgb[keep], ids[keep], ts[keep]


def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray, instance: np.ndarray | None = None,
              frame: np.ndarray | None = None) -> None:
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if instance is not None:
        fields.append(("instance", "<i4"))
    if frame is not None:
        fields.append(("frame", "<i4"))
    data = np.empty(len(xyz), dtype=fields)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if instance is not None:
        data["instance"] = instance
    if frame is not None:
        data["frame"] = frame
    ply_types = {"<f4": "float", "u1": "uchar", "<i4": "int"}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(xyz)}"]
    header += [f"property {ply_types[t]} {name}" for name, t in fields]
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        data.tofile(f)
