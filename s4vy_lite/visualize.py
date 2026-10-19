"""Overlays with persistent instance colors and IDs, and a novel-view render of the 4D point cloud."""

from __future__ import annotations

import colorsys

import cv2
import numpy as np


def palette(n: int) -> np.ndarray:
    """(n+1, 3) uint8 RGB; index 0 (no instance) is black. Golden-ratio hues stay
    distinct for many IDs; every overlay also prints the ID, so color is never the only cue."""
    cols = [(0, 0, 0)]
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.75, 0.95)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(cols, dtype=np.uint8)


def _label_anchor(mask: np.ndarray):
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return int(x), int(y)


def overlay(image_rgb: np.ndarray, id_map: np.ndarray, colors: np.ndarray, alpha: float = 0.55,
            caption: str | None = None) -> np.ndarray:
    """BGR frame with translucent instance fills, outlines and ID labels."""
    out = image_rgb.astype(np.float32)
    present = id_map > 0
    out[present] = (1 - alpha) * out[present] + alpha * colors[id_map[present]]
    out = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)
    for iid in np.unique(id_map[present]):
        m = (id_map == iid).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
        x, y = _label_anchor(m)
        text = str(int(iid))
        cv2.putText(out, text, (x - 6, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (x - 6, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if caption:
        cv2.putText(out, caption, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, caption, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_overlays(images: np.ndarray, id_maps: np.ndarray, colors: np.ndarray, video_path: str,
                   grid_path: str, fps: float = 2.0, grid_cols: int = 4, max_grid: int = 12) -> None:
    frames = [overlay(images[t], id_maps[t], colors, caption=f"frame {t}") for t in range(len(images))]
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    pick = np.linspace(0, len(frames) - 1, min(max_grid, len(frames))).round().astype(int)
    tiles = [cv2.resize(frames[i], (w // 2, h // 2), interpolation=cv2.INTER_AREA) for i in pick]
    while len(tiles) % grid_cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + grid_cols]) for i in range(0, len(tiles), grid_cols)]
    cv2.imwrite(grid_path, np.vstack(rows))


def render_novel_view(xyz: np.ndarray, rgb: np.ndarray, K: np.ndarray, w2c: np.ndarray, hw,
                      yaw_deg: float = 25.0, pitch_deg: float = -12.0, point_px: int = 2) -> np.ndarray:
    """Z-buffered point splat from a camera orbited around the scene centre.

    A novel view exposes the 3D structure: if depth or the instance lifting were
    wrong, objects would smear or tear apart when seen from the side.
    """
    h, w = hw
    c2w = np.linalg.inv(w2c)
    center = np.median(xyz, axis=0)
    ry = cv2.Rodrigues(np.array([0.0, np.radians(yaw_deg), 0.0]))[0]
    rx = cv2.Rodrigues(np.array([np.radians(pitch_deg), 0.0, 0.0]))[0]
    rot = c2w[:3, :3] @ rx @ ry @ c2w[:3, :3].T  # orbit in the camera's own frame
    new_c2w = np.eye(4)
    new_c2w[:3, :3] = rot @ c2w[:3, :3]
    new_c2w[:3, 3] = center + rot @ (c2w[:3, 3] - center)
    view = np.linalg.inv(new_c2w)
    pc = xyz @ view[:3, :3].T + view[:3, 3]
    front = pc[:, 2] > 1e-3
    pc, cols = pc[front], rgb[front]
    u = (K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).round().astype(int)
    v = (K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).round().astype(int)
    order = np.argsort(-pc[:, 2])  # far to near, near points overwrite
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    for dy in range(point_px):
        for dx in range(point_px):
            uu, vv = u[order] + dx, v[order] + dy
            ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            img[vv[ok], uu[ok]] = cols[order][ok]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
