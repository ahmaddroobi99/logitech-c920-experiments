"""Grab a short multi-view clip from the Logitech webcam (or load frames from disk)."""

from __future__ import annotations

import glob
import os
import time

import cv2
import numpy as np

from maverick_c920.camera import UvcCamera, find_logitech_camera

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def capture(out_dir: str, n_frames: int, interval_s: float, device: str | None = None,
            countdown_s: int = 3, display: bool = False) -> list[str]:
    """Save `n_frames` frames `interval_s` apart. Move the camera slowly around the
    scene while it runs: the geometry model needs parallax to recover 3D."""
    os.makedirs(out_dir, exist_ok=True)
    device = device or find_logitech_camera()
    paths = []
    with UvcCamera(device, 640, 480, 30) as cam:
        # Normal webcam behaviour: camera AE, and autofocus back on (OpenCV disables it).
        cam.set("focus_automatic_continuous", 1)
        for s in range(countdown_s, 0, -1):
            print(f"capturing {n_frames} frames from {device} in {s}...", flush=True)
            t_end = time.monotonic() + 1.0
            while time.monotonic() < t_end:
                frame = cam.read()
                if display:
                    cv2.imshow("s4vy-lite capture", frame)
                    cv2.waitKey(1)
        next_t = time.monotonic()
        while len(paths) < n_frames:
            frame = cam.read()
            if display:
                cv2.imshow("s4vy-lite capture", frame)
                cv2.waitKey(1)
            if time.monotonic() >= next_t:
                path = os.path.join(out_dir, f"{len(paths):05d}.jpg")
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                paths.append(path)
                print(f"  frame {len(paths)}/{n_frames}", flush=True)
                next_t += interval_s
        if display:
            cv2.destroyAllWindows()
    return paths


def load_frames(source: str, max_frames: int) -> list[np.ndarray]:
    """RGB frames from an image folder or a video, evenly subsampled to `max_frames`."""
    if os.path.isdir(source):
        paths = sorted(p for p in glob.glob(os.path.join(source, "*")) if p.lower().endswith(IMAGE_EXTS))
        frames = [cv2.imread(p) for p in paths]
    else:
        cap = cv2.VideoCapture(source)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames found in {source}")
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
