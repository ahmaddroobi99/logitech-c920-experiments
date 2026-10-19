"""Frame utility metrics: how useful a frame is for feature-based vision.

`keypoints` stands in for the R2D2 score used offline: FAST corners counted
per grid cell with a per-cell cap, so the score rewards features spread over
the image, not a cluster on one textured patch. `gradient` is the gradient
information metric of Shim et al. (2014); `zhang` is the soft-percentile
gradient metric of Zhang et al. (ICRA 2017).
"""

from __future__ import annotations

import cv2
import numpy as np

UTILITIES = ("keypoints", "gradient")

FAST_THRESHOLD = 20
GRID_COLS, GRID_ROWS, CELL_CAP = 8, 6, 25

SHIM_DELTA, SHIM_LAMBDA = 0.06, 1e3

_fast = cv2.FastFeatureDetector_create(threshold=FAST_THRESHOLD, nonmaxSuppression=True)


def image_gradients(gray_u8: np.ndarray):
    g = gray_u8.astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3, scale=0.25)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3, scale=0.25)
    return gx, gy, cv2.magnitude(gx, gy)


def keypoint_points(gray_u8: np.ndarray) -> np.ndarray:
    kps = _fast.detect(gray_u8, None)
    return np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)


def keypoint_utility(gray_u8: np.ndarray, points: np.ndarray | None = None) -> float:
    """Fraction of the grid's keypoint capacity that is filled (0..1)."""
    pts = keypoint_points(gray_u8) if points is None else points
    if len(pts) == 0:
        return 0.0
    h, w = gray_u8.shape
    counts, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=(GRID_ROWS, GRID_COLS), range=((0, h), (0, w)))
    return float(np.minimum(counts, CELL_CAP).sum() / (GRID_ROWS * GRID_COLS * CELL_CAP))


def gradient_information(gray_u8: np.ndarray, mag: np.ndarray | None = None) -> float:
    """Shim et al. 2014: mean log-compressed gradient magnitude above a noise floor."""
    if mag is None:
        mag = image_gradients(gray_u8)[2]
    strong = mag[mag >= SHIM_DELTA]
    norm = np.log1p(SHIM_LAMBDA * (1.0 - SHIM_DELTA)) * mag.size
    return float(np.log1p(SHIM_LAMBDA * (strong - SHIM_DELTA)).sum() / norm)


def soft_percentile_weights(n: int, percentile: float = 0.5, power: float = 5.0) -> np.ndarray:
    """Zhang et al. 2017 weights over ascending-sorted gradients, peaking at `percentile`."""
    i = np.arange(1, n + 1, dtype=np.float64)
    split = int(np.clip(percentile * n, 1, n - 1))
    w = np.empty(n)
    w[:split] = np.sin(0.5 * np.pi * i[:split] / split) ** power
    w[split:] = np.sin(0.5 * np.pi - 0.5 * np.pi * (i[split:] - split) / (n - split)) ** power
    return w / w.sum()


_weights_cache: dict[tuple, np.ndarray] = {}


def _weights(n: int, percentile: float, power: float) -> np.ndarray:
    key = (n, percentile, power)
    if key not in _weights_cache:
        _weights_cache[key] = soft_percentile_weights(n, percentile, power)
    return _weights_cache[key]


def zhang_metric(gray_u8: np.ndarray, percentile: float = 0.5, power: float = 5.0,
                 dpix_dlogt: np.ndarray | None = None):
    """Soft-percentile gradient metric M and, given a CRF, dM/d ln(exposure).

    Following Zhang et al., the derivative of every pixel's gradient magnitude
    with respect to exposure comes from the CRF: dI/d ln t = 1 / g'(I), so
    d|grad I|/d ln t = grad I . grad(dI/d ln t) / |grad I|.
    """
    small = cv2.resize(gray_u8, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    gx, gy, mag = image_gradients(small)
    flat = mag.ravel()
    order = np.argsort(flat, kind="stable")
    w = _weights(flat.size, percentile, power)
    metric = float(np.dot(w, flat[order]))
    if dpix_dlogt is None:
        return metric, float("nan")
    d_img = dpix_dlogt[small] / 255.0
    dgx = cv2.Sobel(d_img, cv2.CV_32F, 1, 0, ksize=3, scale=0.25)
    dgy = cv2.Sobel(d_img, cv2.CV_32F, 0, 1, ksize=3, scale=0.25)
    dmag = ((gx * dgx + gy * dgy) / np.maximum(mag, 1e-4)).ravel()
    return metric, float(np.dot(w, dmag[order]))


def frame_metrics(gray_u8: np.ndarray) -> dict:
    mag = image_gradients(gray_u8)[2]
    return {
        "mean_intensity": float(gray_u8.mean()),
        "frac_saturated": float(np.mean(gray_u8 >= 250)),
        "frac_dark": float(np.mean(gray_u8 <= 5)),
        "keypoints": keypoint_utility(gray_u8),
        "gradient": gradient_information(gray_u8, mag),
        "zhang": zhang_metric(gray_u8)[0],
    }


def utility(gray_u8: np.ndarray, kind: str) -> float:
    if kind == "keypoints":
        return keypoint_utility(gray_u8)
    if kind == "gradient":
        return gradient_information(gray_u8)
    raise ValueError(f"unknown utility {kind!r}; choose from {UTILITIES}")


def track_survival(prev_gray: np.ndarray, gray_u8: np.ndarray, max_corners: int = 300) -> float:
    """Fraction of Shi-Tomasi corners that KLT tracks forward and back within 1 px.

    Brightness constancy is part of what is being tested: an exposure jump
    between frames breaks KLT just as it would break a VO front end.
    """
    p0 = cv2.goodFeaturesToTrack(prev_gray, max_corners, 0.01, 8)
    if p0 is None or len(p0) < 10:
        return float("nan")
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray_u8, p0, None)
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray_u8, prev_gray, p1, None)
    good = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(p0 - p0r, axis=2).ravel() < 1.0)
    return float(good.mean())
