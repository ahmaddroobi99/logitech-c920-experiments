"""Camera response function (CRF) and single-frame exposure emulation.

This is the live-camera counterpart of BorealRadianceEmulator: instead of
blending bracketed raw captures, we invert the camera's 8-bit response to get
relative irradiance, scale it by the exposure ratio and re-apply the response.
Because the result only depends on the pixel value and the ratio, emulation
is a 256-entry lookup table.

Limits (same as any single-frame emulator): clipped pixels carry no radiance
information, and brightening a frame amplifies its noise and quantization, so
controllers only trust emulation within a few stops of the captured exposure.
"""

from __future__ import annotations

import json

import cv2
import numpy as np


class ResponseCurve:
    """inv[d] = relative exposure (irradiance x time) that produces pixel value d."""

    def __init__(self, inv: np.ndarray, source: str):
        inv = np.asarray(inv, dtype=np.float64).reshape(256)
        inv = np.maximum.accumulate(np.maximum(inv, 1e-9))
        inv = inv + np.arange(256) * 1e-9 * inv.max()  # strictly increasing for inversion
        self.inv = inv / inv[128]
        self.source = source
        self._edges = 0.5 * (self.inv[:-1] + self.inv[1:])
        log_inv = np.log(self.inv)
        slope = np.gradient(log_inv)  # d ln(E t) / d pixel
        # d pixel / d ln(exposure): how much each value brightens per e-fold of exposure.
        dpix = 1.0 / np.maximum(slope, 1e-3)
        taper = np.clip((254.0 - np.arange(256)) / 20.0, 0, 1) * np.clip((np.arange(256) - 1.0) / 4.0, 0, 1)
        self.dpix_dlogt = (dpix * taper).astype(np.float32)

    @classmethod
    def gamma(cls, gamma: float = 2.2) -> "ResponseCurve":
        return cls(((np.arange(256) + 0.5) / 256.0) ** gamma, source=f"gamma-{gamma}")

    @classmethod
    def calibrate_debevec(cls, frames_bgr: list[np.ndarray], exposures_ms: list[float]) -> "ResponseCurve":
        """Debevec-Malik CRF from an exposure sweep of a static scene."""
        times = np.asarray(exposures_ms, dtype=np.float32) / 1000.0
        calib = cv2.createCalibrateDebevec(samples=200, lambda_=10.0, random=False)
        response = calib.process([np.ascontiguousarray(f) for f in frames_bgr], times)  # (256, 1, 3)
        return cls(response.reshape(256, 3).mean(axis=1), source="debevec")

    def lut(self, ratio: float) -> np.ndarray:
        """Pixel mapping for re-exposing a frame by `ratio` (new / captured exposure)."""
        return np.searchsorted(self._edges, self.inv * ratio).clip(0, 255).astype(np.uint8)

    def emulate(self, gray_u8: np.ndarray, ratio: float) -> np.ndarray:
        return cv2.LUT(gray_u8, self.lut(ratio))

    def mean_after(self, hist: np.ndarray, ratio: float) -> float:
        """Mean pixel value of the emulated frame, from the captured histogram."""
        return float(np.dot(hist, self.lut(ratio).astype(np.float64)) / hist.sum())

    def to_json(self) -> dict:
        return {"source": self.source, "inv": self.inv.tolist()}

    @classmethod
    def from_json(cls, data: dict) -> "ResponseCurve":
        return cls(np.asarray(data["inv"]), source=data["source"])

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f)

    @classmethod
    def load(cls, path) -> "ResponseCurve":
        with open(path) as f:
            return cls.from_json(json.load(f))


def emulation_error(curve: ResponseCurve, grays: list[np.ndarray], exposures_ms: list[float],
                    max_stops: float = 1.5) -> float:
    """Mean absolute error (DN) of emulating each sweep frame from its neighbours.

    Only pixels that are unclipped in both frames count, so this measures the
    CRF itself rather than the information lost to clipping.
    """
    errors = []
    for i, (src, e_src) in enumerate(zip(grays, exposures_ms)):
        for j, (dst, e_dst) in enumerate(zip(grays, exposures_ms)):
            if i == j or abs(np.log2(e_dst / e_src)) > max_stops:
                continue
            valid = (src > 4) & (src < 250) & (dst > 4) & (dst < 250)
            if valid.mean() < 0.05:
                continue
            pred = curve.emulate(src, e_dst / e_src)
            errors.append(np.abs(pred[valid].astype(np.float32) - dst[valid]).mean())
    return float(np.mean(errors)) if errors else float("nan")
