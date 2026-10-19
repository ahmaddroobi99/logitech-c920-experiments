"""Exposure controllers. Each one maps (current frame, its exposure) to the next exposure.

* camera-auto     the C920's own aperture-priority AE (optimizes appearance; also moves gain)
* fixed           constant exposure
* zhang-gradient  Zhang et al. (ICRA 2017): gradient ascent on the soft-percentile
                  gradient metric, with dM/d ln t from the camera response function
* predictive      live analog of the surrogate / exposure-optimization step: emulate
                  the current frame at candidate exposures, predict each one's
                  utility and jump to the best; a mid-grey warm start handles frames
                  too clipped to emulate from
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .metrics import utility, zhang_metric
from .response import ResponseCurve

CONTROLLERS = ("zhang-gradient", "predictive", "camera-auto", "fixed")


@dataclass
class Decision:
    next_exposure_ms: float
    warm_start_exposure_ms: float = math.nan
    predicted_exposure_ms: float = math.nan
    predicted_utility: float = math.nan  # predicted utility of the frame at next_exposure_ms
    note: str = ""


class Controller:
    name = "base"
    camera_driven = False  # True: the camera chooses exposure itself

    def __init__(self, bounds_ms: tuple[float, float]):
        self.lo, self.hi = bounds_ms

    def clip(self, exposure_ms: float) -> float:
        return float(np.clip(exposure_ms, self.lo, self.hi))

    def reset(self) -> None:
        pass

    def step(self, gray: np.ndarray, exposure_ms: float) -> Decision:
        raise NotImplementedError


class CameraAuto(Controller):
    name = "camera-auto"
    camera_driven = True

    def step(self, gray, exposure_ms):
        return Decision(math.nan, note="camera AE")


class Fixed(Controller):
    name = "fixed"

    def __init__(self, bounds_ms, exposure_ms: float):
        super().__init__(bounds_ms)
        self.exposure_ms = exposure_ms

    def step(self, gray, exposure_ms):
        return Decision(self.clip(self.exposure_ms))


class ZhangGradient(Controller):
    """Gradient ascent on Zhang's metric in log-exposure.

    The step is gain * (dM/d ln t) / M, clipped to max_step_stops. When most of
    the frame is clipped the CRF derivative vanishes, so a fixed step out of
    saturation/darkness is taken instead.
    """

    name = "zhang-gradient"

    def __init__(self, bounds_ms, response: ResponseCurve, gain: float = 1.0, max_step_stops: float = 0.7,
                 percentile: float = 0.5, power: float = 5.0):
        super().__init__(bounds_ms)
        self.response = response
        self.gain = gain
        self.max_step = max_step_stops * math.log(2)
        self.percentile, self.power = percentile, power

    def step(self, gray, exposure_ms):
        if np.mean(gray >= 250) > 0.5:
            return Decision(self.clip(exposure_ms * math.exp(-self.max_step)), note="saturated guard")
        if np.mean(gray <= 5) > 0.9:
            return Decision(self.clip(exposure_ms * math.exp(self.max_step)), note="dark guard")
        metric, d_metric = zhang_metric(gray, self.percentile, self.power, self.response.dpix_dlogt)
        step = float(np.clip(self.gain * d_metric / max(metric, 1e-6), -self.max_step, self.max_step))
        return Decision(self.clip(exposure_ms * math.exp(step)))


class Predictive(Controller):
    """Predict utility at candidate exposures by emulation and pick the best.

    Candidates span +/- search_stops around the captured exposure (the range in
    which single-frame emulation is trustworthy). A move is only taken when the
    predicted gain beats the captured frame by `hysteresis`, which stops
    dithering on the flat top of the utility curve. On the C920 the FAST score
    fluctuates ~3% frame to frame, so the default sits above that.
    """

    name = "predictive"

    def __init__(self, bounds_ms, response: ResponseCurve, utility_kind: str = "keypoints",
                 search_stops: float = 1.5, n_candidates: int = 13, hysteresis: float = 0.06,
                 target_mean: float = 110.0, warm_start_max_stops: float = 4.0):
        super().__init__(bounds_ms)
        self.response = response
        self.utility_kind = utility_kind
        self.stops = np.linspace(-search_stops, search_stops, n_candidates)
        self.hysteresis = hysteresis
        self.target_mean = target_mean
        self.ws_ratios = 2.0 ** np.linspace(-warm_start_max_stops, warm_start_max_stops, 65)

    def warm_start(self, gray, exposure_ms) -> float:
        """Exposure that brings the mean pixel to `target_mean`, solved through the CRF."""
        hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
        means = np.array([self.response.mean_after(hist, r) for r in self.ws_ratios])
        return self.clip(exposure_ms * self.ws_ratios[np.argmin(np.abs(means - self.target_mean))])

    def step(self, gray, exposure_ms):
        warm = self.warm_start(gray, exposure_ms)
        mean = gray.mean()
        if mean < 30 or mean > 220 or np.mean(gray >= 250) > 0.3 or np.mean(gray <= 5) > 0.5:
            return Decision(warm, warm, warm, math.nan, note="warm start")

        u_now = utility(gray, self.utility_kind)
        best_e, best_u = exposure_ms, u_now
        for s in self.stops:
            e = self.clip(exposure_ms * 2.0 ** s)
            if abs(math.log2(e / exposure_ms)) < 1e-3:
                continue
            u = utility(self.response.emulate(gray, e / exposure_ms), self.utility_kind)
            if u > best_u:
                best_e, best_u = e, u
        if best_u <= u_now * (1.0 + self.hysteresis):
            best_e, best_u = exposure_ms, u_now
        return Decision(best_e, warm, best_e, best_u)


def make_controller(name: str, bounds_ms, response: ResponseCurve, utility_kind: str = "keypoints",
                    fixed_exposure_ms: float = 10.0) -> Controller:
    if name == "zhang-gradient":
        return ZhangGradient(bounds_ms, response)
    if name == "predictive":
        return Predictive(bounds_ms, response, utility_kind=utility_kind)
    if name == "camera-auto":
        return CameraAuto(bounds_ms)
    if name == "fixed":
        return Fixed(bounds_ms, fixed_exposure_ms)
    raise ValueError(f"unknown controller {name!r}; choose from {CONTROLLERS}")
