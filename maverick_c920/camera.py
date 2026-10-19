"""UVC camera access for closed-loop exposure control (tested on a Logitech C920).

Controls are set through raw V4L2 ioctls on a second file descriptor, so they
can change while OpenCV is streaming. Every control we touch is restored on
close, so the webcam goes back to auto exposure for normal use.
"""

from __future__ import annotations

import fcntl
import glob
import os
import re
import struct
import time

import cv2
import numpy as np

VIDIOC_G_CTRL = 0xC008561B  # _IOWR('V', 27, struct v4l2_control)
VIDIOC_S_CTRL = 0xC008561C  # _IOWR('V', 28, struct v4l2_control)

CID = {
    "brightness": 0x00980900,
    "white_balance_automatic": 0x0098090C,
    "gain": 0x00980913,
    "backlight_compensation": 0x0098091C,
    "auto_exposure": 0x009A0901,
    "exposure_time_absolute": 0x009A0902,
    "exposure_dynamic_framerate": 0x009A0903,
    "focus_absolute": 0x009A090A,
    "focus_automatic_continuous": 0x009A090C,
}
AE_MANUAL = 1
AE_APERTURE_PRIORITY = 3

EXPOSURE_UNIT_MS = 0.1  # exposure_time_absolute counts 100 us steps
EXPOSURE_RAW_MIN = 3
EXPOSURE_RAW_MAX = 2047

LOGITECH_USB_VENDOR = "046d"


def find_logitech_camera() -> str:
    """Return the capture node (/dev/videoN) of the first Logitech UVC camera."""
    for node in sorted(glob.glob("/sys/class/video4linux/video*"), key=lambda p: int(re.sub(r"\D", "", p))):
        try:
            with open(os.path.join(node, "index")) as f:
                if f.read().strip() != "0":  # index 1 is the UVC metadata node
                    continue
            with open(os.path.join(node, "name")) as f:
                name = f.read().strip()
        except OSError:
            continue
        usb_dev = os.path.dirname(os.path.realpath(os.path.join(node, "device")))
        vendor = ""
        try:
            with open(os.path.join(usb_dev, "idVendor")) as f:
                vendor = f.read().strip()
        except OSError:
            pass
        if vendor == LOGITECH_USB_VENDOR or re.search(r"logitech|C9\d\d|BRIO", name, re.I):
            return "/dev/" + os.path.basename(node)
    raise RuntimeError("No Logitech UVC camera found; pass --device /dev/videoN")


class UvcCamera:
    """OpenCV capture plus V4L2 exposure control with frame-accurate settling.

    After an exposure change the C920 keeps delivering `settle_frames` frames
    exposed with the old value (measured: 2 at 30 fps). `read()` discards stale
    and in-flight frames so every returned frame was exposed at `exposure_ms`.
    """

    # Values the camera's own AE/AF loops drive; a change in them is not ours to undo.
    _CAMERA_DRIVEN = ("gain", "exposure_time_absolute", "focus_absolute")

    def __init__(self, device: str, width: int = 640, height: int = 480, fps: int = 30,
                 settle_frames: int = 2):
        self.device = device
        self.fd = os.open(device, os.O_RDWR)
        self._saved: dict[str, int] = {}
        self._order: list[str] = []
        # Snapshot before streaming: OpenCV's V4L2 backend switches autofocus off
        # when the stream starts, so values read after open are not the user's.
        self._initial: dict[str, int] = {}
        for name in CID:
            try:
                self._initial[name] = self.get(name)
            except OSError:
                pass
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            os.close(self.fd)
            raise RuntimeError(f"Cannot open {device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        self.settle_frames = settle_frames
        self._pending_settle = 0
        self.exposure_ms = float("nan")
        for _ in range(10):  # let the stream start
            self.cap.read()
        for name, value in self._initial.items():
            if name not in self._CAMERA_DRIVEN and self.get(name) != value:
                self._saved[name] = value
                self._order.append(name)

    # -- V4L2 controls -------------------------------------------------------
    def get(self, name: str) -> int:
        buf = bytearray(struct.pack("Ii", CID[name], 0))
        fcntl.ioctl(self.fd, VIDIOC_G_CTRL, buf)
        return struct.unpack("Ii", buf)[1]

    def set(self, name: str, value: int) -> None:
        if name not in self._saved:
            self._saved[name] = self._initial.get(name, self.get(name))
            self._order.append(name)
        fcntl.ioctl(self.fd, VIDIOC_S_CTRL, bytearray(struct.pack("Ii", CID[name], int(value))))

    def restore(self) -> None:
        """Put back every control we changed, newest first (values before modes)."""
        for name in reversed(self._order):
            try:
                fcntl.ioctl(self.fd, VIDIOC_S_CTRL, bytearray(struct.pack("Ii", CID[name], self._saved[name])))
            except OSError:
                pass  # e.g. focus_absolute is read-only while autofocus is on
        self._saved.clear()
        self._order.clear()

    def use_manual_exposure(self, gain: int | None = None) -> None:
        self.set("auto_exposure", AE_MANUAL)
        self.set("exposure_dynamic_framerate", 0)
        if gain is not None:
            self.set("gain", gain)

    def configure_focus(self, focus: str) -> str:
        """'auto' keeps continuous AF, 'sweep' locks the sharpest position, a number locks that value.

        Continuous AF is a confounder for exposure tests: in dim, low-contrast
        scenes the C920 parks the lens at a near position and every metric collapses.
        """
        if focus == "auto":
            return "auto"
        self.set("focus_automatic_continuous", 0)
        if focus == "sweep":
            return f"locked at {self._lock_sharpest_focus()} (sweep)"
        self.set("focus_absolute", int(focus))
        for _ in range(10):
            self.read()
        return f"locked at {int(focus)}"

    def _lock_sharpest_focus(self) -> int:
        """Coarse-to-fine search over focus_absolute for maximum Laplacian variance."""
        def sharpness(value: int) -> float:
            self.set("focus_absolute", value)
            for _ in range(8):  # lens travel
                self.read()
            return float(cv2.Laplacian(gray(self.read()), cv2.CV_64F).var())

        coarse = {f: sharpness(f) for f in range(0, 251, 25)}
        center = max(coarse, key=coarse.get)
        fine = {f: sharpness(f) for f in range(max(0, center - 20), min(250, center + 20) + 1, 5)}
        best = max(fine, key=fine.get)
        self.set("focus_absolute", best)
        for _ in range(8):
            self.read()
        return best

    def use_auto_exposure(self) -> None:
        self.set("auto_exposure", AE_APERTURE_PRIORITY)
        self._pending_settle = self.settle_frames

    def set_exposure_ms(self, exposure_ms: float) -> float:
        """Command an exposure; returns the value actually applied (100 us grid)."""
        raw = int(np.clip(round(exposure_ms / EXPOSURE_UNIT_MS), EXPOSURE_RAW_MIN, EXPOSURE_RAW_MAX))
        applied = raw * EXPOSURE_UNIT_MS
        if applied != self.exposure_ms:
            self.set("exposure_time_absolute", raw)
            self._pending_settle = self.settle_frames
        self.exposure_ms = applied
        return applied

    # -- frames ----------------------------------------------------------------
    def read(self) -> np.ndarray:
        """Next BGR frame that is fresh and fully exposed at the current setting.

        A grab that returns instantly came from the driver queue (stale); a grab
        that blocks waited for a new frame. After an exposure change we skip
        `settle_frames` new frames before returning one.
        """
        fresh_needed = self._pending_settle + 1
        blocking_s = 0.4 / self.fps
        while fresh_needed > 0:
            t0 = time.monotonic()
            if not self.cap.grab():
                raise RuntimeError(f"Frame grab failed on {self.device}")
            if time.monotonic() - t0 >= blocking_s:
                fresh_needed -= 1
        self._pending_settle = 0
        ok, frame = self.cap.retrieve()
        if not ok:
            raise RuntimeError(f"Frame decode failed on {self.device}")
        return frame

    def measure_settle_frames(self, low_ms: float, high_ms: float, trials: int = 3) -> int | None:
        """Step the exposure and count frames until brightness follows it."""
        counts = []
        for _ in range(trials):
            for start, end in ((low_ms, high_ms), (high_ms, low_ms)):
                self.set_exposure_ms(start)
                for _ in range(4):
                    self.read()
                before = gray(self.read()).mean()
                self.set_exposure_ms(end)
                self._pending_settle = 0
                means = [gray(self.read()).mean() for _ in range(8)]
                after = means[-1]
                if abs(after - before) < 8:
                    return None  # scene too dark or flat to see the step
                crossed = [i for i, m in enumerate(means) if abs(m - before) > 0.5 * abs(after - before)]
                counts.append(crossed[0])
        self.settle_frames = max(counts)
        return self.settle_frames

    def close(self) -> None:
        try:
            self.restore()
        finally:
            self.cap.release()
            os.close(self.fd)

    def __enter__(self) -> "UvcCamera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
