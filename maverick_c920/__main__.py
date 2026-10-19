"""python3 -m maverick_c920 {static,live} ... -- see README.md."""

from __future__ import annotations

import argparse

from .camera import EXPOSURE_RAW_MAX, EXPOSURE_RAW_MIN, EXPOSURE_UNIT_MS, find_logitech_camera
from .controllers import CONTROLLERS
from .metrics import UTILITIES


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", help="V4L2 capture node; default: first Logitech camera")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--gain", type=int, default=0, help="fixed sensor gain for manual control (C920: 0-255)")
    p.add_argument("--focus", default="sweep",
                   help="'sweep' (lock sharpest, default), 'auto' (continuous AF) or a value 0-250 (C920, step 5)")
    p.add_argument("--min-exposure-ms", type=float, default=EXPOSURE_RAW_MIN * EXPOSURE_UNIT_MS)
    p.add_argument("--max-exposure-ms", type=float, default=33.3,
                   help="33.3 keeps 30 fps; the C920 accepts up to %.1f but drops frame rate"
                   % (EXPOSURE_RAW_MAX * EXPOSURE_UNIT_MS))
    p.add_argument("--utility", choices=UTILITIES, default="keypoints", help="objective / evaluation metric")
    p.add_argument("--crf", help="crf.json from a previous static run (default: calibrate or gamma 2.2)")
    p.add_argument("--out", default="runs")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="maverick_c920", description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    st = sub.add_parser("static", help="static-frame convergence test (hold camera and scene still)")
    _common(st)
    st.add_argument("--controllers", nargs="+", choices=CONTROLLERS,
                    default=["zhang-gradient", "predictive", "camera-auto"])
    st.add_argument("--starts", nargs="+", type=float, help="start exposures in ms (default: min and max)")
    st.add_argument("--steps", type=int, default=30)
    st.add_argument("--sweep-points", type=int, default=20)
    st.add_argument("--frames-per-exposure", type=int, default=3)

    lv = sub.add_parser("live", help="closed-loop control on the live stream (move the camera)")
    _common(lv)
    lv.add_argument("--controller", choices=CONTROLLERS, default="predictive")
    lv.add_argument("--seconds", type=float, default=60)
    lv.add_argument("--start-exposure-ms", type=float, default=10.0)
    lv.add_argument("--exposure-ms", type=float, default=10.0, help="exposure for --controller fixed")
    lv.add_argument("--display", action="store_true", help="show a preview window with keypoints")
    lv.add_argument("--record", help="write the annotated preview to this .mp4")

    args = parser.parse_args(argv)
    args.device = args.device or find_logitech_camera()
    if args.mode == "static":
        from .static_test import run_static_test
        run_static_test(args)
    else:
        from .live import run_live
        run_live(args)


if __name__ == "__main__":
    main()
