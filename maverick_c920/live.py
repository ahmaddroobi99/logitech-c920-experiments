"""Closed-loop exposure control on the live camera stream (the 'sequence' mode).

Move the camera through the scene (walk past a window, point at a lamp and
away) while a controller runs; every frame is logged with the same fields as
the offline BorealHDR runs, plus KLT track survival between frames.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import time

import cv2
import numpy as np

from .camera import UvcCamera, gray
from .controllers import make_controller
from .metrics import frame_metrics, keypoint_points, track_survival
from .response import ResponseCurve

FRAME_FIELDS = [
    "frame", "t_s", "controller", "exposure_ms", "gain",
    "warm_start_exposure_ms", "predicted_exposure_ms", "predicted_utility",
    "predicted_utility_for_this_frame", "utility",
    "keypoints", "gradient", "zhang", "mean_intensity", "frac_saturated", "frac_dark",
    "track_survival", "note",
]


def _overlay(frame, row, objective):
    for x, y in keypoint_points(gray(frame)):
        cv2.circle(frame, (int(x), int(y)), 2, (52, 104, 235), -1)
    lines = [
        f"{row['controller']}  exp {row['exposure_ms']:.1f} ms  gain {row['gain']}",
        f"{objective} {row['utility']:.3f}  track {row['track_survival']:.2f}  mean {row['mean_intensity']:.0f}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _latest_crf(runs_dir: str) -> str | None:
    """crf.json from the newest static run, so live mode uses the calibrated response."""
    found = sorted(glob.glob(os.path.join(runs_dir, "static_*", "crf.json")))
    return found[-1] if found else None


def run_live(args) -> str:
    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(args.out, f"live_{args.controller}_{stamp}.csv")
    crf_path = args.crf or _latest_crf(args.out)
    response = ResponseCurve.load(crf_path) if crf_path else ResponseCurve.gamma(2.2)
    lo, hi = args.min_exposure_ms, args.max_exposure_ms
    controller = make_controller(args.controller, (lo, hi), response, args.utility, args.exposure_ms)
    print(f"controller={controller.name} objective={args.utility} crf={crf_path or response.source} "
          f"bounds={lo:.1f}-{hi:.1f} ms; Ctrl-C" + (" or q in the window" if args.display else "") + " to stop")

    writer_video = None
    rows = []
    with UvcCamera(args.device, args.width, args.height, args.fps) as cam:
        cam.use_auto_exposure()
        print(f"focus {cam.configure_focus(args.focus)}")
        if not controller.camera_driven:
            cam.use_manual_exposure(gain=args.gain)
        exposure = cam.set_exposure_ms(args.start_exposure_ms) if not controller.camera_driven else math.nan
        prev_g, predicted_for_this = None, math.nan
        t0 = time.monotonic()
        try:
            while time.monotonic() - t0 < args.seconds:
                frame = cam.read()
                g = gray(frame)
                if controller.camera_driven:
                    exposure = cam.get("exposure_time_absolute") * 0.1
                m = frame_metrics(g)
                decision = controller.step(g, exposure)
                row = {
                    "frame": len(rows), "t_s": time.monotonic() - t0, "controller": controller.name,
                    "exposure_ms": exposure, "gain": cam.get("gain"),
                    "warm_start_exposure_ms": decision.warm_start_exposure_ms,
                    "predicted_exposure_ms": decision.predicted_exposure_ms,
                    "predicted_utility": decision.predicted_utility,
                    "predicted_utility_for_this_frame": predicted_for_this,
                    "utility": m[args.utility], **m,
                    "track_survival": track_survival(prev_g, g) if prev_g is not None else math.nan,
                    "note": decision.note,
                }
                rows.append(row)
                prev_g, predicted_for_this = g, decision.predicted_utility
                if not controller.camera_driven:
                    exposure = cam.set_exposure_ms(decision.next_exposure_ms)
                if args.display or args.record:
                    vis = _overlay(frame.copy(), row, args.utility)
                    if args.record:
                        if writer_video is None:
                            h, w = vis.shape[:2]
                            writer_video = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                        writer_video.write(vis)
                    if args.display:
                        cv2.imshow("maverick-c920", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        except KeyboardInterrupt:
            pass
        finally:
            if writer_video is not None:
                writer_video.release()
            if args.display:
                cv2.destroyAllWindows()

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FRAME_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    if rows:
        util = np.array([r["utility"] for r in rows])
        track = np.array([r["track_survival"] for r in rows], dtype=float)
        exps = np.array([r["exposure_ms"] for r in rows], dtype=float)
        dur = rows[-1]["t_s"]
        print(f"{len(rows)} frames in {dur:.1f}s ({len(rows) / max(dur, 1e-6):.1f} Hz control)")
        print(f"mean {args.utility} utility {util.mean():.4f}   mean track survival {np.nanmean(track):.3f}"
              f"   exposure median {np.nanmedian(exps):.1f} ms (range {np.nanmin(exps):.1f}-{np.nanmax(exps):.1f})")
    print(f"log: {csv_path}")
    return csv_path
