"""Static-frame convergence test on a live UVC camera.

With the camera and scene held still:
  1. measure how many frames an exposure change takes to show up,
  2. sweep exposure to find the oracle (best-utility) exposure for this scene,
  3. calibrate the camera response from the sweep and check emulation error,
  4. run each controller closed-loop from a dark and a bright start and report
     how fast and how close to the oracle it settles.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time

import cv2
import numpy as np

from .camera import AE_MANUAL, UvcCamera, gray
from .controllers import make_controller
from .metrics import frame_metrics
from .response import ResponseCurve, emulation_error

ORACLE_METRICS = ("keypoints", "gradient", "zhang")
LOW_TEXTURE_UTILITY = 0.02

TRAJECTORY_FIELDS = [
    "controller", "start", "step", "t_s", "exposure_ms", "gain",
    "warm_start_exposure_ms", "predicted_exposure_ms", "predicted_utility",
    "predicted_utility_for_this_frame", "utility",
    "keypoints", "gradient", "zhang", "mean_intensity", "frac_saturated", "frac_dark", "note",
]


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sweep(cam: UvcCamera, exposures_ms, frames_per_exposure: int, out_dir: str):
    os.makedirs(os.path.join(out_dir, "sweep"), exist_ok=True)
    rows, frames = [], []
    for e in exposures_ms:
        applied = cam.set_exposure_ms(e)
        shots = [cam.read() for _ in range(frames_per_exposure)]
        per_frame = [frame_metrics(gray(s)) for s in shots]
        row = {"exposure_ms": applied}
        row.update({k: float(np.mean([m[k] for m in per_frame])) for k in per_frame[0]})
        rows.append(row)
        frames.append(shots[0])
        cv2.imwrite(os.path.join(out_dir, "sweep", f"exp_{applied:07.2f}ms.jpg"), shots[0])
    return rows, frames


def choose_response(frames, sweep_rows, crf_path: str | None):
    exps = [r["exposure_ms"] for r in sweep_rows]
    grays = [gray(f) for f in frames]
    candidates = {"gamma-2.2": ResponseCurve.gamma(2.2)}
    if crf_path:
        candidates["loaded"] = ResponseCurve.load(crf_path)
    usable = [i for i, g in enumerate(grays) if 3 < g.mean() < 250]
    if len(usable) >= 5:
        try:
            candidates["debevec"] = ResponseCurve.calibrate_debevec([frames[i] for i in usable],
                                                                    [exps[i] for i in usable])
        except cv2.error as exc:
            print(f"  Debevec calibration failed ({exc}); keeping gamma")
    errors = {name: emulation_error(c, grays, exps) for name, c in candidates.items()}
    best = min(errors, key=lambda k: errors[k] if not math.isnan(errors[k]) else 1e9)
    return candidates[best], errors, best


def oracle(sweep_rows, band: float = 0.95):
    out = {}
    exps = np.array([r["exposure_ms"] for r in sweep_rows])
    for metric in ORACLE_METRICS:
        values = np.array([r[metric] for r in sweep_rows])
        i = int(np.argmax(values))
        near = exps[values >= band * values[i]]
        out[metric] = {"exposure_ms": float(exps[i]), "utility": float(values[i]),
                       "band_ms": [float(near.min()), float(near.max())]}
    return out


def run_controller(cam: UvcCamera, controller, start_ms: float, steps: int, utility_kind: str,
                   gain: int | None = None):
    rows = []
    cam.set("auto_exposure", AE_MANUAL)
    if gain is not None:
        cam.set("gain", gain)  # camera AE moves gain; give every run the same start
    cam.exposure_ms = math.nan  # force the next write even if the value is unchanged
    exposure = cam.set_exposure_ms(start_ms)
    for _ in range(3):
        cam.read()
    controller.reset()
    if controller.camera_driven:
        cam.use_auto_exposure()
    predicted_for_this = math.nan
    t0 = time.monotonic()
    for k in range(steps):
        g = gray(cam.read())
        m = frame_metrics(g)
        if controller.camera_driven:
            exposure = cam.get("exposure_time_absolute") * 0.1
        decision = controller.step(g, exposure)
        rows.append({
            "controller": controller.name, "start": start_ms, "step": k, "t_s": time.monotonic() - t0,
            "exposure_ms": exposure, "gain": cam.get("gain"),
            "warm_start_exposure_ms": decision.warm_start_exposure_ms,
            "predicted_exposure_ms": decision.predicted_exposure_ms,
            "predicted_utility": decision.predicted_utility,
            "predicted_utility_for_this_frame": predicted_for_this,
            "utility": m[utility_kind], **m, "note": decision.note,
        })
        predicted_for_this = decision.predicted_utility
        if not controller.camera_driven:
            exposure = cam.set_exposure_ms(decision.next_exposure_ms)
    cam.set("auto_exposure", AE_MANUAL)
    return rows


def summarize_run(rows, oracle_utility: float, oracle_exposure_ms: float, tail: int = 5):
    exps = np.array([r["exposure_ms"] for r in rows])
    utils = np.array([r["utility"] for r in rows])
    times = np.array([r["t_s"] for r in rows])
    final_exp = float(np.median(exps[-tail:]))
    within = np.abs(np.log2(exps / final_exp)) <= 0.25
    settle = len(rows) - int(np.argmin(within[::-1])) if not within.all() else 0
    settle = min(settle, len(rows) - 1)
    reach = np.nonzero(utils >= 0.9 * oracle_utility)[0]
    pred = np.array([r["predicted_utility_for_this_frame"] for r in rows], dtype=float)
    ok = ~np.isnan(pred)
    return {
        "final_exposure_ms": final_exp,
        "error_vs_oracle_stops": float(math.log2(final_exp / oracle_exposure_ms)),
        "final_utility": float(utils[-tail:].mean()),
        "utility_ratio": float(utils[-tail:].mean() / oracle_utility) if oracle_utility > 0 else math.nan,
        "settle_step": int(settle),
        "settle_time_s": float(times[settle]),
        "first_step_at_90pct": int(reach[0]) if len(reach) else None,
        "first_time_at_90pct_s": float(times[reach[0]]) if len(reach) else None,
        "prediction_mae": float(np.abs(pred[ok] - utils[ok]).mean()) if ok.any() else None,
        "final_gain": int(rows[-1]["gain"]),
    }


def run_static_test(args) -> dict:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.out, f"static_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = args.min_exposure_ms, args.max_exposure_ms

    with UvcCamera(args.device, args.width, args.height, args.fps) as cam:
        cam.use_auto_exposure()  # let the camera's AE light the scene for the focus search
        focus = cam.configure_focus(args.focus)
        cam.use_manual_exposure(gain=args.gain)
        print(f"[1/4] latency  device={args.device} gain={cam.get('gain')} focus {focus}")
        settle = cam.measure_settle_frames(hi / 8, hi / 2)
        print(f"      exposure changes take effect after {cam.settle_frames} frames"
              + ("" if settle is not None else " (default; scene too dark to measure)"))

        print(f"[2/4] sweep    {args.sweep_points} exposures {lo:.1f}-{hi:.1f} ms")
        exposures = np.geomspace(lo, hi, args.sweep_points)
        sweep_rows, frames = sweep(cam, exposures, args.frames_per_exposure, out_dir)
        _write_csv(os.path.join(out_dir, "sweep.csv"), sweep_rows, list(sweep_rows[0]))
        best = oracle(sweep_rows)
        for metric, o in best.items():
            print(f"      oracle[{metric:9s}] = {o['exposure_ms']:6.1f} ms  utility={o['utility']:.4f}"
                  f"  95% band {o['band_ms'][0]:.1f}-{o['band_ms'][1]:.1f} ms")
        if best[args.utility]["utility"] < LOW_TEXTURE_UTILITY:
            print(f"      WARNING: best {args.utility} utility is {best[args.utility]['utility']:.4f}; the scene has"
                  " almost no texture at any exposure. Point the camera at a textured scene (ideally with a"
                  " window or lamp in view) for a meaningful test.")

        print("[3/4] response curve")
        response, crf_errors, crf_name = choose_response(frames, sweep_rows, args.crf)
        response.save(os.path.join(out_dir, "crf.json"))
        for name, err in crf_errors.items():
            print(f"      emulation MAE {name:10s} {err:5.2f} DN" + ("  <- used" if name == crf_name else ""))

        print(f"[4/4] closed loop  {args.steps} steps per run, objective={args.utility}")
        oracle_u = best[args.utility]
        all_rows, summary_runs = [], []
        starts = args.starts or [lo, hi]
        for name in args.controllers:
            controller = make_controller(name, (lo, hi), response, args.utility)
            for start in starts:
                rows = run_controller(cam, controller, start, args.steps, args.utility, args.gain)
                s = summarize_run(rows, oracle_u["utility"], oracle_u["exposure_ms"])
                s.update(controller=name, start_ms=start)
                summary_runs.append(s)
                all_rows.extend(rows)
                cv2.imwrite(os.path.join(out_dir, f"final_{name}_start{start:g}ms.jpg"), cam.read())
                print(f"      {name:15s} start {start:5.1f} ms -> {s['final_exposure_ms']:6.1f} ms "
                      f"({s['error_vs_oracle_stops']:+.2f} stops)  utility {s['utility_ratio']*100:5.1f}% of oracle"
                      f"  settled step {s['settle_step']} ({s['settle_time_s']:.2f}s)"
                      + (f"  [AE gain {s['final_gain']}]" if controller.camera_driven else ""))
        _write_csv(os.path.join(out_dir, "trajectories.csv"), all_rows, TRAJECTORY_FIELDS)

    summary = {
        "device": args.device, "focus": focus, "resolution": [args.width, args.height], "fixed_gain": args.gain,
        "exposure_bounds_ms": [lo, hi], "settle_frames": cam.settle_frames, "objective": args.utility,
        "oracle": best, "response_curve": response.source, "emulation_mae_dn": crf_errors, "runs": summary_runs,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    try:
        from .plots import plot_static
        plot_static(sweep_rows, best, response, all_rows, args.utility, os.path.join(out_dir, "static_test.png"))
    except ImportError as exc:
        print(f"      (no plot: {exc})")
    print(f"results: {out_dir}")
    summary["out_dir"] = out_dir
    return summary
