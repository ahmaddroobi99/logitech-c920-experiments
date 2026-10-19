"""S4VY-lite: exhaustive + promptable 4D instance segmentation on a Logitech webcam (CPU).

  python -m s4vy_lite run --capture            # grab a clip, then segment it in 4D
  python -m s4vy_lite run --input clip.mp4     # or a folder of images
  python -m s4vy_lite prompt RUN --point 250,190 --frame 0
  python -m s4vy_lite prompt RUN --box 40,200,230,370 --frame 3
  python -m s4vy_lite view RUN                 # open the 4D point cloud (Open3D)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from .capture import capture, load_frames
from .geometry import Geometry, estimate_geometry
from .lift import fused_points, instance_stats, write_ply
from .segment import (associate_in_3d, build_mask_generator, build_video_predictor, id_maps_from_logits,
                      prompted_track, select_masks, track_instances)
from .visualize import palette, render_novel_view, write_overlays

CONF_PERCENTILE = 20.0  # drop the least-confident 20% of DA3 depth before lifting to 3D


class Timer:
    def __init__(self):
        self.t0 = time.monotonic()
        self.marks = {}

    def lap(self, name: str) -> float:
        now = time.monotonic()
        self.marks[name] = round(now - self.t0, 1)
        self.t0 = now
        return self.marks[name]


def _write_frames(images: np.ndarray, frames_dir: str) -> None:
    os.makedirs(frames_dir, exist_ok=True)
    for t, img in enumerate(images):
        cv2.imwrite(os.path.join(frames_dir, f"{t:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])


def cmd_run(args) -> None:
    torch.set_num_threads(os.cpu_count() or 4)
    out = args.out or os.path.join("runs", time.strftime("s4vy_%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    timer = Timer()

    if args.capture:
        capture(os.path.join(out, "input"), args.frames, args.interval, args.device, display=args.display)
        source = os.path.join(out, "input")
    else:
        source = args.input
    frames = load_frames(source, args.frames)
    print(f"[1/5] {len(frames)} frames from {source}  ({timer.lap('load')}s)")

    geom = estimate_geometry(frames, args.geometry_model, args.process_res)
    geom.save(os.path.join(out, "geometry.npz"))
    frames_dir = os.path.join(out, "frames")
    _write_frames(geom.images, frames_dir)
    valid = geom.valid_mask(CONF_PERCENTILE)
    scene_scale = float(np.median(geom.depth[valid]))
    voxel = args.voxel * scene_scale
    cams = np.array([np.linalg.inv(m)[:3, 3] for m in geom.w2c])
    baseline = float(np.linalg.norm(cams - cams[0], axis=1).max() / scene_scale)
    print(f"[2/5] geometry {geom.depth.shape[1]}x{geom.depth.shape[2]}, camera baseline "
          f"{baseline:.3f} x scene depth  ({timer.lap('geometry')}s)")
    if baseline < 0.02:
        print("      note: the camera barely moved, so this is a depth video from one viewpoint, not a"
              " multi-view reconstruction. Move the camera while capturing for real 3D.")

    shape = geom.depth.shape[1:]
    amg = build_mask_generator(args.points_per_side)
    keyframes = sorted(set(np.linspace(0, geom.num_frames - 1, args.keyframes).round().astype(int).tolist()))
    keyframe_masks = {}
    with torch.inference_mode():
        for f in keyframes:
            keyframe_masks[f] = select_masks(amg.generate(geom.images[f]), shape)
    del amg
    instances = associate_in_3d(keyframe_masks, geom, valid, voxel)
    instances.sort(key=lambda inst: -inst.area)
    instances = instances[: args.max_objects]
    n_masks = sum(len(v) for v in keyframe_masks.values())
    print(f"[3/5] {n_masks} masks on keyframes {keyframes} -> {len(instances)} instances after 3D association"
          f"  ({timer.lap('keyframe_masks')}s)")

    predictor = build_video_predictor()
    logits = track_instances(predictor, frames_dir, instances, shape)
    id_maps, kept = id_maps_from_logits(logits, min_pixels=args.min_pixels)
    np.savez_compressed(os.path.join(out, "id_maps.npz"), id_maps=id_maps)
    print(f"[4/5] tracked {len(instances)} instances through {geom.num_frames} frames -> {int(id_maps.max())} kept"
          f"  ({timer.lap('tracking')}s)")

    stats = instance_stats(geom, id_maps, valid)
    for s, old in zip(stats, kept):
        s["seed_frames"] = sorted(x.frame for x in instances[old].seeds)
    colors = palette(int(id_maps.max()))
    xyz, rgb, ids, ts = fused_points(geom, id_maps, valid, voxel=args.voxel * 0.25 * scene_scale)
    write_ply(os.path.join(out, "scene_rgb.ply"), xyz, rgb, ids, ts)
    write_ply(os.path.join(out, "instances.ply"), xyz, np.where(ids[:, None] > 0, colors[ids], 160).astype(np.uint8),
              ids, ts)
    write_overlays(geom.images, id_maps, colors, os.path.join(out, "overlay.mp4"), os.path.join(out, "overlay_grid.jpg"))
    inst_rgb = np.where(ids[:, None] > 0, colors[ids], 200).astype(np.uint8)
    nv = [render_novel_view(xyz, c, geom.K[0], geom.w2c[0], shape) for c in (rgb, inst_rgb)]
    cv2.imwrite(os.path.join(out, "novel_view.jpg"), np.hstack(nv))
    timer.lap("export")
    summary = {"source": source, "frames": geom.num_frames, "keyframes": keyframes, "camera_baseline_rel": baseline,
               "scene_scale": scene_scale, "timings_s": timer.marks, "instances": stats}
    with open(os.path.join(out, "instances.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[5/5] 4D instances  ({timer.marks['export']}s)")
    print("      id  frames  seeds     px/frame  size  motion   (size, motion in units of scene depth)")
    for s in stats:
        if s["frames_visible"]:
            print(f"      {s['id']:2d}  {s['frames_visible']:3d}/{geom.num_frames:<3d} {str(s['seed_frames']):9s}"
                  f" {s['mean_pixel_area']:8.0f}  {s['size_rel_scene']:4.2f}  {s['motion_rel_scene']:5.3f}")
    print(f"total {sum(timer.marks.values()):.0f}s -> {out}")


def cmd_prompt(args) -> None:
    torch.set_num_threads(os.cpu_count() or 4)
    geom = Geometry.load(os.path.join(args.run, "geometry.npz"))
    valid = geom.valid_mask(CONF_PERCENTILE)
    point = [float(v) for v in args.point.split(",")] if args.point else None
    box = [float(v) for v in args.box.split(",")] if args.box else None
    t0 = time.monotonic()
    logits = prompted_track(build_video_predictor(), os.path.join(args.run, "frames"), args.frame, point, box,
                            image_rgb=geom.images[args.frame], granularity=args.granularity)
    masks = (logits > 0).astype(np.int16)
    tag = f"point{int(point[0])}_{int(point[1])}" if point else "box" + "_".join(str(int(v)) for v in box)
    out = os.path.join(args.run, f"prompt_f{args.frame}_{tag}")
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "masks.npz"), masks=masks.astype(bool))
    found = instance_stats(geom, masks, valid)
    if not found:
        raise SystemExit("the prompt selected nothing; try a point inside an object")
    stats = found[0]

    # Which exhaustive instance did the prompt pick? (S4VY: the prompt query matches an object query)
    match = None
    id_path = os.path.join(args.run, "id_maps.npz")
    if os.path.exists(id_path):
        id_maps = np.load(id_path)["id_maps"]
        sel = masks.astype(bool)
        best = (-1.0, 0)
        for iid in range(1, int(id_maps.max()) + 1):
            m = id_maps == iid
            iou = (m & sel).sum() / max((m | sel).sum(), 1)
            best = max(best, (float(iou), iid))
        match = {"instance": best[1], "iou_4d": round(best[0], 3)}

    colors = palette(1)
    write_overlays(geom.images, masks, colors, os.path.join(out, "overlay.mp4"), os.path.join(out, "overlay_grid.jpg"))
    xyz, rgb, ids, ts = fused_points(geom, masks, valid, voxel=0.005 * float(np.median(geom.depth[valid])))
    sel = ids > 0
    write_ply(os.path.join(out, "instance.ply"), xyz[sel], rgb[sel], ids[sel], ts[sel])
    with open(os.path.join(out, "instance.json"), "w") as f:
        json.dump({"frame": args.frame, "point": point, "box": box, "stats": stats, "matches_exhaustive": match},
                  f, indent=2)
    print(f"prompted instance visible in {stats['frames_visible']}/{geom.num_frames} frames, "
          f"{stats.get('mean_pixel_area', 0):.0f} px/frame"
          + (f"; matches exhaustive instance {match['instance']} (4D IoU {match['iou_4d']})" if match else "")
          + f"  ({time.monotonic() - t0:.0f}s) -> {out}")


def cmd_view(args) -> None:
    import open3d as o3d

    path = os.path.join(args.run, "scene_rgb.ply" if args.rgb else "instances.ply")
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd], window_name=f"S4VY-lite: {path}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="s4vy_lite", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="exhaustive 4D instance segmentation of a clip")
    src = run.add_mutually_exclusive_group(required=True)
    src.add_argument("--capture", action="store_true", help="grab frames from the Logitech webcam first")
    src.add_argument("--input", help="folder of images or a video file")
    run.add_argument("--device", help="camera node for --capture (default: first Logitech camera)")
    run.add_argument("--display", action="store_true", help="show the camera while capturing")
    run.add_argument("--frames", type=int, default=10, help="frames to capture / subsample to")
    run.add_argument("--interval", type=float, default=0.5, help="seconds between captured frames")
    run.add_argument("--keyframes", type=int, default=3, help="frames that seed instances")
    run.add_argument("--max-objects", type=int, default=20)
    run.add_argument("--points-per-side", type=int, default=16, help="SAM automatic-mask grid density")
    run.add_argument("--min-pixels", type=int, default=400, help="drop instances smaller than this over the clip")
    run.add_argument("--voxel", type=float, default=0.03, help="3D association voxel, fraction of median depth")
    run.add_argument("--process-res", type=int, default=504, help="DA3 processing resolution (long side)")
    run.add_argument("--geometry-model", default="depth-anything/DA3-BASE")
    run.add_argument("--out", help="run directory (default runs/s4vy_<timestamp>)")
    run.set_defaults(func=cmd_run)

    pr = sub.add_parser("prompt", help="select one instance with a point or box, track it in 4D")
    pr.add_argument("run", help="run directory from `run`")
    g = pr.add_mutually_exclusive_group(required=True)
    g.add_argument("--point", help="x,y in the run's frames/ images (processing resolution)")
    g.add_argument("--box", help="x0,y0,x1,y1 in the run's frames/ images")
    pr.add_argument("--frame", type=int, default=0)
    pr.add_argument("--granularity", choices=("whole", "best"), default="whole",
                    help="for --point: largest plausible object (whole) or SAM's top-scoring mask (best)")
    pr.set_defaults(func=cmd_prompt)

    vw = sub.add_parser("view", help="open the 4D point cloud in Open3D")
    vw.add_argument("run")
    vw.add_argument("--rgb", action="store_true", help="real colors instead of instance colors")
    vw.set_defaults(func=cmd_view)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
