# Logitech C920 experiments

Two packages live here:
- **[s4vy_lite](#s4vy_lite-4d-instance-segmentation-on-the-webcam-cpu)**: a CPU stand-in for S4VY (4D segment-anything).
- **[maverick_c920](#maverick_c920-live-exposure-control-tests-on-a-logitech-webcam)**: closed-loop auto-exposure tests.

## s4vy_lite: 4D instance segmentation on the webcam (CPU)

[S4VY](https://evergreen0929.github.io/projects/s4vy.html) (Zhang et al., arXiv:2609.36875)
segments every object in 4D from any set of RGB views. Each persistent object query
tracks one entity across all views, and a point or box selects one instance.

Its code and weights are unreleased (the repo says "coming soon"). It is also sized for
an A100: a VGGT-Ω-1B backbone with rank-256 LoRA, 10.1 s per 100 frames. `s4vy_lite`
reproduces the same pipeline with open models that run on this laptop's CPU:

| S4VY | s4vy_lite |
|---|---|
| VGGT-Ω feed-forward 4D geometry | Depth Anything 3 (DA3-BASE): depth and camera poses from any set of views, run in fp32 (3× faster than its fp16 autocast on this CPU) |
| Space-time query decoder: exhaustive, class-agnostic instances | SAM 2.1-tiny automatic masks on keyframes, **merged across keyframes by 3D voxel overlap** (identity anchored in geometry, not frame order), then tracked through every frame with SAM 2.1, forward and backward |
| Promptable query (point or box) | SAM 2.1 point or box on any frame, propagated both ways and lifted to 4D; a click resolves to the whole object (`--granularity whole`) |
| Agentic language grounding (Qwen3-VL) | not included |

The main difference from S4VY: SAM 2 tracks frame to frame, so input order still matters.
It is also far less accurate than a model trained on 6M annotated frames.

Run it with the `scarf-slam` conda env, which has torch-cpu and Depth Anything 3.
SAM 2 is vendored in `third_party/` from the official repo, so the env is not modified.

```bash
PY=~/miniconda3/envs/scarf-slam/bin/python

# Capture 10 frames 0.5 s apart from the Logitech, then segment them in 4D (~3.5 min).
# Move the camera slowly around the scene while it captures: DA3 needs parallax for 3D.
$PY -m s4vy_lite run --capture --display
$PY -m s4vy_lite run --input some_video.mp4 --frames 12

# Select one instance with a point or box (pixel coords in RUN/frames/*.jpg) and track it in 4D
$PY -m s4vy_lite prompt runs/s4vy_capture --point 285,350 --frame 5
$PY -m s4vy_lite prompt runs/s4vy_capture --box 0,255,178,378 --frame 5

$PY -m s4vy_lite view runs/s4vy_capture          # Open3D viewer (instance colors; --rgb for real colors)
```

Each run directory contains:
- `overlay.mp4` and `overlay_grid.jpg`: masks with persistent IDs.
- `novel_view.jpg`: the fused 4D point cloud seen from a camera orbited 25°.
- `instances.ply` and `scene_rgb.ply`: per-point `instance` and `frame` fields.
- `instances.json`: per-instance visibility, 3D extent, motion and stage timings.
- `geometry.npz` and `id_maps.npz`: the raw outputs.

Measured on the i3-N305 with 10 frames at 504×378:
- Run time: DA3 takes about 40 s, keyframe masks about 22 s, and tracking 12 instances 128 s.
- Peak RAM is 2.6 GB.
- A prompt takes about 15 s.

CPU workarounds, all applied at runtime without patching the vendored code:
- SAM 2.1 runs at 512 px instead of 1024, which made tracking 8.7× faster.
- The image predictor's 1024 feature sizes are overridden.
- The bfloat16 memory bank is cast to fp32 on CPU.

# maverick_c920: live exposure-control tests on a Logitech webcam

A port of the BorealHDR closed-loop exposure benchmark (`maverick/scripts/borealhdr_sequence.py`)
from offline emulation to a real UVC camera. It was written and tested on a Logitech
C920 (`046d:082d`). It picks exposure to maximize how useful a frame is for vision,
not how good it looks.

| Offline (maverick, BorealHDR) | Live (this package) |
|---|---|
| `BorealRadianceEmulator` over raw brackets | `response.py`: Debevec CRF from a live sweep; re-exposes any frame through a 256-entry LUT |
| R2D2 keypoint utility | `metrics.py`: FAST keypoints capped per grid cell (coverage), Shim gradient information |
| `--controller zhang-…` | `zhang-gradient`: gradient ascent on Zhang et al.'s soft-percentile metric, with dM/d ln t from the CRF |
| surrogate / exposure optimization | `predictive`: emulate the frame at ±1.5 stops, predict utility, jump to the best (mid-grey warm start when too clipped) |
| static-frame convergence test | `python3 -m maverick_c920 static` |
| trajectory replay | `python3 -m maverick_c920 live` (move the camera) |
| `FRAME_FIELDS` | same per-frame fields in the CSVs, plus `predicted_utility_for_this_frame` and KLT `track_survival` |

## Run

Requirements: Python 3.10+, `numpy`, `opencv-python`, `matplotlib`. The camera is found automatically.

```bash
# Static-frame convergence test: keep the camera and scene still (~30 s)
python3 -m maverick_c920 static
python3 -m maverick_c920 static --gain 255 --steps 40

# Closed loop on the live stream, with a preview window; press q to stop
python3 -m maverick_c920 live --controller predictive --display
python3 -m maverick_c920 live --controller zhang-gradient --display --record zhang.mp4
python3 -m maverick_c920 live --controller camera-auto --display      # baseline: the camera's own AE
```

Results go to `runs/`. A static run writes `summary.json`, `sweep.csv`, `trajectories.csv`,
`crf.json`, `static_test.png` and the sweep frames. Live mode reuses the newest `crf.json`.

On exit, every camera control is restored, including autofocus, which OpenCV turns off
when it starts streaming. The webcam goes back to normal for video calls.

## C920 facts measured here

- `exposure_time_absolute` is in 100 µs units, from 3 to 2047, and responds continuously (no quantization).
- An exposure change appears **2 frames** later. `UvcCamera.read()` drops stale and in-flight frames, so every frame a controller sees was exposed at the value it commanded. That gives a control rate of about 12 Hz at 30 fps.
- Exposures above 33.3 ms lower the frame rate, even with `exposure_dynamic_framerate=0`. That's why 33.3 ms is the default ceiling.
- In auto mode the camera's AE also raises **gain**. The manual controllers hold gain fixed (`--gain`), so camera-auto is not a like-for-like comparison.
- In dim, low-contrast scenes, continuous autofocus parks the lens at a near position and every metric collapses. By default focus is locked at the sharpest position (`--focus sweep`).

## Getting a test that separates the controllers

The test only discriminates when the best exposure falls *inside* the exposure range.
In a dim room with no light sources, the best exposure is the 33.3 ms ceiling and every
controller "wins" by maxing out. Point the camera at a textured scene with a **window
or lamp in frame**, so long exposures clip highlights and short ones lose the shadows.
For live mode, pan between the bright and dark parts of the room.
