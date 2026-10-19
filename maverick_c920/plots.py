"""Static-test figure: oracle sweep, response curve, exposure and utility trajectories."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
CONTROLLER_COLORS = {"zhang-gradient": "#2a78d6", "predictive": "#eb6834", "camera-auto": "#1baf7a",
                     "fixed": "#6250d6"}
METRIC_STYLES = {"keypoints": "-", "gradient": "--", "zhang": ":"}


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def plot_static(sweep_rows, oracle, response, trajectory_rows, objective, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), facecolor=SURFACE)
    (a_sweep, a_crf), (a_exp, a_util) = axes

    exps = np.array([r["exposure_ms"] for r in sweep_rows])
    _style(a_sweep, "Oracle sweep: utility vs exposure (each metric / its max)")
    for metric, ls in METRIC_STYLES.items():
        v = np.array([r[metric] for r in sweep_rows])
        a_sweep.plot(exps, v / (v.max() or float("nan")), ls, color=INK if metric == objective else INK_2, linewidth=2,
                     marker="o", markersize=4, label=metric + (" (objective)" if metric == objective else ""))
    lo, hi = oracle[objective]["band_ms"]
    a_sweep.axvspan(lo, hi, color=GRID, alpha=0.7, linewidth=0, label=f"{objective} within 95% of best")
    a_sweep.set_xscale("log")
    a_sweep.set_xlabel("exposure (ms)")
    a_sweep.set_ylabel("relative utility")
    a_sweep.legend(frameon=False, fontsize=9, labelcolor=INK_2)

    _style(a_crf, f"Inverse camera response ({response.source})")
    a_crf.plot(np.arange(256), np.log2(response.inv), color=INK, linewidth=2)
    a_crf.set_xlabel("pixel value")
    a_crf.set_ylabel("log2 relative exposure (stops from pixel 128)")

    _style(a_exp, "Closed loop: exposure (solid = dark start, dashed = bright start)")
    _style(a_util, f"Closed loop: {objective} utility / oracle")
    a_exp.axhspan(lo, hi, color=GRID, alpha=0.7, linewidth=0)
    a_exp.axhline(oracle[objective]["exposure_ms"], color=INK_2, linewidth=1)
    a_util.axhline(1.0, color=INK_2, linewidth=1)
    oracle_u = oracle[objective]["utility"] or float("nan")
    runs = {}
    for r in trajectory_rows:
        runs.setdefault((r["controller"], r["start"]), []).append(r)
    starts = sorted({k[1] for k in runs})
    for (name, start), rows in runs.items():
        t = [r["t_s"] for r in rows]
        ls = "-" if start == starts[0] else "--"
        color = CONTROLLER_COLORS.get(name, INK_2)
        label = name if start == starts[0] else None
        a_exp.plot(t, [r["exposure_ms"] for r in rows], ls, color=color, linewidth=2, label=label)
        a_util.plot(t, [r["utility"] / oracle_u for r in rows], ls, color=color, linewidth=2, label=label)
    a_exp.set_yscale("log")
    a_exp.set_xlabel("time (s)")
    a_exp.set_ylabel("exposure (ms)")
    a_util.set_xlabel("time (s)")
    a_util.set_ylabel("utility / oracle")
    a_exp.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    a_util.legend(frameon=False, fontsize=9, labelcolor=INK_2)

    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
