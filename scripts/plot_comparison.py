"""Plot UA, PA and F1 against the allowed detection lag for up to three runs.

Reads `metrics.csv` of each run folder (written by `run_reference.py evaluate`).

Usage:
    python scripts/plot_comparison.py --runs outputs/kalman_cusum outputs/baseline_iqr \
        --labels "Kalman + CUSUM" "JRC IQR baseline" --out docs/figures/scores_by_lag_demo.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # fixed order: first run blue, second orange
INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#fcfcfb",
)
PANELS = (("ua", "User's accuracy"), ("pa", "Producer's accuracy"), ("f1", "F1"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="Demo PSUs: accuracy against allowed detection lag")
    args = ap.parse_args()
    if len(args.runs) != len(args.labels) or len(args.runs) > len(SERIES):
        print("give one label per run, at most three runs")
        return 1
    curves = [pd.read_csv(run / "metrics.csv") for run in args.runs]

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=150, sharey=True, facecolor=SURFACE)
    for ax, (column, name) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        for curve, label, color in zip(curves, args.labels, SERIES):
            ax.plot(curve["lag"], curve[column], color=color, lw=2, label=label)
            ax.annotate(
                f"{curve[column].iloc[-1]:.2f}",
                (curve["lag"].iloc[-1], curve[column].iloc[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                color=SECONDARY,
            )
        ax.set_title(name, loc="left", color=INK, fontsize=10)
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(0, 181, 30))
        ax.set_xlabel("Allowed detection lag (days)", color=SECONDARY)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, length=0)
    axes[0].legend(loc="upper left", frameon=False, labelcolor=SECONDARY)
    fig.suptitle(args.title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.99, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
