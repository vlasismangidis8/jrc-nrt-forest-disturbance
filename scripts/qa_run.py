"""Checks of a monitor run that need no reference labels.

95 of the 100 PSUs have no labels, so nothing can be scored there. What can be seen
without labels is whether a run behaves oddly somewhere:

  - the cube starts too late for the training period or ends before the window does,
  - a large share of the forest is flagged,
  - the alarms of a PSU pile up in a single month (a seasonal artefact or one large event),
  - the flagged pixels are scattered single cells instead of patches.

Reads `runs.csv`, `alarms/` and `rasters/` of a run folder written by `run_monitor.py`,
prints one line per PSU, and writes `qa.csv` and `qa_alarms_by_month.png` into the run
folder. Nothing here feeds back into the monitor.

Usage:
    python scripts/qa_run.py --run-dir outputs/kalman_cusum
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]

HIGH_SHARE = 0.10  # flagged share of the forest that deserves a look
BURST_SHARE = 0.03  # share of the forest alarmed within one calendar month
SINGLE_CELL = 4  # 10 m pixels of one 20 m cell
FIGURE_START = "2021-01"
INK, SECONDARY, MUTED, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb"
RAMP = LinearSegmentedColormap.from_list("blue", [SURFACE, "#a9cbf2", "#2a78d6", "#0f3a75"])


def epoch_day(date: str) -> int:
    return int(np.datetime64(date, "D").astype(np.int64))


def month_index(days: np.ndarray) -> np.ndarray:
    """Months since 1970-01 of epoch days."""
    return days.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)


def latitudes(data_dir: Path) -> dict[int, float]:
    boundaries = json.loads((data_dir / "PSUs_4326.geojson").read_text(encoding="utf-8"))
    return {
        int(f["properties"]["GRD_ID"]): float(
            np.mean([p[1] for p in f["geometry"]["coordinates"][0]])
        )
        for f in boundaries["features"]
    }


def check_psu(row, run_dir: Path, config: dict) -> tuple[dict, pd.Series]:
    """Label-free figures of one PSU and its alarmed share of the forest per month."""
    psu, forest = int(row.psu), int(row.forest_px)
    per_forest = 1.0 / forest if forest else 0.0  # a mask can hold no forest pixel at all
    start, end = epoch_day(config["report_start"]), epoch_day(config["report_end"])
    with rasterio.open(run_dir / "rasters" / f"disturbance_date_{psu:07d}.tif") as src:
        dates = src.read(1)
    flagged = dates > 0
    n_flagged = int(flagged.sum())
    labels, _ = ndimage.label(flagged, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())[1:]
    in_single_cells = int(sizes[sizes <= SINGLE_CELL].sum())

    alarms = np.load(run_dir / "alarms" / f"PSU_{psu:07d}.npz")
    days = np.floor(alarms["day"]).astype(np.int64)
    by_month = pd.Series(month_index(days)).value_counts().sort_index() * per_forest
    reported = month_index(dates[flagged].astype(np.int64))
    peak = pd.Series(reported).value_counts() * per_forest
    rearm = int(config["rearm_days"])
    silenced = np.unique(
        np.stack([alarms["row"], alarms["col"]])[:, (days >= start - rearm) & (days < start)],
        axis=1,
    ).shape[1]

    first, last = epoch_day(row.first_date), epoch_day(row.last_date)
    notes = []
    if not forest:
        notes.append("no forest pixel in the mask")
    if start - first < int(config["min_days"]):
        notes.append(f"training shorter than {config['min_days']} d before the window")
    if last < end:
        notes.append(f"cube ends {row.last_date}")
    if n_flagged * per_forest > HIGH_SHARE:
        notes.append("high flagged share")
    if len(peak) and peak.max() > BURST_SHARE:
        month = str(np.datetime64(int(peak.idxmax()), "M"))
        notes.append(f"{peak.max():.1%} of the forest alarmed in {month}")
    return {
        "psu": psu,
        "first_date": row.first_date,
        "last_date": row.last_date,
        "dates": int(row.dates),
        "forest_px": forest,
        "flagged_share": n_flagged * per_forest,
        "share_2022": float((dates[flagged] <= epoch_day("2022-12-31")).sum()) * per_forest,
        "share_2023": float((dates[flagged] > epoch_day("2022-12-31")).sum()) * per_forest,
        "peak_month_share": float(peak.max()) if len(peak) else 0.0,
        "single_cell_share_of_flagged": in_single_cells / n_flagged if n_flagged else 0.0,
        "silent_at_start_share": silenced * per_forest,
        "notes": "; ".join(notes),
    }, by_month


def plot_months(monthly: pd.DataFrame, order: list[int], out: Path, end: str) -> None:
    """Share of the forest alarmed per month, one row per PSU (sorted south to north)."""
    months = np.arange(
        np.datetime64(FIGURE_START, "M"), np.datetime64(end, "M") + 1, dtype="datetime64[M]"
    )
    grid = monthly.reindex(index=order, columns=months.astype(np.int64)).fillna(0.0).to_numpy()
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8})
    height = max(2.5, 0.16 * len(order) + 1.2)
    fig, ax = plt.subplots(figsize=(9, height), dpi=150, facecolor=SURFACE)
    image = ax.imshow(100 * grid, cmap=RAMP, vmin=0, vmax=5, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(order)), [f"{p:07d}" for p in order], color=MUTED, fontsize=6)
    ticks = [i for i, m in enumerate(months) if str(m)[5:] in ("01", "07")]
    ax.set_xticks(ticks, [str(months[i]) for i in ticks], color=MUTED)
    first_window = int(np.searchsorted(months, np.datetime64("2022-01", "M")))
    ax.axvline(first_window - 0.5, color=SECONDARY, lw=0.8)
    ax.text(first_window - 0.3, -0.6, "reporting window", color=SECONDARY, va="bottom")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_color(AXIS)
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, extend="max")
    bar.set_label("% of the forest pixels alarmed in the month", color=SECONDARY)
    bar.ax.tick_params(colors=MUTED, length=0)
    bar.outline.set_visible(False)
    ax.set_title(
        f"Alarms per month and PSU, south (top) to north ({len(order)} PSUs)",
        loc="left",
        color=INK,
        fontsize=10,
        pad=18,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    args = ap.parse_args()

    sections = yaml.safe_load((args.run_dir / "config.yaml").read_text(encoding="utf-8"))
    config = {k: v for part in sections.values() if isinstance(part, dict) for k, v in part.items()}
    config = {k: str(v) if k.startswith(("report", "event")) else v for k, v in config.items()}
    runs = pd.read_csv(args.run_dir / "runs.csv").drop_duplicates("psu", keep="last")
    runs = runs[
        [(args.run_dir / "rasters" / f"disturbance_date_{p:07d}.tif").exists() for p in runs.psu]
    ]

    rows, monthly = [], {}
    for row in runs.itertuples():
        summary, by_month = check_psu(row, args.run_dir, config)
        rows.append(summary)
        monthly[summary["psu"]] = by_month
    table = pd.DataFrame(rows).sort_values("flagged_share", ascending=False)
    table.to_csv(args.run_dir / "qa.csv", index=False, float_format="%.4f")

    north = latitudes(args.data_dir)
    order = sorted(table.psu, key=lambda p: north.get(p, 0.0))
    figure = args.run_dir / "qa_alarms_by_month.png"
    plot_months(pd.DataFrame(monthly).T, order, figure, config["report_end"])

    print("PSU      dates  flagged  2022   2023  peak month  single cells  silent at start  notes")
    for r in table.itertuples():
        print(
            f"{r.psu:07d}  {r.dates:5d}  {r.flagged_share:6.1%} {r.share_2022:6.1%} {r.share_2023:6.1%}"
            f"  {r.peak_month_share:9.1%}  {r.single_cell_share_of_flagged:11.0%}"
            f"  {r.silent_at_start_share:14.1%}  {r.notes}"
        )
    share = table.flagged_share
    print(
        f"\n{len(table)} PSUs. Flagged share of the forest: median {share.median():.1%}, "
        f"range {share.min():.1%} to {share.max():.1%}; {int((table.notes != '').sum())} with notes."
    )
    print(f"table: {args.run_dir / 'qa.csv'}\nfigure: {figure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
