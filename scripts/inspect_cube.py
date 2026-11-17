"""Inspect one mirrored Sentinel-2 cube and plot the NDVI series of one forest pixel.

Prints the facts needed before any processing (arrays, encodings, grids, time axis,
SCL classes, fill values, reflectance level per year) and writes one figure to
docs/figures/. Reads the local mirror only.

Usage:
    python scripts/inspect_cube.py --psu 4914
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401 - registers the .rio accessor
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
CLEAR_SCL = (4, 5, 7)  # the classes the JRC baseline keeps
SCL_NAMES = {
    0: "no data",
    1: "saturated/defective",
    2: "dark area",
    3: "cloud shadow",
    4: "vegetation",
    5: "not vegetated",
    6: "water",
    7: "unclassified",
    8: "cloud medium prob.",
    9: "cloud high prob.",
    10: "thin cirrus",
    11: "snow/ice",
}
MONITOR_START = np.datetime64("2022-01-01")
EVAL_END = np.datetime64("2023-05-31")
OFFSET_CHANGE = np.datetime64("2022-01-25")  # Sentinel-2 processing baseline 04.00


def describe_group(store: Path, group: str) -> xr.Dataset:
    raw = xr.open_zarr(store, group=group, consolidated=True, mask_and_scale=False, chunks=None)
    dec = xr.open_zarr(store, group=group, consolidated=True, chunks=None)
    x, y = raw["x"].values, raw["y"].values
    print(f"\n[{group}] time/y/x = {raw.sizes['time']} x {raw.sizes['y']} x {raw.sizes['x']}")
    print(f"  CRS            EPSG:{dec.rio.crs.to_epsg()}")
    print(f"  GeoTransform   {raw['spatial_ref'].attrs.get('GeoTransform')}")
    print(f"  x              {x[0]:.1f} .. {x[-1]:.1f} (step {x[1] - x[0]:+.1f})")
    print(f"  y              {y[0]:.1f} .. {y[-1]:.1f} (step {y[1] - y[0]:+.1f})")
    for name in raw.data_vars:
        enc = dec[name].encoding
        print(
            f"  {name:<4} stored {raw[name].dtype!s:<7} decoded {dec[name].dtype!s:<8}"
            f" chunks {enc.get('chunks')} scale_factor {enc.get('scale_factor')}"
            f" _FillValue {enc.get('_FillValue')}"
        )
    return raw


def describe_time(times: np.ndarray) -> None:
    t = pd.DatetimeIndex(times)
    gaps = np.diff(times).astype("timedelta64[h]").astype(float) / 24
    print(f"\n[time] {len(t)} acquisitions, {t[0]:%Y-%m-%d} .. {t[-1]:%Y-%m-%d}")
    print(f"  per year       {t.year.value_counts().sort_index().to_dict()}")
    print(
        f"  gap in days    median {np.median(gaps):.1f}, max {gaps.max():.1f}, min {gaps.min():.2f}"
    )
    print(f"  same-day pairs {int((np.diff(t.normalize()) == np.timedelta64(0)).sum())}")
    hist = (times >= np.datetime64("2019-01-01")) & (times < MONITOR_START)
    mon = (times >= MONITOR_START) & (times < EVAL_END + 1)
    print(f"  before 2019    {int((times < np.datetime64('2019-01-01')).sum())}")
    print(f"  2019 to 2021   {int(hist.sum())} (baseline history)")
    print(f"  2022-01-01 to 2023-05-31  {int(mon.sum())} (monitoring)")
    print(f"  after 2023-05-31          {int((times > EVAL_END).sum())}")


def window_in_cube(raw: xr.Dataset, mask_src, res: float) -> tuple[slice, slice]:
    """Row and column slices of the cube that cover the mask; checks grid alignment."""
    left = raw["x"].values[0] - res / 2
    top = raw["y"].values[0] + res / 2
    col0 = (mask_src.bounds.left - left) / res
    row0 = (top - mask_src.bounds.top) / res
    aligned = abs(col0 - round(col0)) < 1e-6 and abs(row0 - round(row0)) < 1e-6
    print(f"  offset of the mask in the cube: row {row0:g}, col {col0:g} (aligned: {aligned})")
    r, c = round(row0), round(col0)
    inside = r >= 0 and c >= 0 and r + mask_src.height <= raw.sizes["y"]
    inside = inside and c + mask_src.width <= raw.sizes["x"]
    print(f"  mask fully inside the cube: {inside}")
    return slice(r, r + mask_src.height), slice(c, c + mask_src.width)


def plot_pixel(times, ndvi, clear, psu, row, col, x, y, out: Path) -> None:
    ink, muted, grid, axis = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
    blue = "#2a78d6"
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9})
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150, facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(
        times[~clear],
        ndvi[~clear],
        ls="none",
        marker="o",
        ms=2.5,
        color=muted,
        alpha=0.55,
        label=f"masked by SCL ({int((~clear).sum())})",
    )
    ax.plot(
        times[clear],
        ndvi[clear],
        lw=1.0,
        marker="o",
        ms=3,
        color=blue,
        label=f"clear, SCL 4/5/7 ({int(clear.sum())})",
    )
    ax.axvline(MONITOR_START, color="#52514e", lw=1)
    ax.annotate(
        "monitoring starts\n2022-01-01",
        (MONITOR_START, -0.17),
        xytext=(5, 0),
        textcoords="offset points",
        color="#52514e",
        va="bottom",
    )
    ax.set_ylim(-0.2, 1.0)
    ax.set_ylabel("NDVI", color=ink)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=grid, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(axis)
    ax.tick_params(colors=muted, length=0)
    ax.set_title(
        f"PSU {psu:07d}: NDVI of one forest pixel (row {row}, col {col} of the 10 m mask; "
        f"x {x:.0f}, y {y:.0f})",
        loc="left",
        color=ink,
        fontsize=10,
    )
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.1),
        frameon=False,
        labelcolor="#52514e",
        ncols=2,
        borderaxespad=0,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"\nfigure: {out.relative_to(REPO)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--psu", type=int, default=4914)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    ap.add_argument("--fig-dir", type=Path, default=REPO / "docs" / "figures")
    args = ap.parse_args()
    psu = args.psu
    store = args.data_dir / "sentinel2" / f"PSU_{psu:07d}.zarr"
    print(f"PSU {psu:07d}  {store.relative_to(REPO)}")

    r10 = describe_group(store, "r10m")
    r20 = describe_group(store, "r20m")
    times = r10["time"].values
    print(f"\n  r10m and r20m share the time axis: {np.array_equal(times, r20['time'].values)}")
    describe_time(times)

    print("\n[grids]")
    for res in (10, 20, 30):
        with rasterio.open(args.data_dir / "forest_masks" / f"{res}m" / f"mask_{psu:07d}.tif") as m:
            print(f"  mask {res} m: {m.height} x {m.width}, bounds {tuple(m.bounds)}")
    for name, ds, res in (("r10m", r10, 10.0), ("r20m", r20, 20.0)):
        xs, ys = ds["x"].values, ds["y"].values
        print(
            f"  cube {name}: bounds ({xs[0] - res / 2:.0f}, {ys[-1] - res / 2:.0f}, "
            f"{xs[-1] + res / 2:.0f}, {ys[0] + res / 2:.0f})"
        )
    dx = (r20["x"].values[0] - 10.0) - (r10["x"].values[0] - 5.0)
    dy = (r20["y"].values[0] + 10.0) - (r10["y"].values[0] + 5.0)
    print(f"  r20m origin relative to the r10m origin: {dx:+.0f} m in x, {dy:+.0f} m in y")
    with rasterio.open(args.data_dir / "forest_masks" / "20m" / f"mask_{psu:07d}.tif") as m20:
        print("  20 m mask against r20m:")
        window_in_cube(r20, m20, 20.0)
    with rasterio.open(args.data_dir / "forest_masks" / "10m" / f"mask_{psu:07d}.tif") as m10:
        print("  10 m mask against r10m:")
        rows, cols = window_in_cube(r10, m10, 10.0)
        forest = m10.read(1) == 1
    print(f"  forest share of the 10 m mask: {forest.mean():.1%}")

    # Everything below is on the 10 m mask window.
    b04 = r10["B04"].values[:, rows, cols]
    b08 = r10["B08"].values[:, rows, cols]
    # Every 10 m pixel lies inside exactly one 20 m pixel. Find it from the coordinates:
    # the r20m grid does not start at the r10m origin for every PSU.
    left20 = r20["x"].values[0] - 10.0
    top20 = r20["y"].values[0] + 10.0
    ci = np.floor((r10["x"].values[cols] - left20) / 20.0).astype(int)
    ri = np.floor((top20 - r10["y"].values[rows]) / 20.0).astype(int)
    scl = r20["SCL"].values[:, ri][:, :, ci]

    print("\n[SCL] share of all pixel-dates in the mask window")
    counts = np.bincount(scl.ravel(), minlength=12)
    for cls in np.flatnonzero(counts):
        print(f"  {cls:>2} {SCL_NAMES.get(int(cls), '?'):<20} {counts[cls] / scl.size:6.1%}")
    clear = np.isin(scl, CLEAR_SCL)
    print(f"  kept by the baseline (classes 4, 5, 7): {clear.mean():.1%}")

    print("\n[fill value] stored 0 in the reflectance bands")
    for name, band in (("B04", b04), ("B08", b08)):
        zero = band == 0
        print(
            f"  {name}: {zero.mean():.1%} of values are 0; "
            f"{(scl[zero] == 0).mean() if zero.any() else float('nan'):.1%} of those have SCL 0; "
            f"{(zero & clear).sum()} zeros on pixels the baseline keeps; "
            f"{int((band < 0).sum())} negative values"
        )
    n_nodata_dates = int((scl == 0).all(axis=(1, 2)).sum())
    print(f"  dates with no data anywhere in the window: {n_nodata_dates}")

    print("\n[reflectance level] median stored value on clear forest pixels, June to August")
    print("  (a +1000 step after 2022-01-25 would mean the cube is not harmonised)")
    t = pd.DatetimeIndex(times)
    veg = (scl == 4) & forest[None]
    for year in sorted(set(t.year)):
        sel = np.flatnonzero((t.year == year) & (t.month >= 6) & (t.month <= 8))
        m = veg[sel]
        if not m.any():
            continue
        print(
            f"  {year}: B04 {np.median(b04[sel][m]):6.0f}   B08 {np.median(b08[sel][m]):6.0f}"
            f"   B04 1st percentile {np.percentile(b04[sel][m], 1):6.0f}   ({len(sel)} dates)"
        )
    for label, sel in (
        ("60 days before 2022-01-25", (times < OFFSET_CHANGE) & (times >= OFFSET_CHANGE - 60)),
        ("60 days after 2022-01-25 ", (times >= OFFSET_CHANGE) & (times < OFFSET_CHANGE + 60)),
    ):
        m = veg[sel]
        if m.any():
            print(f"  {label}: B04 median {np.median(b04[sel][m]):.0f}, min {b04[sel][m].min()}")

    print("\n[clear observations per forest pixel] SCL 4/5/7 and stored value not 0")
    usable = clear & (b04 != 0) & (b08 != 0)
    for label, sel in (
        (
            "2019 to 2021 (history)   ",
            (times >= np.datetime64("2019-01-01")) & (times < MONITOR_START),
        ),
        ("2022-01-01 to 2023-05-31 ", (times >= MONITOR_START) & (times < EVAL_END + 1)),
    ):
        n = usable[sel].sum(axis=0)[forest]
        print(
            f"  {label}: median {np.median(n):.0f}, 5th percentile {np.percentile(n, 5):.0f}, "
            f"min {n.min()}, of {int(sel.sum())} dates"
        )

    # One forest pixel: the forest pixel closest to the centre of the mask.
    rr, cc = np.nonzero(forest)
    k = np.argmin((rr - forest.shape[0] / 2) ** 2 + (cc - forest.shape[1] / 2) ** 2)
    row, col = int(rr[k]), int(cc[k])
    red = b04[:, row, col].astype("float64") * 1e-4
    nir = b08[:, row, col].astype("float64") * 1e-4
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
    has_data = (b04[:, row, col] != 0) & (b08[:, row, col] != 0)
    px_clear = usable[:, row, col]
    x = r10["x"].values[cols][col]
    y = r10["y"].values[rows][row]
    plot_pixel(
        times[has_data],
        ndvi[has_data],
        px_clear[has_data],
        psu,
        row,
        col,
        x,
        y,
        args.fig_dir / f"ndvi_forest_pixel_PSU_{psu:07d}.png",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
