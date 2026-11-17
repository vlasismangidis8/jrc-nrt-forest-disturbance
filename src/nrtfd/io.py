"""Read one PSU from the local mirror and write the disturbance-date raster.

Everything is expressed on the grid of the 10 m Tree Cover mask, which is the grid of the
submission. The 20 m arrays (B8A, B11, B12, SCL) stay at 20 m in memory; each 10 m pixel
is mapped to the 20 m pixel that contains it, by coordinate, so that the result does not
depend on how the `r20m` grid is aligned with the `r10m` grid.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr

GROUPS = ("r10m", "r20m")


def containing_index(centres: np.ndarray, cube_coords: np.ndarray) -> np.ndarray:
    """Index of the cube pixel that contains each coordinate in `centres`.

    `cube_coords` are the pixel centres along one axis of the cube, evenly spaced, in
    increasing (x) or decreasing (y) order.
    """
    step = cube_coords[1] - cube_coords[0]
    return np.floor((centres - cube_coords[0]) / step + 0.5).astype(np.intp)


@dataclass
class PsuCube:
    """Stored values of one PSU on the window of its 10 m mask."""

    psu: int
    times: np.ndarray  # (T,) datetime64[ns], strictly increasing
    arrays: dict[str, np.ndarray]  # stored values: (T, H, W) at 10 m or (T, h, w) at 20 m
    lookup: dict[str, tuple[np.ndarray, np.ndarray]]  # per array: row and column per forest pixel
    forest: np.ndarray  # (H, W) bool, the 10 m Tree Cover mask
    profile: dict  # CRS, transform and size of the 10 m mask

    @property
    def n_pixels(self) -> int:
        return int(self.forest.sum())

    def pixel_values(self, name: str, t: int) -> np.ndarray:
        """Stored values of array `name` at time index `t`, one per forest pixel."""
        rows, cols = self.lookup[name]
        return np.asarray(self.arrays[name][t])[rows, cols]

    def to_raster(self, values: np.ndarray, dtype="int16") -> np.ndarray:
        """Place one value per forest pixel on the mask grid; other pixels are 0."""
        out = np.zeros(self.forest.shape, dtype=dtype)
        out[self.forest] = values
        return out


def load_cube(
    psu: int, data_dir: Path, names: tuple[str, ...], cache_dir: Path | None = None
) -> PsuCube:
    """Load the arrays `names` (bands and/or SCL) of one PSU.

    With `cache_dir`, the stored values are kept as .npy files and memory-mapped on later
    calls, which makes repeated experiments on the same PSU fast.
    """
    store = data_dir / "sentinel2" / f"PSU_{psu:07d}.zarr"
    if not (store / ".zmetadata").exists():
        raise FileNotFoundError(f"{store} is missing or not completely downloaded")
    with rasterio.open(data_dir / "forest_masks" / "10m" / f"mask_{psu:07d}.tif") as src:
        forest = src.read(1) == 1
        transform = src.transform
        profile = {
            "crs": src.crs,
            "transform": transform,
            "height": src.height,
            "width": src.width,
        }
    xs = transform.c + (np.arange(forest.shape[1]) + 0.5) * transform.a
    ys = transform.f + (np.arange(forest.shape[0]) + 0.5) * transform.e
    forest_rows, forest_cols = np.nonzero(forest)

    groups = {
        g: xr.open_zarr(store, group=g, consolidated=True, mask_and_scale=False, chunks=None)
        for g in GROUPS
    }
    times = groups["r10m"]["time"].values
    if not np.array_equal(times, groups["r20m"]["time"].values):
        raise ValueError(f"PSU {psu}: r10m and r20m have different time axes")
    if not np.all(np.diff(times) > np.timedelta64(0)):
        raise ValueError(f"PSU {psu}: time axis is not strictly increasing")

    arrays, lookup = {}, {}
    for name in names:
        ds = next((groups[g] for g in GROUPS if name in groups[g]), None)
        if ds is None:
            raise KeyError(f"PSU {psu}: array {name} is not in the local mirror")
        rows = containing_index(ys, ds["y"].values)
        cols = containing_index(xs, ds["x"].values)
        if rows.min() < 0 or cols.min() < 0:
            raise ValueError(f"PSU {psu}: the mask extends beyond the cube ({name})")
        if rows.max() >= ds.sizes["y"] or cols.max() >= ds.sizes["x"]:
            raise ValueError(f"PSU {psu}: the mask extends beyond the cube ({name})")
        window = {"y": slice(rows.min(), rows.max() + 1), "x": slice(cols.min(), cols.max() + 1)}
        cache = cache_dir / f"PSU_{psu:07d}" / f"{name}.npy" if cache_dir else None
        if cache is not None and cache.exists():
            data = np.load(cache, mmap_mode="r")
        else:
            data = ds[name].isel(window).values
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache.with_suffix(".part.npy")
                np.save(tmp, data)
                os.replace(tmp, cache)
        expected = (len(times), rows.max() - rows.min() + 1, cols.max() - cols.min() + 1)
        if data.shape != expected:
            raise ValueError(f"PSU {psu}: {name} has shape {data.shape}, expected {expected}")
        arrays[name] = data
        lookup[name] = ((rows - rows.min())[forest_rows], (cols - cols.min())[forest_cols])
    for ds in groups.values():
        ds.close()
    return PsuCube(psu, times, arrays, lookup, forest, profile)


def write_date_raster(path: Path, dates: np.ndarray, profile: dict) -> None:
    """Write detection dates in the format of the JRC baseline.

    One Int16 band of days since 1970-01-01 on the grid of the mask, 0 where nothing was
    detected and outside the forest mask, no nodata tag. The file appears under its final
    name only when it is complete.
    """
    if dates.shape != (profile["height"], profile["width"]):
        raise ValueError(f"raster shape {dates.shape} does not match the mask grid")
    if dates.min() < 0 or dates.max() > np.iinfo(np.int16).max:
        raise ValueError("dates do not fit in Int16")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with rasterio.open(tmp, "w", driver="GTiff", count=1, dtype="int16", **profile) as dst:
        dst.write(dates.astype("int16"), 1)
    os.replace(tmp, path)
