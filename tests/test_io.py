"""Tests of reading a PSU and writing the disturbance-date raster."""

import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
import zarr
from rasterio.transform import from_origin

from nrtfd.io import containing_index, load_cube, write_date_raster

_SPEC = importlib.util.spec_from_file_location(
    "run_reference", Path(__file__).resolve().parents[1] / "scripts" / "run_reference.py"
)
run_reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_reference)

LEFT, TOP = 3_000_000.0, 2_000_000.0  # corner of the 10 m mask


def test_containing_index_on_aligned_and_shifted_grids():
    centres = LEFT + 5.0 + 10.0 * np.arange(6)  # six 10 m pixels
    same_origin = LEFT + 10.0 + 20.0 * np.arange(4)
    assert containing_index(centres, same_origin).tolist() == [0, 0, 1, 1, 2, 2]
    shifted = LEFT - 10.0 + 10.0 + 20.0 * np.arange(4)  # 20 m grid starts 10 m further west
    assert containing_index(centres, shifted).tolist() == [0, 1, 1, 2, 2, 3]
    descending = TOP - 5.0 - 10.0 * np.arange(6)  # y runs north to south
    assert containing_index(descending, TOP - 10.0 - 20.0 * np.arange(4)).tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]


@pytest.fixture
def mirror(tmp_path):
    """A data folder with one PSU: a 6 x 6 mask at 10 m and a cube that is larger than it.

    The r20m grid starts 10 m west of the r10m grid.
    """
    data = tmp_path / "data"
    mask = np.ones((6, 6), "uint8")
    mask[0, 0] = 0
    path = data / "forest_masks" / "10m" / "mask_0000001.tif"
    path.parent.mkdir(parents=True)
    profile = {"driver": "GTiff", "height": 6, "width": 6, "count": 1, "dtype": "uint8"}
    with rasterio.open(
        path, "w", crs="EPSG:3035", transform=from_origin(LEFT, TOP, 10, 10), **profile
    ) as dst:
        dst.write(mask, 1)

    times = np.array(["2018-01-01T10", "2018-01-06T10", "2018-01-11T10"], "datetime64[ns]")
    store = data / "sentinel2" / "PSU_0000001.zarr"
    # r10m: 10 x 10 pixels, two pixels of margin around the mask; value = 100*row + col
    x10 = LEFT - 20 + 5 + 10.0 * np.arange(10)
    y10 = TOP + 20 - 5 - 10.0 * np.arange(10)
    b04 = (100 * np.arange(10)[:, None] + np.arange(10)[None, :]).astype("int16")
    r10 = xr.Dataset(
        {"B04": (("time", "y", "x"), np.stack([b04, b04 + 1000, b04 + 2000]))},
        coords={"time": times, "y": y10, "x": x10},
    )
    # r20m: 6 x 6 pixels starting 10 m west of the r10m grid
    x20 = LEFT - 20 - 10 + 10 + 20.0 * np.arange(6)
    y20 = TOP + 20 - 10 - 20.0 * np.arange(6)
    scl = (10 * np.arange(6)[:, None] + np.arange(6)[None, :]).astype("uint8")
    r20 = xr.Dataset(
        {"SCL": (("time", "y", "x"), np.stack([scl, scl, scl]))},
        coords={"time": times, "y": y20, "x": x20},
    )
    r10.to_zarr(store, group="r10m", zarr_format=2, consolidated=False)
    r20.to_zarr(store, group="r20m", zarr_format=2, consolidated=False)
    zarr.consolidate_metadata(str(store))
    return data


def test_load_cube_maps_both_resolutions_onto_the_mask(mirror, tmp_path):
    cube = load_cube(1, mirror, ("B04", "SCL"))
    assert cube.n_pixels == 35 and cube.forest.shape == (6, 6)
    assert len(cube.times) == 3
    rows, cols = np.nonzero(cube.forest)
    # mask pixel (r, c) is r10m pixel (r + 2, c + 2)
    assert cube.pixel_values("B04", 0).tolist() == (100 * (rows + 2) + cols + 2).tolist()
    assert cube.pixel_values("B04", 2).tolist() == (2000 + 100 * (rows + 2) + cols + 2).tolist()
    # ... and lies in r20m row (r + 2) // 2 and, because of the shift, column (c + 3) // 2
    assert (
        cube.pixel_values("SCL", 1).tolist() == (10 * ((rows + 2) // 2) + (cols + 3) // 2).tolist()
    )

    cached = load_cube(1, mirror, ("B04", "SCL"), cache_dir=tmp_path / "cache")
    again = load_cube(1, mirror, ("B04", "SCL"), cache_dir=tmp_path / "cache")
    for name in ("B04", "SCL"):
        assert np.array_equal(cached.pixel_values(name, 1), cube.pixel_values(name, 1))
        assert np.array_equal(again.pixel_values(name, 1), cube.pixel_values(name, 1))


def test_incomplete_cube_is_refused(mirror):
    (mirror / "sentinel2" / "PSU_0000001.zarr" / ".zmetadata").unlink()
    with pytest.raises(FileNotFoundError, match="not completely downloaded"):
        load_cube(1, mirror, ("B04",))


def test_written_raster_conforms(mirror, tmp_path):
    cube = load_cube(1, mirror, ("B04",))
    dates = np.zeros(cube.n_pixels, np.int32)
    dates[:5] = 19100
    run_dir = tmp_path / "run"
    out = run_dir / "rasters" / "disturbance_date_0000001.tif"
    write_date_raster(out, cube.to_raster(dates), cube.profile)
    with rasterio.open(out) as src:
        assert src.dtypes == ("int16",) and src.nodata is None and src.count == 1
        assert src.read(1)[0, 0] == 0  # outside the forest mask
        assert (src.read(1) == 19100).sum() == 5
    assert not list(out.parent.glob("*.part"))
    args = Namespace(run_dir=run_dir, data_dir=mirror, remote_masks=False)
    assert run_reference.run_check(args) == 0


def test_raster_of_the_wrong_shape_is_refused(mirror, tmp_path):
    cube = load_cube(1, mirror, ("B04",))
    with pytest.raises(ValueError, match="shape"):
        write_date_raster(tmp_path / "x.tif", np.zeros((5, 6), "int16"), cube.profile)
