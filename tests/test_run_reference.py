"""Tests for scripts/run_reference.py: the wrapper around the JRC conformance checker."""

import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

_SPEC = importlib.util.spec_from_file_location(
    "run_reference", Path(__file__).resolve().parents[1] / "scripts" / "run_reference.py"
)
rr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rr)

ORIGIN = (3_000_000.0, 2_000_000.0)


def write_tif(path: Path, array: np.ndarray, origin=ORIGIN, res: float = 10.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": array.dtype,
        "crs": "EPSG:3035",
        "transform": from_origin(origin[0], origin[1], res, res),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


@pytest.fixture
def run(tmp_path):
    """A data folder with one 10 m mask and an empty run folder."""
    write_tif(
        tmp_path / "data" / "forest_masks" / "10m" / "mask_0000001.tif", np.ones((20, 20), "uint8")
    )
    args = Namespace(run_dir=tmp_path / "run", data_dir=tmp_path / "data", remote_masks=False)
    (args.run_dir / "rasters").mkdir(parents=True)
    return args


def dates(value: int = 19100) -> np.ndarray:
    array = np.zeros((20, 20), "int16")
    array[5:8, 5:8] = value
    return array


def test_conforming_raster_passes(run):
    write_tif(run.run_dir / "rasters" / "disturbance_date_0000001.tif", dates())
    assert rr.run_check(run) == 0
    assert "1 Passed, 0 Failed" in (run.run_dir / "conformance.txt").read_text(encoding="utf-8")


def test_shifted_grid_fails(run):
    shifted = (ORIGIN[0] + 5.0, ORIGIN[1])
    write_tif(run.run_dir / "rasters" / "disturbance_date_0000001.tif", dates(), origin=shifted)
    assert rr.run_check(run) == 1


def test_date_before_2022_fails(run):
    write_tif(run.run_dir / "rasters" / "disturbance_date_0000001.tif", dates(18992))
    assert rr.run_check(run) == 1


def test_wrongly_named_file_fails(run):
    # the checker skips it with a message; the wrapper must not report success
    write_tif(run.run_dir / "rasters" / "disturbance_date_0000001.tif", dates())
    write_tif(run.run_dir / "rasters" / "disturbance_type_0000001.tif", dates())
    assert rr.run_check(run) == 1


def test_reference_scripts_are_loaded_without_running_them():
    module = rr.load_reference("00_check_conformance.py")
    assert module.FILENAME_PATTERN.match("disturbance_date_0004914.tif")


# --- scripts/build_archive.py ---------------------------------------------------------------

_ARCHIVE = importlib.util.spec_from_file_location(
    "build_archive", Path(__file__).resolve().parents[1] / "scripts" / "build_archive.py"
)
build_archive = importlib.util.module_from_spec(_ARCHIVE)
_ARCHIVE.loader.exec_module(build_archive)


def test_completeness_reports_missing_unknown_and_stray_files(tmp_path):
    for name in ("disturbance_date_0000001.tif", "disturbance_date_0000009.tif", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    problems = build_archive.completeness(tmp_path, {1, 2, 3})
    assert any("unexpected file notes.txt" in p for p in problems)
    assert any("2 PSU(s) without a raster: [2, 3]" in p for p in problems)
    assert any("unknown PSU(s): [9]" in p for p in problems)
    (tmp_path / "notes.txt").unlink()
    (tmp_path / "disturbance_date_0000009.tif").unlink()
    assert build_archive.completeness(tmp_path, {1}) == []
