"""Tests for scripts/run_reference.py: the wrapper around the JRC conformance checker."""

import importlib.util
import json
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


def write_cube_metadata(store: Path, arrays: list[str]) -> None:
    store.mkdir(parents=True)
    listed = {f"{name}/.zarray": {} for name in arrays} | {".zgroup": {}}
    (store / ".zmetadata").write_text(json.dumps({"metadata": listed}), encoding="utf-8")


def test_cube_arrays_lists_what_was_mirrored(tmp_path):
    assert rr.cube_arrays(tmp_path / "PSU_0000001.zarr") is None  # not downloaded
    write_cube_metadata(tmp_path / "PSU_0000002.zarr", ["r20m/B8A", "r20m/SCL", "r20m/x"])
    assert rr.cube_arrays(tmp_path / "PSU_0000002.zarr") == {"r20m/B8A", "r20m/SCL", "r20m/x"}


def test_baseline_skips_cubes_without_the_10m_bands(run, capsys):
    # PSU 1 has only the 20 m arrays, PSU 2 is not there at all: nothing to run, no error
    write_cube_metadata(run.data_dir / "sentinel2" / "PSU_0000001.zarr", ["r20m/B8A", "r20m/SCL"])
    run.psus, run.force = [1, 2], False
    assert rr.run_baseline(run) == 0
    printed = capsys.readouterr().out
    assert "PSU 0000001: cube has no r10m/B04, r10m/B08 yet, skipped" in printed
    assert "PSU 0000002: cube is not completely downloaded, skipped" in printed
    assert not list((run.run_dir / "rasters").iterdir())


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


# --- scripts/build_submission.py ------------------------------------------------------------

_SUBMISSION = importlib.util.spec_from_file_location(
    "build_submission", Path(__file__).resolve().parents[1] / "scripts" / "build_submission.py"
)
build_submission = importlib.util.module_from_spec(_SUBMISSION)
_SUBMISSION.loader.exec_module(build_submission)


def pack(run_dir: Path, name: str, extra: str | None = None) -> Path:
    """An archive of the run folder's rasters with its checksum file, as build_archive writes."""
    import hashlib
    import tarfile

    archive = run_dir / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted((run_dir / "rasters").iterdir()):
            tar.add(path, arcname=f"{name}/{path.name}")
        if extra:
            tar.add(run_dir / "rasters" / "disturbance_date_0000001.tif", arcname=extra)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (run_dir / f"{name}.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive


def test_archive_problems_checks_checksum_layout_completeness_and_content(tmp_path):
    rasters = tmp_path / "rasters"
    rasters.mkdir()
    for psu in (1, 2):
        (rasters / f"disturbance_date_{psu:07d}.tif").write_bytes(bytes([psu]) * 10)
    archive = pack(tmp_path, "team")
    assert build_submission.archive_problems(archive, rasters, "team", {1, 2}) == []

    problems = build_submission.archive_problems(archive, rasters, "team", {1, 2, 3})
    assert problems == ["1 PSU(s) without a raster: [3]"]

    (rasters / "disturbance_date_0000002.tif").write_bytes(b"changed after packing")
    problems = build_submission.archive_problems(archive, rasters, "team", {1, 2})
    assert problems == ["team/disturbance_date_0000002.tif differs from the run folder"]

    (tmp_path / "team.sha256").write_text("0000  team.tar.gz\n", encoding="utf-8")
    assert any(
        "checksum" in p for p in build_submission.archive_problems(archive, rasters, "team", {1, 2})
    )

    stray = pack(tmp_path, "other", extra="other/readme.txt")
    problems = build_submission.archive_problems(stray, rasters, "other", {1, 2})
    assert any("unexpected entry other/readme.txt" in p for p in problems)
