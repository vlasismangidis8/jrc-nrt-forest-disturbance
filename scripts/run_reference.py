"""Run the JRC reference scripts (scripts/jrc_reference/) against the local mirror.

The reference files stay identical to the challenge page. This launcher imports one of
them as a module, replaces only its path constants (remote URLs -> local copy in data/,
working folder ~/nrt-challenge-demo -> a run folder in outputs/) and calls its entry
function. No processing or scoring logic is changed.

A run folder holds the rasters of one method and what was measured on them:
    outputs/<run>/rasters/disturbance_date_{psu:07d}.tif
    outputs/<run>/conformance.txt
    outputs/<run>/metrics.csv, accuracy_curve.png

Usage:
    python scripts/run_reference.py baseline                 # 5 demo PSUs, skips finished ones
    python scripts/run_reference.py baseline --psus 28788 --force
    python scripts/run_reference.py check     [--run-dir outputs/baseline_iqr]
    python scripts/run_reference.py evaluate  [--run-dir outputs/baseline_iqr]
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # the evaluation script calls plt.show()

REPO = Path(__file__).resolve().parents[1]
REF_DIR = REPO / "scripts" / "jrc_reference"


def load_reference(filename: str):
    """Import a reference script as a module (its __main__ block does not run).

    Two of the scripts create ~/nrt-challenge-demo while being imported. The home
    directory points at a temporary folder during the import so the real one stays clean.
    """
    name = "jrc_" + re.sub(r"\W", "_", Path(filename).stem)
    spec = importlib.util.spec_from_file_location(name, REF_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    saved = {key: os.environ.get(key) for key in ("USERPROFILE", "HOME")}
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["USERPROFILE"] = os.environ["HOME"] = tmp
        try:
            spec.loader.exec_module(module)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return module


class PeakMemory:
    """Samples the resident memory of this process and its children (the Dask worker)."""

    def __init__(self, interval: float = 0.5):
        import psutil

        self._process = psutil.Process()
        self._psutil = psutil
        self._interval = interval
        self._stop = threading.Event()
        self.peak = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            total = 0
            for proc in [self._process, *self._process.children(recursive=True)]:
                with contextlib.suppress(self._psutil.Error):
                    total += proc.memory_info().rss
            self.peak = max(self.peak, total)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


class Tee(io.TextIOBase):
    """Writes to the console and keeps a copy."""

    def __init__(self, stream):
        self.stream = stream
        self.buffer_text = io.StringIO()

    def write(self, text: str) -> int:
        self.buffer_text.write(text)
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()


def run_baseline(args) -> int:
    rasters = args.run_dir / "rasters"
    rasters.mkdir(parents=True, exist_ok=True)
    todo = []
    for psu in args.psus:
        store = args.data_dir / "sentinel2" / f"PSU_{psu:07d}.zarr"
        if not (store / ".zmetadata").exists():
            print(f"PSU {psu:07d}: cube is not completely downloaded, skipped")
        elif (rasters / f"disturbance_date_{psu:07d}.tif").exists() and not args.force:
            print(f"PSU {psu:07d}: raster exists, skipped (use --force to redo)")
        else:
            todo.append(psu)
    if not todo:
        return 0

    module = load_reference("01_run_baseline_iqr.py")
    module.S2_URL = (args.data_dir / "sentinel2").as_posix()
    module.MASKS_URL = (args.data_dir / "forest_masks" / "10m").as_posix()
    module.STORAGE_OPTIONS = None  # the script's value is for HTTP access
    module.OUT_DIR = str(rasters)
    module.DEMO_PSUS = todo

    t0 = time.time()
    with PeakMemory() as memory:
        module.run_baseline_pipeline()
    minutes = (time.time() - t0) / 60
    print(
        f"baseline: {len(todo)} PSU(s) in {minutes:.1f} min, "
        f"peak memory {memory.peak / 1e9:.2f} GB (this process and the Dask worker)"
    )
    return 0


def strict_check(rasters: Path, masks_dir: Path) -> list[str]:
    """Exact checks on top of the JRC checker, against the local Tree Cover masks.

    A raster must sit exactly on the grid of its mask, so the geotransform is compared
    without a tolerance. The band count, the exact data type and the pixels outside the
    forest mask are checked as well.
    The last two checks below follow working assumptions about the reporting window and
    the non-forest value, not a rule stated on the challenge page.
    """
    import rasterio

    problems = []
    for path in sorted(rasters.glob("*.tif")):
        match = re.fullmatch(r"disturbance_date_(\d{7})\.tif", path.name)
        if not match:
            problems.append(f"{path.name}: name is not disturbance_date_<7 digits>.tif")
            continue
        with rasterio.open(path) as out:
            res = round(abs(out.transform.a))
            mask_path = masks_dir / f"{res}m" / f"mask_{int(match[1]):07d}.tif"
            if not mask_path.exists():
                problems.append(f"{path.name}: no local mask {mask_path.name} at {res} m")
                continue
            with rasterio.open(mask_path) as mask:
                if out.crs != mask.crs:
                    problems.append(f"{path.name}: CRS {out.crs}, mask has {mask.crs}")
                if out.shape != mask.shape:
                    problems.append(f"{path.name}: shape {out.shape}, mask has {mask.shape}")
                if not out.transform.almost_equals(mask.transform, precision=1e-6):
                    problems.append(f"{path.name}: geotransform differs from the mask")
                if out.count != 1 or out.dtypes[0] != "int16":
                    problems.append(f"{path.name}: {out.count} band(s) of {out.dtypes[0]}")
                data = out.read(1)
                if ((data != 0) & (data < 18993)).any():
                    problems.append(f"{path.name}: values other than 0 before 2022-01-01")
                if (data > 19508).any():
                    problems.append(f"{path.name}: dates after 2023-05-31 (assumption D1)")
                if out.shape == mask.shape and (data[mask.read(1) != 1] != 0).any():
                    problems.append(f"{path.name}: detections outside the mask (assumption D2)")
    return problems


def run_check(args) -> int:
    rasters = args.run_dir / "rasters"
    module = load_reference("00_check_conformance.py")
    local_masks = args.data_dir / "forest_masks"
    masks = module.BASE_MASKS_DIR if args.remote_masks else local_masks.as_posix()
    tee = Tee(sys.stdout)
    with contextlib.redirect_stdout(tee):
        module.validate_all_outputs(str(rasters.resolve()), masks)
        # The checker only prints. Read its summary so this command can fail in a pipeline.
        n_files = len(list(rasters.glob("*.tif")))
        summary = re.search(
            r"Validation Summary: (\d+) Passed, (\d+) Failed", tee.buffer_text.getvalue()
        )
        jrc_ok = bool(summary) and int(summary[1]) == n_files and int(summary[2]) == 0
        problems = strict_check(rasters, local_masks)
        print(f"\nJRC checker: {'OK' if jrc_ok else 'NOT OK'} ({n_files} GeoTIFF files)")
        print(f"Strict check (exact grid, Int16, dates, mask): {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
    report = "conformance_remote_masks.txt" if args.remote_masks else "conformance.txt"
    (args.run_dir / report).write_text(tee.buffer_text.getvalue(), encoding="utf-8")
    return 0 if jrc_ok and not problems else 1


def run_evaluate(args) -> int:
    module = load_reference("02_run_evaluation.py")
    module.PREDICTIONS_DIR = str(args.run_dir / "rasters")
    module.BASE_REF_URL = (args.data_dir / "demo" / "reference-samples").resolve().as_uri()
    module.SSU_FGB_URL = str(args.data_dir / "demo" / "demo_SSUs.fgb")
    metrics = module.run_assessment(plot=True, save_path=str(args.run_dir / "accuracy_curve.png"))
    metrics.to_csv(args.run_dir / "metrics.csv", index=False)
    print(metrics.to_string(index=False))
    print(f"metrics: {args.run_dir / 'metrics.csv'}")
    return 0


def main() -> int:
    import yaml

    sys.stdout.reconfigure(encoding="utf-8")  # the reference scripts print symbols
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["baseline", "check", "evaluate"])
    ap.add_argument("--psus", nargs="+", type=int, help="PSU ids (default: the 5 demo PSUs)")
    ap.add_argument("--run-dir", type=Path, default=REPO / "outputs" / "baseline_iqr")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    ap.add_argument("--config", type=Path, default=REPO / "config" / "challenge.yaml")
    ap.add_argument("--force", action="store_true", help="baseline: redo existing rasters")
    ap.add_argument(
        "--remote-masks", action="store_true", help="check: read the masks from the JRC server"
    )
    args = ap.parse_args()
    if not args.psus:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        args.psus = list(cfg["demo_psus"])
    args.run_dir.mkdir(parents=True, exist_ok=True)
    return {"baseline": run_baseline, "check": run_check, "evaluate": run_evaluate}[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
