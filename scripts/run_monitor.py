"""Run the Kalman + CUSUM monitor on PSUs and write one disturbance-date raster each.

Resumable: a PSU whose raster exists is skipped, and a PSU whose cube is not completely
downloaded is skipped with a message. Output layout is the one `run_reference.py check`
and `run_reference.py evaluate` expect:

    outputs/<run>/rasters/disturbance_date_{psu:07d}.tif
    outputs/<run>/config.yaml      copy of the configuration used
    outputs/<run>/runs.csv         one line per PSU: runtime and share of forest flagged
    outputs/<run>/alarms/          every alarm of every pixel, for diagnostics

Usage:
    python scripts/run_monitor.py --config config/kc_nbr_v1.yaml              # 5 demo PSUs
    python scripts/run_monitor.py --config config/kc_nbr_v1.yaml --psus all
    python scripts/run_monitor.py --config ... --also-from 2021-01-01         # tuning aid
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nrtfd.io import load_cube, write_date_raster
from nrtfd.monitor import MonitorConfig, epoch_day, run_cube


def resolve_psus(arg: list[str], challenge: dict, data_dir: Path) -> list[int]:
    if arg == ["demo"]:
        return list(challenge["demo_psus"])
    if arg == ["all"]:
        masks = sorted((data_dir / "forest_masks" / "10m").glob("mask_*.tif"))
        return [int(p.stem.split("_")[1]) for p in masks]
    return [int(a) for a in arg]


def process_psu(psu: int, config: MonitorConfig, args, run_dir: Path, out: Path, log_path: Path):
    """Run the monitor on one PSU, write its raster(s) and alarm list; return a summary line."""
    t0 = time.time()
    cube = load_cube(psu, args.data_dir, config.arrays, args.cache_dir)
    t_load = time.time() - t0
    monitor = run_cube(cube, config)
    dates = monitor.first_alarm()
    write_date_raster(out, cube.to_raster(dates), cube.profile)
    if args.also_from:
        extra = run_dir / f"rasters_from_{args.also_from}" / out.name
        early = monitor.first_alarm(start=args.also_from)
        write_date_raster(extra, cube.to_raster(early), cube.profile)

    days, pixels = monitor.alarms
    rows, cols = np.nonzero(cube.forest)
    (run_dir / "alarms").mkdir(exist_ok=True)
    np.savez_compressed(
        run_dir / "alarms" / f"PSU_{psu:07d}.npz",
        day=days,
        row=rows[pixels].astype(np.int16),
        col=cols[pixels].astype(np.int16),
    )
    seconds = time.time() - t0
    stop = np.datetime64(config.report_end) + np.timedelta64(1, "D")
    flagged = int((dates > 0).sum())
    share = flagged / max(cube.n_pixels, 1)
    first, last = str(cube.times[0])[:10], str(cube.times[-1])[:10]
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow(
            [psu, first, last, int((cube.times < stop).sum()), cube.n_pixels, flagged]
            + [f"{share:.4f}", f"{seconds:.1f}"]
        )
    note = "" if cube.times[-1] >= stop - np.timedelta64(1, "D") else f", CUBE ENDS {last}"
    before = int((days < epoch_day(config.report_start)).sum())
    return (
        f"PSU {psu:07d}: {share:6.1%} of forest flagged ({flagged} px), "
        f"{before} alarms before the window, {seconds:.0f} s ({t_load:.0f} s loading){note}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True, help="method configuration (YAML)")
    ap.add_argument("--psus", nargs="+", default=["demo"], help='"demo", "all", or PSU ids')
    ap.add_argument("--run-dir", type=Path, help="default: outputs/<name in the config>")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    ap.add_argument("--cache-dir", type=Path, help="keep the stored values as .npy files here")
    ap.add_argument("--force", action="store_true", help="redo PSUs whose raster exists")
    ap.add_argument(
        "--also-from",
        help="tuning aid: also write rasters whose window starts at this date, to "
        "rasters_from_<date>/ (the monitor run is the same; only the reporting differs)",
    )
    args = ap.parse_args()

    config = MonitorConfig.from_yaml(args.config)
    challenge = yaml.safe_load((REPO / "config" / "challenge.yaml").read_text(encoding="utf-8"))
    run_dir = args.run_dir or REPO / "outputs" / config.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # One run folder holds the results of one configuration only.
    text = args.config.read_text(encoding="utf-8")
    kept = run_dir / "config.yaml"
    if kept.exists() and kept.read_text(encoding="utf-8") != text and not args.force:
        print(f"{run_dir} holds results of a different configuration; use --force or --run-dir")
        return 1
    kept.write_text(text, encoding="utf-8")

    log_path = run_dir / "runs.csv"
    if not log_path.exists():
        log_path.write_text(
            "psu,first_date,last_date,dates,forest_px,flagged_px,flagged_share,seconds\n"
        )

    n_done, failed = 0, []
    for psu in resolve_psus(args.psus, challenge, args.data_dir):
        out = run_dir / "rasters" / f"disturbance_date_{psu:07d}.tif"
        if out.exists() and not args.force:
            print(f"PSU {psu:07d}: raster exists, skipped")
            continue
        try:
            print(process_psu(psu, config, args, run_dir, out, log_path))
            n_done += 1
        except FileNotFoundError as error:
            print(f"PSU {psu:07d}: skipped ({error})")
        except Exception as error:  # noqa: BLE001 - one bad PSU must not stop the others
            failed.append(psu)
            print(f"PSU {psu:07d}: FAILED ({type(error).__name__}: {error})")
    print(f"{n_done} PSU(s) processed, rasters in {run_dir / 'rasters'}")
    if failed:
        print(f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
