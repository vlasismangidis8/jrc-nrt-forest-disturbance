"""Check that the mirrored Sentinel-2 cubes are complete and readable (no network needed).

For every manifest in data/manifests/ this script:
  1. checks that each listed chunk file is on disk and not empty,
  2. opens both resolution groups with xarray,
  3. decodes a few time slabs (first, middle, last) of every data array.

A missing Zarr chunk is silently read as fill values, so step 1 is the one that matters;
steps 2 and 3 catch broken metadata and corrupt chunks.

Usage:
    python scripts/verify_cubes.py                 # every mirrored PSU
    python scripts/verify_cubes.py --psus 4914     # selected PSUs

Exit code 0 if every cube passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
GROUPS = ("r10m", "r20m")


def check_files(store_dir: Path, manifest: dict) -> tuple[int, int, list[str]]:
    """Return (n_files, bytes, problems) for the chunk files listed in the manifest."""
    n_files, nbytes, problems = 0, 0, []
    for array, info in manifest["arrays"].items():
        for name in info["files"]:
            path = store_dir / array / name
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                problems.append(f"missing {array}/{name}")
                continue
            if size == 0:
                problems.append(f"empty {array}/{name}")
            n_files += 1
            nbytes += size
        if len(info["files"]) != info["expected_chunks"]:
            problems.append(
                f"{array}: {len(info['files'])} files listed, {info['expected_chunks']} expected"
            )
    return n_files, nbytes, problems


def check_readable(store_dir: Path) -> tuple[dict, list[str]]:
    """Open both groups and decode a few time slabs of every data array."""
    summary, problems = {}, []
    for group in GROUPS:
        try:
            ds = xr.open_zarr(store_dir, group=group, consolidated=True, mask_and_scale=False)
        except Exception as e:  # noqa: BLE001 - a checker reports every failure, whatever its type
            problems.append(f"{group}: cannot open ({type(e).__name__}: {e})")
            continue
        times = ds["time"].values
        if not np.all(np.diff(times) > np.timedelta64(0)):
            problems.append(f"{group}: time axis is not strictly increasing")
        summary[group] = {
            "arrays": sorted(ds.data_vars),
            "shape": tuple(ds.sizes[d] for d in ("time", "y", "x")),
            "time": (str(times[0])[:10], str(times[-1])[:10]),
        }
        for t in sorted({0, len(times) // 2, len(times) - 1}):
            for name in ds.data_vars:
                if ds[name].ndim != 3:
                    continue
                try:
                    ds[name].isel(time=t).load()
                except Exception as e:  # noqa: BLE001 - same: report, do not crash
                    problems.append(f"{group}/{name} t={t}: {type(e).__name__}: {e}")
        ds.close()
    return summary, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--psus", nargs="+", type=int, help="PSU ids (default: every manifest)")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    args = ap.parse_args()

    paths = sorted((args.data_dir / "manifests").glob("PSU_*.json"))
    if args.psus:
        wanted = {f"PSU_{p:07d}.json" for p in args.psus}
        paths = [p for p in paths if p.name in wanted]
    if not paths:
        print("no manifests found")
        return 1

    n_bad, total_files, total_bytes = 0, 0, 0
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        store_dir = args.data_dir / Path(manifest["store"]).relative_to("data")
        n_files, nbytes, problems = check_files(store_dir, manifest)
        if not manifest.get("complete"):
            problems.append("manifest is not flagged complete")
        summary, read_problems = check_readable(store_dir) if not problems else ({}, [])
        problems += read_problems
        total_files += n_files
        total_bytes += nbytes

        status = "OK  " if not problems else "FAIL"
        bands = manifest.get("bands") or "all bands"
        print(
            f"{status} PSU {manifest['psu']:07d}: {n_files} files, {nbytes / 1e6:.0f} MB, {bands}"
        )
        for group, info in summary.items():
            print(
                f"       {group}: time/y/x = {info['shape']}, "
                f"{info['time'][0]} .. {info['time'][1]}, arrays {', '.join(info['arrays'])}"
            )
        for problem in problems[:20]:
            print(f"       ! {problem}")
        if len(problems) > 20:
            print(f"       ! ... and {len(problems) - 20} more")
        n_bad += bool(problems)

    print(
        f"{len(paths) - n_bad}/{len(paths)} cubes OK, {total_files} files, {total_bytes / 1e9:.2f} GB"
    )
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
