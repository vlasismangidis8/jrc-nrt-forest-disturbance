"""Check a finished run and pack its rasters into the archive that is sent to JRC.

The archive is only written if
  - there is exactly one raster for every PSU of data/PSUs_4326.geojson and nothing else,
  - the JRC conformance checker passes every file,
  - the stricter check of `run_reference.py` (exact grid, Int16, dates, mask) finds nothing.

The rasters go into one top-level folder named like the archive. The challenge page does
not prescribe a layout (docs/challenge_spec.md, question Q5).

Usage:
    python scripts/build_archive.py --run-dir outputs/baseline_iqr --name mangidis_iqr_baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from argparse import Namespace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_reference


def expected_psus(data_dir: Path) -> set[int]:
    boundaries = json.loads((data_dir / "PSUs_4326.geojson").read_text(encoding="utf-8"))
    return {int(feature["properties"]["GRD_ID"]) for feature in boundaries["features"]}


def completeness(rasters: Path, expected: set[int]) -> list[str]:
    """Problems with the set of files: missing PSUs, unknown PSUs, stray files."""
    problems, found = [], set()
    for path in sorted(rasters.iterdir()):
        match = re.fullmatch(r"disturbance_date_(\d{7})\.tif", path.name)
        if match:
            found.add(int(match[1]))
        else:
            problems.append(f"unexpected file {path.name}")
    if missing := sorted(expected - found):
        problems.append(f"{len(missing)} PSU(s) without a raster: {missing[:10]}")
    if unknown := sorted(found - expected):
        problems.append(f"raster(s) for unknown PSU(s): {unknown[:10]}")
    return problems


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--name", required=True, help="archive and folder name, without extension")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    args = ap.parse_args()

    rasters = args.run_dir / "rasters"
    problems = completeness(rasters, expected_psus(args.data_dir))
    for problem in problems:
        print(f"NOT COMPLETE: {problem}")
    check = Namespace(run_dir=args.run_dir, data_dir=args.data_dir, remote_masks=False)
    conforming = run_reference.run_check(check) == 0
    if problems or not conforming:
        print("no archive written")
        return 1

    archive = args.run_dir / f"{args.name}.tar.gz"
    files = sorted(rasters.glob("disturbance_date_*.tif"))
    with tarfile.open(archive, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=f"{args.name}/{path.name}")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (args.run_dir / f"{args.name}.sha256").write_text(f"{digest}  {archive.name}\n")
    print(f"{archive}: {len(files)} rasters, {archive.stat().st_size / 1e6:.1f} MB")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
