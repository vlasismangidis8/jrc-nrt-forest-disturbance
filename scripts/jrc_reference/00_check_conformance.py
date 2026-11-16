#!/usr/bin/env python3
"""
NRT Forest Disturbance Challenge — Submission Conformance Verification
Validates prediction GeoTIFFs against mandatory Tree Cover masks for CRS (EPSG:3035),
pixel dimensions, spatial alignment, data types, and Unix epoch day bounds.
"""

import os
import re
import sys
import glob
import argparse
import rasterio
import numpy as np

BASE_MASKS_DIR = 'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FOREST/NRT/CHALLENGE/data/forest_masks/'
DEFAULT_OUTPUT_DIR = os.path.expanduser('~/nrt-challenge-demo/data')

FILENAME_PATTERN = re.compile(r"^disturbance_date_(\d{1,7})\.tif$")

def validate_all_outputs(output_dir, base_masks_dir):
    tif_files = sorted(glob.glob(os.path.join(output_dir, "*.tif")))
    if not tif_files:
        print(f"[!] No GeoTIFF files found in target directory: {output_dir}")
        return

    print(f"Target Directory: {output_dir}")
    print(f"Found {len(tif_files)} GeoTIFF files. Performing metadata discovery...\n")

    file_metadata = []
    detected_resolutions = set()

    for out_path in tif_files:
        filename = os.path.basename(out_path)
        match = FILENAME_PATTERN.match(filename)
        if not match:
            print(f"❌ INVALID FILENAME: '{filename}' does not follow 'disturbance_date_XXXXXXX.tif' format.")
            continue

        psu_id = int(match.group(1))

        try:
            with rasterio.open(out_path) as src:
                res_x, res_y = round(abs(src.transform.a)), round(abs(src.transform.e))
                if res_x not in [10, 20, 30] or res_x != res_y:
                    print(f"❌ UNSUPPORTED RESOLUTION: {filename} has pixel resolution {res_x}x{res_y}m (Must be 10, 20, or 30m).")
                    continue

                detected_resolutions.add(res_x)
                file_metadata.append({
                    "path": out_path,
                    "filename": filename,
                    "psu_id": psu_id,
                    "resolution": res_x
                })
        except Exception as e:
            print(f"❌ CORRUPTED FILE: Could not open {filename} - {e}")

    if not file_metadata:
        print("\n[!] No valid submission files passed initial filename and format inspection.")
        return

    if len(detected_resolutions) > 1:
        print(f"\n❌ CONSISTENCY ERROR: Mixed spatial resolutions detected across submission files: {detected_resolutions}m.")
        print("    All output files in a submission must share the exact same resolution (either 10m, 20m, OR 30m).")
        return

    chosen_res = detected_resolutions.pop()
    mask_dir_for_res = f"{base_masks_dir.rstrip('/')}/{chosen_res}m/"
    print(f"✅ Metadata Discovery Complete: Inferred uniform resolution of {chosen_res}m.")
    print(f"   Using reference mask directory: {mask_dir_for_res}\n")

    print("Beginning conformance validation against reference masks...\n")
    results = {"passed": 0, "failed": 0}

    for meta in file_metadata:
        out_path = meta["path"]
        filename = meta["filename"]
        psu_id = meta["psu_id"]
        mask_url = f"{mask_dir_for_res}mask_{psu_id:07d}.tif"
        errors = []

        try:
            with rasterio.open(out_path) as out_src, rasterio.open(mask_url) as mask_src:
                out_data = out_src.read(1)

                if out_src.crs != mask_src.crs:
                    errors.append(f"CRS mismatch: expected {mask_src.crs}, got {out_src.crs}")

                if out_src.shape != mask_src.shape:
                    errors.append(f"Dimension mismatch for {chosen_res}m grid: expected {mask_src.shape}, got {out_src.shape}")

                if not np.allclose(out_src.transform, mask_src.transform, atol=1e-5):
                    errors.append(f"Transform mismatch: pixel origin/grid axes do not align with reference {chosen_res}m mask.")

                if not np.issubdtype(out_data.dtype, np.integer):
                    errors.append(f"Data type error: expected integer format, got {out_data.dtype}")

                valid_pixels = out_data[out_data > 0]
                if valid_pixels.size > 0:
                    min_val, max_val = valid_pixels.min(), valid_pixels.max()
                    if min_val < 18993:
                        errors.append(f"Detection dates out of bounds: found min={min_val}, max={max_val} (expected 2022 epoch days)")

        except Exception as e:
            errors.append(f"Reference alignment check failed: {str(e)}")

        if errors:
            results["failed"] += 1
            print(f"❌ FAIL: {filename}")
            for err in errors:
                print(f"    └── {err}")
        else:
            results["passed"] += 1
            print(f"✅ PASS: {filename}")

    print("\n==========================================")
    print(f"Validation Summary: {results['passed']} Passed, {results['failed']} Failed ({chosen_res}m grid)")
    print("==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate GeoTIFF challenge submission outputs.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Path to directory containing output GeoTIFFs (defaults to {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args()

    validate_all_outputs(os.path.abspath(args.output_dir), BASE_MASKS_DIR)
