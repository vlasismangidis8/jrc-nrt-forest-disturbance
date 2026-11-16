#!/usr/bin/env python3
"""
NRT Forest Disturbance Challenge — Baseline Result Generation
Runs the IQR anomaly tracking algorithm over Sentinel-2 GeoZarr cubes for demo PSUs
and exports disturbance date rasters (EPSG:3035) encoded in Unix epoch days.
"""

import os
import gc
import time

import fsspec
import numpy as np
import xarray as xr
import rioxarray
import rasterio
from dask.distributed import LocalCluster, Client
from nrt.monitor.iqr import IQR

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
BASE_DIR = os.path.expanduser("~/nrt-challenge-demo")
OUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

MASKS_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FOREST/NRT/CHALLENGE/data/forest_masks/10m/"
S2_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FOREST/NRT/CHALLENGE/data/sentinel2/"
STORAGE_OPTIONS = {"client_kwargs": {"trust_env": True}}

DEMO_PSUS = [4914, 242173, 141538, 28788, 172977]

# Limit thread contention
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"


def run_baseline_pipeline():
    print("Initializing local Dask cluster...")
    cluster = LocalCluster(n_workers=1, threads_per_worker=6)
    client = Client(cluster)
    print(f"Dask Dashboard available at: {client.dashboard_link}")

    for psu in DEMO_PSUS:
        print(f"\nProcessing PSU {psu:07d}...")
        s2_path = f"{S2_URL.rstrip('/')}/PSU_{psu:07d}.zarr"
        mask_path = f"{MASKS_URL.rstrip('/')}/mask_{psu:07d}.tif"
        out_tif = os.path.join(OUT_DIR, f"disturbance_date_{psu:07d}.tif")

        # 1. Ingest mandatory Tree Cover forest domain mask
        with rasterio.open(mask_path) as src:
            forest_mask = src.read(1)
            mask_bounds = src.bounds

        # 2. Open Sentinel-2 target Zarr data cube (10m & 20m SCL)
        ds = xr.open_dataset(s2_path, engine="zarr", group="r10m", chunks="auto", storage_options=STORAGE_OPTIONS)
        ds = ds.sel(time=slice("2019-01-01", "2023-05-31"))
        ds = ds.rio.clip_box(*mask_bounds)

        SCL = xr.open_dataset(s2_path, engine="zarr", group="r20m", chunks="auto", storage_options=STORAGE_OPTIONS).sel(time=slice("2019-01-01", "2023-05-31")).SCL
        SCL = SCL.interp_like(ds, method="nearest")
        SCL = SCL.rio.clip_box(*mask_bounds)

        # 3. Compute Vegetation Index (NDVI) and mask invalid observations
        ndvi = (ds["B08"] - ds["B04"]) / (ds["B08"] + ds["B04"])
        ndvi = ndvi.where(SCL.isin([4, 5, 7]))  # Vegetation, Bare soils, Unclassified
        ndvi = ndvi.where(~np.isinf(ndvi))

        # 4. Temporal Data Splitting (Baseline vs Monitoring)
        history_ds = ndvi.sel(time=slice("2019-01-01", "2021-12-31")).compute()
        monitoring_ds = ndvi.sel(time=slice("2022-01-01", "2023-05-31")).compute()
        ds.close()
        SCL.close()

        # 5. Fit IQR Baseline Model
        print(f"Fitting historical baseline for PSU {psu:07d}...")
        monitor = IQR(mask=forest_mask, trend=False, harmonic_order=1, sensitivity=4)
        monitor.fit(history_ds, method="RIRLS", maxiter=5, n_threads=6)

        # 6. Online Recursive Monitoring Simulation
        print(f"Running online monitoring for PSU {psu:07d}...")
        for array, date in zip(monitoring_ds.values, monitoring_ds.time.values.astype("datetime64[s]").tolist()):
            monitor.monitor(array=array, date=date)

        # 7. Export Disturbance Date Raster
        monitor.report(out_tif, layers=["detection_date"])
        print(f"✅ Saved prediction raster: {out_tif}")

        # Memory cleanup
        del ds, SCL, ndvi, history_ds, monitoring_ds, monitor
        fsspec.AbstractFileSystem.clear_instance_cache()
        gc.collect()
        time.sleep(1)

    client.close()
    cluster.close()
    print("\n✅ Baseline pipeline execution complete!")


if __name__ == "__main__":
    run_baseline_pipeline()
