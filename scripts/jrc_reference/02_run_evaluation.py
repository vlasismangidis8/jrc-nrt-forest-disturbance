#!/usr/bin/env python3
"""
NRT Forest Disturbance Challenge — Spatio-Temporal Accuracy Assessment
Evaluates prediction GeoTIFFs against JSON visual reference interpretations using nrt-validate.
"""

import os
import json
import urllib.request
import urllib.error

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
from pyproj import Transformer
import matplotlib.pyplot as plt

from nrt.validate.metrics import TemporalEvaluator
from nrt.validate.estimators import TwoStageClusterEstimator

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
BASE_DIR = os.path.expanduser("~/nrt-challenge-demo")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "data")
BASE_REF_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FOREST/NRT/CHALLENGE/demo/reference-samples/"
SSU_FGB_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FOREST/NRT/CHALLENGE/demo/demo_SSUs.fgb"
DEMO_SITE_RANGE = range(0, 100)

TARGET_RESOLUTION = "10m"

TREED_STATES = {"fully-treed", "partially-treed"}
DISTURBED_STATES = {"partial-dieback", "full-dieback", "non-treed"}

REF_DISTURBANCE_CUTOFF = "2022-12-31"
PRED_EVAL_END = "2023-05-31"


def epoch_days_from_iso(date_str: str) -> int:
    dt = datetime.fromisoformat(date_str)
    return (dt - datetime(1970, 1, 1)).days


def extract_disturbance_date(dates: list[str], labels: list[str]) -> int:
    if not dates or not labels:
        return 0
    if labels[0].lower().strip() not in TREED_STATES:
        return 0

    cutoff_epoch = epoch_days_from_iso(REF_DISTURBANCE_CUTOFF)
    for date_str, label in zip(dates, labels):
        if label.lower().strip() in DISTURBED_STATES:
            d_epoch = epoch_days_from_iso(date_str)
            return d_epoch if d_epoch <= cutoff_epoch else 0
    return 0


def load_and_index_ssu_references(base_ref_url: str, ssu_fgb_url: str, target_res: str = "10m") -> pd.DataFrame:
    print(f"Loading master SSU reference frame from FlatGeobuf ({ssu_fgb_url})...")
    gdf_ssus = gpd.read_file(ssu_fgb_url)

    psu_col = "psu_id" if "psu_id" in gdf_ssus.columns else "GRD_ID"
    
    ssu_meta = {}
    for idx_val, row in gdf_ssus.iterrows():
        site_key = int(row["idx"]) if "idx" in row else int(idx_val)
        ssu_meta[site_key] = {
            "psu_id": int(row[psu_col]),
            "stratum_ssu": int(row.get("stratum_ssu", 0)),
            "w1": float(row.get("w1", 250000 / len(gdf_ssus[psu_col].unique()))),
            "w2": float(row.get("w2", 1.0))
        }

    records = []
    fetched_count = 0
    print(f"Fetching remote site JSONs for target resolution '{target_res}'...")

    for site_idx in DEMO_SITE_RANGE:
        if site_idx not in ssu_meta:
            continue

        json_url = f"{base_ref_url.rstrip('/')}/site_{site_idx}_labels.json"
        meta = ssu_meta[site_idx]

        try:
            req = urllib.request.Request(json_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                fc = json.loads(resp.read().decode("utf-8"))
                fetched_count += 1

                for feat in fc.get("features", []):
                    props = feat.get("properties", {})
                    if props.get("resolution") != target_res:
                        continue

                    coords = feat["geometry"]["coordinates"][0]
                    lon = sum(c[0] for c in coords) / len(coords)
                    lat = sum(c[1] for c in coords) / len(coords)

                    dates = props.get("dates", [])
                    labels = props.get("labels", [])
                    y_true_epoch = extract_disturbance_date(dates, labels)

                    records.append({
                        "idx": site_idx,
                        "psu_id": meta["psu_id"],
                        "stratum_ssu": meta["stratum_ssu"],
                        "w1": meta["w1"],
                        "w2": meta["w2"],
                        "w_global": meta["w1"] * meta["w2"],
                        "lon": lon,
                        "lat": lat,
                        "is_edge": props.get("is_edge", False),
                        "y_true": y_true_epoch
                    })
        except urllib.error.HTTPError as e:
            if e.code != 404:
                print(f"⚠️ HTTP Error {e.code} fetching site {site_idx}")
        except Exception as e:
            print(f"⚠️ Failed parsing site {site_idx}: {e}")

    df_ref = pd.DataFrame(records)
    print(f"Retrieved {fetched_count} site JSONs. Compiled {len(df_ref)} SSUs across {df_ref['psu_id'].nunique()} PSUs.")
    return df_ref


def sample_prediction_rasters(df_ref: pd.DataFrame, pred_dir: str) -> pd.DataFrame:
    print("Sampling prediction rasters (disturbance_date_{psu_id}.tif)...")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    
    y_preds = []
    valid_mask = []
    open_rasters = {}

    try:
        for _, row in df_ref.iterrows():
            psu_id = int(row["psu_id"])
            tif_path = os.path.join(pred_dir, f"disturbance_date_{psu_id:07d}.tif")

            if not os.path.exists(tif_path):
                tif_path = os.path.join(pred_dir, f"disturbance_date_{psu_id}.tif")

            if not os.path.exists(tif_path):
                y_preds.append(np.nan)
                valid_mask.append(False)
                continue

            if psu_id not in open_rasters:
                open_rasters[psu_id] = rasterio.open(tif_path)

            src = open_rasters[psu_id]
            x_laea, y_laea = transformer.transform(row["lon"], row["lat"])
            val = list(src.sample([(x_laea, y_laea)]))[0][0]
            y_preds.append(int(val))
            valid_mask.append(True)
    finally:
        for src in open_rasters.values():
            src.close()

    df_eval = df_ref[valid_mask].copy()
    df_eval["y_pred"] = y_preds
    df_eval["y_pred"] = df_eval["y_pred"].astype(int)

    print(f"✅ Successfully matched and extracted predictions for {len(df_eval)} SSUs.")
    return df_eval


def run_assessment(plot: bool = True, save_path: str = None) -> pd.DataFrame:
    df_eval = load_and_index_ssu_references(BASE_REF_URL, SSU_FGB_URL, TARGET_RESOLUTION)
    df_eval = sample_prediction_rasters(df_eval, PREDICTIONS_DIR)

    psu_ids = df_eval["psu_id"].to_numpy()
    global_weights = df_eval["w_global"].to_numpy()
    strata_1_ids = df_eval.get("stratum_psu", pd.Series(np.zeros_like(psu_ids))).to_numpy()

    print(f"\nConfiguring TwoStageClusterEstimator:")
    print(f" • Sampled PSUs: {len(np.unique(psu_ids))}")
    print(f" • Mean Global Weight: {global_weights.mean():.2f} (Min: {global_weights.min():.2f}, Max: {global_weights.max():.2f})")

    estimator = TwoStageClusterEstimator(
        psu_ids=psu_ids,
        strata_1_ids=strata_1_ids,
        global_weights=global_weights,
        se_method="analytical",
    )

    exp_start = epoch_days_from_iso("2022-01-01")
    exp_end = epoch_days_from_iso(PRED_EVAL_END)

    evaluator = TemporalEvaluator(
        y_true=df_eval["y_true"].to_numpy(),
        y_pred=df_eval["y_pred"].to_numpy(),
        estimator=estimator,
        experiment_start=exp_start,
        experiment_end=exp_end,
    )

    lags = np.arange(0, 185, 5)
    print("\nComputing Bullock timeliness-accuracy curves (lag 0 to 180 days)...")
    df_metrics = evaluator.compute_curve(lags=lags, metrics=["ua", "pa", "f1"], negative_tolerance=30)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_metrics["lag"], df_metrics["f1"], label="F1-Score", color="#2563eb", linewidth=2.2)
        ax.plot(df_metrics["lag"], df_metrics["ua"], label="User's Accuracy (Precision)", color="#059669", linewidth=1.8, linestyle="--")
        ax.plot(df_metrics["lag"], df_metrics["pa"], label="Producer's Accuracy (Recall)", color="#d97706", linewidth=1.8, linestyle=":")

        ax.set_title(f"Timeliness–Accuracy Profile ({TARGET_RESOLUTION} Support)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Allowed Detection Lag (days)", fontsize=11)
        ax.set_ylabel("Estimated Accuracy Metric", fontsize=11)
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right", frameon=True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"📊 Accuracy curve saved to: {save_path}")

        plt.show()

    return df_metrics


if __name__ == "__main__":
    run_assessment()
