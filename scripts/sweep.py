"""Try variants of a monitor configuration on the demo PSUs and log every one of them.

Each variant is the base configuration with some parameters replaced. To keep a variant
down to seconds, the monitor runs on the demo SSU pixels plus a fixed random sample of
forest pixels per PSU instead of on whole PSUs. Without `common_mode` that is exact for
the scores, because a pixel's result depends on nothing but its own series. With it the
common error of a date is estimated from the sample instead of from all forest pixels,
so the scores of a whole-PSU run can differ slightly.

For every variant one row is appended to docs/experiments.csv:
  - official window (2022-01-01 on): alarms, hits, kinds of error, UA / PA / F1
  - tuning window (2021-01-01 on, 30 events instead of 13): the same counts
  - share of the sampled forest pixels flagged, per PSU

Usage:
    python scripts/sweep.py --base config/kc_nbr_knots.yaml --grid gate=2.5,3.5 \
        --grid level_drift=0.5,1.0 --note "gate against level drift"
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import yaml
from pyproj import Transformer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import evaluate_run

from nrtfd.io import load_cube
from nrtfd.monitor import (
    DAY,
    KalmanCusumMonitor,
    MonitorConfig,
    index_values,
)

TUNING_START = "2021-01-01"
LOG = REPO / "docs" / "experiments.csv"
SEED = 0


def coerce(config: MonitorConfig, key: str, text: str):
    """Turn a command-line value into the type of the configuration field."""
    kind = {f.name: f.type for f in dataclasses.fields(config)}[key]
    if kind == "int":
        return int(text)
    if kind == "float":
        return float(text)
    if kind == "bool":
        return {"true": True, "false": False}[text.lower()]
    if key == "scl_keep":
        return tuple(int(v) for v in text.split("+"))
    return text


class DemoSeries:
    """Index series of the SSU pixels and of a random forest sample, per demo PSU."""

    def __init__(self, data_dir: Path, cache_dir: Path, psus: list[int], n_sample: int):
        self.data_dir, self.cache_dir, self.psus, self.n_sample = (
            data_dir,
            cache_dir,
            psus,
            n_sample,
        )
        self.module, self.refs = evaluate_run.load_references(data_dir)
        self._series: dict[tuple, dict] = {}

    def get(self, config: MonitorConfig) -> dict:
        key = (config.index, config.scl_keep)
        if key not in self._series:
            self._series[key] = {psu: self._load(psu, config) for psu in self.psus}
        return self._series[key]

    def _load(self, psu: int, config: MonitorConfig) -> dict:
        cube = load_cube(psu, self.data_dir, config.arrays, self.cache_dir)
        number = -np.ones(cube.forest.shape, dtype=np.intp)
        number[cube.forest] = np.arange(cube.n_pixels)
        refs = self.refs[self.refs.psu_id == psu]
        to_laea = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
        with rasterio.open(self.data_dir / "forest_masks" / "10m" / f"mask_{psu:07d}.tif") as src:
            cells = [
                src.index(*to_laea.transform(lon, lat)) for lon, lat in zip(refs.lon, refs.lat)
            ]
        ssu_pixel = np.array([number[r, c] for r, c in cells])  # -1: outside the forest mask
        rng = np.random.default_rng(SEED)
        sample = rng.choice(cube.n_pixels, size=min(self.n_sample, cube.n_pixels), replace=False)
        pixels = np.concatenate([ssu_pixel[ssu_pixel >= 0], sample])
        values = np.stack([index_values(cube, config, t)[pixels] for t in range(len(cube.times))])
        return {
            "times": cube.times,
            "values": values,
            "ssu_index": refs.index.to_numpy(),
            "ssu_in_forest": ssu_pixel >= 0,
            "n_ssu": int((ssu_pixel >= 0).sum()),
        }


def run_variant(series: dict, refs, config: MonitorConfig) -> tuple[dict, dict, dict]:
    """First alarms of the SSUs for both windows, and the flagged share of the samples."""
    official = np.zeros(len(refs), dtype=np.int64)
    tuning = np.zeros(len(refs), dtype=np.int64)
    position = {label: i for i, label in enumerate(refs.index)}
    flagged = {}
    stop = np.datetime64(config.report_end, "D") + DAY
    for psu, data in series.items():
        monitor = KalmanCusumMonitor(data["values"].shape[1], config)
        for t, date in enumerate(data["times"]):
            if date >= stop:
                break
            monitor.monitor(data["values"][t], date)
        rows = [position[label] for label in data["ssu_index"][data["ssu_in_forest"]]]
        official[rows] = monitor.first_alarm()[: data["n_ssu"]]
        tuning[rows] = monitor.first_alarm(start=TUNING_START)[: data["n_ssu"]]
        flagged[psu] = float((monitor.first_alarm()[data["n_ssu"] :] > 0).mean())
    return official, tuning, flagged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, required=True, help="base configuration (YAML)")
    ap.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="KEY=V1,V2",
        help="values to try for one parameter; repeat for a grid (scl_keep as 4+5,4+5+7)",
    )
    ap.add_argument(
        "--one-at-a-time",
        action="store_true",
        help="change one parameter at a time instead of trying every combination",
    )
    ap.add_argument("--note", default="", help="why this sweep is run")
    ap.add_argument("--sample", type=int, default=3000, help="random forest pixels per PSU")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    ap.add_argument("--cache-dir", type=Path, default=REPO / "data" / "cache")
    args = ap.parse_args()

    base = MonitorConfig.from_yaml(args.base)
    challenge = yaml.safe_load((REPO / "config" / "challenge.yaml").read_text(encoding="utf-8"))
    psus = list(challenge["demo_psus"])
    keys = [g.split("=", 1)[0] for g in args.grid]
    choices = [
        [coerce(base, k, v) for v in g.split("=", 1)[1].split(",")] for k, g in zip(keys, args.grid)
    ]
    git = {"capture_output": True, "text": True, "cwd": REPO}
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], check=False, **git
    ).stdout.strip()
    if subprocess.run(
        ["git", "status", "--porcelain", "src", "scripts", "config"], check=False, **git
    ).stdout:
        commit += "+uncommitted"

    demo = DemoSeries(args.data_dir, args.cache_dir, psus, args.sample)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new_log = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        log = csv.writer(f)
        if new_log:
            log.writerow(
                ["time", "commit", "base", "changes", "note"]
                + ["alarms", "hits", "fa_before_window", "fa_no_event", "misses"]
                + ["f1_30", "f1_60", "f1_180", "ua_180", "pa_180", "median_delay"]
                + ["tune_events", "tune_alarms", "tune_hits", "tune_fa_no_event", "tune_f1_180"]
                + [f"flagged_{psu}" for psu in psus]
                + ["parameters"]
            )
        print(
            "changes | official: alarms hits FAbefore FAnone miss | F1@30 @60 @180 UA PA delay"
            " | from 2021: alarms hits FAnone F1@180 | flagged % per PSU"
        )
        if args.one_at_a_time:
            variants = [{k: v} for k, values in zip(keys, choices) for v in values]
        else:
            variants = [dict(zip(keys, combo)) for combo in itertools.product(*choices)]
        for changes in variants:
            config = dataclasses.replace(base, **changes)
            official, tuning, flagged = run_variant(demo.get(config), demo.refs, config)
            results = []
            for start, pred in ((evaluate_run.OFFICIAL_START, official), (TUNING_START, tuning)):
                df = demo.refs.copy()
                df["y_pred"] = pred
                results.append(evaluate_run.score(demo.module, df, start, config.name)["summary"])
            o, t = results
            oc, tc = o["outcomes"], t["outcomes"]
            fa_before = oc.get("false alarm: disturbed before the window", 0)
            fa_none = oc.get("false alarm: no reference event", 0)
            misses = sum(v for k, v in oc.items() if k.startswith("miss"))
            text = " ".join(f"{k}={v}" for k, v in changes.items()) or "(base)"
            s = o["scores"]
            log.writerow(
                [time.strftime("%Y-%m-%d %H:%M"), commit, args.base.stem, text, args.note]
                + [o["alarms"], o["hits"], fa_before, fa_none, misses]
                + [s[30]["f1"], s[60]["f1"], s[180]["f1"], s[180]["ua"], s[180]["pa"]]
                + [o["median_delay_of_hits"]]
                + [t["reference_events"], t["alarms"], t["hits"]]
                + [tc.get("false alarm: no reference event", 0), t["scores"][180]["f1"]]
                + [f"{flagged[psu]:.4f}" for psu in psus]
                + [json.dumps(dataclasses.asdict(config), sort_keys=True)]
            )
            f.flush()
            print(
                f"{text} | {o['alarms']:2d} {o['hits']:2d} {fa_before:2d} {fa_none:2d} {misses:2d} | "
                f"{s[30]['f1']:.2f} {s[60]['f1']:.2f} {s[180]['f1']:.2f} {s[180]['ua']:.2f} "
                f"{s[180]['pa']:.2f} {o['median_delay_of_hits']} | "
                f"{t['alarms']:2d} {t['hits']:2d} {tc.get('false alarm: no reference event', 0):2d} "
                f"{t['scores'][180]['f1']:.2f} | "
                + " ".join(f"{100 * flagged[psu]:.1f}" for psu in psus)
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
