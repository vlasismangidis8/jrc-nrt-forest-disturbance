"""Leave-one-PSU-out check of the parameter choice on the demo PSUs.

The demo scores of the shipped configuration are in-sample: its parameters were chosen on
the events they are scored on. This script measures how much of the score survives when
the choice is made without the PSU that is scored:

  1. Every variant of a parameter grid is run once on the demo SSU pixels and on a random
     sample of forest pixels per PSU (as in `sweep.py`).
  2. For each demo PSU in turn, a variant is chosen by a fixed rule from the other four
     PSUs only, and its alarms on the held-out PSU are kept.
  3. The kept alarms of the five folds are scored together with the JRC evaluation code.

The rule is the one the shipped configuration was chosen by (docs/results_log.md): take
the variants whose hits minus false alarms in the tuning window (from 2021-01-01) are
within `--plateau` of the best, and among them the one that flags the smallest share of
the sampled forest pixels. With a plateau of 0 the flagged share only breaks ties. Both
rules are run by default.

Only the choice inside the grid is cross-validated. The structure of the method (seasonal
knots, monthly variance, the `event_end` rule) was also decided with all five PSUs in
view, and that cannot be repeated by a script.

Writes one row per fold to docs/cross_validation.csv and prints the pooled scores.

Usage:
    python scripts/cross_validate.py --base config/kalman_cusum.yaml \
        --grid index=ndmi,nbr --grid gate=2.0,2.5,3.0,3.5 --grid level_drift=0.3,0.6,1.0 \
        --grid shift=2.0,2.5,3.0 --grid min_anomalies=3,4
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import evaluate_run
import sweep

from nrtfd.monitor import MonitorConfig

LOG = REPO / "docs" / "cross_validation.csv"
MAX_DELAY = 180


def hits_and_false_alarms(refs: pd.DataFrame, pred: np.ndarray, module, start: str):
    """Per SSU: is its alarm a hit, is it a false alarm (the bookkeeping of `evaluate_run`)."""
    first = module.epoch_days_from_iso(start)
    last = module.epoch_days_from_iso(module.PRED_EVAL_END)
    truth = refs["y_true"].to_numpy()
    event = (truth >= first) & (truth <= last)
    alarm = (pred >= first) & (pred <= last)
    delay = pred - truth
    hit = event & alarm & (delay >= -30) & (delay <= MAX_DELAY)
    return hit, alarm & ~hit, event


def choose(net: np.ndarray, flagged: np.ndarray, plateau: int) -> int:
    """Index of the variant the rule picks: near-best net hits, then the fewest pixels flagged."""
    candidates = np.flatnonzero(net >= net.max() - plateau)
    return int(candidates[np.argmin(flagged[candidates])])


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, required=True, help="shipped configuration (YAML)")
    ap.add_argument("--grid", action="append", default=[], metavar="KEY=V1,V2", required=True)
    ap.add_argument(
        "--plateau",
        type=int,
        nargs="+",
        default=[1, 0],
        help="tolerated loss of net hits; one cross-validation per value",
    )
    ap.add_argument("--sample", type=int, default=3000, help="random forest pixels per PSU")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    ap.add_argument("--cache-dir", type=Path, default=REPO / "data" / "cache")
    args = ap.parse_args()

    base = MonitorConfig.from_yaml(args.base)
    challenge = yaml.safe_load((REPO / "config" / "challenge.yaml").read_text(encoding="utf-8"))
    psus = list(challenge["demo_psus"])
    keys = [g.split("=", 1)[0] for g in args.grid]
    choices = [
        [sweep.coerce(base, k, v) for v in g.split("=", 1)[1].split(",")]
        for k, g in zip(keys, args.grid)
    ]
    variants = [dict(zip(keys, combo)) for combo in itertools.product(*choices)]
    shipped = {k: getattr(base, k) for k in keys}
    if shipped not in variants:
        print(f"the grid does not contain the shipped values {shipped}")
        return 1

    demo = sweep.DemoSeries(args.data_dir, args.cache_dir, psus, args.sample)
    refs, module = demo.refs, demo.module
    psu_of = refs["psu_id"].to_numpy()
    official, tuning, flagged = [], [], []
    for number, changes in enumerate(variants, 1):
        config = dataclasses.replace(base, **changes)
        o, t, share = sweep.run_variant(demo.get(config), refs, config)
        official.append(o)
        tuning.append(t)
        flagged.append([share[psu] for psu in psus])
        if number % 20 == 0:
            print(f"{number} of {len(variants)} variants run", flush=True)
    official, tuning, flagged = np.array(official), np.array(tuning), np.array(flagged)

    # Net hits of every variant in the tuning window, per PSU.
    net = np.zeros((len(variants), len(psus)), dtype=int)
    for v in range(len(variants)):
        hit, false_alarm, _ = hits_and_false_alarms(refs, tuning[v], module, sweep.TUNING_START)
        for p, psu in enumerate(psus):
            net[v, p] = int(hit[psu_of == psu].sum()) - int(false_alarm[psu_of == psu].sum())

    i_shipped = variants.index(shipped)
    for plateau in args.plateau:
        kept_official = np.zeros(len(refs), dtype=np.int64)
        kept_tuning = np.zeros(len(refs), dtype=np.int64)
        rows = []
        for p, psu in enumerate(psus):
            others = [q for q in range(len(psus)) if q != p]
            pick = choose(net[:, others].sum(axis=1), flagged[:, others].mean(axis=1), plateau)
            held = psu_of == psu
            kept_official[held] = official[pick, held]
            kept_tuning[held] = tuning[pick, held]
            row = {
                "held_out_psu": psu,
                "chosen": " ".join(f"{k}={v}" for k, v in variants[pick].items()),
            }
            row["is_shipped"] = pick == i_shipped
            for label, pred, start in (
                ("official", official, evaluate_run.OFFICIAL_START),
                ("tuning", tuning, sweep.TUNING_START),
            ):
                for name, v in (("chosen", pick), ("shipped", i_shipped)):
                    hit, false_alarm, event = hits_and_false_alarms(refs, pred[v], module, start)
                    row[f"{label}_events"] = int(event[held].sum())
                    row[f"{label}_hits_{name}"] = int(hit[held].sum())
                    row[f"{label}_false_alarms_{name}"] = int(false_alarm[held].sum())
            row["flagged_chosen"] = round(float(flagged[pick, p]), 4)
            row["flagged_shipped"] = round(float(flagged[i_shipped, p]), 4)
            rows.append(row)
        folds = pd.DataFrame(rows)
        folds.insert(0, "plateau", plateau)
        folds.insert(0, "grid", "; ".join(args.grid))
        folds.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)

        print(
            f"\n{len(variants)} variants, rule: net hits within {plateau} of the best, "
            "then the smallest flagged share"
        )
        for row in rows:
            print(
                f"held out {row['held_out_psu']:07d}: chose {row['chosen']}"
                f"{' (the shipped one)' if row['is_shipped'] else ''}; official window "
                f"{row['official_hits_chosen']} of {row['official_events']} events hit, "
                f"{row['official_false_alarms_chosen']} false alarms "
                f"(shipped: {row['official_hits_shipped']} and {row['official_false_alarms_shipped']})"
            )
        for label, start, cv, own in (
            ("official window", evaluate_run.OFFICIAL_START, kept_official, official[i_shipped]),
            ("tuning window from 2021", sweep.TUNING_START, kept_tuning, tuning[i_shipped]),
        ):
            print(f"\n{label}: cross-validated against in-sample (shipped configuration)")
            results = []
            for pred in (cv, own):
                df = refs.copy()
                df["y_pred"] = pred
                results.append(evaluate_run.score(module, df, start, "cv")["summary"])
            a, b = results
            print(
                f"  events {a['reference_events']}, hits {a['hits']} | {b['hits']}, "
                f"alarms {a['alarms']} | {b['alarms']}, "
                f"median delay {a['median_delay_of_hits']} | {b['median_delay_of_hits']}"
            )
            print("  lag   UA cv  PA cv  F1 cv | UA     PA     F1 in-sample")
            for lag in evaluate_run.LAGS:
                x, y = a["scores"][lag], b["scores"][lag]
                print(
                    f"  {lag:3d}   {x['ua']:.3f}  {x['pa']:.3f}  {x['f1']:.3f} | "
                    f"{y['ua']:.3f}  {y['pa']:.3f}  {y['f1']:.3f}"
                )
    print(f"\nfolds appended to {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
