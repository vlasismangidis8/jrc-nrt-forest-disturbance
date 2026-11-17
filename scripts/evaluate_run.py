"""Score a run on the demo SSUs and show what is behind the scores.

Uses the same functions as the JRC evaluation script (reference dates, raster sampling,
`TwoStageClusterEstimator`, `TemporalEvaluator`), so with the default window the UA / PA
/ F1 values equal those of `run_reference.py evaluate`. On top of the curve it lists
every SSU with a reference event or an alarm and counts the kinds of error.

`--start` moves the start of the evaluation window. With 2021-01-01 the 17 demo events of
2021 count as events too, which gives 30 events instead of 13 to judge detection on. That
is a tuning aid only and needs rasters written with `run_monitor.py --also-from`.

Usage:
    python scripts/evaluate_run.py --run-dir outputs/kc_nbr_knots
    python scripts/evaluate_run.py --run-dir outputs/kc_nbr_knots --start 2021-01-01
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_reference

LAGS = (0, 10, 30, 60, 90, 180)
OFFICIAL_START = "2022-01-01"


def date_of(day: int) -> str:
    return str(np.datetime64(int(day), "D")) if day > 0 else "-"


def load_references(data_dir: Path):
    """The demo SSUs with their reference dates, read by the JRC evaluation code."""
    module = run_reference.load_reference("02_run_evaluation.py")
    with contextlib.redirect_stdout(io.StringIO()):
        df = module.load_and_index_ssu_references(
            (data_dir / "demo" / "reference-samples").resolve().as_uri(),
            str(data_dir / "demo" / "demo_SSUs.fgb"),
            module.TARGET_RESOLUTION,
        )
    return module, df


def evaluate(run_dir: Path, data_dir: Path, start: str, rasters: Path) -> dict:
    module, df = load_references(data_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        df = module.sample_prediction_rasters(df, str(rasters))
    return score(module, df, start, run_dir.name)


def score(module, df, start: str, name: str) -> dict:
    """UA / PA / F1 curve and per-SSU outcomes for predictions in the column `y_pred`."""
    from nrt.validate.estimators import TwoStageClusterEstimator
    from nrt.validate.metrics import TemporalEvaluator

    df = df.copy()
    first = module.epoch_days_from_iso(start)
    last = module.epoch_days_from_iso(module.PRED_EVAL_END)

    estimator = TwoStageClusterEstimator(
        psu_ids=df["psu_id"].to_numpy(),
        strata_1_ids=np.zeros(len(df), dtype=int),
        global_weights=df["w_global"].to_numpy(),
        se_method="analytical",
    )
    evaluator = TemporalEvaluator(
        y_true=df["y_true"].to_numpy(),
        y_pred=df["y_pred"].to_numpy(),
        estimator=estimator,
        experiment_start=first,
        experiment_end=last,
    )
    curve = evaluator.compute_curve(
        lags=np.arange(0, 185, 5), metrics=["ua", "pa", "f1"], negative_tolerance=30
    )

    # The same bookkeeping per SSU, to see which units make up the scores.
    df["ref"] = (df.y_true >= first) & (df.y_true <= last)
    df["ref_before"] = (df.y_true > 0) & (df.y_true < first)
    df["alarm"] = (df.y_pred >= first) & (df.y_pred <= last)
    df["delay"] = np.where(df.ref & df.alarm, df.y_pred - df.y_true, np.nan)
    hit = df.ref & df.alarm & (df.delay >= -30) & (df.delay <= 180)
    df["outcome"] = ""
    df.loc[hit, "outcome"] = "hit"
    df.loc[df.ref & ~df.alarm, "outcome"] = "miss: no alarm"
    df.loc[df.ref & df.alarm & (df.delay > 180), "outcome"] = "miss: alarm too late"
    df.loc[df.ref & df.alarm & (df.delay < -30), "outcome"] = "miss: alarm too early"
    df.loc[df.alarm & df.ref_before, "outcome"] = "false alarm: disturbed before the window"
    df.loc[df.alarm & ~df.ref & ~df.ref_before, "outcome"] = "false alarm: no reference event"
    df["ref_date"] = df.y_true.map(date_of)
    df["alarm_date"] = df.y_pred.map(date_of)

    shown = df[df.outcome != ""].sort_values(["psu_id", "idx"])
    columns = ["idx", "psu_id", "stratum_ssu", "w2", "ref_date", "alarm_date", "delay", "outcome"]
    summary = {
        "run": name,
        "window_start": start,
        "ssus": len(df),
        "reference_events": int(df.ref.sum()),
        "alarms": int(df.alarm.sum()),
        "hits": int(hit.sum()),
        "median_delay_of_hits": float(df.delay[hit].median()) if hit.any() else None,
        "outcomes": {k: int(v) for k, v in shown.outcome.value_counts().items()},
        "scores": {
            int(row.lag): {"ua": round(row.ua, 3), "pa": round(row.pa, 3), "f1": round(row.f1, 3)}
            for row in curve.itertuples()
            if row.lag in LAGS
        },
    }
    return {"summary": summary, "details": shown[columns], "curve": curve}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    ap.add_argument("--start", default=OFFICIAL_START, help="start of the evaluation window")
    ap.add_argument("--quiet", action="store_true", help="print the summary only")
    args = ap.parse_args()

    official = args.start == OFFICIAL_START
    rasters = args.run_dir / ("rasters" if official else f"rasters_from_{args.start}")
    if not rasters.is_dir():
        print(f"{rasters} does not exist (run_monitor.py --also-from {args.start})")
        return 1
    result = evaluate(args.run_dir, args.data_dir, args.start, rasters)
    summary = result["summary"]
    tag = "" if official else f"_from_{args.start}"
    result["details"].to_csv(args.run_dir / f"details{tag}.csv", index=False)
    (args.run_dir / f"summary{tag}.json").write_text(json.dumps(summary, indent=2))

    label = "official window" if official else "tuning window, not the official score"
    print(f"{summary['run']}: window from {args.start} ({label})")
    print(
        f"  {summary['reference_events']} reference events, {summary['alarms']} alarms on "
        f"{summary['ssus']} SSUs, {summary['hits']} hits, "
        f"median delay of hits {summary['median_delay_of_hits']} days"
    )
    for outcome, count in summary["outcomes"].items():
        print(f"  {count:3d}  {outcome}")
    print("  lag   UA     PA     F1")
    for lag, s in summary["scores"].items():
        print(f"  {lag:3d}  {s['ua']:.3f}  {s['pa']:.3f}  {s['f1']:.3f}")
    if not args.quiet:
        print(result["details"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
