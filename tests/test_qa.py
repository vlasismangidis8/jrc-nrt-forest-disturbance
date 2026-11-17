"""Tests for the label-free run checks (qa_run.py) and the cross-validation bookkeeping."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import cross_validate
import qa_run

CONFIG = {
    "report_start": "2022-01-01",
    "report_end": "2023-05-31",
    "rearm_days": 730,
    "min_days": 1095,
}


def day(date: str) -> int:
    return int(np.datetime64(date, "D").astype(np.int64))


def make_run(run_dir: Path, dates: np.ndarray, alarms: list[tuple[str, int, int]]) -> None:
    (run_dir / "rasters").mkdir(parents=True)
    (run_dir / "alarms").mkdir()
    profile = {
        "driver": "GTiff",
        "height": dates.shape[0],
        "width": dates.shape[1],
        "count": 1,
        "dtype": "int16",
        "crs": "EPSG:3035",
        "transform": from_origin(3_000_000.0, 2_000_000.0, 10.0, 10.0),
    }
    with rasterio.open(run_dir / "rasters" / "disturbance_date_0000007.tif", "w", **profile) as dst:
        dst.write(dates, 1)
    np.savez_compressed(
        run_dir / "alarms" / "PSU_0000007.npz",
        day=np.array([day(a[0]) for a in alarms]),
        row=np.array([a[1] for a in alarms], dtype=np.int16),
        col=np.array([a[2] for a in alarms], dtype=np.int16),
    )


def test_check_psu_counts_patches_single_cells_and_bursts(tmp_path):
    dates = np.zeros((20, 20), "int16")
    dates[2:6, 2:6] = day("2022-06-10")  # one patch of 16 pixels
    dates[15, 15] = day("2023-02-03")  # one pixel on its own
    alarms = [("2022-06-10", r, c) for r in range(2, 6) for c in range(2, 6)]
    alarms += [("2023-02-03", 15, 15), ("2021-06-01", 10, 10), ("2019-03-01", 11, 11)]
    make_run(tmp_path, dates, alarms)
    row = SimpleNamespace(
        psu=7, forest_px=400, first_date="2018-01-05", last_date="2023-07-30", dates=300
    )

    summary, by_month = qa_run.check_psu(row, tmp_path, CONFIG)

    assert summary["flagged_share"] == 17 / 400
    assert summary["share_2022"] == 16 / 400
    assert summary["share_2023"] == 1 / 400
    assert summary["single_cell_share_of_flagged"] == 1 / 17
    assert summary["peak_month_share"] == 16 / 400
    assert summary["silent_at_start_share"] == 1 / 400  # the 2021 alarm; 2019 is re-armed
    assert "alarmed in 2022-06" in summary["notes"]
    assert by_month.sum() == 19 / 400


def test_check_psu_notes_short_training_and_early_end(tmp_path):
    make_run(tmp_path, np.zeros((20, 20), "int16"), [])
    row = SimpleNamespace(
        psu=7, forest_px=400, first_date="2019-06-01", last_date="2023-04-30", dates=200
    )

    summary, _ = qa_run.check_psu(row, tmp_path, CONFIG)

    assert "training shorter" in summary["notes"]
    assert "cube ends 2023-04-30" in summary["notes"]
    assert summary["flagged_share"] == 0.0


def test_choose_takes_fewest_flagged_inside_the_plateau():
    net = np.array([10, 9, 7, 10])
    flagged = np.array([0.05, 0.01, 0.001, 0.03])
    assert cross_validate.choose(net, flagged, plateau=1) == 1
    assert cross_validate.choose(net, flagged, plateau=0) == 3


def test_hits_and_false_alarms_follow_the_evaluation_rules():
    module = SimpleNamespace(epoch_days_from_iso=day, PRED_EVAL_END="2023-05-31")
    event = day("2022-06-01")
    refs = pd.DataFrame({"y_true": [event, event, event, day("2021-08-01"), 0, event]})
    pred = np.array([event + 20, event - 31, event + 181, day("2022-02-01"), day("2022-03-01"), 0])

    hit, false_alarm, is_event = cross_validate.hits_and_false_alarms(
        refs, pred, module, "2022-01-01"
    )

    assert hit.tolist() == [True, False, False, False, False, False]
    assert false_alarm.tolist() == [False, True, True, True, True, False]
    assert is_event.tolist() == [True, True, True, False, False, True]
