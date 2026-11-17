"""Tests of the Kalman + CUSUM monitor on synthetic series, including no look-ahead."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from nrtfd import kalman
from nrtfd.io import PsuCube
from nrtfd.monitor import KalmanCusumMonitor, MonitorConfig, epoch_day, run_cube

REPO = Path(__file__).resolve().parents[1]
SIGMA = 0.02

CONFIG = MonitorConfig(
    name="test",
    index="ndmi",
    scl_keep=(4, 5),
    basis="knots",
    basis_size=12,
    level_drift=0.6,
    seasonal_drift=0.3,
    prior_level_var=1e4,
    prior_seasonal_var=1e3,
    variance_bins=12,
    variance_shrink=2.0,
    var_discount=0.99,
    restart_weight=5.0,
    trained_var=2.5,
    gate=3.0,
    gate_min_obs=10,
    max_gated_run=10,
    shift=3.0,
    threshold=9.0,
    min_anomalies=4,
    strong_drop=8.0,
    min_strong_anomalies=3,
    min_obs=30,
    min_days=1095,
    rearm_days=730,
    retrain_days=365,
    report_start="2022-01-01",
    report_end="2023-05-31",
    event_end="2022-12-31",
)


def season(days: np.ndarray) -> np.ndarray:
    """A seasonal course with steep flanks, like deciduous forest."""
    return 0.3 + 0.2 * np.tanh(3.0 * np.sin(2.0 * np.pi * (days - 100.0) / 365.25))


def series(n_pixels: int, seed: int = 0, missing: float = 0.3):
    """Irregular acquisitions from 2018 to mid 2023 with noise and gaps."""
    rng = np.random.default_rng(seed)
    days = np.cumsum(rng.integers(2, 7, size=520)) + epoch_day("2018-01-02")
    days = days[days <= epoch_day("2023-07-15")]
    times = np.datetime64("1970-01-01T10:30", "ns") + days * np.timedelta64(1, "D")
    values = season(days)[:, None] + SIGMA * rng.standard_normal((days.size, n_pixels))
    values[rng.random(values.shape) < missing] = np.nan
    return times, days, values


def feed(config: MonitorConfig, times, values) -> KalmanCusumMonitor:
    monitor = KalmanCusumMonitor(values.shape[1], config)
    for date, row in zip(times, values):
        monitor.monitor(row, date)
    return monitor


def test_knot_weights_interpolate_and_wrap():
    for day in (0.0, 15.0, 200.3, 364.9, 365.25 * 3 + 10):
        h = kalman.regressors(day, "knots", 12)
        assert h.sum() == pytest.approx(1.0) and (h >= 0).all() and (h > 0).sum() <= 2
    b0, b1, w = kalman.season_position(365.0, 12)
    assert (b0, b1) == (11, 0) and 0.0 < w < 1.0


def test_harmonic_regressors():
    h = kalman.regressors(0.0, "harmonics", 2)
    assert np.allclose(h, [1, 1, 0, 1, 0])


def test_forecast_errors_are_standardised_after_training():
    times, days, values = series(300)
    monitor = KalmanCusumMonitor(values.shape[1], CONFIG)
    z = []
    for date, day, row in zip(times, days, values):
        monitor.monitor(row, date)
        if day >= epoch_day("2022-01-01"):
            z.append(monitor.z[np.isfinite(row)])
    z = np.concatenate(z)
    assert abs(z.mean()) < 0.1
    assert 0.8 < z.std() < 1.25


def test_stable_pixels_raise_almost_no_alarms():
    times, _, values = series(2000, seed=1)
    monitor = feed(CONFIG, times, values)
    assert (monitor.first_alarm() > 0).mean() < 0.005


def test_single_and_paired_outliers_do_not_raise_alarms():
    times, days, values = series(300, seed=2)
    start = int(np.searchsorted(days, epoch_day("2021-01-01")))
    for i in range(values.shape[1]):
        seen = np.flatnonzero(np.isfinite(values[:, i]))
        seen = seen[seen >= start]
        # of every 7 observations of the pixel, two in a row (or one) are outliers
        hit = np.arange(seen.size) % 7 < 1 + i % 2
        values[seen[hit], i] -= 0.4  # 20 sigma, as residual cloud would do
    monitor = feed(CONFIG, times, values)
    assert (monitor.first_alarm() > 0).mean() < 0.01


def test_drop_is_detected_within_min_anomalies_and_never_before_it():
    times, days, values = series(50, seed=4)
    changed = np.arange(10)
    t0 = int(np.searchsorted(days, epoch_day("2022-06-01")))
    values[t0:, changed] -= 0.2
    monitor = feed(CONFIG, times, values)
    alarm = monitor.first_alarm()
    for i in changed:
        seen = days[t0:][np.isfinite(values[t0:, i])]
        assert days[t0] <= alarm[i] <= seen[CONFIG.min_anomalies - 1]
    assert (alarm[10:] == 0).all()


@pytest.mark.parametrize(
    ("drop", "acquisitions"),
    [(0.4, CONFIG.min_strong_anomalies), (0.12, CONFIG.min_anomalies)],  # 20 and 6 sigma
)
def test_alarm_date_is_the_date_of_the_acquisition_that_raised_it(drop, acquisitions):
    times, days, values = series(5, seed=5, missing=0.0)
    t0 = int(np.searchsorted(days, epoch_day("2022-03-01")))
    values[t0:, 0] -= drop
    monitor = KalmanCusumMonitor(values.shape[1], CONFIG)
    raised_on = None
    for date, day, row in zip(times, days, values):
        if 0 in monitor.monitor(row, date) and raised_on is None:
            raised_on = day
    # a strong drop needs fewer acquisitions than a moderate one, and neither is back-dated
    assert raised_on == days[t0 + acquisitions - 1]
    assert monitor.first_alarm()[0] == raised_on


def test_shortcut_for_strong_drops_can_be_switched_off():
    times, days, values = series(5, seed=5, missing=0.0)
    t0 = int(np.searchsorted(days, epoch_day("2022-03-01")))
    values[t0:, 0] -= 0.4
    plain = dataclasses.replace(CONFIG, min_strong_anomalies=CONFIG.min_anomalies)
    assert feed(plain, times, values).first_alarm()[0] == days[t0 + CONFIG.min_anomalies - 1]


def test_pixel_disturbed_before_the_window_stays_silent():
    times, days, values = series(20, seed=6)
    t0 = int(np.searchsorted(days, epoch_day("2021-08-10")))
    t1 = int(np.searchsorted(days, epoch_day("2022-09-01")))
    values[t0:, :10] -= 0.2
    values[t1:, :10] -= 0.1  # a second drop inside the window
    monitor = feed(CONFIG, times, values)
    assert (monitor.first_alarm(start="2021-01-01")[:10] < epoch_day("2022-01-01")).all()
    assert (monitor.first_alarm()[:10] == 0).all()


def test_change_after_event_end_is_not_reported_but_a_late_alarm_is():
    times, days, values = series(20, seed=7, missing=0.0)
    late_change = int(np.searchsorted(days, epoch_day("2023-02-10")))
    december = int(np.searchsorted(days, epoch_day("2022-12-28")))
    values[late_change:, :10] -= 0.2
    values[december:, 10:] -= 0.2
    alarm = feed(CONFIG, times, values).first_alarm()
    assert (alarm[:10] == 0).all()
    assert (alarm[10:] > epoch_day("2022-12-31")).all()  # raised in January 2023
    # without the rule the 2023 changes are reported
    free = dataclasses.replace(CONFIG, event_end="2023-05-31")
    assert (feed(free, times, values).first_alarm()[:10] > 0).all()


def test_acquisitions_must_arrive_in_time_order():
    times, _, values = series(3)
    monitor = KalmanCusumMonitor(3, CONFIG)
    monitor.monitor(values[5], times[5])
    with pytest.raises(ValueError, match="time order"):
        monitor.monitor(values[4], times[4])
    with pytest.raises(ValueError, match="time order"):
        monitor.monitor(values[5], times[5])


def test_first_alarm_takes_the_earliest_alarm_in_the_window():
    monitor = KalmanCusumMonitor(4, CONFIG)
    monitor._alarm_days = [np.array([18900, 19000], np.int32), np.array([19100, 19600], np.int32)]
    monitor._alarm_pixels = [np.array([0, 1]), np.array([1, 2])]
    assert monitor.first_alarm().tolist() == [0, 19000, 0, 0]
    assert monitor.first_alarm(start="2021-01-01", end="2023-12-31").tolist() == [
        18900,
        19000,
        19600,
        0,
    ]


# --- no look-ahead, through the whole chain from stored values to alarms -----------------


def stored_cube(seed: int = 0) -> PsuCube:
    """A small cube of stored values (int16 bands, SCL) with changes at several dates."""
    rng = np.random.default_rng(seed)
    times, days, ndmi = series(64, seed=seed, missing=0.0)
    for k, date in enumerate(
        ("2021-05-01", "2022-02-01", "2022-07-01", "2022-11-01", "2023-03-01")
    ):
        ndmi[int(np.searchsorted(days, epoch_day(date))) :, 4 * k : 4 * k + 4] -= 0.2
    shape = (len(times), 8, 8)
    b8a = rng.integers(2500, 3500, size=shape)
    b11 = np.rint(b8a * (1 - ndmi.reshape(shape)) / (1 + ndmi.reshape(shape)))
    scl = np.where(rng.random(shape) < 0.3, 9, 4).astype("uint8")
    forest = np.ones((8, 8), bool)
    rows, cols = np.nonzero(forest)
    arrays = {"B8A": b8a.astype("int16"), "B11": b11.astype("int16"), "SCL": scl}
    return PsuCube(1, times, arrays, dict.fromkeys(arrays, (rows, cols)), forest, {})


def alarm_set(cube: PsuCube, until_day: int) -> set:
    days, pixels = run_cube(cube, CONFIG).alarms
    return {(int(d), int(p)) for d, p in zip(days, pixels) if d <= until_day}


def test_alarms_do_not_depend_on_later_acquisitions():
    full = stored_cube()
    everything = alarm_set(full, 10**6)
    cuts = [epoch_day(d) for d in ("2021-12-31", "2022-06-15", "2022-10-01", "2023-01-20")]
    assert len({d for d, _ in everything}) > len(cuts)  # alarms are spread over time
    rng = np.random.default_rng(9)
    for cut in cuts:
        n = int((full.times < np.datetime64(cut + 1, "D")).sum())
        expected = {a for a in everything if a[0] <= cut}
        assert expected, "the test needs alarms before the cut"

        # the series stops at the cut
        short = dataclasses.replace(
            full, times=full.times[:n], arrays={k: v[:n] for k, v in full.arrays.items()}
        )
        assert alarm_set(short, cut) == expected

        # everything after the cut is replaced by noise
        noisy = {k: v.copy() for k, v in full.arrays.items()}
        for name in ("B8A", "B11"):
            noisy[name][n:] = rng.integers(1, 6000, size=noisy[name][n:].shape)
        noisy["SCL"][n:] = rng.choice([4, 5, 9], size=noisy["SCL"][n:].shape)
        assert alarm_set(dataclasses.replace(full, arrays=noisy), cut) == expected


def test_nothing_after_the_window_is_read():
    class Recorder:
        def __init__(self, array):
            self.array, self.read = array, set()

        def __getitem__(self, t):
            self.read.add(t)
            return self.array[t]

    cube = stored_cube()
    recorders = {k: Recorder(v) for k, v in cube.arrays.items()}
    run_cube(dataclasses.replace(cube, arrays=recorders), CONFIG)
    last = int((cube.times < np.datetime64("2023-06-01")).sum()) - 1
    assert last < len(cube.times) - 1  # the cube does continue after the window
    for recorder in recorders.values():
        assert max(recorder.read) == last


# --- configuration ------------------------------------------------------------------------


def test_shipped_configurations_load():
    paths = [p for p in (REPO / "config").glob("*.yaml") if p.name != "challenge.yaml"]
    assert paths
    for path in paths:
        config = MonitorConfig.from_yaml(path)
        assert isinstance(config.prior_level_var, float) and isinstance(config.min_obs, int)
        assert epoch_day(config.report_start) == 18993


def test_configuration_rejects_unknown_and_missing_keys(tmp_path):
    good = next(p for p in (REPO / "config").glob("*.yaml") if p.name != "challenge.yaml")
    text = good.read_text(encoding="utf-8")
    bad = tmp_path / "typo.yaml"
    bad.write_text(text.replace("  gate:", "  gatee:"), encoding="utf-8")
    with pytest.raises(ValueError, match="gatee"):
        MonitorConfig.from_yaml(bad)
