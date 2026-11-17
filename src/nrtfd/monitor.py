"""Streaming disturbance monitor built on the kernels in `nrtfd.kalman`."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml

from . import kalman
from .io import PsuCube

DAY = np.timedelta64(1, "D")
EPOCH = np.datetime64("1970-01-01")

# Spectral indices as normalised differences (a - b) / (a + b)
INDEX_BANDS = {"ndvi": ("B08", "B04"), "nbr": ("B8A", "B12"), "ndmi": ("B8A", "B11")}


def epoch_day(date) -> int:
    """Whole days since 1970-01-01 (UTC) of a date or timestamp."""
    return int((np.datetime64(date, "D") - EPOCH) / DAY)


@dataclass(frozen=True)
class MonitorConfig:
    """Every parameter of the method. Values come from a YAML file in config/.

    "sigma" below is the standard deviation of the one-step forecast error of a pixel.
    """

    name: str
    # input
    index: str  # ndvi, nbr or ndmi
    scl_keep: tuple[int, ...]  # SCL classes treated as clear
    # model
    basis: str  # "knots" or "harmonics" (see nrtfd.kalman)
    basis_size: int  # number of knots over the year, or number of harmonics
    level_drift: float  # random-walk std of the pixel's level per year, in sigma
    seasonal_drift: float  # same for each knot or harmonic coefficient on its own
    prior_level_var: float  # prior variance of the level, in sigma^2
    prior_seasonal_var: float  # prior variance of each knot or harmonic coefficient
    variance_bins: int  # season bins of the forecast error variance
    variance_shrink: float  # weight of the pooled variance against a bin's own estimate
    var_discount: float  # forgetting factor of the variance estimates, per observation
    restart_weight: float  # weight kept by the old pooled variance after a restart
    trained_var: float  # scaled forecast variance below which a forecast tests the model
    gate: float  # observations with |z| above this are not assimilated
    gate_min_obs: int  # informative errors needed before the gate and the test switch on
    max_gated_run: int  # consecutive rejected observations that force a silent restart
    # detector
    shift: float  # size of the drop the test is tuned to, in sigma
    threshold: float  # alarm when the log-likelihood ratio reaches this
    min_anomalies: int  # no alarm from fewer anomalous acquisitions than this
    strong_drop: float  # a forecast error below -strong_drop (in sigma) is a strong anomaly
    min_strong_anomalies: int  # no alarm from fewer strong anomalies than this
    min_obs: int  # informative errors needed before an alarm may be reported
    min_days: float  # days after a pixel's first observation before reporting
    rearm_days: float  # days after an alarm before the pixel may be reported again
    retrain_days: float  # days after a silent restart before reporting
    # window (inclusive dates)
    report_start: str  # first day whose alarms go into the raster
    report_end: str  # last day whose alarms go into the raster
    event_end: str  # an alarm is raised only if the change can have happened by this day

    @classmethod
    def from_yaml(cls, path: Path) -> MonitorConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        flat = {"name": raw.pop("name")}
        for section in raw.values():
            flat.update(section)
        known = {f.name for f in fields(cls)}
        if set(flat) != known:
            raise ValueError(
                f"{path}: unknown keys {sorted(set(flat) - known)}, "
                f"missing keys {sorted(known - set(flat))}"
            )
        # YAML is loose about types (1.0e4 is read as text, dates as date objects)
        for f in fields(cls):
            if f.type == "float":
                flat[f.name] = float(flat[f.name])
            elif f.type == "int":
                if int(flat[f.name]) != flat[f.name]:
                    raise ValueError(f"{path}: {f.name} must be a whole number")
                flat[f.name] = int(flat[f.name])
            elif f.type == "str":
                flat[f.name] = str(flat[f.name])
        flat["scl_keep"] = tuple(int(v) for v in flat["scl_keep"])
        if flat["index"] not in INDEX_BANDS:
            raise ValueError(f"{path}: index must be one of {sorted(INDEX_BANDS)}")
        if flat["basis"] not in ("knots", "harmonics"):
            raise ValueError(f"{path}: basis must be knots or harmonics")
        if not 1 <= flat["min_strong_anomalies"] <= flat["min_anomalies"]:
            raise ValueError(f"{path}: min_strong_anomalies must be 1 to min_anomalies")
        return cls(**flat)

    @property
    def arrays(self) -> tuple[str, ...]:
        """Cube arrays the method reads."""
        return (*INDEX_BANDS[self.index], "SCL")


class KalmanCusumMonitor:
    """One model per pixel, fed one acquisition at a time in time order.

    The monitor never holds more than the state after the last acquisition and the list
    of alarms raised so far, so no decision can use later data.
    """

    def __init__(self, n_pixels: int, config: MonitorConfig):
        self.config = c = config
        per_day = 1.0 / kalman.YEAR
        if c.basis == "knots":
            k = c.basis_size
            self._q_diag = np.full(k, c.seasonal_drift**2 * per_day)
            self._q_common = c.level_drift**2 * per_day
            self._prior_diag = np.full(k, c.prior_seasonal_var)
            self._prior_common = c.prior_level_var
        else:
            k = 1 + 2 * c.basis_size
            self._q_diag = np.full(k, c.seasonal_drift**2 * per_day)
            self._q_diag[0] = c.level_drift**2 * per_day
            self._q_common = 0.0
            self._prior_diag = np.full(k, c.prior_seasonal_var)
            self._prior_diag[0] = c.prior_level_var
            self._prior_common = 0.0
        self._clip = c.threshold / c.min_anomalies
        self._clip_strong = c.threshold / c.min_strong_anomalies
        self.m = np.zeros((n_pixels, k))
        self.C = np.full((n_pixels, k, k), self._prior_common)
        self.C[:, np.arange(k), np.arange(k)] += self._prior_diag
        self.s = np.zeros(n_pixels)  # pooled forecast error variance
        self.n_eff = np.zeros(n_pixels)
        self.sb = np.zeros((n_pixels, c.variance_bins))  # the same per season bin
        self.nb = np.zeros((n_pixels, c.variance_bins))
        self.n_var = np.zeros(n_pixels, np.int32)
        self.n_gated = np.zeros(n_pixels, np.int32)
        self.t_last = np.full(n_pixels, np.nan)
        self.t_arm = np.full(n_pixels, np.nan)
        self.t_clear = np.full(n_pixels, -np.inf)
        self._event_end = float(epoch_day(c.event_end)) + 1.0  # end of that day
        self.llr = np.zeros(n_pixels)
        self.z = np.full(n_pixels, np.nan)  # standardised forecast errors of the last call
        self._e = np.zeros(n_pixels)
        self._q = np.ones(n_pixels)
        self._var = np.zeros(n_pixels)
        self._fired = np.zeros(n_pixels, np.uint8)
        self._last_day = -np.inf
        self._alarm_days: list[np.ndarray] = []
        self._alarm_pixels: list[np.ndarray] = []

    def monitor(self, values: np.ndarray, date) -> np.ndarray:
        """Process one acquisition; return the indices of the pixels that raise an alarm.

        values: one index value per pixel, NaN where there is no valid observation.
        date:   acquisition time; must be later than every earlier call.
        """
        day = float((np.datetime64(date, "ns") - EPOCH) / DAY)
        if not day > self._last_day:
            raise ValueError("acquisitions must be processed in time order")
        self._last_day = day
        c = self.config
        y = np.ascontiguousarray(values, dtype=np.float64)
        h = kalman.regressors(day, c.basis, c.basis_size)
        b0, b1, w = kalman.season_position(day, c.variance_bins)
        kalman.forecast(
            y,
            h,
            day,
            self._q_diag,
            self._q_common,
            self.m,
            self.C,
            self.t_last,
            self.s,
            self.sb,
            self.nb,
            b0,
            b1,
            w,
            c.variance_shrink,
            self._e,
            self._q,
            self._var,
            self.z,
        )
        kalman.update(
            y,
            h,
            day,
            self._e,
            self._q,
            self._var,
            self.z,
            self.m,
            self.C,
            self.s,
            self.n_eff,
            self.sb,
            self.nb,
            b0,
            b1,
            w,
            self.n_var,
            self.n_gated,
            self.t_arm,
            self.t_clear,
            self.llr,
            self._fired,
            self._prior_diag,
            self._prior_common,
            c.var_discount,
            c.restart_weight,
            c.trained_var,
            c.gate,
            c.gate_min_obs,
            c.max_gated_run,
            c.shift,
            c.threshold,
            self._clip,
            self._clip_strong,
            c.strong_drop,
            c.min_obs,
            c.min_days,
            c.rearm_days,
            c.retrain_days,
            self._event_end,
        )
        pixels = np.flatnonzero(self._fired)
        if pixels.size:
            # The alarm carries the date of the acquisition that raised it, never an earlier one.
            self._alarm_pixels.append(pixels)
            self._alarm_days.append(np.full(pixels.size, int(np.floor(day)), np.int32))
        return pixels

    @property
    def alarms(self) -> tuple[np.ndarray, np.ndarray]:
        """All alarms so far, in the order they were raised: (epoch day, pixel index)."""
        if not self._alarm_days:
            return np.zeros(0, np.int32), np.zeros(0, np.intp)
        return np.concatenate(self._alarm_days), np.concatenate(self._alarm_pixels)

    def first_alarm(self, start=None, end=None) -> np.ndarray:
        """Per pixel, the day of its first alarm inside [start, end]; 0 if there is none."""
        start = epoch_day(self.config.report_start if start is None else start)
        end = epoch_day(self.config.report_end if end is None else end)
        days, pixels = self.alarms
        keep = (days >= start) & (days <= end)
        out = np.zeros(self.s.size, np.int32)
        # alarms are in time order; writing them backwards leaves the earliest per pixel
        out[pixels[keep][::-1]] = days[keep][::-1]
        return out


def index_values(cube: PsuCube, config: MonitorConfig, t: int) -> np.ndarray:
    """Spectral index of every forest pixel at time index `t`, NaN where not usable.

    Usable means: SCL class in `scl_keep` and both bands stored as a positive value
    (0 is the fill value of the cubes; negative reflectance makes the ratio meaningless).
    """
    name_a, name_b = INDEX_BANDS[config.index]
    a = cube.pixel_values(name_a, t).astype(np.float64)
    b = cube.pixel_values(name_b, t).astype(np.float64)
    ok = np.isin(cube.pixel_values("SCL", t), config.scl_keep) & (a > 0) & (b > 0)
    out = np.full(a.shape, np.nan)
    out[ok] = (a[ok] - b[ok]) / (a[ok] + b[ok])
    return out


def run_cube(cube: PsuCube, config: MonitorConfig) -> KalmanCusumMonitor:
    """Feed a PSU to a new monitor, acquisition by acquisition, up to the end of the window.

    Acquisitions after `report_end` are never read.
    """
    monitor = KalmanCusumMonitor(cube.n_pixels, config)
    stop = np.datetime64(config.report_end, "D") + DAY
    for t, date in enumerate(cube.times):
        if date >= stop:
            break
        monitor.monitor(index_values(cube, config, t), date)
    return monitor
