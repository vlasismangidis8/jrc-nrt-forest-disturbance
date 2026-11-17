"""Per-pixel recursive filter and sequential test, one acquisition at a time.

Model, for one pixel and one spectral index y:

    y_t = h_t' x_t + v_t,        v_t ~ N(0, sigma^2)
    x_t = x_s + w_t,             w_t ~ N(0, sigma^2 * Q * (t - s))

x describes the seasonal course of the index. Two bases are available (`basis`):

knots      x holds the value of the index at evenly spaced days of the year. h_t
           interpolates linearly between the two knots around the acquisition day. All
           knots share one random-walk component (the level of the pixel) and each has a
           small one of its own (the shape of the season).
harmonics  x holds a level and the coefficients of annual and higher harmonics.

The coefficients follow random walks, so the process noise grows with the number of days
since the pixel's previous observation and irregular sampling needs no special treatment.

sigma^2 is not known. Covariances are carried in units of sigma^2 (the scaled form of the
dynamic linear model, West and Harrison 1997, ch. 4) and the variance of the one-step
forecast errors is estimated recursively, both pooled and per season (bins over the day
of the year), because the model fits some seasons better than others: around leaf-out
and leaf-fall of deciduous forest the errors are several times larger than in summer.

Detection uses the forecast error standardised with the seasonal variance, z_t. Page's
CUSUM, written as a repeated sequential probability ratio test of "mean of z is -shift"
against "mean of z is 0", accumulates the log-likelihood ratio and raises an alarm when
it reaches a threshold. The contribution of one acquisition is capped, so that no alarm
can come from fewer than a chosen number of anomalous acquisitions. The cap has two
levels: a forecast error beyond `strong_drop` may contribute more, so that a large drop
needs fewer acquisitions than a small one.

Both functions below see exactly one acquisition. Nothing is stored except the state
after that acquisition, so a decision at time t cannot depend on later data.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

YEAR = 365.25


def season_position(day: float, n: int) -> tuple[int, int, float]:
    """Neighbouring points of an n-point grid over the year, and the weight of the second.

    Returns (b0, b1, w): the acquisition lies between grid points b0 and b1 and has weight
    1 - w on b0 and w on b1. The grid is periodic.
    """
    u = (day % YEAR) / YEAR * n
    b0 = int(np.floor(u)) % n
    return b0, (b0 + 1) % n, float(u - np.floor(u))


def regressors(day: float, basis: str, size: int) -> np.ndarray:
    """h_t for an acquisition at `day` (days since 1970-01-01, fractional)."""
    if basis == "knots":
        b0, b1, w = season_position(day, size)
        h = np.zeros(size)
        h[b0] += 1.0 - w
        h[b1] += w
        return h
    h = np.empty(1 + 2 * size)
    h[0] = 1.0
    for j in range(1, size + 1):
        h[2 * j - 1] = np.cos(2.0 * np.pi * j * day / YEAR)
        h[2 * j] = np.sin(2.0 * np.pi * j * day / YEAR)
    return h


@njit(parallel=True, cache=True)
def forecast(y, h, day, q_diag, q_common, m, C, t_last, s, sb, nb, b0, b1, w, shrink, e, q, var, z):
    """Time update and one-step forecast error for every pixel observed at `day`.

    y        (N,)       index value, NaN where the pixel has no valid observation
    q_diag   (k,)       process noise variance per day of each coefficient, units of sigma^2
    q_common scalar     process noise variance per day shared by all coefficients
    m, C     (N,k) (N,k,k)  state mean and scaled covariance, updated in place
    t_last   (N,)       day of the pixel's previous observation, NaN before the first
    s        (N,)       pooled estimate of the forecast error variance
    sb, nb   (N,B)      the same per season bin, and the weight behind each estimate
    b0,b1,w             season bins around `day` and the weight of the second
    shrink   scalar     weight of the pooled estimate when combined with a bin's own
    e, q, var, z (N,)   outputs: forecast error, its scaled variance, the error variance
                        used for this season, and the standardised error
    """
    k = h.shape[0]
    for i in prange(y.shape[0]):
        yi = y[i]
        if np.isnan(yi):
            z[i] = np.nan
            continue
        dt = day - t_last[i]
        if dt > 0.0:  # false for the first observation of the pixel (t_last is NaN)
            for a in range(k):
                if q_common > 0.0:
                    for b in range(k):
                        C[i, a, b] += q_common * dt
                C[i, a, a] += q_diag[a] * dt
        f = 0.0
        qi = 1.0
        for a in range(k):
            f += h[a] * m[i, a]
            acc = 0.0
            for b in range(k):
                acc += C[i, a, b] * h[b]
            qi += h[a] * acc
        ei = yi - f
        v0 = (nb[i, b0] * sb[i, b0] + shrink * s[i]) / (nb[i, b0] + shrink)
        v1 = (nb[i, b1] * sb[i, b1] + shrink * s[i]) / (nb[i, b1] + shrink)
        vi = (1.0 - w) * v0 + w * v1
        e[i] = ei
        q[i] = qi
        var[i] = vi
        t_last[i] = day
        if vi > 0.0:
            z[i] = ei / np.sqrt(vi * qi)
        else:
            z[i] = 0.0


@njit(parallel=True, cache=True)
def update(
    y,
    h,
    day,
    e,
    q,
    var,
    z,
    m,
    C,
    s,
    n_eff,
    sb,
    nb,
    b0,
    b1,
    w,
    n_var,
    n_gated,
    t_arm,
    t_clear,
    llr,
    fired,
    prior_diag,
    prior_common,
    var_discount,
    restart_weight,
    trained_var,
    gate,
    gate_min_obs,
    max_gated_run,
    shift,
    threshold,
    clip,
    clip_strong,
    strong_drop,
    min_obs,
    min_days,
    rearm_days,
    retrain_days,
    event_end,
):
    """Measurement update, variance update and sequential test for one acquisition.

    State per pixel, all updated in place:
    n_eff    discounted number of forecast errors behind the pooled variance s
    n_var    informative forecast errors since the model was (re)started
    n_gated  current run of rejected observations
    t_arm    first day on which an alarm of this pixel may be reported
    t_clear  day of the last tested observation that left no evidence of change
    llr      CUSUM statistic (log-likelihood ratio)
    fired    output: 1 where an alarm is raised by this acquisition
    """
    k = h.shape[0]
    half = 0.5 * shift * shift
    for i in prange(y.shape[0]):
        fired[i] = 0
        if np.isnan(y[i]):
            continue
        if np.isnan(t_arm[i]):  # first observation of this pixel
            t_arm[i] = day + min_days
        zi = z[i]
        ei = e[i]
        qi = q[i]
        # A forecast that rests on coefficients not learned yet says nothing about the pixel.
        informative = qi <= trained_var
        tested = informative and n_var[i] >= gate_min_obs
        rejected = tested and abs(zi) > gate
        restart = False
        retrain = 0.0

        if rejected:
            # Outlier or change: the observation does not touch the coefficients.
            n_gated[i] += 1
            if n_gated[i] >= max_gated_run:
                restart = True  # the model no longer describes the pixel
                retrain = retrain_days
        else:
            n_gated[i] = 0
            rh = np.empty(k)
            for a in range(k):
                acc = 0.0
                for b in range(k):
                    acc += C[i, a, b] * h[b]
                rh[a] = acc
            for a in range(k):
                m[i, a] += rh[a] * ei / qi
                for b in range(k):
                    C[i, a, b] -= rh[a] * rh[b] / qi

        if informative:
            # Squared error in units of the scaled forecast variance; large values are
            # winsorised so that outliers inform the variance without dominating it.
            v = ei * ei / qi
            if n_var[i] >= 3 and v > gate * gate * var[i]:
                v = gate * gate * var[i]
            n_eff[i] = var_discount * n_eff[i] + 1.0
            s[i] += (v - s[i]) / n_eff[i]
            if w < 1.0:
                nb[i, b0] = var_discount * nb[i, b0] + (1.0 - w)
                sb[i, b0] += (1.0 - w) * (v - sb[i, b0]) / nb[i, b0]
            if w > 0.0:
                nb[i, b1] = var_discount * nb[i, b1] + w
                sb[i, b1] += w * (v - sb[i, b1]) / nb[i, b1]
            n_var[i] += 1

        if tested:
            u = -shift * zi - half
            cap = clip_strong if zi <= -strong_drop else clip
            if u > cap:
                u = cap
            elif u < -clip:
                u = -clip
            total = llr[i] + u
            total = max(total, 0.0)
            if total == 0.0:
                t_clear[i] = day
            if total >= threshold:
                restart = True
                retrain = rearm_days
                # The change happened after the pixel last looked normal. It is reported
                # only if that can have been on or before `event_end`.
                if n_var[i] >= min_obs and day >= t_arm[i] and not t_clear[i] > event_end:
                    fired[i] = 1
            llr[i] = total

        if restart:
            # Learn the pixel again; stay silent while doing so. The mean keeps its
            # values, the covariance returns to the prior, the variances adapt quickly.
            for a in range(k):
                for b in range(k):
                    C[i, a, b] = prior_common
                C[i, a, a] += prior_diag[a]
            n_var[i] = 0
            n_gated[i] = 0
            llr[i] = 0.0
            n_eff[i] = min(n_eff[i], restart_weight)
            for b in range(nb.shape[1]):
                nb[i, b] = min(nb[i, b], 1.0)
            t_arm[i] = max(t_arm[i], day + retrain)
