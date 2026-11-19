# JRC NRT Forest Disturbance Monitoring Challenge

Individual entry to the JRC
[Near-Real-Time Forest Disturbance Monitoring Intercomparison Challenge](https://forest.jrc.ec.europa.eu/en/activities/jrc-near-real-time-forest-disturbance-monitoring-challenge/).

**Goal:** a conforming submission (one disturbance-date raster per PSU for 100
PSUs across Europe) plus a short method summary, before **31 January 2027**.

**Method:** one recursive state-space model per forest pixel (a seasonal
profile of the NDMI on 12 knots with a common level), updated by a Kalman
filter at irregular time steps. Forecast errors are standardised with a
seasonal variance and fed to a CUSUM, a repeated sequential probability ratio
test; an alarm needs four anomalous acquisitions, or three strong ones. A
drop that the forest pixels of a PSU share on an acquisition (a dry summer)
is taken out of the evidence first. Strictly online: a decision at time *t*
uses only observations at or before *t*. The code is in `src/nrtfd/`, the
parameters in `config/kalman_cusum.yaml`.

## Layout

| Path | Content |
| --- | --- |
| `config/` | Configuration files: data layout (`challenge.yaml`), method parameters (`kalman_cusum.yaml`) |
| `data/` | Mirrored JRC data. Git-ignored |
| `docs/` | Working notes, experiment logs and the method summary. Kept locally, not part of the repository |
| `notebooks/` | Exploration only; nothing the pipeline depends on |
| `outputs/` | Generated rasters and archives. Git-ignored |
| `scripts/` | Command-line entry points |
| `scripts/jrc_reference/` | JRC reference scripts, verbatim from the challenge page |
| `src/nrtfd/` | The monitoring method: `kalman.py` (filter and test), `monitor.py`, `io.py` |
| `tests/` | Unit tests, including the no-look-ahead test |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (uv installs it if missing).

```
uv sync
```

Exact dependency versions are pinned in `uv.lock`.

On this machine uv was installed with `py -3.14 -m pip install --user uv` and is
called as `py -3.14 -m uv ...` (it is not on `PATH`).

### TLS note

The development machine runs an antivirus that re-signs HTTPS traffic with its
own root certificate, trusted only by the Windows certificate store. Tools that
ship their own CA bundle fail with `CERTIFICATE_VERIFY_FAILED`. The project
therefore verifies against the OS store: `system-certs = true` for uv (in
`pyproject.toml`) and `truststore` inside the Python scripts. Certificate
verification is never disabled.

## Data

```
uv run python scripts/download_data.py --psus demo --dry-run   # list and size only
uv run python scripts/download_data.py --psus demo             # 5 demo PSUs
uv run python scripts/verify_cubes.py                          # completeness and readability
```

The downloader mirrors the JRC FTP layout under `data/`, is resumable (re-run
the same command), and logs failures to `data/download_failures.txt`.

To keep the load on the JRC server low, the demo cubes were mirrored with the
settings below (one connection, at most 5 requests per second, six arrays),
which took 9.5 hours:

```
uv run python scripts/download_data.py --psus demo --what cubes \
    --bands B04 B08 B8A B11 B12 SCL \
    --workers 1 --max-rate 5 --stall-cooldown 300 --max-stalls 150
```

All 100 PSUs are mirrored in two stages with the same settings and
`--psus all`: first `--bands B8A B11 B12 SCL` (the 20 m arrays, all the Kalman
monitor needs, about 2 days), then the six arrays above (adds the 10 m bands
of the JRC baseline, about 3 more days). Progress is in `data/download_all.log`.

Data: (c) European Union, licensed CC BY 4.0 (see `data/copyright.txt`).

## JRC reference pipeline

```
uv run python scripts/inspect_cube.py --psu 4914     # facts about one cube, one figure
uv run python scripts/run_reference.py baseline      # IQR baseline, 5 demo PSUs
uv run python scripts/run_reference.py baseline --psus all   # every cube that has B04 and B08
uv run python scripts/run_reference.py check         # JRC conformance checker
uv run python scripts/run_reference.py evaluate      # nrt-validate on the 100 demo SSUs
```

`run_reference.py` runs the files in `scripts/jrc_reference/` as they are. It
only replaces their path constants, so that they read the local copy in `data/`
instead of the JRC server and write to `outputs/baseline_iqr/` instead of
`~/nrt-challenge-demo`. `check` and `evaluate` take `--run-dir` and work on the
rasters of any method.

## Kalman + CUSUM monitor

```
uv run python scripts/run_monitor.py --config config/kalman_cusum.yaml             # 5 demo PSUs
uv run python scripts/run_monitor.py --config config/kalman_cusum.yaml --psus all  # all downloaded PSUs
uv run python scripts/run_reference.py check    --run-dir outputs/kalman_cusum
uv run python scripts/run_reference.py evaluate --run-dir outputs/kalman_cusum
uv run python scripts/evaluate_run.py --run-dir outputs/kalman_cusum    # scores and every SSU behind them
uv run python scripts/qa_run.py --run-dir outputs/kalman_cusum          # checks that need no labels
uv run python scripts/build_archive.py --run-dir outputs/kalman_cusum --name <archive name>
uv run python scripts/build_submission.py --run-dir outputs/kalman_cusum --name <archive name>
```

`run_monitor.py` is resumable: it skips PSUs whose raster exists and PSUs whose
cube is not completely downloaded. `build_archive.py` writes the `.tar.gz` only
if all 100 PSUs are present and every check passes. All 100 PSUs take about
45 minutes on a 4-core laptop. `build_submission.py` unpacks the archive, runs
the checks again on what comes out, renders the method summary
(`docs/method_summary.md`) to a one-page PDF (needs pandoc and Edge or Chrome)
and puts both into `outputs/submission/`. Nothing is sent by any script.

Parameters are tuned on the demo PSUs only, with `scripts/sweep.py`, which
appends every variant it tries to `docs/experiments.csv`:

```
uv run python scripts/sweep.py --base config/kalman_cusum.yaml --grid gate=2.5,3.0 --grid shift=2.5,3.0
```

`scripts/cross_validate.py` repeats the parameter choice five times, each time
without the PSU that is then scored (results in `docs/cross_validation.csv`).
`scripts/qa_run.py` is for the 95 PSUs without labels: it lists, per PSU, the
share of forest flagged, the months in which the alarms fall, how many flagged
pixels are scattered single cells, and whether the cube covers the training
period and the window.

Tests (`uv run pytest`) include the no-look-ahead test: alarms up to a date
must not change when later acquisitions are removed or replaced by noise.

## Status

0. Setup and discovery: done
1. JRC baseline (IQR monitor from `nrt`) end to end on the 5 demo PSUs: done
2. Baseline on all 100 PSUs, safety-net submission: waits for the 10 m bands
3. Kalman + CUSUM monitor: done. All 100 PSUs are processed, every raster
   passes the conformance checks, and two archives are built:
   `outputs/kalman_cusum/mangidis_kalman_cusum.tar.gz` (with the landscape
   term, the current configuration) and
   `outputs/kalman_cusum_v2/mangidis_kalman_cusum_v2.tar.gz` (without it)
4. Method summary: one page. The submission (archive with the landscape term
   and the summary as PDF) is assembled in `outputs/submission/`

As of 2026-11-19.

## Licence

The code is released under the MIT licence, see [LICENSE](LICENSE). The
three scripts in `scripts/jrc_reference/` are copied from the JRC challenge
page ((c) European Union) and are not covered by it. The data is
(c) European Union, licensed CC BY 4.0.
