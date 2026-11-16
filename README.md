# JRC NRT Forest Disturbance Monitoring Challenge

Individual entry to the JRC
[Near-Real-Time Forest Disturbance Monitoring Intercomparison Challenge](https://forest.jrc.ec.europa.eu/en/activities/jrc-near-real-time-forest-disturbance-monitoring-challenge/).

**Goal:** a conforming submission (one disturbance-date raster per PSU for 100
PSUs across Europe) plus a short method summary, before **31 January 2027**.

**Method (planned):** per-pixel recursive state-space model (level + seasonal
harmonics) updated by a Kalman filter, with a sequential probability ratio
test on the standardized innovations. Strictly online: a decision at time *t*
uses only observations at or before *t*.

The full rule set and all open questions are in
[docs/challenge_spec.md](docs/challenge_spec.md). Scores and experiments are
logged in [docs/results_log.md](docs/results_log.md).

## Layout

| Path | Content |
| --- | --- |
| `config/` | Configuration files (data layout, later: method parameters) |
| `data/` | Mirrored JRC data. Git-ignored |
| `docs/` | Challenge spec, results log, method summary |
| `notebooks/` | Exploration only; nothing the pipeline depends on |
| `outputs/` | Generated rasters and archives. Git-ignored |
| `scripts/` | Command-line entry points |
| `scripts/jrc_reference/` | JRC reference scripts, verbatim from the challenge page |
| `src/` | The monitoring method (phase 3) |
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
```

The downloader mirrors the JRC FTP layout under `data/`, is resumable (re-run
the same command), and logs failures to `data/download_failures.txt`.

Data: (c) European Union, licensed CC BY 4.0 (see `data/copyright.txt`).

## Phases

0. Setup and discovery
1. JRC baseline (IQR monitor from `nrt`) end to end on the 5 demo PSUs
2. Baseline on all 100 PSUs: safety-net submission
3. Kalman + SPRT monitor, tuned on the demo PSUs only
4. Method summary
