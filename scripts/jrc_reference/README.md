# JRC reference implementation scripts

Copied **verbatim** from Section 6 ("Reference Execution Pipeline & Demonstration")
of the challenge page on 2026-11-16:
https://forest.jrc.ec.europa.eu/en/activities/jrc-near-real-time-forest-disturbance-monitoring-challenge/

The scripts are embedded in the page as code blocks (`code-checker`,
`code-baseline`, `code-eval`); they are not on the JRC FTP.

| File | Purpose |
| --- | --- |
| `00_check_conformance.py` | Conformance check of submission rasters against the Tree Cover masks |
| `01_run_baseline_iqr.py` | IQR baseline (`nrt.monitor.iqr.IQR`) on the 5 demo PSUs |
| `02_run_evaluation.py` | Accuracy and timeliness with `nrt-validate` against the demo SSUs |

Do not edit these files. Project-specific wrappers belong in `scripts/`.
