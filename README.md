# Bayesian T1DM

Bayesian forecasting and recommendation pipeline for personal Type 1 diabetes data.

## Mission

This repository turns Tandem pump, CGM, and activity exports into a research-grade analysis pipeline that can:

- forecast near-term glucose trajectories
- quantify uncertainty and calibration
- compare basal, bolus, and I/C ratio scenarios
- generate human-reviewable recommendations for further review

This is a research project and decision-support workbench, not a medical device or automated pump controller.

## Current Status

The active implementation is Python. Legacy R work is archived in [`archive/r/`](./archive/r/), and exploratory notebooks live in [`notebooks/`](./notebooks/).

What is currently in place:

- Tandem Source acquisition through `tconnectsync`
- raw API payload archival as the source of truth
- normalization of CGM, bolus, basal, and activity tables
- manifest generation and coverage checks
- 5-minute feature-table construction
- insulin activity and IOB expansion from bolus events
- Bayesian forecasting model scaffolding in PyMC
- recommendation scoring for candidate pump-setting changes
- sampler diagnostics in run outputs
- recommendation suppression policy tied to validation quality and signal availability
- test fixtures and validation checks
- technical documentation with equations and definitions

What still needs refinement:

- stronger validation on long real-world exports
- prior and threshold calibration against your own data
- richer basal schedule handling
- broader backtesting against simpler baselines
- potential simplification or reparameterization of the latent residual model after more real-data diagnostics

Supported collection path:

- API-first acquisition through `tconnectsync`
- browser or Playwright acquisition is not part of the active workflow
- manual CSV exports can still be staged locally and ingested, but repo-local `data/raw/` is not the canonical long-term home

## Quickstart

1. Create a virtual environment and install the package:

```bash
python3 -m venv bayesian-t1dm
source bayesian-t1dm/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

2. Confirm the CLI is available:

```bash
bayesian-t1dm --help
```

3. Inspect or validate raw Tandem exports:

```bash
bayesian-t1dm normalize-raw
bayesian-t1dm validate-raw
bayesian-t1dm ingest
```

4. Run the full pipeline:

```bash
bayesian-t1dm run
```

For faster real-data validation without the final recommendation fit:

```bash
bayesian-t1dm run --skip-recommendations
```

5. Run tests:

```bash
pytest
```

## Configuration

Credentials and acquisition settings belong in a root-level `.env` file that is never committed.

Required or commonly used variables:

- `TANDEM_SOURCE_EMAIL` or `TCONNECT_EMAIL`
- `TANDEM_SOURCE_PASSWORD` or `TCONNECT_PASSWORD`
- `TANDEM_SOURCE_REGION` or `TCONNECT_REGION`
- `TIMEZONE_NAME`
- `TANDEM_SOURCE_PUMP_SERIAL`, `TANDEM_PUMP_SERIAL`, or `PUMP_SERIAL_NUMBER` if required for your account

The active acquisition flow assumes Tandem Source is the authoritative cloud source for your Mobi data and that `tconnectsync` can query it directly.

## Common Commands

Collect up to a 30-day window:

```bash
bayesian-t1dm collect --start-date 2025-10-01 --end-date 2025-10-30
```

Backfill a longer range:

```bash
bayesian-t1dm backfill --start-date 2025-10-01 --end-date 2026-03-28
```

Validate raw files:

```bash
bayesian-t1dm validate-raw
```

Repair normalized `tconnectsync` windows from archived raw payloads:

```bash
bayesian-t1dm normalize-raw
```

Build coverage and manifest outputs:

```bash
bayesian-t1dm ingest
```

Run the forecasting and recommendation pipeline:

```bash
bayesian-t1dm run
```

Run time-aware validation without the final recommendation fit:

```bash
bayesian-t1dm run --skip-recommendations
```

## Data Flow

The active pipeline is:

1. Tandem Mobi syncs to Tandem Source through the mobile app
2. `tconnectsync` logs in and requests Tandem Source API windows
3. raw API responses are archived under the cloud raw tree
4. normalized CGM, bolus, basal, and activity tables are written beside the raw responses
5. a per-window manifest records coverage, hashes, timestamps, and pump identity
6. the repo can also ingest manual CSV exports staged locally, but the canonical scraped-data home is the cloud project folder
7. a canonical 5-minute time grid is built
8. insulin exposure and IOB are derived from bolus events
9. lagged, rolling, and calendar features are created
10. the Bayesian forecasting model is fit
11. scenario comparison and recommendation ranking are generated

## Review Flow

Canonical repair/review workflow:

1. `bayesian-t1dm normalize-raw`
2. `bayesian-t1dm ingest`
3. `bayesian-t1dm run --skip-recommendations`

This keeps raw-window repair, coverage review, and predictive validation separate from recommendation generation.

Recommended recipes:

- latest real window review:

```bash
bayesian-t1dm normalize-raw --window-id YYYY-MM-DD__YYYY-MM-DD
bayesian-t1dm ingest
bayesian-t1dm run --skip-recommendations
```

- latest 14-day review:

```bash
bayesian-t1dm run --skip-recommendations --eval-folds 2 --draws 200 --tune 200 --chains 2
```

## Storage Layout

Canonical project organization:

- code: `~/Projects/bayesian-t1dm`
- runtime outputs, scratch space, and intermediate artifacts: `~/ProjectsRuntime/bayesian-t1dm`
- cloud project home: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm`

Within the cloud project home:

- scraped/raw acquisition data lives under `data/raw/`
- final published outputs belong under `output/`
- routine working summaries now default to `~/ProjectsRuntime/bayesian-t1dm/output/`
- cloud `output/` should be used for intentional published/final artifacts rather than default working reports

Within the cloud raw tree, the acquisition flow writes:

- raw API payload archives under `data/raw/tconnectsync/<window_id>/raw/`
- normalized CSVs under `data/raw/tconnectsync/<window_id>/normalized/`
- per-window manifests under `data/raw/tconnectsync/<window_id>/window_manifest.csv`
- collection manifest at `data/raw/tandem_export_manifest.csv`

Repo-local `data/raw/` is a convenience/manual staging location, not the canonical scraped-data home.

## Repository Layout

- `src/bayesian_t1dm/` - importable Python package for ingest, acquisition, features, modeling, recommendation, evaluation, and reporting
- `notebooks/` - exploratory notebooks and experiments
- `archive/r/` - archived R prototypes and scratch code
- `docs/` - technical documentation
- `tests/` - fixtures and regression tests

## Outputs

The pipeline produces:

- coverage summaries showing how much history is available
- self-contained HTML review pages for coverage and run inspection
- processed tables with normalized timestamps and derived insulin exposure
- posterior forecasts with uncertainty intervals
- walk-forward metrics against a persistence baseline
- sampler diagnostics for every walk-forward fit and the final fit
- explicit data-quality summaries showing whether source windows are good, degraded, or broken
- recommendation policy metadata showing whether recommendations were generated, suppressed, or skipped
- recommendation summaries for human review when the validation and sampler gates pass, including per-recommendation confidence and warning flags
- runtime reports under `~/ProjectsRuntime/bayesian-t1dm/output/` by default

Default runtime review artifacts:

- `coverage.md`
- `coverage_review.html`
- `run_summary.md`
- `run_summary.json`
- `run_review.html`

## Limitations

- Tandem Source availability and `tconnectsync` behavior still affect acquisition.
- Tandem Source is undocumented, so API drift remains a risk.
- Recommendations are advisory and must be reviewed by you.
- The default insulin action curve is an approximation, not physiology ground truth.
- The Bayesian model is intentionally conservative because personal data can be sparse and irregular.
- A modest MAE win over persistence does not justify recommendations if interval coverage is poor or sampler diagnostics fail.

## Interpretation

Treat predictive discrimination, interval calibration, and recommendation confidence as separate questions.

- `run` may perform multiple expensive fits because walk-forward evaluation is time-aware.
- `--skip-recommendations` is the preferred fast path for real-data validation.
- incomplete source windows remain reviewable, but they suppress recommendations by policy.
- `run_summary.json` includes fit diagnostics and recommendation-policy status.
- `coverage_review.html` and `run_review.html` are the primary visual inspection artifacts for active-pipeline review.
- recommendation records include `confidence` and `flags` so low-signal cases are explicit in machine-readable output.
- An empty recommendation list can mean no scenario cleared the gain threshold, or that recommendations were intentionally suppressed or skipped.

## Technical Notes

For model equations, variable definitions, and recommendation math, see [`docs/technical.md`](./docs/technical.md).
