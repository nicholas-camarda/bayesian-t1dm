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
- test fixtures and validation checks
- technical documentation with equations and definitions

What still needs refinement:

- stronger validation on long real-world exports
- prior and threshold calibration against your own data
- richer basal schedule handling
- explicit CGM gap and missingness indicators
- meal and carb features if Tandem Source exposes them
- broader backtesting against simpler baselines

Supported collection path:

- API-first acquisition through `tconnectsync`
- browser or Playwright acquisition is not part of the active workflow
- manual CSV exports can still be dropped into `data/raw/` and ingested

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
bayesian-t1dm validate-raw
bayesian-t1dm ingest
```

4. Run the full pipeline:

```bash
bayesian-t1dm run
```

5. Run tests:

```bash
pytest
```

## Configuration

Credentials and acquisition settings belong in a root-level `.env` file that is never committed.

Required or commonly used variables:

- `TCONNECT_EMAIL`
- `TCONNECT_PASSWORD`
- `TCONNECT_REGION`
- `TIMEZONE_NAME`
- `PUMP_SERIAL_NUMBER` if required for your account

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

Build coverage and manifest outputs:

```bash
bayesian-t1dm ingest
```

Run the forecasting and recommendation pipeline:

```bash
bayesian-t1dm run
```

## Data Flow

The active pipeline is:

1. Tandem Mobi syncs to Tandem Source through the mobile app
2. `tconnectsync` logs in and requests Tandem Source API windows
3. raw API responses are archived under the cloud raw tree
4. normalized CGM, bolus, basal, and activity tables are written beside the raw responses
5. a per-window manifest records coverage, hashes, timestamps, and pump identity
6. the repo can also ingest manual CSV exports placed in `data/raw/`
7. a canonical 5-minute time grid is built
8. insulin exposure and IOB are derived from bolus events
9. lagged, rolling, and calendar features are created
10. the Bayesian forecasting model is fit
11. scenario comparison and recommendation ranking are generated

## Storage Layout

Local and cloud paths are organized as follows:

- code: `~/Projects/bayesian-t1dm`
- runtime scratch: `~/ProjectsRuntime/bayesian-t1dm`
- cloud raw data and manifests: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw`
- cloud published outputs: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/output`

Within the cloud raw tree, the acquisition flow writes:

- raw API payload archives under `data/raw/tconnectsync/<window_id>/raw/`
- normalized CSVs under `data/raw/tconnectsync/<window_id>/normalized/`
- per-window manifests under `data/raw/tconnectsync/<window_id>/window_manifest.csv`
- collection manifest at `data/raw/tandem_export_manifest.csv`

## Repository Layout

- `src/bayesian_t1dm/` - importable Python package for ingest, acquisition, features, modeling, recommendation, evaluation, and reporting
- `notebooks/` - exploratory notebooks and experiments
- `archive/r/` - archived R prototypes and scratch code
- `docs/` - technical documentation
- `tests/` - fixtures and regression tests

## Outputs

The pipeline produces:

- coverage summaries showing how much history is available
- processed tables with normalized timestamps and derived insulin exposure
- posterior forecasts with uncertainty intervals
- recommendation summaries for human review
- markdown reports suitable for inspection or downstream logging

## Limitations

- Tandem Source availability and `tconnectsync` behavior still affect acquisition.
- Tandem Source is undocumented, so API drift remains a risk.
- Recommendations are advisory and must be reviewed by you.
- The default insulin action curve is an approximation, not physiology ground truth.
- The Bayesian model is intentionally conservative because personal data can be sparse and irregular.

## Design Decisions

- Python is the active path because the current workflow is Python-heavy and easier to test and reuse as a package.
- Notebooks remain for exploratory analysis and visual debugging.
- R is archived because the active R scripts were broken and incomplete as standalone entry points.
- Recommendations are human-reviewable rather than automated because the goal is decision support with uncertainty, not automatic pump control.
- Credentials live in a local `.env` so API acquisition can run without exposing secrets in Git or cloud sync.

## Technical Notes

For model equations, variable definitions, and recommendation math, see [`docs/technical.md`](./docs/technical.md).
