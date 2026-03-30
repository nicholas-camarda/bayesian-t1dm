# Bayesian T1DM

Personal Type 1 diabetes forecasting and recommendation pipeline.

This repository is a workbench for analyzing Tandem pump data, CGM data, activity data, and derived insulin exposure so you can:

- forecast post-meal glucose trajectories
- inspect uncertainty, calibration, and scenario comparisons
- generate human-reviewable recommendations for basal, bolus, and I/C ratio changes

The active implementation is Python. Legacy R work is archived in [`archive/r/`](./archive/r/), and exploratory notebooks live in [`notebooks/`](./notebooks/).

Local runtime artifacts live under `~/ProjectsRuntime/bayesian-t1dm`, while raw Tandem exports and final published outputs live under `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm`.

## Current Status

This repo has moved from scratchpad stage to a structured pipeline, but it is still a research project, not a finished medical product.

What is in place:

- Tandem Source acquisition backed by `tconnectsync` API payload archives
- raw API payloads are preserved as the source of truth
- normalized CGM, bolus, basal, and activity tables are derived from those payloads
- export manifest generation and coverage checks
- 5-minute time-grid construction
- insulin activity and IOB expansion from bolus events
- Bayesian forecasting model scaffolding in PyMC
- recommendation scoring for candidate pump-setting changes
- test fixtures and basic validation checks
- technical documentation with explicit equations and definitions
- live API smoke-tested here against Tandem Source through `tconnectsync`

What remains to be refined:

- stronger validation on real multi-month or multi-year exports
- calibration of priors and scenario thresholds against your own data
- richer handling of basal schedules
- explicit CGM gap and missingness indicators
- meal and carb features if Tandem Source exposes them through `tconnectsync`
- broader backtesting and comparison against simpler baselines
- if you add manual CSV exports yourself, they will be treated as normal raw inputs alongside API payloads

Known statuses:

- fully working: ingest, reporting, and feature/model pipeline on already-existing raw files
- fully working in tests: `tconnectsync`-backed acquisition, raw archival, and normalization plumbing
- live smoke-test confirmed here: Tandem Source login, pump metadata retrieval, and event extraction through `tconnectsync`

## Data Sources

The pipeline treats Tandem Source as the authoritative cloud source for your Mobi data. The active workflow is API-first through `tconnectsync`: request a window, archive the raw API payloads, normalize them into the repo's ingest contract, and then model the normalized tables.

There is no supported Playwright/browser acquisition path in the active workflow. If you need to import data you exported yourself from Tandem Source, drop the CSVs into the raw tree and ingest them as normal inputs.

Typical inputs:

- Tandem Mobi pump data uploaded to Tandem Source through the mobile app sync path
- raw `tconnectsync` payload archives
- manual CSV exports that you drop into `data/raw/` or the cloud raw tree yourself
- CGM readings, bolus events, basal schedule data, and activity exports when available

The repo expects raw Tandem Source artifacts to live under `data/raw/`, then records a manifest of what was seen, what window it covered, and whether the window appears complete. For API-acquired windows, the raw payload archive is the source of truth and the normalized CSVs are derived artifacts.

Raw files should live under `data/raw/`. Processed artifacts can be written to `data/processed/` or `output/` as needed.

Canonical storage locations are:

- local runtime scratch: `~/ProjectsRuntime/bayesian-t1dm`
- cloud raw payloads and normalized outputs: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw`
- cloud published outputs: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/output`

Credentials should live in a root-level `.env` file that is never committed:

- `TCONNECT_EMAIL`
- `TCONNECT_PASSWORD`
- `TCONNECT_REGION`
- `TIMEZONE_NAME`
- optional `PUMP_SERIAL_NUMBER`

## Harvest Workflow

Use this order when you want to recover Tandem data:

1. Put your Tandem credentials in `.env`.
2. Collect a window with the API client:

```bash
bayesian-t1dm collect --start-date 2025-10-01 --end-date 2025-10-30
```

3. If you are backfilling more than 30 days, use:

```bash
bayesian-t1dm backfill --start-date 2025-10-01 --end-date 2026-03-28
```

4. If you have manually exported CSVs, drop them into `data/raw/` and ingest them the same way.

5. Validate the raw files before trusting them:

```bash
bayesian-t1dm validate-raw
```

6. Ingest the raw exports:

```bash
bayesian-t1dm ingest
```

The `validate-raw` step is the quickest way to tell whether a Tandem file is a usable granular CGM export or just a summary artifact:

- granular CGM export means `eventDateTime` plus `egv_estimatedGlucoseValue`
- `bg` may still appear, but it should not be the primary stream when EGV is present

## API Acquisition

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,tconnectsync]'
```

Verify that the cli exists:

```bash
bayesian-t1dm --help
```

Run collection or backfill. The command archives the raw API payloads and writes normalized tables under the cloud raw tree:

```bash
bayesian-t1dm collect --start-date 2024-01-01 --end-date 2024-01-30
bayesian-t1dm backfill --start-date 2023-01-01 --end-date 2024-01-30
```

The acquisition flow is:

1. Install with `.[dev,tconnectsync]`
2. Create `.env` with `TCONNECT_EMAIL`, `TCONNECT_PASSWORD`, `TIMEZONE_NAME`, and optionally `PUMP_SERIAL_NUMBER`
   - `TCONNECT_REGION` defaults to `US` unless your account needs `EU`
3. Run `bayesian-t1dm collect` or `bayesian-t1dm backfill`
4. Inspect the raw payload archive and normalized CSVs under `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/tconnectsync/`
5. Run `bayesian-t1dm ingest` to build the coverage report and model inputs

What gets written where:

- runtime logs and scratch artifacts under `~/ProjectsRuntime/bayesian-t1dm`
- raw API payload archives under `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/tconnectsync/<window_id>/raw/`
- normalized CSVs under `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/tconnectsync/<window_id>/normalized/`
- per-window manifest under `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/tconnectsync/<window_id>/window_manifest.csv`
- collection manifest in `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/tandem_export_manifest.csv`
- acquisition summary in `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/output/tandem_acquisition_summary.md`
- coverage report in `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/output/coverage.md`
- run summary in `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/output/run_summary.md`

Primary collection contract:

- the repo acquires Tandem Source data through `tconnectsync`
- raw API payload archives are the source of truth
- normalized CGM, bolus, basal, and activity CSVs are derived artifacts
- manual CSV exports you add yourself are also valid raw inputs
- `collect` success means raw payloads were archived and normalized outputs were written
- `ingest` success means the raw tree could be parsed into normalized tables and a coverage report

## Repo Layout

- `src/bayesian_t1dm/` - importable Python package for ingest, acquisition, features, modeling, recommendation, evaluation, and reporting
- `notebooks/` - exploratory notebooks and experiments
- `archive/r/` - archived R prototypes and scratch code
- `docs/` - technical documentation
- `tests/` - fixtures and regression tests

## Quickstart

Install in editable mode:

```bash
python -m pip install -e '.[dev,tconnectsync]'
```

Inspect Tandem data coverage:

```bash
bayesian-t1dm ingest
```

Run the full pipeline:

```bash
bayesian-t1dm run
```

Run tests:

```bash
pytest
```

## Pipeline Stages

1. Ingest raw Tandem exports from `data/raw/`
2. Normalize CGM, bolus, basal, and activity tables
3. Expand bolus events into a time-distributed insulin action curve and estimate IOB
4. Build a canonical 5-minute feature table
5. Fit a Bayesian forecasting model
6. Compare current settings against candidate basal, bolus, and I/C scenarios
7. Emit a report with forecast calibration and recommendation summaries

## Outputs

The pipeline produces:

- coverage summaries showing how much history is available
- processed tables with normalized timestamps and derived insulin exposure
- posterior forecasts with uncertainty intervals
- setting-change recommendations for human review
- markdown reports suitable for inspection or downstream logging

## Limitations

- Live Tandem API behavior still depends on your account, Tandem Source availability, and the installed `tconnectsync` version.
- Tandem Source is undocumented, so API drift is still a risk.
- Recommendations are advisory and must be reviewed by you.
- The default insulin action curve is an approximation, not physiology ground truth.
- The Bayesian model is intentionally conservative because personal data can be sparse and irregular.

## Design Decisions

- Python was chosen as the active path because the current notebook workflow is already Python-heavy and the package structure makes testing and reuse much easier.
- Notebooks stay because they are still useful for exploratory analysis and visual debugging.
- R was archived because the active R scripts were broken and incomplete as standalone entrypoints.
- Recommendations are human-reviewable rather than automated because the goal is decision support with uncertainty, not automatic pump control.
- Credentials belong in a local `.env` so API acquisition can run without exposing secrets in Git or cloud sync.

## Technical Details

For model equations, variable definitions, and recommendation math, see [`docs/technical.md`](./docs/technical.md).
