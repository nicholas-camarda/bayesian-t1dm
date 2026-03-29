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

- Tandem Source cloud-sync ingest backed by raw file exports
- selector discovery for Tandem Source browser automation
- export manifest generation and coverage checks
- browser automation for Tandem Source export collection
- login and discovery are substantially more robust than the earlier scratchpad version
- 5-minute time-grid construction
- insulin activity and IOB expansion from bolus events
- Bayesian forecasting model scaffolding in PyMC
- recommendation scoring for candidate pump-setting changes
- test fixtures and basic validation checks
- technical documentation with explicit equations and definitions

What remains to be refined:

- stronger validation on real multi-month or multi-year exports
- calibration of priors and scenario thresholds against your own data
- richer handling of basal schedules and meal tagging if available in exports
- broader backtesting and comparison against simpler baselines
- the Tandem export layer is still under active reverse engineering, so browser success does not imply that a usable tabular artifact was produced

Known statuses:

- fully working: ingest and reporting on already-existing raw files
- partially working: browser login, discovery, and navigation
- still unstable: converting Tandem exports into usable raw CSV or table-like artifacts

## Data Sources

The pipeline treats Tandem Source as the authoritative cloud source for your Mobi data. There is no public Tandem API in the main workflow by default, so the primary contract is browser-driven: discover the page, collect or backfill the export, validate that the artifact is tabular, then ingest it in Python.

Typical inputs:

- Tandem Mobi pump data uploaded to Tandem Source through the mobile app sync path
- Daily Timeline CSV exports or equivalent Tandem Source report exports
- CGM readings, bolus events, basal schedule data, and activity exports when available

The repo expects raw Tandem Source exports to live under `data/raw/`, then records a manifest of what was seen, what window it covered, and whether the window appears complete. If Tandem returns a non-tabular response payload, that is a collection failure for the main pipeline even if the browser action itself succeeded.

Raw files should live under `data/raw/`. Processed artifacts can be written to `data/processed/` or `output/` as needed.

For browser collection, the canonical storage locations are:

- local runtime scratch: `~/ProjectsRuntime/bayesian-t1dm`
- cloud raw exports: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw`
- cloud published outputs: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/output`

Credentials should live in a root-level `.env` file that is never committed:

- `TANDEM_SOURCE_EMAIL`
- `TANDEM_SOURCE_PASSWORD`

## Browser Collection

Install the project with the browser extras if you want the primary browser-driven collection path:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,automation]'
```

Install Playwright browsers:

```bash
playwright install
```

Verify that the cli exists:

```bash
bayesian-t1dm --help
```

Create a root-level `.env` with `TANDEM_SOURCE_EMAIL` and `TANDEM_SOURCE_PASSWORD`. Do not commit it.

Run discovery first so the page map and diagnostics are written:

```bash
bayesian-t1dm discover
```

Inspect the discovery summary and saved page map before collecting:

- `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/output/tandem_discovery_summary.md`
- `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/archive/tandem_page_map.json`

Collect one 30-day window:

```bash
bayesian-t1dm collect --start-date 2024-01-01 --end-date 2024-01-30
```

Backfill a range:

```bash
bayesian-t1dm backfill --start-date 2023-01-01 --end-date 2024-01-30
```

The acquisition flow is:

1. Install with `.[dev,automation]`
2. Run `playwright install`
3. Create `.env`
4. Run `bayesian-t1dm discover`
5. Inspect diagnostics and the saved page map
6. Run `bayesian-t1dm collect` or `bayesian-t1dm backfill`
7. Check whether the collected artifact is actually tabular before trusting `ingest`
8. Run `bayesian-t1dm ingest` only after confirmed raw export artifacts exist

What gets written where:

- runtime logs, traces, screenshots, and browser scratch artifacts under `~/ProjectsRuntime/bayesian-t1dm` and its `logs/`, `traces/`, `downloads/`, `browser-profile/`, and `browser-home/` subdirectories
- selector map in `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/archive/tandem_page_map.json`
- raw collection artifacts in `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/`
- collection manifest in `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/tandem_export_manifest.csv`
- acquisition summary in `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/output/tandem_acquisition_summary.md`
- discovery summary in `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/output/tandem_discovery_summary.md`
- coverage report in `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/output/coverage.md`
- run summary in `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/output/run_summary.md`

Page map:

- partial bootstrap maps are allowed during discovery
- discovery may save an incomplete map if only login or report navigation was discovered
- collection requires a complete map or a path that can finish discovery
- the current code distinguishes `export_csv_launcher` and `export_csv_confirm`

Primary collection contract:

- the repo tries to acquire Tandem report data through browser automation
- the preferred artifact is a tabular export file
- if Tandem returns a non-tabular payload, the main pipeline treats that as a collection failure
- `discover` success means selectors and navigation were learned
- `collect` success means a tabular artifact was written and validated enough to record in the manifest
- export-response capture without tabular output is diagnostic progress, not finished data collection

## Experimental Fallback

The repository also has a Python-only DOM fallback in [`src/bayesian_t1dm/timeline_pull.py`](./src/bayesian_t1dm/timeline_pull.py).

- It is not exposed through the CLI.
- It is a secondary, experimental path for recovering rendered Daily Timeline summary rows from the page DOM when export remains blocked.
- It is useful for analysis rescue work, but it is not equivalent to a raw Tandem CSV export.

## Repo Layout

- `src/bayesian_t1dm/` - importable Python package for ingest, browser acquisition, features, modeling, recommendation, evaluation, and reporting
- `notebooks/` - exploratory notebooks and experiments
- `archive/r/` - archived R prototypes and scratch code
- `docs/` - technical documentation
- `tests/` - fixtures and regression tests

## Quickstart

Install in editable mode:

```bash
python -m pip install -e '.[dev,automation]'
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

- The repo uses browser automation for Tandem Source collection, but the export layer is still under active reverse engineering.
- Browser success does not imply a usable tabular export.
- Recommendations are advisory and must be reviewed by you.
- The default insulin action curve is an approximation, not physiology ground truth.
- The Bayesian model is intentionally conservative because personal data can be sparse and irregular.

## Design Decisions

- Python was chosen as the active path because the current notebook workflow is already Python-heavy and the package structure makes testing and reuse much easier.
- Notebooks stay because they are still useful for exploratory analysis and visual debugging.
- R was archived because the active R scripts were broken and incomplete as standalone entrypoints.
- Recommendations are human-reviewable rather than automated because the goal is decision support with uncertainty, not automatic pump control.
- Credentials belong in a local `.env` so browser automation can run without exposing secrets in Git or cloud sync.

## Technical Details

For model equations, variable definitions, and recommendation math, see [`docs/technical.md`](./docs/technical.md).
