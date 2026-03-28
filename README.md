# Bayesian T1DM

Personal Type 1 diabetes forecasting and recommendation pipeline.

This repository is a workbench for analyzing Tandem pump data, CGM data, activity data, and derived insulin exposure so you can:

- forecast post-meal glucose trajectories
- inspect uncertainty, calibration, and scenario comparisons
- generate human-reviewable recommendations for basal, bolus, and I/C ratio changes

The active implementation is Python. Legacy R work is archived in [`archive/r/`](./archive/r/), and exploratory notebooks live in [`notebooks/`](./notebooks/).

## Current Status

This repo has moved from scratchpad stage to a structured pipeline, but it is still a research project, not a finished medical product.

What is in place:

- normalized Tandem ingest
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

## Data Sources

The pipeline is designed to ingest Tandem exports in CSV or Excel form, then normalize them into a canonical 5-minute table.

Typical inputs:

- CGM readings
- bolus events
- basal schedule or timeline exports
- activity exports

The repo also supports the export style you were already exploring in the notebooks, where monthly Tandem Source/t:connect-derived workbooks are collected and then normalized in Python.

Raw files should live under `data/raw/`. Processed artifacts can be written to `data/processed/` or `output/` as needed.

## Repo Layout

- `src/bayesian_t1dm/` - importable Python package for ingest, features, modeling, recommendation, evaluation, and reporting
- `notebooks/` - exploratory notebooks and experiments
- `archive/r/` - archived R prototypes and scratch code
- `docs/` - technical documentation
- `tests/` - fixtures and regression tests

## Quickstart

Install in editable mode:

```bash
python -m pip install -e .[dev]
```

Inspect Tandem data coverage:

```bash
bayesian-t1dm ingest --raw data/raw --report output/coverage.md
```

Run the full pipeline:

```bash
bayesian-t1dm run --raw data/raw --report output/run_summary.md
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

- The repo is not connected to Tandem programmatically by default; it expects exports or locally saved workbooks/files.
- Recommendations are advisory and must be reviewed by you.
- The default insulin action curve is an approximation, not physiology ground truth.
- The Bayesian model is intentionally conservative because personal data can be sparse and irregular.

## Design Decisions

- Python was chosen as the active path because the current notebook workflow is already Python-heavy and the package structure makes testing and reuse much easier.
- Notebooks stay because they are still useful for exploratory analysis and visual debugging.
- R was archived because the active R scripts were broken and incomplete as standalone entrypoints.
- Recommendations are human-reviewable rather than automated because the goal is decision support with uncertainty, not automatic pump control.

## Technical Details

For model equations, variable definitions, and recommendation math, see [`docs/technical.md`](./docs/technical.md).
