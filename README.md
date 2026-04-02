# Bayesian T1DM

Bayesian forecasting and recommendation pipeline for personal Type 1 diabetes data.

## Mission

This repository exists to turn your personal Tandem and Apple Health data into traceable therapy-setting evidence.

The near-term mission is not just to forecast glucose. It is to answer:

- do the available data identify anything useful about your current settings?
- starting with overnight / fasting basal, what appears identifiable versus blocked?
- what evidence supports a likely direction of change?
- where do the blockers live when the answer is still unclear?

Forecasting, feature engineering, and synthetic validation support that mission, but they are not the headline goal.

This is a research project and decision-support workbench, not a medical device or automated pump controller.

## Current Status

The active implementation is Python. Legacy R work is archived in [`archive/r/`](./archive/r/), and exploratory notebooks live in [`notebooks/`](./notebooks/).

What is currently in place:

- Tandem Source acquisition through `tconnectsync`
- raw API payload archival as the source of truth
- normalization of CGM, bolus, basal, and activity tables
- multi-export Apple Health import with canonical overlap resolution
- Tandem-aligned 5-minute analysis-ready dataset materialization
- Apple Health feature screening against Tandem targets
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

1. Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate bt1dm
```

This environment file installs the package in editable dev mode. If you prefer to create it manually instead of using [`environment.yml`](./environment.yml):

```bash
conda create -n bt1dm python=3.11
conda activate bt1dm
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

4. Prepare model-ready data:

```bash
bayesian-t1dm prepare-model-data --apple-input \
  ~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/apple_health_data
```

`prepare-model-data` is the default data-prep workflow. If Apple Health bundles are available, it imports them, measures Apple/Tandem overlap, backfills Tandem data when credentials are available, and writes an aligned 5-minute model dataset. If Apple Health is absent, it falls back to a Tandem-only dataset and targets roughly one year of Tandem history by default.

5. Run the full pipeline:

```bash
bayesian-t1dm run
```

For faster real-data validation without the final recommendation fit:

```bash
bayesian-t1dm run --skip-recommendations
```

6. Run tests:

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

Import one or more Health Auto Export bundles from a parent directory:

```bash
bayesian-t1dm import-health-auto-export --input \
  ~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/apple_health_data
```

Prepare the default model-ready dataset with optional Apple Health enrichment:

```bash
bayesian-t1dm prepare-model-data --apple-input \
  ~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/apple_health_data
```

Run therapy-setting research with segmented decision-support artifacts:

```bash
bayesian-t1dm research-therapy-settings --apple-input \
  ~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/apple_health_data
```

Run the synthetic therapy infrastructure validator:

```bash
bayesian-t1dm validate-therapy-infra
```

Build the main therapy evidence review surface:

```bash
bayesian-t1dm review-therapy-evidence --apple-input \
  ~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/apple_health_data
```

Materialize the Tandem-aligned 5-minute analysis-ready table:

```bash
bayesian-t1dm build-health-analysis-ready
```

Screen Apple Health context features against Tandem glucose targets:

```bash
bayesian-t1dm screen-health-features
```

## Data Flow

The active pipeline is:

1. Tandem Mobi syncs to Tandem Source through the mobile app
2. `tconnectsync` logs in and requests Tandem Source API windows
3. raw API responses are archived under the cloud raw tree
4. normalized CGM, bolus, basal, and activity tables are written beside the raw responses
5. optional Apple Health year-chunk bundles are imported under `data/raw/health_auto_export/<export_id>/...`
6. imported Apple Health tables are unified with deterministic latest-export-wins dedupe
7. `prepare-model-data` inspects Apple and Tandem spans, requests missing Tandem history when needed, and writes an alignment/preparation report
8. a per-window Tandem manifest records coverage, hashes, timestamps, and pump identity
9. a Tandem 5-minute feature grid is built
10. Apple Health context is merged onto those Tandem timestamps when available to create the final prepared dataset
11. insulin exposure and IOB are derived from bolus events
12. lagged, rolling, and calendar features are created
13. optional therapy research builds segment-level contexts, audits feature engineering, and compares candidate models
14. therapy research now writes a methodological gate, meal-proxy audit, and standardized Tandem/Apple source report cards before model comparison
15. the Bayesian forecasting model is fit
16. scenario comparison and recommendation ranking are generated

## Review Flow

Canonical repair/review workflow:

1. `bayesian-t1dm normalize-raw`
2. `bayesian-t1dm ingest`
3. `bayesian-t1dm run --skip-recommendations`

This keeps raw-window repair, coverage review, and predictive validation separate from recommendation generation.

Canonical Tandem + Apple Health workflow:

1. `bayesian-t1dm normalize-raw`
2. `bayesian-t1dm prepare-model-data --apple-input ~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/apple_health_data`
3. `bayesian-t1dm screen-health-features`
4. `bayesian-t1dm run --skip-recommendations`
5. `bayesian-t1dm run`

Canonical Tandem-only fallback workflow:

1. `bayesian-t1dm normalize-raw`
2. `bayesian-t1dm prepare-model-data`
3. `bayesian-t1dm run --skip-recommendations`
4. `bayesian-t1dm run`

Canonical therapy evidence workflow:

1. `bayesian-t1dm normalize-raw`
2. `bayesian-t1dm prepare-model-data --apple-input ~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/apple_health_data`
3. `bayesian-t1dm review-therapy-evidence`
4. review `therapy_evidence_review.html` first, then inspect the linked supporting artifacts if you need to go deeper

What each step does:

1. `normalize-raw` rebuilds normalized Tandem tables from archived raw API payloads.
2. `prepare-model-data` optionally imports Apple Health bundles, computes Apple/Tandem overlap, backfills Tandem data when needed, and writes the prepared 5-minute model dataset plus a preparation report.
3. `screen-health-features` evaluates Apple context features against Tandem glucose targets using the unified prepared dataset and skips cleanly when Apple Health is absent.
4. `run --skip-recommendations` is the preferred fast validation path because it checks walk-forward forecasting without paying for the final recommendation fit.
5. `run` performs the full modeling and recommendation pipeline. It now uses the same prepared dataset contract: Apple-enriched when Apple data exists, Tandem-only otherwise.
6. `research-therapy-settings` runs a separate research-grade workflow for basal schedule, I/C ratio, and later sensitivity factor analysis. It starts with a methodological gate, builds strict meal-bolus proxy features when explicit carbs are absent, writes Tandem and Apple source report cards, compares candidate models, and writes human-reviewable segment-level evidence rather than auto-changing settings.
7. `validate-therapy-infra` runs synthetic truth-recovery scenarios against the same therapy research stack so the infrastructure has to recover known therapy directions and suppress itself in corrupted or weak-identifiability cases.
8. `review-therapy-evidence` orchestrates the current therapy-facing workflow and writes an interactive HTML review page focused on overnight basal identifiability, evidence quality, exclusion reasons, supporting artifacts, and code-path traceability.

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
- imported Apple Health bundles under `data/raw/health_auto_export/<export_id>/raw/` and `.../normalized/`

Discovery rules:

- active Tandem ingest walks the cloud raw tree but excludes deprecated `archive data/` content
- canonical Apple Health loading reads only `data/raw/health_auto_export/...`
- repo-local `data/raw/` remains a manual staging area, not the canonical long-term source tree

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
- `model_data_preparation.md`
- `prepared_model_data_5min.csv`
- `analysis_ready_health_5min.csv`
- `health_feature_screening.md`
- `health_feature_scores.csv`
- `therapy_feature_audit.md`
- `therapy_feature_registry.csv`
- `therapy_research_gate.md`
- `meal_proxy_audit.md`
- `therapy_model_comparison.md`
- `therapy_segment_evidence.csv`
- `therapy_recommendation_research.md`
- `tandem_source_report_card.md`
- `apple_source_report_card.md`
- `source_numeric_summary.csv`
- `source_missingness_summary.csv`
- `therapy_infra_validation.md`
- `therapy_synthetic_results.csv`
- `therapy_synthetic_recommendation_audit.md`
- `therapy_evidence_review.html`
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
- `therapy_evidence_review.html` is the primary therapy-facing visual review artifact.
- `coverage_review.html` and `run_review.html` remain the primary operational review artifacts for raw coverage and forecasting behavior.
- recommendation records include `confidence` and `flags` so low-signal cases are explicit in machine-readable output.
- An empty recommendation list can mean no scenario cleared the gain threshold, or that recommendations were intentionally suppressed or skipped.

## Technical Notes

For model equations, variable definitions, and recommendation math, see [`docs/technical.md`](./docs/technical.md).
