# Technical Notes

## Data Flow

The active pipeline is:

1. Tandem Mobi syncs to Tandem Source through the mobile app
2. `tconnectsync` logs in to Tandem Source and queries API windows directly
3. raw API responses are archived under `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/tconnectsync/<window_id>/raw/`
4. normalized CGM, bolus, basal, and activity tables are written beside the raw responses
5. optional Health Auto Export bundles are archived under `data/raw/health_auto_export/<export_id>/...`
6. imported Apple Health tables are unified with deterministic latest-export-wins overlap resolution
7. `prepare-model-data` inspects Apple and Tandem coverage, requests missing Tandem history when needed, and writes a preparation/alignment report
8. a per-window Tandem manifest records coverage, hashes, timestamps, and pump identity
9. a Tandem 5-minute feature frame is built
10. canonical Apple Health context is merged onto Tandem feature-frame timestamps when available to create the final prepared dataset
11. derived insulin exposure and IOB
12. lagged, rolling, and calendar features
13. optional therapy research builds day-segmented contexts, audits feature engineering, and compares candidate models for basal schedule and I/C ratio inference
14. Bayesian forecast model
15. scenario comparison and pump-setting recommendation ranking

## Tandem Source Acquisition

The implementation assumes Tandem Source is the authoritative cloud system for the Mobi data. The active pipeline now uses `tconnectsync` as an API adapter.

Operational assumptions:

- the Mobi app is already linked to Tandem Source
- Tandem Source uploads are complete enough to support API windows
- raw API payload archives are saved to the cloud side-project folder before modeling
- credentials are loaded from a local `.env` or shell environment only
- `TCONNECT_REGION` or `TANDEM_SOURCE_REGION` should be set for non-US accounts
- manual CSV exports can be added as supplemental raw inputs

Recommended environment variables:

- `TANDEM_SOURCE_EMAIL` or `TCONNECT_EMAIL`
- `TANDEM_SOURCE_PASSWORD` or `TCONNECT_PASSWORD`
- `TANDEM_SOURCE_REGION` or `TCONNECT_REGION`
- `TIMEZONE_NAME`
- `TANDEM_SOURCE_PUMP_SERIAL`, `TANDEM_PUMP_SERIAL`, or `PUMP_SERIAL_NUMBER` when required

Storage convention:

- code: `~/Projects/bayesian-t1dm`
- runtime outputs, scratch space, and intermediate artifacts: `~/ProjectsRuntime/bayesian-t1dm`
- cloud project home: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm`
- scraped/raw acquisition data: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw`
- final published outputs: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/output`
- default working summaries: `~/ProjectsRuntime/bayesian-t1dm/output`

Raw discovery rules:

- active Tandem ingest excludes deprecated `archive data/` content from generic raw discovery
- canonical Apple Health loading reads only `data/raw/health_auto_export/...`
- `data/raw/apple_health_data/` is a source pool for `import-health-auto-export`, not a direct modeling input

The acquisition manifest captures:

- source file name or raw artifact path
- endpoint family
- requested window start and end dates
- observed first and last timestamps in the downloaded file
- number of rows parsed from the file
- whether the window appears complete
- raw payload hashes and file sizes
- timezone and pump identity used for the request

This is the gating artifact for downstream modeling. If the manifest shows incomplete coverage, the forecast and recommendation outputs should be treated as provisional.

The downstream ingest manifest, which is derived after normalization, still checks for gaps, overlaps, duplicates, and out-of-order windows across the parsed tables.

### Raw Repair and Canonical Timestamp Contract

`bayesian-t1dm normalize-raw` rebuilds `normalized/*.csv` and `window_manifest.csv` from archived `raw/*.json` payloads without refetching.

Normalized CSV contract:

- timestamps are stored as timezone-naive local timestamps in `TIMEZONE_NAME`
- source/raw metadata may still retain offsets
- mixed tz-aware and tz-naive raw payload timestamps are normalized elementwise before manifest building and CSV write-out

Per-window manifest fields now include:

- `requested_start`, `requested_end`
- `observed_first_timestamp`, `observed_last_timestamp`
- `observed_duration_days`
- `coverage_fraction`
- `completeness_reasons`

Standard completeness reason codes:

- `starts_late`
- `ends_early`
- `internal_gap`
- `duplicates`
- `overlap`
- `missing_kind`

Incomplete windows remain usable for review, but they are not recommendation-eligible by default.

## Insulin Action Kernel

We approximate insulin exposure using a fixed-duration action curve inspired by LoopKit-style insulin dynamics.

Let:

- $D$ = bolus dose in units
- $t$ = minutes since bolus completion
- $t_d$ = duration of action in minutes
- $t_p$ = peak time in minutes
- $\tau$ = decay constant
- $a$ = rise-time factor
- $S$ = scale factor

The implementation uses:

$$
\tau = \frac{t_p (1 - t_p / t_d)}{1 - 2 t_p / t_d}
$$

$$
a = \frac{2 \tau}{t_d}
$$

$$
S = \frac{1}{1 - a + (1 + a)e^{-t_d / \tau}}
$$

Insulin activity:

$$
IA(t) = D \cdot \frac{S}{\tau^2} \cdot t \cdot \left(1 - \frac{t}{t_d}\right) e^{-t/\tau}
$$

Insulin on board:

$$
IOB(t) = D \cdot \left[1 - S(1-a)\left(\left(\frac{t^2}{\tau t_d (1-a)} - \frac{t}{\tau} - 1\right)e^{-t/\tau} + 1\right)\right]
$$

The default configuration uses $t_d = 300$ minutes and $t_p = 75$ minutes.

## Feature Definitions

The canonical analysis grid is 5 minutes.

Key features:

- current CGM glucose at time $t$
- lagged CGM values at 5, 10, 15, 30, and 60 minutes
- rolling CGM means
- bolus units in the current 5-minute bin
- insulin activity and IOB derived from bolus expansion
- basal rate carried forward across the current bin
- activity summary variables
- CGM missingness and gap indicators
- carb and meal-derived features when carbohydrate records are available
- calendar features: sine/cosine hour-of-day, sine/cosine day-of-week, weekend indicator

Target:

- $y_{t+h}$, the glucose value $h$ minutes into the future

Default horizon:

- $h = 30$ minutes

## Bayesian Model

The active model is a Bayesian dynamic regression with a latent autoregressive residual process.

Let:

- $z_t$ = feature vector at time $t$
- $y_t$ = observed CGM glucose
- $\alpha$ = intercept
- $\beta$ = regression coefficients
- $x_t$ = latent residual state
- $\rho$ = autoregressive persistence
- $\sigma_{\text{state}}$ = latent state noise scale
- $\sigma_{\text{obs}}$ = observation noise scale

State equation:

$$
x_t = \rho x_{t-1} + \epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0, \sigma_{\text{state}}^2)
$$

Observation equation:

$$
y_t \sim \text{StudentT}(\nu, \mu_t, \sigma_{\text{obs}})
$$

with:

$$
\mu_t = \alpha + \beta^\top z_t + x_t
$$

The Student-t likelihood is used because CGM and pump-derived features can have outliers, short gaps, and occasional sensor noise.

### Prior Choices

The implementation uses shrinkage priors:

- $\alpha \sim \mathcal{N}(0, 1)$ on the standardized scale
- $\beta_j \sim \mathcal{N}(0, 0.5)$
- $\rho \sim \text{Beta}(2, 2)$
- $\sigma_{\text{state}} \sim \text{Exponential}(1)$
- $\sigma_{\text{obs}} \sim \text{Exponential}(1)$

These are intentionally conservative because the personal dataset can be sparse and unevenly distributed across meals, days, and activity states.

## Recommendation Scoring

For each candidate scenario $s$, the pipeline re-scales the exposure features for basal, bolus, and I/C ratio changes, then computes the posterior predictive loss.

One simple loss form is:

$$
L(s) = w_1 \, \mathbb{E}\left[| \hat{y}_t(s) - y^\star |\right]
      + w_2 \, \Pr(\hat{y}_t(s) < 70)
      + w_3 \, \Pr(\hat{y}_t(s) > 180)
$$

where $y^\star$ is the target glucose, typically 110 mg/dL.

The recommendation score is:

$$
\Delta(s) = L(\text{current}) - L(s)
$$

A candidate is surfaced only if:

- it improves expected loss by at least the configured threshold
- the improvement is directionally consistent across posterior draws
- the result is presented as human-reviewable decision support

## Validation

Validation is intentionally time-aware.

Recommended checks:

- walk-forward splits instead of random train/test splits
- calibration of posterior predictive intervals
- posterior predictive checks
- scenario comparison against a naive baseline

Operational notes:

- `bayesian-t1dm run` may perform multiple fits because walk-forward validation refits over chronological folds before any final recommendation fit.
- `bayesian-t1dm run --skip-recommendations` is the preferred fast path when validating real data because it avoids the final full-data recommendation fit.
- Modest MAE improvement over persistence is not enough for recommendation use if interval coverage is poor.

Canonical real-data review flow:

1. `bayesian-t1dm normalize-raw`
2. `bayesian-t1dm ingest`
3. `bayesian-t1dm run --skip-recommendations`

Canonical default model-data flow:

1. `bayesian-t1dm normalize-raw`
2. `bayesian-t1dm prepare-model-data --apple-input ~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/apple_health_data`
3. `bayesian-t1dm screen-health-features`
4. `bayesian-t1dm run --skip-recommendations`
5. `bayesian-t1dm run`

Canonical Tandem-only fallback flow:

1. `bayesian-t1dm normalize-raw`
2. `bayesian-t1dm prepare-model-data`
3. `bayesian-t1dm run --skip-recommendations`
4. `bayesian-t1dm run`

Canonical therapy research flow:

1. `bayesian-t1dm normalize-raw`
2. `bayesian-t1dm prepare-model-data --apple-input ~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/apple_health_data`
3. `bayesian-t1dm research-therapy-settings`
4. `bayesian-t1dm validate-therapy-infra`
5. review the therapy research and validation artifacts before considering any schedule changes

Step-by-step behavior:

1. `normalize-raw` rebuilds normalized Tandem tables from archived `tconnectsync` raw payloads.
2. `prepare-model-data` imports Apple Health bundles when provided, computes Apple/Tandem overlap, backfills Tandem history when needed, and writes a prepared 5-minute model dataset. If Apple Health is absent, it falls back cleanly to Tandem-only preparation and targets roughly one year of Tandem history by default.
3. `screen-health-features` consumes that prepared dataset and skips cleanly when Apple Health is absent.
4. `run --skip-recommendations` is the preferred fast validation path because it exercises walk-forward prediction without the final recommendation fit.
5. `run` executes the full modeling and recommendation stack using the same prepared dataset contract: Apple-enriched when Apple data exists, Tandem-only otherwise.
6. `research-therapy-settings` runs a distinct research-grade therapy workflow: it starts with a methodological gate, builds fasting, meal, and correction contexts, creates strict meal-bolus proxy features when explicit carbs are absent, audits Apple/Tandem feature engineering, compares Bayesian, linear, segmented, tree-boosted, and ensemble-style candidates, and writes segment-level recommendation evidence for basal and I/C ratio. Sensitivity factor remains staged and suppressed by default in the current implementation.
7. `validate-therapy-infra` runs a synthetic truth-recovery suite against the same therapy research contract. It checks basal direction recovery, proxy-only conservatism, Apple-helpful vs Apple-null behavior, and suppression under corrupted or low-identifiability scenarios.

## Output Contract

`run_summary.json` includes:

- `data_quality` with `good`, `degraded`, or `broken` source status
- `walk_forward` aggregate and per-fold metrics
- per-fold `fit_diagnostics`
- top-level `fit_diagnostics` for the final fit, or `null` if recommendations were skipped
- `recommendation_policy` showing whether recommendations were `generated`, `suppressed`, or `skipped`
- `review_artifacts` pointing at the runtime HTML review pages
- recommendation records with per-item `confidence` and `flags`

Sampler diagnostics currently include:

- `draws`, `tune`, `chains`, `target_accept`, `max_treedepth`
- `wall_time_seconds`
- `divergences`
- `max_tree_depth_observed`
- `max_tree_depth_hits`
- `rhat_max` when multiple chains are available
- `ess_bulk_min` and `ess_tail_min` when available

Recommendation output is policy-gated. The pipeline suppresses recommendations when predictive validation, sampler health, or available signal are not strong enough to support them.

In addition to markdown and JSON summaries, the active pipeline writes self-contained interactive HTML review artifacts:

- `coverage_review.html` from `ingest`
- `run_review.html` from `run`

These are the primary visual inspection surfaces for active-pipeline review. They combine source completeness, walk-forward behavior, baseline comparison, and sampler diagnostics in one place.

Apple Health-specific working artifacts:

- `import-health-auto-export` writes per-bundle manifests under `data/raw/health_auto_export/<export_id>/health_auto_export_manifest.csv`
- `prepare-model-data` writes `model_data_preparation.md` and `prepared_model_data_5min.csv` under the runtime output directory by default
- `build-health-analysis-ready` writes a wide Tandem-aligned 5-minute table, by default under `~/ProjectsRuntime/bayesian-t1dm/output/analysis_ready_health_5min.csv`
- `screen-health-features` writes `health_feature_screening.md` and `health_feature_scores.csv` under the runtime output directory
- `research-therapy-settings` writes `therapy_research_gate.md`, `meal_proxy_audit.md`, `therapy_feature_audit.md`, `therapy_feature_registry.csv`, `therapy_model_comparison.md`, `therapy_segment_evidence.csv`, `therapy_recommendation_research.md`, `tandem_source_report_card.md`, `apple_source_report_card.md`, `source_numeric_summary.csv`, and `source_missingness_summary.csv`
- `validate-therapy-infra` writes `therapy_infra_validation.md`, `therapy_synthetic_results.csv`, and `therapy_synthetic_recommendation_audit.md`

## Assumptions and Failure Modes

Assumptions:

- Tandem export timestamps are trustworthy after normalization
- a 5-hour insulin action approximation is acceptable as a first-order model
- current CGM, bolus, and activity history carries signal for short-horizon prediction
- API payloads are stable enough to support windowed backfill

Known failure modes:

- missing export files or partial history
- sparse meal logging
- basal schedule changes not captured in the chosen export
- poor calibration if the insulin action kernel is badly misspecified
- overconfident recommendations if validation is not walk-forward based
- silent API drift if Tandem changes undocumented endpoints

## Acquisition and Manifest Design Decisions

- Tandem Source cloud sync is treated as the source of truth because it is the supported path for Mobi uploads.
- The pipeline is API-first because `tconnectsync` can query Tandem Source directly.
- Manual CSV exports are still accepted as supplemental raw inputs.
- There is no supported Playwright/browser fallback in the active acquisition path; the browser code has been retired.
- A raw acquisition manifest is required so export completeness is checked before modeling rather than after the fact.
- Coverage completeness is defined on the acquisition manifest, while cross-file gap checks live in the normalized ingest manifest.
- Deprecated `archive data/` content is intentionally excluded from active Tandem ingest because it may contain legacy Apple Health exports and non-tabular ECG CSVs that are not part of the modeling contract.
- Canonical Apple Health loading is scoped to `data/raw/health_auto_export/...`; the `apple_health_data/` directory is a source pool for import, not a direct modeling input.
