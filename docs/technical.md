# Technical Notes

## Data Flow

The active pipeline is:

1. Tandem Mobi syncs to Tandem Source through the mobile app
2. Tandem Source browser automation exports CSVs for explicit 30-day windows
3. raw exports are stored in `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw/`
4. local browser session state, traces, and screenshots live in `~/ProjectsRuntime/bayesian-t1dm/`
5. a raw acquisition manifest records file coverage and window completeness
6. normalized CGM, bolus, basal, and activity tables are derived from the raw exports
7. 5-minute canonical time grid
8. derived insulin exposure and IOB
9. lagged, rolling, and calendar features
10. Bayesian forecast model
11. scenario comparison and pump-setting recommendation ranking

## Tandem Source Acquisition

The implementation assumes Tandem Source is the authoritative cloud system for the Mobi data. The pipeline does not require a public Tandem API.

Operational assumptions:

- the Mobi app is already linked to Tandem Source
- Tandem Source uploads are complete enough to support export windows
- browser automation runs against a local Playwright profile
- raw reports are saved to the cloud side-project folder before modeling
- credentials are loaded from a local `.env` or shell environment only

Storage convention:

- code: `~/Projects/bayesian-t1dm`
- runtime scratch: `~/ProjectsRuntime/bayesian-t1dm`
- cloud raw data and manifest: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/data/raw`
- cloud published outputs: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/bayesian-t1dm/output`

The acquisition manifest captures:

- source file name
- requested window start and end dates
- observed first and last timestamps in the downloaded file
- number of rows parsed from the file
- whether the window appears complete
- the browser trace and screenshot paths used for debugging
- a content hash and file size for the raw export

This is the gating artifact for downstream modeling. If the manifest shows incomplete coverage, the forecast and recommendation outputs should be treated as provisional.

The downstream ingest manifest, which is derived after normalization, still checks for gaps, overlaps, duplicates, and out-of-order windows across the parsed tables.

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

## Assumptions and Failure Modes

Assumptions:

- Tandem export timestamps are trustworthy after normalization
- a 5-hour insulin action approximation is acceptable as a first-order model
- current CGM, bolus, and activity history carries signal for short-horizon prediction

Known failure modes:

- missing export files or partial history
- sparse meal logging
- basal schedule changes not captured in the chosen export
- poor calibration if the insulin action kernel is badly misspecified
- overconfident recommendations if validation is not walk-forward based

## Acquisition and Manifest Design Decisions

- Tandem Source cloud sync is treated as the source of truth because it is the supported path for Mobi uploads.
- The pipeline is file-based because there is no public Tandem API in the implementation.
- A raw export manifest is required so export completeness is checked before modeling rather than after the fact.
- Coverage completeness is defined on the acquisition manifest, while cross-file gap checks live in the normalized ingest manifest.

## Design Decisions

- Python is the active pipeline language because it supports a cleaner package structure and better notebook-to-production migration.
- Notebooks remain for exploration, but they are not the source of truth.
- R is archived because the previous R scripts were not runnable as a standalone workflow.
- Recommendations are human-reviewable rather than automatic because the repo is meant to support personal diabetes decisions, not control the pump directly.
