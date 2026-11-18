# Technical Notes

## Goals

| Goal | Question | Subgoals | Status |
| --- | --- | --- | --- |
| A | Where do glucose outcomes stand, and in which hours? | A1 consensus metrics overall; A2 by pump segment; A3 baseline 14 and 90 days | `report-outcomes` |
| B | Which breakfast I:C change does the evidence support? | B1 pooled estimate per pump setting; B2 agreement checks; B3 proposal and safety rule | breakfast evidence page; proposals next |
| C | Did an entered change help? | C1 one-command refresh; C2 proposal record; C3 before/after evaluation | planned |
| D | Can logged food sharpen the carb estimate? | D1 unlogged-meal flags; D2 fat and protein; D3 Bevel as a second measurement | planned |
| E | Lunch and dinner I:C | E1 per segment evidence; E2 stricter eligibility; E3 proposals | planned |
| F | Basal from the controller pattern | F1 commanded vs profile rate; F2 lows as a hard stop; F3 proposals | planned |
| G | Correction factor | G1 correction-only boluses | planned |

The goal throughout is more time in 70-180 mg/dL with time below 70 under 4% and below 54 under 1%. A proposal changes one setting by at most 10% and is held 14 days before the next. The system proposes and measures; the person enters the change on the pump.

## Breakfast I:C proposals (goal B)

`bayesian-t1dm propose-breakfast-icr` writes `runtime/output/proposals/breakfast_proposal.md` and `.csv`.

**What the page leads with.** A table with one row per pump segment of the profile now running (segments are never merged: each has its own I:C, CF and basal). For each: the settings it holds now, glucose over its own hours (time in range, below 70, below 54, above 180) for the last 90 days with the last 14 in brackets, a one-line reading (meeting the 70% target; below it with lows within limits; or lows over a limit, so no added insulin), how many breakfasts have been eaten at the current I:C (of the 5 needed), and what breakfasts at the earlier I:C say. A setting changed recently has few breakfasts at the new value; the table then shows the earlier setting's evidence beside it and says whether it supports the change.

**Settings periods.** Breakfasts are pooled by pump segment: the segment of the pump's own profile that covers the breakfast's time of day (`profile_segment_start_minutes`, from the profile history while edits are visible and from the segments as last seen afterwards). Within one pump, profile and pump segment, a period is a run of breakfasts with the same I:C and CF (CF rounded to 1 mg/dL/U). A stay on another profile in between does not split a period; an edit to this profile's own setting does. Each breakfast is compared with its own setting, so the estimate is the I:C that breakfasts at that setting behaved like.

**Checks.** Every period is re-pooled three more ways: without the looser tiers (baseline drift and exploratory windows dropped), on clean reference breakfasts alone, and without breakfasts whose calculator recorded a correction bolus (a missing record counts as none). The two logged-food (Bevel) sensitivities are existing analyses. A check with fewer than 5 breakfasts is reported as not assessable and does not block.

**Rule.** A proposal needs: at least 5 breakfasts at the current setting; the programmed I:C outside the primary 90% interval; every assessable check on the same side of the programmed value as the primary; and, when it adds insulin (a smaller g/U), time below 70 under 4% and below 54 under 1% over the segment's hours plus the 4 hours after, in the last 90 days (a change that removes insulin needs no such condition). The step is the smaller of 10% of the current value and the distance to the estimate, rounded to 0.1 g/U, toward the estimate. Only the segment covering most breakfasts of the last 90 days on the pump and profile now running is proposed for. How to read it: with a current I:C of 7.5 and an estimate of 6.5 (90% range 6.0 to 7.1), all checks agreeing and no lows, the proposal is 6.8: 10% of 7.5 is 0.75, less than the distance 1.0. How it could fail: carb-counting bias is absorbed into the estimate, so it is the ratio that works with how the carbs are counted; a correction factor that is wrong shifts every estimate the same way.

**Reproduction on real data.** For the Medium profile's segment starting 07:00 at 7.5 g/U (54 breakfasts to 2026-11-16, estimate 6.5, 90% range 6.0 to 7.1, both logged-food sensitivities agreeing, 0.2% below 70 and 0% below 54 over 07:00-13:00) the rule gives 7.5 to 6.8 by hand and by the code. The setting now running (Main, 7.0 g/U since 2026-11-16) has 1 breakfast, so the current result is "no change proposed: 5 are needed".

## Food the pump did not log (goal D1)

`meal_flags.py` compares the pump, Bevel (logged food) and the CGM, reading each flag in one direction only. **Food without pump carbs:** a Bevel entry of at least 5 g carbohydrate with no pump carb entry or bolus (0.5 U or more) within 30 minutes. **Possible unlogged food:** a CGM rise of at least 40 mg/dL within an hour that starts with no pump carbs, bolus or Bevel entry from 2.5 hours before to 1 hour after (a logged meal explains a rise for its absorption time). Both mark the 30 minutes before to 3 hours after the event; a breakfast window that overlaps one gets the exclusion reason `unlogged_food_in_window`, is left out of the I:C evidence, and is counted in the exclusion summary. **Bevel missing** (a pump carb entry with no Bevel entry within 30 minutes) is normal, since not every meal is logged in Bevel: it is reported and never read as "no meal", and analyses that need Bevel use only the pump carbs of that meal. On the real data: 47 Bevel entries without pump carbs and 1,062 possible unlogged rises over 2.8 years; of 124 breakfast windows that passed every other check, 14 overlap one (their rise was typically after treating a low or a snack that was never logged). The effective I:C at the earlier 7.0 g/U setting moved from 6.5 (90 breakfasts) to 6.4 (72), range 6.1 to 6.7. How it could fail: a late rise from slow-absorbing food eaten at the breakfast itself looks like a second, unlogged meal and costs a usable breakfast; the cost is a smaller sample, not a bias in one direction.

**Fat and protein (goal D2): tested in simulation, not adopted.** The plan was to lengthen the estimator's absorption-time prior by the logged fat and protein. The simulation harness has a `fat_protein_meals` scenario (0 to 4 fat-protein units, FPU = (9 x fat + 4 x protein) / 100; the Warsaw method's absorption durations of about 3, 4, 5 and 8 hours for 1, 2, 3 and 4 FPU hold back part of the carbohydrate). On the path that sets the effective I:C (logged carbs, vague carb prior) the estimated carbohydrate was already 9% above the truth with no lengthening (7% for meals above 2 FPU), and each added coefficient of 0.1 per FPU on the prior raised it (to 11% at 0.2 and 15% at 0.5; the error spread also grew). On the bolus-prior path of `validate-latent-meal` the scenario shows a bias of -8 g with no pre-bolus delay and +1 g with a 10-minute pre-bolus, so the sign depends on timing rather than on fat and protein. With no setting of the coefficient that improves recovery, the estimator is left as it is; breakfasts high in fat or protein stay in the primary and the existing sensitivity without them remains the check. How it could fail: real absorption of mixed meals that is slower than the simulated shapes would bias the estimate low, and the sensitivity analysis is what would show it.

## Refresh, change record and evaluation (goal C)

**Refresh.** `bayesian-t1dm refresh` runs by hand, in order: `collect` (latest 30 days from Tandem Source, including segment edits), `build-dataset`, `report-outcomes`, `propose-breakfast-icr --record`. Apple Health arrives by manual export into the raw tree before the run. It writes a run folder `runtime/output/refresh/<date>_<commit>/` with a `run_record.json` (commit, whether the working tree was clean, each step's exit code, status) and then publishes it to the Project Vault `outputs/<date>/` byte-identical with `MANIFEST.sha256`. It refuses to publish a run that is incomplete, started from a dirty working tree, or whose tree has changed since, and it never overwrites an earlier publish. `--no-collect` skips the Tandem download, `--no-publish` stops before the copy.

**Change record.** `data/records/setting_proposals.csv` in the Vault has one row per changed value. A proposal enters as `proposed`; when the pump's profile history shows the proposed value on the same segment it becomes `entered`; once its evaluation has run it is `evaluated`. A change the person made on the pump without a proposal is recorded as `origin = person` and evaluated the same way; only the profile now running is tracked, and each row shows how many other changes fell within 24 hours (they confound each other). One change is open at a time: a new proposal is not recorded while another is proposed or entered. The previous file is copied to `records/archive/` before every rewrite. Changes are read from the pump's own segment edits: a value is a change only when it differs from what the same segment held, so re-logged segments are not.

**Evaluation.** For each entered change: the 14 days before the change's day against the 14 days after it, over the segment's hours plus 4 hours, once all 14 after-days have been observed (until then the record says how many have). The measure is the mean of daily time in range, 70-180 mg/dL; the difference carries a day-level bootstrap 90% interval (4,000 resamples of days within each side, fixed seed), with time below 70 and below 54 reported against their 4% and 1% limits after the change. A day counts with at least 70% of its readings for those hours; days in `context_events.csv` are left out; at least 7 days per side are needed. How to read it: "in range 61.0% -> 68.5% (difference +7.5, 90% interval +2.1 to +12.9)" says the days after were higher by about 7 points and the interval excludes zero; the lows must still be within their limits. Before-versus-after in one person is associational: weather, illness, and other settings changed in the same days move it too, which is why the other changes within 24 hours are listed.

## Outcome metrics (goal A)

`bayesian-t1dm report-outcomes` writes `runtime/output/outcomes/outcomes_report.html` and CSV tables. Metrics follow the international consensus on CGM: time in range 70-180 mg/dL; time below 70 and 54; time above 180 and 250; mean glucose; coefficient of variation; glucose management indicator (3.31 + 0.02392 x mean). A day counts only when it holds at least 70% of its readings (for a part of the day, 70% of that part's readings). Readings are one per 5-minute slot. Time-of-day bins are the segments of the profile the latest pump runs, as last visible in its profile history. How to read it: a row whose time below 70 is 4% or more, or below 54 is 1% or more, is over a low limit and is not a candidate for more insulin in those hours. How it could fail: a profile edited since the last download leaves the bins one edit behind until the next `fetch-profile-edits`.

## Data Flow

The active pipeline is:

1. Tandem Mobi syncs to Tandem Source through the mobile app
2. `tconnectsync` logs in to Tandem Source and queries API windows directly
3. raw API responses are archived under `~/Library/CloudStorage/OneDrive-Personal/Project Vault/Personal Projects/bayesian-t1dm/data/raw/tconnectsync/<window_id>/raw/`
4. normalized CGM, bolus, basal, and activity tables are written beside the raw responses
5. optional Health Auto Export JSON files are read directly from anywhere under `data/raw/health_auto_export/`
6. `build-dataset` fingerprints every raw input (SHA-256), and only when one changed: normalizes changed Tandem windows in parallel, reads Tandem once, adds every Apple Health export JSON (byte-identical copies once, newest export wins where they overlap), builds the 5-minute dataset and writes it with `validation_report.md` to `data/merged/`. Analyses load that dataset and refuse to run if a raw file changed since the build.
7. a file whose size and modification time match the previous build keeps its recorded hash instead of being read again, so the staleness check takes about a second; any file whose modification time changed is re-hashed, so a sync that touches a file without changing it does not trigger a rebuild. A first build, `--force` or a change of `DATASET_VERSION` in the code re-normalizes every window
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

- code: `~/Workspaces/bayesian-t1dm/source`
- runtime outputs, scratch space, and intermediate artifacts: `~/Workspaces/bayesian-t1dm/runtime`
- cloud project home: `~/Library/CloudStorage/OneDrive-Personal/Project Vault/Personal Projects/bayesian-t1dm`
- scraped/raw acquisition data: `~/Library/CloudStorage/OneDrive-Personal/Project Vault/Personal Projects/bayesian-t1dm/data/raw`
- final published outputs: `~/Library/CloudStorage/OneDrive-Personal/Project Vault/Personal Projects/bayesian-t1dm/output`
- default working summaries: `~/Workspaces/bayesian-t1dm/runtime/output`

Raw discovery rules:

- active Tandem ingest excludes deprecated `archive data/` content from generic raw discovery
- canonical Apple Health loading reads only `data/raw/health_auto_export/...`
- `data/raw/health_auto_export/` holds Apple Health exports as dropped (`HealthAutoExport-*.json`); the raw JSON is read directly, nothing is copied

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

`build-dataset` rebuilds `normalized/*.csv` and `window_manifest.csv` from the archived raw responses itself, without refetching. `bayesian-t1dm normalize-raw` does the same for one window or all windows by hand and then refreshes `tandem_export_manifest.csv`.

Normalized CSV contract:

- timestamps are stored as timezone-naive local wall-clock timestamps in `TIMEZONE_NAME`
- a value with a UTC offset is converted to `TIMEZONE_NAME`; a value without one is already local time; a value that is present but not ISO 8601 stops the build with an error rather than dropping the row
- a CGM reading's time is the sensor's measurement time on the pump clock, stored without an offset. Example: a reading measured at 01:58 on the day clocks jump forward and received at 03:01 is stored at 01:58. Attaching the receipt's offset (EDT) to the measurement (EST) would have moved it an hour
- Apple Health times keep the wall-clock time the phone recorded and drop its offset

## Bolus calculator settings

Each breakfast is compared with the I:C ratio and correction factor (CF) the bolus calculator used for it.

- **t:slim** records the ratio on `LID_BOLUS_REQUESTED_MSG1` (in milligrams per unit) and the CF and target on `LID_BOLUS_REQUESTED_MSG2`.
- **Mobi** (boluses from the phone app) records all three as 0. `LID_BOLUS_REQUESTED_MSG3` records the food and correction portions of the bolus, which are enough to recover them.

**I:C.** The food portion is carbs / I:C, so I:C = carbs / food portion, rounded to the pump's 0.1 g/U steps. That holds only while the calculator has not folded a below-target correction or insulin on board into the food portion, which is the case when the bolus includes a positive correction or glucose was at or above the recorded target. On 1,405 t:slim meals that record the ratio, the recovered value matched in all 1,082 meals meeting that condition; in the other 323 the food portion was reduced and no ratio is recovered.

Example: 35 g with a food portion of 4.67 U and a 0.38 U correction gives 35 / 4.67 = 7.49, so 7.5 g/U. The same 35 g at glucose 95 with no correction may have had 0.3 U taken off the food portion (glucose below a 110 target with CF 50), which would read as 8.0 g/U; that meal gets no recovered ratio.

**CF and target.** The calculator's correction is (glucose - target) / CF - insulin on board. Within a run of breakfasts on one pump with one I:C, correction + insulin on board is therefore a straight line in glucose with slope 1 / CF and intercept -target / CF. Over breakfasts with a correction of at least 0.3 U (smaller ones are too coarsely rounded), a Theil-Sen fit finds the majority setting, and a least-squares refit of the breakfasts within 5% of it gives the run's CF and target; the refit keeps a few breakfasts with another CF from pulling the estimate. A breakfast whose own (glucose - target) / (correction + insulin on board) is more than 5% from the run's CF keeps its own value, because a setting changed within the run. On t:slim groups with a recorded CF and target, the fit recovers both to within 0.1 mg/dL/U and 0.1 mg/dL.

Example: on the Mobi from June 2026, 41 morning corrections fit CF 60.0 and target 110.1 with no residual above 0.01 U.

**Pump profile history.** The pump logs every profile edit and switch: `LID_IDP_TD_SEG` (event 68) gives one time-of-day segment of a profile after a change, with its start time, basal, CF, target and I:C, and `LID_IDP` (event 69) records a profile being created, copied, deleted or activated. tconnectsync does not decode either; `pump_profiles.py` reads them from the archived raw events. Tandem Source stores each 4-byte word of the pump's record in reverse byte order, and once that is undone the layout is pumpX2's, except that the I:C is a 16-bit value in mg/U. Replaying the events in order gives, for any moment, the active profile and the segment covering that time of day. The `modificationType` of a segment record is 1 when the segment is deleted (a copied profile is trimmed by deleting segments one at a time) and otherwise leaves the segment at its index with the logged values; a copy starts from the source profile's segments at the copy time, and a delete clears the profile. A profile the archive never saw created may have segments it never logged, so the segment after the last one seen is unknown. Compared with the carb ratio recorded by the t:slim's own calculator, the replay agrees for 1,985 of 1,987 entries that have both (an earlier reading of the codes, 1 = insert and 2 = delete, agreed for 99.7% and left duplicate start times in 93 snapshots; none remain). How it could fail: a modification code with another meaning in a pump firmware the archive does not show would place a segment at the wrong index.

Event 68 reaches the archive two ways: the binary event files of windows collected with tconnectsync 2.x (to April 22, 2026 for the Mobi), and `pump_profile_edits_<pump>_<device>.json`, which the collector writes for every window by asking Tandem Source for event code 68 by name (the server's default event list leaves it out, and the pump-logs JSON of tconnectsync 3.x carries profile switches, event 69, without segment edits). `fetch-profile-edits` adds those files to windows already archived without rewriting anything in them. A pump's visible span runs over the first unbroken run of windows with either source; for the Mobi it now reaches the day of the last download, so a profile's values are known at every meal. With the 2026 edits, the replay agrees with the carb ratio recorded or recovered by the calculator for 2,333 of 2,336 entries (99.9%). Each carb entry keeps the moment the pump recorded (`event_time`) beside its 5-minute grid slot (`timestamp`), and the profile is read at `event_time`, so an edit made inside a slot applies to the meals after it and not to those before it (this changed the settings of 9 of about 3,680 entries). Glucose, insulin on board and basal stay on the 5-minute grid: across a 5-minute step CGM changes by a median 3 mg/dL (90th percentile 11), so reading the grid slot instead of the exact moment costs about 1.5 mg/dL at the median; basal is a step function that is held, not interpolated, and insulin on board is computed from exact bolus times. `build-dataset` writes the decoded events to `tables/pump_profile_events.parquet` and the visible span per pump to the validation report. Example from the data: on 2026-11-17 at 09:29 the Main profile's 05:00 and 09:00 segments went from 7.0 to 6.5 g/U and the 21:00 segment from 7.0 to 6.8; a 07:30 breakfast that morning was dosed at 7.0, and the 11:50 meal at 6.5.

**Precedence.** For each breakfast, the I:C and CF each come from the first of: the bolus calculator (recorded, or the I:C recovered from the food portion); the profile history; the CF recovered from the corrections; the nearest breakfast in the same unbroken stretch on one pump, profile and edit visibility (a switch to another profile and back starts a new stretch); the programmed profile. The source shown is the weakest one used. On Mobi breakfasts between August 2025 and April 2026, the recovered I:C and CF agreed with the profile history in all 44 breakfasts that had both. Carried values were the weak point: 36 of 115 carried CFs disagreed with the history, because carrying crossed profile switches.

Example: on the Mobi, Main at 07:00 is 7.0 g/U with CF 65, and Medium is 8.0 g/U with CF 60 and target 120. A breakfast at 07:30 on September 12, 2025, after Medium was activated on September 10, uses Medium's values. A breakfast in June 2026, after the binary archive ends, uses its own recovered I:C and CF, or those of nearby June breakfasts on Medium. It does not use the 8.0 last seen in 2025, because Medium was later changed to 7.5 without the edit appearing in the download.

**Breakfasts with nothing usable** (no correction and glucose below target, no visible profile history, or a hand-logged meal) take the setting of the nearest earlier breakfast on the same pump and profile, or the nearest later one, so they do not start a settings period of their own. A value missing for every breakfast on that pump and profile comes from the programmed profile, and only that value is replaced. A settings period is a run of breakfasts with the same I:C and CF. How it could fail: a setting changed and changed back between two usable breakfasts, after the visible history ends, would be missed. With breakfasts most days, that window is about a day.

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
- across a CGM gap the model-facing `glucose` column carries the last reading forward (`FeatureConfig.cgm_fill_method = "locf"`). Linear interpolation (`"linear"`) is available for retrospective plots but must not be used for forecasting: a gap row would be filled using the reading at the far end of the gap, which is often its own target. `glucose_observed` is never filled.
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

For each candidate scenario $s$, the pipeline re-scales the exposure features for basal, bolus, and I/C ratio changes, recomputes the derived insulin features, and predicts with every posterior draw $k$.

The implemented loss is the mean absolute deviation of the predicted mean trajectory from the glucose target $y^\star$ (110 mg/dL by default):

$$
L_k(s) = \frac{1}{T} \sum_{t=1}^{T} \left| \mu_{t,k}(s) - y^\star \right|
$$

Hypo- and hyperglycaemia terms are not part of the loss; a scenario that trades a little mean deviation for more lows is not penalised for it.

Two numbers are reported per scenario:

- expected gain: $\frac{1}{T}\sum_t |\bar\mu_t(\text{current}) - y^\star| - \frac{1}{T}\sum_t |\bar\mu_t(s) - y^\star|$ using posterior-mean predictions
- posterior probability better: $\Pr_k\left[L_k(s) < L_k(\text{current})\right]$, the share of posterior draws in which the scenario beats current settings. Draws are paired: both scenarios use the same parameter draw and the same latent-state innovations.

A candidate is surfaced only if the expected gain clears the configured threshold, the posterior probability is at least 0.75, and the recommendation policy (validation, sampler diagnostics, data quality, carb and activity signal) passes.

Two caveats that matter more than the arithmetic:

- The model is a predictive regression on observational, closed-loop data. Its coefficients are not causal effects of insulin, so a scenario is a sensitivity analysis of the fitted model, not a prediction of what would happen if the setting were changed.
- With the current Tandem archive carbohydrate records are absent, so the policy suppresses every recommendation. This lane has never produced a recommendation on real data.

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

Canonical research loop:

1. `bayesian-t1dm build-dataset` to normalize and merge any new raw data into `data/merged/` (about a second when nothing changed, under a minute for a full rebuild).
2. `bayesian-t1dm research-latent-meal-icr --research-scope foundation` to inspect honest meal-truth semantics and first-meal clean-window readiness.
3. `bayesian-t1dm research-latent-meal-icr --research-scope full` only after the foundation cohort is actually ready for latent fitting.
4. `bayesian-t1dm status` when you want the combined snapshot across therapy, forecast, and latent-meal lanes.
5. `bayesian-t1dm run --skip-recommendations` only as a secondary forecast diagnostic.
6. `bayesian-t1dm review-therapy-evidence` only when you need secondary therapy identifiability context.

Artifact reading order:

1. `current_status.html` for the combined overview.
2. `therapy_review.html` for secondary therapy identifiability.
3. `forecast_review.html` for secondary forecast diagnostics.
4. `output/latent_meal/` and `output/fixture/` for the scientific priority lane.

If the latent-meal foundation cohort is blocked or empty, debug cohort quality, source truth, and freshness first. Do not treat that as a model-fitting failure.

Step-by-step behavior:

1. `normalize-raw` rebuilds normalized Tandem tables from archived `tconnectsync` raw payloads by hand; `build-dataset` does this itself for changed windows.
2. `build-dataset` merges Tandem and Apple Health into one deduplicated 5-minute dataset when raw inputs change, and writes `validation_report.md` (inputs, spans, row counts, completeness and plausibility checks).
3. `research-latent-meal-icr --research-scope foundation` audits meal-truth semantics, builds the first-meal clean-window cohort, and tells you whether the accepted cohort is ready for Phase C. This is the main scientific checkpoint.
4. `research-latent-meal-icr --research-scope full` runs the constrained first-meal latent-response approximation on accepted windows only and writes research-only posterior, confidence, and model-comparison artifacts.
5. `status` refreshes the combined snapshot and publishes the latest top-level HTML entrypoints, but it is still a combined overview rather than the primary scientific loop.
6. `screen-health-features` consumes the prepared dataset and skips cleanly when Apple Health is absent.
7. `run --skip-recommendations` is the preferred fast validation path for forecast behavior because it exercises walk-forward prediction without the final recommendation fit.
8. `run` executes the full modeling and recommendation stack using the same prepared dataset contract: Apple-enriched when Apple data exists, Tandem-only otherwise.
9. `research-therapy-settings` runs a distinct research-grade therapy workflow: it starts with a methodological gate, builds fasting, meal, and correction contexts, creates strict meal-bolus proxy features when explicit carbs are absent, audits Apple/Tandem feature engineering, compares Bayesian, linear, segmented, tree-boosted, and ensemble-style candidates, and writes segment-level recommendation evidence for basal and I/C ratio. Sensitivity factor remains staged and suppressed by default in the current implementation.
10. `validate-therapy-infra` runs a synthetic truth-recovery suite against the same therapy research contract. It checks basal direction recovery, proxy-only conservatism, Apple-helpful vs Apple-null behavior, and suppression under corrupted or low-identifiability scenarios.
11. `review-therapy-evidence` is the therapy-facing orchestration command. It prepares the model dataset, writes the therapy research and validation artifacts, computes the overnight basal evidence summary, and renders one interactive HTML page that explains what is identifiable, what is blocked, and where in the code and artifacts the logic lives.

Workflow crosswalk:

| Step | Question | Main artifact(s) | Main code path(s) | Interpretation |
| --- | --- | --- | --- | --- |
| `normalize-raw` | Did we correctly rebuild normalized Tandem windows? | `normalize_raw_summary.md` | `src/bayesian_t1dm/acquisition.py::normalize_tconnectsync_archive` | Confirms raw-to-normalized reconstruction before modeling. |
| `build-dataset` | What data went in, and is the merged dataset valid? | `data/merged/validation_report.md`, `data/merged/dataset.json`, `output/prepare/model_data_preparation.md` | `src/bayesian_t1dm/dataset.py::build_merged_dataset` |
| `research-latent-meal-icr --research-scope foundation` | Is the first-meal cohort ready for Phase C? | `output/latent_meal/latent_meal_research_gate.md`, `output/latent_meal/first_meal_clean_window_audit.md` | `src/bayesian_t1dm/therapy_research.py::run_latent_meal_icr_research` | Main scientific checkpoint for latent-meal readiness. |
| `research-latent-meal-icr --research-scope full` | Does the accepted first-meal cohort support useful latent meal fitting yet? | `output/latent_meal/latent_meal_fit_summary.md`, `output/latent_meal/latent_meal_posterior_meals.csv`, `output/latent_meal/latent_meal_model_comparison.md` | `src/bayesian_t1dm/therapy_research.py::run_latent_meal_icr_research` | Research-only constrained latent-response fit on accepted windows. |
| `validate-latent-meal` | Does the latent carb estimator recover known truth in simulation? | `output/latent_meal/validation/latent_meal_truth_recovery.md` | `src/bayesian_t1dm/latent_meal_validation.py::run_latent_meal_truth_recovery` | Simulated meals with known carbs; scores bolus-implied, the old peak/AUC heuristic, and the trajectory estimator. Exits non-zero if the trajectory estimator stops beating bolus x I:C. |
| `status` | What is the combined current state across lanes? | `current_status.html`, `therapy_review.html`, `forecast_review.html` | `src/bayesian_t1dm/status.py::derive_current_status` | Combined overview, not the primary science loop. |
| `run --skip-recommendations` | Does walk-forward forecasting still behave sensibly? | `forecast_review.html`, `cache/forecast/run_summary.json` | `src/bayesian_t1dm/report.py::build_run_summary`, `src/bayesian_t1dm/review.py::write_run_review_html` | Secondary forecast diagnostic. |
| `research-therapy-settings` | What therapy contexts and identifiability gates are available? | `therapy_research_gate.md`, `therapy_feature_audit.md`, `therapy_model_comparison.md` | `src/bayesian_t1dm/therapy_research.py::build_therapy_research_frame`, `src/bayesian_t1dm/therapy_research.py::run_therapy_research` | Shows whether basal / I:C / ISF are directly observed, proxy-supported, weakly identified, or blocked. |
| `validate-therapy-infra` | Does the therapy workflow recover known truth and suppress itself correctly? | `therapy_infra_validation.md`, `therapy_synthetic_results.csv` | `src/bayesian_t1dm/therapy_research.py::validate_therapy_infra` | Tests truth-recovery and false-positive suppression, not just predictive fit. |
| `review-therapy-evidence` | What is actually happening with overnight basal evidence in the current data? | `output/therapy/therapy_review.html` | `src/bayesian_t1dm/review.py::write_therapy_evidence_review_html` | Secondary therapy-facing explanation surface. |

## Latent Carb Estimator

Code: `src/bayesian_t1dm/latent_meal_trajectory.py`, called from `therapy_research.py::_latent_carb_posterior_for_window`.

For an accepted first-meal window, with $t$ in minutes after the bolus, the forward model is

$$
G(t) = G_0 + b_0 + b_1 t + \frac{\text{CF}}{\text{ICR}}\, C \, A(t - \delta;\, a, m) - \text{CF}\,(1 + d)\, U(t) + \varepsilon(t)
$$

- $C$ = meal carbohydrate (grams), the quantity of interest
- $A$ = fraction of the meal absorbed, a gamma CDF with shape $a \in \{1.5, 2.5, 4, 7\}$ and mean absorption time $m$ (35 to 170 min); $\delta \in \{0, 10, 20\}$ min is the delay between bolus and carbohydrate appearance
- $U(t)$ = units of insulin absorbed since the anchor. The bolus part is reconstructed from the frame as $\text{IOB}(0) + \text{boluses after } 0 - \text{IOB}(t)$, so later boluses and insulin already on board are included. Closed-loop basal is added on top: delivery above or below the scheduled rate is pushed through the same action curve (negative when basal was suspended). The schedule is not in the frame, so it is estimated as the median delivery at the same time of day over the surrounding 43 days
- CF, ICR = programmed correction factor and I:C ratio for the segment
- $b_0, b_1, d$ = baseline offset (SD 5 mg/dL), linear drift (SD 8 mg/dL/h) and relative correction-factor error (SD 0.15). They have zero-mean Gaussian priors and are integrated out in closed form
- $\varepsilon$ = AR(1) noise, SD 8 mg/dL, correlation 0.7 per 5 minutes

The prior on $C$ is centred on bolus x ICR with SD $\max(20, 0.5 \times \text{mean})$ grams (tighter when a carb entry was logged). The posterior over $(C, a, m, \delta)$ is evaluated on a grid. Analysis tiers scale every noise and nuisance SD by the tier's uncertainty multiplier.

Per-meal outputs include `estimator` (`trajectory` or `summary_heuristic`), `prior_contraction` ($1 - \text{posterior var}/\text{prior var}$: how much the trace told us beyond the bolus) and `trajectory_fit_rmse_mgdl`.

What it cannot do:

- Glucose identifies $C \cdot \text{CF}/\text{ICR}$, not $C$. The estimate is a carb-equivalent under the programmed settings. Carb-counting error and a wrong I:C ratio are not separable from a single meal without carb truth.
- Carbohydrate absorbed after the 180-minute window is invisible, so slow high-fat meals are biased low.
- Basal modulation is counted relative to an inferred schedule. If the controller modulates basal at the same time on most days, the inferred schedule absorbs it and the modulation goes uncounted.

The older estimator (peak and AUC summaries with fixed response constants) is retained as a fallback for windows whose trace or insulin columns are unusable, and is flagged `summary_heuristic_estimator`.

### Synthetic truth recovery

`bayesian-t1dm validate-latent-meal` simulates meals with known carbohydrate, each bolused for a mis-counted amount, and scores each estimator's posterior median against truth. With 200 meals per scenario and the default seed:

| scenario | bolus x ICR MAE (g) | heuristic MAE / 90% coverage | trajectory MAE / 90% coverage |
| --- | ---: | ---: | ---: |
| matched assumptions | 17.2 | 13.8 / 0.66 | 3.9 / 0.99 |
| long-tail absorption | 17.2 | 17.7 / 0.55 | 2.5 / 1.00 |
| biphasic absorption | 17.2 | 17.5 / 0.55 | 17.8 / 0.71 |
| pre-bolus timing | 16.0 | 11.7 / 0.72 | 4.4 / 1.00 |
| wrong settings (15%) | 17.2 | 15.5 / 0.57 | 7.9 / 0.91 |
| baseline drift | 19.5 | 14.9 / 0.62 | 6.2 / 0.98 |
| closed-loop basal | 17.2 | 13.5 / 0.66 | 4.2 / 1.00 |
| everything at once | 16.8 | 15.5 / 0.55 | 9.1 / 0.91 |

The simulator is linear and has no exercise and no sensor lag; its closed-loop controller is a simple threshold rule, not Control-IQ. These numbers say the estimator is internally sound and robust to the listed misspecifications; they do not say it is accurate on real meals.

### Effective I:C from logged meals

Code: `latent_meal_trajectory.py::observe_carb_sensitivity`, `pool_effective_icr`; wired in `therapy_research.py::_effective_icr_evidence`.

When an accepted window carries a logged carb count $S_i$, the ambiguity above disappears for that meal. The trajectory estimator is re-run with a deliberately vague carb prior (SD at least 60 g) to get the carb-equivalent $\hat{C}_i$ the trace supports, and the meal contributes

$$
\log r_i = \log\left(\hat{C}_i / S_i\right), \qquad \text{sd}(\log r_i) \approx \text{sd}(\hat{C}_i)/\hat{C}_i
$$

$r_i > 1$ means glucose rose more per logged gram than CF / ICR predicts. Meals in a segment are pooled with a DerSimonian-Laird random-effects model (between-meal variance $\tau^2$) and a Hartung-Knapp $t$ interval, so genuine meal-to-meal variability widens the result. The effective ratio is

$$
\text{ICR}_{\text{effective}} = \text{ICR}_{\text{programmed}} / r
$$

At least 5 usable logged meals are required; below that the status is `insufficient_logged_meals`. The estimate assumes the programmed correction factor is right: a correction factor that is off by x% moves the I:C estimate by about x%.

Carb logs enter the pipeline as a `carbs.csv` (or `carb_log.csv`) with `timestamp,carb_grams` columns anywhere in the raw data tree; this hand-kept file is the one table accepted outside a `normalized/` directory. Entries from 20 minutes before to 30 minutes after the first-meal bolus are summed as that meal's logged carbs.

Synthetic recovery (`validate-latent-meal`, 40 cohorts of 20 logged meals, programmed I:C 10, with long-tail absorption, pre-bolus timing, drift, noisy CGM and 10% logging error):

| true I:C | mean estimate | SD across cohorts | 90% coverage | flagged as different |
| ---: | ---: | ---: | ---: | ---: |
| 8.0 | 7.99 | 0.25 | 0.90 | 1.00 |
| 10.0 | 9.87 | 0.34 | 0.95 | 0.05 |
| 12.5 | 12.16 | 0.49 | 0.95 | 1.00 |

## Output Contract

`cache/forecast/run_summary.json` includes:

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

- `current_status.html` from `status`
- `coverage_review.html` from `ingest`
- `forecast_review.html` from `run`
- `therapy_review.html` from `review-therapy-evidence` / `status`

These are the primary visual inspection surfaces for active-pipeline review. They combine source completeness, walk-forward behavior, baseline comparison, and sampler diagnostics in one place.

Apple Health-specific working artifacts:

- `build-dataset` writes the merged tables, `prepared_model_data_5min.parquet`, `dataset.json` (input fingerprints, spans, row counts) and `validation_report.md` under `data/merged/`
- `build-dataset` also writes `model_data_preparation.md` under `output/prepare/` when it rebuilds
- CGM rows come only from sensor-reading events (`LID_CGM_DATA_*`), timed by the sensor's measurement time (`egvTimeStamp`); the pump's receipt time lags it for readings backfilled after a signal dropout
- window completeness covers the download only (`missing_kind`, `starts_late`, `ends_early`); `cgm_slot_coverage` and `cgm_gaps_ge_2h` are reported per window as data quality
- logged food from Apple Health (`carbohydrates`, `protein`, `total_fat`, one row per entry time) is matched to each breakfast within 30 minutes. It never replaces the pump-entered carbs: the effective I:C is rerun (a) without breakfasts with at least 20 g fat or 25 g protein and (b) with each meal's carbs as the geometric mean of the pump entry and the logged-food carbs divided by their median ratio over all meals that have both
- `context_events.csv` in the raw folder lists date ranges (`start_date,end_date,reason`) left out of the meal and overnight-basal analyses
- `screen-health-features` writes `health_feature_screening.md` under `output/prepare/` and `health_feature_scores.csv` under `cache/prepare/`
- `research-latent-meal-icr` writes foundation or full-scope latent-meal artifacts under `output/latent_meal/`
- `build-latent-meal-fixture` writes the fixture audit bundle under `output/fixture/` and the reduced prepared dataset under `cache/latent_meal/`
- `research-therapy-settings` writes its research artifacts under `output/therapy/`
- `validate-therapy-infra` writes `therapy_infra_validation.md`, `therapy_synthetic_results.csv`, and `therapy_synthetic_recommendation_audit.md` under `output/therapy/validation/`
- `review-therapy-evidence` writes `output/therapy/therapy_review.html` and publishes the latest top-level `output/therapy_review.html`

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
