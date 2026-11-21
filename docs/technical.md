# Technical reference

How each step of the pipeline works, in the order the data move through it: download, normalization, the merged dataset, the settings in force, the shared definitions, the I:C and basal evidence, and the readout. [`methods.md`](methods.md) states the goal and the methods in brief; this document gives the rules, parameters and failure modes. Run numbers live in the generated readout and its diagnostics workbook, not here.

## Contents

- [Pipeline at a glance](#pipeline-at-a-glance)
- [Download from Tandem Source](#download-from-tandem-source)
- [Normalization and timestamps](#normalization-and-timestamps)
- [Merged dataset](#merged-dataset)
- [Settings in force](#settings-in-force)
- [Shared definitions](#shared-definitions)
- [Outcome metrics](#outcome-metrics)
- [Insulin action kernel](#insulin-action-kernel)
- [Effective I:C from meals](#effective-ic-from-meals)
- [Basal from the controller pattern](#basal-from-the-controller-pattern)
- [Personal readout](#personal-readout)
- [Assumptions and failure modes](#assumptions-and-failure-modes)

## Pipeline at a glance

| Step | Command | Reads | Writes |
| --- | --- | --- | --- |
| Download | `collect` (one window, at most 30 days), `backfill` (a range), `fetch-profile-edits` | Tandem Source through tconnectsync | raw responses and normalized tables under `<cloud root>/data/raw/tconnectsync/<window>/`, `tandem_export_manifest.csv` |
| Normalize | `normalize-raw` (by hand; `build-dataset` does it for changed windows) | archived raw responses | `normalized/*.csv`, `window_manifest.csv` |
| Build | `build-dataset` | every raw file (Tandem, Apple Health exports, carb logs, `context_events.csv`) | `<cloud root>/data/merged/` with `validation_report.md` |
| Readout | `readout` | the merged dataset | `runtime/output/readout/` (page, workbook, CSVs), page copied to `<cloud root>/readout/readout.html` |
| All of it | `refresh` | | collect, build-dataset, readout in order, stopping at the first failure |

Every command logs to `runtime/logs/<command>/<run>/events.jsonl`. The cloud root is the Project Vault folder (`BAYESIAN_T1DM_CLOUD_ROOT`), the runtime root holds generated work (`BAYESIAN_T1DM_RUNTIME_ROOT`); both default as in [`paths.py`](../src/bayesian_t1dm/paths.py).

## Download from Tandem Source

Tandem Source is the source of truth for pump data: the Mobi app uploads to it and `tconnectsync` queries it directly (API first; there is no browser path). Credentials come from `.env` or the environment only: `TANDEM_SOURCE_EMAIL`, `TANDEM_SOURCE_PASSWORD`, `TANDEM_SOURCE_REGION` (non-US accounts), `TIMEZONE_NAME`, and a pump serial when the account has more than one pump.

Each window keeps the raw API responses untouched under `raw/` and writes normalized CGM, bolus, basal and activity tables beside them. The acquisition manifest (`tandem_export_manifest.csv`) records, per window: the requested and observed first and last timestamps, the observed duration, `coverage_fraction`, raw payload hashes and sizes, the time zone and pump identity of the request, and `completeness_reasons` (`starts_late`, `ends_early`, `internal_gap`, `duplicates`, `overlap`, `missing_kind`). CGM coverage within a window (`cgm_slot_coverage`, `cgm_gaps_ge_2h`) is reported as data quality and does not make a window incomplete.

The pump's profile edits (event 68) are not in tconnectsync's default event list, so the collector asks for them by name and writes `pump_profile_edits_<pump>_<device>.json` for every window; `fetch-profile-edits` adds that file to windows archived before this was done, without rewriting anything else.

How to read it: a window with `ends_early` stopped before its requested end (the pump had not synced the last hours); the next `collect` fills it. How it could fail: Tandem Source can change undocumented endpoints, which shows up as a failed collect or a window with `missing_kind`, never as silently shorter data.

## Normalization and timestamps

`build-dataset` rebuilds `normalized/*.csv` and `window_manifest.csv` from the archived raw responses for every window whose raw files changed, without refetching; `normalize-raw` does the same by hand for one window or all of them and refreshes the manifest.

- Timestamps are stored as time-zone-naive local wall-clock times in `TIMEZONE_NAME`. A value with a UTC offset is converted; a value without one is already local; a value that is present but not ISO 8601 stops the build instead of dropping the row.
- A CGM reading's time is the sensor's measurement time on the pump clock (`egvTimeStamp`), not the time the pump received it. Example: a reading measured at 01:58 on the night clocks go forward and received at 03:01 is stored at 01:58.
- CGM rows come only from sensor-reading events; readings backfilled after a signal dropout keep their measurement time.
- Apple Health times keep the wall-clock time the phone recorded.

How it could fail: a pump whose clock was not yet set logs events in 2020; events outside the span the archive covers for that pump are dropped when the dataset is built.

## Merged dataset

`build-dataset` fingerprints every raw input with SHA-256 (a file whose size and modification time match the previous build keeps its recorded hash) and does work only when a fingerprint changed, `--force` is given or `DATASET_VERSION` in [`dataset.py`](../src/bayesian_t1dm/dataset.py) was bumped. It then normalizes the changed Tandem windows, reads Tandem once, adds every Apple Health export (`data/raw/health_auto_export/HealthAutoExport-*.json`; the newest export wins where two overlap), and writes to `<cloud root>/data/merged/`:

- `tables/*.parquet`: CGM, bolus, carbohydrate, basal delivery, Apple Health measurements (heart rate, resting heart rate, HRV, respiratory rate, steps, weight), sleep, workouts and logged food, each deduplicated;
- `tables/pump_profile_events.parquet` and the decoded Control-IQ mode changes and site fills;
- `analysis_grid_5min.parquet`: every 5-minute slot with observed glucose (empty where there is none), boluses, carbohydrate, insulin on board and the calculator's carb ratio;
- `prepared_model_data_5min.parquet`: the 5-minute frame the meal estimator reads, with Apple Health context joined where available;
- `dataset.json` (input fingerprints, spans, row counts, the build's commit) and `validation_report.md` (inputs, spans, completeness and plausibility checks).

Every analysis loads the dataset with `load_merged_dataset`, which refuses to return data whose raw inputs changed since the build. Two hand-kept inputs live in the raw folder: `context_events.csv` (`start_date,end_date,reason`: days such as illness or a steroid injection that analyses leave out) and an optional carb log (`carbs.csv` or `carb_log.csv` with `timestamp,carb_grams`, the one table accepted outside a `normalized/` folder).

How to read it: when the readout and the validation report disagree about a span or a count, the validation report describes what went in and is the first place to look. How it could fail: a rule changed in code without a `DATASET_VERSION` bump leaves an old dataset in place; the version bump forces the rebuild.

## Settings in force

The I:C, correction factor, basal rate and target in force at any moment come from replaying the pump's own profile events ([`pump_profiles.py`](../src/bayesian_t1dm/pump_profiles.py)).

**Profile history.** `LID_IDP_TD_SEG` (event 68) gives one time-of-day segment of a profile after a change (start time, basal, correction factor, target, I:C); `LID_IDP` (event 69) records a profile being created, copied, deleted or activated. Tandem Source stores each 4-byte word of the record in reverse byte order; once that is undone the layout is pumpX2's, with the I:C as a 16-bit value in mg/U. A segment record with modification type 1 deletes the segment, 0 inserts one at its index, and any other value sets the segment at its index; a copy starts from the source profile's segments at that moment. Replaying the events gives, for any moment, the active profile and the segment covering that time of day. A carb entry keeps the exact moment the pump recorded (`event_time`) beside its 5-minute slot, and the profile is read at that moment, so an edit made inside a slot applies only to the meals after it.

**A profile's last slot.** A profile created before the archive shows its slots one edit at a time, so the archive alone cannot say whether another slot follows the last one seen. The pump logs the rate the active profile calls for about every five minutes (event 279). The last slot is taken to run to midnight over a stretch in which it keeps its start, rate and index when at least 12 of those records from 23:00 on equal the slot's rate in 95% or more of cases; otherwise its hours stay unknown. How it could fail: a hidden slot with the same basal rate would go unnoticed by the rate log.

**The bolus calculator's own settings.** Each meal is also compared with the I:C and correction factor its bolus calculator used. The t:slim records them; the Mobi records zeros, and they are recovered from the food and correction portions of the bolus: I:C = carbohydrate / food portion, rounded to 0.1 g/U, only when the calculator did not fold a below-target correction or insulin on board into the food portion; the correction factor and target from a Theil-Sen line through correction + insulin on board against glucose, refit by least squares on the meals within 5% of it. For each meal the I:C and correction factor come from the first available of: the calculator (recorded or recovered), the profile history, the recovered correction factor, the nearest meal in the same unbroken stretch on one pump and profile, the programmed profile. Example: 35 g with a food portion of 4.67 U and a 0.38 U correction gives 35 / 4.67 = 7.49, so 7.5 g/U.

How to read it: pick a meal, read the pump's profile screen at that moment, and the settings attached to the meal should match it. How it could fail: a setting changed and changed back between two meals, after the visible profile history ends, would be missed.

## Shared definitions

`analysis_data.load_analysis_data` reads the merged dataset once and gives every analysis the same pieces: valid CGM, the 5-minute grid, the replayed profile history, the latest pump and its active profile, one moment at which the time-of-day segments and settings are read (the last day the profile edits are visible), Control-IQ modes and site fills, unlogged-food intervals, exercise intervals and context-event days. [`context_rules.py`](../src/bayesian_t1dm/context_rules.py) states the shared rules once:

- *CGM validity*: Tandem readings from 40 to 400 mg/dL count. Apple Health blood glucose is not used.
- *Exercise*: a workout of 10 minutes or more (walks included), from its start until 4 hours after its end, and exercise mode on the pump from its start until 4 hours after it ends.
- *Analysis day*: a time-of-day segment that wraps midnight belongs to the date it started.

**Food the pump did not log** ([`meal_flags.py`](../src/bayesian_t1dm/meal_flags.py)). *Food without pump carbs*: a logged-food entry of at least 5 g carbohydrate with no pump carb entry or bolus of 0.5 U or more within 30 minutes. *Possible unlogged food*: a CGM rise of at least 40 mg/dL within an hour with no pump carbs, bolus or logged food from 2.5 hours before to 1 hour after. Both mark 30 minutes before to 3 hours after the event; meal windows and clean basal time that overlap one are left out. How it could fail: slow food eaten at a logged meal can produce a late rise that looks unlogged; the cost is a smaller sample, not a bias in one direction.

## Outcome metrics

Consensus CGM metrics ([`outcomes.py`](../src/bayesian_t1dm/outcomes.py)): time in range 70-180 mg/dL, below 70 and 54, above 180 and 250, mean glucose, coefficient of variation and the glucose management indicator (3.31 + 0.02392 x mean). Readings are one per 5-minute slot; a day (or a part of a day) counts only with at least 70% of its expected readings. Targets: time in range above 70%, below 70 under 4%, below 54 under 1%. Time-of-day segments are the segments of the profile the latest pump runs. How to read it: a segment with time below 70 of 4% or more is over a low limit and gets no suggestion that adds insulin. How it could fail: a profile edited since the last download leaves the segments one edit behind until the next `fetch-profile-edits` and `build-dataset`.

## Insulin action kernel

Insulin exposure uses a fixed-duration action curve ([`insulin.py`](../src/bayesian_t1dm/insulin.py)) with dose $D$, minutes since the dose $t$, duration $t_d = 300$ and peak $t_p = 75$:

$$
\tau = \frac{t_p (1 - t_p / t_d)}{1 - 2 t_p / t_d}, \qquad a = \frac{2 \tau}{t_d}, \qquad S = \frac{1}{1 - a + (1 + a)e^{-t_d / \tau}}
$$

$$
IA(t) = D \cdot \frac{S}{\tau^2} \cdot t \cdot \left(1 - \frac{t}{t_d}\right) e^{-t/\tau}
$$

$$
IOB(t) = D \cdot \left[1 - S(1-a)\left(\left(\frac{t^2}{\tau t_d (1-a)} - \frac{t}{\tau} - 1\right)e^{-t/\tau} + 1\right)\right]
$$

How it could fail: a real insulin curve that peaks later or lasts longer than 5 hours shifts every meal's insulin accounting the same way.

## Effective I:C from meals

The question: do meals in a pump segment behave like the I:C that segment holds?

**Meals.** The first bolus of 1.5 U or more in each of the morning (06:00-11:00), afternoon (11:00-17:00) and evening (17:00-24:00) windows of a day is a candidate. It is eligible with logged carbohydrate, no bolus in the previous 4 hours, insulin on board under 1 U, glucose 80 to 130 mg/dL before the meal without a steep slope, no exercise, no controller change just before, no unlogged food in the window and no context-event day. Each eligible meal is placed in the pump segment that covers its time of day.

**Per meal.** With $t$ in minutes after the bolus, the forward model ([`latent_meal_trajectory.py`](../src/bayesian_t1dm/latent_meal_trajectory.py)) is

$$
G(t) = G_0 + b_0 + b_1 t + \frac{\text{CF}}{\text{ICR}}\, C \, A(t - \delta;\, a, m) - \text{CF}\,(1 + d)\, U(t) + \varepsilon(t)
$$

$C$ is the carbohydrate-equivalent; $A$ is a gamma absorption curve (shape 1.5 to 7, mean 35 to 170 minutes) with a 0 to 20 minute delay $\delta$; $U(t)$ is the insulin absorbed since the bolus, including insulin already on board, later boluses and basal delivered above or below the scheduled rate (the schedule is the median delivery at that time of day over the surrounding 43 days); $b_0$, $b_1$ and $d$ are baseline offset, drift and correction-factor error, integrated out; $\varepsilon$ is AR(1) noise. The posterior over $(C, a, m, \delta)$ is evaluated on a grid with a deliberately vague prior on $C$ (SD at least 60 g).

**Pooling.** Each meal gives $\log r_i = \log(\hat{C}_i / S_i)$, the carbohydrate-equivalent over the logged grams, with $\text{sd} \approx \text{sd}(\hat{C}_i)/\hat{C}_i$. Within a pump segment and settings period (a run of meals with the same I:C and correction factor), meals are pooled by DerSimonian-Laird random effects with a Hartung-Knapp 90% interval, so meal-to-meal variability widens the result. The effective I:C is the programmed I:C divided by the pooled $r$; at least 5 meals are needed.

**Checked in simulation.** The test suite simulates meals with known carbohydrate under misspecified absorption, pre-bolus timing, drift, noisy CGM and logging error, and checks that the trajectory estimator beats bolus x I:C, that cohorts of 20 logged meals recover a true I:C of 8, 10 and 12.5 g/U to within 6% on average with 90% intervals covering the truth in at least 70% of cohorts, that a setting 20% off is flagged in at least 90% of cohorts, and that a correct setting is flagged in at most 25%.

How to read it: a pooled effective I:C of 8 g/U (90% interval 7 to 9) under a setting of 10 g/U says meals raised glucose as if one unit covered about 8 g, so the setting gives less insulin than meals ask for; an interval of 7 to 13 would say nothing about the setting. How it could fail: glucose identifies carbohydrate times CF / ICR, so habitually under-counted carbohydrate lowers the estimate exactly as a weak ratio would; a wrong correction factor shifts every estimate by about the same percentage; slow fat and protein absorption after the 3-hour window is invisible.

## Basal from the controller pattern

The question: does Control-IQ keep working against the programmed basal at the same hours ([`controller.py`](../src/bayesian_t1dm/controller.py))?

**Data.** The commanded rate every ~5 minutes for the whole record (it equals event 279's `commandedRate` where both exist) against the programmed rate of the segment in force from the profile history.

**Clean time.** Time is left out for 2, 3 or 4 hours after a meal by its size (carbohydrate of 15 g or more, or a bolus of 1 U or more; under 20 g or 2 U is 2 hours, under 40 g or 4 U is 3 hours), for exercise (the shared rule), unlogged-food intervals and context-event days. Small carbohydrate entries and the pump's own automatic corrections do not set a window; the corrections are the behaviour being measured. A day-segment counts with at least an hour of clean time.

**Signal.** For each segment, over the last 28 days (widening to 56, then 90, when clean time is short), each day's commanded insulin is compared with the segment's *current* programmed rate, weighted by clean hours. "Basal likely too low": the controller commanded 25% or more above the rate for at least half of the clean time, mean glucose over it is above 120 mg/dL, and time below 70 and 54 are within their limits; "too high" is the mirror case; under 10 days or 14 clean hours gives no signal. The pooled extra delivery and its 90% range come from resampling days (2,000 draws, fixed seed).

How to read it: "+43% (+14% to +71%)" means the controller delivered 1.43 times the programmed rate over the clean hours. How it could fail: the reference rate matters (the same commanded rates read differently against two programmed rates); sleep mode has a narrower treatment range than daytime; 28-day windows are noisy where clean time is thin, which is why the window widens and says so.

## Personal readout

`bayesian-t1dm readout` ([`readout_report.py`](../src/bayesian_t1dm/readout_report.py)) writes `runtime/output/readout/`: `readout.html` (one self-contained page drawn for a phone), `diagnostics.xlsx`, the driver effects and the settings check as CSV, and a run record `readout_run_<run id>.log`. When both models pass the sampler gate the page is copied to `<cloud root>/readout/readout.html`; a failed gate fails the run and nothing is copied.

**Day table** ([`readout.py`](../src/bayesian_t1dm/readout.py)). One row per calendar day: the consensus glucose metrics, bolus, delivered basal and carbohydrate from the pump, and the context before the day. Sleep is the night that ended on the morning of the day, one record per night from the first device in a fixed order (Oura, then Apple Watch). Resting heart rate and HRV are daily means, each compared with the median and robust spread (median absolute deviation x 1.4826) of the previous 28 days on the same device, with at least 10 such days, so a change of ring or watch is not read as a change in the person. Workout energy is 0 on a day a wearable recorded without a workout and unknown on a day nothing was recording. Site age is the time since the last cannula fill at midnight. Insulin and carbohydrate are compared with the median of the previous 28 days (unknown when that median is zero).

**Drivers.** Fixed before any fit: sleep under 6 hours the night before; low recovery the day before (resting heart rate more than one spread above, or HRV more than one spread below, its baseline); at least 200 kcal of workout active energy the day before, and two days before; 10 g or more of carbohydrate entered at or after 21:00 the evening before; an infusion site 3 or more days old at midnight (its fourth day or later). A day enters the model when it counts, is not a context-event day, has site changes observed and has every driver measured. A driver needs 15 modelled days with it and 15 without; with none estimable the run stops with a message.

**Model** ([`readout_model.py`](../src/bayesian_t1dm/readout_model.py)). For time in range and, separately, time below 70:

$$
y_d \sim \text{BetaBinomial}(n_d,\ \mu_d \phi,\ (1 - \mu_d)\phi), \qquad \text{logit}(\mu_d) = \alpha + \text{month}_d + \gamma\,\text{weekend}_d + \sum_k \beta_k x_{dk}
$$

The month effect is partially pooled (half-normal scale, prior 0.5), so a driver is compared with days of the same month and slow change in settings, season and habits does not load onto it. Each $\beta_k$ has a Normal(0, 0.5) prior on the logit scale. The reported effect is the average marginal effect: over the modelled days, the mean difference in the predicted share of the day with the driver set to yes and to no, in percentage points, with a 90% posterior interval. Sampling: 4 chains of 1,000 draws after 1,000 tuning steps; the sampler gate (R-hat under 1.01, bulk and tail effective sample size over 400, no divergences) applies to both fits.

**Settings check** ([`readout_settings.py`](../src/bayesian_t1dm/readout_settings.py)). For each segment of the profile running now: basal from the controller signal and I:C from the effective I:C, each compared with the value the segment holds now. Statuses: fits; needs more insulin; needs less insulin; held (the evidence points to more insulin but lows over the last 90 days are over a limit); not enough data. A suggestion is one step of 10% or 20%, the step nearer the gap (more insulin: basal up, I:C down). Setting changes of the last 90 days on the running profile are listed with a 14-day before/after comparison of time in range over the segment's hours ([`changes.py`](../src/bayesian_t1dm/changes.py)), with a day-resampling 90% interval, once 14 days have passed.

**Page** ([`readout_page.py`](../src/bayesian_t1dm/readout_page.py)). Rings for the latest full day (time in range, colored by whether it meets 70%; insulin against the 28-day median; recovery the day before, 100 x the normal percentile of (HRV z minus resting-heart-rate z) / sqrt(2), 50 being a usual day), the context of that day with how days like it have gone, this week's pooled time in range against the 10th to 90th percentile of 7-day stretches in the 90 days before, the driver effects, the settings check, recent changes, and the last seven days as glucose traces with sleep, workouts, carbohydrate entries and site changes marked. Colors per mark are defined once in [`figures.py`](../src/bayesian_t1dm/figures.py).

**Workbook.** `Specification_Coding`, `Denominators_Flow`, `Driver_Counts`, `Driver_Effects`, `Sampler_Health`, `Settings_Check`, `Basal_Signal`, `ICR_By_Segment`, `Recent_Changes`, `Record_Issues` (each left-out day and why; reconciled with the flow), `Warnings_Limitations`, `Artifacts_Provenance`. The workbook is read back and its effects and flow checked against the fit before the page is copied.

How to read it: a driver at -3.0 points (90% interval -5.5 to -0.5) says days after it had about 3 percentage points less of the day in range (roughly 45 minutes) than days of the same month without it; -1.0 (-4.0 to +2.0) is no clear link. How it could fail: a driver can travel with something unmeasured (an illness behind both a high resting heart rate and high glucose); a rare driver has a wide interval however large its effect; thresholds were chosen once and another cut-off could tell a different story; devices measure sleep and HRV differently, which per-device baselines handle for recovery but not for the absolute sleep threshold.

## Assumptions and failure modes

- Tandem Source is complete for the windows downloaded; an incomplete window is named in the manifest.
- Normalized timestamps are local wall-clock time; a wrong `TIMEZONE_NAME` shifts pump and Apple Health data against each other by hours.
- A 5-hour insulin action curve is a first-order approximation.
- Control-IQ changes basal all the time, so the I:C and basal evidence and every driver effect are observational.
- Apple Health exports are only as complete as the wearable was worn and the export was run; the readout treats days with nothing recording as unknown, not as zero.
