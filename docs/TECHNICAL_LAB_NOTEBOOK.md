# Technical Lab Notebook: Bayesian T1DM

**Updated**: November 17, 2026
**Purpose**: Scientific status, open problems and roadmap. Commands are in the [README](../README.md); implementation detail is in [technical.md](technical.md). Real-data numbers live in the generated reports, which trace to a run, not in this file.

## Contents

- [Summary](#summary)
- [What the data contain](#what-the-data-contain)
- [What has been verified](#what-has-been-verified)
- [Problems found and fixed](#problems-found-and-fixed)
- [Structural limits that no code change removes](#structural-limits-that-no-code-change-removes)
- [Lane status](#lane-status)
- [Scientific frame and priority cohort](#scientific-frame-and-priority-cohort)
- [Roadmap](#roadmap)
- [Decision rules for new work](#decision-rules-for-new-work)

## Summary

The repository is a personal diabetes research workbench, not a therapy-recommendation product. Its main question is whether the morning insulin-to-carb (I:C) ratio fits how glucose actually responds to logged breakfasts.

All real-data results produced before November 17, 2026 are void. Until then the Tandem data were stored in UTC because `TIMEZONE_NAME` was unset and the code fell back to UTC silently, while Apple Health was in local time. Time-of-day segments covered the wrong hours, Apple glucose filled CGM gaps at the wrong times and pump settings were looked up at the wrong hour. `TIMEZONE_NAME` is now required.

## What the data contain

- **Pump-entered carbs exist.** Carbs typed into the bolus calculator are on the pump's `LID_BOLUS_REQUESTED_MSG1` event, together with the carb ratio the calculator used; `LID_BOLUS_REQUESTED_MSG2` holds the correction factor and target. Each breakfast is judged against the setting in force for it, and the effective I:C is reported per settings period, with the current period as the headline. These carbs are the user's estimates, not weighed truth.
- **Logged food from Apple Health** (mostly Bevel; entries that carry protein and fat with the carbs) is matched to breakfasts within 30 minutes. It is supplementary only: the headline stays on pump entries, and two labelled sensitivity analyses (without high fat or protein breakfasts; pump and logged-food carbs combined after rescaling) show whether conclusions move. Whether an entry came from a barcode scan or a photo estimate is not in the export.
- **CGM** comes only from sensor-reading events, timed by the sensor's measurement clock (`egvTimeStamp`). Readings backfilled after a signal dropout used to pile into one 5-minute slot.
- **Excluded periods** are listed in `data/raw/context_events.csv` (for example the 14 days after the October 22, 2026 C-spine steroid injection). Breakfasts and overnight periods inside them are a hard exclusion in every analysis.
- Counts, spans and checks for the current dataset are in `data/merged/validation_report.md`.

## What has been verified

These results come from tests and simulation, not from the personal data.

- The test suite runs without clinical data in CI on Python 3.11 and 3.13.
- The trajectory-based latent carb estimator recovers simulated carbohydrate. Against meals bolused for a mis-counted amount, bolus x I:C is off by about 17 g on average; the trajectory estimator is off by 2.5 to 9 g in seven of eight scenarios, with 90% intervals that cover truth at least 90% of the time. Reproduce with `bayesian-t1dm validate-latent-meal`.
- With 20 logged meals, the effective I:C estimator recovers a true ratio of 8 or 12.5 against a programmed 10 to within about 3%.
- The full command chain runs end to end on a simulated raw tree laid out like the Tandem archive (an automated test).
- Data-pipeline rewrites for speed were checked to produce identical Tandem tables, Apple Health tables and basal features on the real data.

## Problems found and fixed

Each of these changed results; anything computed before the fix should be regenerated, not reused.

1. **Timezone.** Tandem times were stored in UTC (see Summary).
2. **Look-ahead leak in forecast features.** CGM gaps were filled by linear interpolation from the reading at the far end of the gap. The fill is now last-observation-carried-forward.
3. **Original latent carb estimator was miscalibrated** (about +12 g bias, 55 to 72% interval coverage). It is now only a fallback.
4. **"Posterior probability better" was not a probability.** It is now the share of paired posterior draws in which a scenario wins.
5. **Meal-window CGM coverage was overstated**; it is now measured against the rows a complete window should have.
6. **Pump-entered carbs were thrown away**; they are now read from `LID_BOLUS_REQUESTED_MSG1` and kept only for delivered boluses.
7. **Tandem Source retired the tconnectsync 2.x endpoints** in mid-2026 (HTTP 403). Collection uses 3.x; windows collected with 2.x keep their event files, which 3.x still decodes.
8. **The forecast sampler never converged** (R-hat about 2.9). The forecast model is now a regression of glucose 30 minutes ahead on current features with Student-t errors.
9. **CGM readings took the pump's receipt time**, so backfilled readings collided. They now use sensor time, kept as pump wall-clock time so a daylight-saving change between measurement and receipt cannot move them an hour.
10. **Pump settings were dated by the window they were archived with.** Every window collected on the same day holds the same current snapshot, so ordering is now by the snapshot's upload time.
11. **Unreadable timestamps were dropped silently**; they now stop the build.

## Structural limits that no code change removes

- **Carbs and I:C are not separately identifiable from glucose.** A post-meal trace identifies `carbs x CF / ICR`. With pump-entered carbs, "my counts run low" and "my I:C is too weak" produce the same trace, so the effective I:C absorbs any systematic carb-counting bias. That is useful clinically (it is the ratio that works with how you count), but it is not the physiological ratio. Weighed or packaged meals break the tie.
- **Late absorption is invisible.** Carbohydrate that lands after the 180-minute window biases estimates low (about -18 g in a biphasic simulation). This is why high fat or protein breakfasts get a sensitivity analysis.
- **Closed-loop basal is only approximately accounted for.** Control-IQ changes basal during the post-meal window. The estimator counts delivery above or below a schedule inferred from the median delivery at that time of day, which is right when modulation varies from day to day and wrong when the controller does the same thing most mornings. The residual error on real data is unknown.
- **Retrospective personal data** support associations, not causal claims about a setting change.

## Lane status

| lane | status |
| --- | --- |
| Acquisition and normalization (Tandem, Apple Health) | Runs on real data. A full rebuild takes under a minute and an unchanged check about a second |
| Merged 5-minute dataset | Runs on real data; validated in `validation_report.md` |
| Latent meal: breakfast effective I:C | Primary lane. Runs on real data with pump-entered carbs; validated in simulation. Real-data result is in `output/latent_meal/latent_meal_review.html` |
| Therapy research (overnight basal identifiability) | Runs on real data. Its synthetic "truth recovery" uses frames without meal or insulin dynamics, so it is weaker evidence than the name suggests |
| Forecast validation | Runs on real data. Kept only as a check on model quality; the answer is on `output/forecast/forecast_review.html` |
| Forecast-based recommendations | Always suppressed. The scenario math is a sensitivity analysis of a predictive regression, not a causal estimate |

## Scientific frame and priority cohort

Observed glucose is a response to meals, insulin, basal and closed-loop behaviour, exercise, sleep, stress and measurement gaps. The project does not try to read true carbs off the data. It estimates meal response under uncertainty and suppresses results where a window is too confounded.

The priority cohort is the **first meal of the day** with a long fast beforehand, premeal glucose near target, low insulin on board, no recent correction or workout, and enough CGM through the post-meal horizon. It is the least confounded setting in the data and can later calibrate messier meals.

## Roadmap

### Now: land the real-data pipeline

1. Open a pull request for the `real-data-pipeline` branch so CI runs, then merge. Nothing on the branch has been pushed yet.
2. Read the regenerated pages: they are the first valid real-data results. In particular, check whether the current settings period has enough breakfasts for a headline (the page says "too early to judge" below 5).

### Next: make the I:C answer trustworthy

3. **Weighed or packaged breakfasts.** Log about 20 breakfasts with weighed or packaged carbs over 2 to 3 weeks. This is the only way to separate carb-counting bias from the ratio, and it also calibrates Bevel (how far its estimates are from known carbs). Needs a way to mark those meals as weighed, for example a column in `carbs.csv`.
4. **Keep `context_events.csv` current** (illness, steroids, travel, sensor problems). Excluded periods are the main defence against confounded mornings.
5. **Travel and time zones.** Storage uses one `TIMEZONE_NAME`. Days spent in another time zone are stored on home time (Tandem) or phone time (Apple Health), which can disagree. Until that is handled, list travel days in `context_events.csv`.
6. **Decide whether CGM gaps should block anything.** Any multi-year record has 2-hour gaps and a few values outside 40 to 400 mg/dL; the plausibility check currently reports these as "failed", which permanently blocks forecast recommendations. Gaps are already handled per meal window.
7. **Measure the Control-IQ residual** on real mornings: how often the controller does the same thing most mornings, which is when the basal adjustment is wrong.

### Later

8. Overnight basal evidence from the therapy lane, once the I:C lane is settled.
9. Decide whether the forecast lane earns its keep. If the model does not beat "glucose stays the same" by the 10% bar on current data, retire it.
10. Phase E: test whether the clean first-meal estimates transfer to less clean meals, with wider uncertainty.
11. Housekeeping: `tandem_export_manifest.csv` is written by `collect`, `backfill`, `normalize-raw` and `build-dataset`. It should have one writer, `build-dataset`, and `normalize-raw` may then be unnecessary.

### Explicitly deferred

- global basal optimization as the headline
- correction-factor inference as a primary target
- therapy recommendations across all meal contexts
- claiming exact carb truth for any single bolus

## Decision rules for new work

Ask first: **does this help the clean first-meal I:C program, or is it a side quest?**

- Work that improves source truth, missingness semantics or meal-window quality is high priority.
- Work that improves identifiability of the morning I:C is high priority.
- Work that only makes dashboards prettier is low priority.
- Work that assumes recommendations are already trustworthy is premature.

Update this notebook when the active question, the main data-truth conclusion, the primary lane or the deferred list changes.
