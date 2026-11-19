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

- **Pump-entered carbs exist.** Carbs typed into the bolus calculator are on the pump's `LID_BOLUS_REQUESTED_MSG1` event. The t:slim also records the carb ratio there and the correction factor and target on `LID_BOLUS_REQUESTED_MSG2`. The Mobi, bolusing from the phone app, records all three as 0; they are recovered from how its calculator split the bolus (method in [technical.md](technical.md#bolus-calculator-settings)). Each breakfast is judged against the setting in force for it, and the effective I:C is reported per settings period, with the current period as the headline. These carbs are the user's estimates, not weighed truth.
- **The pump logs its profile history.** Every profile edit (event 68, one segment's start, basal, CF, target and I:C) and every profile switch (event 69) is in the archived raw events, decoded by `pump_profiles.py` (see [technical.md](technical.md#bolus-calculator-settings)). Edits come from the binary archive (t:slim January 2024 to September 2025; Mobi August 2025 to April 22, 2026) and, after that, from event 68 requested by name from Tandem Source (the default request omits it, which led to an earlier wrong note here that edits were invisible after April); `fetch-profile-edits` adds them to archived windows. The history shows frequent switching between Main, Low and Medium on the Mobi in August to September 2025, Main from mid-September 2025 to June 2026, Medium (8.0, edited to 7.5 g/U) from June 2026 with a Main interlude on July 26 to 29, and Main again from November 16, 2026.
- **Logged food from Apple Health** (mostly Bevel; entries that carry protein and fat with the carbs) is matched to breakfasts within 30 minutes. It is supplementary only: the headline stays on pump entries, and two labelled sensitivity analyses (without high fat or protein breakfasts; pump and logged-food carbs combined after rescaling) show whether conclusions move. Whether an entry came from a barcode scan or a photo estimate is not in the export.
- **CGM** comes only from sensor-reading events, timed by the sensor's measurement clock (`egvTimeStamp`). Readings backfilled after a signal dropout used to pile into one 5-minute slot.
- **Excluded periods** are listed in `data/raw/context_events.csv` (for example the 14 days after the October 22, 2026 C-spine steroid injection). Breakfasts and overnight periods inside them are a hard exclusion in every analysis.
- Counts, spans and checks for the current dataset are in `data/merged/validation_report.md`.

## What has been verified

These results come from tests and simulation, not from the personal data.

- The test suite runs without clinical data in CI on Python 3.11, the version in `environment.yml`.
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
12. **Every Mobi breakfast was compared with today's profile.** The Mobi records its calculator settings as 0, so all breakfasts since August 2025 were compared with the snapshot uploaded on the day of the pull (Main, 6.5), while they were dosed at 7.0 and then 7.5 g/U. Settings are now recovered per breakfast, and a breakfast without a usable record takes the setting of the nearest breakfast on the same pump instead of today's profile.
14. **Review of 94fecbd..HEAD (Opus 5.5, high), fixed:** the profile replay read modification 1 as an insert (it is a delete) and ignored copies and deletes of whole profiles (93 snapshots had duplicate start times; none now, and agreement with the calculator is 99.9%); the carry grouped non-adjacent meals, so a Main to Medium to Main sequence carried Main's values across the switch (now stretches of adjacent meals); per-meal carb estimates used the latest profile's I:C (6.5 for all 693 breakfasts, now each meal's own 5.0 to 8.0); removed raw files and timezone changes did not trigger a rebuild; the fingerprint missed tables the ingest reads; a CGM value of 0 or below, or above 1000, passed plausibility; commands that clear an output folder did so before the analysis. **Open, decided in phase 2:** a settings period breaks whenever a meal's own I:C or CF differs, so 149 of 173 periods have fewer than 5 breakfasts and the current-setting row has one; profile lookup uses carb times rounded down to 5 minutes (9 of about 3,680 entries cross an edit); the largest I:C still sets the plausible-carb ceiling from the latest snapshot.
13. **Carried settings crossed profile switches.** A breakfast without a usable record took the setting of the nearest breakfast on the same pump, even across a switch to another profile. Checked against the pump's profile history, 36 of 115 carried Mobi correction factors were wrong, and the "7.0 g/U, CF 55, September to October 2025" period was an artifact of one carried value. The profile history now supplies settings where it shows them, and carrying stays within one pump and profile and does not cross the end of visible edits.

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
| 5-minute glucose forecast and its scenario recommendations | Retired 2026-11-19 (tag `pre-forecast-lane-removal-20261119`); see Retired in `docs/technical.md` |

## Scientific frame and priority cohort

Observed glucose is a response to meals, insulin, basal and closed-loop behaviour, exercise, sleep, stress and measurement gaps. The project does not try to read true carbs off the data. It estimates meal response under uncertainty and suppresses results where a window is too confounded.

The priority cohort is the **first meal of the day** with a long fast beforehand, premeal glucose near target, low insulin on board, no recent correction or workout, and enough CGM through the post-meal horizon. It is the least confounded setting in the data and can later calibrate messier meals.

## Roadmap

Each phase is its own branch and pull request. A phase is done when: tests for its rules are added; its real-data run is reread against the source tables; its rendered page is checked; the docs are updated; CI on the pull request is green. The live status is kept in `PROGRESS.md` (local, not committed).

| Phase | Goal | Content |
| --- | --- | --- |
| 0 | land the branch | commit, review, fix, pull request |
| 1 | A outcome metrics | consensus metrics overall and by pump segment (`outcomes.py`, `report-outcomes`) |
| 2 | B breakfast I:C proposals | pool by (pump, I:C, CF, segment); tier, clean-direction, no-correction checks; proposal and safety rule (`proposals.py`) |
| 3 | C refresh, change log, evaluation | one `refresh` command; event 68 request; durable proposal record; 14-day before/after evaluation |
| 4 | D Bevel in the carb estimator | unlogged-meal flags, fat and protein absorption, Bevel as a second measurement |
| 5 | E lunch and dinner I:C | the breakfast machinery per segment, with stricter eligibility |
| 6 | F basal from the controller pattern | Control-IQ's commanded vs profile rate by time of day, lows as a hard stop |
| 7 | G correction factor | after phases 2 to 6 |

Done in phase 2: settings periods are formed per pump segment, which cut periods with fewer than 5 breakfasts from 149 of 173 to 22 of 44. Still open from the review: profile lookup uses carb times rounded down to 5 minutes; the plausible-carb ceiling comes from the latest snapshot.

Still open from earlier: keep `context_events.csv` current; travel and time zones; whether CGM gaps should block anything (the plausibility check no longer fails on readings at the sensor's limits, only on impossible values); `tandem_export_manifest.csv` should have one writer.

Explicitly deferred: global basal optimization as the headline; claiming exact carb truth for any single bolus; weighed breakfasts (not planned, so the effective I:C keeps absorbing carb-counting bias, which is the ratio that works with how the carbs are counted).

## Decision rules for new work

Ask first: **does this help the clean first-meal I:C program, or is it a side quest?**

- Work that improves source truth, missingness semantics or meal-window quality is high priority.
- Work that improves identifiability of the morning I:C is high priority.
- Work that only makes dashboards prettier is low priority.
- Work that assumes recommendations are already trustworthy is premature.

Update this notebook when the active question, the main data-truth conclusion, the primary lane or the deferred list changes.
