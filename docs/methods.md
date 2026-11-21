# Methods

The goal, the data processing and the statistical methods in one document. [`technical.md`](technical.md) gives each step's rules, parameters, a plain "how to read it" and how it could fail. Run numbers are not stated here: they live in the generated readout and its diagnostics workbook, which trace to a run.

## Contents

- [Goals](#goals)
- [Data processing](#data-processing)
- [Statistical methods](#statistical-methods)
- [Claims and their limits](#claims-and-their-limits)

## Goals

The aim is a personal readout that shows how the last day and week went, what has gone with glucose (sleep, recovery, exercise, late meals, infusion-site age), and whether the pump's basal and I:C settings are performing, with time in range 70-180 mg/dL as the main outcome and time below 70 under 4% and below 54 under 1% as limits. The readout informs a conversation with the endocrinologist; the person enters any change on the pump.

| Goal | Question | Subgoals | Command | Output |
| --- | --- | --- | --- | --- |
| A | Is the record complete and correctly merged? | A1 download and acquisition manifest; A2 normalization and timestamps; A3 merged dataset and validation report | `collect`, `backfill`, `fetch-profile-edits`, `normalize-raw`, `build-dataset` | `<cloud root>/data/raw/`, `<cloud root>/data/merged/` |
| B | How did the last day and week go? | B1 consensus CGM metrics for the day; B2 this week against the usual range | `readout` | readout page: rings and This week |
| C | What has gone with glucose? | C1 day table and prespecified drivers; C2 drivers model of next-day time in range and time below 70 | `readout` | readout page: What has gone with your glucose; workbook |
| D | Are the settings performing? | D1 basal from the controller pattern; D2 effective I:C from meals; D3 recent setting changes before and after | `readout` | readout page: Settings check and Recent setting changes; workbook |

`refresh` runs collect, build-dataset and readout in order.

## Data processing

Sources are Tandem Source (the pump's own event log: CGM, boluses, carbohydrate entries, basal delivery, profile edits, Control-IQ modes, site changes), Apple Health through Health Auto Export (heart rate, resting heart rate, HRV, sleep, workouts, steps, logged food), `context_events.csv` (days to leave out) and an optional hand-kept carb log. Details are in [Download from Tandem Source](technical.md#download-from-tandem-source), [Normalization and timestamps](technical.md#normalization-and-timestamps) and [Merged dataset](technical.md#merged-dataset).

- **Clock.** All times are local pump time (`TIMEZONE_NAME`); glucose, insulin and carbohydrate sit on a 5-minute grid.
- **Glucose validity.** Tandem CGM readings from 40 to 400 mg/dL count; a day counts when at least 70% of its readings are present.
- **Settings in force.** The I:C, correction factor, basal rate and target at each moment are reconstructed by replaying the profile edits the pump logged and checked against the bolus calculator's own settings; a moment the record cannot show is unknown, never filled. See [Settings in force](technical.md#settings-in-force).
- **Shared exclusions.** Exercise windows, context-event days and unlogged-food intervals are defined once and used by every analysis. See [Shared definitions](technical.md#shared-definitions).
- **Day table.** One row per day: glucose outcomes, insulin, carbohydrate, and the context before the day, with heart-rate measures compared with the person's own 28-day baseline on the same device. See [Personal readout](technical.md#personal-readout).

## Statistical methods

| Goal | Method | Estimand and uncertainty | Where specified |
| --- | --- | --- | --- |
| A | Fingerprinted rebuild, deduplication, completeness and plausibility checks | Counts and spans in the validation report; no inference | [Merged dataset](technical.md#merged-dataset) |
| B | Consensus CGM metrics with the day-coverage rule; the week against the 10th to 90th percentile of earlier 7-day stretches | Descriptive percentages | [Outcome metrics](technical.md#outcome-metrics) |
| C | Beta-binomial regression of each day's readings in range (and below 70) on the drivers measured before the day, with a partially pooled month effect and a weekend term | Average marginal effect of each driver in percentage points of the day, 90% posterior interval; sampler gate | [Personal readout](technical.md#personal-readout) |
| D1 | Controller's commanded rate against the current programmed rate over clean time, pooled over days | Percent above or below the programmed rate with a day-resampling 90% interval | [Basal from the controller pattern](technical.md#basal-from-the-controller-pattern) |
| D2 | Per-meal grid posterior of the carbohydrate-equivalent from the glucose trace; ratio to logged carbohydrate pooled per segment by random effects with a Hartung-Knapp interval | Effective I:C in g/U with a 90% interval | [Effective I:C from meals](technical.md#effective-ic-from-meals) |
| D3 | 14 days before against 14 days after a change, over the segment's hours | Difference in mean daily time in range with a day-resampling 90% interval | [Personal readout](technical.md#personal-readout) |

A driver effect is reported as "went with more (or less) time in range" when its 90% interval excludes zero and as "no clear link" otherwise. A setting is shown as needing more or less insulin when the 90% interval of its evidence lies wholly on one side of the value set now; a suggestion is one step of 10% or 20%, and nothing that adds insulin is suggested where lows are over a limit.

## Claims and their limits

The record is retrospective and from one person, and Control-IQ adjusts insulin all the time. The readout makes descriptive and associational statements: a driver went with a difference in time in range on days like it, and a setting's evidence points one way. These are not statements that changing a habit or a setting would cause the difference.
