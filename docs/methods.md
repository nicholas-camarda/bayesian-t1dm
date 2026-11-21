# Methods

The goals, the data processing and the statistical methods of the project in one document. [`technical.md`](technical.md) is the technical reference (formats, rules, parameters, validation); each goal below links to its section there, where the plain-language "how to read it" paragraph with a generic worked example is also given. Run numbers are not stated in tracked documents: they live in the generated reports and the front page, which trace to a run.

## Contents

- [Goals](#goals)
- [Data processing](#data-processing)
- [Statistical methods](#statistical-methods)
- [Claims and their limits](#claims-and-their-limits)

## Goals

The aim is more time in 70-180 mg/dL with time below 70 under 4% and below 54 under 1% (the international consensus targets). A proposal changes one setting by at most 10% and is held 14 days before the next. The system proposes and measures; the person enters any change on the pump.

| Goal | Question | Subgoals | Command or module | Output folder in a run |
| --- | --- | --- | --- | --- |
| A | Where do glucose outcomes stand, and in which hours? | A1 consensus metrics overall; A2 by pump segment; A3 baseline 14 and 90 days | `report-outcomes` | `A_outcomes/` |
| B | Which breakfast I:C change does the evidence support? | B1 pooled estimate per pump setting; B2 agreement checks; B3 proposal and safety rule | `propose-breakfast-icr` | `B_icr/` |
| C | Did an entered change help? | C1 one-command refresh; C2 change record; C3 before/after evaluation | `refresh` | `C_changes/` |
| D | Can logged food sharpen the carb estimate? | D1 unlogged-meal flags; D2 fat and protein (tested, not adopted); D3 Bevel as a second measurement | `propose-breakfast-icr` (flags and sensitivities) | `B_icr/supporting/` |
| E | Lunch and dinner I:C | E1 per-segment evidence; E2 stricter eligibility; E3 proposals | `propose-breakfast-icr --meal-segments` | `B_icr/` |
| F | Basal from the controller pattern | F1 commanded vs programmed rate; F2 lows as a hard stop; F3 proposals | `propose-breakfast-icr` | `F_basal/` |
| G | Correction factor | G1 correction-only boluses | planned; the calculator's factor is read for goals B and E | none |
| H | What happened in each long high or low, and how to respond | H1 episode finder; H2 pattern and evidence; H3 repeats by pump segment; H4 playbook | `propose-breakfast-icr` | `H_episodes/` |
| I | Does the hour-day model forecast 14-day time in range better than the last 28 days? | I1 prespecified win test; I2 exploratory comparators | `python -m bayesian_t1dm.forecast_harness` | `I_forecast/` |
| J | Do the model's predicted effects of past setting changes match what happened? | J1 replay as predictor (slope, power, settings information) | `python -m bayesian_t1dm.replay_predictor` | `J_replay_predictor/` |
| K | How did the last day and week go, what has gone with glucose, and are the settings performing? | K1 daily rings and context; K2 drivers of next-day time in range and time below 70; K3 settings check (basal, I:C); K4 recent changes | `python -m bayesian_t1dm.readout_report` | `runtime/output/readout/`, page copied to the Vault `readout/` |

The shared record checks sit in `00_General/` (the data-quality audit, `python -m bayesian_t1dm.data_audit`, and the dataset validation report). The exploratory replay of past changes (`python -m bayesian_t1dm.replay`) and the hour-day outcomes model (`python -m bayesian_t1dm.outcomes_model`) are the building blocks of goals I and J.

## Data processing

Sources are Tandem Source (the pump's own event log: CGM, boluses, carbohydrate entries, basal delivery, profile edits, Control-IQ modes, site changes), Apple Health (workouts, sleep, resting heart rate, heart rate variability, Bevel's food log), `context_events.csv` (days to leave out) and an optional hand-kept carb log. Every raw file is fingerprinted and `build-dataset` rebuilds only what changed. Details are in [Design overview](technical.md#design-overview), [Data Flow](technical.md#data-flow) and [Tandem Source Acquisition](technical.md#tandem-source-acquisition).

- **Clock.** All times are local pump time (`TIMEZONE_NAME`); glucose, insulin and carbohydrate sit on a 5-minute grid.
- **Glucose validity.** Tandem CGM readings from 40 to 400 mg/dL count; a day counts when at least 70% of its readings are present.
- **Settings in force.** The pump's I:C, correction factor, basal rate and target at each moment are reconstructed by replaying the profile edits the pump logged, and checked against the settings the bolus calculator recorded and against the basal rate the pump logged. A moment the record cannot show is "unknown", with a stated reason; it is never filled. See [Hour-day table and settings in force](technical.md#hour-day-table-and-settings-in-force).
- **Hour-day table.** One row per clock hour: counts of valid readings in range, below 70 and below 54, the settings in force, and exposures (carbohydrate in the prior 3 hours, exercise, sleep mode, cannula age). Covariates for a forecast must be known at the forecast origin; settings and calendar are, same-hour exposures are not (they are resampled from the 28 days before the origin).
- **Shared exclusions.** Exercise windows, context-event days and unlogged-food intervals are defined once in `context_rules` and used by every analysis (see "One loader, one set of definitions" in [Design overview](technical.md#design-overview)).

## Statistical methods

Each analysis follows the same shape: describe what the record shows, estimate with its uncertainty, apply a written rule, judge the result afterwards. The table gives the method and where its estimand, rule and "how to read it" are written.

| Goal | Method | Estimand and uncertainty | Where specified, with how to read it |
| --- | --- | --- | --- |
| A | Consensus CGM metrics with the day-coverage rule | Descriptive percentages by pump segment; no inference | [Outcome metrics](technical.md#outcome-metrics-goal-a) |
| B, E | Per-meal grid posterior over carbohydrate and absorption shape from the glucose trace; sensitivity ratio (carbohydrate-equivalent over logged carbohydrate) pooled per pump segment and setting by random effects with Hartung-Knapp variance | Effective I:C in g/U with a 90% interval; a proposal needs the programmed value outside the interval, every check on the same side and the low limits held | [Breakfast I:C proposals](technical.md#breakfast-ic-proposals-goal-b), [Lunch and dinner](technical.md#lunch-and-dinner-goal-e) |
| C | Before/after of 14 days each side over the segment's hours plus 4 | Difference in mean daily time in range (percentage points) with a day-level bootstrap 90% interval | [Refresh, change record and evaluation](technical.md#refresh-change-record-and-evaluation-goal-c) |
| D | Rule-based flags comparing the pump, Bevel and the CGM | Counts of flagged windows, which are excluded and counted | [Food the pump did not log](technical.md#food-the-pump-did-not-log-goal-d1) |
| F | Controller's commanded rate minus programmed rate over clean time, pooled over days | Percent above or below the programmed rate (U/h) with a day-resampling interval | [Basal from the controller pattern](technical.md#basal-from-the-controller-pattern-goal-f) |
| H | Ordered rules over insulin, carbohydrate, site changes and workouts | Wilson 90% intervals for rates | [Problem episodes and responses](technical.md#problem-episodes-and-responses-goal-h) |
| Audit | Pure checks on the loaded record with a status per check (no concern, concern, not assessed, not applicable) | Coverage, CGM quality, clock, settings reconstruction, meals, missingness, leakage | [Data-quality audit](technical.md#data-quality-audit-step-a) |
| Model | Beta-binomial hour-day model of the share of readings in an outcome, settings and exposures as covariates, random clock-hour and day effects | Posterior of each covariate effect in percentage points; sampler gate on R-hat, effective sample size and divergences | [Hour-day outcomes model](technical.md#hour-day-outcomes-model) |
| Replay | Replay of each past settings change against five matched no-change dates | `delta`, observed minus predicted minus placebo, in percentage points with a bootstrap 90% interval | [Replay of past setting changes](technical.md#replay-of-past-setting-changes-exploratory) |
| I | Weekly origins over the last year; CRPS of the 14-day aggregate share against the last-28-day baseline; paired block bootstrap | Win only if the 90% interval of the mean CRPS difference (model minus baseline) lies entirely below zero | [Forecast evaluation harness](technical.md#forecast-evaluation-harness-step-2) |
| J | The model fitted before each change predicts the change under new and old settings; slope of observed on predicted change, settings-information CRPS difference, power simulation | Slope in percentage points per percentage point, cluster bootstrap 90% interval | [Replay as predictor](technical.md#replay-as-predictor-step-4) |
| K | Beta-binomial regression of each day's readings in range (and below 70) on the prespecified drivers measured before the day, with a partially pooled month effect and a weekend term | Average marginal effect of each driver in percentage points of the day with a 90% posterior interval; within-person association | [Personal readout](technical.md#personal-readout-goal-k) |

The win rule of goal I is fixed before any fit and the verdict is one of win, tie or lose: a lower mean CRPS whose interval spans zero is a tie. A tie against a comparator means the data do not separate the two; it does not show the model carries no information.

## Claims and their limits

Retrospective records from one person support descriptive, associational and predictive statements. They do not support causal statements about a setting change, onset timing, optimal surveillance intervals or clinical utility. The forecast verdict is a statement about prediction of time in range; whether the settings add information beyond the long-run level is a separate question, which goal J tests with the changes that the record contains. Ranked candidate changes are produced only after the settings effect is validated; until then no ranking is offered.
