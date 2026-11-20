"""Replay as predictor: do the outcomes model's predicted effects of a setting change match what happened after the past changes?

Two questions, asked of every past change in the settings in force that the replay (``replay``) scored.

1. Tracking. The hour-day outcomes model is fitted to the days before the change. It then predicts the share of readings in the
   outcome (time in range, below 70, below 54) over the replay's affected hours on the 14 days after the change, once under the
   settings in force after the change ("new") and once with every setting held at its value before the change ("old"), with the
   same posterior draw, the same day effects and the same resampled exposures in both. The predicted change is new minus old, in
   percentage points, with a 90% credible interval. It is set beside the replay's observed change minus placebo change
   (``replay`` ``delta`` per event: the change's miss from the last-28-day predictor minus the mean miss on its five matched
   no-change dates). Reported: the correlation and the least-squares slope of observed on predicted (a slope of 1 means the
   predicted size is right, 0 means the predictions carry no information about what happened), the share of changes whose
   direction matches, and how often the intervals hold the observed change, each with a 90% cluster-bootstrap interval (clusters are
   the replay's clusters of changes within 14 days of each other).
2. Settings information. The CRPS of the 14-day aggregate share under the new settings is compared with the CRPS of the same fit with
   the settings held at their old values, paired across changes (new minus old, percentage points; negative means the settings in
   force predict what happened better), with a 90% cluster-bootstrap interval.

Inputs known at the change date only. The fit uses the rows dated before the change's first day (a change within 14 days after
an earlier one reuses the earlier change's fit, so a fit never sees a day on or after the change it serves); the old settings are the
profile in force at the last moment of the day before; exposures (carbohydrate, exercise, sleep mode, site age) are resampled from
the 28 days before. Reading counts per hour are the realised ones on the days the replay counted (the count of readings is not glucose
information). The new settings are the settings actually in force on those days: they are not known at the change date, they are the
change under test. Claims are predictive and associational: one person's record, retrospective, no randomisation.

``python -m bayesian_t1dm.replay_predictor`` runs it on the real record.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import forecast_harness as fh
from . import outcomes_model as om
from . import replay
from .hour_day import HOURS_PER_DAY
from .win_test import crps_points

LOGGER = logging.getLogger(__name__)

OUTCOMES = ("tir", "below70", "below54")
PRIMARY_OUTCOME = "tir"
MIN_TRAIN_DAYS = 90
FIT_SHARE_DAYS = 14
WINDOW_DAYS = fh.WINDOW_DAYS
assert WINDOW_DAYS == replay.AFTER_DAYS, "The model's window and the replay's after window must be the same 14 days"
DRAWS = fh.DRAWS
BOOTSTRAP_DRAWS = 10_000
SEED = om.SEED
INTERVAL = (5.0, 95.0)
MIN_CLUSTERS = 10
SETTING_UNITS = (("icr", "I:C", "g/U"), ("cf", "CF", "mg/dL per U"), ("basal", "basal", "U/h"), ("target", "target", "mg/dL"))
GROUPS = ("isolated changes", "all changes")
COVERAGE_RANGE = (0.80, 0.98)
SETTING_PRIOR_SD = 0.5  # the prior SD of every beta in outcomes_model.build_model (a test checks the model source still says so)
PRIOR_DOMINATED_RATIO = 0.8
POWER_SLOPES = (0.5, 1.0, 2.0, 4.0)
POWER_SIMULATIONS = 300
POWER_BOOTSTRAP_DRAWS = 1000


class NotPredicted(Exception):
    """A change cannot be predicted; the reason is listed in the record issues (the change is excluded, never filled in)."""


# ---------------------------------------------------------------------------
# Targets: what each change asks of the model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventTarget:
    """One replay change as the model sees it.

    ``counts`` (14 x 24) are the valid readings of the replay's counted days in the affected hours (zero elsewhere). ``old`` (24 x 4) is the settings profile in force at the last
    moment of the day before the change and ``new`` (14 x 24 x 4) the profile in force after it (the profile at the last moment of the change's last day, held for every day of the
    window; icr, cf, basal, target). ``baseline`` (days x 24 x 4) holds the settings actually in force on each of the replay's baseline days (the qualifying days of the 28 before
    the change) and ``baseline_weights`` their readings in the affected hours: the replay's last-28-day predictor is built from those days. ``pool`` is the days x 24 x 4 exposure pool
    known at the change date, ``moved`` marks the settings (icr, cf, basal, target) the change alters at its changed hours, and ``cut`` is the first day the fit may not see.
    """

    event_id: int
    event_date: pd.Timestamp
    window_start: pd.Timestamp
    cluster_id: int
    isolated: bool
    direction: str
    settings: str
    description: str
    counted_days: int
    training_days: int
    counts: np.ndarray
    old: np.ndarray
    new: np.ndarray
    baseline: np.ndarray
    baseline_weights: np.ndarray
    pool: np.ndarray
    moved: np.ndarray
    observed: dict[str, float]
    cut: pd.Timestamp


def describe_change(old: np.ndarray, new: np.ndarray, hours: np.ndarray) -> str:
    """The change in the units the pump uses, as the mean over the clock hours of ``hours`` where that setting differs (``old`` and ``new`` are known 24 x 4 profiles at ``hours``)."""
    if not (np.isfinite(old[hours]).all() and np.isfinite(new[hours]).all()):
        raise ValueError("A setting is unknown at an hour being described")
    parts = []
    for index, (_, label, unit) in enumerate(SETTING_UNITS):
        differs = hours & (np.abs(new[:, index] - old[:, index]) >= fh.NO_SETTINGS_CHANGE)
        if not differs.any():
            continue
        digits = {"basal": 3, "I:C": 2, "CF": 2}.get(label, 1)
        parts.append(f"{label} {old[differs, index].mean():.{digits}f} to {new[differs, index].mean():.{digits}f} {unit}")
    return "; ".join(parts) if parts else "no setting value differs"


def settings_grid(table: pd.DataFrame) -> np.ndarray:
    """The settings of every day and clock hour (days x 24 x 4: icr, cf, basal, target) of a complete hour-day table, NaN where the archive cannot show them at all twelve moments."""
    values = table[list(fh.SETTING_COLUMNS)].to_numpy(dtype=float).copy()
    values[(table["settings_known_slots"] != fh.SLOTS_PER_HOUR).to_numpy()] = np.nan
    return values.reshape(-1, HOURS_PER_DAY, 4)


def check_profiles_agree(table: pd.DataFrame, grid: replay.DayGrid, profiles: fh.SettingsProfiles) -> int:
    """The profile frozen from the pump's history at the start of each day must equal the hour-day table's settings on every day the replay can say had no change (no changed hour and no hour whose day-to-day difference is unknown, where an edit during the day is not visible to the replay); returns how many day-hours were compared.

    The change's old and new settings come from the frozen profiles and the baseline settings from the table. A day without a change where they disagree means the two readings of the
    pump's settings differ, which is an error and not a difference to be predicted.
    """
    table_settings = settings_grid(table)
    frozen = profiles.values
    dates = pd.DatetimeIndex(sorted(table["date"].unique()))
    if not frozen.shape[0] == table_settings.shape[0] == len(dates) or not (profiles.dates == dates).all():
        raise AssertionError("The frozen profiles and the hour-day table do not cover the same days")
    undetermined = table.groupby("date")[["d_icr", "d_cf", "d_basal", "d_target"]].agg(lambda column: column.isna().any()).any(axis=1).reindex(dates).to_numpy()
    quiet = ~grid.changed.any(axis=1) & ~undetermined
    quiet[0] = False
    both = np.isfinite(frozen) & np.isfinite(table_settings) & quiet[:, None, None]
    worst = np.abs(np.where(both, frozen - table_settings, 0.0))
    if (worst >= fh.NO_SETTINGS_CHANGE).any():
        days = int((worst >= fh.NO_SETTINGS_CHANGE).any(axis=(1, 2)).sum())
        raise AssertionError(f"The frozen profile and the hour-day table disagree on {days} days without a change (largest difference {worst.max():.4g})")
    return int(both.sum() // 4)


def plan_targets(table: pd.DataFrame, grid: replay.DayGrid, events: list[replay.Event], clusters: dict[int, int], isolated: set[int], scores: pd.DataFrame, replay_issues: pd.DataFrame,
                 profiles: fh.SettingsProfiles, daily: pd.DataFrame, tdd: pd.Series, *, min_train_days: int = MIN_TRAIN_DAYS) -> tuple[list[EventTarget], pd.DataFrame, pd.DataFrame]:
    """Every replay change that can be predicted (with the fit cut it will use), the flow of changes through the stages, and the record issues of those left out.

    A change is left out, with the reason, when: the replay did not score it; fewer than ``min_train_days`` days with usable hours precede it; a setting is unknown at an affected hour
    before the change, after it, or on a baseline day (the qualifying days of the 28 before it, whose settings the observed change is measured against); no complete day of exposure
    precedes it; or the setting values at its changed hours are equal before and after it (nothing to test). The frozen profiles and the hour-day table must agree on days without a
    change (an error otherwise).
    """
    check_profiles_agree(table, grid, profiles)
    by_day = settings_grid(table)
    scored = set(scores.loc[scores["kind"].eq("change") & scores["predictor"].eq(replay.PRIMARY), "event_id"])
    reasons = replay_issues.set_index("event_id") if len(replay_issues) else pd.DataFrame()
    usable_dates = np.sort(table.loc[om.usable_rows(table), "date"].unique())
    targets: list[EventTarget] = []
    issues: list[dict[str, object]] = []
    stage = {"changes in the record": len(events), "scored by the replay": 0, "with enough training days": 0, "with settings known and exposures available": 0, "with different setting values before and after": 0}
    for event in events:
        date = grid.dates[event.first]
        identity = {"event_id": event.event_id, "change_date": date.date(), "change_days": event.span}
        if event.event_id not in scored:
            why = reasons.loc[event.event_id] if event.event_id in reasons.index else None
            issues.append({**identity, "step": "scored by the replay", "reason": "not scored by the replay" + (f": {why['reason']}" if why is not None else ""),
                           "field": why["field"] if why is not None else "", "value": why["value"] if why is not None else ""})
            continue
        stage["scored by the replay"] += 1
        training_days = int(np.searchsorted(usable_dates, np.datetime64(date)))
        if training_days < min_train_days:
            issues.append({**identity, "step": "enough training days", "reason": f"fewer than {min_train_days} days with usable hours before the change", "field": "training days", "value": training_days})
            continue
        stage["with enough training days"] += 1
        start = event.first + event.span
        window_start = grid.dates[start]
        positions, _ = replay._window(grid, start, start + replay.AFTER_DAYS, event.affected_hours)
        counts = np.zeros((WINDOW_DAYS, HOURS_PER_DAY))
        counts[positions - start] = grid.counts["n_valid"][positions] * event.affected_hours[None, :]
        baseline_days, baseline_counts = replay._window(grid, event.first - replay.BEFORE_DAYS, event.first, event.affected_hours)
        baseline = by_day[baseline_days]
        known = fh.known_at(date, PRIMARY_OUTCOME, table, daily, tdd, profiles)
        old = known.profiles.at(date)
        after = profiles.at(window_start)
        affected = event.affected_hours
        try:
            if not np.isfinite(old[affected]).all():
                raise NotPredicted(("settings before the change are unknown at an affected hour", "hours unknown before", int((~np.isfinite(old[affected]).all(axis=1)).sum())))
            if not np.isfinite(after[affected]).all():
                raise NotPredicted(("settings after the change are unknown at an affected hour", "hours unknown after", int((~np.isfinite(after[affected]).all(axis=1)).sum())))
            if not np.isfinite(baseline[:, affected]).all():
                raise NotPredicted(("settings are unknown at an affected hour on a baseline day (the 28 days before the change)", "baseline day-hours unknown", int((~np.isfinite(baseline[:, affected]).all(axis=2)).sum())))
            try:
                pool = fh.exposure_pool(known)
            except fh.NotScored as problem:
                raise NotPredicted((str(problem), "complete exposure days", 0)) from problem
        except NotPredicted as problem:
            reason, field, value = problem.args[0]
            issues.append({**identity, "step": "settings known and exposures available", "reason": reason, "field": field, "value": value})
            continue
        stage["with settings known and exposures available"] += 1
        moved = (np.abs(after[event.changed_hours] - old[event.changed_hours]) >= fh.NO_SETTINGS_CHANGE).any(axis=0)
        if not moved.any():
            issues.append({**identity, "step": "different setting values", "reason": "the setting values at the changed hours are equal before and after the change, so the predicted change is exactly zero and there is no setting to test",
                           "field": "largest difference in a setting value at the changed hours", "value": 0.0})
            continue
        stage["with different setting values before and after"] += 1
        observed, n_days = replay.observe(grid, event.first, event.span, event.affected_hours)
        targets.append(EventTarget(event.event_id, date, window_start, clusters[event.event_id], event.event_id in isolated, event.direction, event.settings,
                                   describe_change(old, after, np.asarray(event.changed_hours) & (np.abs(after - old) >= fh.NO_SETTINGS_CHANGE).any(axis=1)), n_days, training_days, counts, old,
                                   np.broadcast_to(after, (WINDOW_DAYS, HOURS_PER_DAY, 4)).copy(), baseline, baseline_counts["n_valid"].astype(float), pool, moved, observed, date))
    flow = pd.DataFrame(list(stage.items()), columns=["stage", "changes"])
    return assign_cuts(targets), flow, pd.DataFrame(issues, columns=["event_id", "change_date", "change_days", "step", "reason", "field", "value"])


def assign_cuts(targets: list[EventTarget]) -> list[EventTarget]:
    """Give each change the fit cut it will use: a change within 14 days after the first change of its group reuses that change's fit (the fit then ends before both)."""
    out: list[EventTarget] = []
    head: pd.Timestamp | None = None
    for target in sorted(targets, key=lambda item: item.event_date):
        if head is None or (target.event_date - head).days > FIT_SHARE_DAYS:
            head = target.event_date
        out.append(replace(target, cut=head))
    return out


def check_no_leakage(table: pd.DataFrame, target: EventTarget) -> pd.DataFrame:
    """The rows a change's fit trains on: those dated before its cut, which is on or before the change date. Raises when any later row would reach the fit."""
    train = table[table["date"] < target.cut]
    if target.cut > target.event_date or train.empty or train["date"].max() >= target.event_date:
        raise AssertionError(f"Rows from the change date {target.event_date.date()} or later would reach its fit (cut {target.cut.date()})")
    return train


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_event(posterior: fh.ModelPosterior, target: EventTarget, seed: int) -> dict[str, np.ndarray]:
    """Expected share of the post-change window under three settings, and predictive draws under two, one posterior draw at a time.

    ``expected_new``: the profile in force after the change. ``expected_baseline``: the settings actually in force on each baseline day (the days the replay's last-28-day predictor pools),
    averaged with the readings of each day as weight; the observed change is measured against that baseline, so ``expected_new - expected_baseline`` is the quantity the replay observes.
    ``expected_old``: the profile in force the day before the change held for the window (sensitivity). All three share the posterior draw, day effects and resampled exposures.
    ``simulated_new`` and ``simulated_old`` are beta-binomial predictive draws of the share (new and old settings, same seed) for the CRPS comparison.
    """
    rng = np.random.default_rng(seed)
    draws = len(posterior.alpha)
    exposures = target.pool[rng.integers(0, len(target.pool), size=(draws, WINDOW_DAYS))]
    day_noise = rng.normal(size=(draws, WINDOW_DAYS))
    weights = target.counts.reshape(-1) / target.counts.sum()

    def expected(profile: np.ndarray) -> np.ndarray:
        return fh.window_mu(posterior, fh.settings_covariates(np.broadcast_to(profile, (WINDOW_DAYS, HOURS_PER_DAY, 4))), exposures, target.counts, day_noise) @ weights

    old = np.broadcast_to(target.old, (WINDOW_DAYS, HOURS_PER_DAY, 4))
    day_weights = target.baseline_weights / target.baseline_weights.sum()
    result: dict[str, np.ndarray] = {"expected_new": expected(target.new[0]), "expected_old": expected(target.old),
                                     "expected_baseline": sum(weight * expected(profile) for weight, profile in zip(day_weights, target.baseline))}
    for name, profile in (("new", target.new), ("old", old)):
        result[f"simulated_{name}"] = fh.simulate_window_share(posterior, fh.settings_covariates(profile), exposures, target.counts, np.random.default_rng([seed, 1]))
    return result


def moved_prior_ratio(posterior: fh.ModelPosterior, moved: np.ndarray) -> float:
    """Largest posterior SD over prior SD among the settings a change moves: near 1 means the fit has learnt nothing about that setting's coefficient and the prediction rests on the prior."""
    return float((posterior.beta[:, :4].std(axis=0) / SETTING_PRIOR_SD)[moved].max())


def event_rows(posteriors: dict[pd.Timestamp, fh.ModelPosterior], targets: list[EventTarget], outcome: str) -> pd.DataFrame:
    """One row per change for ``outcome``: predicted change (percentage points, mean and 90% credible interval) against the baseline days' settings, the same against the day-before profile,
    the observed window, and the CRPS under the new and the old (day-before) settings."""
    rows = []
    for target in targets:
        posterior = posteriors[target.cut]
        check_posterior_precedes(posterior, target)
        draws = predict_event(posterior, target, seed=int(np.random.default_rng([SEED, target.event_id, OUTCOMES.index(outcome)]).integers(2**31)))
        change = 100.0 * (draws["expected_new"] - draws["expected_baseline"])
        frozen = 100.0 * (draws["expected_new"] - draws["expected_old"])
        low, high = np.percentile(change, INTERVAL)
        observed = target.observed[outcome] / 100.0
        crps_new, crps_old = crps_points(draws["simulated_new"], observed), crps_points(draws["simulated_old"], observed)
        new_low, new_high = np.percentile(draws["simulated_new"], INTERVAL)
        ratio = moved_prior_ratio(posterior, target.moved)
        rows.append({
            "event_id": target.event_id, "change_date": target.event_date.date(), "cluster_id": target.cluster_id, "isolated": target.isolated, "direction": target.direction,
            "change": target.description, "outcome": outcome, "fit_cut": target.cut.date(), "training_days": target.training_days, "counted_days": target.counted_days,
            "predicted_change_points": float(change.mean()), "predicted_lower_5th": float(low), "predicted_upper_95th": float(high), "predicted_change_day_before_points": float(frozen.mean()),
            "prior_sd_ratio_moved_settings": ratio, "prediction_basis": "prior only" if ratio >= PRIOR_DOMINATED_RATIO else "data and prior", "prior_dominated": bool(ratio >= PRIOR_DOMINATED_RATIO),
            "expected_new_pct": 100.0 * float(draws["expected_new"].mean()), "expected_baseline_pct": 100.0 * float(draws["expected_baseline"].mean()),
            "observed_window_pct": target.observed[outcome], "predictive_new_lower_5th": 100.0 * float(new_low), "predictive_new_upper_95th": 100.0 * float(new_high),
            "crps_new_points": crps_new, "crps_old_points": crps_old, "crps_new_minus_old_points": crps_new - crps_old,
        })
    return pd.DataFrame(rows)


def check_posterior_precedes(posterior: fh.ModelPosterior, target: EventTarget) -> None:
    """The stored fit must be the one for the change's cut."""
    if posterior.origin != target.cut:
        raise AssertionError(f"The fit for {posterior.origin.date()} was given to a change that needs the fit for {target.cut.date()}")


# ---------------------------------------------------------------------------
# Comparison with what happened
# ---------------------------------------------------------------------------

def observed_changes(scores: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """The replay's observed change minus placebo change per event (percentage points), and the noise range of such a change when nothing changed.

    The observed change is the change's miss minus the mean of its 5 matched placebo misses, so with no effect its variance is 1 + 1/5 times that of one miss. The range is the 5th and 95th
    percentiles of the placebo misses centred on their mean and widened by the square root of 1.2.
    """
    contrast = replay._contrasts(scores, replay.PRIMARY, "matched placebo", outcome)
    placebo = scores[scores["predictor"].eq(replay.PRIMARY) & scores["kind"].eq("matched placebo")][f"{outcome}_residual"].to_numpy()
    low, high = np.percentile(placebo - placebo.mean(), INTERVAL) * np.sqrt(1.0 + 1.0 / replay.PLACEBOS_PER_EVENT)
    out = contrast[["contrast"]].rename(columns={"contrast": "observed_change_points"}).reset_index()
    out["no_change_noise_lower_5th"], out["no_change_noise_upper_95th"] = float(low), float(high)
    return out


def join_observed(rows: pd.DataFrame, scores: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """The event rows with the replay's observed change, and whether the predicted change's credible interval and the predicted change plus the no-change noise range hold it."""
    merged = rows.merge(observed_changes(scores, outcome), on="event_id", how="left", validate="one_to_one")
    if merged["observed_change_points"].isna().any():
        raise RuntimeError("A predicted change has no observed change in the replay")
    merged["inside_credible_interval"] = (merged["observed_change_points"] >= merged["predicted_lower_5th"]) & (merged["observed_change_points"] <= merged["predicted_upper_95th"])
    merged["inside_noise_range"] = ((merged["observed_change_points"] >= merged["predicted_change_points"] + merged["no_change_noise_lower_5th"])
                                    & (merged["observed_change_points"] <= merged["predicted_change_points"] + merged["no_change_noise_upper_95th"]))
    merged["direction_agrees"] = np.sign(merged["predicted_change_points"]) == np.sign(merged["observed_change_points"])
    return merged


# ---------------------------------------------------------------------------
# Cluster bootstrap and the statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Interval:
    """A statistic on the whole sample with the 90% percentile interval over cluster resamples and how many resamples left it undefined."""

    estimate: float
    lower: float
    upper: float
    undefined: int
    clusters: int


def slope(values: np.ndarray) -> float:
    """Least-squares slope of column 1 on column 0 (NaN when column 0 does not vary or fewer than 3 rows)."""
    x, y = values[:, 0], values[:, 1]
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan")
    return float(np.cov(x, y, bias=True)[0, 1] / x.var())


def correlation(values: np.ndarray) -> float:
    """Pearson correlation of columns 0 and 1 (NaN when either does not vary or fewer than 3 rows)."""
    x, y = values[:, 0], values[:, 1]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def agreement(values: np.ndarray) -> float:
    """Share of rows where the signs of columns 0 and 1 are the same."""
    return float(np.mean(np.sign(values[:, 0]) == np.sign(values[:, 1])))


def column_mean(values: np.ndarray) -> float:
    """Mean of column 0."""
    return float(values[:, 0].mean())


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, statistic: Callable[[np.ndarray], float], *, draws: int = BOOTSTRAP_DRAWS, seed: int = SEED) -> Interval:
    """``statistic`` of ``values`` (rows x columns) with a 90% percentile interval from resampling whole clusters (rows that share a cluster label stay together) with replacement."""
    values = np.asarray(values, dtype=float)
    labels, inverse = np.unique(clusters, return_inverse=True)
    groups = [values[inverse == index] for index in range(len(labels))]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(groups), size=(draws, len(groups)))
    with np.errstate(invalid="ignore", divide="ignore"):
        resampled = np.array([statistic(np.concatenate([groups[index] for index in row])) for row in picks])
    finite = resampled[np.isfinite(resampled)]
    if finite.size == 0:
        return Interval(float(statistic(values)), float("nan"), float("nan"), int(draws), len(groups))
    lower, upper = np.percentile(finite, INTERVAL)
    return Interval(float(statistic(values)), float(lower), float(upper), int(draws - finite.size), len(groups))


def group_rows(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """The changes of a reporting group: the isolated ones (no other change within 14 days; primary) or all."""
    return frame[frame["isolated"]] if group == "isolated changes" else frame


def tracking_table(joined: pd.DataFrame, *, draws: int = BOOTSTRAP_DRAWS, predicted: str = "predicted_change_points", variant: str = "primary") -> pd.DataFrame:
    """Per outcome and group: how the predicted changes line up with the observed ones (slope, correlation, direction agreement, interval coverage), each with a cluster-bootstrap interval.

    A group with fewer than 3 changes or fewer than 2 clusters gets a row that says it was not estimated and why.
    """
    rows = []
    for outcome, frame_outcome in joined.groupby("outcome", sort=False):
        for group in GROUPS:
            frame = group_rows(frame_outcome, group)
            row = {"variant": variant, "outcome": outcome, "group": group, "changes": len(frame), "clusters": int(frame["cluster_id"].nunique())}
            if len(frame) < 3 or frame["cluster_id"].nunique() < 2:
                row["reading"] = f"not estimated: {len(frame)} changes in {row['clusters']} clusters (at least 3 changes in 2 clusters are needed)"
                rows.append(row)
                continue
            pair = frame[[predicted, "observed_change_points"]].to_numpy()
            clusters = frame["cluster_id"].to_numpy()
            row.update({"mean_predicted_points": float(frame[predicted].mean()), "mean_observed_points": float(frame["observed_change_points"].mean())})
            for name, statistic in (("slope", slope), ("correlation", correlation), ("direction_agreement", agreement)):
                result = cluster_bootstrap(pair, clusters, statistic, draws=draws)
                row.update({name: result.estimate, f"{name}_lower_5th": result.lower, f"{name}_upper_95th": result.upper, f"{name}_undefined_resamples": result.undefined})
            row["inside_credible_interval"] = float(frame["inside_credible_interval"].mean())
            row["inside_noise_range"] = float(frame["inside_noise_range"].mean())
            row["reading"] = tracking_reading(row)
            rows.append(row)
    return pd.DataFrame(rows)


def tracking_reading(row: dict[str, object]) -> str:
    """What the slope's interval says: observed changes rise with the predicted ones, move against them, or the record does not tell."""
    low, high = row["slope_lower_5th"], row["slope_upper_95th"]
    if not (np.isfinite(low) and np.isfinite(high)):
        return "the slope is undefined in the resamples; no reading"
    if low > 0:
        return "the slope interval is above zero: observed changes rise with the predicted ones"
    if high < 0:
        return "the slope interval is below zero: observed changes moved against the predicted ones"
    return "the slope interval holds zero: the record does not show that the predicted changes track what happened"


def information_table(joined: pd.DataFrame, *, draws: int = BOOTSTRAP_DRAWS) -> pd.DataFrame:
    """Per outcome and group: mean CRPS under the new and the old settings and their paired difference (new minus old, percentage points) with a cluster-bootstrap interval."""
    rows = []
    for outcome, frame_outcome in joined.groupby("outcome", sort=False):
        for group in GROUPS:
            frame = group_rows(frame_outcome, group)
            row = {"outcome": outcome, "group": group, "changes": len(frame), "clusters": int(frame["cluster_id"].nunique())}
            if len(frame) < 3 or row["clusters"] < 2:
                row["reading"] = f"not estimated: {len(frame)} changes in {row['clusters']} clusters (at least 3 changes in 2 clusters are needed)"
                rows.append(row)
                continue
            result = cluster_bootstrap(frame[["crps_new_minus_old_points"]].to_numpy(), frame["cluster_id"].to_numpy(), column_mean, draws=draws)
            row.update({"mean_crps_new_points": float(frame["crps_new_points"].mean()), "mean_crps_old_points": float(frame["crps_old_points"].mean()), "mean_difference_points": result.estimate,
                        "lower_5th": result.lower, "upper_95th": result.upper, "share_changes_new_better": float((frame["crps_new_minus_old_points"] < 0).mean()), "reading": information_reading(result)})
            rows.append(row)
    return pd.DataFrame(rows)


def information_reading(result: Interval) -> str:
    """What the paired CRPS difference says about the settings in force: better, no different from, or worse than the old settings at predicting what happened."""
    if result.upper < 0:
        return "the interval is below zero: the settings in force predict the window better than the old settings"
    if result.lower > 0:
        return "the interval is above zero: the old settings predict the window better than the settings in force"
    return "the interval holds zero: the record does not show that the settings in force predict the window better than the old settings"


def latest_isolated(frame: pd.DataFrame) -> pd.Series:
    """The most recent isolated change of one outcome's rows."""
    return frame[frame["isolated"]].sort_values("change_date").iloc[-1]


def sensitivity_table(joined: pd.DataFrame, *, draws: int = BOOTSTRAP_DRAWS) -> pd.DataFrame:
    """The tracking estimates under three variations: the old arm as the profile the day before the change instead of the baseline days' settings; the latest isolated change left out
    (isolated group only, time in range); and the changes whose prediction rests on the prior left out."""
    parts = [tracking_table(joined, draws=draws, predicted="predicted_change_day_before_points", variant="predicted against the profile the day before the change")]
    tir = joined[joined["outcome"].eq(PRIMARY_OUTCOME)]
    if tir["isolated"].any():
        latest = latest_isolated(tir)
        without = tracking_table(tir[tir["event_id"] != latest["event_id"]], draws=draws, variant=f"latest isolated change ({latest['change_date']}) left out")
        parts.append(without[without["group"].eq("isolated changes")])
    kept = joined[~joined["prior_dominated"]]
    if len(kept) < len(joined):
        parts.append(tracking_table(kept, draws=draws, variant="changes predicted from the prior alone left out"))
    return pd.concat(parts, ignore_index=True)


def power_table(joined: pd.DataFrame, *, simulations: int | None = None, draws: int | None = None, seed: int = SEED) -> pd.DataFrame:
    """Power of the slope interval to find a true slope: simulate observed changes as ``slope`` times the real predicted changes plus normal noise with the real residual SD of the observed changes, keep the real clusters, and count how often the 90% cluster-bootstrap interval lies above zero.

    For the primary outcome, per group, for true slopes of ``POWER_SLOPES`` (1 = the predicted sizes are right).
    """
    simulations = POWER_SIMULATIONS if simulations is None else simulations
    draws = POWER_BOOTSTRAP_DRAWS if draws is None else draws
    rows = []
    for outcome, frame_outcome in joined[joined["outcome"].eq(PRIMARY_OUTCOME)].groupby("outcome", sort=False):
        for group in GROUPS:
            frame = group_rows(frame_outcome, group)
            if len(frame) < 3 or frame["cluster_id"].nunique() < 2 or frame["predicted_change_points"].nunique() < 2:
                continue
            x = frame["predicted_change_points"].to_numpy()
            y = frame["observed_change_points"].to_numpy()
            fit = np.polyfit(x, y, 1)
            noise = float(np.std(y - np.polyval(fit, x), ddof=2))
            clusters = frame["cluster_id"].to_numpy()
            rng = np.random.default_rng(seed)
            row = {"outcome": outcome, "group": group, "changes": len(frame), "clusters": int(frame["cluster_id"].nunique()), "predicted_change_sd_points": float(x.std(ddof=1)), "noise_sd_points": noise}
            for true_slope in POWER_SLOPES:
                hits = 0
                for _ in range(simulations):
                    simulated = true_slope * x + rng.normal(0.0, noise, len(x))
                    hits += int(cluster_bootstrap(np.column_stack([x, simulated]), clusters, slope, draws=draws, seed=int(rng.integers(2**31))).lower > 0)
                row[f"power_true_slope_{true_slope:g}"] = hits / simulations
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def run(raw: str | None, output: str | None, *, outcomes: tuple[str, ...], draws: int, tune: int, chains: int, workers: int, reuse: bool, target_accept: float, min_train_days: int, command: str, note: str) -> int:
    from .analysis_data import load_analysis_data
    from .hour_day import build_hour_day_table
    from .paths import ProjectPaths
    from .replay_predictor_report import write_outputs

    paths = ProjectPaths.from_root(Path.cwd())
    raw = raw or str(paths.cloud_raw)
    folder = Path(output or paths.output_replay_predictor)
    folder.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(folder / f"replay_predictor_run_{run_id}.log")])
    started = time.time()
    LOGGER.info("replay predictor started run_id=%s outcomes=%s raw=%s", run_id, ",".join(outcomes), raw)
    analysis = load_analysis_data(raw)
    table = build_hour_day_table(analysis)
    daily = fh.daily_counts(table, PRIMARY_OUTCOME)
    tdd = fh.daily_insulin(analysis.grid, analysis.merged.tandem_data.basal)
    profiles = fh.frozen_profiles(analysis.history, table)
    results = replay.run_replay(table)
    scores = results["scores"]
    grid = replay.day_grid(table)
    events = replay.find_events(grid)
    clusters, isolated = replay.cluster_events(events)
    targets, flow, issues = plan_targets(table, grid, events, clusters, isolated, scores, results["issues"], profiles, daily, tdd, min_train_days=min_train_days)
    for target in targets:
        check_no_leakage(table, target)
    cuts = sorted({target.cut for target in targets})
    LOGGER.info("changes flow\n%s\n%d changes predicted from %d fits per outcome", flow.to_string(index=False), len(targets), len(cuts))
    posteriors: dict[str, dict[pd.Timestamp, fh.ModelPosterior]] = {}
    joined = []
    for outcome in outcomes:
        posteriors[outcome] = fh.fit_all_origins(table, cuts, outcome=outcome, draws=draws, tune=tune, chains=chains, workers=workers, store=folder / "posteriors" / outcome if reuse else None, target_accept=target_accept)
        joined.append(join_observed(event_rows(posteriors[outcome], targets, outcome), scores, outcome))
        LOGGER.info("%s: %d changes predicted", outcome, len(joined[-1]))
    joined_frame = pd.concat(joined, ignore_index=True)
    write_outputs(folder=folder, run_id=run_id, raw=raw, table=table, flow=flow, issues=issues, joined=joined_frame, targets=targets, posteriors=posteriors, replay_results=results, draws=draws, tune=tune,
                  chains=chains, target_accept=target_accept, min_train_days=min_train_days, seconds=time.time() - started, command=command, note=note)
    LOGGER.info("replay predictor finished in %.1f s", time.time() - started)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", default=None, help="Raw data root (default: the Project Vault data/raw)")
    parser.add_argument("--output", default=None, help="Output folder (default: runtime/output/replay_predictor)")
    parser.add_argument("--outcomes", default=",".join(OUTCOMES), help="Comma-separated outcomes to fit and score (tir, below70, below54)")
    parser.add_argument("--draws", type=int, default=3000, help="Posterior draws per chain of each fit (the harness setting)")
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=om.TARGET_ACCEPT, help="NUTS target acceptance of every fit")
    parser.add_argument("--workers", type=int, default=4, help="Fits run at the same time")
    parser.add_argument("--min-train-days", type=int, default=MIN_TRAIN_DAYS, help="Fewest days with usable hours before a change for it to be predicted")
    parser.add_argument("--reuse-posteriors", action="store_true", help="Store each fit in the output folder and reuse it when the training data, model code and sampler settings are identical")
    parser.add_argument("--run-note", default="", help="Free text recorded in the summary and the workbook")
    args = parser.parse_args(argv)
    outcomes = tuple(name for name in args.outcomes.split(",") if name)
    unknown = sorted(set(outcomes) - set(OUTCOMES))
    if unknown or PRIMARY_OUTCOME not in outcomes:
        raise SystemExit(f"--outcomes must include {PRIMARY_OUTCOME} and only name {OUTCOMES}; got {args.outcomes}")
    command = " ".join(["python -m bayesian_t1dm.replay_predictor", f"--outcomes {args.outcomes}", f"--draws {args.draws}", f"--tune {args.tune}", f"--chains {args.chains}", f"--workers {args.workers}",
                        f"--min-train-days {args.min_train_days}", *(["--reuse-posteriors"] if args.reuse_posteriors else []), *([f'--run-note "{args.run_note}"'] if args.run_note else [])])
    return run(args.raw, args.output, outcomes=outcomes, draws=args.draws, tune=args.tune, chains=args.chains, workers=args.workers, reuse=args.reuse_posteriors, target_accept=args.target_accept, min_train_days=args.min_train_days,
               command=command, note=args.run_note)


if __name__ == "__main__":
    raise SystemExit(main())
