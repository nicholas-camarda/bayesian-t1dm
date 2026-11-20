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
from scipy.special import expit

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
SETTING_PRIOR_SD = 0.5


class NotPredicted(Exception):
    """A change cannot be predicted; the reason is listed in the record issues (the change is excluded, never filled in)."""


# ---------------------------------------------------------------------------
# Targets: what each change asks of the model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventTarget:
    """One replay change as the model sees it.

    ``counts`` (14 x 24) are the valid readings of the replay's counted days in the affected hours (zero elsewhere); ``old`` (24 x 4) is the settings profile in
    force before the change and ``new`` (14 x 24 x 4) the settings in force on each day of the window (icr, cf, basal, target; NaN where unknown); ``pool`` is
    the days x 24 x 4 exposure pool known at the change date; ``cut`` is the first day the fit may not see.
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
    pool: np.ndarray
    observed: dict[str, float]
    cut: pd.Timestamp


def describe_change(old: np.ndarray, new: np.ndarray, hours: np.ndarray) -> str:
    """The change in the units the pump uses, as the mean over the clock hours of ``hours`` where that setting differs (``old`` and ``new`` are 24 x 4 profiles)."""
    parts = []
    for index, (_, label, unit) in enumerate(SETTING_UNITS):
        differs = hours & ~(np.abs(new[:, index] - old[:, index]) < fh.NO_SETTINGS_CHANGE)
        before, after = old[differs, index], new[differs, index]
        if not differs.any():
            continue
        if not (np.isfinite(before).all() and np.isfinite(after).all()):
            continue
        if np.abs(after - before).max() < fh.NO_SETTINGS_CHANGE:
            continue
        digits = {"basal": 3, "I:C": 2, "CF": 2}.get(label, 1)
        parts.append(f"{label} {before.mean():.{digits}f} to {after.mean():.{digits}f} {unit}")
    return "; ".join(parts) if parts else "differs only on later days of the window"


def plan_targets(table: pd.DataFrame, grid: replay.DayGrid, events: list[replay.Event], clusters: dict[int, int], isolated: set[int], scores: pd.DataFrame, replay_issues: pd.DataFrame,
                 profiles: fh.SettingsProfiles, daily: pd.DataFrame, tdd: pd.Series, *, min_train_days: int = MIN_TRAIN_DAYS) -> tuple[list[EventTarget], pd.DataFrame, pd.DataFrame]:
    """Every replay change that can be predicted (with the fit cut it will use), the flow of changes through the stages, and the record issues of those left out.

    A change is left out, with the reason, when the replay did not score it, when fewer than ``min_train_days`` days with usable hours precede it, when a setting is unknown
    at an hour the replay counted (before or after the change), or when no complete day of exposure precedes it.
    """
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
        known = fh.known_at(date, PRIMARY_OUTCOME, table, daily, tdd, profiles)
        old = known.profiles.at(date)
        new = fh.window_for(table, window_start).actual_settings
        try:
            if not np.isfinite(old[event.affected_hours]).all():
                raise NotPredicted(("settings before the change are unknown at an affected hour", "hours unknown before", int((~np.isfinite(old[event.affected_hours]).all(axis=1)).sum())))
            if not np.isfinite(new[counts > 0]).all():
                raise NotPredicted(("settings after the change are unknown at an affected hour the replay counted", "cells unknown after", int((~np.isfinite(new[counts > 0]).all(axis=1)).sum())))
            try:
                pool = fh.exposure_pool(known)
            except fh.NotScored as problem:
                raise NotPredicted((str(problem), "complete exposure days", 0)) from problem
        except NotPredicted as problem:
            reason, field, value = problem.args[0]
            issues.append({**identity, "step": "settings known and exposures available", "reason": reason, "field": field, "value": value})
            continue
        stage["with settings known and exposures available"] += 1
        counted = counts > 0
        if np.abs(new[counted] - np.broadcast_to(old, new.shape)[counted]).max() < fh.NO_SETTINGS_CHANGE:
            issues.append({**identity, "step": "different setting values", "reason": "the setting values in the counted hours are the same before and after (only a segment start time moved), so the predicted change is exactly zero and the change has no setting to test",
                           "field": "largest difference in a setting value", "value": 0.0})
            continue
        stage["with different setting values before and after"] += 1
        observed, n_days = replay.observe(grid, event.first, event.span, event.affected_hours)
        after_old = profiles.at(window_start)
        targets.append(EventTarget(event.event_id, date, window_start, clusters[event.event_id], event.event_id in isolated, event.direction, event.settings,
                                   describe_change(old, after_old, event.affected_hours), n_days, training_days, counts, old, new, pool, observed, date))
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

def linear_predictor(posterior: fh.ModelPosterior, settings: np.ndarray, exposures: np.ndarray, counts: np.ndarray, day_noise: np.ndarray) -> np.ndarray:
    """The model's linear predictor (draws x 336 cells of 14 days x 24 hours) for the model-scale ``settings`` (14 x 24 x 4), exposures (draws x 14 x 24 x 4) and day effects (draws x 14).

    Cells without readings are not used and take no settings or exposure.
    """
    draws = len(posterior.alpha)
    hours = np.tile(np.arange(HOURS_PER_DAY), WINDOW_DAYS)
    present = counts.reshape(-1) > 0
    flat = settings.reshape(-1, 4)
    if not np.isfinite(flat[present]).all():
        raise NotPredicted(("a setting is unknown at an hour with readings", "cells", int((~np.isfinite(flat[present]).all(axis=1)).sum())))
    centered = np.where(present[:, None], (flat - posterior.center[:4]) / posterior.scale[:4], 0.0)
    exposure = np.where(present[None, :, None], (exposures.reshape(draws, -1, 4) - posterior.center[4:]) / posterior.scale[4:], 0.0)
    return (posterior.alpha[:, None] + posterior.hour_scale[:, None] * posterior.z_hour[:, hours] + posterior.day_scale[:, None] * day_noise[:, np.repeat(np.arange(WINDOW_DAYS), HOURS_PER_DAY)]
            + posterior.beta[:, :4] @ centered.T + np.einsum("dpk,dk->dp", exposure, posterior.beta[:, 4:]))


def simulate_share(mu: np.ndarray, log_phi: np.ndarray, counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Predictive draws of the aggregate share: a beta draw of each cell's share and binomial counts at the realised number of readings."""
    phi = np.exp(log_phi)[:, None]
    flat = counts.reshape(-1).astype(int)
    hits = rng.binomial(flat[None, :], rng.beta(mu * phi, (1.0 - mu) * phi))
    return hits.sum(axis=1) / flat.sum()


def predict_event(posterior: fh.ModelPosterior, target: EventTarget, seed: int) -> dict[str, np.ndarray]:
    """Expected and simulated aggregate share of the window under the new settings and under the old, one posterior draw at a time.

    The two settings share the posterior draw, the day effects and the resampled exposures, so the expected change is the settings' contribution alone.
    """
    rng = np.random.default_rng(seed)
    draws = len(posterior.alpha)
    exposures = target.pool[rng.integers(0, len(target.pool), size=(draws, WINDOW_DAYS))]
    day_noise = rng.normal(size=(draws, WINDOW_DAYS))
    old = np.broadcast_to(target.old, (WINDOW_DAYS, HOURS_PER_DAY, 4))
    weights = target.counts.reshape(-1) / target.counts.sum()
    result: dict[str, np.ndarray] = {}
    for name, profile in (("new", target.new), ("old", old)):
        mu = expit(linear_predictor(posterior, fh.settings_covariates(profile), exposures, target.counts, day_noise))
        result[f"expected_{name}"] = mu @ weights
        result[f"simulated_{name}"] = simulate_share(mu, posterior.log_phi, target.counts, np.random.default_rng([seed, 1]))
    return result


def event_rows(posteriors: dict[pd.Timestamp, fh.ModelPosterior], targets: list[EventTarget], outcome: str) -> pd.DataFrame:
    """One row per change for ``outcome``: predicted change (percentage points, mean and 90% credible interval), the two expected shares, the observed share, and the CRPS under the new and old settings."""
    rows = []
    for target in targets:
        posterior = posteriors[target.cut]
        check_posterior_precedes(posterior, target)
        draws = predict_event(posterior, target, seed=int(np.random.default_rng([SEED, target.event_id, OUTCOMES.index(outcome)]).integers(2**31)))
        change = 100.0 * (draws["expected_new"] - draws["expected_old"])
        low, high = np.percentile(change, INTERVAL)
        observed = target.observed[outcome] / 100.0
        crps_new, crps_old = crps_points(draws["simulated_new"], observed), crps_points(draws["simulated_old"], observed)
        new_low, new_high = np.percentile(draws["simulated_new"], INTERVAL)
        rows.append({
            "event_id": target.event_id, "change_date": target.event_date.date(), "cluster_id": target.cluster_id, "isolated": target.isolated, "direction": target.direction,
            "change": target.description, "outcome": outcome, "fit_cut": target.cut.date(), "training_days": target.training_days, "counted_days": target.counted_days,
            "predicted_change_points": float(change.mean()), "predicted_lower_5th": float(low), "predicted_upper_95th": float(high),
            "expected_new_pct": 100.0 * float(draws["expected_new"].mean()), "expected_old_pct": 100.0 * float(draws["expected_old"].mean()),
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
    """The replay's observed change minus placebo change per event (percentage points), and the 5th and 95th percentiles of the misses on matched placebo dates (the no-change spread)."""
    contrast = replay._contrasts(scores, replay.PRIMARY, "matched placebo", outcome)
    placebo = scores[scores["predictor"].eq(replay.PRIMARY) & scores["kind"].eq("matched placebo")][f"{outcome}_residual"]
    low, high = np.percentile(placebo, INTERVAL)
    out = contrast[["contrast"]].rename(columns={"contrast": "observed_change_points"}).reset_index()
    out["placebo_spread_lower_5th"], out["placebo_spread_upper_95th"] = float(low), float(high)
    return out


def join_observed(rows: pd.DataFrame, scores: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """The event rows with the replay's observed change, and whether the predicted change's credible interval and the placebo-widened interval hold it."""
    merged = rows.merge(observed_changes(scores, outcome), on="event_id", how="left", validate="one_to_one")
    if merged["observed_change_points"].isna().any():
        raise RuntimeError("A predicted change has no observed change in the replay")
    merged["inside_credible_interval"] = (merged["observed_change_points"] >= merged["predicted_lower_5th"]) & (merged["observed_change_points"] <= merged["predicted_upper_95th"])
    merged["inside_placebo_widened_interval"] = ((merged["observed_change_points"] >= merged["predicted_change_points"] + merged["placebo_spread_lower_5th"])
                                                  & (merged["observed_change_points"] <= merged["predicted_change_points"] + merged["placebo_spread_upper_95th"]))
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


def tracking_table(joined: pd.DataFrame, *, draws: int = BOOTSTRAP_DRAWS) -> pd.DataFrame:
    """Per outcome and group: how the predicted changes line up with the observed ones (slope, correlation, direction agreement, interval coverage), each with a cluster-bootstrap interval."""
    rows = []
    for outcome, frame_outcome in joined.groupby("outcome", sort=False):
        for group in GROUPS:
            frame = group_rows(frame_outcome, group)
            if len(frame) < 3 or frame["cluster_id"].nunique() < 2:
                continue
            pair = frame[["predicted_change_points", "observed_change_points"]].to_numpy()
            clusters = frame["cluster_id"].to_numpy()
            row = {"outcome": outcome, "group": group, "changes": len(frame), "clusters": int(frame["cluster_id"].nunique()),
                   "mean_predicted_points": float(frame["predicted_change_points"].mean()), "mean_observed_points": float(frame["observed_change_points"].mean())}
            for name, statistic in (("slope", slope), ("correlation", correlation), ("direction_agreement", agreement)):
                result = cluster_bootstrap(pair, clusters, statistic, draws=draws)
                row.update({name: result.estimate, f"{name}_lower_5th": result.lower, f"{name}_upper_95th": result.upper, f"{name}_undefined_resamples": result.undefined})
            row["inside_credible_interval"] = float(frame["inside_credible_interval"].mean())
            row["inside_placebo_widened_interval"] = float(frame["inside_placebo_widened_interval"].mean())
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
        return "the slope interval is below zero: observed changes move against the predicted ones"
    return "the slope interval holds zero: the record does not show that the predicted changes track what happened"


def information_table(joined: pd.DataFrame, *, draws: int = BOOTSTRAP_DRAWS) -> pd.DataFrame:
    """Per outcome and group: mean CRPS under the new and the old settings and their paired difference (new minus old, percentage points) with a cluster-bootstrap interval."""
    rows = []
    for outcome, frame_outcome in joined.groupby("outcome", sort=False):
        for group in GROUPS:
            frame = group_rows(frame_outcome, group)
            if len(frame) < 3 or frame["cluster_id"].nunique() < 2:
                continue
            result = cluster_bootstrap(frame[["crps_new_minus_old_points"]].to_numpy(), frame["cluster_id"].to_numpy(), column_mean, draws=draws)
            rows.append({"outcome": outcome, "group": group, "changes": len(frame), "clusters": result.clusters, "mean_crps_new_points": float(frame["crps_new_points"].mean()),
                         "mean_crps_old_points": float(frame["crps_old_points"].mean()), "mean_difference_points": result.estimate, "lower_5th": result.lower, "upper_95th": result.upper,
                         "share_changes_new_better": float((frame["crps_new_minus_old_points"] < 0).mean()), "reading": information_reading(result)})
    return pd.DataFrame(rows)


def information_reading(result: Interval) -> str:
    """What the paired CRPS difference says about the settings in force: better, no different from, or worse than the old settings at predicting what happened."""
    if result.upper < 0:
        return "the interval is below zero: the settings in force predict the window better than the old settings"
    if result.lower > 0:
        return "the interval is above zero: the old settings predict the window better than the settings in force"
    return "the interval holds zero: the record does not show that the settings in force predict the window better than the old settings"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def run(raw: str | None, output: str | None, *, outcomes: tuple[str, ...], draws: int, tune: int, chains: int, workers: int, reuse: bool, min_train_days: int, command: str, note: str) -> int:
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
        posteriors[outcome] = fh.fit_all_origins(table, cuts, outcome=outcome, draws=draws, tune=tune, chains=chains, workers=workers, store=folder / "posteriors" / outcome if reuse else None)
        joined.append(join_observed(event_rows(posteriors[outcome], targets, outcome), scores, outcome))
        LOGGER.info("%s: %d changes predicted", outcome, len(joined[-1]))
    joined_frame = pd.concat(joined, ignore_index=True)
    write_outputs(folder=folder, run_id=run_id, raw=raw, table=table, flow=flow, issues=issues, joined=joined_frame, targets=targets, posteriors=posteriors, replay_results=results, draws=draws, tune=tune,
                  chains=chains, min_train_days=min_train_days, seconds=time.time() - started, command=command, note=note)
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
    return run(args.raw, args.output, outcomes=outcomes, draws=args.draws, tune=args.tune, chains=args.chains, workers=args.workers, reuse=args.reuse_posteriors, min_train_days=args.min_train_days,
               command=command, note=args.run_note)


if __name__ == "__main__":
    raise SystemExit(main())
