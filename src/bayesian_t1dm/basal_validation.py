"""Walk-forward scoring of one future eligible day per controller segment and context."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .analysis_data import load_analysis_data
from .controller import basal_records, daily_controller_pattern, meal_windows

HALF_LIFE_DAYS = 28
BASELINE_DAYS = 28
THRESHOLD = 0.10
HOLDOUT_DAYS = 120
EVENING_SEGMENT = "19:00-21:00"


def controller_forecast_rows(analysis) -> pd.DataFrame:
    """Separate fasting stretches from those after a logged evening meal."""
    parts, _ = analysis.day_parts()
    rates = basal_records(analysis.merged, analysis.history)
    data = analysis.merged.tandem_data
    common = dict(unlogged=analysis.unlogged, min_clean_fraction=None, workouts=analysis.merged.data.workouts)
    fasting = daily_controller_pattern(rates, analysis.mode_events, analysis.cgm, meal_windows(data.bolus, data.carbs),
                                      parts, **common).assign(context="Eligible fasting")
    # A usual post-dinner row requires logged carbohydrate from a qualifying evening meal.
    dinner = data.carbs.loc[data.carbs.timestamp.dt.hour.ge(17)]
    windows = meal_windows(data.bolus.iloc[:0], dinner)
    evening = [part for part in parts if part.name == EVENING_SEGMENT]
    post = daily_controller_pattern(rates, analysis.mode_events, analysis.cgm, windows, evening, post_meal=True,
                                    **common).assign(context="Post-dinner")
    rows = pd.concat([fasting, post], ignore_index=True)
    rows["command_ratio"] = rows.commanded_units / rows.profile_units
    rows["context_event_day"] = rows.date.isin(analysis.excluded_days)
    return rows.sort_values(["date", "part", "context"]).reset_index(drop=True)


def baseline_distribution(history: pd.DataFrame, origin: pd.Timestamp, *, decay: bool) -> dict[str, float] | None:
    """Hours-weighted log-rate distribution with separate zero mass and a calendar-day lookback."""
    age = (origin - history.date).dt.days
    keep = age.gt(0)
    if not decay:
        keep &= age.le(BASELINE_DAYS)
    age = age.loc[keep].to_numpy()
    units = history.loc[keep, "command_ratio"].to_numpy()
    hours = history.loc[keep, "clean_hours"].to_numpy()
    weights = hours * np.exp2(-age / HALF_LIFE_DAYS) if decay else hours
    if not len(units):
        return None
    weights = weights / weights.sum()
    # Jeffreys smoothing on the effective weighted days keeps rare zero outcomes scoreable.
    effective_days = 1 / np.sum(weights**2)
    zero = units == 0
    zero_mass = (weights[zero].sum() * effective_days + 0.5) / (effective_days + 1)
    positive = np.log(units[~zero])
    if len(positive) < 2:
        return None
    positive_weights = weights[~zero]
    mean = np.average(positive, weights=positive_weights)
    sd = np.sqrt(np.average((positive - mean)**2, weights=positive_weights))
    if sd <= 0:
        return None
    # Match the weighted arithmetic mean of the positive rate ratios.
    mean = np.log(np.average(units[~zero], weights=positive_weights)) - 0.5 * sd**2
    return dict(mean=mean, sd=sd, zero_mass=zero_mass, training_days=len(units), training_hours=hours.sum())


def score_distribution(distribution: dict[str, float], observed: float) -> dict[str, float]:
    """Predictive interval and threshold probabilities on the ratio scale, with log score on the log-rate scale."""
    mean, sd, zero = (distribution[name] for name in ("mean", "sd", "zero_mass"))
    lo, hi = [float(np.exp(norm.ppf((q - zero) / (1 - zero), loc=mean, scale=sd))) if q > zero else 0.0 for q in (0.05, 0.95)]
    above = (1 - zero) * norm.sf(np.log1p(THRESHOLD), loc=mean, scale=sd)
    below = zero + (1 - zero) * norm.cdf(np.log1p(-THRESHOLD), loc=mean, scale=sd)
    log_score = np.log(zero) if observed == 0 else np.log1p(-zero) + norm.logpdf(np.log(observed), loc=mean, scale=sd)
    return dict(lower_90=lo, upper_90=hi, covered=lo <= observed <= hi, p_above=above, p_below=below,
                hit_above=observed >= 1 + THRESHOLD, hit_below=observed <= 1 - THRESHOLD, log_score=log_score)


def rolling_baselines(rows: pd.DataFrame) -> pd.DataFrame:
    """Use identical held-out evenings for both comparators, with no same-day outcome in training."""
    start = rows.date.max() - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    results = []
    for (part, context), group in rows.groupby(["part", "context"]):
        for row in group[group.date.ge(start)].itertuples():
            distributions = [baseline_distribution(group, row.date, decay=decay) for decay in (True, False)]
            for model, dist in zip(("28-day half-life", "last 28 days"), distributions):
                identity = dict(date=row.date, part=part, context=context, model=model,
                                observed=row.command_ratio, hours=row.clean_hours, paired=all(d is not None for d in distributions))
                results.append({**identity, **dist, **score_distribution(dist, row.command_ratio)} if dist else identity)
    return pd.DataFrame(results)


def write_baseline_validation(raw: Path, output: Path) -> None:
    rows = controller_forecast_rows(load_analysis_data(raw))
    results = rolling_baselines(rows)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "controller_evenings.csv", index=False)
    results.to_csv(output / "walk_forward.csv", index=False)
    scored = results.loc[results.paired].copy()
    scored["width"] = scored.upper_90 - scored.lower_90
    scored["brier_above"] = (scored.p_above - scored.hit_above) ** 2
    scored["brier_below"] = (scored.p_below - scored.hit_below) ** 2
    summary = scored.groupby(["part", "context", "model"]).agg(
        evenings=("date", "size"), hours=("hours", "sum"), mean_log_score=("log_score", "mean"),
        coverage=("covered", "mean"), mean_width=("width", "mean"),
        p_above=("p_above", "mean"), observed_above=("hit_above", "mean"), brier_above=("brier_above", "mean"),
        p_below=("p_below", "mean"), observed_below=("hit_below", "mean"), brier_below=("brier_below", "mean"),
    )
    summary.to_csv(output / "walk_forward_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_baseline_validation(args.raw, args.output)
