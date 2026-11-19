"""Research model for the next eligible basal observation, conditional on a fixed profile."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

from .basal_validation import HALF_LIFE_DAYS, HOLDOUT_DAYS, THRESHOLD
from .model import FitDiagnostics, extract_fit_diagnostics

WEEK_DAYS = 7
TARGET_ACCEPT = 0.95
MAX_TREEDEPTH = 10
R_HAT_LIMIT = 1.01
ESS_LIMIT = 400
SEED = 20261118


@dataclass(frozen=True)
class ControllerDesign:
    outcome: np.ndarray
    hours: np.ndarray
    group: np.ndarray
    week: np.ndarray
    design: np.ndarray
    n_groups: int
    n_weeks: int


def build_model(data: ControllerDesign):
    """Use a shared weekly random walk, group-level variation and a separate point mass for suspension."""
    import pymc as pm
    import pytensor.tensor as pt

    observed = data.outcome
    positive = observed > 0
    missing = ~np.isfinite(data.design)
    values = np.where(missing, 0.0, data.design)
    with pm.Model() as model:
        intercept = pm.Normal("intercept", sigma=0.3)
        group_scale = pm.HalfNormal("group_scale", sigma=0.25)
        group_raw = pm.Normal("group_raw", shape=data.n_groups)
        shared_scale = pm.HalfNormal("shared_week_scale", sigma=0.05)
        shared_step = pm.Normal("shared_week_step", shape=data.n_weeks - 1)
        group_week_scale = pm.HalfNormal("group_week_scale", sigma=0.05)
        group_step = pm.Normal("group_week_step", shape=(data.n_groups, data.n_weeks - 1))
        shared_level = shared_scale * pt.concatenate((pt.zeros(1), pt.cumsum(shared_step)))
        group_level = group_week_scale * pt.concatenate((pt.zeros((data.n_groups, 1)), pt.cumsum(group_step, axis=1)), axis=1)
        level = pm.Deterministic("level", intercept + group_scale * group_raw[:, None] + shared_level + group_level)
        beta = pm.Normal("beta", sigma=0.15, shape=values.shape[1])
        day_scale = pm.HalfNormal("day_scale", sigma=0.5)
        hour_scale = pm.HalfNormal("hour_scale", sigma=0.25)
        mean = level[data.group[positive], data.week[positive]] + pt.dot(values[positive], beta)
        # Marginalize missing standardized inputs under independent N(0, 1) priors.
        var = day_scale**2 + hour_scale**2 / data.hours[positive] + pt.dot(missing[positive].astype(float), beta**2)
        pm.Normal("positive_log_ratio", mu=mean, sigma=pt.sqrt(var), observed=np.log(observed[positive]))
        zero_center = pm.Normal("zero_center", mu=-4.0, sigma=1.5)
        zero_scale = pm.HalfNormal("zero_group_scale", sigma=0.75)
        zero_effect = pm.Normal("zero_group", shape=data.n_groups)
        zero_probability = pm.Deterministic("p_zero", pm.math.sigmoid(zero_center + zero_scale * zero_effect))
        pm.Bernoulli("suspended", p=zero_probability[data.group], observed=~positive)
    return model


def sampler_gate(diagnostics: FitDiagnostics) -> bool:
    rhat, bulk, tail = diagnostics.rhat_max, diagnostics.ess_bulk_min, diagnostics.ess_tail_min
    return bool(rhat is not None and bulk is not None and tail is not None and
                rhat < R_HAT_LIMIT and min(bulk, tail) > ESS_LIMIT and diagnostics.divergences == 0)


def fit_model(data: ControllerDesign, *, draws: int = 600, tune: int = 600, chains: int = 4):
    import pymc as pm

    started = time.perf_counter()
    with build_model(data):
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=chains, random_seed=SEED,
                          target_accept=TARGET_ACCEPT, nuts={"max_treedepth": MAX_TREEDEPTH}, progressbar=False)
    diagnostics = extract_fit_diagnostics(idata, draws=draws, tune=tune, chains=chains, target_accept=TARGET_ACCEPT,
                                          max_treedepth=MAX_TREEDEPTH, wall_time_seconds=time.perf_counter() - started)
    return idata, diagnostics


def score_prediction(mean, sd, p_zero, observed: float, *, rng: np.random.Generator) -> dict[str, float]:
    """Score a posterior mixture with respect to point mass at zero and density of the positive log ratio."""
    predicted = np.where(rng.random(len(mean)) < p_zero, 0.0, np.exp(rng.normal(mean, sd)))
    lo, hi = np.quantile(predicted, [0.05, 0.95])
    log_density = np.log(p_zero) if observed == 0 else np.log1p(-p_zero) + norm.logpdf(np.log(observed), mean, sd)
    return dict(lower_90=lo, upper_90=hi, covered=lo <= observed <= hi, p_zero=p_zero.mean(),
                hit_above=observed >= 1 + THRESHOLD, hit_below=observed <= 1 - THRESHOLD,
                p_above=np.mean((1 - p_zero) * norm.sf(np.log1p(THRESHOLD), mean, sd)),
                p_below=np.mean(p_zero + (1 - p_zero) * norm.cdf(np.log1p(-THRESHOLD), mean, sd)),
                log_score=logsumexp(log_density) - np.log(len(mean)))


def design_for_training(train: pd.DataFrame, predictors: list[str]) -> tuple[ControllerDesign, pd.Series, pd.Series, pd.Index]:
    """Standardize on training rows alone and maintain whole calendar weeks through gaps."""
    groups = pd.MultiIndex.from_frame(train[["part", "context"]].drop_duplicates())
    group = groups.get_indexer(pd.MultiIndex.from_frame(train[["part", "context"]]))
    week = ((train.date - train.date.min()).dt.days // WEEK_DAYS).to_numpy()
    design = train[predictors]
    center = design.mean().fillna(0.0)
    scale = design.std(ddof=0).replace(0, 1.0).fillna(1.0)
    data = ControllerDesign(train.command_ratio.to_numpy(), train.clean_hours.to_numpy(), group, week,
                            ((design - center) / scale).to_numpy(), len(groups), int(week.max()) + 1)
    return data, center, scale, groups


def asof_previous(rows: pd.DataFrame, other: pd.DataFrame, right_on: str, **options) -> pd.DataFrame:
    """Match each row to the latest earlier record. The result keeps the order of `rows`; only the merge is sorted."""
    left = rows[["date"]].assign(_row=np.arange(len(rows))).sort_values("date", kind="stable")
    merged = pd.merge_asof(left, other.sort_values(right_on), left_on="date", right_on=right_on, **options)
    return merged.sort_values("_row").reset_index(drop=True)


def attach_origin_features(rows: pd.DataFrame, analysis) -> pd.DataFrame:
    """Assemble features known at midnight, before any outcome from that calendar day."""
    from .context_rules import EXERCISE_MIN_MINUTES

    rows = rows.copy().reset_index(drop=True)
    origin = rows.date
    rows["weekend"] = origin.dt.dayofweek.ge(5).astype(int)
    rows["prior_sleep_hours"] = np.nan
    sleep = analysis.merged.data.sleep.dropna(subset=["sleep_end"]).sort_values("sleep_end")
    if not sleep.empty:
        available = asof_previous(rows, sleep[["sleep_end", "total_sleep_hours"]], "sleep_end",
                                  tolerance=pd.Timedelta(hours=36))
        rows["prior_sleep_hours"] = available.total_sleep_hours
    for stream, date_column, destination, value in [
        (analysis.site_events, "timestamp", "days_since_site", None),
        (analysis.cgm, "timestamp", "prior_glucose", "glucose"),
    ]:
        data = stream.dropna(subset=[date_column]).sort_values(date_column)
        times = pd.DatetimeIndex(data[date_column])
        right = times.searchsorted(origin, side="left")
        if value is None:
            matched = times.to_series(index=np.arange(len(times))).reindex(right - 1)
            rows[destination] = (origin - matched.reset_index(drop=True)).dt.total_seconds() / 86400
        else:
            left = times.searchsorted(origin - pd.Timedelta(days=1))
            numbers = data[value].to_numpy(dtype=float)
            finite = np.isfinite(numbers)
            mass = np.r_[0, np.cumsum(np.where(finite, numbers, 0.0))]
            seen = np.r_[0, np.cumsum(finite)]
            count = seen[right] - seen[left]
            rows[destination] = np.divide(mass[right] - mass[left], count, out=np.full(len(count), np.nan), where=count > 0)
    workouts = analysis.merged.data.workouts
    workouts = workouts.loc[workouts.duration_seconds.ge(EXERCISE_MIN_MINUTES * 60)].dropna(subset=["end_timestamp"]).sort_values("end_timestamp")
    ends = pd.DatetimeIndex(workouts.end_timestamp)
    right = ends.searchsorted(origin, side="left")
    left = ends.searchsorted(origin - pd.Timedelta(days=1))
    matched = ends.to_series(index=np.arange(len(ends))).reindex(right - 1)
    rows["hours_since_workout"] = (origin - matched.reset_index(drop=True)).dt.total_seconds() / 3600
    energy = np.r_[0, workouts.active_energy_value.fillna(0).cumsum()]
    rows["workout_energy_24h"] = np.where(origin.between(ends.min(), ends.max()), energy[right] - energy[left], np.nan)
    rows["previous_sleep_mode_share"] = rows.groupby(["part", "context"]).sleep_share.shift(1)
    # A forecast for post-dinner behavior conditions on the exposure category. Use only earlier logged food values.
    nutrition = analysis.merged.data.nutrition
    if not nutrition.empty:
        columns = [name for name in ("carb_grams", "fat_grams", "protein_grams") if name in nutrition]
        if columns:
            meals = nutrition.loc[nutrition.timestamp.dt.hour.ge(17), ["timestamp", *columns]].sort_values("timestamp")
            previous = asof_previous(rows, meals, "timestamp", allow_exact_matches=False, tolerance=pd.Timedelta(hours=36))
            for name in columns:
                rows["prior_dinner_" + name] = previous[name]
            rows["hours_since_dinner_log"] = (origin - previous.timestamp).dt.total_seconds() / 3600
    rows["context_event_day"] = rows.context_event_day.astype(int)
    return rows


def predict_at_origin(train, test, predictors, *, output_dir):
    """Fit one elapsed-data fold. Diagnostics are saved first; a failed sampler gate stops the run."""
    from dataclasses import asdict
    from pathlib import Path

    from .basal_validation import baseline_distribution, score_distribution
    import pymc as pm

    from .model import BayesianGlucoseModel

    data, center, scale, groups = design_for_training(train, predictors)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with build_model(data):
        prior = pm.sample_prior_predictive(samples=100, random_seed=SEED, var_names=["positive_log_ratio", "p_zero"])
    prior_draws = np.asarray(prior.prior_predictive["positive_log_ratio"])
    odds = np.asarray(prior.prior["p_zero"])
    pd.DataFrame([
        {"quantity": quantity, "q05": np.quantile(draws, 0.05), "median": np.median(draws), "q95": np.quantile(draws, 0.95)}
        for quantity, draws in [("positive commanded/programmed ratio", np.exp(prior_draws)), ("zero probability", odds)]
    ]).to_csv(output / "prior_prediction.csv", index=False)
    idata, diagnostics = fit_model(data)
    pd.DataFrame([asdict(diagnostics) | {"sampler_gate": sampler_gate(diagnostics)}]).to_csv(output / "fit_diagnostics.csv", index=False)
    if not sampler_gate(diagnostics):
        raise RuntimeError(f"Sampling diagnostics failed; forecasts are not scored. See {output / 'fit_diagnostics.csv'}")

    draws = BayesianGlucoseModel._stack_draws
    beta = draws(idata, "beta")
    last = draws(idata, "level")[:, :, -1]
    shared_scale = draws(idata, "shared_week_scale")
    group_scale = draws(idata, "group_week_scale")
    day_sd = draws(idata, "day_scale")
    hour_sd = draws(idata, "hour_scale")
    zero = draws(idata, "p_zero")
    rng = np.random.default_rng(SEED)
    predictions = []
    for row in test.itertuples():
        key = (row.part, row.context)
        if key not in groups:
            continue
        idx = groups.get_loc(key)
        history = train.loc[train.part.eq(row.part) & train.context.eq(row.context)]
        base = [baseline_distribution(history, row.origin, decay=decay) for decay in (True, False)]
        if any(result is None for result in base):
            continue
        x = (pd.Series(row._asdict()).reindex(predictors).astype(float) - center) / scale
        missing = x.isna().to_numpy(dtype=float)
        steps = max(0, (row.date - train.date.min()).days // WEEK_DAYS - (data.n_weeks - 1))
        state_noise = np.sqrt(steps * (shared_scale**2 + group_scale**2))
        mean = last[:, idx] + rng.normal(0, state_noise) + beta @ x.fillna(0).to_numpy()
        age = (row.origin - history.date).dt.days.to_numpy()
        weights = np.exp2(-age / HALF_LIFE_DAYS)
        hours = rng.choice(history.clean_hours.to_numpy(), size=len(mean), p=weights / weights.sum())
        sd = np.sqrt(day_sd**2 + hour_sd**2 / hours + (beta**2) @ missing)
        y = row.command_ratio
        identity = dict(part=row.part, context=row.context, origin=row.origin, date=row.date, observed=y, hours=row.clean_hours)
        predictions.append(identity | {"model": "Bayesian local level"} | score_prediction(mean, sd, zero[:, idx], y, rng=rng))
        for name, distribution in zip(("28-day half-life", "last 28 days"), base):
            predictions.append(identity | {"model": name} | score_distribution(distribution, y))
    return pd.DataFrame(predictions)


def validate(raw, output, *, pilot=False):
    """Check first eligible observations from each predeclared fortnightly forecast origin."""
    from pathlib import Path

    from .analysis_data import load_analysis_data
    from .basal_validation import controller_forecast_rows

    analysis = load_analysis_data(raw)
    source = controller_forecast_rows(analysis)
    rows = attach_origin_features(source, analysis)
    keys = ["part", "context"]
    predictors = [name for name in rows if name.startswith("prior_") or name in (
        "weekend", "days_since_site", "hours_since_workout", "workout_energy_24h", "previous_sleep_mode_share",
        "context_event_day", "hours_since_dinner_log", "profile_rate",
    )]
    stop = rows.date.max().normalize()
    origins = list(pd.date_range(stop - pd.Timedelta(days=HOLDOUT_DAYS), stop, freq="14D"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_folds = []
    for origin in origins[:1] if pilot else origins:
        train = rows[rows.date.lt(origin)]
        future = source[source.date.ge(origin) & source.date.lt(origin + pd.Timedelta(days=14))]
        test = future.groupby(keys, sort=False, as_index=False).head(1).copy()
        # Only one exposure outcome per context and segment goes into this fold. Features come from the forecast origin.
        test["outcome_date"] = test.date
        test["date"] = origin
        test = attach_origin_features(test, analysis)
        # The profile rate in force on the outcome day, kept to confirm the profile did not change after the origin.
        outcome_rate = test.profile_rate.to_numpy()
        test["previous_sleep_mode_share"] = pd.MultiIndex.from_frame(test[keys]).map(
            train.groupby(keys).sleep_share.last()
        ).to_numpy()
        # The programmed rate must be knowable at the origin. Resolve from the last active pump in the record before it.
        pump_rows = analysis.merged.tandem_data.basal.loc[lambda x: x.start_timestamp.lt(origin)]
        from .pump_profiles import pump_key

        if pump_rows.empty:
            raise ValueError(f"No pump basal record before forecast origin {origin.date()}")
        active_pump = pump_key(pump_rows.sort_values("start_timestamp").pump_serial.iloc[-1])
        profile_id = analysis.history.active_profile(active_pump, origin)
        parts, _ = analysis.day_parts()
        by_part = {part.name: part.start_minutes for part in parts}
        for i, row in test.iterrows():
            programmed = analysis.history.segment_in_force(active_pump, profile_id, origin, minute_of_day=by_part[row.part])
            if programmed is None:
                test.loc[i, "profile_rate"] = np.nan
            else:
                test.loc[i, "profile_rate"] = programmed["basal_units_per_hour"]
        test["origin"] = origin
        test["date"] = test.pop("outcome_date")
        # Hold the profile fixed for the comparison. An intervening edit makes this forecast conditional on the wrong exposure.
        test = test.loc[np.isclose(test.profile_rate, outcome_rate, atol=0.00051)].copy()
        test.to_csv(output / f"{origin.date()}-holdout.csv", index=False)
        scored = predict_at_origin(train, test, predictors, output_dir=output / f"{origin.date()}")
        all_folds.append(scored)
    pd.concat(all_folds, ignore_index=True).to_csv(output / "paired_scores.csv", index=False)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", action="store_true", help="Run the first forecast origin only, with the production sampler")
    args = parser.parse_args()
    validate(args.raw, args.output, pilot=args.pilot)
