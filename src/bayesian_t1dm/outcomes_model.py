"""Hour-day outcomes model: how many readings of an hour are in range (or below 70, or below 54), given the settings in force and the context.

One row per clock hour of every day (``hour_day.build_hour_day_table``). For hour ``i`` of day ``d`` the
count ``y_i`` of readings in the outcome out of ``n_i`` valid readings is beta-binomial (readings within an hour are
correlated, so the counts are over-dispersed relative to a binomial):

    y_i ~ BetaBinomial(n_i, mu_i * phi, (1 - mu_i) * phi)
    logit(mu_i) = alpha + sigma_h * z_h[hour_i] + sigma_d * z_d[day_i] + x_i . beta

``z_h`` (24 clock-hour effects, constrained to sum to zero so they do not trade off with ``alpha``) and ``z_d`` (one
effect per day) are standard normal and scaled by ``sigma_h`` and ``sigma_d`` (non-centered), and ``x_i`` holds the
standardized settings in force and context. The settings enter on the log scale
(I:C, correction factor, basal) or per 10 mg/dL (target), so ``beta`` is per standard deviation of the log setting.

Claims stay predictive and associational. With hour and day effects in the model, ``beta`` for a setting is
identified from how its changes over time line up with the outcome in the same clock hour, not from a comparison between
hours; settings and context change together, so a coefficient is not the effect of changing that setting. Bolus,
insulin on board and delivered basal are left out: they lie on the path from the settings to glucose.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .hour_day import SLOTS_PER_HOUR
from scipy.special import expit

from .sampler_health import ESS_LIMIT, R_HAT_LIMIT, FitDiagnostics, extract_fit_diagnostics, sampler_gate, stack_draws

LOGGER = logging.getLogger(__name__)

OUTCOME_COLUMNS = {"tir": "n_in_range", "below70": "n_below_70", "below54": "n_below_54"}
OUTCOME_PRIOR_MEAN = {"tir": 0.70, "below70": 0.03, "below54": 0.005}
COVARIATES = ("log_icr", "log_cf", "log_basal", "target_10", "log_carbs_3h", "exercise_share", "sleep_mode_share", "site_age_capped")
SITE_AGE_CAP_DAYS = 5.0
SEED = 20261119
TARGET_ACCEPT = 0.95
MAX_TREEDEPTH = 10
VIF_CONCERN = 5.0
VIF_STOP = 10.0


@dataclass(frozen=True)
class OutcomeDesign:
    """Arrays for one outcome: the counts, the hour and day index of each row and the standardized covariates."""

    outcome: str
    y: np.ndarray
    n: np.ndarray
    hour: np.ndarray
    day: np.ndarray
    x: np.ndarray
    center: pd.Series
    scale: pd.Series
    days: pd.DatetimeIndex

    @property
    def rows(self) -> int:
        return len(self.y)


def covariate_frame(table: pd.DataFrame) -> pd.DataFrame:
    """The covariates of the hour-day table on their modelling scale (not yet standardized)."""
    return pd.DataFrame({
        "log_icr": np.log(table["icr"]),
        "log_cf": np.log(table["cf"]),
        "log_basal": np.log(table["basal"]),
        "target_10": table["target"] / 10.0,
        "log_carbs_3h": np.log1p(table["carbs_prev3h_g"]),
        "exercise_share": table["exercise_share"],
        "sleep_mode_share": table["sleep_mode_share"],
        "site_age_capped": table["site_age_days"].clip(upper=SITE_AGE_CAP_DAYS),
    }, index=table.index)


def _usable_mask(table: pd.DataFrame, covariates: pd.DataFrame) -> pd.Series:
    return (table["n_valid"] > 0) & (table["settings_known_slots"] == SLOTS_PER_HOUR) & ~table["excluded_day"] & np.isfinite(covariates).all(axis=1)


def usable_rows(table: pd.DataFrame) -> pd.Series:
    """Hours with readings, every setting known for all twelve moments, every covariate present and not on a context-event day."""
    return _usable_mask(table, covariate_frame(table))


def design_flow(table: pd.DataFrame) -> pd.DataFrame:
    """Rows left after each exclusion, for the diagnostics workbook."""
    covariates = covariate_frame(table)
    steps = [
        ("clock hours in the record", pd.Series(True, index=table.index)),
        ("with at least one reading", table["n_valid"] > 0),
        ("with every setting known at all twelve moments", table["settings_known_slots"] == SLOTS_PER_HOUR),
        ("not on a context-event day", ~table["excluded_day"]),
        ("with every covariate present", np.isfinite(covariates).all(axis=1)),
    ]
    keep = pd.Series(True, index=table.index)
    rows = []
    for name, condition in steps:
        keep &= condition
        rows.append((name, int(keep.sum())))
    return pd.DataFrame(rows, columns=["stage", "rows"])


def build_design(table: pd.DataFrame, *, outcome: str, center: pd.Series | None = None, scale: pd.Series | None = None) -> OutcomeDesign:
    """Design arrays from the usable rows of ``table``.

    Covariates are standardized on these rows unless ``center`` and ``scale`` (from the training rows) are given.
    """
    if outcome not in OUTCOME_COLUMNS:
        raise ValueError(f"Unknown outcome {outcome!r}; choose from {sorted(OUTCOME_COLUMNS)}")
    covariates = covariate_frame(table)
    keep = _usable_mask(table, covariates)
    rows = table.loc[keep].reset_index(drop=True)
    if rows.empty:
        raise ValueError("No usable hour-day rows")
    covariates = covariates.loc[keep].reset_index(drop=True)
    center = covariates.mean() if center is None else center
    scale = covariates.std(ddof=0).replace(0, 1.0) if scale is None else scale
    days = pd.DatetimeIndex(sorted(rows["date"].unique()))
    day = days.get_indexer(pd.DatetimeIndex(rows["date"]))
    return OutcomeDesign(
        outcome=outcome, y=rows[OUTCOME_COLUMNS[outcome]].to_numpy(dtype=int), n=rows["n_valid"].to_numpy(dtype=int),
        hour=rows["hour"].to_numpy(dtype=int), day=day, x=((covariates - center) / scale).to_numpy(),
        center=center, scale=scale, days=days,
    )


def collinearity(design: OutcomeDesign) -> pd.DataFrame:
    """Variance inflation factors of the covariates, raw and after removing each clock hour's mean; a covariate that does not vary within clock hours has no within-hour value (not assessed)."""
    out = {}
    hours = pd.Series(design.hour)
    frame = pd.DataFrame(design.x, columns=list(COVARIATES))
    demeaned = frame - frame.groupby(hours.to_numpy()).transform("mean")
    for label, values in (("vif_raw", frame), ("vif_within_hour", demeaned)):
        varying = values.std(ddof=0) > 1e-12
        vif = pd.Series(np.nan, index=values.columns)
        vif[varying] = np.diag(np.linalg.inv(np.corrcoef(values.loc[:, varying].to_numpy(), rowvar=False)))
        out[label] = vif.to_numpy()
    table = pd.DataFrame(out, index=list(COVARIATES)).rename_axis("covariate").reset_index()
    values = table[["vif_raw", "vif_within_hour"]]
    table["status"] = np.where(values.max(axis=1) >= VIF_CONCERN, "concern", np.where(values.isna().any(axis=1), "not assessed", "no concern"))
    table["condition_number"] = float(np.linalg.cond(design.x))
    return table


def build_model(design: OutcomeDesign):
    """The PyMC model of the module docstring."""
    import pymc as pm
    import pytensor.tensor as pt

    with pm.Model() as model:
        logit_mean = float(np.log(OUTCOME_PRIOR_MEAN[design.outcome] / (1 - OUTCOME_PRIOR_MEAN[design.outcome])))
        alpha = pm.Normal("alpha", mu=logit_mean, sigma=1.0)
        beta = pm.Normal("beta", mu=0.0, sigma=0.5, shape=design.x.shape[1])
        hour_scale = pm.HalfNormal("hour_scale", sigma=1.0)
        day_scale = pm.HalfNormal("day_scale", sigma=0.5)
        z_hour = pm.ZeroSumNormal("z_hour", sigma=1.0, shape=24)
        z_day = pm.Normal("z_day", 0.0, 1.0, shape=len(design.days))
        log_phi = pm.Normal("log_phi", 0.0, 1.5)
        eta = alpha + hour_scale * z_hour[design.hour] + day_scale * z_day[design.day] + pt.dot(design.x, beta)
        mu = pm.math.invlogit(eta)
        phi = pt.exp(log_phi)
        pm.BetaBinomial("y", n=design.n, alpha=mu * phi, beta=(1 - mu) * phi, observed=design.y)
    return model


def prior_predictive(design: OutcomeDesign, *, samples: int = 200) -> dict[str, float]:
    """Prior predictive check: the 5th and 95th percentiles, over prior draws, of the overall share of readings in the outcome."""
    import pymc as pm

    with build_model(design):
        draws = pm.sample_prior_predictive(draws=samples, random_seed=SEED, var_names=["y"]).prior_predictive["y"].to_numpy().reshape(samples, -1)
    share = draws.sum(axis=1) / design.n.sum()
    return {"overall_share_5th": float(np.percentile(share, 5)), "overall_share_95th": float(np.percentile(share, 95)), "prior_mean": OUTCOME_PRIOR_MEAN[design.outcome]}


def fit(design: OutcomeDesign, *, draws: int, tune: int, chains: int, target_accept: float = TARGET_ACCEPT) -> tuple[object, FitDiagnostics, dict[str, float]]:
    """Sample the model; returns the draws, the sampler diagnostics and the timing (model build, compile of the gradient, sampling) in seconds."""
    import pymc as pm

    timing: dict[str, float] = {}
    started = time.perf_counter()
    model = build_model(design)
    timing["build_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    model.compile_dlogp()(model.initial_point())
    timing["compile_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=chains, random_seed=SEED, target_accept=target_accept,
                          nuts={"max_treedepth": MAX_TREEDEPTH}, progressbar=False)
    timing["sample_seconds"] = time.perf_counter() - started
    diagnostics = extract_fit_diagnostics(idata, draws=draws, tune=tune, chains=chains, target_accept=target_accept,
                                          max_treedepth=MAX_TREEDEPTH, wall_time_seconds=timing["sample_seconds"])
    return idata, diagnostics, timing


def _posterior_sample(idata, draws: int, seed: int) -> np.ndarray:
    """Sorted indices of ``draws`` posterior samples (chains x draws stacked) picked without replacement."""
    total = stack_draws(idata, "alpha").shape[0]
    return np.sort(np.random.default_rng(seed).choice(total, size=min(draws, total), replace=False))


def predict_day_shares(idata, held_out: OutcomeDesign, *, draws: int = 500, seed: int = SEED) -> np.ndarray:
    """Predictive draws of each held-out day's share of readings in the outcome (draws x days), given that day's settings and context.

    Day effects are new standard-normal draws, so the spread covers a day the model has not seen. Counts are simulated from the beta-binomial of each hour.
    """
    rng = np.random.default_rng(seed)
    pick = _posterior_sample(idata, draws, seed)
    alpha, beta, z_hour, hour_scale, day_scale, log_phi = (stack_draws(idata, name)[pick] for name in ("alpha", "beta", "z_hour", "hour_scale", "day_scale", "log_phi"))
    n_draws = len(pick)
    eta = (alpha[:, None] + hour_scale[:, None] * z_hour[:, held_out.hour]
           + day_scale[:, None] * rng.normal(size=(n_draws, len(held_out.days)))[:, held_out.day] + beta @ held_out.x.T)
    mu = expit(eta)
    phi = np.exp(log_phi)[:, None]
    share = rng.beta(mu * phi, (1.0 - mu) * phi)
    counts = rng.binomial(held_out.n[None, :], share)
    by_day = np.zeros((n_draws, len(held_out.days)))
    np.add.at(by_day.T, held_out.day, counts.T)
    return by_day / np.bincount(held_out.day, weights=held_out.n, minlength=len(held_out.days))[None, :]


def posterior_eta(idata, design: OutcomeDesign, *, draws: int = 300, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Linear predictor of every training row for a thinned set of posterior draws (draws x rows), and the draw indices used."""
    pick = _posterior_sample(idata, draws, seed)
    alpha, beta, z_hour, z_day, hour_scale, day_scale = (stack_draws(idata, name)[pick] for name in ("alpha", "beta", "z_hour", "z_day", "hour_scale", "day_scale"))
    eta = alpha[:, None] + hour_scale[:, None] * z_hour[:, design.hour] + day_scale[:, None] * z_day[:, design.day] + beta @ design.x.T
    return eta, pick


def hourly_check(idata, design: OutcomeDesign, *, draws: int = 300) -> pd.DataFrame:
    """Share of readings in the outcome by clock hour: observed, and the posterior predictive 5th, 50th and 95th percentiles."""
    eta, pick = posterior_eta(idata, design, draws=draws)
    rng = np.random.default_rng(SEED + 1)
    mu = expit(eta)
    phi = np.exp(stack_draws(idata, "log_phi")[pick])[:, None]
    counts = rng.binomial(design.n[None, :], rng.beta(mu * phi, (1.0 - mu) * phi))
    readings = np.bincount(design.hour, weights=design.n, minlength=24)
    simulated = np.stack([np.bincount(design.hour, weights=row, minlength=24) for row in counts]) / readings
    low, median, high = np.percentile(simulated, [5, 50, 95], axis=0)
    observed = np.bincount(design.hour, weights=design.y, minlength=24) / readings
    return pd.DataFrame({"hour": np.arange(24), "readings": readings.astype(int), "observed": observed, "predicted_5th": low, "predicted_median": median, "predicted_95th": high,
                         "inside_90_interval": (observed >= low) & (observed <= high)})


def effects_table(idata, design: OutcomeDesign, *, draws: int = 300) -> pd.DataFrame:
    """Each covariate's coefficient (log odds per standard deviation) and its change in the share of readings in the outcome, in percentage points per standard deviation."""
    eta, pick = posterior_eta(idata, design, draws=draws)
    mu = expit(eta)
    slope = (mu * (1.0 - mu)).mean(axis=1)
    beta = stack_draws(idata, "beta")[pick]
    points = 100.0 * beta * slope[:, None]
    rows = []
    for index, name in enumerate(COVARIATES):
        low, high = np.percentile(beta[:, index], [5, 95])
        low_points, high_points = np.percentile(points[:, index], [5, 95])
        rows.append({"covariate": name, "sd_of_covariate": float(design.scale[name]), "coefficient": float(beta[:, index].mean()), "coefficient_5th": float(low), "coefficient_95th": float(high),
                     "points_per_sd": float(points[:, index].mean()), "points_5th": float(low_points), "points_95th": float(high_points),
                     "probability_negative": float((beta[:, index] < 0).mean())})
    return pd.DataFrame(rows)


def scale_summary(idata) -> pd.DataFrame:
    """Posterior mean and 90% interval of the hour and day scales and the dispersion."""
    rows = []
    for name in ("hour_scale", "day_scale"):
        values = stack_draws(idata, name)
        rows.append((name, float(values.mean()), *np.percentile(values, [5, 95])))
    phi = np.exp(stack_draws(idata, "log_phi"))
    rows.append(("phi", float(phi.mean()), *np.percentile(phi, [5, 95])))
    return pd.DataFrame(rows, columns=["parameter", "mean", "5th", "95th"])


def daily_shares(design: OutcomeDesign) -> np.ndarray:
    """Share of readings in the outcome of each day of a design (days with usable hours only)."""
    return np.bincount(design.day, weights=design.y, minlength=len(design.days)) / np.bincount(design.day, weights=design.n, minlength=len(design.days))


def rolling_origin(table: pd.DataFrame, *, outcome: str, origins: list[pd.Timestamp], span_days: int = 14, draws: int, tune: int, chains: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on the days before each origin and score the next ``span_days`` days' daily share against the last-28-days baseline.

    The model predicts each held-out day from that day's actual settings and context; the baseline draws one day at random from the 28 days before the origin and ignores the settings. Both are scored by CRPS and by whether the 90% interval holds the observed share. Every fit must pass the sampler gate. An origin whose window has no usable day is not scored and is returned in the second table with the reason.
    """
    from .replay import crps

    rows, skipped = [], []
    for origin in origins:
        train_table = table[table["date"] < origin]
        held_table = table[(table["date"] >= origin) & (table["date"] < origin + pd.Timedelta(days=span_days))]
        if not usable_rows(held_table).any():
            LOGGER.warning("rolling origin %s skipped: no usable day in the next %d days", origin.date(), span_days)
            skipped.append({"origin": origin.date(), "reason": f"no usable day in the {span_days} days from the origin (context-event days, unknown settings or no readings)"})
            continue
        train = build_design(train_table, outcome=outcome)
        held = build_design(held_table, outcome=outcome, center=train.center, scale=train.scale)
        idata, diagnostics, _ = fit(train, draws=draws, tune=tune, chains=chains)
        if not sampler_gate(diagnostics):
            raise RuntimeError(f"Rolling-origin fit before {origin.date()} fails the sampler gate: {diagnostics}")
        model_draws = predict_day_shares(idata, held)
        recent = daily_shares(train)[train.days >= origin - pd.Timedelta(days=28)]
        if recent.size == 0:
            LOGGER.warning("rolling origin %s skipped: no usable training day in the 28 days before it", origin.date())
            skipped.append({"origin": origin.date(), "reason": "no usable training day in the 28 days before the origin for the baseline"})
            continue
        observed = daily_shares(held)
        rng = np.random.default_rng([SEED, int(origin.strftime("%Y%m%d"))])
        baseline = rng.choice(recent, size=2000)
        for position, date in enumerate(held.days):
            for name, predicted in (("hour-day model", model_draws[:, position]), ("last 28 days", baseline)):
                low, high = np.percentile(predicted, [5, 95])
                rows.append({"origin": origin.date(), "date": date.date(), "predictor": name, "observed": observed[position], "predicted_mean": float(predicted.mean()),
                             "lower_90": float(low), "upper_90": float(high), "crps": crps(predicted, observed[position]), "inside_90": bool(low <= observed[position] <= high)})
    return pd.DataFrame(rows), pd.DataFrame(skipped, columns=["origin", "reason"])


def fortnight_origins(table: pd.DataFrame, count: int, span_days: int = 14) -> list[pd.Timestamp]:
    """The first days of the last ``count`` consecutive ``span_days`` windows of the record."""
    end = table["date"].max() + pd.Timedelta(days=1)
    return [end - pd.Timedelta(days=span_days * k) for k in range(count, 0, -1)]


def main(argv: list[str] | None = None) -> int:
    from .analysis_data import load_analysis_data
    from .hour_day import build_hour_day_table
    from .outcomes_model_report import write_outputs
    from .paths import ProjectPaths

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", default=None, help="Raw data root (default: the Project Vault data/raw)")
    parser.add_argument("--output", default=None, help="Output folder (default: runtime/output/outcomes_model)")
    parser.add_argument("--outcome", default="tir", choices=sorted(OUTCOME_COLUMNS))
    parser.add_argument("--origins", type=int, default=4, help="Fortnightly rolling origins to score at the end (0 skips)")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args(argv)
    paths = ProjectPaths.from_root(Path.cwd())
    raw = args.raw or str(paths.cloud_raw)
    output = Path(args.output or paths.output_outcomes_model)
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(output / f"outcomes_model_run_{run_id}.log")])
    started = time.time()
    LOGGER.info("outcomes model started run_id=%s outcome=%s raw=%s", run_id, args.outcome, raw)
    table = build_hour_day_table(load_analysis_data(raw))
    design = build_design(table, outcome=args.outcome)
    LOGGER.info("design: %d rows, %d days\n%s", design.rows, len(design.days), design_flow(table).to_string(index=False))
    vif = collinearity(design)
    LOGGER.info("collinearity\n%s", vif.to_string(index=False))
    worst = float(np.max(vif[["vif_raw", "vif_within_hour"]].to_numpy()))
    if not np.isfinite(worst) or worst >= VIF_STOP:
        raise RuntimeError("Covariates are too collinear, or one does not vary within clock hours, to fit (VIF >= 10 or undefined); see the log")
    prior = prior_predictive(design)
    LOGGER.info("prior predictive %s", json.dumps(prior))
    idata, diagnostics, timing = fit(design, draws=args.draws, tune=args.tune, chains=args.chains)
    LOGGER.info("timing %s", json.dumps(timing))
    LOGGER.info("diagnostics %s", diagnostics)
    (output / f"outcomes_model_fit_{run_id}.json").write_text(json.dumps({"diagnostics": diagnostics.__dict__, "timing": timing}, indent=2))
    if not sampler_gate(diagnostics):
        raise RuntimeError(f"The fit fails the sampler gate (R-hat < {R_HAT_LIMIT}, ESS > {ESS_LIMIT}, no divergences): {diagnostics}")
    scores, skipped = rolling_origin(table, outcome=args.outcome, origins=fortnight_origins(table, args.origins), draws=args.draws, tune=args.tune, chains=args.chains) if args.origins else (None, None)
    write_outputs(outcome=args.outcome, table=table, design=design, idata=idata, diagnostics=diagnostics, timing=timing, vif=vif, prior=prior, origin_scores=scores, skipped_origins=skipped,
                  draws=args.draws, tune=args.tune, chains=args.chains, output=output, run_id=run_id, raw=raw, seconds=time.time() - started)
    LOGGER.info("outcomes model finished in %.1f s", time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
