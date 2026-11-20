"""Evaluation harness: how well each predictor forecasts the next 14 days' share of readings in range, scored at weekly origins.

The question. At an origin ``t0`` (the first forecast day) a predictor sees everything recorded before ``t0 00:00`` and
returns a predictive distribution (4,000 draws) of the share of valid CGM readings in the outcome (time in range by default)
over ``t0`` to ``t0 + 13``. The predictor is scored by CRPS in percentage points. A candidate is better than the reference
only if the 90% circular-block-bootstrap interval of its paired CRPS difference lies entirely below zero (``win_test``).

Origins. Evaluation origins are weekly, ``t0_k = L - 13 - 7k`` for ``k = 0..51``, where ``L`` is the last CGM day (the last window
ends on ``L``). Development origins are every second week over 2025; every model choice is made on them and none on the
evaluation origins. A window holding fewer than 70% of its 4,032 possible readings is not scored and is listed with the reason.

Target. Sum of readings in the outcome over the sum of valid readings of every hour of the 14 days, so context-event days and hours
with unknown settings are in the target (the exclusion of a day is decided after its outcome is seen, and dropping it would bias the target).

Known at the origin. All glucose, the settings profile in force at the last moment of ``t0 - 1`` (frozen: it is repeated over the
window, whatever the pump does later), delivered insulin and the calendar. Not known: carbohydrate, exercise, sleep mode, site age and
context events inside the window. The harness hands each predictor a :class:`Known` that holds only rows dated before ``t0``.

Predictors.

- ``last 28 days`` (prespecified; decides the win), ``last 14 days`` and ``recency-weighted`` (exponential day weights, half-life 14
  days, over the last 90 days): the point is the readings-weighted share over the lookback; the spread comes from the same baseline's own
  past errors, ``logit(observed) - logit(point)`` at every earlier daily origin whose 14-day window ended before ``t0``. A win over the
  baseline therefore cannot come from the baseline having the wrong spread.
- ``ridge``: one row per earlier daily origin, response ``logit(observed)``, features known at that origin (logit share of the last 7,
  14, 28 and 90 days, the frozen settings relative to the trailing 28-day total daily dose, month); ridge penalty by blocked 5-fold
  cross-validation in time; spread from the last 120 out-of-fold residuals. A comparator, not a candidate: it says whether a model
  built on the settings adds anything over a recency regression with the same inputs.
- ``hour-day model``: the hour-day outcomes model refitted on the rows before every origin (its training rule unchanged). Its
  concurrent covariates are not known at the origin, so each predictive draw resamples 14 whole days of carbohydrate, exercise, sleep-mode
  and site-age exposure from the 28 days before ``t0``; its settings are the frozen profile; hourly reading counts are taken from the
  realised window (the count of readings is not glucose information). Variants that share the same fit and never gate:
  ``hour-day model (planned settings)`` (the settings actually in force), ``hour-day model (settings pinned)`` (every setting at the
  training centre, so the settings terms drop out) and ``hour-day model (trailing-mean counts)`` (the mean reading counts of the
  28 days before ``t0`` in place of the realised counts).
- ``last 365 days`` and ``all history``: means over a year and over every earlier day. Added after the first results as comparators of the
  model's level, which has no drift term; they never gate.

``python -m bayesian_t1dm.forecast_harness`` runs it on the real record.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from . import outcomes_model as om
from .hour_day import HOURS_PER_DAY, SETTING_FIELDS, SLOTS_PER_HOUR
from .sampler_health import FitDiagnostics, sampler_gate
from .win_test import crps_points

LOGGER = logging.getLogger(__name__)

SEED = om.SEED
WINDOW_DAYS = 14
READINGS_PER_DAY = 288
MIN_WINDOW_COVERAGE = 0.70
EVALUATION_ORIGIN_COUNT = 52
EVALUATION_STEP_DAYS = 7
DEVELOPMENT_START = pd.Timestamp("2025-01-07")
DEVELOPMENT_END = pd.Timestamp("2025-11-04")
DEVELOPMENT_STEP_DAYS = 14
DRAWS = 4000
LOGIT_CLIP = 1e-3
RECENCY_HALF_LIFE_DAYS = 14
RECENCY_LOOKBACK_DAYS = 90
MIN_RESIDUALS = 30
TDD_DAYS = 28
TDD_MIN_COVERAGE = 0.70
TDD_MIN_DAYS = 14
EXPOSURE_POOL_DAYS = 28
RIDGE_LOOKBACKS = (7, 14, 28, 90)
RIDGE_PENALTIES = tuple(float(value) for value in np.logspace(-2, 3, 11))
RIDGE_FOLDS = 5
RIDGE_RESIDUALS = 120
RIDGE_MIN_ROWS = 150
SETTING_COLUMNS = ("icr", "cf", "basal", "target")
MODEL = "hour-day model"
# Model variants share the model's fit: the settings actually in force (planned), the settings pinned at the training centre, and trailing mean reading counts in place of the realised ones.
MODEL_VARIANTS = {MODEL: "frozen", "hour-day model (planned settings)": "planned", "hour-day model (settings pinned)": "pinned", "hour-day model (trailing-mean counts)": "trailing counts"}
BASELINES = ("last 28 days", "last 14 days", "recency-weighted", "last 365 days", "all history")
# The order fixes each predictor's random stream: new predictors are appended.
PREDICTORS = ("last 28 days", "last 14 days", "recency-weighted", "ridge", *MODEL_VARIANTS, "last 365 days", "all history")
REFERENCE = "last 28 days"
LOOKBACK_DAYS = {"last 28 days": 28, "last 14 days": 14, "last 365 days": 365}
ALL_HISTORY_MIN_DAYS = 28
COUNT_DAYS = 28
NO_SETTINGS_CHANGE = 1e-9


class NotScored(Exception):
    """A predictor cannot forecast an origin; the reason is listed in the diagnostics (the origin is excluded, never filled in)."""


# ---------------------------------------------------------------------------
# Origins, target, windows
# ---------------------------------------------------------------------------

def evaluation_origins(last_day: pd.Timestamp, *, count: int = EVALUATION_ORIGIN_COUNT, step: int = EVALUATION_STEP_DAYS) -> list[pd.Timestamp]:
    """Weekly origins ``last_day - 13 - step * k`` for ``k = 0..count-1``, in time order (the last window ends on ``last_day``)."""
    last = pd.Timestamp(last_day).normalize()
    return [last - pd.Timedelta(days=WINDOW_DAYS - 1 + step * k) for k in range(count - 1, -1, -1)]


def development_origins(start: pd.Timestamp = DEVELOPMENT_START, end: pd.Timestamp = DEVELOPMENT_END, step: int = DEVELOPMENT_STEP_DAYS) -> list[pd.Timestamp]:
    """Every ``step`` days from ``start`` while not after ``end``."""
    return list(pd.date_range(start, end, freq=f"{step}D"))


def check_origins_in_record(origins: list[pd.Timestamp], first_day: pd.Timestamp, last_day: pd.Timestamp) -> None:
    """Every window of every origin must lie inside the CGM record."""
    if not origins:
        raise ValueError("No origins")
    if min(origins) < first_day or max(origins) + pd.Timedelta(days=WINDOW_DAYS - 1) > last_day:
        raise ValueError(f"Origins {min(origins).date()} to {max(origins).date()} do not all fit inside the record {first_day.date()} to {last_day.date()}")


def daily_counts(table: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Valid readings ``n`` and readings in the outcome ``y`` of each calendar day of the hour-day table (every row counts: no exclusions)."""
    column = om.OUTCOME_COLUMNS[outcome]
    daily = table.groupby("date").agg(n=("n_valid", "sum"), y=(column, "sum"))
    days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(days, fill_value=0).rename_axis("date").astype(float)


def window_target(table: pd.DataFrame, origin: pd.Timestamp, outcome: str) -> tuple[float, float]:
    """Readings-weighted share of the outcome over the 14 days from ``origin``, and the window's coverage of its possible readings."""
    rows = table[(table["date"] >= origin) & (table["date"] < origin + pd.Timedelta(days=WINDOW_DAYS))]
    readings = float(rows["n_valid"].sum())
    coverage = readings / (WINDOW_DAYS * READINGS_PER_DAY)
    return (float(rows[om.OUTCOME_COLUMNS[outcome]].sum()) / readings if readings > 0 else float("nan")), coverage


@dataclass(frozen=True)
class Window:
    """The 14 forecast days as far as an origin may use them: dates and reading counts per clock hour, and the settings actually in force."""

    origin: pd.Timestamp
    n_valid: np.ndarray  # 14 x 24
    actual_settings: np.ndarray  # 14 x 24 x 4 (icr, cf, basal, target); NaN where the archive cannot show the setting


def window_for(table: pd.DataFrame, origin: pd.Timestamp) -> Window:
    """The realised window's hourly reading counts and the settings in force (settings are known in advance only as a plan)."""
    rows = table[(table["date"] >= origin) & (table["date"] < origin + pd.Timedelta(days=WINDOW_DAYS))].sort_values(["date", "hour"])
    if len(rows) != WINDOW_DAYS * HOURS_PER_DAY:
        raise ValueError(f"The window at {origin.date()} holds {len(rows)} clock hours, not {WINDOW_DAYS * HOURS_PER_DAY}")
    settings = rows[list(SETTING_COLUMNS)].to_numpy(dtype=float).copy()
    settings[(rows["settings_known_slots"] != SLOTS_PER_HOUR).to_numpy()] = np.nan
    return Window(origin=origin, n_valid=rows["n_valid"].to_numpy(dtype=float).reshape(WINDOW_DAYS, HOURS_PER_DAY), actual_settings=settings.reshape(WINDOW_DAYS, HOURS_PER_DAY, 4))


# ---------------------------------------------------------------------------
# What is known at an origin
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SettingsProfiles:
    """The 24 x 4 hourly settings profile frozen at the last moment of the day before each date (icr, cf, basal, target)."""

    dates: pd.DatetimeIndex
    values: np.ndarray  # days x 24 x 4, NaN where any of the hour's twelve moments is unknown

    def upto(self, date: pd.Timestamp) -> "SettingsProfiles":
        """The profiles of every date up to and including ``date`` (a profile frozen at ``t0`` uses only what preceded ``t0``)."""
        keep = self.dates <= date
        return SettingsProfiles(self.dates[keep], self.values[keep])

    def at(self, date: pd.Timestamp) -> np.ndarray:
        position = self.dates.get_loc(date)
        return self.values[position]


@dataclass(frozen=True)
class Known:
    """Everything a predictor may use at an origin: rows dated before ``origin`` and profiles frozen at or before it."""

    origin: pd.Timestamp
    outcome: str
    table: pd.DataFrame
    daily: pd.DataFrame
    tdd: pd.Series
    profiles: SettingsProfiles


def known_at(origin: pd.Timestamp, outcome: str, table: pd.DataFrame, daily: pd.DataFrame, tdd: pd.Series, profiles: SettingsProfiles) -> Known:
    """Cut every source at ``origin 00:00``; nothing dated on or after it is passed on."""
    known = Known(origin=origin, outcome=outcome, table=table[table["date"] < origin], daily=daily[daily.index < origin], tdd=tdd[tdd.index < origin], profiles=profiles.upto(origin))
    if known.table["date"].max() >= origin or known.daily.index.max() >= origin or known.tdd.index.max() >= origin or known.profiles.dates.max() > origin:
        raise AssertionError(f"Rows from the origin {origin.date()} or later reached the predictors")
    return known


def trailing_tdd(tdd: pd.Series) -> pd.Series:
    """Mean total daily dose over the 28 days before each day (the day itself excluded), when at least 14 of them have a dose."""
    return tdd.shift(1).rolling(TDD_DAYS, min_periods=TDD_MIN_DAYS).mean()


def daily_insulin(grid: pd.DataFrame, basal: pd.DataFrame) -> pd.Series:
    """Delivered insulin per day (units): boluses on the analysis grid plus the pump's delivered basal (rate times duration of each logged interval, assigned to the day it starts).

    Under Control-IQ the logged basal intervals are the automated delivery. A day whose basal intervals cover under 70% of 24 hours has no value.
    """
    bolus = grid.groupby(pd.to_datetime(grid["timestamp"]).dt.normalize())["bolus_units"].sum()
    start = pd.to_datetime(basal["start_timestamp"])
    hours = (pd.to_datetime(basal["end_timestamp"]) - start).dt.total_seconds() / 3600.0
    by_day = pd.DataFrame({"day": start.dt.normalize(), "units": basal["basal_units_per_hour"] * hours, "hours": hours}).groupby("day").sum()
    days = pd.date_range(min(bolus.index.min(), by_day.index.min()), max(bolus.index.max(), by_day.index.max()), freq="D")
    total = bolus.reindex(days, fill_value=0.0) + by_day["units"].reindex(days, fill_value=0.0)
    covered = by_day["hours"].reindex(days, fill_value=0.0) / 24.0 >= TDD_MIN_COVERAGE
    return total.where(covered).rename_axis("date")


def frozen_profiles(history, table: pd.DataFrame) -> SettingsProfiles:
    """For each date, the profile in force at 23:55 of the day before, read at every clock hour's twelve moments (the hour-day settings logic, frozen).

    The pump is the one whose readings fill the last hour of the previous day. An hour is NaN when any of its twelve moments is unknown to the archive.
    """
    last_hour_pump = table[table["hour"] == HOURS_PER_DAY - 1].set_index("date")["pump_serial"]
    dates = pd.DatetimeIndex(sorted(table["date"].unique()))
    values = np.full((len(dates), HOURS_PER_DAY, len(SETTING_COLUMNS)), np.nan)
    fields = list(SETTING_FIELDS.values())
    for position, date in enumerate(dates):
        previous = date - pd.Timedelta(days=1)
        if previous not in last_hour_pump.index:
            continue
        pump = last_hour_pump[previous]
        moment = (date - pd.Timedelta(minutes=5)).to_pydatetime()
        profile = history.active_profile(pump, moment)
        if profile is None:
            continue
        for hour in range(HOURS_PER_DAY):
            slots = [history.segment_in_force(pump, profile, moment, minute_of_day=hour * 60 + 5 * slot) for slot in range(SLOTS_PER_HOUR)]
            if any(segment is None for segment in slots):
                continue
            values[position, hour] = [np.mean([segment[field] for segment in slots]) for field in fields]
    return SettingsProfiles(dates, values)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _clipped_logit(share: np.ndarray | float) -> np.ndarray | float:
    return logit(np.clip(share, LOGIT_CLIP, 1.0 - LOGIT_CLIP))


def _window_outcomes(y: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Share and coverage of the 14-day window starting at each position ``s`` whose window lies inside the series (NaN share when coverage is under 70%)."""
    count = len(y) - WINDOW_DAYS + 1
    if count <= 0:
        return np.array([]), np.array([])
    sum_y = np.concatenate([[0.0], np.cumsum(y)])
    sum_n = np.concatenate([[0.0], np.cumsum(n)])
    window_y = sum_y[WINDOW_DAYS:WINDOW_DAYS + count] - sum_y[:count]
    window_n = sum_n[WINDOW_DAYS:WINDOW_DAYS + count] - sum_n[:count]
    coverage = window_n / (WINDOW_DAYS * READINGS_PER_DAY)
    share = np.where(coverage >= MIN_WINDOW_COVERAGE, window_y / np.maximum(window_n, 1.0), np.nan)
    return share, coverage


def lookback_points(y: np.ndarray, n: np.ndarray, kind: str) -> np.ndarray:
    """Point forecast for a window starting at every position ``s = 0..len(y)`` from the days before ``s`` only; NaN where the lookback is not fully inside the series or holds no readings."""
    length = len(y)
    positions = np.arange(length + 1)
    if kind == "all history":
        sum_y = np.concatenate([[0.0], np.cumsum(y)])
        sum_n = np.concatenate([[0.0], np.cumsum(n)])
        return np.where((positions >= ALL_HISTORY_MIN_DAYS) & (sum_n[positions] > 0), sum_y[positions] / np.maximum(sum_n[positions], 1.0), np.nan)
    if kind in LOOKBACK_DAYS:
        days = LOOKBACK_DAYS[kind]
        sum_y = np.concatenate([[0.0], np.cumsum(y)])
        sum_n = np.concatenate([[0.0], np.cumsum(n)])
        low = positions - days
        num = sum_y[positions] - sum_y[np.maximum(low, 0)]
        den = sum_n[positions] - sum_n[np.maximum(low, 0)]
        return np.where((low >= 0) & (den > 0), num / np.maximum(den, 1.0), np.nan)
    if kind == "recency-weighted":
        weights = 0.5 ** (np.arange(RECENCY_LOOKBACK_DAYS) / RECENCY_HALF_LIFE_DAYS)
        weighted_y = np.concatenate([[0.0], np.convolve(y, weights)[:length]])
        weighted_n = np.concatenate([[0.0], np.convolve(n, weights)[:length]])
        return np.where((positions >= RECENCY_LOOKBACK_DAYS) & (weighted_n > 0), weighted_y / np.maximum(weighted_n, 1e-12), np.nan)
    raise ValueError(f"Unknown baseline {kind!r}")


def residual_pool(y: np.ndarray, n: np.ndarray, kind: str) -> np.ndarray:
    """Logit errors of the baseline at every earlier daily origin whose 14-day window ends inside the series (so before the origin that asks)."""
    points = lookback_points(y, n, kind)
    share, _ = _window_outcomes(y, n)
    point = points[:len(share)]
    usable = np.isfinite(share) & np.isfinite(point)
    return np.asarray(_clipped_logit(share[usable]) - _clipped_logit(point[usable]))


def baseline_draws(known: Known, kind: str, rng: np.random.Generator, draws: int = DRAWS) -> np.ndarray:
    """Predictive draws of the baseline: its point at the origin shifted by errors resampled from its own past errors."""
    y, n = known.daily["y"].to_numpy(), known.daily["n"].to_numpy()
    point = lookback_points(y, n, kind)[len(y)]
    if not np.isfinite(point):
        raise NotScored(f"{kind}: no readings in the lookback before the origin or the lookback does not fit inside the record")
    pool = residual_pool(y, n, kind)
    if len(pool) < MIN_RESIDUALS:
        raise NotScored(f"{kind}: only {len(pool)} earlier windows to learn the spread from (at least {MIN_RESIDUALS} needed)")
    return expit(_clipped_logit(point) + rng.choice(pool, size=draws))


# ---------------------------------------------------------------------------
# Ridge comparator
# ---------------------------------------------------------------------------

RIDGE_FEATURES = tuple(f"logit_share_{days}d" for days in RIDGE_LOOKBACKS) + ("log_icr_x_tdd", "log_cf_x_tdd", "log_basal_fraction", "month_sin", "month_cos")


def ridge_features(known: Known) -> tuple[np.ndarray, np.ndarray]:
    """Features known at every daily position ``s = 0..len(daily)`` (rows) and the date of each: recency shares, the frozen settings relative to the trailing 28-day dose, and month."""
    y, n = known.daily["y"].to_numpy(), known.daily["n"].to_numpy()
    length = len(y)
    dates = pd.date_range(known.daily.index[0], periods=length + 1, freq="D")
    columns = []
    for days in RIDGE_LOOKBACKS:
        sum_y = np.concatenate([[0.0], np.cumsum(y)])
        sum_n = np.concatenate([[0.0], np.cumsum(n)])
        positions = np.arange(length + 1)
        low = positions - days
        den = sum_n[positions] - sum_n[np.maximum(low, 0)]
        share = np.where((low >= 0) & (den > 0), (sum_y[positions] - sum_y[np.maximum(low, 0)]) / np.maximum(den, 1.0), np.nan)
        columns.append(_clipped_logit(share))
    tdd28 = trailing_tdd(known.tdd.reindex(dates)).to_numpy()  # the origin's own (unknown) dose is NaN and never enters its trailing mean
    profile = np.full((length + 1, HOURS_PER_DAY, 4), np.nan)
    present = known.profiles.dates.intersection(dates)
    profile[dates.get_indexer(present)] = known.profiles.values[known.profiles.dates.get_indexer(present)]
    with np.errstate(invalid="ignore", divide="ignore"):
        columns.append(np.mean(np.log(profile[:, :, 0]), axis=1) + np.log(tdd28))
        columns.append(np.mean(np.log(profile[:, :, 1]), axis=1) + np.log(tdd28))
        columns.append(np.log(profile[:, :, 2].sum(axis=1) / tdd28))
    month = dates.month.to_numpy()
    columns.append(np.sin(2 * np.pi * month / 12))
    columns.append(np.cos(2 * np.pi * month / 12))
    return np.column_stack(columns), dates.to_numpy()


@dataclass(frozen=True)
class RidgeModel:
    """A fitted ridge regression on standardized features."""

    intercept: float
    coefficients: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    def predict(self, z: np.ndarray) -> np.ndarray:
        return self.intercept + ((z - self.mean) / self.scale) @ self.coefficients


def _ridge_fit(z: np.ndarray, target: np.ndarray, penalty: float) -> RidgeModel:
    """Ridge regression of ``target`` on the standardized columns of ``z`` (intercept unpenalized)."""
    mean = z.mean(axis=0)
    scale = np.where(z.std(axis=0) == 0, 1.0, z.std(axis=0))
    standardized = (z - mean) / scale
    coefficients = np.linalg.solve(standardized.T @ standardized + penalty * np.eye(z.shape[1]), standardized.T @ (target - target.mean()))
    return RidgeModel(float(target.mean()), coefficients, mean, scale)


def blocked_cross_validation(z: np.ndarray, target: np.ndarray, positions: np.ndarray) -> tuple[float, np.ndarray, pd.DataFrame]:
    """Penalty by blocked K-fold cross-validation in time, and the out-of-fold residuals at that penalty.

    Folds are contiguous in time. A test row's outcome (its 14-day window) must not reach a training row either through that row's own window
    (a training start within 13 days before the fold's first start) or through its features, which look back up to 90 days from the row's start
    (a training start up to 13 + 90 days after the fold's last start). Those training rows are left out of the fold's fit.
    """
    folds = np.array_split(np.arange(len(target)), RIDGE_FOLDS)
    errors = np.zeros((len(RIDGE_PENALTIES), len(target)))
    for fold in folds:
        train = ridge_training_mask(positions, fold)
        for index, penalty in enumerate(RIDGE_PENALTIES):
            errors[index, fold] = target[fold] - _ridge_fit(z[train], target[train], penalty).predict(z[fold])
    rmse = np.sqrt((errors ** 2).mean(axis=1))
    best = int(np.argmin(rmse))
    table = pd.DataFrame({"penalty": RIDGE_PENALTIES, "cv_rmse_logit": rmse})
    return RIDGE_PENALTIES[best], errors[best], table


def ridge_training_mask(positions: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Rows that may train a fold: none whose window overlaps a test window or whose features look back into the test windows' outcomes."""
    low = positions[fold[0]] - (WINDOW_DAYS - 1)
    high = positions[fold[-1]] + (WINDOW_DAYS - 1) + max(RIDGE_LOOKBACKS)
    return ~((positions >= low) & (positions <= high))


def ridge_draws(known: Known, rng: np.random.Generator, draws: int = DRAWS) -> tuple[np.ndarray, dict]:
    """Predictive draws of the window-level ridge and what was fitted (training rows, penalty, cross-validated error)."""
    features, dates = ridge_features(known)
    y, n = known.daily["y"].to_numpy(), known.daily["n"].to_numpy()
    share, _ = _window_outcomes(y, n)
    rows = np.flatnonzero(np.isfinite(share) & np.isfinite(features[:len(share)]).all(axis=1))
    if len(rows) < RIDGE_MIN_ROWS:
        raise NotScored(f"ridge: only {len(rows)} earlier windows have every feature (at least {RIDGE_MIN_ROWS} needed)")
    at_origin = features[-1]
    if not np.isfinite(at_origin).all():
        missing = [name for name, value in zip(RIDGE_FEATURES, at_origin) if not np.isfinite(value)]
        raise NotScored(f"ridge: feature(s) unknown at the origin: {', '.join(missing)}")
    target = np.asarray(_clipped_logit(share[rows]))
    z = features[rows]
    penalty, residuals, cv = blocked_cross_validation(z, target, rows)
    point = float(_ridge_fit(z, target, penalty).predict(at_origin[None, :])[0])
    pool = residuals[-RIDGE_RESIDUALS:]
    info = {"training_windows": int(len(rows)), "penalty": float(penalty), "cv_rmse_logit": float(cv["cv_rmse_logit"].min()), "residuals_used": int(len(pool))}
    return expit(point + rng.choice(pool, size=draws)), info


# ---------------------------------------------------------------------------
# The current hour-day model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPosterior:
    """The posterior draws the forecast needs (thinned to ``DRAWS``) with the standardization of the training rows, the sampler diagnostics and the training size."""

    origin: pd.Timestamp
    alpha: np.ndarray
    beta: np.ndarray
    z_hour: np.ndarray
    hour_scale: np.ndarray
    day_scale: np.ndarray
    log_phi: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    diagnostics: FitDiagnostics
    timing: dict
    rows: int
    days: int
    fingerprint: str
    model_hash: str


def design_fingerprint(design: om.OutcomeDesign) -> str:
    """Hash of the training arrays, so a stored posterior can be reused only for exactly the same training data."""
    digest = hashlib.sha256()
    for array in (design.y, design.n, design.hour, design.day, np.round(design.x, 10)):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def model_hash(draws: int, tune: int, chains: int, target_accept: float = om.TARGET_ACCEPT) -> str:
    """Hash of the model code (the source of ``outcomes_model``, which holds the priors, likelihood and sampler constants), the seed and the sampler settings: a stored fit is reused only when it is unchanged."""
    text = inspect.getsource(om) + f"|{om.SEED}|{target_accept}|{om.MAX_TREEDEPTH}|{draws}|{tune}|{chains}|{DRAWS}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fit_origin(table: pd.DataFrame, origin: pd.Timestamp, outcome: str, draws: int, tune: int, chains: int, target_accept: float = om.TARGET_ACCEPT) -> ModelPosterior:
    """Fit the outcomes model to the rows before ``origin``; the sampler gate must pass."""
    from .sampler_health import stack_draws

    train_table = table[table["date"] < origin]
    design = om.build_design(train_table, outcome=outcome)
    idata, diagnostics, timing = om.fit(design, draws=draws, tune=tune, chains=chains, target_accept=target_accept)
    if not sampler_gate(diagnostics):
        raise RuntimeError(f"The fit before {origin.date()} fails the sampler gate: {diagnostics}")
    pick = om._posterior_sample(idata, DRAWS, SEED + int(origin.strftime("%Y%m%d")))

    def take(name: str) -> np.ndarray:
        return stack_draws(idata, name)[pick]

    return ModelPosterior(origin=origin, alpha=take("alpha"), beta=take("beta"), z_hour=take("z_hour"), hour_scale=take("hour_scale"), day_scale=take("day_scale"), log_phi=take("log_phi"),
                          center=design.center.to_numpy(), scale=design.scale.to_numpy(), diagnostics=diagnostics, timing=timing, rows=design.rows, days=len(design.days), fingerprint=design_fingerprint(design),
                          model_hash=model_hash(draws, tune, chains, target_accept))


def _fit_worker(arguments: tuple) -> ModelPosterior:
    return fit_origin(*arguments)


def save_posterior(posterior: ModelPosterior, folder: Path) -> Path:
    """Store a fitted origin so a later run can reuse it when the training data, outcome and sampler settings are identical."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"posterior_{posterior.origin.date()}.npz"
    meta = json.dumps({"origin": str(posterior.origin.date()), "diagnostics": posterior.diagnostics.__dict__, "timing": posterior.timing, "rows": posterior.rows, "days": posterior.days, "fingerprint": posterior.fingerprint, "model_hash": posterior.model_hash})
    np.savez_compressed(path, alpha=posterior.alpha, beta=posterior.beta, z_hour=posterior.z_hour, hour_scale=posterior.hour_scale, day_scale=posterior.day_scale, log_phi=posterior.log_phi,
                        center=posterior.center, scale=posterior.scale, meta=np.array(meta))
    return path


def load_posterior(path: Path) -> ModelPosterior:
    stored = np.load(path, allow_pickle=False)
    meta = json.loads(str(stored["meta"]))
    return ModelPosterior(origin=pd.Timestamp(meta["origin"]), alpha=stored["alpha"], beta=stored["beta"], z_hour=stored["z_hour"], hour_scale=stored["hour_scale"], day_scale=stored["day_scale"],
                          log_phi=stored["log_phi"], center=stored["center"], scale=stored["scale"], diagnostics=FitDiagnostics(**meta["diagnostics"]), timing=meta["timing"], rows=meta["rows"], days=meta["days"],
                          fingerprint=meta["fingerprint"], model_hash=meta["model_hash"])


def fit_all_origins(table: pd.DataFrame, origins: list[pd.Timestamp], *, outcome: str, draws: int, tune: int, chains: int, workers: int, store: Path | None, target_accept: float = om.TARGET_ACCEPT) -> dict[pd.Timestamp, ModelPosterior]:
    """Refit the outcomes model at every origin (exact: no state carried over), several origins at once. A stored posterior is reused only when its training data and settings match."""
    posteriors: dict[pd.Timestamp, ModelPosterior] = {}
    pending = []
    for origin in origins:
        path = None if store is None else store / f"posterior_{origin.date()}.npz"
        if path is not None and path.exists():
            stored = load_posterior(path)
            design = om.build_design(table[table["date"] < origin], outcome=outcome)
            same = (stored.fingerprint == design_fingerprint(design) and stored.model_hash == model_hash(draws, tune, chains, target_accept)
                    and (stored.diagnostics.draws, stored.diagnostics.tune, stored.diagnostics.chains, stored.diagnostics.target_accept) == (draws, tune, chains, target_accept))
            if not same:
                raise RuntimeError(f"The stored posterior {path} was fitted to different data, model code or sampler settings; remove it to refit")
            posteriors[origin] = stored
            LOGGER.info("reused stored posterior for origin %s", origin.date())
        else:
            pending.append(origin)
    if pending:
        LOGGER.info("fitting %d origins, %d at a time", len(pending), workers)
        arguments = [(table[table["date"] < origin].copy(), origin, outcome, draws, tune, chains, target_accept) for origin in pending]
        failures = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(_fit_worker, argument): argument[1] for argument in arguments}
            for future in as_completed(futures):
                origin = futures[future]
                try:
                    posterior = future.result()
                except RuntimeError as problem:  # a fit that fails the sampler gate; the other origins finish and are stored before the run fails
                    LOGGER.error("fit %s failed: %s", origin.date(), problem)
                    failures.append(str(problem))
                    continue
                posteriors[origin] = posterior
                LOGGER.info("fit %s: rows %d, rhat %.4f, ess %.0f/%.0f, divergences %d, %.0f s", origin.date(), posterior.rows, posterior.diagnostics.rhat_max, posterior.diagnostics.ess_bulk_min,
                            posterior.diagnostics.ess_tail_min, posterior.diagnostics.divergences, posterior.timing["sample_seconds"])
                if store is not None:
                    save_posterior(posterior, store)
        if failures:
            raise RuntimeError(f"{len(failures)} fit(s) failed the sampler gate: {failures[0]}")
    return posteriors


def exposure_pool(known: Known) -> np.ndarray:
    """Whole days of exposure (days x 24 x 4: log carbohydrate of the last 3 h, exercise share, sleep-mode share, site age capped) from the 28 days before the origin that are complete."""
    recent = known.table[known.table["date"] >= known.origin - pd.Timedelta(days=EXPOSURE_POOL_DAYS)].sort_values(["date", "hour"])
    frame = pd.DataFrame({"date": recent["date"], "carbs": np.log1p(recent["carbs_prev3h_g"]), "exercise": recent["exercise_share"], "sleep": recent["sleep_mode_share"],
                          "site": recent["site_age_days"].clip(upper=om.SITE_AGE_CAP_DAYS)})
    days = []
    for _, day in frame.groupby("date"):
        values = day[["carbs", "exercise", "sleep", "site"]].to_numpy(dtype=float)
        if len(values) == HOURS_PER_DAY and np.isfinite(values).all():
            days.append(values)
    if not days:
        raise NotScored(f"hour-day model: no complete day of exposures in the {EXPOSURE_POOL_DAYS} days before the origin")
    return np.stack(days)


def settings_covariates(profile: np.ndarray) -> np.ndarray:
    """Model covariates (log I:C, log correction factor, log basal, target per 10 mg/dL) of an ``... x 4`` array of (icr, cf, basal, target)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([np.log(profile[..., 0]), np.log(profile[..., 1]), np.log(profile[..., 2]), profile[..., 3] / 10.0], axis=-1)


def window_mu(posterior: ModelPosterior, settings: np.ndarray, exposures: np.ndarray, n_valid: np.ndarray, day_noise: np.ndarray) -> np.ndarray:
    """Expected share of readings in the outcome of each of the 14 x 24 cells (draws x 336) given the model-scale ``settings`` (14 x 24 x 4), ``exposures`` (draws x 14 x 24 x 4) and ``day_noise`` (draws x 14 standard-normal day effects).

    Cells without readings (``n_valid`` of 0) take no settings or exposure; a setting unknown at a cell with readings is an error.
    """
    draws = len(posterior.alpha)
    hours = np.tile(np.arange(HOURS_PER_DAY), WINDOW_DAYS)
    present = n_valid.reshape(-1) > 0
    flat_settings = settings.reshape(-1, 4)
    if not np.isfinite(flat_settings[present]).all():
        raise NotScored("hour-day model: a setting is unknown at an hour of the window that has readings")
    centered = np.where(present[:, None], (flat_settings - posterior.center[:4]) / posterior.scale[:4], 0.0)
    exposure = np.where(present[None, :, None], (exposures.reshape(draws, -1, 4) - posterior.center[4:]) / posterior.scale[4:], 0.0)
    eta = (posterior.alpha[:, None] + posterior.hour_scale[:, None] * posterior.z_hour[:, hours]
           + posterior.day_scale[:, None] * day_noise[:, np.repeat(np.arange(WINDOW_DAYS), HOURS_PER_DAY)]
           + posterior.beta[:, :4] @ centered.T + np.einsum("dpk,dk->dp", exposure, posterior.beta[:, 4:]))
    return expit(eta)


def simulate_window_share(posterior: ModelPosterior, settings: np.ndarray, exposures: np.ndarray, n_valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Predictive draws of the window's share of readings in the outcome from the hour-day model.

    ``settings`` (14 x 24 x 4, model scale) are the same for every draw; ``exposures`` (draws x 14 x 24 x 4) are one resampled set of days per draw; ``n_valid`` (14 x 24) are the
    realised reading counts. Each draw uses its own posterior draw, a fresh day effect per forecast day, a beta draw of each hour's share and binomial counts.
    """
    draws = len(posterior.alpha)
    counts = n_valid.reshape(-1).astype(int)
    mu = window_mu(posterior, settings, exposures, n_valid, rng.normal(size=(draws, WINDOW_DAYS)))
    phi = np.exp(posterior.log_phi)[:, None]
    share = rng.beta(mu * phi, (1.0 - mu) * phi)
    hits = rng.binomial(counts[None, :], share)
    return hits.sum(axis=1) / counts.sum()


def trailing_counts(known: Known) -> np.ndarray:
    """Reading counts per clock hour known at the origin: the mean over the 28 days before it, rounded, the same for each of the 14 days (14 x 24)."""
    recent = known.table[known.table["date"] >= known.origin - pd.Timedelta(days=COUNT_DAYS)]
    return np.tile(np.rint(recent.groupby("hour")["n_valid"].mean().reindex(range(HOURS_PER_DAY)).fillna(0.0).to_numpy()), (WINDOW_DAYS, 1))


def _variant_inputs(known: Known, window: Window, variant: str, posterior: ModelPosterior | None) -> tuple[np.ndarray, np.ndarray]:
    """Settings on the model scale (14 x 24 x 4) and reading counts (14 x 24) of one model variant."""
    counts = trailing_counts(known) if variant == "trailing counts" else window.n_valid
    if variant == "pinned":
        centre = np.zeros(4) if posterior is None else posterior.center[:4]
        return np.broadcast_to(centre, (WINDOW_DAYS, HOURS_PER_DAY, 4)), counts
    profile = window.actual_settings if variant == "planned" else np.broadcast_to(known.profiles.at(known.origin), window.actual_settings.shape)
    return settings_covariates(profile), counts


def model_draws(posterior: ModelPosterior, known: Known, window: Window, rng: np.random.Generator, *, variant: str = "frozen") -> np.ndarray:
    """Predictive draws of the window's share from the fitted model and resampled exposures.

    Variants: ``frozen`` (the profile in force at the origin; the model's forecast), ``planned`` (the settings actually in force), ``pinned`` (every setting at the training centre, so the
    settings terms drop out) and ``trailing counts`` (the frozen profile, with the mean reading counts of the 28 days before the origin in place of the realised counts).
    """
    pool = exposure_pool(known)
    chosen = rng.integers(0, len(pool), size=(len(posterior.alpha), WINDOW_DAYS))
    settings, counts = _variant_inputs(known, window, variant, posterior)
    return simulate_window_share(posterior, settings, pool[chosen], counts, rng)


def model_unscorable_reason(known: Known, window: Window, *, variant: str = "frozen") -> str:
    """Why the model cannot forecast this window (no complete day of exposures, no readings, or a setting unknown at an hour with readings); empty when it can."""
    try:
        exposure_pool(known)
    except NotScored as problem:
        return str(problem)
    settings, counts = _variant_inputs(known, window, variant, None)
    if counts.sum() <= 0:
        return "hour-day model: no reading counts to simulate"
    if not np.isfinite(settings[counts > 0]).all():
        return "hour-day model: a setting is unknown at an hour of the window that has readings" + (" (the frozen profile)" if variant != "planned" else "")
    return ""


def settings_changed_in_window(window: Window, frozen: np.ndarray) -> bool:
    """True when the settings actually in force differ from the frozen profile at any hour of the window that has readings and known settings."""
    difference = np.abs(window.actual_settings - frozen[None, :, :])
    seen = (window.n_valid > 0)[:, :, None] & np.isfinite(difference)
    return bool((difference[seen] > NO_SETTINGS_CHANGE).any())


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def rng_for(origin: pd.Timestamp, predictor: str) -> np.random.Generator:
    return np.random.default_rng([SEED, int(origin.strftime("%Y%m%d")), PREDICTORS.index(predictor)])


def score_draws(draws: np.ndarray, observed: float, rng: np.random.Generator) -> dict[str, float]:
    """Score one predictive distribution of a share (percentage points): CRPS, mean, median, 90% interval, coverage and randomized PIT."""
    below, ties = float((draws < observed).mean()), float((draws == observed).mean())
    low, median, high = np.percentile(draws, [5, 50, 95])
    return {"observed_pct": 100.0 * observed, "mean_pct": 100.0 * float(draws.mean()), "median_pct": 100.0 * float(median), "lower_90_pct": 100.0 * float(low), "upper_90_pct": 100.0 * float(high),
            "crps_points": crps_points(draws, observed), "inside_90": bool(low <= observed <= high), "pit": below + float(rng.random()) * ties}


def score_origins(table: pd.DataFrame, origins: list[pd.Timestamp], *, outcome: str, daily: pd.DataFrame, tdd: pd.Series, profiles: SettingsProfiles, set_name: str,
                  posteriors: dict[pd.Timestamp, ModelPosterior] | None, predictors: tuple[str, ...] = PREDICTORS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score every predictor at every origin.

    Returns the scores (one row per origin and predictor), the origin table (coverage, whether the window was scored, why not, whether the settings changed inside it) and the ridge fits.
    """
    score_rows, origin_rows, ridge_rows = [], [], []
    for origin in origins:
        observed, coverage = window_target(table, origin, outcome)
        window = window_for(table, origin)
        known = known_at(origin, outcome, table, daily, tdd, profiles)
        frozen = known.profiles.at(origin)
        row = {"set": set_name, "origin": origin, "window_end": origin + pd.Timedelta(days=WINDOW_DAYS - 1), "coverage": coverage, "scored": coverage >= MIN_WINDOW_COVERAGE, "reason": "",
               "frozen_profile_known": bool(np.isfinite(frozen).all()), "settings_changed": settings_changed_in_window(window, frozen),
               "hours_with_unknown_settings": int(((window.n_valid > 0) & ~np.isfinite(window.actual_settings).all(axis=2)).sum())}
        if not row["scored"]:
            row["reason"] = f"window holds {coverage:.0%} of its possible readings (under {MIN_WINDOW_COVERAGE:.0%})"
            origin_rows.append(row)
            continue
        for name in predictors:
            rng = rng_for(origin, name)
            try:
                if name in BASELINES:
                    draws = baseline_draws(known, name, rng)
                elif name == "ridge":
                    draws, info = ridge_draws(known, rng)
                    ridge_rows.append({"origin": origin, **info})
                elif posteriors is None:
                    continue
                else:
                    variant = MODEL_VARIANTS[name]
                    reason = model_unscorable_reason(known, window, variant="frozen") or model_unscorable_reason(known, window, variant=variant)
                    if reason:
                        raise NotScored(reason)
                    draws = model_draws(posteriors[origin], known, window, rng, variant=variant)
            except NotScored as problem:
                score_rows.append({"set": set_name, "origin": origin, "predictor": name, "reason": str(problem)})
                continue
            score_rows.append({"set": set_name, "origin": origin, "predictor": name, "reason": "", **score_draws(draws, observed, rng)})
        origin_rows.append(row)
    return pd.DataFrame(score_rows), pd.DataFrame(origin_rows), pd.DataFrame(ridge_rows)


def scorable_for_model(table: pd.DataFrame, origins: list[pd.Timestamp], profiles: SettingsProfiles, *, outcome: str, daily: pd.DataFrame, tdd: pd.Series) -> list[pd.Timestamp]:
    """Origins whose window has enough readings and whose frozen-settings forecast the model can make (a fit is run only for these)."""
    keep = []
    for origin in origins:
        if window_target(table, origin, outcome)[1] < MIN_WINDOW_COVERAGE:
            continue
        if not model_unscorable_reason(known_at(origin, outcome, table, daily, tdd, profiles), window_for(table, origin), variant="frozen"):
            keep.append(origin)
    return keep


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def run(raw: str | None, output: str | None, *, outcome: str, draws: int, tune: int, chains: int, workers: int, score_model: bool, reuse: bool, model_development: bool, command: str, note: str) -> int:
    from .analysis_data import load_analysis_data
    from .forecast_harness_report import write_outputs
    from .hour_day import build_hour_day_table
    from .paths import ProjectPaths

    paths = ProjectPaths.from_root(Path.cwd())
    raw = raw or str(paths.cloud_raw)
    folder = Path(output or paths.output_forecast_harness)
    folder.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(folder / f"forecast_harness_run_{run_id}.log")])
    started = time.time()
    LOGGER.info("forecast harness started run_id=%s outcome=%s raw=%s score_model=%s", run_id, outcome, raw, score_model)
    analysis = load_analysis_data(raw)
    table = build_hour_day_table(analysis)
    daily = daily_counts(table, outcome)
    tdd = daily_insulin(analysis.grid, analysis.merged.tandem_data.basal)
    profiles = frozen_profiles(analysis.history, table)
    last_day = daily.index.max()
    evaluation = evaluation_origins(last_day)
    development = development_origins()
    check_origins_in_record(evaluation + development, daily.index.min(), last_day)
    LOGGER.info("record %s to %s; %d evaluation origins %s to %s; %d development origins", daily.index.min().date(), last_day.date(), len(evaluation), evaluation[0].date(), evaluation[-1].date(), len(development))
    posteriors = None
    if score_model:
        todo = scorable_for_model(table, evaluation + (development if model_development else []), profiles, outcome=outcome, daily=daily, tdd=tdd)
        posteriors = fit_all_origins(table, todo, outcome=outcome, draws=draws, tune=tune, chains=chains, workers=workers, store=folder / "posteriors" if reuse else None)
    results = {}
    for set_name, origins in (("evaluation", evaluation), ("development", development)):
        scores, origin_table, ridge = score_origins(table, origins, outcome=outcome, daily=daily, tdd=tdd, profiles=profiles, set_name=set_name,
                                                    posteriors=posteriors if (set_name == "evaluation" or model_development) else None,
                                                    predictors=PREDICTORS if (set_name == "evaluation" or model_development) else tuple(name for name in PREDICTORS if name not in MODEL_VARIANTS))
        results[set_name] = (scores, origin_table, ridge)
        LOGGER.info("%s origins: %d windows scored", set_name, int(origin_table["scored"].sum()))
    write_outputs(folder=folder, run_id=run_id, raw=raw, outcome=outcome, table=table, results=results, posteriors=posteriors, draws=draws, tune=tune, chains=chains, seconds=time.time() - started, score_model=score_model,
                  model_development=model_development, command=command, note=note)
    LOGGER.info("forecast harness finished in %.1f s", time.time() - started)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", default=None, help="Raw data root (default: the Project Vault data/raw)")
    parser.add_argument("--output", default=None, help="Output folder (default: runtime/output/forecast_harness)")
    parser.add_argument("--outcome", default="tir", choices=sorted(om.OUTCOME_COLUMNS))
    parser.add_argument("--draws", type=int, default=3000, help="Posterior draws per chain of each refit")
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4, help="Origins fitted at the same time")
    parser.add_argument("--skip-model", action="store_true", help="Score the baselines and the ridge only (the power check, before any model fit)")
    parser.add_argument("--reuse-posteriors", action="store_true", help="Store each origin's fit in the output folder and reuse it when the training data and settings are identical")
    parser.add_argument("--score-model-development", action="store_true", help="Also refit and score the model at the development origins (never used for the win test)")
    parser.add_argument("--run-note", default="", help="Free text recorded in the summary and the workbook (for example why the sampler settings were chosen)")
    args = parser.parse_args(argv)
    command = " ".join(["python -m bayesian_t1dm.forecast_harness", f"--outcome {args.outcome}", f"--draws {args.draws}", f"--tune {args.tune}", f"--chains {args.chains}", f"--workers {args.workers}",
                        *(["--skip-model"] if args.skip_model else []), *(["--reuse-posteriors"] if args.reuse_posteriors else []), *(["--score-model-development"] if args.score_model_development else []),
                        *([f'--run-note "{args.run_note}"'] if args.run_note else [])])
    return run(args.raw, args.output, outcome=args.outcome, draws=args.draws, tune=args.tune, chains=args.chains, workers=args.workers, score_model=not args.skip_model, reuse=args.reuse_posteriors,
               model_development=args.score_model_development and not args.skip_model, command=command, note=args.run_note)


if __name__ == "__main__":
    raise SystemExit(main())
