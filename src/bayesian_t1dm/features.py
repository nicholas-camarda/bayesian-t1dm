from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .ingest import IngestedData
from .insulin import expand_bolus_to_grid


@dataclass(frozen=True)
class FeatureConfig:
    freq: str = "5min"
    iob_duration_minutes: int = 300
    horizon_minutes: int = 30
    cgm_lags: tuple[int, ...] = (5, 10, 15, 30, 60)
    rolling_windows_minutes: tuple[int, ...] = (30, 60, 120)
    activity_windows_minutes: tuple[int, ...] = (30, 60, 120)


@dataclass
class FeatureFrame:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    horizon_minutes: int
    config: FeatureConfig


def build_time_grid(data: IngestedData, freq: str = "5min") -> pd.DataFrame:
    timestamps: list[pd.Series] = []
    if not data.cgm.empty and "timestamp" in data.cgm.columns:
        timestamps.append(pd.to_datetime(data.cgm["timestamp"]))
    if not data.bolus.empty and "timestamp" in data.bolus.columns:
        timestamps.append(pd.to_datetime(data.bolus["timestamp"]))
    if not data.activity.empty and "timestamp" in data.activity.columns:
        timestamps.append(pd.to_datetime(data.activity["timestamp"]))
    if not data.basal.empty:
        if "start_timestamp" in data.basal.columns:
            timestamps.append(pd.to_datetime(data.basal["start_timestamp"]))
        if "end_timestamp" in data.basal.columns:
            timestamps.append(pd.to_datetime(data.basal["end_timestamp"]))
    if not timestamps:
        return pd.DataFrame(columns=["timestamp"])
    all_ts = pd.concat(timestamps, ignore_index=True).dropna()
    if all_ts.empty:
        return pd.DataFrame(columns=["timestamp"])
    start = all_ts.min().floor(freq)
    end = all_ts.max().ceil(freq)
    return pd.DataFrame({"timestamp": pd.date_range(start=start, end=end, freq=freq)})


def _aggregate_cgm(data: IngestedData, grid: pd.DataFrame) -> pd.DataFrame:
    if data.cgm.empty:
        out = grid.copy()
        out["glucose"] = np.nan
        return out
    cgm = data.cgm.copy()
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"]).dt.floor("5min")
    cgm = cgm.groupby("timestamp", as_index=False).agg(glucose=("glucose", "mean"))
    out = grid.merge(cgm, on="timestamp", how="left")
    out["glucose"] = out["glucose"].interpolate(method="linear", limit_direction="both")
    return out


def _aggregate_activity(data: IngestedData, grid: pd.DataFrame) -> pd.DataFrame:
    if data.activity.empty:
        out = grid.copy()
        out["activity_value"] = 0.0
        return out
    activity = data.activity.copy()
    activity["timestamp"] = pd.to_datetime(activity["timestamp"]).dt.floor("5min")
    activity = activity.groupby("timestamp", as_index=False).agg(activity_value=("activity_value", "sum"))
    return grid.merge(activity, on="timestamp", how="left")


def _aggregate_basal(data: IngestedData, grid: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()
    out["basal_units_per_hour"] = 0.0
    if data.basal.empty or "basal_units_per_hour" not in data.basal.columns:
        return out
    basal = data.basal.copy()
    if "start_timestamp" in basal.columns:
        basal = basal.sort_values("start_timestamp")
        basal["start_timestamp"] = pd.to_datetime(basal["start_timestamp"])
        if "end_timestamp" in basal.columns:
            basal["end_timestamp"] = pd.to_datetime(basal["end_timestamp"])
        records = []
        for row in basal.itertuples(index=False):
            start = getattr(row, "start_timestamp", None)
            end = getattr(row, "end_timestamp", None)
            rate = float(getattr(row, "basal_units_per_hour"))
            if pd.isna(start):
                continue
            if pd.isna(end):
                end = grid["timestamp"].max() + pd.Timedelta(minutes=5)
            mask = (out["timestamp"] >= start) & (out["timestamp"] < end)
            records.append((mask, rate))
        for mask, rate in records:
            out.loc[mask, "basal_units_per_hour"] = rate
    return out


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hour = out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60.0
    dow = out["timestamp"].dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["is_weekend"] = (dow >= 5).astype(int)
    return out


def _add_lags(df: pd.DataFrame, lags: Sequence[int]) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"glucose_lag_{lag}m"] = out["glucose"].shift(lag // 5)
    return out


def _add_rolling_features(df: pd.DataFrame, windows_minutes: Sequence[int]) -> pd.DataFrame:
    out = df.copy()
    for window in windows_minutes:
        periods = max(int(window / 5), 1)
        out[f"glucose_roll_mean_{window}m"] = out["glucose"].rolling(periods, min_periods=1).mean()
        out[f"iob_roll_sum_{window}m"] = out["iob_units"].rolling(periods, min_periods=1).sum()
        out[f"activity_roll_sum_{window}m"] = out["activity_value"].rolling(periods, min_periods=1).sum()
    return out


def build_feature_frame(data: IngestedData, config: FeatureConfig | None = None) -> FeatureFrame:
    config = config or FeatureConfig()
    grid = build_time_grid(data, freq=config.freq)
    if grid.empty:
        frame = pd.DataFrame(columns=["timestamp"])
        return FeatureFrame(frame=frame, feature_columns=[], target_column="target_glucose", horizon_minutes=config.horizon_minutes, config=config)

    frame = _aggregate_cgm(data, grid)
    frame = _aggregate_activity(data, frame)
    frame = _aggregate_basal(data, frame)
    frame = expand_bolus_to_grid(data.bolus, frame, duration_minutes=config.iob_duration_minutes, step_minutes=5)
    frame = _add_calendar_features(frame)
    frame = _add_lags(frame, config.cgm_lags)
    frame = _add_rolling_features(frame, config.rolling_windows_minutes)
    horizon_steps = int(config.horizon_minutes / 5)
    frame["target_glucose"] = frame["glucose"].shift(-horizon_steps)
    frame["target_delta"] = frame["target_glucose"] - frame["glucose"]

    numeric_candidates = [
        "bolus_units",
        "insulin_activity_units",
        "iob_units",
        "basal_units_per_hour",
        "activity_value",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "glucose",
        "glucose_lag_5m",
        "glucose_lag_10m",
        "glucose_lag_15m",
        "glucose_lag_30m",
        "glucose_lag_60m",
        "glucose_roll_mean_30m",
        "glucose_roll_mean_60m",
        "glucose_roll_mean_120m",
        "iob_roll_sum_30m",
        "iob_roll_sum_60m",
        "iob_roll_sum_120m",
        "activity_roll_sum_30m",
        "activity_roll_sum_60m",
        "activity_roll_sum_120m",
    ]
    feature_columns = [column for column in numeric_candidates if column in frame.columns]
    required_columns = list(dict.fromkeys(feature_columns + ["target_glucose"]))
    frame = frame.dropna(subset=required_columns).reset_index(drop=True)
    return FeatureFrame(
        frame=frame,
        feature_columns=feature_columns,
        target_column="target_glucose",
        horizon_minutes=config.horizon_minutes,
        config=config,
    )
