from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Sequence

import numpy as np
import pandas as pd

from .ingest import IngestedData
from .insulin import expand_bolus_series_to_grid, expand_bolus_to_grid


@dataclass(frozen=True)
class FeatureConfig:
    freq: str = "5min"
    iob_duration_minutes: int = 300
    horizon_minutes: int = 30
    fill_cgm_interior_only: bool = True
    drop_imputed_targets: bool = True
    gap_flag_thresholds_minutes: tuple[int, ...] = (10, 20, 30)
    carb_windows_minutes: tuple[int, ...] = (30, 60, 120)
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


def _freq_minutes(freq: str) -> int:
    return int(pd.Timedelta(freq).total_seconds() // 60)


def build_time_grid(data: IngestedData, freq: str = "5min") -> pd.DataFrame:
    timestamps: list[pd.Series] = []
    if not data.cgm.empty and "timestamp" in data.cgm.columns:
        timestamps.append(pd.to_datetime(data.cgm["timestamp"]))
    if not data.bolus.empty and "timestamp" in data.bolus.columns:
        timestamps.append(pd.to_datetime(data.bolus["timestamp"]))
    if not data.carbs.empty and "timestamp" in data.carbs.columns:
        timestamps.append(pd.to_datetime(data.carbs["timestamp"]))
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


def _aggregate_cgm(
    data: IngestedData,
    grid: pd.DataFrame,
    *,
    freq: str,
    fill_interior_only: bool,
    gap_flag_thresholds_minutes: Sequence[int],
) -> pd.DataFrame:
    out = grid.copy()
    out["glucose_observed"] = np.nan
    out["glucose"] = np.nan
    out["missing_cgm"] = 1
    out["minutes_since_last_cgm"] = np.nan
    if data.cgm.empty or "timestamp" not in data.cgm.columns or "glucose" not in data.cgm.columns:
        for threshold in gap_flag_thresholds_minutes:
            out[f"cgm_gap_ge_{threshold}m"] = 0
        return out

    cgm = data.cgm.copy()
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"]).dt.floor(freq)
    cgm = cgm.groupby("timestamp", as_index=False).agg(glucose_observed=("glucose", "mean"))
    out = out.merge(cgm, on="timestamp", how="left")
    out["glucose_observed"] = out["glucose_observed_y"].combine_first(out["glucose_observed_x"]) if "glucose_observed_y" in out.columns else out["glucose_observed"]
    out = out.drop(columns=[column for column in ["glucose_observed_x", "glucose_observed_y"] if column in out.columns])
    out["glucose"] = out["glucose_observed"]
    if fill_interior_only:
        out["glucose"] = out["glucose"].interpolate(method="linear", limit_area="inside")
    else:
        out["glucose"] = out["glucose"].interpolate(method="linear", limit_direction="both")
    out["missing_cgm"] = out["glucose_observed"].isna().astype(int)
    last_observed_ts = out["timestamp"].where(out["glucose_observed"].notna()).ffill()
    out["minutes_since_last_cgm"] = (out["timestamp"] - last_observed_ts).dt.total_seconds().div(60.0)
    for threshold in gap_flag_thresholds_minutes:
        out[f"cgm_gap_ge_{threshold}m"] = out["minutes_since_last_cgm"].ge(threshold).fillna(False).astype(int)
    return out


def _aggregate_activity(data: IngestedData, grid: pd.DataFrame, *, freq: str) -> pd.DataFrame:
    out = grid.copy()
    out["activity_value"] = 0.0
    if data.activity.empty or "timestamp" not in data.activity.columns or "activity_value" not in data.activity.columns:
        return out
    activity = data.activity.copy()
    activity["timestamp"] = pd.to_datetime(activity["timestamp"]).dt.floor(freq)
    activity = activity.groupby("timestamp", as_index=False).agg(activity_value=("activity_value", "sum"))
    out = out.merge(activity, on="timestamp", how="left", suffixes=("", "_agg"))
    out["activity_value"] = out["activity_value_agg"].combine_first(out["activity_value"])
    out = out.drop(columns=[column for column in ["activity_value_agg"] if column in out.columns])
    out["activity_value"] = out["activity_value"].fillna(0.0)
    return out


def _aggregate_basal(data: IngestedData, grid: pd.DataFrame, *, freq: str) -> pd.DataFrame:
    step_minutes = _freq_minutes(freq)
    out = grid.copy()
    out["basal_units_per_hour"] = 0.0
    out["basal_units_delivered"] = 0.0
    out["basal_schedule_change"] = 0
    out["minutes_since_basal_change"] = 0.0
    if data.basal.empty or "basal_units_per_hour" not in data.basal.columns or "start_timestamp" not in data.basal.columns:
        return out

    basal = data.basal.copy()
    basal["start_timestamp"] = pd.to_datetime(basal["start_timestamp"], errors="coerce")
    if "end_timestamp" in basal.columns:
        basal["end_timestamp"] = pd.to_datetime(basal["end_timestamp"], errors="coerce")
    else:
        basal["end_timestamp"] = pd.NaT
    basal["basal_units_per_hour"] = pd.to_numeric(basal["basal_units_per_hour"], errors="coerce")
    basal = basal.dropna(subset=["start_timestamp", "basal_units_per_hour"]).sort_values("start_timestamp").reset_index(drop=True)
    if basal.empty:
        return out

    bin_start = out["timestamp"].to_numpy(dtype="datetime64[ns]")
    bin_end = bin_start + np.timedelta64(step_minutes, "m")
    overlap_minutes = np.zeros(len(out), dtype=float)
    delivered_units = np.zeros(len(out), dtype=float)
    rate_weighted_minutes = np.zeros(len(out), dtype=float)
    overlap_count = np.zeros(len(out), dtype=int)
    schedule_change = np.zeros(len(out), dtype=int)

    previous_rate: float | None = None
    last_end = pd.Timestamp(out["timestamp"].max()) + pd.Timedelta(minutes=step_minutes)
    for row in basal.itertuples(index=False):
        start = pd.Timestamp(getattr(row, "start_timestamp"))
        end = pd.Timestamp(getattr(row, "end_timestamp")) if not pd.isna(getattr(row, "end_timestamp")) else last_end
        rate = float(getattr(row, "basal_units_per_hour"))
        if pd.isna(start) or pd.isna(rate) or end <= start:
            continue
        start64 = np.datetime64(start.to_datetime64())
        end64 = np.datetime64(end.to_datetime64())
        overlap_start = np.maximum(bin_start, start64)
        overlap_end = np.minimum(bin_end, end64)
        overlap = (overlap_end - overlap_start) / np.timedelta64(1, "m")
        mask = overlap > 0
        if not np.any(mask):
            previous_rate = rate
            continue
        overlap_minutes[mask] += overlap[mask]
        delivered_units[mask] += rate * overlap[mask] / 60.0
        rate_weighted_minutes[mask] += rate * overlap[mask]
        overlap_count[mask] += 1
        if previous_rate is not None and not np.isclose(rate, previous_rate):
            change_idx = np.searchsorted(bin_start, start64, side="right") - 1
            if 0 <= change_idx < len(schedule_change):
                schedule_change[change_idx] = 1
        previous_rate = rate

    schedule_change = np.where(overlap_count > 1, 1, schedule_change)
    valid = overlap_minutes > 0
    out.loc[valid, "basal_units_per_hour"] = rate_weighted_minutes[valid] / overlap_minutes[valid]
    out["basal_units_delivered"] = delivered_units
    out["basal_schedule_change"] = schedule_change
    change_ts = out["timestamp"].where(out["basal_schedule_change"].astype(bool)).ffill()
    out["minutes_since_basal_change"] = (out["timestamp"] - change_ts).dt.total_seconds().div(60.0).fillna(0.0)
    return out


def _aggregate_carbs(data: IngestedData, grid: pd.DataFrame, *, freq: str, windows_minutes: Sequence[int]) -> pd.DataFrame:
    step_minutes = _freq_minutes(freq)
    out = grid.copy()
    out["carb_grams"] = 0.0
    out["meal_event"] = 0
    out["minutes_since_last_meal"] = 0.0
    for window in windows_minutes:
        out[f"carb_roll_sum_{window}m"] = 0.0
    if data.carbs.empty or "timestamp" not in data.carbs.columns or "carb_grams" not in data.carbs.columns:
        return out

    carbs = data.carbs.copy()
    carbs["timestamp"] = pd.to_datetime(carbs["timestamp"], errors="coerce").dt.floor(freq)
    carbs["carb_grams"] = pd.to_numeric(carbs["carb_grams"], errors="coerce")
    carbs = carbs.dropna(subset=["timestamp", "carb_grams"])
    if carbs.empty:
        return out

    carbs = carbs.groupby("timestamp", as_index=False).agg(carb_grams=("carb_grams", "sum"))
    out = out.merge(carbs, on="timestamp", how="left", suffixes=("", "_agg"))
    out["carb_grams"] = out["carb_grams_agg"].combine_first(out["carb_grams"])
    out = out.drop(columns=[column for column in ["carb_grams_agg"] if column in out.columns])
    out["carb_grams"] = out["carb_grams"].fillna(0.0)
    out["meal_event"] = (out["carb_grams"] > 0).astype(int)
    last_meal_ts = out["timestamp"].where(out["meal_event"].astype(bool)).ffill()
    out["minutes_since_last_meal"] = (out["timestamp"] - last_meal_ts).dt.total_seconds().div(60.0).fillna(0.0)
    for window in windows_minutes:
        periods = max(int(window / step_minutes), 1)
        out[f"carb_roll_sum_{window}m"] = out["carb_grams"].rolling(periods, min_periods=1).sum()
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


def _add_lags(df: pd.DataFrame, lags: Sequence[int], *, step_minutes: int) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        lagged = out["glucose"].shift(max(int(lag / step_minutes), 1))
        out[f"glucose_lag_{lag}m"] = lagged.fillna(out["glucose"])
    return out


def _add_rolling_features(
    df: pd.DataFrame,
    windows_minutes: Sequence[int],
    *,
    step_minutes: int,
    activity_windows_minutes: Sequence[int] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    for window in windows_minutes:
        periods = max(int(window / step_minutes), 1)
        out[f"glucose_roll_mean_{window}m"] = out["glucose"].rolling(periods, min_periods=1).mean()
        out[f"iob_roll_sum_{window}m"] = out["iob_units"].rolling(periods, min_periods=1).sum()
    for window in activity_windows_minutes or windows_minutes:
        periods = max(int(window / step_minutes), 1)
        out[f"activity_roll_sum_{window}m"] = out["activity_value"].rolling(periods, min_periods=1).sum()
    return out


def _add_carb_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "carb_roll_sum_60m" in out.columns and "bolus_units" in out.columns:
        out["carb_bolus_interaction_60m"] = out["carb_roll_sum_60m"] * out["bolus_units"]
    if "carb_roll_sum_60m" in out.columns and "iob_units" in out.columns:
        out["carb_iob_interaction_60m"] = out["carb_roll_sum_60m"] * out["iob_units"]
    return out


def recompute_scenario_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    if "bolus_units" in out.columns:
        exposure = expand_bolus_series_to_grid(out[["timestamp", "bolus_units"]], duration_minutes=300, step_minutes=5)
        if "insulin_activity_units" in out.columns and "insulin_activity_units" in exposure.columns:
            out["insulin_activity_units"] = exposure["insulin_activity_units"].to_numpy()
        if "iob_units" in out.columns and "iob_units" in exposure.columns:
            out["iob_units"] = exposure["iob_units"].to_numpy()
    for column in list(out.columns):
        match = re.fullmatch(r"iob_roll_sum_(\d+)m", column)
        if match and "iob_units" in out.columns:
            window = int(match.group(1))
            periods = max(int(window / 5), 1)
            out[column] = out["iob_units"].rolling(periods, min_periods=1).sum()
    out = _add_carb_interactions(out)
    return out


def build_feature_frame(data: IngestedData, config: FeatureConfig | None = None) -> FeatureFrame:
    import warnings
    config = config or FeatureConfig()
    if data.carbs.empty:
        warnings.warn(
            "Carb data is empty — carb_grams will be 0 everywhere. "
            "If you expect carb data, check that your Tandem export includes carbohydrate records.",
            UserWarning,
            stacklevel=2,
        )
    grid = build_time_grid(data, freq=config.freq)
    if grid.empty:
        frame = pd.DataFrame(columns=["timestamp"])
        return FeatureFrame(frame=frame, feature_columns=[], target_column="target_glucose", horizon_minutes=config.horizon_minutes, config=config)

    step_minutes = _freq_minutes(config.freq)
    frame = _aggregate_cgm(
        data,
        grid,
        freq=config.freq,
        fill_interior_only=config.fill_cgm_interior_only,
        gap_flag_thresholds_minutes=config.gap_flag_thresholds_minutes,
    )
    frame = _aggregate_activity(data, frame, freq=config.freq)
    frame = _aggregate_basal(data, frame, freq=config.freq)
    frame = expand_bolus_to_grid(data.bolus, frame, duration_minutes=config.iob_duration_minutes, step_minutes=step_minutes)
    frame = _aggregate_carbs(data, frame, freq=config.freq, windows_minutes=config.carb_windows_minutes)
    frame = _add_calendar_features(frame)
    frame = _add_lags(frame, config.cgm_lags, step_minutes=step_minutes)
    frame = _add_rolling_features(
        frame,
        config.rolling_windows_minutes,
        step_minutes=step_minutes,
        activity_windows_minutes=config.activity_windows_minutes,
    )
    frame = _add_carb_interactions(frame)
    horizon_steps = max(int(config.horizon_minutes / step_minutes), 1)
    target_source = frame["glucose_observed"]
    frame["target_glucose"] = target_source.shift(-horizon_steps)
    if not config.drop_imputed_targets:
        fallback_target = frame["glucose"].shift(-horizon_steps)
        frame["target_glucose"] = frame["target_glucose"].combine_first(fallback_target)
    frame["target_delta"] = frame["target_glucose"] - frame["glucose"]

    numeric_candidates = [
        "bolus_units",
        "insulin_activity_units",
        "iob_units",
        "basal_units_per_hour",
        "basal_units_delivered",
        "basal_schedule_change",
        "minutes_since_basal_change",
        "activity_value",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "glucose",
        "missing_cgm",
        "minutes_since_last_cgm",
        "carb_grams",
        "meal_event",
        "minutes_since_last_meal",
        "carb_bolus_interaction_60m",
        "carb_iob_interaction_60m",
    ]
    numeric_candidates.extend(f"glucose_lag_{lag}m" for lag in config.cgm_lags)
    numeric_candidates.extend(f"cgm_gap_ge_{threshold}m" for threshold in config.gap_flag_thresholds_minutes)
    numeric_candidates.extend(f"glucose_roll_mean_{window}m" for window in config.rolling_windows_minutes)
    numeric_candidates.extend(f"iob_roll_sum_{window}m" for window in config.rolling_windows_minutes)
    numeric_candidates.extend(f"activity_roll_sum_{window}m" for window in config.activity_windows_minutes)
    numeric_candidates.extend(f"carb_roll_sum_{window}m" for window in config.carb_windows_minutes)
    feature_columns = [column for column in numeric_candidates if column in frame.columns and column != "glucose_observed"]
    if feature_columns:
        fill_columns = [column for column in feature_columns if column != "glucose"]
        frame[fill_columns] = frame[fill_columns].fillna(0.0)
    frame = frame.dropna(subset=["glucose", "target_glucose"]).reset_index(drop=True)
    return FeatureFrame(
        frame=frame,
        feature_columns=feature_columns,
        target_column="target_glucose",
        horizon_minutes=config.horizon_minutes,
        config=config,
    )
