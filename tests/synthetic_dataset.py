"""A seven-day synthetic 5-minute dataset for tests: meals at fixed hours, a fixed basal schedule and optional Apple Health columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bayesian_t1dm.features import FeatureConfig
from bayesian_t1dm.health_auto_export import AnalysisReadyHealthDataset


def synthetic_base_dataset(
    *,
    apple: bool,
    explicit_carbs: bool,
    proxy_only: bool = False,
) -> AnalysisReadyHealthDataset:
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=7 * 288, freq="5min")
    minute_of_day = timestamps.hour * 60 + timestamps.minute
    overnight = ((minute_of_day < 360)).astype(float)
    morning = ((minute_of_day >= 360) & (minute_of_day < 660)).astype(float)
    afternoon = ((minute_of_day >= 660) & (minute_of_day < 1020)).astype(float)
    evening = ((minute_of_day >= 1020)).astype(float)
    breakfast = ((minute_of_day >= 7 * 60) & (minute_of_day < 9 * 60)).astype(float)
    lunch = ((minute_of_day >= 12 * 60) & (minute_of_day < 13 * 60 + 30)).astype(float)
    dinner = ((minute_of_day >= 18 * 60) & (minute_of_day < 19 * 60 + 30)).astype(float)
    meal_event = (((breakfast + lunch + dinner) > 0) & (pd.Series(timestamps).dt.minute.eq(0))).astype(float)

    basal = 0.85 + 0.05 * morning + 0.15 * afternoon - 0.18 * overnight
    carbs = meal_event * np.where(morning > 0, 50.0, np.where(afternoon > 0, 60.0, 70.0))
    bolus_units = meal_event * np.where(morning > 0, 3.8, np.where(afternoon > 0, 4.4, 4.8))
    if proxy_only:
        carbs = carbs * 0.0
    activity_value = np.where((minute_of_day >= 17 * 60) & (minute_of_day < 18 * 60), 40.0, 0.0)

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "glucose": 112.0 + 14.0 * overnight + 7.0 * evening + 6.0 * np.sin(np.arange(len(timestamps)) / 18.0),
            "missing_cgm": 0,
            "minutes_since_last_cgm": 0.0,
            "basal_units_per_hour": basal,
            "basal_units_delivered": basal * (5.0 / 60.0),
            "basal_schedule_change": (pd.Series(basal).diff().abs().fillna(0.0) > 0.05).astype(int),
            "minutes_since_basal_change": 0.0,
            "bolus_units": bolus_units,
            "activity_value": activity_value,
            "hour_sin": np.sin(2 * np.pi * (timestamps.hour + timestamps.minute / 60.0) / 24.0),
            "hour_cos": np.cos(2 * np.pi * (timestamps.hour + timestamps.minute / 60.0) / 24.0),
            "dow_sin": np.sin(2 * np.pi * pd.Series(timestamps).dt.dayofweek / 7.0),
            "dow_cos": np.cos(2 * np.pi * pd.Series(timestamps).dt.dayofweek / 7.0),
            "is_weekend": (pd.Series(timestamps).dt.dayofweek >= 5).astype(int),
            "meal_event": meal_event if explicit_carbs and not proxy_only else 0.0,
            "carb_grams": carbs if explicit_carbs and not proxy_only else 0.0,
        }
    )
    frame["carb_roll_sum_60m"] = frame["carb_grams"].rolling(12, min_periods=1).sum()
    frame["carb_roll_sum_120m"] = frame["carb_grams"].rolling(24, min_periods=1).sum()
    frame["minutes_since_last_meal"] = (
        frame["timestamp"] - frame["timestamp"].where(frame["meal_event"].gt(0)).ffill()
    ).dt.total_seconds().div(60.0).fillna(1e6)
    frame["insulin_activity_units"] = frame["bolus_units"].rolling(12, min_periods=1).sum() / 6.0
    frame["iob_units"] = frame["bolus_units"].rolling(24, min_periods=1).sum()
    frame["iob_roll_sum_60m"] = frame["iob_units"].rolling(12, min_periods=1).sum()
    frame["iob_roll_sum_120m"] = frame["iob_units"].rolling(24, min_periods=1).sum()
    frame["glucose_roll_mean_30m"] = frame["glucose"].rolling(6, min_periods=1).mean()
    frame["glucose_roll_mean_60m"] = frame["glucose"].rolling(12, min_periods=1).mean()
    frame["glucose_lag_5m"] = frame["glucose"].shift(1).fillna(frame["glucose"])
    frame["glucose_lag_30m"] = frame["glucose"].shift(6).fillna(frame["glucose"])
    frame["activity_roll_sum_60m"] = frame["activity_value"].rolling(12, min_periods=1).sum()
    frame["carb_bolus_interaction_60m"] = frame["carb_roll_sum_60m"] * frame["bolus_units"]
    frame["carb_iob_interaction_60m"] = frame["carb_roll_sum_60m"] * frame["iob_units"]

    if apple:
        poor_sleep = np.where(pd.Series(timestamps).dt.dayofweek.isin([0, 1]), 1.0, 0.0)
        frame["prior_night_total_sleep_hours"] = 7.8 - 1.6 * poor_sleep
        frame["in_sleep"] = (overnight > 0).astype(int)
        frame["recent_workout_12h"] = np.where((minute_of_day >= 17 * 60) | (minute_of_day < 6 * 60), 1, 0)
        frame["health_activity_roll_sum_60m"] = frame["activity_roll_sum_60m"]
        frame["hrv_latest"] = 46.0 + 7.0 * overnight - 4.0 * poor_sleep
        frame["heart_rate_avg_latest"] = 68.0 + 8.0 * frame["activity_value"].gt(0).astype(float) + 5.0 * poor_sleep
        frame["respiratory_rate_latest"] = 14.0 + 0.8 * frame["activity_value"].gt(0).astype(float)
    target_delta = (
        0.18 * (110.0 - frame["glucose"])
        - 8.0 * frame["basal_units_per_hour"]
        - 2.4 * frame["bolus_units"]
        + 0.14 * frame["carb_roll_sum_60m"]
        - 0.03 * frame["activity_value"]
        + 7.5 * overnight
    )
    frame["target_delta"] = target_delta
    frame["target_glucose"] = (frame["glucose"] + frame["target_delta"]).clip(lower=65.0, upper=260.0)

    tandem_features = [
        "glucose",
        "missing_cgm",
        "minutes_since_last_cgm",
        "basal_units_per_hour",
        "basal_units_delivered",
        "basal_schedule_change",
        "minutes_since_basal_change",
        "bolus_units",
        "insulin_activity_units",
        "iob_units",
        "iob_roll_sum_60m",
        "iob_roll_sum_120m",
        "carb_grams",
        "meal_event",
        "minutes_since_last_meal",
        "carb_roll_sum_60m",
        "carb_roll_sum_120m",
        "activity_value",
        "activity_roll_sum_60m",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "glucose_lag_5m",
        "glucose_lag_30m",
        "glucose_roll_mean_30m",
        "glucose_roll_mean_60m",
        "carb_bolus_interaction_60m",
        "carb_iob_interaction_60m",
    ]
    health_features = []
    if apple:
        health_features = [
            "prior_night_total_sleep_hours",
            "in_sleep",
            "recent_workout_12h",
            "health_activity_roll_sum_60m",
            "hrv_latest",
            "heart_rate_avg_latest",
            "respiratory_rate_latest",
        ]
    return AnalysisReadyHealthDataset(
        frame=frame,
        feature_columns=tandem_features + health_features,
        tandem_feature_columns=tandem_features,
        health_feature_columns=health_features,
        target_column="target_glucose",
        horizon_minutes=30,
        config=FeatureConfig(horizon_minutes=30),
        mode="apple_enriched" if apple else "tandem_only",
        apple_available=apple,
        explicit_carb_source_available=bool(explicit_carbs and not proxy_only),
    )
