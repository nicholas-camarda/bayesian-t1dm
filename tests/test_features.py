from __future__ import annotations

import pandas as pd
import pytest

from bayesian_t1dm.features import FeatureConfig, build_feature_frame
from bayesian_t1dm.ingest import IngestedData, load_tandem_exports


def _frame(columns: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_build_feature_frame_creates_lags_and_target(tandem_fixture_dir):
    data = load_tandem_exports(tandem_fixture_dir)
    frame = build_feature_frame(data, FeatureConfig(horizon_minutes=30))

    assert frame.target_column == "target_glucose"
    assert "glucose_lag_30m" in frame.feature_columns
    assert "iob_roll_sum_60m" in frame.feature_columns
    assert frame.frame["timestamp"].is_monotonic_increasing
    assert frame.frame["target_glucose"].notna().all()
    assert frame.frame["target_delta"].notna().all()


def test_build_feature_frame_tracks_cgm_gaps_and_drops_imputed_targets():
    data = IngestedData(
        cgm=_frame(
            {
                "timestamp": pd.to_datetime(["2023-06-01 00:00:00", "2023-06-01 00:25:00", "2023-06-01 00:30:00"]),
                "glucose": [100.0, 125.0, 130.0],
                "source_file": ["fixture.csv", "fixture.csv", "fixture.csv"],
            }
        ),
    )
    frame = build_feature_frame(
        data,
        FeatureConfig(horizon_minutes=5, cgm_lags=(5,), rolling_windows_minutes=(30,), activity_windows_minutes=(30,), carb_windows_minutes=(30,)),
    )

    row = frame.frame.loc[frame.frame["timestamp"] == pd.Timestamp("2023-06-01 00:20:00")].iloc[0]

    assert pd.isna(row["glucose_observed"])
    assert row["glucose"] == pytest.approx(120.0)
    assert row["missing_cgm"] == 1
    assert row["minutes_since_last_cgm"] == pytest.approx(20.0)
    assert row["cgm_gap_ge_10m"] == 1
    assert row["cgm_gap_ge_20m"] == 1
    assert row["cgm_gap_ge_30m"] == 0
    assert row["target_glucose"] == pytest.approx(125.0)
    assert frame.frame["target_glucose"].notna().all()


def test_build_feature_frame_adds_basal_and_meal_exposures():
    data = IngestedData(
        cgm=_frame(
            {
                "timestamp": pd.date_range("2023-06-01 00:00:00", periods=5, freq="5min"),
                "glucose": [100.0, 102.0, 104.0, 106.0, 108.0],
                "source_file": ["fixture.csv"] * 5,
            }
        ),
        bolus=_frame(
            {
                "timestamp": [pd.Timestamp("2023-06-01 00:05:00")],
                "bolus_units": [2.0],
                "source_file": ["fixture.csv"],
            }
        ),
        carbs=_frame(
            {
                "timestamp": [pd.Timestamp("2023-06-01 00:00:00"), pd.Timestamp("2023-06-01 00:05:00")],
                "carb_grams": [20.0, 25.0],
                "source_file": ["fixture.csv", "fixture.csv"],
            }
        ),
        basal=_frame(
            {
                "start_timestamp": [pd.Timestamp("2023-06-01 00:00:00"), pd.Timestamp("2023-06-01 00:07:00")],
                "end_timestamp": [pd.Timestamp("2023-06-01 00:07:00"), pd.Timestamp("2023-06-01 00:20:00")],
                "basal_units_per_hour": [1.0, 2.0],
                "source_file": ["fixture.csv", "fixture.csv"],
            }
        ),
    )
    frame = build_feature_frame(
        data,
        FeatureConfig(horizon_minutes=5, cgm_lags=(5,), rolling_windows_minutes=(30,), activity_windows_minutes=(30,), carb_windows_minutes=(30, 60)),
    )

    meal_row = frame.frame.loc[frame.frame["timestamp"] == pd.Timestamp("2023-06-01 00:05:00")].iloc[0]
    basal_row = frame.frame.loc[frame.frame["timestamp"] == pd.Timestamp("2023-06-01 00:05:00")].iloc[0]
    next_row = frame.frame.loc[frame.frame["timestamp"] == pd.Timestamp("2023-06-01 00:10:00")].iloc[0]

    assert meal_row["meal_event"] == 1
    assert meal_row["carb_roll_sum_30m"] == pytest.approx(45.0)
    assert meal_row["minutes_since_last_meal"] == pytest.approx(0.0)
    assert meal_row["carb_bolus_interaction_60m"] == pytest.approx(meal_row["carb_roll_sum_60m"] * meal_row["bolus_units"])
    assert meal_row["carb_iob_interaction_60m"] == pytest.approx(meal_row["carb_roll_sum_60m"] * meal_row["iob_units"])

    assert basal_row["basal_schedule_change"] == 1
    assert basal_row["basal_units_delivered"] == pytest.approx((1.0 * 2.0 + 2.0 * 3.0) / 60.0)
    assert basal_row["minutes_since_basal_change"] == pytest.approx(0.0)
    assert next_row["minutes_since_basal_change"] == pytest.approx(5.0)
