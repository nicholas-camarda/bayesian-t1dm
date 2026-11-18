from __future__ import annotations

from pathlib import Path

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
    # Causal fill: the gap row carries the last reading (100 at 00:00) forward. Linear
    # interpolation would give 120 here by peeking at the 00:25 reading, which is this
    # row's own forecast target.
    assert row["glucose"] == pytest.approx(100.0)
    assert row["missing_cgm"] == 1
    assert row["minutes_since_last_cgm"] == pytest.approx(20.0)
    assert row["cgm_gap_ge_10m"] == 1
    assert row["cgm_gap_ge_20m"] == 1
    assert row["cgm_gap_ge_30m"] == 0
    assert row["target_glucose"] == pytest.approx(125.0)
    assert frame.frame["target_glucose"].notna().all()


def test_build_feature_frame_excludes_invalid_cgm_from_observed_and_targets():
    data = IngestedData(
        cgm=_frame(
            {
                "timestamp": pd.to_datetime(["2023-06-01 00:00:00", "2023-06-01 00:05:00", "2023-06-01 00:10:00"]),
                "glucose": [100.0, 0.0, 110.0],
                "source_file": ["fixture.csv"] * 3,
            }
        ),
    )

    frame = build_feature_frame(
        data,
        FeatureConfig(horizon_minutes=5, cgm_lags=(5,), rolling_windows_minutes=(30,), activity_windows_minutes=(30,), carb_windows_minutes=(30,)),
    )

    invalid_row = frame.frame.loc[frame.frame["timestamp"] == pd.Timestamp("2023-06-01 00:05:00")].iloc[0]
    assert pd.isna(invalid_row["glucose_observed"])
    assert invalid_row["cgm_invalid"] == 1
    assert invalid_row["target_glucose"] == pytest.approx(110.0)
    assert not frame.frame["target_glucose"].eq(0.0).any()


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


def test_build_feature_frame_warns_when_tconnectsync_has_no_explicit_carb_records():
    data = IngestedData(
        cgm=_frame(
            {
                "timestamp": pd.date_range("2023-06-01 00:00:00", periods=3, freq="5min"),
                "glucose": [100.0, 102.0, 104.0],
                "source_file": ["fixture.csv"] * 3,
            }
        ),
        bolus=_frame(
            {
                "timestamp": [pd.Timestamp("2023-06-01 00:05:00")],
                "bolus_units": [2.0],
                "source_file": ["fixture.csv"],
            }
        ),
        source_files=[Path("/tmp/tconnectsync/2026-03-02__2026-03-31/normalized/bolus.csv")],
    )

    with pytest.warns(UserWarning, match="tconnectsync/Tandem payloads in this dataset do not appear to include explicit carbohydrate records"):
        frame = build_feature_frame(data, FeatureConfig(horizon_minutes=5))

    assert not frame.frame.empty


def _gap_data() -> IngestedData:
    return IngestedData(
        cgm=_frame(
            {
                "timestamp": pd.to_datetime(["2023-06-01 00:00:00", "2023-06-01 00:25:00", "2023-06-01 00:30:00"]),
                "glucose": [100.0, 125.0, 130.0],
                "source_file": ["fixture.csv"] * 3,
            }
        ),
    )


def _gap_config(**overrides) -> FeatureConfig:
    return FeatureConfig(
        horizon_minutes=5,
        cgm_lags=(5,),
        rolling_windows_minutes=(30,),
        activity_windows_minutes=(30,),
        carb_windows_minutes=(30,),
        **overrides,
    )


def test_default_cgm_fill_never_uses_readings_after_the_row_timestamp():
    with pytest.warns(UserWarning):
        frame = build_feature_frame(_gap_data(), _gap_config()).frame

    gap_rows = frame.loc[frame["missing_cgm"] == 1]
    assert not gap_rows.empty
    # Every gap row sits between the 100 reading and the 125 reading; a causal fill can
    # only ever report the earlier one.
    assert gap_rows["glucose"].eq(100.0).all()
    assert gap_rows["glucose_lag_5m"].eq(100.0).all()


def test_linear_cgm_fill_remains_available_for_retrospective_use():
    with pytest.warns(UserWarning):
        frame = build_feature_frame(_gap_data(), _gap_config(cgm_fill_method="linear")).frame

    row = frame.loc[frame["timestamp"] == pd.Timestamp("2023-06-01 00:20:00")].iloc[0]
    assert row["glucose"] == pytest.approx(120.0)


def test_unknown_cgm_fill_method_is_rejected():
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="cgm_fill_method"):
        build_feature_frame(_gap_data(), _gap_config(cgm_fill_method="spline"))
