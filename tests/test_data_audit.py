from __future__ import annotations

import base64
import struct
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from test_hour_day import DAY1, PUMP, _analysis

from bayesian_t1dm import data_audit as audit
from bayesian_t1dm import data_audit_report as report
from bayesian_t1dm.diagnostics_workbook import read_workbook, write_workbook
from bayesian_t1dm.hour_day import build_hour_day_table
from bayesian_t1dm.paths import ProjectPaths

STATUSES = {"no concern", "concern", "not assessed", "not applicable"}


def _slots(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="5min")


def test_runs_returns_start_and_exclusive_stop_of_each_run():
    flags = np.array([False, True, True, False, True, False, True, True])
    assert audit._runs(flags) == [(1, 3), (4, 5), (6, 8)]
    assert audit._runs(np.zeros(4, dtype=bool)) == []


def test_active_time_windows_count_readings_per_window_and_mark_the_partial_last_one():
    table = pd.DataFrame({"date": np.repeat(pd.date_range("2025-09-01", periods=20), 24), "n_valid": 12})
    table.loc[table["date"] < "2025-09-08", "n_valid"] = 6  # half the readings in the first week
    windows = audit.active_time_windows(table)
    assert windows["days"].tolist() == [14, 6] and windows["full"].tolist() == [True, False]
    assert windows["active_pct"].iloc[0] == pytest.approx(100 * (7 * 0.5 + 7) / 14)
    assert windows["active_pct"].iloc[1] == pytest.approx(100.0)


def test_gap_distribution_bins_missing_slots_and_shares_sum_to_100():
    stamps = _slots("2025-09-01", 400)
    keep = np.ones(len(stamps), dtype=bool)
    keep[10] = False         # one missing slot: a 5 minute gap
    keep[20:23] = False      # three: 10-25 min
    keep[100:112] = False    # twelve: 1-2 h
    keep[200:300] = False    # one hundred: 6 h or more
    bins = audit.gap_distribution(pd.DataFrame({"timestamp": stamps[keep]})).set_index("gap_length")
    assert bins["gaps"].to_dict() == {"5 min": 1, "10-25 min": 1, "30-55 min": 0, "1-2 h": 1, "2-6 h": 0, "6 h or more": 1}
    assert bins.loc["1-2 h", "missing_minutes"] == 60
    assert bins["share_of_missing_minutes_pct"].sum() == pytest.approx(100.0)


def test_impossible_values_and_flat_runs_are_listed_as_record_issues():
    stamps = _slots("2025-09-01", 40)
    glucose = np.linspace(100, 160, 40).round()
    glucose[5], glucose[6] = 0.0, 38.0           # below the sensor floor
    glucose[20:34] = 120.0                       # fourteen identical readings: a flat run
    raw = pd.DataFrame({"timestamp": stamps, "glucose": glucose, "pump_serial": PUMP, "duplicate_count": 1})
    valid = raw[raw["glucose"] >= 40]
    checks, issues = audit.cgm_duplicates_and_values(raw, valid)
    status = checks.set_index("check")["status"]
    assert status["Readings below the sensor floor (LOW)"] == "concern" and status["Flat runs"] == "concern"
    assert status["Duplicate timestamps"] == "no concern"
    assert issues.groupby("step")["n"].sum().to_dict() == {"CGM impossible value": 2, "CGM flat run": 14}
    assert set(checks["status"]) <= STATUSES


def test_calculator_and_basal_checks_agree_with_the_profile_and_flag_a_wrong_setting():
    history = _analysis().history
    meal = pd.Timestamp("2025-09-01 10:00")
    carbs = pd.DataFrame({"event_time": [meal], "timestamp": [meal], "pump_serial": [PUMP], "carb_ratio_g_per_unit": [10.0], "isf_mgdl_per_unit": [65.0], "target_bg_mgdl": [110.0]})
    checks, issues = audit.calculator_agreement(carbs, history)
    assert checks["status"].tolist() == ["no concern"] * 3 and issues.empty
    wrong = carbs.assign(carb_ratio_g_per_unit=7.0)   # the value before the 09:07 edit
    checks, issues = audit.calculator_agreement(wrong, history)
    assert checks["status"].tolist() == ["concern", "no concern", "no concern"]
    assert issues["step"].str.contains("I:C").all() and issues["n"].sum() == 1
    rates = pd.DataFrame({"pump_serial": [PUMP, PUMP], "timestamp": [meal, meal + pd.Timedelta(hours=1)], "rate": [0.8, 0.8]})
    assert audit.basal_log_agreement(rates, history)[0]["status"].tolist() == ["no concern"]
    checks, issues = audit.basal_log_agreement(rates.assign(rate=0.9), history)
    assert checks["status"].tolist() == ["concern"] and issues["n"].sum() == 2


def _jump(stamp: str, minutes: float, sequence: int, event: int = audit.TIME_CHANGED_EVENT, pump: str = PUMP) -> dict:
    when = pd.Timestamp(stamp)
    return {"pump_serial": pump, "sequence": sequence, "timestamp": when, "previous_timestamp": when - pd.Timedelta(minutes=minutes), "event": event, "jump_minutes": minutes}


YEAR = pd.DataFrame({"hour_start": pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h"), "date": pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h").normalize(), "pump_serial": PUMP})
NEW_YORK = ZoneInfo("America/New_York")


def test_a_time_change_and_the_date_change_that_follows_it_net_to_one_shift():
    jumps = pd.DataFrame([
        _jump("2025-06-02 00:00", -1462, 10, event=90),      # the new-day event stamped a day too early
        _jump("2025-06-02 00:22", 22, 11, event=audit.DATE_CHANGED_EVENT),
        _jump("2025-06-02 21:22", 1260, 12),                  # the time set 21 hours on
    ])
    changes, checks, issues = audit.clock_jumps(jumps, YEAR, zone=NEW_YORK)
    assert changes["class"].tolist() == ["time-zone change (2-3.5 hours)"] and changes["net_minutes"].iloc[0] == pytest.approx(-180)   # one change of 3 hours, not three jumps
    assert checks.set_index("check").loc["Timestamp inversions", "result"] == 0               # the negative jump belongs to the clock change
    assert issues["step"].tolist() == ["pump clock jump"] and issues["value"].tolist() == ["-180"]


def test_clock_changes_are_classed_by_their_net_shift():
    jumps = pd.DataFrame([
        _jump("2025-03-09 08:00", 60, 100),                          # spring forward on 2025-03-09; the person sets the pump five hours after the wall clock jumps
        _jump("2025-04-01 09:00", 5, 300),                           # under 50 minutes: no clock change
        _jump("2025-06-01 09:00", 180, 500),
        _jump("2025-06-20 09:00", -180, 700),                        # back from a clock three hours away
        _jump("2025-07-04 09:00", 600, 900),                         # a clock far from the right time
        _jump("2025-08-01 09:00", -30, 1100, event=90),              # an event stamped before the one preceding it, outside a clock change
        _jump("2025-09-01 09:00", 1e7, 1300),                        # more than a month: the pump's clock set at first use
    ])
    changes, checks, issues = audit.clock_jumps(jumps, YEAR, zone=NEW_YORK)
    assert changes["class"].tolist() == ["one hour", "time-zone change (2-3.5 hours)", "time-zone change (2-3.5 hours)", "other jump", "pump set-up"]
    assert changes["follows_clock_change"].tolist()[0] and changes["hours_after_clock_change"].iloc[0] == pytest.approx(5.0)   # the wall clock reads 03:00 right after the transition
    status = checks.set_index("check")["status"]
    assert status["Other clock jumps"] == "concern" and status["Timestamp inversions"] == "concern"
    assert status["Shared clock of glucose, bolus and settings"] == "not assessed"
    assert issues["step"].value_counts().to_dict() == {"pump clock jump": 5, "timestamp inversion": 1}
    assert set(checks["status"]) <= STATUSES


def test_time_away_from_home_follows_the_running_sum_and_ends_when_the_pump_is_back():
    out_and_back = pd.DataFrame([_jump("2025-06-01 09:00", 180, 500), _jump("2025-06-11 09:00", -180, 7)])
    _, checks, _ = audit.clock_jumps(out_and_back, YEAR, zone=NEW_YORK)
    assert checks.set_index("check").loc["Time spent on a clock away from home", "result"].startswith("about 10 days")
    with_other = pd.DataFrame([_jump("2025-06-01 09:00", 180, 500), _jump("2025-06-05 09:00", 960, 700), _jump("2025-06-11 09:00", -1140, 9)])   # an odd 16 hours on top of the 3, then home
    _, checks, _ = audit.clock_jumps(with_other, YEAR, zone=NEW_YORK)
    assert checks.set_index("check").loc["Time spent on a clock away from home", "result"].startswith("about 10 days")


def test_a_local_clock_change_the_pump_never_followed_is_counted():
    followed = pd.DataFrame([_jump("2025-03-09 08:00", 60, 1)])
    _, checks, _ = audit.clock_jumps(followed, YEAR, zone=NEW_YORK)
    row = checks.set_index("check").loc["Clock changes at local clock-change dates"]
    assert row["result"].startswith("1 of 2 transitions uncorrected") and row["status"] == "concern"
    both = pd.DataFrame([_jump("2025-03-09 08:00", 60, 100), _jump("2025-11-02 09:00", -60, 3)])
    _, checks, _ = audit.clock_jumps(both, YEAR, zone=NEW_YORK)
    assert checks.set_index("check").loc["Clock changes at local clock-change dates", "status"] == "no concern"


def _unknown_table() -> pd.DataFrame:
    """Four weeks of hours: hours 0-5 of the second half have unknown settings and the same time in range as known hours of those hours."""
    hours = pd.date_range("2025-01-01", periods=28 * 24, freq="h")
    rng = np.random.default_rng(3)
    table = pd.DataFrame({"hour_start": hours, "date": hours.normalize(), "hour": hours.hour, "n_valid": 12, "excluded_day": False})
    table["settings_known_slots"] = 12
    table.loc[(table["date"] >= "2025-01-15") & (table["hour"] < 6), "settings_known_slots"] = 0
    night = table["hour"] < 6
    table["n_in_range"] = np.where(night, rng.binomial(12, 0.55, len(table)), rng.binomial(12, 0.85, len(table)))
    return table


def test_composition_test_removes_a_gap_that_is_only_the_hour_of_day_mix():
    summary, check = audit.composition_test(_unknown_table(), draws=200, seed=1)
    crude, standardised, unknown = summary["time_in_range_pct"]
    assert unknown < crude - 10                      # unknown hours look worse as observed: they are night hours
    assert abs(standardised - unknown) < 6           # the known night hours match them
    assert check["status"].iloc[0] == "no concern"
    repeat, _ = audit.composition_test(_unknown_table(), draws=200, seed=1)
    pd.testing.assert_frame_equal(summary, repeat)   # the seed fixes the intervals


def test_composition_test_is_not_applicable_without_unknown_hours():
    table = _unknown_table().assign(settings_known_slots=12)
    summary, check = audit.composition_test(table)
    assert summary.empty and check["status"].tolist() == ["not applicable"]


def test_leakage_table_flags_columns_the_forecast_cannot_know_when_the_model_uses_them():
    table = pd.DataFrame({name: np.arange(400.0) % 3 for name in ("icr", "cf", "basal", "target")})
    leak = audit.leakage_table(table, ["log_icr", "log_cf", "log_basal", "target_10", "log_carbs_3h", "exercise_share", "sleep_mode_share", "site_age_capped"]).set_index("column")
    status = leak["status"]
    assert status["hour, date, dst_day"] == "no concern" and status["icr, cf, basal, target"] == "no concern"
    for column in ("carbs_prev3h_g", "exercise_share", "sleep_mode_share", "site_age_days", "excluded_day"):
        assert status[column] == "concern"
    assert status["apple_asleep_share"] == "not applicable" and status["bolus_u, carbs_g"] == "not applicable"
    assert leak.loc["icr, cf, basal, target", "statistic"].startswith("hours whose value differs")
    unused = audit.leakage_table(table, ["log_icr"]).set_index("column")["status"]
    assert unused["carbs_prev3h_g"] == "not applicable" and unused["excluded_day"] == "concern"   # the row filter is not optional


def test_meal_checks_pair_entries_and_boluses_within_ten_minutes():
    entries = pd.DataFrame({"event_time": pd.to_datetime(["2025-09-01 08:00", "2025-09-01 12:00", "2025-09-01 18:00"]), "carb_grams": [30.0, 45.0, 20.0], "pump_serial": PUMP})
    entries["timestamp"] = entries["event_time"]
    bolus = pd.DataFrame({"timestamp": pd.to_datetime(["2025-09-01 08:02", "2025-09-01 12:50", "2025-09-01 21:00"]), "bolus_units": [3.0, 1.0, 2.0], "bolus_type_raw": [16, 8, 16],
                          "bolus_source_raw": [1, 7, 1], "requested_later_units": [0.0, 0.0, 0.0], "pump_serial": PUMP})
    checks, issues = audit.meal_checks(entries, bolus)
    assert issues.groupby("step")["n"].sum().to_dict() == {"carbohydrate entry without a bolus": 2, "carbohydrate bolus without an entry": 1}
    assert "nearest bolus 50 min away" in issues["value"].tolist()[0]
    assert set(checks["status"]) <= STATUSES


def test_event_log_reader_finds_clock_changes_in_a_binary_window(tmp_path):
    def record(event: int, seconds: int, sequence: int) -> bytes:
        return struct.pack(">HII", event, seconds, sequence) + bytes(26 - 10)

    start = int((pd.Timestamp("2025-09-01 08:00") - report.TANDEM_EPOCH).total_seconds())
    records = [record(90, start, 1), record(90, start + 300, 2), record(13, start + 3600 + 600, 3), record(90, start + 3600 + 900, 4)]
    raw = tmp_path / "tconnectsync" / "2025-09-01__2025-09-02" / "raw"
    raw.mkdir(parents=True)
    (raw / f"pump_events_raw_{PUMP}_x.txt").write_text(base64.b64encode(b"".join(records)).decode())
    log = report.read_event_log(tmp_path / "tconnectsync")
    assert log["jumps"]["event"].tolist() == [13] and log["jumps"]["jump_minutes"].iloc[0] == pytest.approx(65.0)
    assert log["overview"]["events"].tolist() == [4]


def test_the_audit_has_its_own_output_folder():
    paths = ProjectPaths.from_root(Path("/nonexistent/project"))
    assert paths.output_data_audit.parts[-2:] == ("output", "data_audit")
    assert paths.output_data_audit.parent == paths.output_outcomes_model.parent


def test_cli_passes_its_options_to_the_runner(monkeypatch):
    seen = {}
    monkeypatch.setattr(report, "run", lambda raw, output: seen.update(raw=raw, output=output) or 0)
    assert audit.main(["--raw", "a", "--output", "b"]) == 0 and seen == {"raw": "a", "output": "b"}


def _long_analysis(days: int = 120):
    """The two-day synthetic record stretched to ``days`` days: the profile history shows only the first two, so the rest has unknown settings."""
    analysis = _analysis()
    slots = pd.date_range(DAY1, periods=days * 288, freq="5min")
    rng = np.random.default_rng(11)
    glucose = np.round(np.clip(130 + 45 * np.sin(slots.hour.to_numpy() / 24 * 2 * np.pi) + rng.normal(0, 25, len(slots)), 45, 380))
    analysis.cgm = pd.DataFrame({"timestamp": slots, "glucose": glucose, "pump_serial": PUMP})
    grid = pd.DataFrame({"timestamp": slots, "carb_grams": 0.0, "bolus_units": 0.0})
    meals = grid["timestamp"].dt.hour.isin([8, 13, 19]) & grid["timestamp"].dt.minute.eq(5)
    grid.loc[meals, ["carb_grams", "bolus_units"]] = [40.0, 4.0]
    analysis.grid = grid
    return analysis


def test_workbook_written_from_the_checks_reconciles_with_the_hour_day_table(tmp_path):
    analysis = _long_analysis()
    table = build_hour_day_table(analysis)
    analysis.merged = _merged(analysis)
    results = report.run_checks(analysis, table, {"jumps": pd.DataFrame(columns=["pump_serial", "sequence", "timestamp", "previous_timestamp", "event", "jump_minutes"]),
                                                  "sessions": pd.DataFrame(columns=["pump_serial", "timestamp", "kind"]), "overview": pd.DataFrame()})
    sheets = report.build_sheets(results, table, analysis, raw="synthetic", run_id="test", seconds=1.0, files=["diagnostics.xlsx"])
    names = list(sheets)
    assert names[0] == "Specification_Coding" and names[1] == "Denominators_Flow" and names[-1] == "Artifacts_Provenance" and names[-2] == "Warnings_Limitations"
    for name in report.CHECK_SHEETS:
        assert set(sheets[name]["status"]) <= STATUSES and sheets[name]["reason"].notna().all()
    path = write_workbook(tmp_path / "diagnostics.xlsx", sheets)
    report.reconcile(path, table, results)
    flow = read_workbook(path)["Denominators_Flow"].set_index("stage")["n"]
    assert flow["hours without readings"] == 0 and flow["hours on context-event days"] == 24
    issues = read_workbook(path)["Record_Issues"]
    assert issues.loc[issues["step"].eq("context-event day (hours)"), "n"].sum() == 24
    # a wrong count in the workbook is caught
    broken = {**sheets, "Denominators_Flow": sheets["Denominators_Flow"].assign(n=lambda f: np.where(f["stage"].eq("clock hours in the record"), 49, f["n"]))}
    with pytest.raises(RuntimeError, match="clock hours in the record"):
        report.reconcile(write_workbook(tmp_path / "broken.xlsx", broken), table, results)
    unflagged = {**sheets, "Leakage": sheets["Leakage"].assign(status=lambda f: np.where(f["column"].eq("site_age_days"), "no concern", f["status"]))}
    with pytest.raises(RuntimeError, match="site_age_days"):
        report.reconcile(write_workbook(tmp_path / "unflagged.xlsx", unflagged), table, results)
    miscounted = {**sheets, "Denominators_Flow": sheets["Denominators_Flow"].assign(n=lambda f: np.where(f["stage"].str.startswith("hours with readings and settings"), f["n"] + 1, f["n"]))}
    with pytest.raises(RuntimeError, match="reasons leave"):
        report.reconcile(write_workbook(tmp_path / "miscounted.xlsx", miscounted), table, results)
    summary = tmp_path / "summary.md"
    report.write_summary(summary, read_workbook(path), "test")
    assert "test" in summary.read_text()
    figures = report.figure_cgm(results, table, tmp_path) + report.figure_settings(results, table, tmp_path)
    assert all(p.exists() and p.stat().st_size > 0 for p in figures)


def _merged(analysis):
    from types import SimpleNamespace

    grid = analysis.grid
    meal = grid[grid["carb_grams"] > 0]
    bolus = grid[grid["bolus_units"] > 0]
    carbs = pd.DataFrame({"timestamp": meal["timestamp"], "event_time": meal["timestamp"], "carb_grams": meal["carb_grams"], "pump_serial": PUMP,
                          "carb_ratio_g_per_unit": 10.0, "isf_mgdl_per_unit": 65.0, "target_bg_mgdl": 110.0})
    boluses = pd.DataFrame({"timestamp": bolus["timestamp"], "bolus_units": bolus["bolus_units"], "pump_serial": PUMP, "bolus_type_raw": 16, "bolus_source_raw": 1, "requested_later_units": 0.0})
    basal = pd.DataFrame({"start_timestamp": [DAY1], "end_timestamp": [DAY1 + pd.Timedelta(days=2)], "basal_units_per_hour": 0.8, "pump_serial": PUMP})
    cgm = analysis.cgm.assign(duplicate_count=1, source_event_name="GXB")
    rates = pd.DataFrame({"pump_serial": [PUMP], "timestamp": [DAY1 + pd.Timedelta(hours=12)], "rate": [0.8]})
    return SimpleNamespace(tandem_data=SimpleNamespace(cgm=cgm, basal=basal, bolus=boluses, carbs=carbs), profile_events=analysis.history.events, profile_rates=rates,
                           data=analysis.merged.data, manifest={"dataset_version": "synthetic", "built_at": "n/a"})


def test_zero_readings_are_reported_apart_from_readings_of_1_to_39():
    stamps = _slots("2025-09-01", 30)
    glucose = np.linspace(100, 160, 30).round()
    glucose[3], glucose[4], glucose[5] = 0.0, 0.0, 35.0
    raw = pd.DataFrame({"timestamp": stamps, "glucose": glucose, "pump_serial": PUMP, "duplicate_count": 1})
    checks, issues = audit.cgm_duplicates_and_values(raw, raw[raw["glucose"] >= 40])
    result = checks.set_index("check")["result"]
    assert result["Zero-valued readings (sensor error)"] == 2 and result["Readings below the sensor floor (LOW)"].startswith("1;")
    assert issues["reason"].value_counts().to_dict() == {"reading of 0 mg/dL (sensor error)": 2, "reading of 1 to 39 mg/dL (LOW)": 1}


def test_pump_eras_overlap_is_read_from_slots_not_whole_days():
    table = pd.DataFrame({"n_valid": [12, 12]})
    first, second = _slots("2025-09-01 20:00", 12), _slots("2025-09-01 21:00", 12)       # the hand-over is on one date, in different slots
    raw = pd.DataFrame({"timestamp": list(first) + list(second), "pump_serial": ["900001"] * 12 + ["900002"] * 12})
    row = audit.pump_eras(raw, table).set_index("check").loc["Pump eras overlap"]
    assert row["status"] == "no concern" and row["result"] == "0 slots; 1 dates"
    raw.loc[raw["pump_serial"].eq("900002"), "timestamp"] = list(first)
    assert audit.pump_eras(raw, table).set_index("check").loc["Pump eras overlap", "status"] == "concern"


def test_a_pump_change_with_no_readings_in_a_block_is_not_assessed():
    hours = pd.date_range("2025-01-01", periods=60 * 24, freq="h")
    table = pd.DataFrame({"date": hours.normalize(), "hour": hours.hour, "pump_serial": np.where(hours < "2025-02-10", "900001", "900002"), "n_valid": 0,
                          "n_in_range": 0, "n_below_70": 0, "glucose_sum": 0.0, "n_below_54": 0, "n_above_180": 0})
    assert audit.pump_change_break(table)["status"].tolist() == ["not assessed"] * 3


def _gap_table(low_next_to_gap: int, low_elsewhere: int) -> pd.DataFrame:
    hours = pd.date_range("2025-01-01", periods=200 * 24, freq="h")
    table = pd.DataFrame({"date": hours.normalize(), "hour": hours.hour, "n_valid": 12, "n_in_range": 10, "n_below_70": low_elsewhere, "n_above_180": 0, "settings_known_slots": 12, "excluded_day": False})
    table.loc[::24, "n_valid"] = 0
    table.loc[table["n_valid"].eq(0), "n_below_70"] = 0
    neighbours = (table["n_valid"].shift(1).eq(0) | table["n_valid"].shift(-1).eq(0)) & table["n_valid"].gt(0)
    table.loc[neighbours, "n_below_70"] = low_next_to_gap
    return table


def test_hours_next_to_a_gap_are_judged_by_a_ratio_of_time_below_70():
    status = lambda t: audit.missingness(t).set_index("check").loc["Hours next to a gap in readings (is missing not at random?)", "status"]   # noqa: E731
    assert status(_gap_table(3, 1)) == "concern"      # 3 times as much time below 70 next to a gap, though the difference is only a few points
    assert status(_gap_table(1, 1)) == "no concern"


def test_unknown_hours_clustered_in_one_month_are_a_concern_even_when_spread_over_the_day():
    hours = pd.date_range("2025-01-01", periods=300 * 24, freq="h")
    table = pd.DataFrame({"date": hours.normalize(), "hour": hours.hour, "n_valid": 12, "n_in_range": 9, "n_below_70": 0, "n_above_180": 0, "settings_known_slots": 12, "excluded_day": False})
    table.loc[(table["date"] < "2025-01-15"), "settings_known_slots"] = 0
    row = audit.missingness(table).set_index("check").loc["Hours with settings not known for all 12 moments"]
    assert row["status"] == "concern" and "in 2025-01" in row["reason"]


def test_composition_test_leaves_out_unknown_hours_in_cells_without_known_hours():
    table = _unknown_table()
    table.loc[(table["date"] >= "2025-01-15") & (table["hour"] == 23), "settings_known_slots"] = 0   # unknown at 23:00 whole record: no known hour in those cells after mid-January
    table.loc[(table["date"] < "2025-01-15") & (table["hour"] == 23), "n_valid"] = 0                 # and no known 23:00 hour before
    summary, check = audit.composition_test(table, draws=100, seed=2)
    assert summary["readings"].iloc[2] < (table["settings_known_slots"].lt(12) & table["n_valid"].gt(0)).mul(table["n_valid"]).sum()
    assert "left out of both sides" in check["result"].iloc[0] and not check["result"].iloc[0].split("; ")[-1].startswith("0.0%")


def test_composition_test_leaves_out_context_event_days():
    table = _unknown_table()
    with_all, _ = audit.composition_test(table, draws=50, seed=2)
    table.loc[table["date"].between("2025-01-16", "2025-01-20"), "excluded_day"] = True
    without, _ = audit.composition_test(table, draws=50, seed=2)
    assert without["readings"].iloc[2] < with_all["readings"].iloc[2]


def test_carbohydrate_windows_are_whole_windows_and_a_window_without_entries_counts():
    days = pd.date_range("2025-01-01", periods=60)
    when = list(days[:14]) + list(days[28:56]) + list(days[56:58])            # window 1 (days 14-27) has no entry; days 56-59 are a partial window
    entries = pd.DataFrame({"event_time": pd.to_datetime(when) + pd.Timedelta(hours=8), "carb_grams": 30.0, "pump_serial": PUMP})
    entries["timestamp"] = entries["event_time"]
    bolus = pd.DataFrame({"timestamp": entries["event_time"], "bolus_units": 3.0, "bolus_type_raw": 16, "bolus_source_raw": 1, "requested_later_units": 0.0, "pump_serial": PUMP})
    row = audit.meal_checks(entries, bolus)[0].set_index("check").loc["Carbohydrate entries per 14-day window"]
    assert row["result"].startswith("1 of 4 whole windows (1 with no entry") and row["status"] == "concern"


def test_stationarity_leaves_out_the_days_the_model_drops():
    days = pd.date_range("2025-01-01", periods=70)
    table = pd.DataFrame({"date": np.repeat(days, 24), "hour": np.tile(np.arange(24), 70), "n_valid": 12, "n_in_range": 9, "n_below_70": 0, "n_below_54": 0, "n_above_180": 3, "glucose_sum": 12 * 120.0,
                          "settings_known_slots": 12, "excluded_day": False})
    table.loc[table["date"].between("2025-03-01", "2025-03-10"), ["n_in_range", "excluded_day"]] = [0, True]   # ten days of poor glucose, all context-event days
    assert audit.fortnight_series(table)["tir_pct"].eq(75.0).all()
    assert audit.daily_series(table)["tir_pct"].eq(75.0).all()
    assert len(audit.daily_series(table)) == 60


def test_calculator_agreement_is_not_assessed_without_comparable_meals():
    history = _analysis().history
    meal = pd.Timestamp("2026-03-01 10:00")           # outside the span the profile history shows
    carbs = pd.DataFrame({"event_time": [meal], "timestamp": [meal], "pump_serial": [PUMP], "carb_ratio_g_per_unit": [10.0], "isf_mgdl_per_unit": [65.0], "target_bg_mgdl": [110.0]})
    checks, issues = audit.calculator_agreement(carbs, history)
    assert checks["status"].tolist() == ["not assessed"] * 3 and issues.empty


def test_the_reason_of_an_hour_with_tied_moments_does_not_depend_on_the_hash_seed():
    from types import SimpleNamespace

    first, second = audit.UNKNOWN_REASONS[1], audit.UNKNOWN_REASONS[4]
    row = [first] * 6 + [second] * 6
    history = SimpleNamespace(unknown_reasons=lambda pumps, moments: np.array((row * 3)[: len(moments)], dtype=object))
    table = pd.DataFrame({"hour_start": pd.date_range("2025-09-01", periods=3, freq="h"), "pump_serial": PUMP})
    assert audit.hour_unknown_reasons(history, table).tolist() == [first] * 3


def test_reason_and_unknown_colors_are_not_the_pump_colors():
    from bayesian_t1dm.figures import PUMP_STYLE, REASON_COLORS, UNKNOWN_SETTINGS_COLOR

    pumps = {style.color for style in PUMP_STYLE.values()}
    assert not pumps & set(REASON_COLORS.values()) and UNKNOWN_SETTINGS_COLOR not in pumps
    assert len(set(REASON_COLORS.values())) == len(REASON_COLORS)
