from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesian_t1dm import replay
from bayesian_t1dm.diagnostics_workbook import STATUSES, read_workbook, write_workbook
from bayesian_t1dm.figures import assert_ascii, format_p
from bayesian_t1dm.replay_report import write_replay_outputs

START = pd.Timestamp("2024-01-01")
CHANGE_DAYS = list(range(60, 400, 35))  # 10 changes, 35 days apart: each is isolated


def _table(*, shift: float = 0.0, days: int = 460, change_days=CHANGE_DAYS, seed: int = 3) -> pd.DataFrame:
    """A synthetic hour-day table: 70% of readings in range on average, a settings change on ``change_days`` at hours 8-12, and ``shift`` added to the in-range share on the 14 days after each change."""
    rng = np.random.default_rng(seed)
    day_level = np.clip(0.70 + rng.normal(0, 0.04, days), 0.4, 0.95)
    for day in change_days:
        day_level[day + 1: day + 1 + replay.AFTER_DAYS] += shift
    hour_start = pd.date_range(START, periods=days * 24, freq="h")
    p_range = np.repeat(day_level, 24)
    below = 0.03
    counts = np.array([rng.multinomial(12, [p, below, max(1 - p - below, 0)]) for p in p_range])
    table = pd.DataFrame({"hour_start": hour_start, "date": hour_start.normalize(), "hour": hour_start.hour,
                          "n_valid": 12, "n_in_range": counts[:, 0], "n_below_70": counts[:, 1], "n_below_54": 0, "n_above_180": counts[:, 2],
                          "glucose_sum": 12 * 130.0, "settings_known_slots": 12, "settings_mixed": False, "excluded_day": False})
    table["settings_changed"] = False
    for name in ("icr", "cf", "basal", "target"):
        table["d_" + name] = 0.0
    for day in change_days:
        rows = (table["date"] == START + pd.Timedelta(days=day)) & table["hour"].between(8, 12)
        table.loc[rows, "settings_changed"] = True
        table.loc[rows, "d_icr"] = -0.5  # a smaller I:C gives more insulin
    return table


def test_a_planted_shift_is_recovered_and_no_shift_covers_zero():
    shifted = replay.summarize(replay.run_replay(_table(shift=0.10))["scores"])
    row = shifted.query("group == 'all changes' and predictor == @replay.PRIMARY and measure == 'tir'").iloc[0]
    assert row["delta"] == pytest.approx(10.0, abs=2.5)
    assert row["delta_lower_90"] > 0


def test_without_a_planted_shift_the_estimate_is_near_zero_and_its_interval_usually_covers_it():
    covered, deltas = 0, []
    for seed in (3, 4, 5, 6):   # a 90% interval misses zero by chance about one time in ten, so judge several records
        plain = replay.summarize(replay.run_replay(_table(shift=0.0, seed=seed))["scores"])
        row = plain.query("group == 'all changes' and predictor == @replay.PRIMARY and measure == 'tir'").iloc[0]
        covered += int(row["delta_lower_90"] < 0 < row["delta_upper_90"])
        deltas.append(row["delta"])
    assert covered >= 3
    assert np.abs(np.mean(deltas)) < 1.0


def test_changes_five_days_apart_share_a_cluster_and_are_not_isolated():
    grid = replay.day_grid(_table(change_days=[60, 65, 200]))
    events = replay.find_events(grid)
    clusters, isolated = replay.cluster_events(events)
    assert [event.first for event in events] == [60, 65, 200]
    assert clusters[0] == clusters[1] != clusters[2]
    assert isolated == {2}


def test_consecutive_days_form_one_event_with_a_direction_and_affected_hours():
    table = _table(change_days=[60, 61])
    events = replay.find_events(replay.day_grid(table))
    assert len(events) == 1 and events[0].span == 2
    assert events[0].direction == "more insulin" and events[0].settings == "I:C"
    assert np.flatnonzero(events[0].affected_hours).tolist() == list(range(8, 17))   # changed hours 8-12 plus 4 more


def test_placebo_dates_sit_clear_of_every_change():
    results = replay.run_replay(_table())
    grid = replay.day_grid(_table())
    flagged = np.flatnonzero(grid.changed.any(axis=1))
    position = {date.date(): index for index, date in enumerate(grid.dates)}
    placebos = results["scores"].query("kind != 'change'")["date"]
    assert len(placebos) > 0
    for date in placebos:
        assert np.abs(flagged - position[date]).min() > replay.CLEARANCE_DAYS


def test_predictions_use_only_days_before_the_change():
    table = _table()
    grid = replay.day_grid(table)
    event = replay.find_events(grid)[2]
    altered = table.copy()
    after = altered["hour_start"] >= START + pd.Timedelta(days=event.first)
    altered.loc[after, "n_in_range"] = 0
    altered.loc[after, "n_below_70"] = 12
    base = replay.score_target(grid, event.first, event.span, event.affected_hours, key=(event.event_id, 0))
    mutated = replay.score_target(replay.day_grid(altered), event.first, event.span, event.affected_hours, key=(event.event_id, 0))
    for predictor in replay.PREDICTORS:
        assert mutated[predictor]["tir_predicted"] == base[predictor]["tir_predicted"]
        assert mutated[predictor]["tir_observed"] != base[predictor]["tir_observed"]  # the outcome did change, the prediction did not


def test_changes_without_enough_data_are_listed_with_a_reason():
    results = replay.run_replay(_table(change_days=[10, *CHANGE_DAYS]))   # day 10 has only 10 days of history
    first = results["issues"].iloc[0]
    assert (first["event_id"], first["step"], first["field"], first["value"]) == (0, "enough data", "qualifying days before", 10)
    assert results["flow"].set_index("stage").loc["changes (consecutive days merged)", "n"] == len(CHANGE_DAYS) + 1


def test_the_run_fails_when_no_change_can_be_scored():
    with pytest.raises(RuntimeError, match="no events left"):
        replay.run_replay(_table(change_days=[5]))


def test_outputs_workbook_and_figure(tmp_path):
    table = _table(shift=0.05)
    results = replay.run_replay(table)
    write_replay_outputs(results, table, tmp_path, run_id="test", raw="synthetic", seconds=1.0)
    workbook = read_workbook(tmp_path / "diagnostics.xlsx")
    assert list(workbook) == ["Specification_Coding", "Denominators_Flow", "Replay_Checks", "Record_Issues", "Warnings_Limitations", "Artifacts_Provenance"]
    assert set(workbook["Replay_Checks"]["status"]) <= set(STATUSES)
    assert (tmp_path / "replay_figure.png").stat().st_size > 10_000 and (tmp_path / "replay_figure.svg").exists()
    assert (tmp_path / "replay_summary.csv").exists() and (tmp_path / "replay_scores.csv").exists()


def test_workbook_rejects_a_status_outside_the_four_and_missing_sheets(tmp_path):
    sheets = {name: pd.DataFrame({"a": [1]}) for name in ("Specification_Coding", "Denominators_Flow", "Warnings_Limitations", "Artifacts_Provenance")}
    write_workbook(tmp_path / "ok.xlsx", sheets)
    with pytest.raises(ValueError, match="statuses"):
        write_workbook(tmp_path / "bad.xlsx", {**sheets, "Checks": pd.DataFrame({"status": ["passed"]})})
    with pytest.raises(ValueError, match="lacks required sheets"):
        write_workbook(tmp_path / "missing.xlsx", {"Specification_Coding": sheets["Specification_Coding"]})


def test_figure_text_helpers():
    assert format_p(0.0004) == "p < 0.001" and format_p(0.0123) == "p = 0.012"
    assert_ascii("time in range 70-180 (percentage points)")
    with pytest.raises(ValueError):
        assert_ascii("± 5")
