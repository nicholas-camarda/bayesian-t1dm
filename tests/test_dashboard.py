from __future__ import annotations

import json

import numpy as np
import pandas as pd

from bayesian_t1dm import dashboard
from bayesian_t1dm.dashboard_page import render_dashboard


def test_clean_makes_values_json_safe():
    value = {"a": np.float64("nan"), "b": np.int64(3), "c": pd.Timestamp("2026-01-02 03:04"), "d": [np.inf, 1.5], "e": pd.NaT, "f": np.bool_(True)}
    assert dashboard.clean(value) == {"a": None, "b": 3, "c": "2026-01-02T03:04:00", "d": [None, 1.5], "e": None, "f": True}
    json.dumps(dashboard.clean(value), allow_nan=False)


def test_script_json_cannot_close_the_script_element():
    text = dashboard.to_script_json({"x": "</script><!-- hi"})
    assert "</" not in text and "<!--" not in text and json.loads(text.replace("<\\/", "</").replace("<\\!--", "<!--"))["x"] == "</script><!-- hi"


def _segments(**over):
    row = {"hours": "15:00-19:00", "tir_90d": 61.0, "tir_14d": 44.0, "below_70_90d": 0.4, "below_54_90d": 0.0, "above_180_90d": 38.0}
    row.update(over)
    return pd.DataFrame([row])


EMPTY_EPISODES = {"days": 90, "highs": [0] * 24, "lows": [0] * 24, "high_total": 0, "low_total": 0}
NO_SIGNALS = pd.DataFrame(columns=["part", "window_days", "days", "clean_hours", "share_over_x", "share_over_lo", "share_over_hi", "pooled_excess", "mean_glucose", "signal"])


def test_problems_list_lows_first_then_evidence_then_what_to_watch():
    evidence = [{"segment": "19:00", "programmed": 6.8, "effective": 5.0, "lower": 4.5, "upper": 5.5, "n": 48, "current": True, "now": 6.8, "latest": False}]
    signals = pd.DataFrame([{"part": "00:00-05:00", "window_days": 28, "days": 28, "clean_hours": 120.0, "share_over_x": 0.68, "share_over_lo": 0.5, "share_over_hi": 0.8, "pooled_excess": 0.46, "mean_glucose": 148.0, "signal": "basal likely too low", "strict_signal": "basal likely too low"}])
    segments = pd.concat([_segments(), _segments(hours="00:00-05:00", below_70_90d=5.0)], ignore_index=True)
    areas = dashboard.find_problem_areas(segments, evidence, signals, EMPTY_EPISODES, {})
    titles = [a["title"] for a in areas]
    assert titles[0].startswith("Lows over a limit") and areas[0]["level"] == "act"
    assert titles[1] == "I:C at 19:00 looks too weak" and titles[2].startswith("Basal likely too low")
    assert any(a["level"] == "watch" and "below the 70% target, getting worse" in a["title"] for a in areas)


def test_a_setting_already_changed_is_reported_as_a_change_that_fits_not_as_a_problem():
    evidence = [{"segment": "05:00", "programmed": 7.0, "effective": 6.4, "lower": 6.1, "upper": 6.7, "n": 72, "current": False, "now": 6.5, "latest": True}]
    areas = dashboard.find_problem_areas(_segments(tir_90d=86.0, tir_14d=90.0), evidence, NO_SIGNALS, EMPTY_EPISODES, {})
    assert areas[0]["level"] == "ok" and "was changed to 6.5" in areas[0]["title"] and "fits them" in areas[0]["summary"]


def test_nothing_to_report_says_so():
    areas = dashboard.find_problem_areas(_segments(tir_90d=80.0, tir_14d=80.0), [], NO_SIGNALS, EMPTY_EPISODES, {})
    assert [a["title"] for a in areas] == ["Nothing needs attention"]


def test_agp_and_episode_counts_on_a_known_series():
    index = pd.date_range("2026-01-01", periods=288 * 10, freq="5min")
    glucose = np.where((index.hour >= 20) & (index.hour < 23), 300.0, 120.0)  # three hours at 300 every evening
    cgm = pd.DataFrame({"timestamp": index, "glucose": glucose})
    as_of = pd.Timestamp("2026-01-11")
    agp = dashboard.ambulatory_glucose_profile(cgm, as_of=as_of)
    assert len(agp) == 96 and agp[0]["p50"] == 120.0 and [r for r in agp if r["minute"] == 21 * 60][0]["p50"] == 300.0
    episodes = dashboard.episodes_by_hour(cgm, as_of=as_of)
    assert episodes["highs"][20] == 10 and episodes["high_total"] == 10 and episodes["low_total"] == 0


def test_the_page_embeds_its_data_and_has_every_tab():
    page = render_dashboard({"run": {}, "as_of": "2026-01-01", "problems": []})
    assert '"as_of":"2026-01-01"' in page and all(name in page for name in ("Today", "Glucose", "Meals (I:C)", "Basal", "Changes", "About"))
    assert "http://" not in page.replace("http://www.w3.org/2000/svg", "")  # self-contained: no network references


def test_basal_sensitivity_disagreement_is_visible_on_landing_page():
    signals = pd.DataFrame([{"part": "21:00-00:00", "window_days": 56, "days": 23, "clean_hours": 40.0,
                            "pooled_excess": 0.69, "mean_glucose": 181.0, "signal": "basal likely too low",
                            "strict_signal": "no consistent pattern", "strict_excess": 0.09}])
    areas = dashboard.find_problem_areas(_segments(tir_90d=80), [], signals, EMPTY_EPISODES, {})
    assert areas[0]["level"] == "watch"
    assert "depends on the clean-time rule" in areas[0]["title"]
    assert "+69%" in areas[0]["summary"] and "+9%" in areas[0]["summary"]
