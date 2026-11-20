"""The front page on synthetic outputs: the verdict states, a missing output, the status chips and the one-place rule (no clinical data)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from bayesian_t1dm import front_page as fp
from bayesian_t1dm import front_page_page as page
from bayesian_t1dm.cli import main
from bayesian_t1dm.diagnostics_workbook import write_workbook
from bayesian_t1dm.figures import CHANGE_SET_STYLE, PREDICTOR_STYLE, STATUS_STYLE, VERDICT_STYLE
from bayesian_t1dm.figures import INK_BLUE, INK_GOLD

FLOW = {"Denominators_Flow": pd.DataFrame({"set": ["evaluation"] * 2, "stage": ["weekly origins", "windows with at least 70% of their possible readings"], "n": [52, 51]})}
REQUIRED = {"Specification_Coding": pd.DataFrame({"item": ["Question"], "value": ["q"]}), "Denominators_Flow": pd.DataFrame({"stage": ["x"], "n": [1]}),
            "Warnings_Limitations": pd.DataFrame({"limitation": ["l"], "detail": ["d"]})}


def provenance(run_id: str, span: str = "2024-01-01 to 2026-11-17", clean: str = "yes") -> pd.DataFrame:
    return pd.DataFrame({"item": ["run id", "commit", "working tree clean", "record span", "bootstrap resamples"], "value": [run_id, "abcdef0123456789", clean, span, 10000]})


def win_row(candidate: str, reference: str, role: str, mean: float, lower: float, upper: float, block: int = 4) -> dict:
    verdict = "win" if upper < 0 else "loss" if lower > 0 else "tie"
    return {"candidate": candidate, "reference": reference, "role": role, "block_length_origins": block, "origins": 52, "mean_difference_points": mean, "lower_5th_points": lower,
            "upper_95th_points": upper, "verdict": verdict, "status": "not applicable", "reason": ""}


def write_forecast(run: Path, interval: tuple[float, float, float] = (-0.43, -0.67, -0.18), model_vs_365: tuple[float, float, float] = (-0.02, -0.20, 0.16)) -> None:
    mean, lower, upper = interval
    rows = [win_row("hour-day model", "last 28 days", "win test", mean, lower, upper), win_row("last 14 days", "last 28 days", "power check", 0.46, 0.14, 0.77)]
    rows += [win_row(name, "last 28 days", "reported, does not gate", -0.2, -0.5, 0.1) for name in PREDICTOR_STYLE if name not in ("last 28 days", "last 14 days", "hour-day model", "last 365 days")]
    rows.append(win_row("last 365 days", "last 28 days", "reported, does not gate", -0.41, -0.66, -0.15))
    rows += [win_row("hour-day model", "last 365 days", "pairwise, does not gate", *model_vs_365), win_row("hour-day model", "all history", "pairwise, does not gate", -0.30, -0.57, -0.04),
             win_row("hour-day model", "hour-day model (settings pinned)", "pairwise, does not gate", -0.63, -1.07, -0.21),
             win_row("hour-day model", "last 28 days", "sensitivity of the win test (block length)", mean, lower, upper, block=2)]
    warnings = pd.DataFrame({"limitation": ["Resampled exposures"], "detail": ["each draw uses 14 days resampled from the 28 days before it"]})
    sheets = {**REQUIRED, **FLOW, "Warnings_Limitations": warnings, "Win_Test": pd.DataFrame(rows),
              "Settings_Changes": pd.DataFrame({"windows": ["settings changed inside the window", "settings unchanged"], "origins": [13, 39]}), "Artifacts_Provenance": provenance("20260101T000000Z")}
    write_workbook(run / fp.FORECAST_FILE, sheets)


def write_replay(run: Path, primary: tuple[float, float, float] | None = (-1.1, -7.6, 0.4), power: float | None = 0.3, drop_power: bool = False) -> None:
    def row(outcome: str, group: str, changes: int, clusters: int, slope: float, lower: float, upper: float) -> dict:
        return {"variant": "primary", "outcome": outcome, "group": group, "changes": changes, "clusters": clusters, "slope": slope, "slope_lower_5th": lower, "slope_upper_95th": upper,
                "reading": "the slope interval holds zero" if lower == lower else "too few changes to estimate a slope"}
    tracking = pd.DataFrame([row("tir", "isolated changes", 10, 10, *(primary or (float("nan"),) * 3)), row("tir", "all changes", 18, 15, -0.9, -2.8, 0.1), row("below54", "all changes", 18, 15, -3.8, -7.5, -0.6)])
    information = pd.DataFrame([{"outcome": "tir", "group": "isolated changes", "changes": 10, "clusters": 10, "mean_crps_new_points": 4.46, "mean_crps_old_points": 3.57, "mean_difference_points": 0.89,
                                 "lower_5th": 0.31, "upper_95th": 1.52, "reading": "r"}])
    powers = pd.DataFrame([{"outcome": "tir", "group": group, "changes": 1, "clusters": 1, "power_true_slope_1": power} for group in ("isolated changes", "all changes") if not drop_power])
    if drop_power:
        powers = pd.DataFrame(columns=["outcome", "group", "changes", "clusters", "power_true_slope_1"])
    flow = pd.DataFrame({"stage": ["changes in the record", "predicted (after the fits)", "of which isolated"], "n": [45, 18, 10], "unit": ["changes"] * 3})
    write_workbook(run / fp.REPLAY_FILE, {**REQUIRED, "Denominators_Flow": flow, "Tracking": tracking, "Settings_Information": information, "Power": powers, "Artifacts_Provenance": provenance("20260102T000000Z")})


def write_audit(run: Path, *, not_assessed: bool = True) -> None:
    def sheet(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
        return pd.DataFrame([{"check": c, "source": "s", "statistic": "t", "result": f"result of {c}", "rule": "r", "status": status, "reason": f"reason {c}"} for c, status, _ in rows])
    shared = "not assessed" if not_assessed else "no concern"
    sheets = {
        "Inventory_Coverage": sheet([("cgm rows", "concern", "")]), "CGM_Quality": sheet([("duplicates", "no concern", "")]),
        "Devices_Clock": sheet([("Time spent on a clock away from home", "concern", ""), ("Other clock jumps", "no concern", ""), ("Shared clock of glucose, bolus and settings", shared, "")]),
        "Settings_Reconstruction": sheet([("Hours with readings whose settings are not known for all 12 moments", "concern", "")]), "Meals_Insulin": sheet([("entries", "not applicable", "")]),
        "Missingness_Stationarity": sheet([("Latest fortnight against earlier fortnights", "concern", ""), ("Trend in monthly time in range", "no concern", "")]),
        "Leakage": pd.DataFrame([{"column": "carbs_prev3h_g", "built_from": "b", "known_when": "concurrent with the hour", "used_by_outcomes_model": "yes", "statistic": "s", "status": "concern", "reason": "after the origin"},
                                 {"column": "icr", "built_from": "b", "known_when": "known at the forecast origin", "used_by_outcomes_model": "yes", "statistic": "s", "status": "no concern", "reason": "ok"},
                                 {"column": "bolus_u", "built_from": "b", "known_when": "concurrent", "used_by_outcomes_model": "no", "statistic": "s", "status": "not applicable", "reason": "unused"}]),
        "Artifacts_Provenance": provenance("20260103T000000Z"),
    }
    write_workbook(run / fp.AUDIT_FILE, {**REQUIRED, **sheets})


def write_proposal(run: Path) -> None:
    (run / fp.PROPOSAL_FILE).parent.mkdir(parents=True, exist_ok=True)
    (run / fp.PROPOSAL_FILE).write_text("| table |\n\n# Breakfast I:C proposal (2026-11-17)\n\n**No change proposed.** Waiting: earlier changes are still being evaluated.\n\n## Checks\n\n- x\n", encoding="utf-8")


@pytest.fixture
def run(tmp_path) -> Path:
    folder = tmp_path / "run"
    folder.mkdir()
    return folder


@pytest.fixture
def full(run) -> Path:
    write_forecast(run)
    write_replay(run)
    write_audit(run)
    write_proposal(run)
    return run


def render(run: Path, tmp_path: Path | None = None) -> str:
    return page.render_front_page(fp.gather(run), run)


def visible(text: str) -> str:
    """The page without tags and without the text alternatives, which repeat a chart's numbers for a screen reader."""
    return re.sub(r"<[^>]+>", " ", re.sub(r"aria-label='[^']*'", "", text))


@pytest.mark.parametrize(("interval", "state", "words"), [((-0.43, -0.67, -0.18), "win", "forecast 14-day time in range better"), ((-0.1, -0.5, 0.3), "tie", "cannot be told apart"),
                                                          ((0.4, 0.1, 0.7), "loss", "forecast 14-day time in range worse")])
def test_the_verdict_follows_the_interval(full, interval, state, words):
    write_forecast(full, interval)
    text = render(full)
    strip = text[text.index("<nav class='strip'"):text.index("</nav>")]
    assert f"{VERDICT_STYLE[state].label}</span>" in strip.split("2. Settings effects")[0]
    assert words in text[text.index("class='headline'"):].split("</p>")[0]
    assert f"{interval[0]:+.2f} points" in text and f"({interval[1]:+.2f} to {interval[2]:+.2f}" in text


def test_a_missing_output_is_a_stated_gap_not_a_fill(run):
    write_replay(run)
    write_audit(run)
    text = render(run)
    assert "The forecast harness result is not available" in text and "forecast_harness" in text and "python -m bayesian_t1dm.forecast_harness" in text
    assert "Prespecified win test" not in text and "chip outline" in text.split("2. Can the record")[0] and VERDICT_STYLE["missing"].label in text
    assert "The breakfast I:C proposal is not available" in text


def test_with_nothing_written_every_section_says_so(run):
    text = render(run)
    for what in ("forecast harness result", "replay-as-predictor result", "data-quality audit", "breakfast I:C proposal"):
        assert f"The {what} is not available" in text
    assert "Ranked setting changes are not shown" in text


def test_a_check_that_was_not_assessed_never_looks_like_a_pass(full):
    text = render(full)
    assert STATUS_STYLE["not assessed"].outline and not STATUS_STYLE["no concern"].outline
    not_assessed = re.search(r"<li class='check'>(?:(?!</li>).)*Shared clock(?:(?!</li>).)*</li>", text, re.S).group(0)
    assert "Not assessed" in not_assessed and "No concern" not in not_assessed and "chip outline" in not_assessed
    assert "never a pass" in text and "could not be run" in text
    clean = render_with(full, not_assessed=False)
    assert "could not be run" not in clean


def render_with(run, *, not_assessed: bool) -> str:
    write_audit(run, not_assessed=not_assessed)
    return render(run)


def test_the_overall_data_status_is_a_concern_while_any_check_is_a_concern(full):
    text = render(full)
    strip = text[text.index("<nav class='strip'"):text.index("</nav>")]
    assert "Concern" in strip.split("3. Data quality")[1] and "Concerns in" in strip


def test_settings_effects_are_not_shown_when_the_slope_interval_holds_zero(full):
    text = render(full)
    assert "Ranked setting changes are not shown" in text and VERDICT_STYLE["not shown"].label in text and "low power" in text
    write_replay(full, primary=(2.0, 0.4, 3.5), power=0.9)
    shown = render(full)
    assert VERDICT_STYLE["shown"].label in shown and "Ranked setting changes are not shown" not in shown and "low power" not in shown


def test_a_group_has_one_color_marker_and_line_type_everywhere(full):
    text = render(full)
    for name in ("hour-day model", "last 365 days"):
        assert PREDICTOR_STYLE[name].color in text
    isolated, every = CHANGE_SET_STYLE["isolated changes"], CHANGE_SET_STYLE["all changes"]
    assert isolated.color in text and every.color in text and isolated.linestyle == "-" and every.linestyle == "--"
    assert "border-top-style:dashed" in text and "border-top-style:solid" in text


def test_the_page_is_ascii_has_no_script_and_each_headline_number_is_printed_once(full):
    text = render(full)
    assert text.isascii() and "<script" not in text and "<img" not in text
    assert visible(text).count("-0.43") == 1 and visible(text).count("-0.67") == 1
    assert "lang='en'" in text and "name='viewport'" in text and "role='img'" in text


def test_the_proposal_keeps_its_own_wording(full):
    text = render(full)
    assert "<strong>No change proposed.</strong> Waiting: earlier changes are still being evaluated." in text and "Breakfast I:C proposal (2026-11-17)" in text


def test_a_workbook_missing_a_check_the_page_names_fails_loudly(full):
    sheets = pd.read_excel(full / fp.AUDIT_FILE, sheet_name=None)
    sheets["Devices_Clock"] = sheets["Devices_Clock"][~sheets["Devices_Clock"]["check"].eq("Other clock jumps")]
    write_workbook(full / fp.AUDIT_FILE, sheets)
    with pytest.raises(ValueError, match="Other clock jumps"):
        fp.gather(full)


@pytest.mark.parametrize("row", [{"candidate": "hour-day model", "role": "win test"}, {"candidate": "last 365 days", "role": "reported, does not gate"}])
def test_a_verdict_that_disagrees_with_its_interval_is_refused(full, row):
    sheets = pd.read_excel(full / fp.FORECAST_FILE, sheet_name=None)
    wanted = sheets["Win_Test"]["candidate"].eq(row["candidate"]) & sheets["Win_Test"]["role"].eq(row["role"]) & sheets["Win_Test"]["block_length_origins"].eq(4)
    sheets["Win_Test"].loc[wanted, "verdict"] = "loss"
    write_workbook(full / fp.FORECAST_FILE, sheets)
    with pytest.raises(ValueError, match="disagrees"):
        fp.gather(full)


def test_links_point_only_at_files_that_are_in_the_run(full):
    (full / "I_forecast").mkdir(exist_ok=True)
    (full / "I_forecast" / "forecast_harness_summary.md").write_text("s")
    (full / "dashboard.html").write_text("d")
    text = page.render_front_page(fp.gather(full), full)
    assert "href='I_forecast/forecast_harness_summary.md'" in text and "href='dashboard.html'" in text
    assert "href='J_replay_predictor/replay_predictor_summary.md'" not in text and "Figure (not written)" in text


def test_the_front_page_command_reads_the_runs_own_copies(full, tmp_path, monkeypatch):
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(tmp_path / "empty_runtime"))
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(tmp_path / "vault"))
    assert main(["--root", str(tmp_path / "source"), "front-page", "--run-dir", str(full)]) == 0
    text = (full / "index.html").read_text(encoding="utf-8")
    assert "Can this record say which pump settings to change?" in text and "Prespecified win test" in text


def _strip_card(text: str, title: str) -> str:
    strip = text[text.index("<nav class='strip'"):text.index("</nav>")]
    return strip.split(title)[1].split("<a class='scard'")[0]


@pytest.mark.parametrize(("interval", "model_vs_365", "wanted"), [((-0.43, -0.67, -0.18), (-0.02, -0.2, 0.16), True), ((-0.1, -0.5, 0.3), (-0.02, -0.2, 0.16), False),
                                                                    ((0.4, 0.1, 0.7), (-0.02, -0.2, 0.16), False), ((-0.43, -0.67, -0.18), (-0.3, -0.5, -0.1), False)])
def test_the_365_day_sentence_prints_only_when_the_comparison_supports_it(full, interval, model_vs_365, wanted):
    write_forecast(full, interval, model_vs_365)
    text = render(full)
    assert ("cannot be told apart from the model, so this test does not show" in text) is wanted
    assert ("A plain 365-day mean beats the last 28 days too" in _strip_card(text, "1. Forecast")) is wanted
    assert "does as well" not in text


def test_the_pinned_settings_caveat_and_run_provenance_are_stated(full):
    text = render(full)
    assert "does not show that the settings add forecasting information" in text and "its level depends on its settings terms" in text
    assert "commit abcdef01, clean tree" in text
    write_audit(full)
    write_workbook(full / fp.AUDIT_FILE, {**pd.read_excel(full / fp.AUDIT_FILE, sheet_name=None), "Artifacts_Provenance": provenance("r", clean="no")})
    assert "uncommitted changes" in render(full)


def test_no_denominator_or_resample_count_is_typed_into_the_page(full):
    write_workbook(full / fp.FORECAST_FILE, {**pd.read_excel(full / fp.FORECAST_FILE, sheet_name=None), "Artifacts_Provenance": provenance("r").replace({"10000": "7777", 10000: 7777})})
    text = render(full)
    assert "7,777 resamples" in text and "10,000" not in text.split("Intervals: circular")[1].split("</p>")[0] and "51 windows with at least 70%" in text


def test_a_slope_that_could_not_be_estimated_is_not_a_reading(full):
    write_replay(full, primary=None, power=None, drop_power=True)
    text = render(full)
    assert "not estimated: too few changes to estimate a slope" in text and "power not assessed" in text
    assert "nan" not in text.lower().replace("nan", "", 0) or "nan%" not in text
    assert "Ranked setting changes are not shown: the primary slope could not be estimated" in text
    assert "Power was not assessed" in text


def test_slope_intervals_that_exclude_one_are_counted(full):
    text = render(full)
    assert "of 3 slope intervals lie wholly below 1" in text


def test_status_text_meets_contrast_on_its_chip():
    def luminance(color: str) -> float:
        channels = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    def tint(color: str) -> str:
        return "#" + "".join(f"{round(int(color[i:i + 2], 16) * 0.09 + 255 * 0.91):02x}" for i in (1, 3, 5))

    for style in (*STATUS_STYLE.values(), *VERDICT_STYLE.values()):
        background = "#ffffff" if style.outline else tint(style.color)
        hi, lo = sorted((luminance(style.text), luminance(background)), reverse=True)
        assert (hi + 0.05) / (lo + 0.05) >= 4.5, style.label
    assert INK_BLUE != STATUS_STYLE["no concern"].color and INK_GOLD != STATUS_STYLE["not assessed"].color


def test_a_p_value_is_written_the_same_way_everywhere():
    from bayesian_t1dm.formatting import format_p

    assert format_p(0.0004) == "p < 0.001" and format_p(0.0123) == "p = 0.012"
