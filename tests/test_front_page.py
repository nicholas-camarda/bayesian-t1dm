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
from bayesian_t1dm.paths import ProjectPaths

REQUIRED = {"Specification_Coding": pd.DataFrame({"item": ["Question"], "value": ["q"]}), "Denominators_Flow": pd.DataFrame({"stage": ["x"], "n": [1]}),
            "Warnings_Limitations": pd.DataFrame({"limitation": ["l"], "detail": ["d"]})}


def provenance(run_id: str, span: str = "2024-01-01 to 2026-11-17") -> pd.DataFrame:
    return pd.DataFrame({"item": ["run id", "record span"], "value": [run_id, span]})


def win_row(candidate: str, reference: str, role: str, mean: float, lower: float, upper: float, block: int = 4) -> dict:
    verdict = "win" if upper < 0 else "loss" if lower > 0 else "tie"
    return {"candidate": candidate, "reference": reference, "role": role, "block_length_origins": block, "origins": 52, "mean_difference_points": mean, "lower_5th_points": lower,
            "upper_95th_points": upper, "verdict": verdict, "status": "not applicable", "reason": ""}


def write_forecast(folder: Path, interval: tuple[float, float, float] = (-0.43, -0.67, -0.18), model_vs_365: tuple[float, float, float] = (-0.02, -0.20, 0.16)) -> None:
    mean, lower, upper = interval
    rows = [win_row("hour-day model", "last 28 days", "win test", mean, lower, upper), win_row("last 14 days", "last 28 days", "power check", 0.46, 0.14, 0.77)]
    rows += [win_row(name, "last 28 days", "reported, does not gate", -0.2, -0.5, 0.1) for name in PREDICTOR_STYLE if name not in ("last 28 days", "last 14 days", "hour-day model", "last 365 days")]
    rows.append(win_row("last 365 days", "last 28 days", "reported, does not gate", -0.41, -0.66, -0.15))
    rows += [win_row("hour-day model", "last 365 days", "pairwise, does not gate", *model_vs_365), win_row("hour-day model", "all history", "pairwise, does not gate", -0.30, -0.57, -0.04),
             win_row("hour-day model", "hour-day model (settings pinned)", "pairwise, does not gate", -0.63, -1.07, -0.21),
             win_row("hour-day model", "last 28 days", "sensitivity of the win test (block length)", mean, lower, upper, block=2)]
    sheets = {**REQUIRED, "Win_Test": pd.DataFrame(rows),
              "Settings_Changes": pd.DataFrame({"windows": ["settings changed inside the window", "settings unchanged"], "origins": [13, 39]}), "Artifacts_Provenance": provenance("20260101T000000Z")}
    write_workbook(folder / "diagnostics.xlsx", sheets)


def write_replay(folder: Path, primary: tuple[float, float, float] = (-1.1, -7.6, 0.4), power: float = 0.3) -> None:
    def row(outcome: str, group: str, changes: int, clusters: int, slope: float, lower: float, upper: float) -> dict:
        return {"variant": "primary", "outcome": outcome, "group": group, "changes": changes, "clusters": clusters, "slope": slope, "slope_lower_5th": lower, "slope_upper_95th": upper}
    tracking = pd.DataFrame([row("tir", "isolated changes", 10, 10, *primary), row("tir", "all changes", 18, 15, -0.9, -2.8, 0.1), row("below54", "all changes", 18, 15, -3.8, -7.5, -0.6)])
    information = pd.DataFrame([{"outcome": "tir", "group": "isolated changes", "changes": 10, "clusters": 10, "mean_crps_new_points": 4.46, "mean_crps_old_points": 3.57, "mean_difference_points": 0.89,
                                 "lower_5th": 0.31, "upper_95th": 1.52}])
    powers = pd.DataFrame([{"outcome": "tir", "group": group, "changes": 1, "clusters": 1, "power_true_slope_1": power} for group in ("isolated changes", "all changes")])
    flow = pd.DataFrame({"stage": ["changes in the record", "predicted (after the fits)", "of which isolated"], "n": [45, 18, 10], "unit": ["changes"] * 3})
    write_workbook(folder / "diagnostics.xlsx", {**REQUIRED, "Denominators_Flow": flow, "Tracking": tracking, "Settings_Information": information, "Power": powers, "Artifacts_Provenance": provenance("20260102T000000Z")})


def write_audit(folder: Path, *, not_assessed: bool = True) -> None:
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
    write_workbook(folder / "diagnostics.xlsx", {**REQUIRED, **sheets})


def write_proposal(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "breakfast_proposal.md").write_text("| table |\n\n# Breakfast I:C proposal (2026-11-17)\n\n**No change proposed.** Waiting: earlier changes are still being evaluated.\n\n## Checks\n\n- x\n", encoding="utf-8")


@pytest.fixture
def paths(tmp_path) -> ProjectPaths:
    return ProjectPaths.from_root(tmp_path / "source", runtime_root=tmp_path / "runtime", cloud_root=tmp_path / "vault").ensure()


@pytest.fixture
def full(paths) -> ProjectPaths:
    write_forecast(paths.output_forecast_harness)
    write_replay(paths.output_replay_predictor)
    write_audit(paths.output_data_audit)
    write_proposal(paths.output_proposals)
    return paths


def render(paths: ProjectPaths, tmp_path: Path) -> str:
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    return page.render_front_page(fp.gather(paths), run)


def visible(text: str) -> str:
    """The page without tags and without the text alternatives, which repeat a chart's numbers for a screen reader."""
    return re.sub(r"<[^>]+>", " ", re.sub(r"aria-label='[^']*'", "", text))


@pytest.mark.parametrize(("interval", "state", "words"), [((-0.43, -0.67, -0.18), "win", "forecast 14-day time in range better"), ((-0.1, -0.5, 0.3), "tie", "cannot be told apart"),
                                                          ((0.4, 0.1, 0.7), "loss", "forecast 14-day time in range worse")])
def test_the_verdict_follows_the_interval(full, tmp_path, interval, state, words):
    write_forecast(full.output_forecast_harness, interval)
    text = render(full, tmp_path)
    strip = text[text.index("<nav class='strip'"):text.index("</nav>")]
    assert f"{VERDICT_STYLE[state].label}</span>" in strip.split("2. Settings effects")[0]
    assert words in text[text.index("class='headline'"):].split("</p>")[0]
    assert f"{interval[0]:+.2f} points" in text and f"({interval[1]:+.2f} to {interval[2]:+.2f}" in text


def test_a_missing_output_is_a_stated_gap_not_a_fill(paths, tmp_path):
    write_replay(paths.output_replay_predictor)
    write_audit(paths.output_data_audit)
    text = render(paths, tmp_path)
    assert "The forecast harness result is not available" in text and "forecast_harness" in text and "python -m bayesian_t1dm.forecast_harness" in text
    assert "Prespecified win test" not in text and "chip outline" in text.split("2. Can the record")[0] and VERDICT_STYLE["missing"].label in text
    assert "The breakfast I:C proposal is not available" in text


def test_with_nothing_written_every_section_says_so(paths, tmp_path):
    text = render(paths, tmp_path)
    for what in ("forecast harness result", "replay-as-predictor result", "data-quality audit", "breakfast I:C proposal"):
        assert f"The {what} is not available" in text
    assert "Ranked setting changes are not shown" in text


def test_a_check_that_was_not_assessed_never_looks_like_a_pass(full, tmp_path):
    text = render(full, tmp_path)
    assert STATUS_STYLE["not assessed"].outline and not STATUS_STYLE["no concern"].outline
    not_assessed = re.search(r"<li class='check'>(?:(?!</li>).)*Shared clock(?:(?!</li>).)*</li>", text, re.S).group(0)
    assert "Not assessed" in not_assessed and "No concern" not in not_assessed and "chip outline" in not_assessed
    assert "never a pass" in text and "could not be run" in text
    clean = render_with(full, tmp_path, not_assessed=False)
    assert "could not be run" not in clean


def render_with(paths, tmp_path, *, not_assessed: bool) -> str:
    write_audit(paths.output_data_audit, not_assessed=not_assessed)
    return render(paths, tmp_path)


def test_the_overall_data_status_is_a_concern_while_any_check_is_a_concern(full, tmp_path):
    text = render(full, tmp_path)
    strip = text[text.index("<nav class='strip'"):text.index("</nav>")]
    assert "Concern" in strip.split("3. Data quality")[1] and "Concerns in" in strip


def test_settings_effects_are_not_shown_when_the_slope_interval_holds_zero(full, tmp_path):
    text = render(full, tmp_path)
    assert "Ranked setting changes are not shown" in text and VERDICT_STYLE["not shown"].label in text and "low power" in text
    write_replay(full.output_replay_predictor, primary=(2.0, 0.4, 3.5), power=0.9)
    shown = render(full, tmp_path)
    assert VERDICT_STYLE["shown"].label in shown and "Ranked setting changes are not shown" not in shown and "low power" not in shown


def test_a_group_has_one_color_marker_and_line_type_everywhere(full, tmp_path):
    text = render(full, tmp_path)
    for name in ("hour-day model", "last 365 days"):
        assert PREDICTOR_STYLE[name].color in text
    isolated, every = CHANGE_SET_STYLE["isolated changes"], CHANGE_SET_STYLE["all changes"]
    assert isolated.color in text and every.color in text and isolated.linestyle == "-" and every.linestyle == "--"
    assert "border-top-style:dashed" in text and "border-top-style:solid" in text


def test_the_page_is_ascii_has_no_script_and_each_headline_number_is_printed_once(full, tmp_path):
    text = render(full, tmp_path)
    assert text.isascii() and "<script" not in text and "<img" not in text
    assert visible(text).count("-0.43") == 1 and visible(text).count("-0.67") == 1
    assert "lang='en'" in text and "name='viewport'" in text and "role='img'" in text


def test_the_proposal_keeps_its_own_wording(full, tmp_path):
    text = render(full, tmp_path)
    assert "<strong>No change proposed.</strong> Waiting: earlier changes are still being evaluated." in text and "Breakfast I:C proposal (2026-11-17)" in text


def test_a_workbook_missing_a_check_the_page_names_fails_loudly(full):
    sheets = pd.read_excel(full.output_data_audit / "diagnostics.xlsx", sheet_name=None)
    sheets["Devices_Clock"] = sheets["Devices_Clock"][~sheets["Devices_Clock"]["check"].eq("Other clock jumps")]
    write_workbook(full.output_data_audit / "diagnostics.xlsx", sheets)
    with pytest.raises(ValueError, match="Other clock jumps"):
        fp.gather(full)


@pytest.mark.parametrize("row", [{"candidate": "hour-day model", "role": "win test"}, {"candidate": "last 365 days", "role": "reported, does not gate"}])
def test_a_verdict_that_disagrees_with_its_interval_is_refused(full, row):
    sheets = pd.read_excel(full.output_forecast_harness / "diagnostics.xlsx", sheet_name=None)
    wanted = sheets["Win_Test"]["candidate"].eq(row["candidate"]) & sheets["Win_Test"]["role"].eq(row["role"]) & sheets["Win_Test"]["block_length_origins"].eq(4)
    sheets["Win_Test"].loc[wanted, "verdict"] = "loss"
    write_workbook(full.output_forecast_harness / "diagnostics.xlsx", sheets)
    with pytest.raises(ValueError, match="disagrees"):
        fp.gather(full)


def test_links_point_only_at_files_that_are_in_the_run(full, tmp_path):
    run = tmp_path / "run"
    (run / "I_forecast").mkdir(parents=True)
    (run / "I_forecast" / "forecast_harness_summary.md").write_text("s")
    (run / "dashboard.html").write_text("d")
    text = page.render_front_page(fp.gather(full), run)
    assert "href='I_forecast/forecast_harness_summary.md'" in text and "href='dashboard.html'" in text
    assert "href='J_replay_predictor/replay_predictor_summary.md'" not in text and "Figure (not written)" in text


def test_the_front_page_command_writes_index_html_into_a_run_folder(full, tmp_path, monkeypatch):
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(full.runtime))
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(full.cloud_root))
    run = tmp_path / "somerun"
    run.mkdir()
    assert main(["--root", str(full.root), "front-page", "--run-dir", str(run)]) == 0
    assert "Can this record say which pump settings to change?" in (run / "index.html").read_text(encoding="utf-8")
