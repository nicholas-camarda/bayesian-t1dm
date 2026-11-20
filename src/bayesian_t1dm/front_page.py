"""What the front page says, read from the runs' own outputs: nothing on the page is typed by hand.

``gather`` reads the diagnostics workbooks and summaries the research commands wrote under ``runtime/output`` (the forecast
harness, the replay-as-predictor test and the data audit) and the breakfast proposal, and returns plain values. A source that
was never written becomes a ``Missing`` state that the page prints as such; a source that exists but lacks a sheet, a column or a
check the page needs raises, so a changed workbook fails the refresh instead of leaving a gap. Nothing here imports a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import win_test as wt
from .diagnostics_workbook import STATUSES, read_workbook
from .paths import ProjectPaths

REFERENCE = "last 28 days"
MODEL = "hour-day model"
PRIMARY_BLOCK = 4
# The comparisons the harness reports after seeing its results (candidate, reference); none gates the win test.
EXPLORATORY = (("last 365 days", REFERENCE), (MODEL, "last 365 days"), ("all history", REFERENCE), (MODEL, "all history"), (MODEL, "hour-day model (settings pinned)"))
PRIMARY_OUTCOME = "tir"
PRIMARY_SET = "isolated changes"
POWER_FLOOR = 0.80  # a test that finds a true effect less often than this cannot say an interval holding zero means no effect
# The audit checks the front page names (sheet, check): each must be in the workbook.
AUDIT_HEADLINE = (
    ("Settings reconstruction", "Settings_Reconstruction", "Hours with readings whose settings are not known for all 12 moments"),
    ("Clock", "Devices_Clock", "Time spent on a clock away from home"),
    ("Clock", "Devices_Clock", "Other clock jumps"),
    ("Clock", "Devices_Clock", "Shared clock of glucose, bolus and settings"),
    ("Recent regime", "Missingness_Stationarity", "Latest fortnight against earlier fortnights"),
    ("Recent regime", "Missingness_Stationarity", "Trend in monthly time in range"),
)
AUDIT_CHECK_SHEETS = ("Inventory_Coverage", "CGM_Quality", "Devices_Clock", "Settings_Reconstruction", "Meals_Insulin", "Missingness_Stationarity", "Leakage")


@dataclass(frozen=True)
class Missing:
    """A source that was not written: what is missing, where it was looked for and how to make it."""

    what: str
    path: Path
    command: str


@dataclass(frozen=True)
class Comparison:
    """One paired CRPS difference, candidate minus reference, in percentage points of the 14-day share (negative favours the candidate)."""

    candidate: str
    reference: str
    role: str
    mean: float
    lower: float
    upper: float
    origins: int

    @property
    def verdict(self) -> str:
        return wt.verdict_from_interval(self.lower, self.upper)


@dataclass(frozen=True)
class Forecast:
    run_id: str
    record_span: str
    origins: int
    win: Comparison
    half_width: float
    chart: tuple[Comparison, ...]
    exploratory: tuple[Comparison, ...]
    windows_with_change: int


@dataclass(frozen=True)
class Slope:
    outcome: str
    group: str
    changes: int
    clusters: int
    slope: float
    lower: float
    upper: float
    power_at_one: float | None

    @property
    def reading(self) -> str:
        return "above zero" if self.lower > 0 else "below zero" if self.upper < 0 else "holds zero"


@dataclass(frozen=True)
class Information:
    outcome: str
    group: str
    changes: int
    crps_new: float
    crps_old: float
    difference: float
    lower: float
    upper: float

    @property
    def verdict(self) -> str:
        return wt.verdict_from_interval(self.lower, self.upper)


@dataclass(frozen=True)
class Replay:
    run_id: str
    changes_in_record: int
    predicted: int
    isolated: int
    slopes: tuple[Slope, ...]
    information: tuple[Information, ...]

    @property
    def primary(self) -> Slope:
        return next(row for row in self.slopes if row.outcome == PRIMARY_OUTCOME and row.group == PRIMARY_SET)

    @property
    def validated(self) -> bool:
        """The record shows the predicted effects track what happened only when the primary slope interval lies above zero."""
        return self.primary.lower > 0

    @property
    def underpowered(self) -> bool:
        return all(row.power_at_one is not None and row.power_at_one < POWER_FLOOR for row in self.slopes if row.outcome == PRIMARY_OUTCOME)


@dataclass(frozen=True)
class Check:
    group: str
    check: str
    result: str
    status: str
    reason: str


@dataclass(frozen=True)
class Audit:
    run_id: str
    counts: dict[str, int]
    checks: tuple[Check, ...]
    not_assessed: tuple[Check, ...]


@dataclass(frozen=True)
class Proposal:
    title: str
    paragraph: str


@dataclass(frozen=True)
class FrontPage:
    forecast: Forecast | Missing
    replay: Replay | Missing
    audit: Audit | Missing
    proposal: Proposal | Missing


def _sheet(sheets: dict[str, pd.DataFrame], name: str, path: Path) -> pd.DataFrame:
    if name not in sheets:
        raise ValueError(f"{path} has no sheet {name!r}; rerun the command that wrote it")
    return sheets[name]


def _provenance(sheets: dict[str, pd.DataFrame], path: Path) -> dict[str, str]:
    table = _sheet(sheets, "Artifacts_Provenance", path)
    return {str(row.item): str(row.value) for row in table.itertuples(index=False)}


def read_forecast(folder: Path) -> Forecast | Missing:
    """The win test and its comparators from the forecast harness workbook."""
    path = folder / "diagnostics.xlsx"
    if not path.exists():
        return Missing("The forecast harness result", path, "python -m bayesian_t1dm.forecast_harness --outcome tir")
    sheets = read_workbook(path)
    table = _sheet(sheets, "Win_Test", path)
    block = table[table["block_length_origins"].eq(PRIMARY_BLOCK)]

    def comparison(row: pd.Series) -> Comparison:
        found = Comparison(str(row["candidate"]), str(row["reference"]), str(row["role"]), float(row["mean_difference_points"]), float(row["lower_5th_points"]),
                           float(row["upper_95th_points"]), int(row["origins"]))
        if found.verdict != str(row["verdict"]):
            raise ValueError(f"{path}: the verdict for {found.candidate} against {found.reference} disagrees with its interval")
        return found

    def find(candidate: str, reference: str) -> Comparison:
        rows = block[block["candidate"].eq(candidate) & block["reference"].eq(reference)]
        if len(rows) != 1:
            raise ValueError(f"{path}: expected one Win_Test row for {candidate} against {reference} at block {PRIMARY_BLOCK}, found {len(rows)}")
        return comparison(rows.iloc[0])

    gating = block[block["role"].eq("win test")]
    if len(gating) != 1:
        raise ValueError(f"{path}: expected one win-test row, found {len(gating)}")
    win = comparison(gating.iloc[0])
    power = block[block["role"].eq("power check")]
    if len(power) != 1:
        raise ValueError(f"{path}: expected one power-check row, found {len(power)}")
    half_width = (float(power.iloc[0]["upper_95th_points"]) - float(power.iloc[0]["lower_5th_points"])) / 2.0
    chart = tuple(comparison(row) for _, row in block[block["reference"].eq(REFERENCE)].iterrows())
    changes = _sheet(sheets, "Settings_Changes", path)
    changed = changes[changes["windows"].eq("settings changed inside the window")]
    if len(changed) != 1:
        raise ValueError(f"{path}: Settings_Changes has no row for windows with a settings change")
    provenance = _provenance(sheets, path)
    return Forecast(provenance["run id"], provenance["record span"], win.origins, win, half_width, chart,
                    tuple(find(candidate, reference) for candidate, reference in EXPLORATORY), int(changed.iloc[0]["origins"]))


def read_replay(folder: Path) -> Replay | Missing:
    """The replay-as-predictor slopes, power and settings-information test from its workbook."""
    path = folder / "diagnostics.xlsx"
    if not path.exists():
        return Missing("The replay-as-predictor result", path, "python -m bayesian_t1dm.replay_predictor --outcomes tir,below70,below54")
    sheets = read_workbook(path)
    tracking = _sheet(sheets, "Tracking", path)
    tracking = tracking[tracking["variant"].eq("primary")]
    power = _sheet(sheets, "Power", path).set_index(["outcome", "group"])
    slopes = tuple(
        Slope(str(row.outcome), str(row.group), int(row.changes), int(row.clusters), float(row.slope), float(row.slope_lower_5th), float(row.slope_upper_95th),
              float(power.loc[(row.outcome, row.group), "power_true_slope_1"]) if (row.outcome, row.group) in power.index else None)
        for row in tracking.itertuples(index=False)
    )
    information = tuple(
        Information(str(row.outcome), str(row.group), int(row.changes), float(row.mean_crps_new_points), float(row.mean_crps_old_points), float(row.mean_difference_points),
                    float(row.lower_5th), float(row.upper_95th))
        for row in _sheet(sheets, "Settings_Information", path).itertuples(index=False)
    )
    flow = _sheet(sheets, "Denominators_Flow", path).set_index("stage")["n"]
    replay = Replay(_provenance(sheets, path)["run id"], int(flow["changes in the record"]), int(flow["predicted (after the fits)"]), int(flow["of which isolated"]), slopes, information)
    replay.primary  # raises when the primary row is absent
    return replay


def read_audit(folder: Path) -> Audit | Missing:
    """Status counts over every audit check, the named headline checks and the checks that were not assessed."""
    path = folder / "diagnostics.xlsx"
    if not path.exists():
        return Missing("The data-quality audit", path, "python -m bayesian_t1dm.data_audit")
    sheets = read_workbook(path)
    counts = dict.fromkeys(STATUSES, 0)
    not_assessed: list[Check] = []
    for name in AUDIT_CHECK_SHEETS:
        table = _sheet(sheets, name, path)
        label = "column" if name == "Leakage" else "check"
        for row in table.to_dict("records"):
            counts[str(row["status"])] += 1
            if row["status"] == "not assessed":
                not_assessed.append(Check(name, str(row[label]), str(row.get("result", row.get("known_when", ""))), "not assessed", str(row["reason"])))
    checks = []
    for group, name, wanted in AUDIT_HEADLINE:
        rows = _sheet(sheets, name, path)
        rows = rows[rows["check"].eq(wanted)]
        if len(rows) != 1:
            raise ValueError(f"{path}: expected one check {wanted!r} in {name}, found {len(rows)}")
        row = rows.iloc[0]
        checks.append(Check(group, wanted, str(row["result"]), str(row["status"]), str(row["reason"])))
    leakage = _sheet(sheets, "Leakage", path)
    for row in leakage[leakage["used_by_outcomes_model"].astype(str).str.startswith(("yes", "no (row filter)"))].to_dict("records"):
        checks.append(Check("Leakage", str(row["column"]), str(row["known_when"]), str(row["status"]), str(row["reason"])))
    return Audit(_provenance(sheets, path)["run id"], counts, tuple(checks), tuple(not_assessed))


def read_proposal(folder: Path) -> Proposal | Missing:
    """The breakfast proposal's own heading and first paragraph, word for word."""
    path = folder / "breakfast_proposal.md"
    if not path.exists():
        return Missing("The breakfast I:C proposal", path, "propose-breakfast-icr")
    lines = path.read_text(encoding="utf-8").splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("# Breakfast I:C proposal")]
    if not starts:
        raise ValueError(f"{path} has no '# Breakfast I:C proposal' heading")
    body: list[str] = []
    for line in lines[starts[0] + 1:]:
        if line.startswith("## "):
            break
        body.append(line)
    paragraph = " ".join(part.strip() for part in "\n".join(body).strip().split("\n\n")[:1])
    return Proposal(lines[starts[0]].lstrip("# ").strip(), paragraph)


def gather(paths: ProjectPaths) -> FrontPage:
    """Everything the front page shows, read from the runs' outputs."""
    return FrontPage(read_forecast(paths.output_forecast_harness), read_replay(paths.output_replay_predictor), read_audit(paths.output_data_audit),
                     read_proposal(paths.output_proposals))


def plain_markdown(text: str) -> str:
    """Bold markers only: the proposal's own wording, with ``**x**`` as ``<strong>x</strong>`` and everything else escaped."""
    import html

    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text))
