"""What the front page says, read from the runs' own outputs: nothing on the page is typed by hand.

``gather`` reads the diagnostics workbooks and the proposal that a run folder holds (copied there from ``runtime/output`` by
``layout.assemble_run``) and returns plain values. A source that is not in the run becomes a ``Missing`` state that the page prints as
such; a source that is there but lacks a sheet, a column or a check the page needs raises with a message naming it, so a changed
workbook fails the refresh instead of leaving a gap. A result the workbook could not estimate (a group with too few changes) is carried
as ``None`` with the workbook's own reason, never as a number. Nothing here imports a model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import win_test as wt
from .diagnostics_workbook import STATUSES, read_workbook

REFERENCE = "last 28 days"
MODEL = "hour-day model"
PRIMARY_BLOCK = 4
# The comparisons the harness reports after seeing its results (candidate, reference); none gates the win test.
EXPLORATORY = (("last 365 days", REFERENCE), (MODEL, "last 365 days"), ("all history", REFERENCE), (MODEL, "all history"), (MODEL, "hour-day model (settings pinned)"))
PRIMARY_OUTCOME = "tir"
PRIMARY_SET = "isolated changes"
POWER_FLOOR = 0.80  # a test that finds a true effect less often than this cannot say an interval holding zero means no effect
# Where a run folder keeps each source (copied there by ``layout.assemble_run``).
FORECAST_FILE = "I_forecast/supporting/diagnostics.xlsx"
REPLAY_FILE = "J_replay_predictor/supporting/diagnostics.xlsx"
AUDIT_FILE = "00_General/diagnostics.xlsx"
PROPOSAL_FILE = "B_icr/icr_proposal.md"
# The audit checks the front page names (group, sheet, check): each must be in the workbook.
AUDIT_HEADLINE = (
    ("Settings reconstruction", "Settings_Reconstruction", "Hours with readings whose settings are not known for all 12 moments"),
    ("Clock", "Devices_Clock", "Time spent on a clock away from home"),
    ("Clock", "Devices_Clock", "Other clock jumps"),
    ("Clock", "Devices_Clock", "Shared clock of glucose, bolus and settings"),
    ("Recent regime", "Missingness_Stationarity", "Latest fortnight against earlier fortnights"),
    ("Recent regime", "Missingness_Stationarity", "Trend in monthly time in range"),
)
AUDIT_CHECK_SHEETS = ("Inventory_Coverage", "CGM_Quality", "Devices_Clock", "Settings_Reconstruction", "Meals_Insulin", "Missingness_Stationarity", "Leakage")
RESAMPLED_EXPOSURES = "Resampled exposures"


@dataclass(frozen=True)
class Missing:
    """A source that is not in the run: what is missing, the file looked for (relative to the run) and the command that writes it."""

    what: str
    path: str
    command: str


@dataclass(frozen=True)
class Provenance:
    """Which run an evidence source came from."""

    run_id: str
    commit: str
    clean: bool


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
    provenance: Provenance
    record_span: str
    origins: int
    windows_scored: int
    resamples: int
    win: Comparison
    half_width: float
    chart: tuple[Comparison, ...]
    exploratory: tuple[Comparison, ...]
    windows_with_change: int
    exposure_note: str

    def exploratory_row(self, candidate: str, reference: str) -> Comparison:
        return next(row for row in self.exploratory if (row.candidate, row.reference) == (candidate, reference))


@dataclass(frozen=True)
class Slope:
    """The slope of observed on predicted change with its 90% interval; ``slope`` is None when the workbook could not estimate it (``reading`` says why)."""

    outcome: str
    group: str
    changes: int
    clusters: int
    slope: float | None
    lower: float | None
    upper: float | None
    power_at_one: float | None
    reason: str

    @property
    def estimated(self) -> bool:
        return self.slope is not None

    @property
    def reading(self) -> str:
        if not self.estimated:
            return "not estimated"
        return "above zero" if self.lower > 0 else "below zero" if self.upper < 0 else "holds zero"

    @property
    def excludes_one(self) -> bool:
        """The interval lies wholly below 1: a correctly sized prediction (slope 1) is ruled out at the nominal level."""
        return self.estimated and self.upper < 1


@dataclass(frozen=True)
class Information:
    """CRPS with the settings in force against the old settings, new minus old; None when it could not be estimated."""

    outcome: str
    group: str
    changes: int
    crps_new: float | None
    crps_old: float | None
    difference: float | None
    lower: float | None
    upper: float | None
    reason: str

    @property
    def estimated(self) -> bool:
        return self.difference is not None

    @property
    def verdict(self) -> str:
        return wt.verdict_from_interval(self.lower, self.upper) if self.estimated else "not estimated"


@dataclass(frozen=True)
class Replay:
    provenance: Provenance
    resamples: int
    changes_in_record: int
    predicted: int
    isolated: int
    slopes: tuple[Slope, ...]
    information: tuple[Information, ...]

    @property
    def primary(self) -> Slope:
        for row in self.slopes:
            if row.outcome == PRIMARY_OUTCOME and row.group == PRIMARY_SET:
                return row
        raise ValueError(f"The replay-as-predictor workbook has no primary row ({PRIMARY_OUTCOME}, {PRIMARY_SET})")

    def information_for(self, outcome: str, group: str) -> Information | None:
        return next((row for row in self.information if row.outcome == outcome and row.group == group), None)

    @property
    def validated(self) -> bool:
        """The record shows the predicted effects track what happened only when the primary slope interval lies above zero."""
        return self.primary.estimated and self.primary.lower > 0

    @property
    def power_known(self) -> bool:
        return all(row.power_at_one is not None for row in self.slopes if row.outcome == PRIMARY_OUTCOME and row.estimated)

    @property
    def underpowered(self) -> bool:
        """Known power below the floor for every primary-outcome slope: an interval holding zero is then uninformative."""
        rows = [row for row in self.slopes if row.outcome == PRIMARY_OUTCOME and row.estimated]
        return bool(rows) and self.power_known and all(row.power_at_one < POWER_FLOOR for row in rows)

    @property
    def intervals_excluding_one(self) -> tuple[int, int]:
        """How many estimated slope intervals lie wholly below 1, of how many were estimated."""
        estimated = [row for row in self.slopes if row.estimated]
        return sum(row.excludes_one for row in estimated), len(estimated)


@dataclass(frozen=True)
class Check:
    group: str
    check: str
    result: str
    status: str
    reason: str


@dataclass(frozen=True)
class Audit:
    provenance: Provenance
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


def _lookup(table: pd.DataFrame, key_column: str, value_column: str, key: str, path: Path, sheet: str) -> object:
    rows = table[table[key_column].eq(key)]
    if rows.empty:
        raise ValueError(f"{path}: sheet {sheet} has no row {key!r}")
    return rows.iloc[0][value_column]


def _provenance(sheets: dict[str, pd.DataFrame], path: Path) -> tuple[Provenance, dict[str, str]]:
    table = _sheet(sheets, "Artifacts_Provenance", path)
    values = {str(row.item): str(row.value) for row in table.itertuples(index=False)}
    for key in ("run id", "commit", "working tree clean"):
        if key not in values:
            raise ValueError(f"{path}: Artifacts_Provenance has no {key!r}")
    return Provenance(values["run id"], values["commit"][:8], values["working tree clean"] == "yes"), values


def _number(value: object) -> float | None:
    """A workbook number, or None when the cell is empty (NaN)."""
    number = float(value)
    return None if math.isnan(number) else number


def read_forecast(root: Path) -> Forecast | Missing:
    """The win test and its comparators from the forecast harness workbook."""
    path = root / FORECAST_FILE
    if not path.exists():
        return Missing("The forecast harness result", FORECAST_FILE, "python -m bayesian_t1dm.forecast_harness --outcome tir")
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
    flow = _sheet(sheets, "Denominators_Flow", path)
    flow = flow[flow["set"].eq("evaluation")].set_index("stage")["n"]
    scored = [stage for stage in flow.index if stage.startswith("windows with at least")]
    if len(scored) != 1 or "weekly origins" not in flow.index:
        raise ValueError(f"{path}: Denominators_Flow lacks the evaluation rows 'weekly origins' and 'windows with at least ...'")
    provenance, values = _provenance(sheets, path)
    for key in ("record span", "bootstrap resamples"):
        if key not in values:
            raise ValueError(f"{path}: Artifacts_Provenance has no {key!r}")
    warnings = _sheet(sheets, "Warnings_Limitations", path)
    return Forecast(provenance, values["record span"], int(flow["weekly origins"]), int(flow[scored[0]]), int(values["bootstrap resamples"]), win, half_width, chart,
                    tuple(find(candidate, reference) for candidate, reference in EXPLORATORY), int(changed.iloc[0]["origins"]),
                    str(_lookup(warnings, "limitation", "detail", RESAMPLED_EXPOSURES, path, "Warnings_Limitations")))


def read_replay(root: Path) -> Replay | Missing:
    """The replay-as-predictor slopes, power and settings-information test from its workbook."""
    path = root / REPLAY_FILE
    if not path.exists():
        return Missing("The replay-as-predictor result", REPLAY_FILE, "python -m bayesian_t1dm.replay_predictor --outcomes tir,below70,below54")
    sheets = read_workbook(path)
    tracking = _sheet(sheets, "Tracking", path)
    tracking = tracking[tracking["variant"].eq("primary")]
    power = _sheet(sheets, "Power", path).set_index(["outcome", "group"])

    def power_at_one(outcome: str, group: str) -> float | None:
        return _number(power.loc[(outcome, group), "power_true_slope_1"]) if (outcome, group) in power.index else None

    slopes = tuple(
        Slope(str(row.outcome), str(row.group), int(row.changes), int(row.clusters), _number(row.slope), _number(row.slope_lower_5th), _number(row.slope_upper_95th),
              power_at_one(row.outcome, row.group), str(row.reading))
        for row in tracking.itertuples(index=False)
    )
    information = tuple(
        Information(str(row.outcome), str(row.group), int(row.changes), _number(row.mean_crps_new_points), _number(row.mean_crps_old_points), _number(row.mean_difference_points),
                    _number(row.lower_5th), _number(row.upper_95th), str(row.reading))
        for row in _sheet(sheets, "Settings_Information", path).itertuples(index=False)
    )
    flow = _sheet(sheets, "Denominators_Flow", path)
    counts = {stage: int(_lookup(flow, "stage", "n", stage, path, "Denominators_Flow")) for stage in ("changes in the record", "predicted (after the fits)", "of which isolated")}
    provenance, values = _provenance(sheets, path)
    if "bootstrap resamples" not in values:
        raise ValueError(f"{path}: Artifacts_Provenance has no 'bootstrap resamples'")
    replay = Replay(provenance, int(values["bootstrap resamples"]), counts["changes in the record"], counts["predicted (after the fits)"], counts["of which isolated"], slopes, information)
    replay.primary  # raises, naming the row, when the primary row is absent
    return replay


def read_audit(root: Path) -> Audit | Missing:
    """Status counts over every audit check, the named headline checks and the checks that were not assessed."""
    path = root / AUDIT_FILE
    if not path.exists():
        return Missing("The data-quality audit", AUDIT_FILE, "python -m bayesian_t1dm.data_audit")
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
    provenance, _ = _provenance(sheets, path)
    return Audit(provenance, counts, tuple(checks), tuple(not_assessed))


def read_proposal(root: Path) -> Proposal | Missing:
    """The breakfast proposal's own heading and first paragraph, word for word."""
    path = root / PROPOSAL_FILE
    if not path.exists():
        return Missing("The breakfast I:C proposal", PROPOSAL_FILE, "propose-breakfast-icr")
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


def gather(run_dir: Path) -> FrontPage:
    """Everything the front page shows, read from the copies a run folder holds."""
    return FrontPage(read_forecast(run_dir), read_replay(run_dir), read_audit(run_dir), read_proposal(run_dir))


def plain_markdown(text: str) -> str:
    """Bold markers only: the proposal's own wording, with ``**x**`` as ``<strong>x</strong>`` and everything else escaped."""
    import html

    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text))
