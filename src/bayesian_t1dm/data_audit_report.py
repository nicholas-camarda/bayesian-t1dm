"""Reads the archive, runs the data-quality audit, and writes its diagnostics workbook, figures and short summary.

The checks themselves are in ``data_audit`` (pure); this module does the input and output: it reads the pump's event log for
the clock and sensor-session checks, assembles the sheets in the diagnostics-workbook order, reads the workbook back and
reconciles it with the hour-day table, and writes three figures and a markdown summary generated from the check rows.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data_audit as audit
from .diagnostics_workbook import read_workbook, write_workbook
from .figures import NEUTRAL_BAND, NEUTRAL_BAR, NEUTRAL_LINE, PUMP_STYLE, REASON_COLORS, assert_ascii
from .hour_day import SLOTS_PER_HOUR, build_hour_day_table, hour_day_flow
from .outcomes_model import COVARIATES, design_flow
from .publish import current_commit, working_tree_is_clean
from .pump_profiles import UNKNOWN_REASONS, pump_key

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
EVENT_RECORD_LEN = 26
TANDEM_EPOCH = datetime(2008, 1, 1)
TIME_CHANGED_EVENT = 13
DATE_CHANGED_EVENT = 14
SESSION_EVENTS = {213: "join", 394: "join", 214: "stop", 447: "stop"}  # Gx and G7 sensor-session join and stop
JUMP_MINUTES = 20
FIGURE_FILES = ("data_audit_cgm", "data_audit_settings", "data_audit_stationarity")
CHECK_SHEETS = ("Inventory_Coverage", "CGM_Quality", "Devices_Clock", "Settings_Reconstruction", "Meals_Insulin", "Missingness_Stationarity")


# ---------------------------------------------------------------------------
# Reading the pump's event log
# ---------------------------------------------------------------------------

def _binary_event_ids(path: Path, pump: str) -> pd.DataFrame:
    """Event id, time and sequence of every 26-byte record of one archived binary window (no payload decoding)."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return pd.DataFrame(columns=["pump_serial", "sequence", "timestamp", "event"])
    if text.startswith('"'):
        text = json.loads(text)
    data = base64.b64decode(text)
    records = np.frombuffer(data[: len(data) - len(data) % EVENT_RECORD_LEN], dtype=np.uint8).reshape(-1, EVENT_RECORD_LEN)
    event = ((records[:, 0].astype(np.int64) << 8) | records[:, 1]) & 0x0FFF
    seconds = records[:, 2:6].astype(np.int64) @ np.array([1 << 24, 1 << 16, 1 << 8, 1])
    sequence = records[:, 6:10].astype(np.int64) @ np.array([1 << 24, 1 << 16, 1 << 8, 1])
    return pd.DataFrame({"pump_serial": pump, "sequence": sequence, "timestamp": pd.Timestamp(TANDEM_EPOCH) + pd.to_timedelta(seconds, unit="s"), "event": event})


def _json_event_ids(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pump = pump_key(payload["pump_serial"])
    rows = [(pump, int(event["sequenceNumber"]), pd.Timestamp(event["pumpDateTime"]).tz_localize(None) if pd.Timestamp(event["pumpDateTime"]).tzinfo else pd.Timestamp(event["pumpDateTime"]), int(event["eventCode"]))
            for chunk in payload["chunks"] for event in chunk["response"].get("events") or []]
    return pd.DataFrame(rows, columns=["pump_serial", "sequence", "timestamp", "event"])


def read_event_log(tandem_root: str | Path) -> dict[str, pd.DataFrame]:
    """Clock jumps and sensor-session events from the archived pump events (every window, binary and JSON), deduplicated by pump and sequence.

    Returns:
        ``jumps``: one row per pair of consecutive events (by sequence) whose timestamps differ by more than 20 minutes or go backward;
        ``sessions``: join and stop events of the sensor (``kind``); ``overview``: events, first and last time per pump.
    """
    tandem_root = Path(tandem_root)
    pieces = []
    for window in sorted(path for path in tandem_root.iterdir() if path.is_dir() and "__" in path.name):
        for path in sorted((window / "raw").glob("pump_events_raw_*.txt")):
            pump = pump_key(path.stem.removeprefix("pump_events_raw_").rpartition("_")[0])
            pieces.append(_binary_event_ids(path, pump))
        for path in sorted((window / "raw").glob("pump_logs_*.json")):
            if not path.name.endswith(".meta.json"):
                pieces.append(_json_event_ids(path))
    events = pd.concat(pieces, ignore_index=True).drop_duplicates(["pump_serial", "sequence"]).sort_values(["pump_serial", "sequence"]).reset_index(drop=True)
    events["previous_timestamp"] = events.groupby("pump_serial")["timestamp"].shift()
    events["jump_minutes"] = (events["timestamp"] - events["previous_timestamp"]).dt.total_seconds() / 60.0
    jumps = events[(events["jump_minutes"].abs() > JUMP_MINUTES) | (events["jump_minutes"] < -1)][["pump_serial", "sequence", "timestamp", "previous_timestamp", "event", "jump_minutes"]].reset_index(drop=True)
    sessions = events[events["event"].isin(SESSION_EVENTS)].assign(kind=lambda f: f["event"].map(SESSION_EVENTS))[["pump_serial", "timestamp", "kind"]].reset_index(drop=True)
    overview = events.groupby("pump_serial").agg(events=("sequence", "size"), first=("timestamp", "min"), last=("timestamp", "max")).reset_index()
    return {"jumps": jumps, "sessions": sessions, "overview": overview}


# ---------------------------------------------------------------------------
# Running the checks
# ---------------------------------------------------------------------------

def run_checks(analysis, table: pd.DataFrame, event_log: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame | list[pd.DataFrame]]:
    """Every check on the loaded record: the sheets that are check tables, the detail tables and the record issues."""
    merged = analysis.merged
    raw = merged.tandem_data.cgm
    valid = analysis.cgm
    data = merged.tandem_data
    streams = {
        "cgm": valid["timestamp"], "basal": data.basal["start_timestamp"], "bolus": data.bolus["timestamp"], "carbs": data.carbs["timestamp"],
        "exercise": analysis.exercise["start"], "pump mode changes": analysis.mode_events["timestamp"],
        "cannula fills": analysis.site_events.loc[analysis.site_events["kind"].eq("cannula filled"), "timestamp"],
        "profile events": merged.profile_events["timestamp"], "phone sleep periods": merged.data.sleep["sleep_start"],
        "context-event days": pd.Series(analysis.excluded_days, dtype="datetime64[ns]"),
    }
    issues: list[pd.DataFrame] = []
    cgm_rows = [audit.cgm_active_time(table), audit.cgm_gaps(valid)]
    values, value_issues = audit.cgm_duplicates_and_values(raw, valid)
    cgm_rows += [values, audit.compression_lows(valid)]
    eras, era_check = audit.cgm_eras(raw, valid)
    cgm_rows += [era_check, audit.sensor_sessions(event_log["sessions"], valid)]
    issues.append(value_issues)

    jump_table, clock_checks, clock_issues = audit.clock_jumps(event_log["jumps"], table)
    issues.append(clock_issues)
    devices = pd.concat([audit.pump_eras(raw, table), audit.pump_change_break(table), clock_checks, audit.clock_alignment(table)], ignore_index=True)

    reasons = audit.hour_unknown_reasons(analysis.history, table)
    unknown_checks, unknown_detail, unknown_issues = audit.unknown_settings(table, reasons)
    events_table, crosscheck, crosscheck_issues = audit.replay_event_crosscheck(table, analysis.history, merged.profile_events)
    calculator_checks, calculator_issues = audit.calculator_agreement(data.carbs, analysis.history)
    basal_checks, basal_issues = audit.basal_log_agreement(merged.profile_rates, analysis.history)
    settings = pd.concat([unknown_checks, calculator_checks, basal_checks, crosscheck], ignore_index=True)
    issues += [unknown_issues, calculator_issues, basal_issues, crosscheck_issues]

    meal_checks, meal_issues = audit.meal_checks(data.carbs, data.bolus)
    issues.append(meal_issues)
    composition, composition_check = audit.composition_test(table)
    missing = pd.concat([audit.missingness(table), composition_check, audit.stationarity(table)], ignore_index=True)
    no_reading = _no_reading_issues(table)
    excluded = _excluded_day_issues(table)
    issues += [no_reading, excluded]
    return {
        "Inventory_Coverage": audit.inventory_coverage(streams),
        "CGM_Quality": pd.concat(cgm_rows, ignore_index=True),
        "Devices_Clock": devices,
        "Settings_Reconstruction": settings,
        "Meals_Insulin": pd.concat([meal_checks, audit.insulin_checks(data.bolus, data.basal)], ignore_index=True),
        "Missingness_Stationarity": missing,
        "Leakage": audit.leakage_table(table, COVARIATES),
        "Replay_Events": events_table,
        "Unknown_Settings_Detail": unknown_detail,
        "Monthly_Detail": audit.monthly_table(table),
        "issues": pd.concat(issues, ignore_index=True),
        "composition": composition, "eras": eras, "windows": audit.active_time_windows(table), "gaps": audit.gap_distribution(valid),
        "fortnights": audit.fortnight_series(table), "jump_table": jump_table, "reasons": reasons,
    }


def _no_reading_issues(table: pd.DataFrame) -> pd.DataFrame:
    flags = table["n_valid"].eq(0).to_numpy()
    return audit._issues(audit._run_issues(table, flags, step="no readings (hours)", reason="no valid CGM reading in the hour", field="n_valid", value="0"))


def _excluded_day_issues(table: pd.DataFrame) -> pd.DataFrame:
    flags = table["excluded_day"].to_numpy(dtype=bool)
    return audit._issues(audit._run_issues(table, flags, step="context-event day (hours)", reason="day listed in context_events.csv", field="excluded_day", value="True"))


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

def _specification() -> pd.DataFrame:
    rows = [
        ("Question", "Is the record fit to be modelled: what does each stream hold, where is it unreliable, and does anything in how the hour-day table is built change how a result must be read?"),
        ("Claim type", "Descriptive. Nothing here estimates an effect; the checks say where later estimates could be distorted."),
        ("Population and unit", "The whole record: every Tandem CGM reading, bolus, basal, carbohydrate entry, profile event, mode and site event, and the hour-day table (one row per clock hour of every day; the unit of the checks is stated in each row)."),
        ("Statuses", "no concern: the check ran and met its rule. concern: it ran and did not. not assessed: it could not be run (the reason says why). not applicable: descriptive, no rule."),
        ("Thresholds", f"Active time {int(100 * audit.ACTIVE_TIME_MIN)}% per {audit.WINDOW_DAYS}-day window (CGM consensus); flat run {audit.FLAT_RUN_SLOTS}+ identical readings; sensor range {audit.SENSOR_FLOOR_MGDL:.0f}-{audit.SENSOR_CEILING_MGDL:.0f} mg/dL; reconstruction agrees at {int(100 * audit.AGREEMENT_MIN)}% or more of compared moments; carbohydrate heaping at most {int(100 * audit.HEAPING_MAX_SHARE)}% multiples of 5 g. The thresholds are flags for reading results, not tests."),
        ("Unknown settings", "An hour's settings are unknown when any of its twelve 5-minute moments is not shown by the pump's profile history; the reason is the branch of ProfileHistory.values_at that returned no value (UNKNOWN_REASONS)."),
        ("Composition test", "Hours with known settings are standardised to the clock-hour and calendar-quarter mix of the unknown hours (direct standardisation, weights = the unknown hours' share of readings in each clock-hour x quarter cell); 90% intervals resample days."),
        ("Pump-change break", "The step in 28-day outcomes at the pump change against the steps between all other adjacent non-overlapping 28-day blocks."),
        ("Interpretation", "A concern is a reason to look before relying on a result, not a finding about the person's glucose."),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def _flow(table: pd.DataFrame, analysis, reasons: pd.Series) -> pd.DataFrame:
    flow = hour_day_flow(table)
    readings = int(table["n_valid"].sum())
    unknown_with = int((table["n_valid"].gt(0) & table["settings_known_slots"].lt(SLOTS_PER_HOUR)).sum())
    rows = [("clock hours in the record", flow.hours), ("hours with at least one reading", flow.hours_with_readings), ("hours without readings", flow.hours - flow.hours_with_readings),
            ("hours with all twelve moments' settings known", flow.hours_with_all_settings), ("hours with readings and settings not known for all twelve moments", unknown_with),
            ("hours with mixed settings (an edit inside the hour)", flow.hours_with_mixed_settings), ("hours on context-event days", flow.hours_on_excluded_days),
            ("days in the record", flow.hours // 24), ("valid CGM readings (sensor range, one per slot)", len(analysis.cgm)), ("readings counted in the hour-day table", readings)]
    rows += [(f"outcomes model rows: {stage}", int(n)) for stage, n in design_flow(table).itertuples(index=False)]
    return pd.DataFrame(rows, columns=["stage", "n"])


STEP_FLOW = {"no readings (hours)": "hours without readings", "settings unknown (hours with readings)": "hours with readings and settings not known for all twelve moments", "context-event day (hours)": "hours on context-event days"}


def _warnings() -> pd.DataFrame:
    rows = [
        ("Record_Issues holds identifiers", "Its rows carry dates, hours and pump serials: the workbook goes to the Project Vault with the outputs and never to Git or public documentation."),
        ("Not an estimate", "The audit reads one person's record and states what the record contains; thresholds flag a check for review and are not tests."),
        ("Pump clock", "Hours of day are the pump's wall-clock hours. Whether CGM readings, boluses and settings share that clock is checked through jumps in the pump's own event sequence and the phone's sleep periods, not by comparing the CGM stamp with the pump's receipt time."),
        ("LOW and HIGH", "The archive stores a glucose value per reading; readings below the sensor floor or above the ceiling are counted here and dropped by every analysis, so how they would enter an outcome is a separate decision."),
        ("Compression lows", "The compression-low check is a pattern (abrupt fall, short low, fast recovery at night), not a classification of a reading."),
        ("Leakage table", "The classification of when each covariate is known is a design judgement written in code; the settings column of the table gives how often the settings change over two weeks."),
        ("Event log ids", "Sensor sessions are read from event ids 213/394 (join) and 214/447 (stop). Time changes are read as jumps in time between consecutive events, not from the payload of event 13."),
    ]
    return pd.DataFrame(rows, columns=["limitation", "detail"])


def _provenance(table: pd.DataFrame, analysis, raw: str, run_id: str, seconds: float, files: list[str]) -> pd.DataFrame:
    manifest = analysis.merged.manifest
    dirty = not working_tree_is_clean(REPO_ROOT)
    rows = [("run id", run_id), ("commit", current_commit(REPO_ROOT)), ("working tree clean", "no" if dirty else "yes"), ("raw data root", raw), ("dataset version", manifest.get("dataset_version")), ("dataset built", manifest.get("built_at")),
            ("record span", f"{table['date'].min().date()} to {table['date'].max().date()}"), ("seed", audit.SEED), ("bootstrap draws", audit.DRAWS), ("command", "python -m bayesian_t1dm.data_audit"),
            ("wall time (seconds)", round(seconds, 1)), ("files", json.dumps(files))]
    return pd.DataFrame(rows, columns=["item", "value"])


def build_sheets(results: dict, table: pd.DataFrame, analysis, *, raw: str, run_id: str, seconds: float, files: list[str]) -> dict[str, pd.DataFrame]:
    """The workbook sheets in the diagnostics order: specification and flow first, method sheets, warnings and provenance last."""
    issues = results["issues"]
    sheets = {"Specification_Coding": _specification(), "Denominators_Flow": _flow(table, analysis, results["reasons"])}
    sheets.update({name: results[name] for name in CHECK_SHEETS[:3]})
    sheets["Replay_Events"] = results["Replay_Events"]
    sheets.update({name: results[name] for name in CHECK_SHEETS[3:]})
    sheets["Leakage"] = results["Leakage"]
    sheets["Unknown_Settings_Detail"] = results["Unknown_Settings_Detail"]
    sheets["Monthly_Detail"] = results["Monthly_Detail"]
    sheets["Record_Issues"] = issues.assign(start=issues["start"].astype(str), end=issues["end"].astype(str))
    sheets["Warnings_Limitations"] = _warnings()
    sheets["Artifacts_Provenance"] = _provenance(table, analysis, raw, run_id, seconds, files)
    return sheets


def reconcile(path: Path, table: pd.DataFrame, results: dict) -> None:
    """Read the workbook back: the flow must equal the hour-day table's, and Record_Issues must account for the hours the flow says were lost."""
    workbook = read_workbook(path)
    flow = workbook["Denominators_Flow"].set_index("stage")["n"]
    expected = hour_day_flow(table)
    for stage, value in (("clock hours in the record", expected.hours), ("hours with at least one reading", expected.hours_with_readings), ("hours with all twelve moments' settings known", expected.hours_with_all_settings),
                         ("hours with mixed settings (an edit inside the hour)", expected.hours_with_mixed_settings), ("hours on context-event days", expected.hours_on_excluded_days)):
        if int(flow[stage]) != value:
            raise RuntimeError(f"Workbook flow says {int(flow[stage])} for {stage!r}, the hour-day table says {value}")
    if int(flow["readings counted in the hour-day table"]) != int(flow["valid CGM readings (sensor range, one per slot)"]):
        raise RuntimeError("Readings in the hour-day table differ from the valid CGM readings")
    issues = workbook["Record_Issues"]
    for step, stage in STEP_FLOW.items():
        listed = int(issues.loc[issues["step"].eq(step), "n"].sum())
        if listed != int(flow[stage]):
            raise RuntimeError(f"Record_Issues lists {listed} hours for {step!r}, the flow says {int(flow[stage])}")
    if len(workbook["Replay_Events"]) != len(results["Replay_Events"]):
        raise RuntimeError("Replay_Events lost rows in the workbook")
    for name in CHECK_SHEETS:
        if workbook[name]["status"].isna().any():
            raise RuntimeError(f"Sheet {name} has a check without a status")
    leakage = workbook["Leakage"]
    for column in ("carbs_prev3h_g", "exercise_share", "sleep_mode_share", "site_age_days", "excluded_day"):
        row = leakage[leakage["column"].str.contains(column)]
        if row.empty or row["status"].iloc[0] != "concern":
            raise RuntimeError(f"Leakage table does not flag {column}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save(figure, output: Path, stem: str) -> list[Path]:
    import matplotlib.pyplot as plt

    paths = []
    for suffix in ("png", "svg"):
        path = output / f"{stem}.{suffix}"
        figure.savefig(path, dpi=200)
        paths.append(path)
    plt.close(figure)
    return paths


def _pump_change(table: pd.DataFrame) -> pd.Timestamp | None:
    pumps = table["pump_serial"].astype(str).to_numpy()
    change = np.flatnonzero(pumps[1:] != pumps[:-1]) + 1
    return pd.Timestamp(table["hour_start"].iloc[change[0]]) if len(change) else None


def _era_of(dates: pd.Series, change: pd.Timestamp | None) -> pd.Series:
    return pd.Series(np.where(dates < change, "first pump", "second pump") if change is not None else "first pump", index=dates.index)


def figure_cgm(results: dict, table: pd.DataFrame, output: Path) -> list[Path]:
    """Active time per 14-day window with the pump change, and where the missing minutes are (by gap length)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows, gaps = results["windows"], results["gaps"]
    change = _pump_change(table)
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [2.3, 1.1], "wspace": 0.28})
    era = _era_of(windows["start"], change)
    for name, style in PUMP_STYLE.items():
        part = windows[era.eq(name)]
        left.bar(part["start"] + pd.Timedelta(days=7), part["active_pct"], width=12.5, color=style.color, label=style.label, alpha=0.9)
    left.axhline(100 * audit.ACTIVE_TIME_MIN, color=NEUTRAL_LINE, linewidth=1.6, linestyle="--")
    left.text(windows["start"].min(), 100 * audit.ACTIVE_TIME_MIN - 4, "70% (consensus minimum)", fontsize=9, color=NEUTRAL_LINE, zorder=6, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    left.set_ylim(0, 118)
    left.set_ylabel("Readings present in the window (%)", fontsize=10)
    left.set_title("Sensor active time in each 14-day window", fontsize=11, loc="left")
    left.legend(frameon=False, fontsize=9, loc="upper center", ncol=2)
    right.barh(gaps["gap_length"], gaps["share_of_missing_minutes_pct"], color=NEUTRAL_BAR)
    for position, (share, count) in enumerate(zip(gaps["share_of_missing_minutes_pct"], gaps["gaps"])):
        right.text(share + 1, position, f"{share:.0f}%  ({count} gaps)", va="center", fontsize=8.5)
    right.set_xlim(0, max(60, gaps["share_of_missing_minutes_pct"].max() * 1.5))
    right.invert_yaxis()
    right.set_xlabel("Share of all missing minutes (%)", fontsize=10)
    right.set_title("Missing minutes by gap length", fontsize=11, loc="left")
    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
    figure.suptitle("How to read: a bar above the dashed line is a 14-day window with enough readings; the right panel shows which gap lengths make up the missing time", fontsize=9.5, x=0.01, ha="left", y=0.995)
    assert_ascii(*[t.get_text() for t in figure.findobj(lambda a: hasattr(a, "get_text"))])
    figure.subplots_adjust(top=0.86, bottom=0.14, left=0.07, right=0.97)
    return _save(figure, output, FIGURE_FILES[0])


def figure_settings(results: dict, table: pd.DataFrame, output: Path) -> list[Path]:
    """Hours with readings and unknown settings: clock hour by month, and by reason and pump."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reasons = results["reasons"]
    unknown = table[reasons.ne("") & table["n_valid"].gt(0)].assign(reason=reasons, month=lambda f: pd.to_datetime(f["date"]).dt.to_period("M").astype(str))
    months = sorted(table["date"].dt.to_period("M").astype(str).unique())
    grid = unknown.pivot_table(index="hour", columns="month", values="n_valid", aggfunc="size", fill_value=0).reindex(index=range(24), columns=months, fill_value=0)
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [2.4, 1.3], "wspace": 0.45})
    image = left.imshow(grid.to_numpy(), aspect="auto", cmap="Blues", vmin=0, origin="upper")
    left.set_yticks(range(0, 24, 3))
    left.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], fontsize=9)
    ticks = list(range(0, len(months), 4))
    left.set_xticks(ticks)
    left.set_xticklabels([months[i] for i in ticks], rotation=45, ha="right", fontsize=8.5)
    left.set_title("Hours with readings and unknown settings\n(count in each clock hour and month)", fontsize=11, loc="left")
    colorbar = figure.colorbar(image, ax=left, fraction=0.04, pad=0.02)
    colorbar.set_label("Hours", fontsize=9)
    if unknown.empty:
        right.text(0.5, 0.5, "No hour with readings has unknown settings", ha="center", va="center", transform=right.transAxes, fontsize=10)
    else:
        counts = unknown.groupby(["pump_serial", "reason"]).size().unstack(fill_value=0)
        pumps = sorted(counts.index, key=lambda p: table.loc[table["pump_serial"].eq(p), "date"].min())
        labels = [list(PUMP_STYLE.values())[i].label for i in range(len(pumps))]
        bottom = np.zeros(len(pumps))
        for reason in UNKNOWN_REASONS:
            if reason not in counts.columns:
                continue
            values = counts.reindex(pumps)[reason].to_numpy()
            right.bar(labels, values, bottom=bottom, color=REASON_COLORS[reason], label=reason.replace("_", " "))
            bottom += values
        right.set_ylim(0, bottom.max() * 1.45)
        right.legend(frameon=False, fontsize=8, loc="upper right")
        right.set_ylabel("Hours with readings and unknown settings", fontsize=10)
        right.set_title("Why the settings are unknown", fontsize=11, loc="left")
    right.spines[["top", "right"]].set_visible(False)
    left.set_ylabel("Clock hour (pump wall clock)", fontsize=10)
    figure.suptitle("How to read: a dark row means a cause tied to that hour of the day; a dark column means a cause tied to that month", fontsize=9.5, x=0.01, ha="left", y=0.995)
    assert_ascii(*[t.get_text() for t in figure.findobj(lambda a: hasattr(a, "get_text"))])
    figure.subplots_adjust(top=0.86, bottom=0.17, left=0.07, right=0.97)
    return _save(figure, output, FIGURE_FILES[1])


def figure_stationarity(results: dict, table: pd.DataFrame, output: Path) -> list[Path]:
    """Time in range of consecutive 14-day windows with the pump change, and the composition test."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fortnights, composition = results["fortnights"], results["composition"]
    change = _pump_change(table)
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2.3, 1.5], "wspace": 0.55})
    era = _era_of(fortnights["start"], change)
    earlier = fortnights.iloc[:-1]
    low, high = np.percentile(earlier["tir_pct"], [5, 95])
    left.axhspan(low, high, color=NEUTRAL_BAND, zorder=0)
    left.text(fortnights["start"].min(), high + 0.8, "5th to 95th percentile of earlier fortnights", fontsize=8.5, color=NEUTRAL_LINE)
    for name, style in PUMP_STYLE.items():
        part = fortnights[era.eq(name)]
        left.plot(part["start"], part["tir_pct"], marker=style.marker, linestyle=style.linestyle, color=style.color, linewidth=2.0, markersize=5, label=style.label)
    last = fortnights.iloc[-1]
    left.scatter([last["start"]], [last["tir_pct"]], s=110, facecolors="none", edgecolors=NEUTRAL_LINE, linewidths=1.8, zorder=4)
    left.annotate("latest fortnight", (last["start"], last["tir_pct"]), textcoords="offset points", xytext=(-14, -3), ha="right", fontsize=9)
    left.set_ylabel("Time in range 70-180 mg/dL (%)", fontsize=10)
    left.set_title("Time in range of consecutive 14-day windows", fontsize=11, loc="left")
    left.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    left.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    left.legend(frameon=False, fontsize=9, loc="lower left")
    if composition.empty:
        right.text(0.5, 0.5, "No hour with unknown settings", ha="center", va="center", transform=right.transAxes)
    else:
        labels = ["Known settings,\nas observed", "Known settings,\nstandardised to the\nunknown hours' mix", "Unknown settings"]
        colors = [NEUTRAL_BAR, NEUTRAL_LINE, PUMP_STYLE["second pump"].color]
        for position, (_, row) in enumerate(composition.iterrows()):
            lo, hi = (float(x) for x in row["interval_90"].split(" to "))
            right.errorbar(row["time_in_range_pct"], position, xerr=[[row["time_in_range_pct"] - lo], [hi - row["time_in_range_pct"]]], fmt="o", color=colors[position], capsize=4, linewidth=2.2, markersize=7)
            right.text(row["time_in_range_pct"], position + 0.25, f"{row['time_in_range_pct']:.1f}", ha="center", fontsize=9)
        right.set_yticks(range(3))
        right.set_yticklabels(labels, fontsize=9)
        right.set_ylim(2.7, -0.5)
        right.set_xlabel("Time in range (%), 90% interval over days", fontsize=10)
        right.set_title("Do unknown-settings hours differ\nbeyond when they happen?", fontsize=11, loc="left")
    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
    figure.suptitle("How to read: a fortnight outside the grey band is unusual for this record. Right: if the black and red dots agree, unknown hours differ only by when they fall", fontsize=9.5, x=0.01, ha="left", y=0.995)
    assert_ascii(*[t.get_text() for t in figure.findobj(lambda a: hasattr(a, "get_text"))])
    figure.subplots_adjust(top=0.84, bottom=0.14, left=0.07, right=0.97)
    return _save(figure, output, FIGURE_FILES[2])


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(path: Path, sheets: dict[str, pd.DataFrame], run_id: str) -> None:
    """A short markdown summary generated from the workbook's check rows: the flow, then every concern with its reason."""
    flow = sheets["Denominators_Flow"].set_index("stage")["n"]
    lines = [f"# Data-quality audit, run {run_id}", "", "Generated from `diagnostics.xlsx` in this folder; every number is in the workbook.", "", "## Flow", ""]
    for stage in ("clock hours in the record", "hours with at least one reading", "hours with all twelve moments' settings known", "hours with readings and settings not known for all twelve moments", "hours on context-event days"):
        lines.append(f"- {stage}: {int(flow[stage])}")
    status_counts = pd.concat([sheets[name]["status"] for name in (*CHECK_SHEETS, "Leakage")]).value_counts()
    lines += ["", "## Check statuses", "", ", ".join(f"{status}: {count}" for status, count in status_counts.items()), "", "## Concerns", ""]
    for name in (*CHECK_SHEETS, "Leakage"):
        frame = sheets[name]
        for _, row in frame[frame["status"].eq("concern")].iterrows():
            title = row["check"] if "check" in frame.columns else row["column"]
            result = row["result"] if "result" in frame.columns else row["known_when"]
            lines += [f"- **{title}** ({name}): {result}. {row['reason']}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(raw: str | None, output: str | None) -> int:
    """Load the record, run every check, write the workbook, figures and summary, and read the workbook back."""
    from .acquisition import load_local_env_file
    from .analysis_data import load_analysis_data
    from .paths import ProjectPaths

    paths = ProjectPaths.from_root(Path.cwd())
    for key, value in load_local_env_file(paths.root / ".env").items():
        os.environ.setdefault(key, value)
    raw_root = raw or str(paths.cloud_raw)
    folder = Path(output or paths.output_data_audit)
    folder.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(folder / f"data_audit_run_{run_id}.log")])
    started = time.time()
    LOGGER.info("data audit started run_id=%s raw=%s", run_id, raw_root)
    analysis = load_analysis_data(raw_root)
    table = build_hour_day_table(analysis)
    LOGGER.info("hour-day table %s", json.dumps(hour_day_flow(table).__dict__))
    event_log = read_event_log(Path(raw_root) / "tconnectsync")
    results = run_checks(analysis, table, event_log)
    files = [f"{stem}.{suffix}" for stem in FIGURE_FILES for suffix in ("png", "svg")] + ["diagnostics.xlsx", "data_audit_summary.md"]
    figure_cgm(results, table, folder)
    figure_settings(results, table, folder)
    figure_stationarity(results, table, folder)
    sheets = build_sheets(results, table, analysis, raw=raw_root, run_id=run_id, seconds=time.time() - started, files=files)
    path = write_workbook(folder / "diagnostics.xlsx", sheets)
    reconcile(path, table, results)
    write_summary(folder / "data_audit_summary.md", read_workbook(path), run_id)
    LOGGER.info("data audit finished in %.1f s; outputs in %s", time.time() - started, folder)
    return 0
