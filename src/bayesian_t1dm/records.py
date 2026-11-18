"""The durable record of setting changes: proposed by the system or made by the person, and how each turned out.

One row per changed value. ``status`` runs proposed -> entered (the pump's profile history shows the new value) ->
evaluated (the 14-day before/after comparison has run). A change the person made on their own is recorded as
``origin = person`` so it is evaluated the same way. The file lives in the Project Vault; a copy of the previous
version is kept before every rewrite.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .changes import evaluate_change
from .outcomes import DayPart
from .proposals import EVALUATION, Proposal, affected_part

RECORD_COLUMNS = [
    "record_id", "origin", "proposed_on", "setting", "pump_serial", "profile_id", "segment_start_minutes", "affected_hours",
    "old_value", "new_value", "reason", "evaluation_plan", "status", "entered_on", "concurrent_changes", "evaluated_on", "result",
]
OPEN_STATUSES = ("proposed", "entered")


def load_records(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=RECORD_COLUMNS)
    return pd.read_csv(path, dtype={"pump_serial": str, "record_id": str}).reindex(columns=RECORD_COLUMNS)


def save_records(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write the record, keeping the previous file under ``archive/`` beside it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        archive = path.parent / "archive"
        archive.mkdir(exist_ok=True)
        shutil.copy2(path, archive / f"{path.stem}_{pd.Timestamp.now():%Y%m%d_%H%M%S}{path.suffix}")
    frame.reindex(columns=RECORD_COLUMNS).to_csv(path, index=False)
    return path


def _record_id(when: Any, setting: str, segment_start: float) -> str:
    return f"{pd.Timestamp(when):%Y%m%d}-{setting}-{int(segment_start):04d}"


def register_proposal(records: pd.DataFrame, proposal: Proposal, *, as_of: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    """Add a proposed change, unless one is already open (one setting at a time, held until evaluated)."""
    if proposal.status != "proposed":
        return records, "no proposal to record"
    open_rows = records[records["status"].isin(OPEN_STATUSES)]
    if not open_rows.empty:
        return records, f"not recorded: {len(open_rows)} earlier change(s) are still open ({', '.join(open_rows['record_id'])})"
    row = {
        "record_id": _record_id(as_of, "carb_ratio_g_per_unit", proposal.segment_start_minutes) + f"-{proposal.pump_serial}", "origin": "system",
        "proposed_on": as_of.date(), "setting": "carb_ratio_g_per_unit", "pump_serial": proposal.pump_serial, "profile_id": proposal.profile_id,
        "segment_start_minutes": proposal.segment_start_minutes, "affected_hours": proposal.affected_hours, "old_value": proposal.current_value,
        "new_value": proposal.proposed_value, "reason": proposal.reason, "evaluation_plan": EVALUATION, "status": "proposed",
    }
    return pd.concat([records, pd.DataFrame([row])], ignore_index=True), "recorded"


def sync_with_pump(records: pd.DataFrame, changes: pd.DataFrame, parts_by_start: dict[float, DayPart], *, since: pd.Timestamp) -> pd.DataFrame:
    """Mark proposals the pump now shows as entered, and record changes the person made on their own.

    A change counts for a proposal when it is on the same pump, profile, segment and setting, made on or after the
    proposal's day, and reaches the proposed value. Every other change since ``since`` becomes a ``person`` record.
    """
    records = records.copy()
    changes = changes[pd.to_datetime(changes["timestamp"]) >= since].sort_values("timestamp")
    used: set[int] = set()
    for index, record in records[records["status"].eq("proposed")].iterrows():
        match = changes[
            changes["pump_serial"].eq(str(record["pump_serial"])) & changes["profile_id"].eq(int(record["profile_id"]))
            & changes["segment_start_minutes"].eq(float(record["segment_start_minutes"])) & changes["setting"].eq(record["setting"])
            & np.isclose(changes["new_value"], float(record["new_value"]))
            & (pd.to_datetime(changes["timestamp"]).dt.date >= pd.Timestamp(record["proposed_on"]).date())
        ]
        if not match.empty:
            records.loc[index, ["status", "entered_on"]] = ["entered", pd.Timestamp(match.iloc[0]["timestamp"])]
            used.add(match.index[0])
    known = set(records["record_id"])
    new_rows = []
    for index, change in changes.iterrows():
        if index in used or change["setting"] == "start_minutes":
            continue
        rid = _record_id(change["timestamp"], change["setting"], change["segment_start_minutes"]) + f"-{change['pump_serial']}"
        if rid in known:
            continue
        part = parts_by_start.get(float(change["segment_start_minutes"]))
        hours = affected_part(int(part.start_minutes), int(part.end_minutes)).name if part is not None else ""
        nearby = abs(pd.to_datetime(changes["timestamp"]) - pd.Timestamp(change["timestamp"])) <= pd.Timedelta(hours=24)
        new_rows.append(
            {
                "record_id": rid, "origin": "person", "proposed_on": pd.NaT, "setting": change["setting"], "pump_serial": change["pump_serial"],
                "profile_id": change["profile_id"], "segment_start_minutes": change["segment_start_minutes"], "affected_hours": hours,
                "old_value": change["old_value"], "new_value": change["new_value"], "reason": "change made on the pump; found in the profile history",
                "status": "entered", "entered_on": pd.Timestamp(change["timestamp"]), "concurrent_changes": int((nearby & (changes.index != index)).sum()),
            }
        )
        known.add(rid)
    if new_rows:
        records = pd.concat([records, pd.DataFrame(new_rows)], ignore_index=True)
    return records


def evaluate_open(records: pd.DataFrame, cgm: pd.DataFrame, parts_by_start: dict[float, DayPart], *, now: pd.Timestamp, excluded_days=()) -> pd.DataFrame:
    """Run the before/after comparison for every entered change whose after-window has been observed."""
    records = records.copy()
    for index, record in records[records["status"].eq("entered")].iterrows():
        part = parts_by_start.get(float(record["segment_start_minutes"]))
        if part is None:
            continue
        window = affected_part(int(part.start_minutes), int(part.end_minutes))
        result = evaluate_change(cgm, pd.Timestamp(record["entered_on"]), window, now=now, excluded_days=excluded_days)
        if result["status"] == "pending":
            records.loc[index, "result"] = f"pending: {result['reason']}"
        elif result["status"] == "insufficient":
            records.loc[index, "result"] = f"not evaluable: {result['reason']}"
        else:
            records.loc[index, ["status", "evaluated_on", "result"]] = [
                "evaluated",
                now.date(),
                f"{result['verdict']}: in range {result['tir_before']:.1f}% -> {result['tir_after']:.1f}% "
                f"(difference {result['tir_difference']:+.1f} points, 90% interval {result['tir_difference_lower_90']:+.1f} to {result['tir_difference_upper_90']:+.1f}); "
                f"below 70 {result['below_70_after']:.2f}%, below 54 {result['below_54_after']:.2f}% ({result['low_limits_after']}) over {result['hours']}",
            ]
    return records
