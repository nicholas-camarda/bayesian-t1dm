from __future__ import annotations

import numpy as np
import pandas as pd

from bayesian_t1dm.outcomes import DayPart
from bayesian_t1dm.proposals import Proposal
from bayesian_t1dm.records import RECORD_COLUMNS, evaluate_open, load_records, register_proposal, save_records, sync_with_pump

PARTS = {300.0: DayPart("05:00-09:00", 300, 540), 540.0: DayPart("09:00-15:00", 540, 900)}
PROPOSAL = Proposal(setting="breakfast I:C", status="proposed", reason="carbs behaved like 6.5", pump_serial="1", profile_id=0.0,
                    segment_start_minutes=300.0, affected_hours="05:00-13:00", current_value=7.0, proposed_value=6.5, direction="adds insulin")


def _change(when, start, setting="carb_ratio_g_per_unit", old=7.0, new=6.5, profile=0, pump="1"):
    return {"timestamp": pd.Timestamp(when), "pump_serial": pump, "profile_id": profile, "segment_index": 1, "segment_start_minutes": start, "setting": setting, "old_value": old, "new_value": new}


def test_a_second_proposal_is_not_recorded_while_one_is_open():
    records, note = register_proposal(load_records("/nonexistent/x.csv"), PROPOSAL, as_of=pd.Timestamp("2026-11-18"))
    assert note == "recorded" and records.loc[0, "status"] == "proposed"
    again, note = register_proposal(records, PROPOSAL, as_of=pd.Timestamp("2026-11-19"))
    assert len(again) == 1 and note.startswith("not recorded")


def test_a_matching_pump_change_marks_a_proposal_entered_and_other_changes_become_person_records():
    records, _ = register_proposal(load_records("/nonexistent/x.csv"), PROPOSAL, as_of=pd.Timestamp("2026-11-18"))
    changes = pd.DataFrame([
        _change("2026-11-19 09:30", 300.0),
        _change("2026-11-19 09:31", 540.0),
        _change("2026-11-10 08:00", 300.0, new=7.5, old=8.0),  # before 'since'
    ])
    synced = sync_with_pump(records, changes, PARTS, since=pd.Timestamp("2026-11-15"))
    system = synced[synced["origin"].eq("system")].iloc[0]
    assert system["status"] == "entered" and pd.Timestamp(system["entered_on"]) == pd.Timestamp("2026-11-19 09:30")
    person = synced[synced["origin"].eq("person")]
    assert len(person) == 1 and person.iloc[0]["segment_start_minutes"] == 540.0 and person.iloc[0]["concurrent_changes"] == 1
    assert person.iloc[0]["affected_hours"] == "09:00-19:00"


def test_evaluation_waits_then_records_the_result():
    cgm = pd.DataFrame({"timestamp": pd.date_range("2026-10-20", "2026-12-10", freq="5min"), "glucose": 120.0})
    records = pd.DataFrame([{"record_id": "r", "origin": "person", "setting": "carb_ratio_g_per_unit", "segment_start_minutes": 300.0, "status": "entered", "entered_on": pd.Timestamp("2026-11-17 09:29")}])
    waiting = evaluate_open(records, cgm, PARTS, now=pd.Timestamp("2026-11-25"))
    assert waiting.loc[0, "status"] == "entered" and waiting.loc[0, "result"].startswith("pending")
    done = evaluate_open(records, cgm, PARTS, now=pd.Timestamp("2026-12-05"))
    assert done.loc[0, "status"] == "evaluated" and "no clear change in time in range" in done.loc[0, "result"]


def test_saving_keeps_the_previous_version(tmp_path):
    path = tmp_path / "records" / "setting_proposals.csv"
    first, _ = register_proposal(load_records(path), PROPOSAL, as_of=pd.Timestamp("2026-11-18"))
    save_records(first, path)
    save_records(first.assign(status="entered"), path)
    assert list(load_records(path).columns) == RECORD_COLUMNS and len(list((path.parent / "archive").glob("*.csv"))) == 1
    assert not np.isnan(load_records(path).loc[0, "old_value"])


def test_the_serial_form_does_not_change_whether_a_proposal_is_matched():
    from bayesian_t1dm.pump_profiles import pump_key

    assert {pump_key(v) for v in ("1453346.0", "1453346", 1453346.0, " 1453346 ")} == {"1453346"} and pump_key(np.nan) == "" and pump_key(None) == ""
    proposal = Proposal(setting="breakfast I:C", status="proposed", reason="r", pump_serial="1453346.0", profile_id=0.0, segment_start_minutes=300.0,
                        affected_hours="05:00-13:00", current_value=7.0, proposed_value=6.5, direction="adds insulin")
    records, _ = register_proposal(load_records("/nonexistent/x.csv"), proposal, as_of=pd.Timestamp("2026-11-18"))
    synced = sync_with_pump(records, pd.DataFrame([_change("2026-11-19 09:30", 300.0, pump="1453346")]), PARTS, since=pd.Timestamp("2026-11-15"))
    assert synced["origin"].tolist() == ["system"] and synced.loc[0, "status"] == "entered" and synced.loc[0, "pump_serial"] == "1453346"
