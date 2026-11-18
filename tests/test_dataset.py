from __future__ import annotations

import os
from dataclasses import asdict

import bayesian_t1dm.dataset as dataset


def test_fingerprint_reuses_recorded_hashes_and_rehashes_edited_files(tmp_path, monkeypatch):
    carbs = tmp_path / "carbs.csv"
    carbs.write_text("timestamp,carb_grams\n2025-05-24 08:00:00,30\n", encoding="utf-8")
    first = dataset.fingerprint_inputs(tmp_path)
    assert [item.path for item in first] == ["carbs.csv"]

    hashed: list[str] = []
    real_sha = dataset._sha256
    monkeypatch.setattr(dataset, "_sha256", lambda path: hashed.append(path.name) or real_sha(path))

    # Same size and modification time: the recorded hash is reused without reading the file.
    again = dataset.fingerprint_inputs(tmp_path, known=[asdict(item) for item in first])
    assert again == first and hashed == []

    # An edit changes the modification time, so the file is read and the new hash differs.
    carbs.write_text("timestamp,carb_grams\n2025-05-24 08:00:00,45\n", encoding="utf-8")
    os.utime(carbs, ns=(first[0].mtime_ns + 1_000_000_000, first[0].mtime_ns + 1_000_000_000))
    edited = dataset.fingerprint_inputs(tmp_path, known=[asdict(item) for item in first])
    assert hashed == ["carbs.csv"]
    assert dataset._input_diff([asdict(item) for item in first], edited)["changed"] == ["carbs.csv"]


def test_cgm_gaps_are_reported_but_only_impossible_values_fail_plausibility():
    import pandas as pd

    from bayesian_t1dm.quality import assess_prepared_feature_plausibility

    frame = pd.DataFrame({"glucose_observed": [120.0, 39.0, 150.0], "minutes_since_last_cgm": [5.0, 150.0, 5.0], "bolus_units": [0.0, 1.0, 0.0]})
    assert assess_prepared_feature_plausibility(frame) == ("good", {"cgm_gap_ge_120m_rows": 1, "invalid_cgm_rows": 1})
    frame.loc[0, "bolus_units"] = -1.0
    status, reasons = assess_prepared_feature_plausibility(frame)
    assert status == "failed" and reasons["negative_bolus_rows"] == 1
