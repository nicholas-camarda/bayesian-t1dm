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
    frame.loc[0, "bolus_units"] = 0.0
    frame.loc[2, "glucose_observed"] = -5.0  # no sensor returns this: it fails, unlike a reading at the sensor's limit
    status, reasons = assess_prepared_feature_plausibility(frame)
    assert status == "failed" and reasons["impossible_cgm_rows"] == 1


def test_a_removed_raw_file_rebuilds_its_window(tmp_path):
    diff = {"added": [], "changed": [], "removed": ["tconnectsync/w1/raw/a.json", "tconnectsync/w2/raw/b.json", "carbs.csv"]}
    assert dataset._changed_tandem_windows(diff) == ["w1", "w2"]


def test_a_timezone_change_forces_a_full_rebuild_record():
    import inspect

    source = inspect.getsource(dataset.build_merged_dataset)
    assert 'previous.get("timezone")' in source and '"timezone": timezone_name' in source


def test_every_table_the_ingest_reads_is_fingerprinted_and_derived_tables_are_not(tmp_path):
    import pandas as pd

    (tmp_path / "sub").mkdir()
    pd.DataFrame({"timestamp": ["2025-05-24 08:00:00"], "carb_grams": [30]}).to_parquet(tmp_path / "sub" / "carb_log.parquet")
    (tmp_path / "normalized").mkdir()
    (tmp_path / "normalized" / "cgm.csv").write_text("timestamp,glucose\n", encoding="utf-8")
    found = {(item.path, item.kind) for item in dataset.fingerprint_inputs(tmp_path)}
    assert found == {("sub/carb_log.parquet", "carb_log")}


def test_normalized_tables_of_a_window_whose_raw_files_were_removed_are_deleted(tmp_path):
    root = tmp_path / "tconnectsync"
    (root / "w1" / "raw").mkdir(parents=True)
    (root / "w1" / "normalized").mkdir()
    (root / "w2" / "normalized").mkdir(parents=True)  # a window kept as derived tables only is left alone unless its raw files were removed
    (root / "w3" / "normalized").mkdir(parents=True)
    removed = ["tconnectsync/w2/raw/events.txt"]
    dataset._normalize_windows(None, tmp_path / "tconnectsync", {"added": [], "changed": [], "removed": removed}, rebuild_all=False, log=lambda *_: None)
    assert (root / "w1" / "normalized").exists() and (root / "w3" / "normalized").exists() and not (root / "w2" / "normalized").exists()
