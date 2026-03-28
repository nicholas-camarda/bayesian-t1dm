from __future__ import annotations

from bayesian_t1dm.ingest import load_tandem_exports, summarize_coverage


def test_load_tandem_exports_normalizes_fixture_data(tandem_fixture_dir):
    data = load_tandem_exports(tandem_fixture_dir)

    assert len(data.source_files) == 4
    assert {"timestamp", "glucose", "source_file"}.issubset(data.cgm.columns)
    assert {"timestamp", "bolus_units", "source_file"}.issubset(data.bolus.columns)
    assert {"start_timestamp", "end_timestamp", "basal_units_per_hour"}.issubset(data.basal.columns)
    assert {"timestamp", "activity_value", "source_file"}.issubset(data.activity.columns)

    coverage = summarize_coverage(data)
    assert coverage.cgm_rows == 7
    assert coverage.bolus_rows == 2
    assert coverage.basal_rows == 1
    assert coverage.activity_rows == 2
    assert str(coverage.first_timestamp) == "2023-06-01 00:00:00"
    assert str(coverage.last_timestamp) == "2023-06-02 00:00:00"
