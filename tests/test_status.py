from __future__ import annotations

import json

import pandas as pd

from bayesian_t1dm.features import FeatureConfig
from bayesian_t1dm.health_auto_export import AnalysisReadyHealthDataset, ModelDataPreparationResult
from bayesian_t1dm.paths import ProjectPaths
from bayesian_t1dm.status import cleanup_legacy_top_level_output, derive_current_status, finalize_status_logs, reset_output_directory, write_status_json


def _dataset(*, apple: bool = True) -> AnalysisReadyHealthDataset:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01 00:00:00", periods=96, freq="5min"),
            "glucose": [110.0] * 96,
            "target_delta": [0.0] * 96,
            "heart_rate_avg_latest": [70.0] * 96,
        }
    )
    if not apple:
        frame = frame.drop(columns=["heart_rate_avg_latest"], errors="ignore")
    return AnalysisReadyHealthDataset(
        frame=frame,
        feature_columns=["glucose"],
        tandem_feature_columns=["glucose"],
        health_feature_columns=["heart_rate_avg_latest"] if apple else [],
        target_column="target_delta",
        horizon_minutes=30,
        config=FeatureConfig(horizon_minutes=30),
        mode="apple_enriched" if apple else "tandem_only",
        apple_available=apple,
    )


def _preparation(dataset: AnalysisReadyHealthDataset) -> ModelDataPreparationResult:
    start = pd.Timestamp("2025-01-01 00:00:00")
    end = pd.Timestamp("2025-01-05 00:00:00")
    return ModelDataPreparationResult(
        dataset=dataset,
        apple_available=dataset.apple_available,
        apple_span_start=start,
        apple_span_end=end,
        tandem_span_before_start=start,
        tandem_span_before_end=end,
        tandem_span_after_start=start,
        tandem_span_after_end=end,
        requested_tandem_start=start,
        requested_tandem_end=end,
        overlap_start=start,
        overlap_end=end,
        final_dataset_start=start,
        final_dataset_end=end,
        final_row_count=int(len(dataset.frame)),
        backfill_status="not_needed",
    )


def _data_quality(status: str = "good") -> dict[str, object]:
    return {"status": status, "incomplete_window_count": 0 if status == "good" else 2}


def test_a_complete_source_is_data_ready_and_says_nothing_about_settings():
    payload = derive_current_status(preparation=_preparation(_dataset()), data_quality=_data_quality(), run_id="run123")

    assert payload["overall_state"] == "data_ready"
    assert payload["primary_blockers"] == []
    assert "therapy" not in payload
    assert payload["data_prep"]["final_row_count"] == 96


def test_degraded_source_quality_blocks_the_status():
    payload = derive_current_status(preparation=_preparation(_dataset()), data_quality=_data_quality("degraded"), run_id="run123")

    assert payload["overall_state"] == "blocked"
    assert [item["code"] for item in payload["primary_blockers"]] == ["degraded_source_quality"]
    assert payload["source_quality"]["incomplete_window_count"] == 2


def test_output_cleanup_moves_top_level_clutter_into_runtime_archive(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    paths = ProjectPaths.from_root(repo_root, runtime_root=tmp_path / "runtime", cloud_root=tmp_path / "cloud").ensure()
    (paths.reports / "old.csv").write_text("old", encoding="utf-8")
    (paths.reports / "scratch").mkdir(parents=True, exist_ok=True)
    (paths.reports / "scratch" / "note.txt").write_text("note", encoding="utf-8")

    manifest_path = cleanup_legacy_top_level_output(paths)

    assert manifest_path is not None
    assert not (paths.reports / "old.csv").exists()
    moved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(moved["moved"]) == 2
    assert moved["migrated_logs"] == []
    assert manifest_path.parent == paths.legacy_output_archive

    log_dir = paths.logs / "status" / "run123"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")
    assert finalize_status_logs(paths, log_dir) == log_dir


def test_output_cleanup_migrates_legacy_output_logs_into_runtime_logs(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    paths = ProjectPaths.from_root(repo_root, runtime_root=tmp_path / "runtime", cloud_root=tmp_path / "cloud").ensure()
    legacy_log_dir = paths.reports / "logs" / "status" / "run123"
    legacy_log_dir.mkdir(parents=True, exist_ok=True)
    (legacy_log_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (paths.reports / "logs" / "status" / "latest.json").write_text('{"run_dir":"old"}\n', encoding="utf-8")

    manifest_path = cleanup_legacy_top_level_output(paths)

    assert manifest_path is not None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(payload["migrated_logs"]) == 2
    assert not (paths.reports / "logs").exists()
    assert (paths.logs / "status" / "run123" / "events.jsonl").exists()
    assert (paths.logs / "status" / "latest.json").exists()


def test_reset_output_directory_replaces_stale_contents(tmp_path):
    directory = tmp_path / "output" / "stale"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stale.md").write_text("stale", encoding="utf-8")

    reset_output_directory(directory)

    assert directory.exists()
    assert list(directory.iterdir()) == []


def test_write_status_json_persists_expected_keys(tmp_path):
    path = tmp_path / "current_status.json"
    write_status_json({"overall_state": "blocked", "run_id": "abc123"}, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["overall_state"] == "blocked"
    assert payload["run_id"] == "abc123"
