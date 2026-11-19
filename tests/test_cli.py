from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import bayesian_t1dm.cli as cli
from bayesian_t1dm.cli import main


def _fake_merged(preparation):
    """A merged dataset with the parts the research commands read: no food log, no workouts, no pump modes, an empty analysis grid."""
    grid = pd.DataFrame(columns=["timestamp", "glucose", "carb_grams", "bolus_units"])
    return SimpleNamespace(preparation=preparation, data_quality=DataQualitySummary(status="degraded", contributing_window_ids=[], incomplete_window_count=2, reason_counts={}, evaluation_touches_incomplete_windows=False), grid=grid, mode_events=pd.DataFrame(), data=SimpleNamespace(nutrition=pd.DataFrame(), workouts=pd.DataFrame()))
from bayesian_t1dm.acquisition import ExportWindow, NormalizedWindowResult
from bayesian_t1dm.health_auto_export import ModelDataPreparationResult
from bayesian_t1dm.quality import DataQualitySummary
from bayesian_t1dm.therapy_research import TherapyInfraValidationResult, TherapyResearchResult, _synthetic_base_dataset


def _latest_log_dir(runtime_root: Path, command: str) -> Path:
    latest = json.loads((runtime_root / "logs" / command / "latest.json").read_text(encoding="utf-8"))
    return Path(latest["run_dir"])


def _write_tandem_cgm_export(raw_dir: Path, *, start: str = "2025-05-24 00:00:00", periods: int = 288) -> Path:
    workbook = raw_dir / "therapy_events_2025-05.xlsx"
    frame = pd.DataFrame(
        {
            "eventDateTime": pd.date_range(start, periods=periods, freq="5min"),
            "egv_estimatedGlucoseValue": [110 + (index % 18) for index in range(periods)],
        }
    )
    with pd.ExcelWriter(workbook) as writer:
        frame.to_excel(writer, index=False)
    return workbook


def _fake_status_research_result() -> TherapyResearchResult:
    dataset = _synthetic_base_dataset(apple=True, explicit_carbs=True)
    research_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01 00:00:00", periods=12, freq="5min"),
            "therapy_segment": ["overnight"] * 12,
            "basal_context": [0] * 12,
            "therapy_stable_epoch": [0] * 12,
            "glucose": [110.0] * 12,
            "target_delta": [0.0] * 12,
            "recent_meal_120m": [1] * 12,
            "recent_bolus_120m": [1] * 12,
            "recent_exercise_context": [0] * 12,
            "closed_loop_confounding_flag": [1] * 12,
            "missing_cgm": [0] * 12,
        }
    )
    gate = pd.DataFrame(
        [
            {
                "parameter": "basal",
                "identifiability": "not_identified",
                "gate_status": "diagnostics_only",
                "source_quality_status": "good",
                "direct_meal_rows": 0,
                "proxy_meal_rows": 12,
                "basal_context_rows": 0,
                "correction_context_rows": 12,
                "closed_loop_confounding_risk": "high",
                "apple_alignment_status": "credible",
            }
        ]
    )
    recommendations = pd.DataFrame(
        [
            {
                "parameter": "basal",
                "segment": "overnight",
                "status": "suppressed",
                "proposed_change_percent": None,
                "expected_direction": "hold",
                "mean_expected_gain": 0.1,
                "fold_better_fraction": 0.5,
                "confidence": "low",
                "reasons_for": "",
                "reasons_against": "expected_gain_too_small",
                "identifiability": "not_identified",
            }
        ]
    )
    return TherapyResearchResult(
        prepared_dataset=dataset,
        research_frame=research_frame,
        research_gate=gate,
        feature_registry=pd.DataFrame(),
        meal_proxy_audit=pd.DataFrame(),
        model_comparison=pd.DataFrame(),
        segment_evidence=pd.DataFrame(),
        recommendations=recommendations,
        research_gate_markdown="# gate\n",
        feature_audit_markdown="# audit\n",
        meal_proxy_audit_markdown="# meal\n",
        model_comparison_markdown="# model\n",
        recommendation_markdown="# recs\n",
        tandem_source_report_markdown="# tandem\n",
        apple_source_report_markdown="# apple\n",
        source_numeric_summary=pd.DataFrame(),
        source_missingness_summary=pd.DataFrame(),
        segments=tuple(),
        include_models=("ridge",),
        meal_proxy_mode="strict",
        ic_policy="exploratory_only",
    )



def _built_raw_tree(tmp_path: Path, workspace_root: Path, fixture_dir: Path) -> Path:
    """Copy a fixture raw tree into the test cloud folder and build its merged dataset."""
    raw_root = tmp_path / "cloud" / "data" / "raw"
    shutil.copytree(fixture_dir, raw_root)
    assert main(["--root", str(workspace_root), "build-dataset", "--raw", str(raw_root)]) == 0
    return raw_root


def test_validate_raw_command_writes_summary_report(tmp_path):
    workbook = tmp_path / "therapy_events_2023-02.xlsx"
    frame = pd.DataFrame(
        {
            "eventDateTime": pd.date_range("2023-02-04 09:00:00", periods=4, freq="5min"),
            "egv_estimatedGlucoseValue": [100, 101, 102, 103],
        }
    )
    with pd.ExcelWriter(workbook) as writer:
        frame.to_excel(writer, index=False)

    report = tmp_path / "validation.md"
    exit_code = main(["--root", str(tmp_path), "validate-raw", "--raw", str(tmp_path), "--report", str(report)])

    assert exit_code == 0
    text = report.read_text(encoding="utf-8")
    assert "dense_cgm_files: 1" in text
    assert "therapy_events_2023-02.xlsx" in text


class _FakeApiClient:
    last_instance = None

    def __init__(self, workspace, **kwargs) -> None:
        self.workspace = workspace
        self.kwargs = kwargs
        self.calls: list[tuple[str, str | None]] = []
        _FakeApiClient.last_instance = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def login(self, credentials, step_log=None):
        self.calls.append(("login", credentials.email))

    def start_trace(self):
        self.calls.append(("trace_start", None))

    def stop_trace(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("trace", encoding="utf-8")
        return path

    def capture_screenshot(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("shot", encoding="utf-8")
        return path

    def capture_page_diagnostics(self, stem: str):
        return None

    def export_daily_timeline_window(self, window: ExportWindow, download_dir: Path, step_log=None):
        self.calls.append(("export", window.window_id))
        window_dir = self.workspace.cloud_raw / "tconnectsync" / window.window_id / "normalized"
        window_dir.mkdir(parents=True, exist_ok=True)
        is_complete_window = not window.window_id.endswith("2024-02-29")
        pd.DataFrame(
            {
                "timestamp": pd.date_range(window.start_date, periods=2, freq="5min"),
                "glucose": [100, 101],
                "source_file": [f"tconnectsync::{window.window_id}::tconnectsync"] * 2,
                "source_label": ["tconnectsync"] * 2,
            }
        ).to_csv(window_dir / "cgm.csv", index=False)
        pd.DataFrame(
            {
                "window_id": [window.window_id],
                "kind": ["cgm"],
                "normalized_path": [str(window_dir / "cgm.csv")],
            }
        ).to_csv(self.workspace.cloud_raw / "tconnectsync" / window.window_id / "window_manifest.csv", index=False)
        return NormalizedWindowResult(
            window_id=window.window_id,
            requested_start=window.start_date.isoformat(),
            requested_end=window.end_date.isoformat(),
            endpoint_family="tconnectsync",
            source_label=f"tconnectsync::{window.window_id}::tconnectsync",
            raw_artifacts=(),
            normalized_paths={"cgm": str(window_dir / "cgm.csv")},
            row_counts={"cgm": 2},
            observed_first_timestamp=pd.Timestamp(window.start_date).isoformat(),
            observed_last_timestamp=pd.Timestamp(window.end_date if is_complete_window else window.start_date).isoformat(),
            observed_window_days=1 if is_complete_window else 10,
            is_complete_window=is_complete_window,
            activity_present=False,
            manifest_path=self.workspace.cloud_raw / "tconnectsync" / window.window_id / "window_manifest.csv",
        )


def test_collect_command_uses_api_client_by_default(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    monkeypatch.setenv("TANDEM_SOURCE_EMAIL", "me@example.com")
    monkeypatch.setenv("TANDEM_SOURCE_PASSWORD", "secret")
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(tmp_path / "cloud"))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr(cli, "TConnectSyncSourceClient", lambda paths, **kwargs: _FakeApiClient(paths, **kwargs))

    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "collect",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-01",
        ]
    )

    assert exit_code == 0
    client = _FakeApiClient.last_instance
    assert client is not None
    assert any(call[0] == "export" for call in client.calls)
    assert (client.workspace.cloud_raw / "tconnectsync").exists()
    assert (client.workspace.cloud_raw / "tconnectsync" / "2024-01-01__2024-01-01" / "window_manifest.csv").exists()
    assert (client.workspace.output_source / "tandem_acquisition_summary.md").exists()


def test_collect_command_writes_redacted_run_logs(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    monkeypatch.setenv("TANDEM_SOURCE_EMAIL", "me@example.com")
    monkeypatch.setenv("TANDEM_SOURCE_PASSWORD", "secret")
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(tmp_path / "cloud"))
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setattr(cli, "TConnectSyncSourceClient", lambda paths, **kwargs: _FakeApiClient(paths, **kwargs))

    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "collect",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-01",
        ]
    )

    assert exit_code == 0
    run_dir = _latest_log_dir(runtime_root, "collect")
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    event_names = {event["event"] for event in events}
    assert "login.start" in event_names
    assert "window.complete" in event_names
    login_start = next(event for event in events if event["event"] == "login.start")
    assert login_start["fields"]["email"] == "[REDACTED]"
    assert (run_dir / "run.log").exists()
    assert (run_dir / "run_meta.json").exists()


def test_backfill_command_continues_on_partial_window(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    monkeypatch.setenv("TANDEM_SOURCE_EMAIL", "me@example.com")
    monkeypatch.setenv("TANDEM_SOURCE_PASSWORD", "secret")
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(tmp_path / "cloud"))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr(cli, "TConnectSyncSourceClient", lambda paths, **kwargs: _FakeApiClient(paths, **kwargs))

    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "backfill",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-02-29",
            "--window-days",
            "30",
        ]
    )

    assert exit_code == 0
    client = _FakeApiClient.last_instance
    assert client is not None
    export_calls = [call for call in client.calls if call[0] == "export"]
    assert len(export_calls) == 2
    assert (client.workspace.cloud_raw / "tandem_export_manifest.csv").exists()
    assert (client.workspace.output_source / "tandem_acquisition_summary.md").exists()


def test_ingest_command_defaults_report_to_runtime_output(tmp_path, tandem_fixture_dir, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    cloud_root = tmp_path / "cloud"
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(cloud_root))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))

    raw_root = _built_raw_tree(tmp_path, workspace_root, tandem_fixture_dir)
    exit_code = main(["--root", str(workspace_root), "ingest", "--raw", str(raw_root)])

    assert exit_code == 0
    assert (runtime_root / "output" / "source" / "coverage.md").exists()
    assert (runtime_root / "output" / "source" / "coverage_review.html").exists()
    assert (cloud_root / "data" / "raw" / "tandem_export_manifest.csv").exists()
    coverage_text = (runtime_root / "output" / "source" / "coverage.md").read_text(encoding="utf-8")
    assert "[coverage_review_html](coverage_review.html)" in coverage_text


def test_normalize_raw_command_rebuilds_archived_window(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    cloud_root = tmp_path / "cloud"
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(cloud_root))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))

    raw_dir = cloud_root / "data" / "raw" / "tconnectsync" / "2024-05-01__2024-05-03" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for kind, payload in {
        "cgm": [{"eventDateTime": "2024-05-01T00:00:00-04:00", "egv_estimatedGlucoseValue": 110}],
        "bolus": [{"completionDateTime": "2024-05-01T08:00:00-04:00", "actualTotalBolusRequested": 2.5}],
        "basal": [{"startDateTime": "2024-05-01T00:00:00-04:00", "endDateTime": "2024-05-02 00:00:00", "basalRate": 0.7}],
        "activity": [],
    }.items():
        (raw_dir / f"{kind}.json").write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(["--root", str(workspace_root), "normalize-raw"])

    assert exit_code == 0
    assert (cloud_root / "data" / "raw" / "tconnectsync" / "2024-05-01__2024-05-03" / "window_manifest.csv").exists()
    assert (runtime_root / "output" / "source" / "normalize_raw_summary.md").exists()


def test_build_dataset_without_apple_writes_tandem_only_dataset_and_skips_unchanged_inputs(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    cloud_root = tmp_path / "cloud"
    runtime_root = tmp_path / "runtime"
    raw_root = cloud_root / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    _write_tandem_cgm_export(raw_root)
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(cloud_root))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))

    assert main(["--root", str(workspace_root), "build-dataset", "--raw", str(raw_root)]) == 0

    merged = cloud_root / "data" / "merged"
    prepared = pd.read_parquet(merged / "prepared_model_data_5min.parquet")
    assert "health_activity_value" not in prepared.columns
    text = (runtime_root / "output" / "prepare" / "model_data_preparation.md").read_text(encoding="utf-8")
    assert "- mode: tandem_only" in text
    assert "- apple_available: False" in text
    manifest = json.loads((merged / "dataset.json").read_text(encoding="utf-8"))
    built_at = manifest["built_at"]
    run_dir = _latest_log_dir(runtime_root, "build-dataset")
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["event"] == "command.stage.start" and event["stage"] == "dataset.build" for event in events)

    # Unchanged inputs: nothing is rebuilt.
    assert main(["--root", str(workspace_root), "build-dataset", "--raw", str(raw_root)]) == 0
    assert json.loads((merged / "dataset.json").read_text(encoding="utf-8"))["built_at"] == built_at

    # A new raw file makes analyses refuse the old dataset until it is rebuilt.
    (raw_root / "carbs.csv").write_text("timestamp,carb_grams\n2025-05-24 08:00:00,30\n", encoding="utf-8")
    from bayesian_t1dm.dataset import StaleDatasetError, load_merged_dataset

    try:
        load_merged_dataset(raw_root)
    except StaleDatasetError as exc:
        assert "1 added" in str(exc)
    else:
        raise AssertionError("expected a stale-dataset error after adding a raw file")

def test_validate_therapy_infra_command_writes_artifacts(tmp_path):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    report_dir = tmp_path / "reports"

    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "validate-therapy-infra",
            "--report-dir",
            str(report_dir),
            "--include-models",
            "ridge,segmented_ridge,tree_boost",
        ]
    )

    assert exit_code == 0
    assert (report_dir / "therapy_infra_validation.md").exists()
    assert (report_dir / "therapy_synthetic_results.csv").exists()
    assert (report_dir / "therapy_synthetic_recommendation_audit.md").exists()


def test_validate_latent_meal_command_writes_artifacts(tmp_path, capsys):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    report_dir = tmp_path / "latent_validation"

    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "validate-latent-meal",
            "--report-dir",
            str(report_dir),
            "--meals",
            "25",
        ]
    )

    assert exit_code == 0
    assert "pass" in capsys.readouterr().out
    report = (report_dir / "latent_meal_truth_recovery.md").read_text(encoding="utf-8")
    assert "everything_at_once" in report
    assert (report_dir / "latent_meal_truth_recovery_summary.csv").exists()
    assert (report_dir / "latent_meal_truth_recovery_meals.csv").exists()


def test_review_therapy_evidence_command_writes_report_and_supporting_artifacts(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    dataset = _synthetic_base_dataset(apple=True, explicit_carbs=True)
    preparation = ModelDataPreparationResult(
        dataset=dataset,
        apple_available=True,
        apple_span_start=pd.Timestamp("2025-01-01 00:00:00"),
        apple_span_end=pd.Timestamp("2025-01-07 23:55:00"),
        tandem_span_before_start=pd.Timestamp("2025-01-01 00:00:00"),
        tandem_span_before_end=pd.Timestamp("2025-01-07 23:55:00"),
        tandem_span_after_start=pd.Timestamp("2025-01-01 00:00:00"),
        tandem_span_after_end=pd.Timestamp("2025-01-07 23:55:00"),
        requested_tandem_start=pd.Timestamp("2025-01-01 00:00:00"),
        requested_tandem_end=pd.Timestamp("2025-01-07 23:55:00"),
        overlap_start=pd.Timestamp("2025-01-01 00:00:00"),
        overlap_end=pd.Timestamp("2025-01-07 23:55:00"),
        final_dataset_start=pd.Timestamp("2025-01-01 00:00:00"),
        final_dataset_end=pd.Timestamp("2025-01-07 23:55:00"),
        final_row_count=len(dataset.frame),
    )
    monkeypatch.setattr(cli, "_load_dataset", lambda args, session=None: _fake_merged(preparation))

    report = tmp_path / "therapy_evidence_review.html"
    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "review-therapy-evidence",
            "--report",
            str(report),
            "--include-models",
            "ridge,segmented_ridge,tree_boost",
        ]
    )

    assert exit_code == 0
    assert report.exists()
    assert (tmp_path / "model_data_preparation.md").exists()
    assert (tmp_path / "therapy_research_gate.md").exists()
    assert (tmp_path / "therapy_infra_validation.md").exists()


def test_research_latent_meal_icr_command_writes_artifacts_and_review(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    dataset = _synthetic_base_dataset(apple=True, explicit_carbs=True)
    preparation = ModelDataPreparationResult(
        dataset=dataset,
        apple_available=True,
        apple_span_start=pd.Timestamp("2025-01-01 00:00:00"),
        apple_span_end=pd.Timestamp("2025-01-07 23:55:00"),
        tandem_span_before_start=pd.Timestamp("2025-01-01 00:00:00"),
        tandem_span_before_end=pd.Timestamp("2025-01-07 23:55:00"),
        tandem_span_after_start=pd.Timestamp("2025-01-01 00:00:00"),
        tandem_span_after_end=pd.Timestamp("2025-01-07 23:55:00"),
        requested_tandem_start=pd.Timestamp("2025-01-01 00:00:00"),
        requested_tandem_end=pd.Timestamp("2025-01-07 23:55:00"),
        overlap_start=pd.Timestamp("2025-01-01 00:00:00"),
        overlap_end=pd.Timestamp("2025-01-07 23:55:00"),
        final_dataset_start=pd.Timestamp("2025-01-01 00:00:00"),
        final_dataset_end=pd.Timestamp("2025-01-07 23:55:00"),
        final_row_count=len(dataset.frame),
    )
    monkeypatch.setattr(cli, "_load_dataset", lambda args, session=None: _fake_merged(preparation))

    report_dir = tmp_path / "latent_meal"
    review_html = report_dir / "latent_meal_review.html"
    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "research-latent-meal-icr",
            "--report-dir",
            str(report_dir),
            "--review-html",
            str(review_html),
        ]
    )

    assert exit_code == 0
    assert (report_dir / "latent_meal_research_gate.md").exists()
    assert (report_dir / "meal_truth_semantics_report.md").exists()
    assert (report_dir / "meal_event_registry.csv").exists()
    assert (report_dir / "first_meal_clean_window_registry.csv").exists()
    assert (report_dir / "first_meal_clean_window_audit.md").exists()
    assert (report_dir / "first_meal_exclusion_summary.csv").exists()
    assert review_html.exists()


def test_research_latent_meal_icr_full_scope_writes_outputs(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    report_dir = tmp_path / "latent_meal_full"
    review_html = report_dir / "latent_meal_review.html"
    raw_root = tmp_path / "raw"
    metadata_dir = raw_root / "tconnectsync" / "2026-03-02__2026-03-31" / "raw"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "pump_event_metadata.json").write_text(
        json.dumps(
            {
                "requested_end": "2026-03-31",
                "selected_pump": {
                    "lastUpload": {
                        "lastUploadedAt": "2026-04-01T02:15:16.6",
                        "settings": {
                            "profiles": {
                                "activeIdp": 0,
                                "profile": [
                                    {
                                        "idp": 0,
                                        "name": "Main",
                                        "insulinDuration": 300,
                                        "maxBolus": 25000,
                                        "tDependentSegs": [
                                            {"startTime": 0, "carbRatio": 7000, "isf": 60, "targetBg": 110},
                                            {"startTime": 900, "carbRatio": 6800, "isf": 60, "targetBg": 110},
                                        ],
                                    }
                                ],
                            },
                            "pumpSettings": {"maxBolus": 25000},
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    dataset = _synthetic_base_dataset(apple=True, explicit_carbs=True)
    preparation = ModelDataPreparationResult(
        dataset=dataset,
        apple_available=True,
        apple_span_start=pd.Timestamp("2025-01-01 00:00:00"),
        apple_span_end=pd.Timestamp("2025-01-07 23:55:00"),
        tandem_span_before_start=pd.Timestamp("2025-01-01 00:00:00"),
        tandem_span_before_end=pd.Timestamp("2025-01-07 23:55:00"),
        tandem_span_after_start=pd.Timestamp("2025-01-01 00:00:00"),
        tandem_span_after_end=pd.Timestamp("2025-01-07 23:55:00"),
        requested_tandem_start=pd.Timestamp("2025-01-01 00:00:00"),
        requested_tandem_end=pd.Timestamp("2025-01-07 23:55:00"),
        overlap_start=pd.Timestamp("2025-01-01 00:00:00"),
        overlap_end=pd.Timestamp("2025-01-07 23:55:00"),
        final_dataset_start=pd.Timestamp("2025-01-01 00:00:00"),
        final_dataset_end=pd.Timestamp("2025-01-07 23:55:00"),
        final_row_count=len(dataset.frame),
    )
    monkeypatch.setattr(cli, "_load_dataset", lambda args, session=None: _fake_merged(preparation))

    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "research-latent-meal-icr",
            "--raw",
            str(raw_root),
            "--research-scope",
            "full",
            "--report-dir",
            str(report_dir),
            "--review-html",
            str(review_html),
        ]
    )

    assert exit_code == 0
    assert (report_dir / "latent_meal_research_gate.md").exists()
    assert (report_dir / "meal_event_registry.csv").exists()
    assert (report_dir / "meal_window_audit.md").exists()
    assert (report_dir / "latent_meal_window_audit.csv").exists()
    assert (report_dir / "latent_carb_estimator_report.md").exists()
    assert (report_dir / "latent_carb_estimates.csv").exists()
    assert (report_dir / "latent_meal_fit_summary.md").exists()
    assert (report_dir / "latent_meal_posterior_meals.csv").exists()
    assert (report_dir / "latent_meal_confidence_report.md").exists()
    assert (report_dir / "latent_meal_model_comparison.md").exists()
    cache_dir = cli.ProjectPaths.from_root(workspace_root).cache_latent_meal
    assert (cache_dir / "latent_carb_estimates.csv").exists()
    assert (cache_dir / "therapy_settings_used.json").exists()
    assert (cache_dir / "latent_carb_posterior_samples.parquet").exists()
    assert review_html.exists()


def test_build_latent_meal_fixture_command_writes_fixture_bundle(tmp_path):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    prepared_csv = tmp_path / "prepared_model_data_5min.csv"
    output_dir = tmp_path / "latent_meal_fixture"
    review_html = output_dir / "latent_meal_review.html"
    dataset = _synthetic_base_dataset(apple=True, explicit_carbs=True)
    dataset.frame.to_csv(prepared_csv, index=False)

    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "build-latent-meal-fixture",
            "--prepared-csv",
            str(prepared_csv),
            "--output-dir",
            str(output_dir),
            "--review-html",
            str(review_html),
            "--background-days",
            "2",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "prepared_model_data_5min.csv").exists()
    assert (output_dir / "latent_meal_fixture_summary.md").exists()
    assert (output_dir / "latent_meal_fixture_day_manifest.csv").exists()
    assert (output_dir / "latent_meal_research_gate.md").exists()
    assert (output_dir / "meal_truth_semantics_report.md").exists()
    assert (output_dir / "first_meal_clean_window_registry.csv").exists()
    assert review_html.exists()


def test_status_command_writes_curated_latest_outputs_and_cache_artifacts(tmp_path, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    cloud_root = tmp_path / "cloud"
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(cloud_root))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))

    dataset = _synthetic_base_dataset(apple=True, explicit_carbs=True)
    preparation = ModelDataPreparationResult(
        dataset=dataset,
        apple_available=True,
        apple_span_start=pd.Timestamp("2025-01-01 00:00:00"),
        apple_span_end=pd.Timestamp("2025-01-07 23:55:00"),
        tandem_span_before_start=pd.Timestamp("2025-01-01 00:00:00"),
        tandem_span_before_end=pd.Timestamp("2025-01-07 23:55:00"),
        tandem_span_after_start=pd.Timestamp("2025-01-01 00:00:00"),
        tandem_span_after_end=pd.Timestamp("2025-01-07 23:55:00"),
        requested_tandem_start=pd.Timestamp("2025-01-01 00:00:00"),
        requested_tandem_end=pd.Timestamp("2025-01-07 23:55:00"),
        overlap_start=pd.Timestamp("2025-01-01 00:00:00"),
        overlap_end=pd.Timestamp("2025-01-07 23:55:00"),
        final_dataset_start=pd.Timestamp("2025-01-01 00:00:00"),
        final_dataset_end=pd.Timestamp("2025-01-07 23:55:00"),
        final_row_count=len(dataset.frame),
    )
    monkeypatch.setattr(cli, "_load_dataset", lambda args, session=None: _fake_merged(preparation))
    monkeypatch.setattr(cli, "run_therapy_research", lambda *args, **kwargs: _fake_status_research_result())
    monkeypatch.setattr(
        cli,
        "validate_therapy_infra",
        lambda *args, **kwargs: TherapyInfraValidationResult(
            scenario_results=pd.DataFrame([{"scenario": "x", "passed": True}]),
            report_markdown="# validation\n",
            recommendation_audit_markdown="# audit\n",
        ),
    )
    output_root = runtime_root / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "old_artifact.md").write_text("old", encoding="utf-8")

    exit_code = main(
        [
            "--root",
            str(workspace_root),
            "status",
            "--include-models",
            "ridge",
        ]
    )

    assert exit_code == 0
    current_payload = json.loads((runtime_root / "cache" / "status" / "current_status.json").read_text(encoding="utf-8"))
    assert current_payload["overall_state"] == "blocked"
    assert (output_root / "current_status.html").exists()
    assert (output_root / "therapy_review.html").exists()
    assert not (output_root / "old_artifact.md").exists()
    assert (output_root / "therapy" / "therapy_review.html").exists()
    assert (output_root / "therapy" / "model_data_preparation.md").exists()
    assert _latest_log_dir(runtime_root, "status").joinpath("events.jsonl").exists()


def test_ingest_failure_is_logged(tmp_path, tandem_fixture_dir, monkeypatch):
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("BAYESIAN_T1DM_CLOUD_ROOT", str(tmp_path / "cloud"))
    monkeypatch.setenv("BAYESIAN_T1DM_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setattr(cli, "load_merged_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    try:
        main(
            [
                "--root",
                str(workspace_root),
                "ingest",
                "--raw",
                str(tandem_fixture_dir),
            ]
        )
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected ingest to fail")

    run_dir = _latest_log_dir(runtime_root, "ingest")
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["event"] == "command.error" and event["fields"]["error_type"] == "RuntimeError" for event in events)
    assert any(event["event"] == "command.complete" and event["status"] == "failed" for event in events)
