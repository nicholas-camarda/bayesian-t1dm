from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import date, timedelta
from pathlib import Path
import sys

import pandas as pd

import warnings

from .evaluate import run_walk_forward
from .acquisition import (
    AcquisitionError,
    ExportWindow,
    backfill_tandem_exports,
    collect_tandem_exports,
    load_tandem_credentials,
    normalize_tconnectsync_archive,
    StepLogger,
    TConnectSyncSourceClient,
)
from .features import FeatureConfig
from .health_auto_export import (
    build_prepared_model_dataset,
    has_apple_health_data,
    import_health_auto_export_batch,
    summarize_apple_health_span,
    summarize_tandem_data_span,
    screen_health_features,
    write_health_screening_report,
    write_model_data_preparation_report,
    ModelDataPreparationResult,
    intersect_spans,
)
from .ingest import build_export_manifest, load_tandem_exports, summarize_coverage, summarize_tandem_raw_dir, write_export_manifest
from .model import BayesianGlucoseModel
from .observability import LoggingSession, setup_run_logging
from .paths import ProjectPaths
from .quality import assess_data_quality
from .recommend import RecommendationPolicy, recommend_setting_changes
from .report import build_run_summary, write_json_report, write_markdown_report
from .review import write_coverage_review_html, write_current_status_html, write_run_review_html, write_therapy_evidence_review_html
from .features import FeatureFrame
from .status import cleanup_legacy_top_level_output, create_status_bundle, derive_current_status, finalize_status_logs, publish_latest_entrypoints, write_status_json
from .therapy_research import (
    parse_model_list,
    parse_therapy_segments,
    run_therapy_research,
    validate_therapy_infra,
    write_therapy_infra_validation_artifacts,
    write_therapy_research_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    logging_parent = argparse.ArgumentParser(add_help=False)
    logging_parent.add_argument("--log-level", choices=["ERROR", "WARNING", "INFO", "DEBUG"], default="INFO")
    logging_parent.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)
    logging_parent.add_argument("--unsafe-debug-logging", action=argparse.BooleanOptionalAction, default=False)

    parser = argparse.ArgumentParser(prog="bayesian-t1dm", parents=[logging_parent])
    parser.add_argument("--root", default=".", help="Project root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_command(name: str, help_text: str) -> argparse.ArgumentParser:
        return subparsers.add_parser(name, help=help_text, parents=[logging_parent])

    ingest = add_command("ingest", "Load Tandem exports and write a coverage report")
    ingest.add_argument("--raw", default=None)
    ingest.add_argument("--report", default=None)
    ingest.add_argument("--manifest", default=None)

    normalize_raw = add_command("normalize-raw", "Rebuild normalized tconnectsync windows from archived raw payloads")
    normalize_raw.add_argument("--raw", default=None)
    normalize_raw.add_argument("--window-id", default=None)
    normalize_raw.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    normalize_raw.add_argument("--report", default=None)
    normalize_raw.add_argument("--manifest", default=None)

    validate_raw = add_command("validate-raw", "Classify Tandem raw files as dense CGM or summary-only")
    validate_raw.add_argument("--raw", default=None)
    validate_raw.add_argument("--report", default=None)

    import_health = add_command("import-health-auto-export", "Archive and normalize a Health Auto Export JSON bundle")
    import_health.add_argument("--input", required=True, help="Path to the exported Health Auto Export directory")

    prepare_model = add_command("prepare-model-data", "Prepare the best available model dataset with optional Apple Health enrichment")
    prepare_model.add_argument("--raw", default=None)
    prepare_model.add_argument("--apple-input", default=None, help="Optional Health Auto Export parent directory")
    prepare_model.add_argument("--output", default=None)
    prepare_model.add_argument("--report", default=None)
    prepare_model.add_argument("--horizon", type=int, default=30)
    prepare_model.add_argument("--history-days", type=int, default=365)
    prepare_model.add_argument("--min-history-days", type=int, default=180)
    prepare_model.add_argument("--skip-backfill", action=argparse.BooleanOptionalAction, default=False)
    prepare_model.add_argument("--env-file", default=None, help="Optional .env file with Tandem credentials")

    analysis_ready = add_command("build-health-analysis-ready", "Build a Tandem-aligned 5-minute analysis-ready dataset with unified Apple Health context")
    analysis_ready.add_argument("--raw", default=None)
    analysis_ready.add_argument("--output", default=None)
    analysis_ready.add_argument("--horizon", type=int, default=30)

    screen_health = add_command("screen-health-features", "Screen imported Apple Health context features against Tandem glucose targets")
    screen_health.add_argument("--raw", default=None)
    screen_health.add_argument("--report", default=None)
    screen_health.add_argument("--scores", default=None)
    screen_health.add_argument("--horizon", type=int, default=30)

    research_therapy = add_command("research-therapy-settings", "Run research-grade therapy setting analysis on prepared Tandem and Apple Health data")
    research_therapy.add_argument("--raw", default=None)
    research_therapy.add_argument("--apple-input", default=None, help="Optional Health Auto Export parent directory")
    research_therapy.add_argument("--horizon", type=int, default=30)
    research_therapy.add_argument("--segments", default=None, help="Comma-separated day segments like overnight=00:00-06:00,morning=06:00-11:00")
    research_therapy.add_argument("--include-models", default=None, help="Comma-separated model families such as bayesian,ridge,elastic_net,segmented_ridge,tree_boost,ensemble")
    research_therapy.add_argument("--skip-backfill", action=argparse.BooleanOptionalAction, default=False)
    research_therapy.add_argument("--report-dir", default=None)
    research_therapy.add_argument("--env-file", default=None, help="Optional .env file with Tandem credentials")
    research_therapy.add_argument("--meal-proxy-mode", choices=["strict", "broad", "off"], default="strict")
    research_therapy.add_argument("--ic-policy", choices=["exploratory_only", "conservative", "off"], default="exploratory_only")
    research_therapy.add_argument("--write-source-report-cards", action=argparse.BooleanOptionalAction, default=True)
    research_therapy.add_argument("--write-research-gate", action=argparse.BooleanOptionalAction, default=True)

    validate_therapy = add_command("validate-therapy-infra", "Run synthetic truth-recovery validation for the therapy research infrastructure")
    validate_therapy.add_argument("--report-dir", default=None)
    validate_therapy.add_argument("--include-models", default=None, help="Comma-separated model families such as ridge,segmented_ridge,tree_boost,ensemble")
    validate_therapy.add_argument("--meal-proxy-mode", choices=["strict", "broad", "off"], default="strict")
    validate_therapy.add_argument("--ic-policy", choices=["exploratory_only", "conservative", "off"], default="exploratory_only")

    review_therapy = add_command("review-therapy-evidence", "Build a therapy evidence report focused on identifiability and overnight basal evidence")
    review_therapy.add_argument("--raw", default=None)
    review_therapy.add_argument("--apple-input", default=None, help="Optional Health Auto Export parent directory")
    review_therapy.add_argument("--report", default=None)
    review_therapy.add_argument("--horizon", type=int, default=30)
    review_therapy.add_argument("--segments", default=None)
    review_therapy.add_argument("--include-models", default=None)
    review_therapy.add_argument("--skip-backfill", action=argparse.BooleanOptionalAction, default=False)
    review_therapy.add_argument("--env-file", default=None, help="Optional .env file with Tandem credentials")
    review_therapy.add_argument("--meal-proxy-mode", choices=["strict", "broad", "off"], default="strict")
    review_therapy.add_argument("--ic-policy", choices=["exploratory_only", "conservative", "off"], default="exploratory_only")

    status = add_command("status", "Run the primary end-to-end status workflow and publish the latest dashboards")
    status.add_argument("--raw", default=None)
    status.add_argument("--apple-input", default=None, help="Optional Health Auto Export parent directory")
    status.add_argument("--horizon", type=int, default=30)
    status.add_argument("--history-days", type=int, default=365)
    status.add_argument("--min-history-days", type=int, default=180)
    status.add_argument("--segments", default=None)
    status.add_argument("--include-models", default=None)
    status.add_argument("--skip-backfill", action=argparse.BooleanOptionalAction, default=False)
    status.add_argument("--env-file", default=None, help="Optional .env file with Tandem credentials")
    status.add_argument("--meal-proxy-mode", choices=["strict", "broad", "off"], default="strict")
    status.add_argument("--ic-policy", choices=["exploratory_only", "conservative", "off"], default="exploratory_only")
    status.add_argument("--eval-folds", type=int, default=4)
    status.add_argument("--draws", type=int, default=1000)
    status.add_argument("--tune", type=int, default=1000)
    status.add_argument("--chains", type=int, default=2)
    status.add_argument("--target-accept", type=float, default=0.95)
    status.add_argument("--max-treedepth", type=int, default=12)

    run = add_command("run", "Run the full forecasting and recommendation pipeline")
    run.add_argument("--raw", default=None)
    run.add_argument("--report", default=None)
    run.add_argument("--manifest", default=None)
    run.add_argument("--horizon", type=int, default=30)
    run.add_argument("--eval-folds", type=int, default=4)
    run.add_argument("--draws", type=int, default=1000)
    run.add_argument("--tune", type=int, default=1000)
    run.add_argument("--chains", type=int, default=2)
    run.add_argument("--target-accept", type=float, default=0.95)
    run.add_argument("--max-treedepth", type=int, default=12)
    run.add_argument("--skip-recommendations", action=argparse.BooleanOptionalAction, default=False)

    collect = add_command("collect", "Fetch one Tandem Source window through tconnectsync")
    collect.add_argument("--start-date", default=None, help="Requested window start date (YYYY-MM-DD)")
    collect.add_argument("--end-date", default=None, help="Requested window end date (YYYY-MM-DD)")
    collect.add_argument("--manifest", default=None)
    collect.add_argument("--report", default=None)
    collect.add_argument("--env-file", default=None, help="Optional .env file with Tandem credentials")
    collect.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)

    backfill = add_command("backfill", "Backfill Tandem Source windows through tconnectsync")
    backfill.add_argument("--start-date", required=True, help="Earliest requested date (YYYY-MM-DD)")
    backfill.add_argument("--end-date", required=True, help="Latest requested date (YYYY-MM-DD)")
    backfill.add_argument("--window-days", type=int, default=30)
    backfill.add_argument("--direction", choices=["backward", "forward"], default="backward")
    backfill.add_argument("--manifest", default=None)
    backfill.add_argument("--report", default=None)
    backfill.add_argument("--env-file", default=None, help="Optional .env file with Tandem credentials")
    backfill.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    backfill.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    return parser


def _coerce_date_bounds(
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if start is None or end is None:
        return None, None
    return pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()


def _build_feature_frame_from_prepared(dataset) -> FeatureFrame:
    return FeatureFrame(
        frame=dataset.frame,
        feature_columns=list(dataset.feature_columns),
        target_column=dataset.target_column,
        horizon_minutes=dataset.horizon_minutes,
        config=dataset.config,
    )


def _session_stage(session: LoggingSession | None, stage: str, **fields: object):
    return session.stage(stage, **fields) if session is not None else nullcontext()


def _prepare_model_data(
    *,
    args,
    paths: ProjectPaths,
    session: LoggingSession | None = None,
) -> ModelDataPreparationResult:
    warnings_list: list[str] = []
    raw_root = Path(args.raw)
    history_days = int(getattr(args, "history_days", 365))
    min_history_days = int(getattr(args, "min_history_days", 180))

    if getattr(args, "apple_input", None):
        with _session_stage(session, "prepare_model_data.apple_import", apple_input=str(args.apple_input)):
            try:
                imported = import_health_auto_export_batch(args.apple_input, paths.ensure())
                if session is not None:
                    session.log_event(
                        "prepare_model_data.apple_import.complete",
                        stage="prepare_model_data.apple_import",
                        imported_bundle_count=len(imported),
                    )
            except ValueError as exc:
                warnings_list.append(f"Apple Health import skipped: {exc}")
                if session is not None:
                    session.log_event(
                        "prepare_model_data.apple_import.unavailable",
                        level="WARNING",
                        message=str(exc),
                        stage="prepare_model_data.apple_import",
                    )
    elif session is not None:
        session.log_event(
            "prepare_model_data.apple_import.reused_existing",
            message="No new Apple Health import requested; reusing any previously imported Apple Health tables.",
            stage="prepare_model_data.apple_import",
        )

    with _session_stage(session, "prepare_model_data.load_inputs", raw_root=str(raw_root)):
        if session is not None:
            session.log_event(
                "prepare_model_data.tandem_inputs.loading",
                stage="prepare_model_data.load_inputs",
                message="Loading Tandem normalized inputs.",
            )
        tandem_before = load_tandem_exports(raw_root, include_health_auto_export=False)
        if session is not None:
            session.log_event(
                "prepare_model_data.tandem_inputs.loaded",
                stage="prepare_model_data.load_inputs",
                tandem_cgm_rows=int(len(tandem_before.cgm)),
                tandem_bolus_rows=int(len(tandem_before.bolus)),
            )
            session.log_event(
                "prepare_model_data.health_inputs.loading",
                stage="prepare_model_data.load_inputs",
                message="Loading Tandem plus already-imported Apple Health inputs.",
            )
        health_before = load_tandem_exports(raw_root, include_health_auto_export=True)
        tandem_span_before_start, tandem_span_before_end = summarize_tandem_data_span(tandem_before)
        apple_available = has_apple_health_data(health_before)
        apple_span_start, apple_span_end = summarize_apple_health_span(health_before)
    if session is not None:
        if apple_available:
            session.log_event(
                "prepare_model_data.apple_health.detected",
                message="Apple Health data detected in the canonical raw tree.",
                stage="prepare_model_data.load_inputs",
                apple_span_start=apple_span_start,
                apple_span_end=apple_span_end,
            )
        else:
            session.log_event(
                "prepare_model_data.apple_health.not_detected",
                message="No Apple Health data detected in the canonical raw tree; proceeding Tandem-only.",
                stage="prepare_model_data.load_inputs",
            )
        session.log_event(
            "prepare_model_data.input_summary",
            stage="prepare_model_data.load_inputs",
            tandem_cgm_rows=int(len(tandem_before.cgm)),
            tandem_bolus_rows=int(len(tandem_before.bolus)),
            apple_available=apple_available,
            health_activity_rows=int(len(health_before.health_activity)),
            health_measurement_rows=int(len(health_before.health_measurements)),
            sleep_rows=int(len(health_before.sleep)),
            workout_rows=int(len(health_before.workouts)),
        )

    requested_tandem_start: pd.Timestamp | None = None
    requested_tandem_end: pd.Timestamp | None = None
    if apple_available and apple_span_start is not None and apple_span_end is not None:
        requested_tandem_start, requested_tandem_end = _coerce_date_bounds(apple_span_start, apple_span_end)
    else:
        anchor_end = (
            pd.Timestamp(tandem_span_before_end).normalize()
            if tandem_span_before_end is not None
            else pd.Timestamp(date.today())
        )
        requested_tandem_end = anchor_end
        requested_tandem_start = anchor_end - pd.Timedelta(days=max(history_days, 1) - 1)

    backfill_status = "not_needed"
    needs_backfill = (
        requested_tandem_start is not None
        and requested_tandem_end is not None
        and (
            tandem_span_before_start is None
            or tandem_span_before_end is None
            or pd.Timestamp(tandem_span_before_start).normalize() > requested_tandem_start
            or pd.Timestamp(tandem_span_before_end).normalize() < requested_tandem_end
        )
    )
    if session is not None:
        session.log_event(
            "prepare_model_data.backfill_decision",
            stage="prepare_model_data.backfill_decision",
            needs_backfill=needs_backfill,
            requested_tandem_start=requested_tandem_start,
            requested_tandem_end=requested_tandem_end,
            tandem_span_before_start=tandem_span_before_start,
            tandem_span_before_end=tandem_span_before_end,
            apple_available=apple_available,
        )

    if needs_backfill and getattr(args, "skip_backfill", False):
        backfill_status = "skipped_by_flag"
        warnings_list.append("Requested Tandem backfill was skipped by flag; using currently available Tandem history.")
        if session is not None:
            session.log_event(
                "prepare_model_data.backfill.skipped",
                level="WARNING",
                message="Requested Tandem backfill skipped by flag.",
                stage="prepare_model_data.backfill",
                backfill_status=backfill_status,
            )
    elif needs_backfill:
        with _session_stage(
            session,
            "prepare_model_data.backfill",
            requested_tandem_start=requested_tandem_start,
            requested_tandem_end=requested_tandem_end,
        ):
            try:
                credentials = load_tandem_credentials(args.root, getattr(args, "env_file", None))
                client_kwargs = {
                    "region": credentials.region,
                    "timezone": credentials.timezone,
                    "pump_serial": credentials.pump_serial,
                }
                client_cm = TConnectSyncSourceClient(paths, **client_kwargs)
                with client_cm as client:
                    backfill_tandem_exports(
                        client,
                        start_date=requested_tandem_start.date().isoformat(),
                        end_date=requested_tandem_end.date().isoformat(),
                        workspace=paths,
                        credentials=credentials,
                        window_days=30,
                        direction="backward",
                        manifest_path=str(paths.cloud_raw / "tandem_export_manifest.csv"),
                        report_path=str(paths.reports / "tandem_acquisition_summary.md"),
                        resume=True,
                        strict=False,
                        step_log=StepLogger(session=session) if session is not None else None,
                    )
                backfill_status = "completed"
            except AcquisitionError as exc:
                backfill_status = "unavailable"
                warnings_list.append(f"Tandem backfill unavailable: {exc}")
            except Exception as exc:
                backfill_status = "failed"
                warnings_list.append(f"Tandem backfill failed: {exc}")
        if session is not None:
            session.log_event(
                "prepare_model_data.backfill.complete",
                stage="prepare_model_data.backfill",
                backfill_status=backfill_status,
            )
    elif session is not None:
        session.log_event(
            "prepare_model_data.backfill.skipped",
            message="Backfill not needed.",
            stage="prepare_model_data.backfill",
            backfill_status=backfill_status,
        )

    with _session_stage(session, "prepare_model_data.reload_inputs", raw_root=str(raw_root)):
        if session is not None:
            session.log_event(
                "prepare_model_data.tandem_inputs.reloading",
                stage="prepare_model_data.reload_inputs",
                message="Reloading Tandem inputs after backfill decision.",
            )
        tandem_after = load_tandem_exports(raw_root, include_health_auto_export=False)
        if session is not None:
            session.log_event(
                "prepare_model_data.health_inputs.reloading",
                stage="prepare_model_data.reload_inputs",
                message="Reloading Tandem plus Apple Health inputs after backfill decision.",
            )
        health_after = load_tandem_exports(raw_root, include_health_auto_export=True)
        export_manifest = build_export_manifest(health_after)
        write_export_manifest(export_manifest, paths.cloud_raw / "tandem_export_manifest.csv")

    with _session_stage(session, "prepare_model_data.build_dataset", horizon_minutes=args.horizon):
        dataset = build_prepared_model_dataset(
            tandem_data=tandem_after,
            health_data=health_after,
            config=FeatureConfig(horizon_minutes=args.horizon),
        )
        tandem_span_after_start, tandem_span_after_end = summarize_tandem_data_span(tandem_after)
        apple_span_start, apple_span_end = summarize_apple_health_span(health_after)
    overlap_start, overlap_end = intersect_spans(
        apple_span_start,
        apple_span_end,
        tandem_span_after_start,
        tandem_span_after_end,
    )
    final_dataset_start = pd.to_datetime(dataset.frame["timestamp"], errors="coerce").min() if not dataset.frame.empty else None
    final_dataset_end = pd.to_datetime(dataset.frame["timestamp"], errors="coerce").max() if not dataset.frame.empty else None
    if final_dataset_start is not None and pd.notna(final_dataset_start) and final_dataset_end is not None and pd.notna(final_dataset_end):
        actual_history_days = int((pd.Timestamp(final_dataset_end).normalize() - pd.Timestamp(final_dataset_start).normalize()).days + 1)
        if actual_history_days < min_history_days:
            warnings_list.append(
                f"Final dataset history is only {actual_history_days} days, below the requested minimum of {min_history_days} days."
            )
    elif dataset.frame.empty:
        warnings_list.append("Final prepared dataset is empty.")
    if session is not None:
        session.log_event(
            "prepare_model_data.dataset_summary",
            stage="prepare_model_data.build_dataset",
            final_row_count=int(len(dataset.frame)),
            final_dataset_start=None if pd.isna(final_dataset_start) else pd.Timestamp(final_dataset_start),
            final_dataset_end=None if pd.isna(final_dataset_end) else pd.Timestamp(final_dataset_end),
            backfill_status=backfill_status,
            apple_available=dataset.apple_available,
        )

    return ModelDataPreparationResult(
        dataset=dataset,
        apple_available=dataset.apple_available,
        apple_span_start=apple_span_start,
        apple_span_end=apple_span_end,
        tandem_span_before_start=tandem_span_before_start,
        tandem_span_before_end=tandem_span_before_end,
        tandem_span_after_start=tandem_span_after_start,
        tandem_span_after_end=tandem_span_after_end,
        requested_tandem_start=requested_tandem_start,
        requested_tandem_end=requested_tandem_end,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        final_dataset_start=None if pd.isna(final_dataset_start) else pd.Timestamp(final_dataset_start),
        final_dataset_end=None if pd.isna(final_dataset_end) else pd.Timestamp(final_dataset_end),
        final_row_count=int(len(dataset.frame)),
        backfill_status=backfill_status,
        warnings=tuple(warnings_list),
    )


def _run_therapy_analysis(
    *,
    args,
    paths: ProjectPaths,
    session: LoggingSession | None = None,
    preparation_stage: str = "therapy.prepare_data",
    research_stage: str = "therapy.research",
    validation_stage: str = "therapy.validation",
    include_validation: bool = True,
):
    with _session_stage(session, preparation_stage):
        preparation = _prepare_model_data(args=args, paths=paths, session=session)
    with _session_stage(session, research_stage):
        research_result = run_therapy_research(
            preparation.dataset,
            segments=parse_therapy_segments(args.segments),
            include_models=parse_model_list(args.include_models),
            meal_proxy_mode=args.meal_proxy_mode,
            ic_policy=args.ic_policy,
        )
    validation_result = None
    if include_validation:
        with _session_stage(session, validation_stage):
            validation_result = validate_therapy_infra(
                meal_proxy_mode=args.meal_proxy_mode,
                ic_policy=args.ic_policy,
                include_models=parse_model_list("ridge,segmented_ridge,tree_boost,ensemble"),
            )
    return preparation, research_result, validation_result


def _write_therapy_outputs(
    *,
    preparation: ModelDataPreparationResult,
    research_result,
    validation_result,
    supporting_dir: Path,
    review_path: Path,
    artifact_href_prefix: str,
) -> None:
    supporting_dir.mkdir(parents=True, exist_ok=True)
    write_model_data_preparation_report(preparation, supporting_dir / "model_data_preparation.md")
    preparation.dataset.frame.to_csv(supporting_dir / "prepared_model_data_5min.csv", index=False)
    write_therapy_research_artifacts(research_result, supporting_dir)
    write_therapy_infra_validation_artifacts(validation_result, supporting_dir)
    write_therapy_evidence_review_html(
        preparation,
        research_result,
        review_path,
        validation_result=validation_result,
        artifact_root=supporting_dir,
        artifact_href_prefix=artifact_href_prefix,
    )


def _build_forecast_summary(
    *,
    args,
    paths: ProjectPaths,
    session: LoggingSession | None = None,
    skip_recommendations: bool = False,
) -> dict[str, object]:
    with _session_stage(session, "run.load_data", raw=args.raw):
        tandem_data = load_tandem_exports(args.raw, include_health_auto_export=False)
        data = load_tandem_exports(args.raw, include_health_auto_export=True)
        prepared = build_prepared_model_dataset(
            tandem_data=tandem_data,
            health_data=data,
            config=FeatureConfig(horizon_minutes=args.horizon),
        )
        feature_frame = _build_feature_frame_from_prepared(prepared)
    with _session_stage(session, "run.assess_inputs", manifest_path=args.manifest):
        export_manifest = build_export_manifest(data)
        write_export_manifest(export_manifest, args.manifest)
        coverage = summarize_coverage(data)
        data_quality, quality_rows = assess_data_quality(args.raw, export_manifest=export_manifest)
    if args.chains < 2:
        warnings.warn(
            f"--chains={args.chains} may produce unreliable diagnostics; use at least 2 chains for MCMC.",
            UserWarning,
            stacklevel=2,
        )
    with _session_stage(session, "run.walk_forward", eval_folds=args.eval_folds):
        model = BayesianGlucoseModel(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            max_treedepth=args.max_treedepth,
        )
        walk_forward = run_walk_forward(feature_frame, model, n_folds=args.eval_folds)
    fit_diagnostics = None
    recommendations = []
    if skip_recommendations:
        recommendation_policy = RecommendationPolicy(
            status="skipped",
            reasons=["skipped_by_flag"],
            validation_passed=False,
            sampler_passed=False,
            signal_passed=False,
        )
    else:
        with _session_stage(session, "run.recommendations", skip_recommendations=False):
            fit = model.fit(feature_frame)
            fit_diagnostics = fit.diagnostics
            recommendations, _, recommendation_policy = recommend_setting_changes(
                model,
                fit,
                feature_frame.frame,
                walk_forward=walk_forward,
                data_quality=data_quality,
                carbs_present=not data.carbs.empty,
                activity_present=not data.activity.empty,
            )
    return build_run_summary(
        coverage=coverage,
        walk_forward=walk_forward,
        recommendations=recommendations,
        fit_diagnostics=fit_diagnostics,
        data_quality=data_quality,
        recommendation_policy=recommendation_policy,
        review_artifacts={},
    )


def _apply_command_defaults(args: argparse.Namespace, paths: ProjectPaths) -> None:
    if args.command in {"ingest", "run", "screen-health-features", "build-health-analysis-ready", "prepare-model-data", "research-therapy-settings", "review-therapy-evidence", "status"}:
        args.raw = args.raw or str(paths.cloud_raw)
        if args.command == "ingest":
            args.report = args.report or str(paths.reports / "coverage.md")
            args.manifest = args.manifest or str(paths.cloud_raw / "tandem_export_manifest.csv")
        elif args.command == "run":
            args.report = args.report or str(paths.reports / "run_summary.md")
            args.manifest = args.manifest or str(paths.cloud_raw / "tandem_export_manifest.csv")
        elif args.command == "status":
            args.manifest = str(paths.cloud_raw / "tandem_export_manifest.csv")
        elif args.command == "prepare-model-data":
            args.output = args.output or str(paths.reports / "prepared_model_data_5min.csv")
            args.report = args.report or str(paths.reports / "model_data_preparation.md")
        elif args.command == "research-therapy-settings":
            args.report_dir = args.report_dir or str(paths.reports)
        elif args.command == "review-therapy-evidence":
            args.report = args.report or str(paths.reports / "therapy_evidence_review.html")
        elif args.command == "build-health-analysis-ready":
            args.output = args.output or str(paths.reports / "analysis_ready_health_5min.csv")
        else:
            args.report = args.report or str(paths.reports / "health_feature_screening.md")
            args.scores = args.scores or str(paths.reports / "health_feature_scores.csv")
    if args.command == "validate-therapy-infra":
        args.report_dir = args.report_dir or str(paths.reports / "therapy_infra_validation")
    if args.command == "normalize-raw":
        args.raw = args.raw or str(paths.cloud_raw / "tconnectsync")
        args.report = args.report or str(paths.reports / "normalize_raw_summary.md")
        args.manifest = args.manifest or str(paths.cloud_raw / "tandem_export_manifest.csv")


def _dispatch_command(
    *,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    paths: ProjectPaths,
    session: LoggingSession,
) -> int:
    if args.command == "ingest":
        with session.stage("ingest.load_data", raw=args.raw):
            data = load_tandem_exports(args.raw, include_health_auto_export=True)
        with session.stage("ingest.manifest", manifest_path=args.manifest):
            export_manifest = build_export_manifest(data)
            write_export_manifest(export_manifest, args.manifest)
        with session.stage("ingest.coverage", report_path=args.report):
            coverage = summarize_coverage(data)
            data_quality, quality_rows = assess_data_quality(args.raw, export_manifest=export_manifest)
        review_path = str(Path(args.report).with_name("coverage_review.html"))
        with session.stage("ingest.reporting", report_path=args.report, review_path=review_path):
            summary = build_run_summary(
                coverage=coverage,
                data_quality=data_quality,
                review_artifacts={"coverage_review_html": review_path},
            )
            write_markdown_report(summary, args.report)
            write_coverage_review_html(summary, quality_rows, review_path)
        return 0

    if args.command == "normalize-raw":
        with session.stage("normalize_raw.rebuild", raw_root=args.raw, report_path=args.report):
            normalize_tconnectsync_archive(
                paths,
                raw_root=args.raw,
                window_id=args.window_id,
                force=args.force,
                report_path=args.report,
                manifest_path=args.manifest,
            )
        return 0

    if args.command == "import-health-auto-export":
        with session.stage("health.import", input_path=args.input):
            import_health_auto_export_batch(args.input, paths.ensure())
        return 0

    if args.command == "prepare-model-data":
        with session.stage("prepare_model_data.build", output_path=args.output):
            preparation = _prepare_model_data(args=args, paths=paths, session=session)
        with session.stage("prepare_model_data.write", output_path=args.output, report_path=args.report):
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            preparation.dataset.frame.to_csv(args.output, index=False)
            write_model_data_preparation_report(preparation, args.report)
        return 0

    if args.command == "research-therapy-settings":
        preparation, result, _ = _run_therapy_analysis(args=args, paths=paths, session=session, include_validation=False)
        with session.stage("therapy.write_artifacts", report_dir=args.report_dir):
            write_therapy_research_artifacts(
                result,
                args.report_dir,
                write_source_report_cards=args.write_source_report_cards,
                write_research_gate=args.write_research_gate,
            )
        return 0

    if args.command == "validate-therapy-infra":
        with session.stage("therapy.validate_infra", report_dir=args.report_dir):
            result = validate_therapy_infra(
                meal_proxy_mode=args.meal_proxy_mode,
                ic_policy=args.ic_policy,
                include_models=parse_model_list(args.include_models or "ridge,segmented_ridge,tree_boost,ensemble"),
            )
            write_therapy_infra_validation_artifacts(result, args.report_dir)
        return 0

    if args.command == "review-therapy-evidence":
        report_path = Path(args.report)
        report_dir = report_path.parent
        preparation, research_result, validation_result = _run_therapy_analysis(args=args, paths=paths, session=session)
        with session.stage("therapy.write_review", report_path=args.report):
            _write_therapy_outputs(
                preparation=preparation,
                research_result=research_result,
                validation_result=validation_result,
                supporting_dir=report_dir,
                review_path=report_path,
                artifact_href_prefix="",
            )
        return 0

    if args.command == "status":
        with session.stage("status.initialize", output_root=str(paths.reports)):
            cleanup_legacy_top_level_output(paths)
            bundle = create_status_bundle(paths, session.context.run_id)
        preparation, research_result, validation_result = _run_therapy_analysis(
            args=args,
            paths=paths,
            session=session,
            preparation_stage="status.prepare_data",
            research_stage="status.therapy_research",
            validation_stage="status.therapy_validation",
        )
        with session.stage("status.write_therapy_bundle", bundle_root=str(bundle.root)):
            _write_therapy_outputs(
                preparation=preparation,
                research_result=research_result,
                validation_result=validation_result,
                supporting_dir=bundle.supporting_dir,
                review_path=bundle.therapy_dir / "therapy_evidence_review.html",
                artifact_href_prefix="../supporting/",
            )
            _write_therapy_outputs(
                preparation=preparation,
                research_result=research_result,
                validation_result=validation_result,
                supporting_dir=bundle.supporting_dir,
                review_path=paths.reports / "therapy_evidence_review.html",
                artifact_href_prefix=f"runs/{session.context.run_id}/supporting/",
            )
        with session.stage("status.forecast_validation", bundle_root=str(bundle.root)):
            forecast_summary = _build_forecast_summary(args=args, paths=paths, session=session, skip_recommendations=True)
        with session.stage("status.write_forecast_bundle", bundle_root=str(bundle.root)):
            bundle_run_summary_md = bundle.supporting_dir / "run_summary.md"
            bundle_run_summary_json = bundle.supporting_dir / "run_summary.json"
            write_markdown_report(
                {
                    **forecast_summary,
                    "review_artifacts": {"run_review_html": "../forecast/run_review.html"},
                },
                bundle_run_summary_md,
            )
            write_json_report(forecast_summary, bundle_run_summary_json)
            write_run_review_html(forecast_summary, bundle.forecast_dir / "run_review.html")
            write_run_review_html(forecast_summary, paths.reports / "run_review.html")
        with session.stage("status.write_status", bundle_root=str(bundle.root)):
            top_level_status_html = paths.reports / "current_status.html"
            top_level_status_json = paths.reports / "current_status.json"
            artifact_paths = {
                "current_status_html": str(top_level_status_html),
                "current_status_json": str(top_level_status_json),
                "therapy_evidence_review_html": str(paths.reports / "therapy_evidence_review.html"),
                "run_review_html": str(paths.reports / "run_review.html"),
                "bundle_root": str(bundle.root),
                "bundle_status_html": str(bundle.status_dir / "current_status.html"),
                "bundle_status_json": str(bundle.status_dir / "current_status.json"),
                "supporting_dir": str(bundle.supporting_dir),
            }
            payload = derive_current_status(
                preparation=preparation,
                research_result=research_result,
                forecast_summary=forecast_summary,
                run_id=session.context.run_id,
                artifact_paths=artifact_paths,
            )
            write_current_status_html(
                payload,
                bundle.status_dir / "current_status.html",
                therapy_href="../therapy/therapy_evidence_review.html",
                forecast_href="../forecast/run_review.html",
            )
            write_status_json(payload, bundle.status_dir / "current_status.json")
            write_current_status_html(
                payload,
                top_level_status_html,
                therapy_href="therapy_evidence_review.html",
                forecast_href="run_review.html",
            )
            write_status_json(payload, top_level_status_json)
            publish_latest_entrypoints(
                paths=paths,
                run_id=session.context.run_id,
                payload=payload,
                bundle=bundle,
                top_level_status_html=top_level_status_html,
                top_level_status_json=top_level_status_json,
                top_level_therapy_html=paths.reports / "therapy_evidence_review.html",
                top_level_forecast_html=paths.reports / "run_review.html",
            )
        return 0

    if args.command == "build-health-analysis-ready":
        with session.stage("analysis_ready.load_data", raw=args.raw):
            tandem_data = load_tandem_exports(args.raw, include_health_auto_export=False)
            health_data = load_tandem_exports(args.raw, include_health_auto_export=True)
        with session.stage("analysis_ready.build", output_path=args.output):
            analysis_ready = build_prepared_model_dataset(
                tandem_data=tandem_data,
                health_data=health_data,
                config=FeatureConfig(horizon_minutes=args.horizon),
            )
        with session.stage("analysis_ready.write", output_path=args.output):
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            analysis_ready.frame.to_csv(args.output, index=False)
        return 0

    if args.command == "screen-health-features":
        with session.stage("health_screen.load_data", raw=args.raw):
            tandem_data = load_tandem_exports(args.raw, include_health_auto_export=False)
            health_data = load_tandem_exports(args.raw, include_health_auto_export=True)
        with session.stage("health_screen.score", report_path=args.report):
            screening = screen_health_features(tandem_data=tandem_data, health_data=health_data, horizon_minutes=args.horizon)
        with session.stage("health_screen.write", report_path=args.report, scores_path=args.scores):
            Path(args.scores).parent.mkdir(parents=True, exist_ok=True)
            screening.scores.to_csv(args.scores, index=False)
            write_health_screening_report(screening, args.report)
        return 0

    if args.command == "run":
        summary = _build_forecast_summary(
            args=args,
            paths=paths,
            session=session,
            skip_recommendations=bool(args.skip_recommendations),
        )
        review_path = str(Path(args.report).with_name("run_review.html"))
        with session.stage("run.reporting", report_path=args.report, review_path=review_path):
            write_markdown_report({**summary, "review_artifacts": {"run_review_html": review_path}}, args.report)
            json_path = Path(args.report).with_suffix(".json")
            write_json_report(summary, json_path)
            write_run_review_html(summary, review_path)
        return 0

    if args.command == "validate-raw":
        args.raw = args.raw or str(paths.cloud_raw)
        args.report = args.report or str(paths.reports / "tandem_raw_validation.md")
        with session.stage("validate_raw.summarize", raw=args.raw, report_path=args.report):
            summary = summarize_tandem_raw_dir(args.raw)
            report_lines = [
                "# Tandem Raw Validation",
                "",
                f"- raw_dir: {args.raw}",
                f"- file_count: {len(summary)}",
            ]
            if summary.empty:
                report_lines.extend(["", "No Tandem raw files were found."])
            else:
                report_lines.extend(
                    [
                        "",
                        "| file | dense_cgm | cgm_rows | all_cgm_rows | median_spacing_minutes | first_timestamp | last_timestamp |",
                        "| --- | --- | ---: | ---: | ---: | --- | --- |",
                    ]
                )
                for row in summary.itertuples(index=False):
                    report_lines.append(
                        "| {file} | {dense} | {cgm_rows} | {all_rows} | {spacing} | {first} | {last} |".format(
                            file=row.source_file,
                            dense="yes" if bool(row.has_dense_cgm_stream) else "no",
                            cgm_rows=int(row.cgm_rows),
                            all_rows=int(row.all_cgm_rows),
                            spacing="NA" if pd.isna(row.median_spacing_minutes) else f"{float(row.median_spacing_minutes):.1f}",
                            first="" if pd.isna(row.first_timestamp) else str(pd.Timestamp(row.first_timestamp)),
                            last="" if pd.isna(row.last_timestamp) else str(pd.Timestamp(row.last_timestamp)),
                        )
                    )
                dense = summary.loc[summary["has_dense_cgm_stream"].fillna(False)]
                report_lines.extend(
                    [
                        "",
                        f"- dense_cgm_files: {len(dense)}",
                        f"- dense_cgm_rows_total: {int(dense['cgm_rows'].sum()) if not dense.empty else 0}",
                    ]
                )
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        return 0

    if args.command in {"collect", "backfill"}:
        with session.stage("acquisition.credentials", command=args.command):
            credentials = load_tandem_credentials(args.root, args.env_file)
        manifest_path = args.manifest or str(paths.cloud_raw / "tandem_export_manifest.csv")
        report_path = args.report or str(paths.reports / "tandem_acquisition_summary.md")
        client_kwargs = {
            "region": credentials.region,
            "timezone": credentials.timezone,
            "pump_serial": credentials.pump_serial,
        }
        step_log = StepLogger(session=session)
        with session.stage("acquisition.run", manifest_path=manifest_path, report_path=report_path):
            client_cm = TConnectSyncSourceClient(paths, **client_kwargs)
            with client_cm as client:
                if args.command == "collect":
                    start_date = date.fromisoformat(args.start_date) if args.start_date else date.today() - timedelta(days=29)
                    end_date = date.fromisoformat(args.end_date) if args.end_date else date.today()
                    if (end_date - start_date).days + 1 > 30:
                        parser.error("collect accepts at most a 30-day window; use backfill for larger ranges")
                    windows = [ExportWindow(start_date=start_date, end_date=end_date)]
                    collect_tandem_exports(
                        client,
                        windows,
                        paths,
                        credentials,
                        manifest_path=manifest_path,
                        report_path=report_path,
                        resume=True,
                        strict=args.strict,
                        step_log=step_log,
                    )
                    return 0
                backfill_tandem_exports(
                    client,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    workspace=paths,
                    credentials=credentials,
                    window_days=args.window_days,
                    direction=args.direction,
                    manifest_path=manifest_path,
                    report_path=report_path,
                    resume=args.resume,
                    strict=args.strict,
                    step_log=step_log,
                )
                return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = ProjectPaths.from_root(args.root).ensure()
    _apply_command_defaults(args, paths)

    session = setup_run_logging(
        paths,
        command=args.command,
        argv=list(argv) if argv is not None else sys.argv[1:],
        log_level=args.log_level,
        quiet=args.quiet,
        unsafe_debug_logging=args.unsafe_debug_logging,
    )
    session.start()
    session.log_event(
        "command.start",
        message=f"{args.command} started",
        status="started",
        root=str(paths.root),
    )
    if session.startup_warning is not None:
        session.log_event(
            "command.warning",
            level="WARNING",
            message=session.startup_warning,
            status="degraded",
        )

    try:
        exit_code = _dispatch_command(args=args, parser=parser, paths=paths, session=session)
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
        if not session.error_logged:
            session.log_event(
                "command.error",
                level="ERROR",
                message=str(exc),
                stage=session.current_stage,
                status="failed",
                error_type="SystemExit",
                exit_code=exit_code,
            )
        session.finalize(exit_code=exit_code, status="failed")
        if args.command == "status":
            bundle = create_status_bundle(paths, session.context.run_id)
            finalize_status_logs(bundle, session.context.run_dir)
        raise
    except Exception as exc:
        if not session.error_logged:
            session.log_event(
                "command.error",
                level="ERROR",
                message=str(exc),
                stage=session.current_stage,
                status="failed",
                error_type=type(exc).__name__,
            )
        session.finalize(exit_code=1, status="failed")
        if args.command == "status":
            bundle = create_status_bundle(paths, session.context.run_id)
            finalize_status_logs(bundle, session.context.run_dir)
        raise

    session.finalize(exit_code=exit_code, status="success" if exit_code == 0 else "failed")
    if args.command == "status":
        bundle = create_status_bundle(paths, session.context.run_id)
        finalize_status_logs(bundle, session.context.run_dir)
    return exit_code
