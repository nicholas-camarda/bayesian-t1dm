from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from datetime import date, timedelta
from pathlib import Path
import sys

import pandas as pd

import warnings

from .evaluate import run_walk_forward
from .acquisition import (
    load_local_env_file,
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
    AnalysisReadyHealthDataset,
    screen_health_features,
    write_health_screening_report,
    write_model_data_preparation_report,
    ModelDataPreparationResult,
)
from .latent_meal_validation import run_effective_icr_recovery, run_latent_meal_truth_recovery, write_latent_meal_recovery_artifacts
from .dataset import MergedDataset, build_merged_dataset, load_merged_dataset, merged_dir
from .ingest import summarize_tandem_raw_dir
from .model import BayesianGlucoseModel
from .observability import LoggingSession, setup_run_logging
from .outcomes_report import write_outcomes
from .paths import ProjectPaths
from .recommend import RecommendationPolicy, recommend_setting_changes
from .report import build_run_summary, write_json_report, write_markdown_report
from .review import write_coverage_review_html, write_current_status_html, write_latent_meal_review_html, write_run_review_html, write_therapy_evidence_review_html
from .features import FeatureFrame
from .status import cleanup_legacy_top_level_output, derive_current_status, finalize_status_logs, publish_html_entrypoint, reset_output_directory, write_status_json
from .therapy_research import (
    load_excluded_periods,
    build_representative_latent_meal_fixture,
    extract_latent_meal_therapy_settings_from_tandem_raw,
    load_latent_meal_therapy_settings,
    parse_model_list,
    parse_therapy_segments,
    run_latent_meal_icr_research,
    run_therapy_research,
    validate_therapy_infra,
    write_latent_meal_research_artifacts,
    write_representative_latent_meal_fixture_artifacts,
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

    normalize_raw = add_command("normalize-raw", "Rebuild normalized tconnectsync windows from archived raw payloads")
    normalize_raw.add_argument("--raw", default=None)
    normalize_raw.add_argument("--window-id", default=None)
    normalize_raw.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    normalize_raw.add_argument("--report", default=None)
    normalize_raw.add_argument("--manifest", default=None)

    validate_raw = add_command("validate-raw", "Classify Tandem raw files as dense CGM or summary-only")
    validate_raw.add_argument("--raw", default=None)
    validate_raw.add_argument("--report", default=None)

    build_dataset = add_command(
        "build-dataset",
        "Merge, deduplicate and validate all raw Tandem and Apple Health data into one dataset (only when the raw files changed)",
    )
    build_dataset.add_argument("--raw", default=None)
    build_dataset.add_argument("--force", action=argparse.BooleanOptionalAction, default=False, help="Rebuild even if no raw file changed")

    outcomes = add_command("report-outcomes", "Report consensus glucose outcomes (time in range, lows, highs) overall and by pump time-of-day segment")
    outcomes.add_argument("--raw", default=None)
    outcomes.add_argument("--report-dir", default=None)

    screen_health = add_command("screen-health-features", "Screen imported Apple Health context features against Tandem glucose targets")
    screen_health.add_argument("--raw", default=None)
    screen_health.add_argument("--report", default=None)
    screen_health.add_argument("--scores", default=None)

    research_therapy = add_command("research-therapy-settings", "Run research-grade therapy setting analysis on prepared Tandem and Apple Health data")
    research_therapy.add_argument("--raw", default=None)
    research_therapy.add_argument("--segments", default=None, help="Comma-separated day segments like overnight=00:00-06:00,morning=06:00-11:00")
    research_therapy.add_argument("--include-models", default=None, help="Comma-separated model families such as bayesian,ridge,elastic_net,segmented_ridge,tree_boost,ensemble")
    research_therapy.add_argument("--report-dir", default=None)
    research_therapy.add_argument("--meal-proxy-mode", choices=["strict", "broad", "off"], default="strict")
    research_therapy.add_argument("--ic-policy", choices=["exploratory_only", "conservative", "off"], default="exploratory_only")
    research_therapy.add_argument("--write-source-report-cards", action=argparse.BooleanOptionalAction, default=True)
    research_therapy.add_argument("--write-research-gate", action=argparse.BooleanOptionalAction, default=True)

    latent_meal = add_command("research-latent-meal-icr", "Run an experimental latent meal and I/C research workflow")
    latent_meal.add_argument("--raw", default=None)
    latent_meal.add_argument("--segments", default=None, help="Comma-separated day segments like overnight=00:00-06:00,morning=06:00-11:00")
    latent_meal.add_argument("--report-dir", default=None)
    latent_meal.add_argument("--review-html", default=None)
    latent_meal.add_argument("--meal-proxy-mode", choices=["strict", "broad", "off"], default="strict")
    latent_meal.add_argument("--research-scope", choices=["foundation", "full"], default="foundation")
    latent_meal.add_argument("--therapy-settings", default=None, help="Optional JSON override for I:C, correction factor, target, insulin action, and carb absorption priors; full scope auto-extracts Tandem settings when omitted")

    latent_fixture = add_command("build-latent-meal-fixture", "Build a representative prepared-data fixture for latent meal foundation debugging")
    latent_fixture.add_argument("--prepared-csv", default=None, help="Prepared model dataset CSV to subset; defaults to the merged dataset")
    latent_fixture.add_argument("--output-dir", default=None, help="Directory to write the fixture dataset and foundation artifacts")
    latent_fixture.add_argument("--review-html", default=None, help="Optional HTML review path for the fixture")
    latent_fixture.add_argument("--segments", default=None, help="Comma-separated day segments like overnight=00:00-06:00,morning=06:00-11:00")
    latent_fixture.add_argument("--meal-proxy-mode", choices=["strict", "broad", "off"], default="strict")
    latent_fixture.add_argument("--background-days", type=int, default=7)
    latent_fixture.add_argument("--seed", type=int, default=17)
    latent_fixture.add_argument("--horizon", type=int, default=30)

    validate_therapy = add_command("validate-therapy-infra", "Run synthetic truth-recovery validation for the therapy research infrastructure")
    validate_therapy.add_argument("--report-dir", default=None)
    validate_therapy.add_argument("--include-models", default=None, help="Comma-separated model families such as ridge,segmented_ridge,tree_boost,ensemble")
    validate_therapy.add_argument("--meal-proxy-mode", choices=["strict", "broad", "off"], default="strict")
    validate_therapy.add_argument("--ic-policy", choices=["exploratory_only", "conservative", "off"], default="exploratory_only")

    validate_latent = add_command("validate-latent-meal", "Run synthetic truth-recovery validation for the latent carb estimators")
    validate_latent.add_argument("--report-dir", default=None)
    validate_latent.add_argument("--meals", type=int, default=200, help="Simulated meals per scenario")
    validate_latent.add_argument("--seed", type=int, default=20260403)
    validate_latent.add_argument("--icr-cohorts", type=int, default=40, help="Simulated 20-meal cohorts per true I:C for the logged-meal recovery check; 0 skips it")

    review_therapy = add_command("review-therapy-evidence", "Build a therapy evidence report focused on identifiability and overnight basal evidence")
    review_therapy.add_argument("--raw", default=None)
    review_therapy.add_argument("--report", default=None)
    review_therapy.add_argument("--segments", default=None)
    review_therapy.add_argument("--include-models", default=None)
    review_therapy.add_argument("--meal-proxy-mode", choices=["strict", "broad", "off"], default="strict")
    review_therapy.add_argument("--ic-policy", choices=["exploratory_only", "conservative", "off"], default="exploratory_only")

    status = add_command("status", "Run the primary end-to-end status workflow and publish the latest dashboards")
    status.add_argument("--raw", default=None)
    status.add_argument("--segments", default=None)
    status.add_argument("--include-models", default=None)
    status.add_argument("--meal-proxy-mode", choices=["strict", "broad", "off"], default="strict")
    status.add_argument("--ic-policy", choices=["exploratory_only", "conservative", "off"], default="exploratory_only")
    status.add_argument("--eval-folds", type=int, default=4)
    status.add_argument("--draws", type=int, default=1000)
    status.add_argument("--tune", type=int, default=1000)
    status.add_argument("--chains", type=int, default=4)
    status.add_argument("--target-accept", type=float, default=0.95)
    status.add_argument("--max-treedepth", type=int, default=12)

    run = add_command("run", "Run the full forecasting and recommendation pipeline")
    run.add_argument("--raw", default=None)
    run.add_argument("--report", default=None)
    run.add_argument("--eval-folds", type=int, default=4)
    run.add_argument("--draws", type=int, default=1000)
    run.add_argument("--tune", type=int, default=1000)
    run.add_argument("--chains", type=int, default=4)
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


def _load_dataset(args, session: LoggingSession | None = None) -> MergedDataset:
    with _session_stage(session, "dataset.load", raw=args.raw):
        return load_merged_dataset(args.raw)


def _run_therapy_analysis(
    *,
    args,
    merged: MergedDataset,
    session: LoggingSession | None = None,
    research_stage: str = "therapy.research",
    validation_stage: str = "therapy.validation",
    include_validation: bool = True,
):
    preparation = merged.preparation
    with _session_stage(session, research_stage):
        research_result = run_therapy_research(
            preparation.dataset,
            segments=parse_therapy_segments(args.segments),
            include_models=parse_model_list(args.include_models),
            meal_proxy_mode=args.meal_proxy_mode,
            ic_policy=args.ic_policy,
            excluded_periods=load_excluded_periods(args.raw),
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
    merged: MergedDataset,
    session: LoggingSession | None = None,
    skip_recommendations: bool = False,
) -> dict[str, object]:
    prepared = merged.preparation.dataset
    feature_frame = _build_feature_frame_from_prepared(prepared)
    data = merged.data
    coverage = merged.coverage
    data_quality = merged.data_quality
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
                carbs_present=bool(getattr(prepared, "explicit_carb_source_available", False)),
                activity_present=not data.activity.empty or not data.health_activity.empty,
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


def _run_latent_meal_analysis(
    *,
    args,
    paths: ProjectPaths,
    session: LoggingSession | None = None,
):
    therapy_settings = None
    if getattr(args, "therapy_settings", None):
        therapy_settings = load_latent_meal_therapy_settings(args.therapy_settings)
    elif getattr(args, "research_scope", "foundation") == "full":
        therapy_settings = extract_latent_meal_therapy_settings_from_tandem_raw(
            args.raw,
            segments=parse_therapy_segments(args.segments),
        )
    merged = _load_dataset(args, session)
    preparation = merged.preparation
    with _session_stage(session, "latent_meal.research"):
        result = run_latent_meal_icr_research(
            preparation.dataset,
            segments=parse_therapy_segments(args.segments),
            meal_proxy_mode=args.meal_proxy_mode,
            research_scope=args.research_scope,
            therapy_settings=therapy_settings,
            excluded_periods=load_excluded_periods(args.raw),
            nutrition=merged.data.nutrition,
        )
    return preparation, result


def _load_prepared_dataset_csv(
    prepared_csv: str | Path,
    *,
    horizon_minutes: int,
) -> AnalysisReadyHealthDataset:
    path = Path(prepared_csv)
    if not path.exists():
        raise FileNotFoundError(f"Prepared dataset CSV not found: {path}")
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    semantic_columns = {
        "timestamp",
        "target_glucose",
        "target_delta",
        "explicit_carb_grams",
        "explicit_meal_event",
        "minutes_since_last_explicit_meal",
        "explicit_carb_source_available",
        "meal_truth_status",
    }
    feature_columns = [column for column in frame.columns if column not in semantic_columns]
    health_markers = (
        "sleep",
        "workout",
        "hrv",
        "heart_rate",
        "respiratory",
        "resting_",
        "health_",
    )
    health_feature_columns = [
        column
        for column in feature_columns
        if any(marker in column for marker in health_markers)
    ]
    tandem_feature_columns = [column for column in feature_columns if column not in health_feature_columns]
    explicit_carb_source_available = False
    if "explicit_carb_source_available" in frame.columns:
        explicit_carb_source_available = bool(pd.to_numeric(frame["explicit_carb_source_available"], errors="coerce").fillna(0).astype(int).max())
    elif "meal_truth_status" in frame.columns:
        explicit_carb_source_available = bool(frame["meal_truth_status"].astype(str).eq("observed_explicit").any())
    elif "explicit_carb_grams" in frame.columns:
        explicit_carb_source_available = bool(pd.to_numeric(frame["explicit_carb_grams"], errors="coerce").notna().any())
    return AnalysisReadyHealthDataset(
        frame=frame,
        feature_columns=feature_columns,
        tandem_feature_columns=tandem_feature_columns,
        health_feature_columns=health_feature_columns,
        target_column="target_glucose" if "target_glucose" in frame.columns else "target_glucose",
        horizon_minutes=int(horizon_minutes),
        config=FeatureConfig(horizon_minutes=int(horizon_minutes)),
        mode="apple_enriched" if bool(health_feature_columns) else "tandem_only",
        apple_available=bool(health_feature_columns),
        explicit_carb_source_available=explicit_carb_source_available,
    )


def _run_latent_meal_fixture_builder(
    *,
    args,
    paths: ProjectPaths,
    session: LoggingSession | None = None,
):
    with _session_stage(session, "latent_meal_fixture.load_prepared", prepared_csv=args.prepared_csv):
        if args.prepared_csv:
            dataset = _load_prepared_dataset_csv(args.prepared_csv, horizon_minutes=args.horizon)
        else:
            dataset = load_merged_dataset(paths.cloud_raw).preparation.dataset
    with _session_stage(session, "latent_meal_fixture.build", background_days=args.background_days):
        result = build_representative_latent_meal_fixture(
            dataset,
            segments=parse_therapy_segments(args.segments),
            meal_proxy_mode=args.meal_proxy_mode,
            background_days=args.background_days,
            seed=args.seed,
        )
    return result


def _path_matches(candidate: str | Path | None, expected: str | Path) -> bool:
    if candidate is None:
        return False
    return Path(candidate).expanduser().resolve() == Path(expected).expanduser().resolve()


def _prepare_runtime_output(paths: ProjectPaths, *directories: str | Path) -> None:
    cleanup_legacy_top_level_output(paths)
    for directory in directories:
        reset_output_directory(directory)


def _apply_command_defaults(args: argparse.Namespace, paths: ProjectPaths) -> None:
    if args.command in {"ingest", "run", "screen-health-features", "build-dataset", "research-therapy-settings", "research-latent-meal-icr", "review-therapy-evidence", "status", "report-outcomes"}:
        args.raw = args.raw or str(paths.cloud_raw)
        if args.command == "ingest":
            args.report = args.report or str(paths.output_source / "coverage.md")
        elif args.command == "run":
            args.report = args.report or str(paths.output_forecast / "run_summary.md")
        elif args.command == "research-therapy-settings":
            args.report_dir = args.report_dir or str(paths.output_therapy)
        elif args.command == "research-latent-meal-icr":
            args.report_dir = args.report_dir or str(paths.output_latent_meal)
            args.review_html = args.review_html or str(Path(args.report_dir) / "latent_meal_review.html")
        elif args.command == "review-therapy-evidence":
            args.report = args.report or str(paths.output_therapy / "therapy_review.html")
        elif args.command == "report-outcomes":
            args.report_dir = args.report_dir or str(paths.output_outcomes)
        elif args.command in {"status", "build-dataset"}:
            pass
        else:
            args.report = args.report or str(paths.output_prepare / "health_feature_screening.md")
            args.scores = args.scores or str(paths.cache_prepare / "health_feature_scores.csv")
    if args.command == "build-latent-meal-fixture":
        args.output_dir = args.output_dir or str(paths.output_fixture)
        args.review_html = args.review_html or str(Path(args.output_dir) / "latent_meal_review.html")
    if args.command == "validate-therapy-infra":
        args.report_dir = args.report_dir or str(paths.output_therapy / "validation")
    if args.command == "validate-latent-meal":
        args.report_dir = args.report_dir or str(paths.output_latent_meal / "validation")
    if args.command == "normalize-raw":
        args.raw = args.raw or str(paths.cloud_raw / "tconnectsync")
        args.report = args.report or str(paths.output_source / "normalize_raw_summary.md")
        args.manifest = args.manifest or str(paths.cloud_raw / "tandem_export_manifest.csv")


def _dispatch_command(
    *,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    paths: ProjectPaths,
    session: LoggingSession,
) -> int:
    if args.command == "ingest":
        if _path_matches(args.report, paths.output_source / "coverage.md"):
            _prepare_runtime_output(paths, paths.output_source)
        merged = _load_dataset(args, session)
        coverage, data_quality, quality_rows = merged.coverage, merged.data_quality, merged.quality_rows
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
        if _path_matches(args.report, paths.output_source / "normalize_raw_summary.md"):
            _prepare_runtime_output(paths, paths.output_source)
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

    if args.command == "report-outcomes":
        merged = _load_dataset(args, session)
        with session.stage("outcomes.write", report_dir=args.report_dir):
            written = write_outcomes(merged, args.raw, args.report_dir)
        print(f"Outcomes report: {written['page']}")
        return 0

    if args.command == "build-dataset":
        with session.stage("dataset.build", raw=args.raw, force=args.force):
            merged, rebuilt = build_merged_dataset(paths, raw_root=args.raw, force=args.force)
        if rebuilt:
            _prepare_runtime_output(paths, paths.output_prepare)
            write_model_data_preparation_report(merged.preparation, paths.output_prepare / "model_data_preparation.md")
        print(f"Validation report: {merged_dir(args.raw) / 'validation_report.md'}")
        return 0

    if args.command == "research-therapy-settings":
        preparation, result, _ = _run_therapy_analysis(args=args, merged=_load_dataset(args, session), session=session, include_validation=False)
        if _path_matches(args.report_dir, paths.output_therapy):
            _prepare_runtime_output(paths, paths.output_therapy)  # only after the analysis succeeded: a failure leaves the last output
        with session.stage("therapy.write_artifacts", report_dir=args.report_dir):
            write_therapy_research_artifacts(
                result,
                args.report_dir,
                write_source_report_cards=args.write_source_report_cards,
                write_research_gate=args.write_research_gate,
            )
        return 0

    if args.command == "research-latent-meal-icr":
        _, result = _run_latent_meal_analysis(args=args, paths=paths, session=session)
        if _path_matches(args.report_dir, paths.output_latent_meal):
            _prepare_runtime_output(paths, paths.output_latent_meal)
        with session.stage("latent_meal.write_artifacts", report_dir=args.report_dir, review_html=args.review_html):
            write_latent_meal_research_artifacts(result, args.report_dir, cache_dir=paths.cache_latent_meal)
            write_latent_meal_review_html(
                result,
                args.review_html,
                artifact_root=args.report_dir,
                artifact_href_prefix="",
            )
        return 0

    if args.command == "build-latent-meal-fixture":
        if _path_matches(args.output_dir, paths.output_fixture):
            _prepare_runtime_output(paths, paths.output_fixture)
        result = _run_latent_meal_fixture_builder(args=args, paths=paths, session=session)
        with session.stage("latent_meal_fixture.write_artifacts", output_dir=args.output_dir, review_html=args.review_html):
            prepared_csv_path = paths.cache_latent_meal / "prepared_model_data_5min.csv" if _path_matches(args.output_dir, paths.output_fixture) else None
            write_representative_latent_meal_fixture_artifacts(result, args.output_dir, prepared_csv_path=prepared_csv_path)
            write_latent_meal_review_html(
                result.fixture_result,
                args.review_html,
                artifact_root=args.output_dir,
                artifact_href_prefix="",
            )
        return 0

    if args.command == "validate-therapy-infra":
        if _path_matches(args.report_dir, paths.output_therapy / "validation"):
            cleanup_legacy_top_level_output(paths)
            reset_output_directory(args.report_dir)
        with session.stage("therapy.validate_infra", report_dir=args.report_dir):
            result = validate_therapy_infra(
                meal_proxy_mode=args.meal_proxy_mode,
                ic_policy=args.ic_policy,
                include_models=parse_model_list(args.include_models or "ridge,segmented_ridge,tree_boost,ensemble"),
            )
            write_therapy_infra_validation_artifacts(result, args.report_dir)
        return 0

    if args.command == "validate-latent-meal":
        with session.stage("latent_meal.validate_estimators", report_dir=args.report_dir, meals=args.meals, seed=args.seed):
            recovery = run_latent_meal_truth_recovery(n_meals_per_scenario=args.meals, seed=args.seed)
            icr_recovery = run_effective_icr_recovery(n_cohorts=args.icr_cohorts, seed=args.seed) if args.icr_cohorts > 0 else None
            recovery_paths = write_latent_meal_recovery_artifacts(recovery, args.report_dir, icr_recovery=icr_recovery)
        print(f"Latent meal truth recovery: {'pass' if recovery.passed() else 'FAIL'} -> {recovery_paths['report']}")
        return 0 if recovery.passed() else 1

    if args.command == "review-therapy-evidence":
        report_path = Path(args.report)
        report_dir = report_path.parent
        publish_latest = _path_matches(report_path, paths.output_therapy / "therapy_review.html")
        preparation, research_result, validation_result = _run_therapy_analysis(args=args, merged=_load_dataset(args, session), session=session)
        if publish_latest:
            _prepare_runtime_output(paths, paths.output_therapy)
        with session.stage("therapy.write_review", report_path=args.report):
            _write_therapy_outputs(
                preparation=preparation,
                research_result=research_result,
                validation_result=validation_result,
                supporting_dir=report_dir,
                review_path=report_path,
                artifact_href_prefix="",
            )
            if publish_latest:
                write_therapy_evidence_review_html(
                    preparation,
                    research_result,
                    paths.reports / "therapy_review.html",
                    validation_result=validation_result,
                    artifact_root=report_dir,
                    artifact_href_prefix="therapy/",
                )
        return 0

    if args.command == "status":
        merged = _load_dataset(args, session)
        preparation, research_result, validation_result = _run_therapy_analysis(
            args=args,
            merged=merged,
            session=session,
            research_stage="status.therapy_research",
            validation_stage="status.therapy_validation",
        )
        with session.stage("status.forecast_validation", forecast_root=str(paths.output_forecast)):
            forecast_summary = _build_forecast_summary(args=args, merged=merged, session=session, skip_recommendations=True)
        with session.stage("status.initialize", output_root=str(paths.reports)):
            _prepare_runtime_output(paths, paths.output_therapy, paths.output_forecast)  # only after both analyses succeeded
        with session.stage("status.write_therapy_bundle", therapy_root=str(paths.output_therapy)):
            _write_therapy_outputs(
                preparation=preparation,
                research_result=research_result,
                validation_result=validation_result,
                supporting_dir=paths.output_therapy,
                review_path=paths.output_therapy / "therapy_review.html",
                artifact_href_prefix="",
            )
            write_therapy_evidence_review_html(
                preparation,
                research_result,
                paths.reports / "therapy_review.html",
                validation_result=validation_result,
                artifact_root=paths.output_therapy,
                artifact_href_prefix="therapy/",
            )
        with session.stage("status.write_forecast_bundle", forecast_root=str(paths.output_forecast)):
            forecast_report_path = paths.output_forecast / "run_summary.md"
            forecast_review_path = paths.output_forecast / "forecast_review.html"
            forecast_json_path = paths.cache_forecast / "run_summary.json"
            write_markdown_report(
                {
                    **forecast_summary,
                    "review_artifacts": {"forecast_review_html": "forecast_review.html"},
                },
                forecast_report_path,
            )
            write_json_report(forecast_summary, forecast_json_path)
            write_run_review_html(forecast_summary, forecast_review_path)
            publish_html_entrypoint(forecast_review_path, paths.reports / "forecast_review.html")
        with session.stage("status.write_status", output_root=str(paths.reports)):
            top_level_status_html = paths.reports / "current_status.html"
            cached_status_json = paths.cache_status / "current_status.json"
            artifact_paths = {
                "current_status_html": str(top_level_status_html),
                "current_status_json": str(cached_status_json),
                "therapy_review_html": str(paths.reports / "therapy_review.html"),
                "forecast_review_html": str(paths.reports / "forecast_review.html"),
                "therapy_dir": str(paths.output_therapy),
                "forecast_dir": str(paths.output_forecast),
                "merged_dataset_dir": str(merged_dir(args.raw)),
                "forecast_summary_json": str(paths.cache_forecast / "run_summary.json"),
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
                top_level_status_html,
                therapy_href="therapy_review.html",
                forecast_href="forecast_review.html",
            )
            write_status_json(payload, cached_status_json)
        return 0

    if args.command == "screen-health-features":
        if _path_matches(args.report, paths.output_prepare / "health_feature_screening.md"):
            _prepare_runtime_output(paths, paths.output_prepare)
        merged = _load_dataset(args, session)
        with session.stage("health_screen.score", report_path=args.report):
            screening = screen_health_features(tandem_data=merged.tandem_data, health_data=merged.data, horizon_minutes=merged.preparation.dataset.horizon_minutes)
        with session.stage("health_screen.write", report_path=args.report, scores_path=args.scores):
            Path(args.scores).parent.mkdir(parents=True, exist_ok=True)
            screening.scores.to_csv(args.scores, index=False)
            write_health_screening_report(screening, args.report)
        return 0

    if args.command == "run":
        publish_latest = _path_matches(args.report, paths.output_forecast / "run_summary.md")
        summary = _build_forecast_summary(
            args=args,
            merged=_load_dataset(args, session),
            session=session,
            skip_recommendations=bool(args.skip_recommendations),
        )
        if publish_latest:
            # Clear the previous run's output only once the new fit has succeeded, so a failed run leaves it in place.
            _prepare_runtime_output(paths, paths.output_forecast)
        review_path = str(Path(args.report).with_name("forecast_review.html"))
        with session.stage("run.reporting", report_path=args.report, review_path=review_path):
            write_markdown_report({**summary, "review_artifacts": {"forecast_review_html": review_path}}, args.report)
            json_path = paths.cache_forecast / "run_summary.json" if publish_latest else Path(args.report).with_suffix(".json")
            write_json_report(summary, json_path)
            write_run_review_html(summary, review_path)
            if publish_latest:
                publish_html_entrypoint(review_path, paths.reports / "forecast_review.html")
        return 0

    if args.command == "validate-raw":
        args.raw = args.raw or str(paths.cloud_raw)
        args.report = args.report or str(paths.output_source / "tandem_raw_validation.md")
        if _path_matches(args.report, paths.output_source / "tandem_raw_validation.md"):
            _prepare_runtime_output(paths, paths.output_source)
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
        report_path = args.report or str(paths.output_source / "tandem_acquisition_summary.md")
        if _path_matches(report_path, paths.output_source / "tandem_acquisition_summary.md"):
            _prepare_runtime_output(paths, paths.output_source)
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
    # Settings such as TIMEZONE_NAME live in the repository's .env; the shell environment wins.
    for key, value in load_local_env_file(paths.root / ".env").items():
        os.environ.setdefault(key, value)
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
            finalize_status_logs(paths, session.context.run_dir)
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
            finalize_status_logs(paths, session.context.run_dir)
        raise

    session.finalize(exit_code=exit_code, status="success" if exit_code == 0 else "failed")
    if args.command == "status":
        finalize_status_logs(paths, session.context.run_dir)
    return exit_code
