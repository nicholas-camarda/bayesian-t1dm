from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

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
from .paths import ProjectPaths
from .quality import assess_data_quality
from .recommend import RecommendationPolicy, recommend_setting_changes
from .report import build_run_summary, write_json_report, write_markdown_report
from .review import write_coverage_review_html, write_run_review_html
from .features import FeatureFrame
from .therapy_research import (
    parse_model_list,
    parse_therapy_segments,
    run_therapy_research,
    validate_therapy_infra,
    write_therapy_infra_validation_artifacts,
    write_therapy_research_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bayesian-t1dm")
    parser.add_argument("--root", default=".", help="Project root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Load Tandem exports and write a coverage report")
    ingest.add_argument("--raw", default=None)
    ingest.add_argument("--report", default=None)
    ingest.add_argument("--manifest", default=None)

    normalize_raw = subparsers.add_parser("normalize-raw", help="Rebuild normalized tconnectsync windows from archived raw payloads")
    normalize_raw.add_argument("--raw", default=None)
    normalize_raw.add_argument("--window-id", default=None)
    normalize_raw.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    normalize_raw.add_argument("--report", default=None)
    normalize_raw.add_argument("--manifest", default=None)

    validate_raw = subparsers.add_parser("validate-raw", help="Classify Tandem raw files as dense CGM or summary-only")
    validate_raw.add_argument("--raw", default=None)
    validate_raw.add_argument("--report", default=None)

    import_health = subparsers.add_parser("import-health-auto-export", help="Archive and normalize a Health Auto Export JSON bundle")
    import_health.add_argument("--input", required=True, help="Path to the exported Health Auto Export directory")

    prepare_model = subparsers.add_parser("prepare-model-data", help="Prepare the best available model dataset with optional Apple Health enrichment")
    prepare_model.add_argument("--raw", default=None)
    prepare_model.add_argument("--apple-input", default=None, help="Optional Health Auto Export parent directory")
    prepare_model.add_argument("--output", default=None)
    prepare_model.add_argument("--report", default=None)
    prepare_model.add_argument("--horizon", type=int, default=30)
    prepare_model.add_argument("--history-days", type=int, default=365)
    prepare_model.add_argument("--min-history-days", type=int, default=180)
    prepare_model.add_argument("--skip-backfill", action=argparse.BooleanOptionalAction, default=False)
    prepare_model.add_argument("--env-file", default=None, help="Optional .env file with Tandem credentials")

    analysis_ready = subparsers.add_parser("build-health-analysis-ready", help="Build a Tandem-aligned 5-minute analysis-ready dataset with unified Apple Health context")
    analysis_ready.add_argument("--raw", default=None)
    analysis_ready.add_argument("--output", default=None)
    analysis_ready.add_argument("--horizon", type=int, default=30)

    screen_health = subparsers.add_parser("screen-health-features", help="Screen imported Apple Health context features against Tandem glucose targets")
    screen_health.add_argument("--raw", default=None)
    screen_health.add_argument("--report", default=None)
    screen_health.add_argument("--scores", default=None)
    screen_health.add_argument("--horizon", type=int, default=30)

    research_therapy = subparsers.add_parser("research-therapy-settings", help="Run research-grade therapy setting analysis on prepared Tandem and Apple Health data")
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

    validate_therapy = subparsers.add_parser("validate-therapy-infra", help="Run synthetic truth-recovery validation for the therapy research infrastructure")
    validate_therapy.add_argument("--report-dir", default=None)
    validate_therapy.add_argument("--include-models", default=None, help="Comma-separated model families such as ridge,segmented_ridge,tree_boost,ensemble")
    validate_therapy.add_argument("--meal-proxy-mode", choices=["strict", "broad", "off"], default="strict")
    validate_therapy.add_argument("--ic-policy", choices=["exploratory_only", "conservative", "off"], default="exploratory_only")

    run = subparsers.add_parser("run", help="Run the full forecasting and recommendation pipeline")
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

    collect = subparsers.add_parser("collect", help="Fetch one Tandem Source window through tconnectsync")
    collect.add_argument("--start-date", default=None, help="Requested window start date (YYYY-MM-DD)")
    collect.add_argument("--end-date", default=None, help="Requested window end date (YYYY-MM-DD)")
    collect.add_argument("--manifest", default=None)
    collect.add_argument("--report", default=None)
    collect.add_argument("--env-file", default=None, help="Optional .env file with Tandem credentials")
    collect.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)

    backfill = subparsers.add_parser("backfill", help="Backfill Tandem Source windows through tconnectsync")
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


def _prepare_model_data(
    *,
    args,
    paths: ProjectPaths,
) -> ModelDataPreparationResult:
    warnings_list: list[str] = []
    raw_root = Path(args.raw)
    history_days = int(getattr(args, "history_days", 365))
    min_history_days = int(getattr(args, "min_history_days", 180))

    if getattr(args, "apple_input", None):
        try:
            import_health_auto_export_batch(args.apple_input, paths.ensure())
        except ValueError as exc:
            warnings_list.append(f"Apple Health import skipped: {exc}")

    tandem_before = load_tandem_exports(raw_root, include_health_auto_export=False)
    health_before = load_tandem_exports(raw_root, include_health_auto_export=True)
    tandem_span_before_start, tandem_span_before_end = summarize_tandem_data_span(tandem_before)
    apple_available = has_apple_health_data(health_before)
    apple_span_start, apple_span_end = summarize_apple_health_span(health_before)

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

    if needs_backfill and getattr(args, "skip_backfill", False):
        backfill_status = "skipped_by_flag"
        warnings_list.append("Requested Tandem backfill was skipped by flag; using currently available Tandem history.")
    elif needs_backfill:
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
                )
            backfill_status = "completed"
        except AcquisitionError as exc:
            backfill_status = "unavailable"
            warnings_list.append(f"Tandem backfill unavailable: {exc}")
        except Exception as exc:
            backfill_status = "failed"
            warnings_list.append(f"Tandem backfill failed: {exc}")

    tandem_after = load_tandem_exports(raw_root, include_health_auto_export=False)
    health_after = load_tandem_exports(raw_root, include_health_auto_export=True)
    export_manifest = build_export_manifest(health_after)
    write_export_manifest(export_manifest, paths.cloud_raw / "tandem_export_manifest.csv")

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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = ProjectPaths.from_root(args.root).ensure()

    if args.command in {"ingest", "run", "screen-health-features", "build-health-analysis-ready", "prepare-model-data", "research-therapy-settings"}:
        args.raw = args.raw or str(paths.cloud_raw)
        if args.command == "ingest":
            args.report = args.report or str(paths.reports / "coverage.md")
            args.manifest = args.manifest or str(paths.cloud_raw / "tandem_export_manifest.csv")
        elif args.command == "run":
            args.report = args.report or str(paths.reports / "run_summary.md")
            args.manifest = args.manifest or str(paths.cloud_raw / "tandem_export_manifest.csv")
        elif args.command == "prepare-model-data":
            args.output = args.output or str(paths.reports / "prepared_model_data_5min.csv")
            args.report = args.report or str(paths.reports / "model_data_preparation.md")
        elif args.command == "research-therapy-settings":
            args.report_dir = args.report_dir or str(paths.reports)
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

    if args.command == "ingest":
        data = load_tandem_exports(args.raw, include_health_auto_export=True)
        export_manifest = build_export_manifest(data)
        write_export_manifest(export_manifest, args.manifest)
        coverage = summarize_coverage(data)
        data_quality, quality_rows = assess_data_quality(args.raw, export_manifest=export_manifest)
        review_path = str(Path(args.report).with_name("coverage_review.html"))
        summary = build_run_summary(
            coverage=coverage,
            data_quality=data_quality,
            review_artifacts={"coverage_review_html": review_path},
        )
        write_markdown_report(summary, args.report)
        write_coverage_review_html(summary, quality_rows, review_path)
        return 0

    if args.command == "normalize-raw":
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
        import_health_auto_export_batch(args.input, paths.ensure())
        return 0

    if args.command == "prepare-model-data":
        preparation = _prepare_model_data(args=args, paths=paths)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        preparation.dataset.frame.to_csv(args.output, index=False)
        write_model_data_preparation_report(preparation, args.report)
        return 0

    if args.command == "research-therapy-settings":
        preparation = _prepare_model_data(args=args, paths=paths)
        result = run_therapy_research(
            preparation.dataset,
            segments=parse_therapy_segments(args.segments),
            include_models=parse_model_list(args.include_models),
            meal_proxy_mode=args.meal_proxy_mode,
            ic_policy=args.ic_policy,
        )
        write_therapy_research_artifacts(
            result,
            args.report_dir,
            write_source_report_cards=args.write_source_report_cards,
            write_research_gate=args.write_research_gate,
        )
        return 0

    if args.command == "validate-therapy-infra":
        result = validate_therapy_infra(
            meal_proxy_mode=args.meal_proxy_mode,
            ic_policy=args.ic_policy,
            include_models=parse_model_list(args.include_models or "ridge,segmented_ridge,tree_boost,ensemble"),
        )
        write_therapy_infra_validation_artifacts(result, args.report_dir)
        return 0

    if args.command == "build-health-analysis-ready":
        tandem_data = load_tandem_exports(args.raw, include_health_auto_export=False)
        health_data = load_tandem_exports(args.raw, include_health_auto_export=True)
        analysis_ready = build_prepared_model_dataset(
            tandem_data=tandem_data,
            health_data=health_data,
            config=FeatureConfig(horizon_minutes=args.horizon),
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        analysis_ready.frame.to_csv(args.output, index=False)
        return 0

    if args.command == "screen-health-features":
        tandem_data = load_tandem_exports(args.raw, include_health_auto_export=False)
        health_data = load_tandem_exports(args.raw, include_health_auto_export=True)
        screening = screen_health_features(tandem_data=tandem_data, health_data=health_data, horizon_minutes=args.horizon)
        Path(args.scores).parent.mkdir(parents=True, exist_ok=True)
        screening.scores.to_csv(args.scores, index=False)
        write_health_screening_report(screening, args.report)
        return 0

    if args.command == "run":
        tandem_data = load_tandem_exports(args.raw, include_health_auto_export=False)
        data = load_tandem_exports(args.raw, include_health_auto_export=True)
        prepared = build_prepared_model_dataset(
            tandem_data=tandem_data,
            health_data=data,
            config=FeatureConfig(horizon_minutes=args.horizon),
        )
        feature_frame = _build_feature_frame_from_prepared(prepared)
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
        if args.skip_recommendations:
            recommendation_policy = RecommendationPolicy(
                status="skipped",
                reasons=["skipped_by_flag"],
                validation_passed=False,
                sampler_passed=False,
                signal_passed=False,
            )
        else:
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
        review_path = str(Path(args.report).with_name("run_review.html"))
        summary = build_run_summary(
            coverage=coverage,
            walk_forward=walk_forward,
            recommendations=recommendations,
            fit_diagnostics=fit_diagnostics,
            data_quality=data_quality,
            recommendation_policy=recommendation_policy,
            review_artifacts={"run_review_html": review_path},
        )
        write_markdown_report(summary, args.report)
        json_path = Path(args.report).with_suffix(".json")
        write_json_report(summary, json_path)
        write_run_review_html(summary, review_path)
        return 0

    if args.command == "validate-raw":
        args.raw = args.raw or str(paths.cloud_raw)
        args.report = args.report or str(paths.reports / "tandem_raw_validation.md")
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
        credentials = load_tandem_credentials(args.root, args.env_file)
        manifest_path = args.manifest or str(paths.cloud_raw / "tandem_export_manifest.csv")
        report_path = args.report or str(paths.reports / "tandem_acquisition_summary.md")
        client_kwargs = {
            "region": credentials.region,
            "timezone": credentials.timezone,
            "pump_serial": credentials.pump_serial,
        }
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
            )
            return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
