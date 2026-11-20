from __future__ import annotations

import argparse
import dataclasses
import shutil
import json
import os
from contextlib import nullcontext
from datetime import date, timedelta
from pathlib import Path
import sys

import pandas as pd


from .acquisition import (
    load_local_env_file,
    ExportWindow,
    backfill_profile_edits,
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
)
from .latent_meal_validation import run_effective_icr_recovery, run_latent_meal_truth_recovery, write_latent_meal_recovery_artifacts
from .dataset import MergedDataset, build_merged_dataset, load_merged_dataset, merged_dir
from .ingest import summarize_tandem_raw_dir
from .observability import LoggingSession, setup_run_logging
from .analysis_data import AnalysisData, exercise_for, load_analysis_data, unlogged_for
from .outcomes_report import write_outcomes
from .controller import basal_signal, controller_inputs, programmed_units_by_slot
from .dashboard import build_dashboard_data, episodes_payload
from .dashboard_page import render_dashboard
from .changes import EVALUATION_DAYS, SETTING_FIELDS, detect_setting_changes
from .outcomes import day_part_table
from .result_flow import write_result_flow
from .front_page_page import write_front_page
from .layout import activate_current, assemble_run
from .publish import current_commit, publish_run, working_tree_is_clean
from .records import evaluate_open, load_records, register_proposal, save_records, sync_with_pump
from .proposals import propose_breakfast_icr, proposal_markdown, segment_ends_from_parts, segment_markdown, segment_summary
from .paths import ProjectPaths
from .report import build_coverage_summary, write_coverage_markdown
from .review import write_coverage_review_html, write_current_status_html, write_latent_meal_review_html
from .status import cleanup_legacy_top_level_output, derive_current_status, finalize_status_logs, reset_output_directory, write_status_json
from .therapy_research import (
    load_excluded_periods,
    build_representative_latent_meal_fixture,
    extract_latent_meal_therapy_settings_from_tandem_raw,
    load_latent_meal_therapy_settings,
    parse_therapy_segments,
    run_latent_meal_icr_research,
    write_latent_meal_research_artifacts,
    write_representative_latent_meal_fixture_artifacts,
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

    propose = add_command("propose-breakfast-icr", "Propose one small breakfast I:C change when the evidence, its checks and the low limits support it")
    propose.add_argument("--raw", default=None)
    propose.add_argument("--segments", default=None)
    propose.add_argument("--report-dir", default=None)
    propose.add_argument("--meal-proxy-mode", choices=["strict", "broad", "off"], default="strict")
    propose.add_argument("--research-scope", choices=["foundation", "full"], default="full")
    propose.add_argument("--therapy-settings", default=None)
    propose.add_argument("--meal-segments", default="morning,afternoon,evening", help="Therapy segments whose first meal of the day counts as evidence (breakfast, lunch, dinner)")
    propose.add_argument("--record", action=argparse.BooleanOptionalAction, default=False, help="Update the durable change record in the Project Vault (proposals, entered changes, evaluations)")

    refresh = add_command("refresh", "Run the whole pipeline by hand: collect, build, outcomes, proposal, change record, then publish to the Vault")
    refresh.add_argument("--env-file", default=None, help="Optional .env file with Tandem credentials")
    refresh.add_argument("--collect", action=argparse.BooleanOptionalAction, default=True, help="Fetch the latest 30 days from Tandem Source first")
    refresh.add_argument("--record", action=argparse.BooleanOptionalAction, default=True, help="Write evaluated changes and eligible proposals to the durable record")
    refresh.add_argument("--publish", action=argparse.BooleanOptionalAction, default=True)

    front = add_command("front-page", "Rewrite the front page (index.html) of a run folder from the research outputs, without a refresh")
    front.add_argument("--run-dir", default=None, help="Run folder to write into; defaults to runtime/current")

    screen_health = add_command("screen-health-features", "Screen imported Apple Health context features against Tandem glucose targets")
    screen_health.add_argument("--raw", default=None)
    screen_health.add_argument("--report", default=None)
    screen_health.add_argument("--scores", default=None)

    latent_meal = add_command("research-latent-meal-icr", "Run an experimental latent meal and I/C research workflow")
    latent_meal.add_argument("--raw", default=None)
    latent_meal.add_argument("--segments", default=None, help="Comma-separated day segments like overnight=00:00-06:00,morning=06:00-11:00")
    latent_meal.add_argument("--report-dir", default=None)
    latent_meal.add_argument("--review-html", default=None)
    latent_meal.add_argument("--meal-proxy-mode", choices=["strict", "broad", "off"], default="strict")
    latent_meal.add_argument("--research-scope", choices=["foundation", "full"], default="foundation")
    latent_meal.add_argument("--meal-segments", default="morning", help="Therapy segments whose first meal of the day counts as evidence")
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

    validate_latent = add_command("validate-latent-meal", "Run synthetic truth-recovery validation for the latent carb estimators")
    validate_latent.add_argument("--report-dir", default=None)
    validate_latent.add_argument("--meals", type=int, default=200, help="Simulated meals per scenario")
    validate_latent.add_argument("--seed", type=int, default=20260403)
    validate_latent.add_argument("--icr-cohorts", type=int, default=40, help="Simulated 20-meal cohorts per true I:C for the logged-meal recovery check; 0 skips it")

    status = add_command("status", "Report whether the source data windows are complete")
    status.add_argument("--raw", default=None)
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

    edits = add_command("fetch-profile-edits", "Fetch pump profile segment edits (event 68) for archived windows that lack them")
    edits.add_argument("--env-file", default=None, help="Optional .env file with Tandem credentials")

    return parser


def _session_stage(session: LoggingSession | None, stage: str, **fields: object):
    return session.stage(stage, **fields) if session is not None else nullcontext()


def _load_dataset(args, session: LoggingSession | None = None) -> MergedDataset:
    with _session_stage(session, "dataset.load", raw=args.raw):
        return load_merged_dataset(args.raw)


def _run_latent_meal_analysis(
    *,
    args,
    paths: ProjectPaths,
    session: LoggingSession | None = None,
    analysis: AnalysisData | None = None,
):
    therapy_settings = None
    if getattr(args, "therapy_settings", None):
        therapy_settings = load_latent_meal_therapy_settings(args.therapy_settings)
    elif getattr(args, "research_scope", "foundation") == "full":
        therapy_settings = extract_latent_meal_therapy_settings_from_tandem_raw(
            args.raw,
            segments=parse_therapy_segments(args.segments),
        )
    merged = analysis.merged if analysis is not None else _load_dataset(args, session)
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
            unlogged=analysis.unlogged if analysis is not None else unlogged_for(merged),
            exercise=analysis.exercise if analysis is not None else exercise_for(merged),
            programmed_units=programmed_units_by_slot(analysis) if analysis is not None else None,
            meal_segments=tuple(part.strip() for part in getattr(args, "meal_segments", "morning").split(",") if part.strip()),
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
    if args.command in {"ingest", "screen-health-features", "build-dataset", "research-latent-meal-icr", "status", "report-outcomes", "propose-breakfast-icr"}:
        args.raw = args.raw or str(paths.cloud_raw)
        if args.command == "ingest":
            args.report = args.report or str(paths.output_source / "coverage.md")
        elif args.command == "research-latent-meal-icr":
            args.report_dir = args.report_dir or str(paths.output_latent_meal)
            args.review_html = args.review_html or str(Path(args.report_dir) / "latent_meal_review.html")
        elif args.command == "report-outcomes":
            args.report_dir = args.report_dir or str(paths.output_outcomes)
        elif args.command == "propose-breakfast-icr":
            args.report_dir = args.report_dir or str(paths.output_proposals)
        elif args.command in {"status", "build-dataset"}:
            pass
        else:
            args.report = args.report or str(paths.output_prepare / "health_feature_screening.md")
            args.scores = args.scores or str(paths.cache_prepare / "health_feature_scores.csv")
    if args.command == "build-latent-meal-fixture":
        args.output_dir = args.output_dir or str(paths.output_fixture)
        args.review_html = args.review_html or str(Path(args.output_dir) / "latent_meal_review.html")
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
            summary = build_coverage_summary(
                coverage=coverage,
                data_quality=data_quality,
                review_artifacts={"coverage_review_html": review_path},
            )
            write_coverage_markdown(summary, args.report)
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

    if args.command == "refresh":
        return _run_refresh(args, paths, parser)

    if args.command == "front-page":
        run_dir = Path(args.run_dir) if args.run_dir else paths.current
        if not run_dir.is_dir():
            raise FileNotFoundError(f"{run_dir} is not a run folder; run refresh first or pass --run-dir")
        with session.stage("front_page.write", run_dir=str(run_dir)):
            print(f"Front page: {write_front_page(run_dir)}")
        return 0

    if args.command == "propose-breakfast-icr":
        merged = _load_dataset(args, session)
        analysis = load_analysis_data(args.raw, merged)
        _, result = _run_latent_meal_analysis(args=args, paths=paths, session=session, analysis=analysis)
        with session.stage("proposals.breakfast_icr", report_dir=args.report_dir):
            parts, _ = analysis.day_parts()
            now_settings = analysis.segment_settings()
            cgm = analysis.cgm
            as_of = pd.Timestamp(cgm["timestamp"].max()).normalize()
            proposal = propose_breakfast_icr(
                result.effective_icr_evidence, cgm, segment_ends_from_parts(parts), as_of=as_of,
                current_icr={start: item["icr"] for start, item in now_settings.items()},
            )
            out = Path(args.report_dir)
            out.mkdir(parents=True, exist_ok=True)
            table = segment_summary(
                parts, now_settings, day_part_table(cgm, parts, as_of=as_of, days=90), day_part_table(cgm, parts, as_of=as_of, days=14),
                result.effective_icr_evidence, pump_serial=proposal.pump_serial, profile_id=proposal.profile_id,
            )
            table.to_csv(out / "segment_summary.csv", index=False)
            result.effective_icr_evidence.to_csv(out / "effective_icr_evidence.csv", index=False)
            result.icr_support.to_csv(out / "icr_support.csv", index=False)
            change_log, proposal, records = _update_change_record(paths, analysis, proposal, parts, as_of, record=args.record)
            (out / "change_log.md").write_text(change_log, encoding="utf-8")
            pd.DataFrame([{"as_of": as_of.date(), **proposal.to_row()}]).to_csv(out / "breakfast_proposal.csv", index=False)
            current_rates = {part.name: now_settings[float(part.start_minutes)]["basal"] for part in parts if float(part.start_minutes) in now_settings}
            primary_flow, strict_flow = [], []
            controller_daily, rates, unlogged = controller_inputs(analysis, parts, flow=primary_flow)
            strict_daily, _, _ = controller_inputs(analysis, parts, strict=True, rates=rates, flow=strict_flow)
            pd.concat([pd.DataFrame(primary_flow).assign(rule="size-scaled"), pd.DataFrame(strict_flow).assign(rule="strict")],
                      ignore_index=True).to_csv(out / "controller_flow.csv", index=False)
            episode_data, classified_episodes = episodes_payload(
                cgm, analysis.grid, parts, analysis.site_events, rates, unlogged, analysis.exercise, merged.data.workouts, site_observed_from=analysis.site_observed_from(), as_of=as_of,
            )
            classified_episodes.drop(columns=["site_change"]).assign(site_change=classified_episodes["site_change"]).to_csv(out / "episodes.csv", index=False)
            if not controller_daily.empty:
                controller_daily.to_csv(out / "controller_daily.csv", index=False)
                basal_signal(controller_daily, current_rates).to_csv(out / "basal_signal.csv", index=False)
                if not strict_daily.empty:
                    basal_signal(strict_daily, current_rates).to_csv(out / "basal_signal_strict.csv", index=False)
            page = build_dashboard_data(
                cgm=cgm, parts=parts, settings_now=now_settings, segments=table, evidence=result.effective_icr_evidence, proposal_row=proposal.to_row(),
                records=records, controller_daily=controller_daily, controller_daily_strict=strict_daily, current_rates=current_rates, as_of=as_of,
                run={"run_id": f"{pd.Timestamp.now():%Y-%m-%d}_{current_commit(paths.root)[:8]}", "commit": current_commit(paths.root), "built": f"{pd.Timestamp.now():%Y-%m-%d %H:%M}"},
                pump_serial=proposal.pump_serial, profile_id=proposal.profile_id, episodes=episode_data,
            )
            (out / "dashboard.html").write_text(render_dashboard(page), encoding="utf-8")
            (out / "breakfast_proposal.md").write_text(segment_markdown(table) + "\n" + proposal_markdown(proposal, as_of=as_of) + ("\n" + change_log if change_log else ""), encoding="utf-8")
        print((out / "breakfast_proposal.md").read_text(encoding="utf-8"))
        return 0

    if args.command == "report-outcomes":
        merged = _load_dataset(args, session)
        with session.stage("outcomes.write", report_dir=args.report_dir):
            written = write_outcomes(load_analysis_data(args.raw, merged), args.report_dir)
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

    if args.command == "validate-latent-meal":
        with session.stage("latent_meal.validate_estimators", report_dir=args.report_dir, meals=args.meals, seed=args.seed):
            recovery = run_latent_meal_truth_recovery(n_meals_per_scenario=args.meals, seed=args.seed)
            icr_recovery = run_effective_icr_recovery(n_cohorts=args.icr_cohorts, seed=args.seed) if args.icr_cohorts > 0 else None
            recovery_paths = write_latent_meal_recovery_artifacts(recovery, args.report_dir, icr_recovery=icr_recovery)
        print(f"Latent meal truth recovery: {'pass' if recovery.passed() else 'FAIL'} -> {recovery_paths['report']}")
        return 0 if recovery.passed() else 1

    if args.command == "status":
        merged = _load_dataset(args, session)
        with session.stage("status.initialize", output_root=str(paths.reports)):
            _prepare_runtime_output(paths)
        with session.stage("status.write_status", output_root=str(paths.reports)):
            top_level_status_html = paths.reports / "current_status.html"
            cached_status_json = paths.cache_status / "current_status.json"
            artifact_paths = {
                "current_status_html": str(top_level_status_html),
                "current_status_json": str(cached_status_json),
                "merged_dataset_dir": str(merged_dir(args.raw)),
            }
            payload = derive_current_status(
                preparation=merged.preparation,
                data_quality=dataclasses.asdict(merged.data_quality),
                run_id=session.context.run_id,
                artifact_paths=artifact_paths,
            )
            write_current_status_html(payload, top_level_status_html)
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

    if args.command == "fetch-profile-edits":
        credentials = load_tandem_credentials(args.root, args.env_file)
        with session.stage("acquisition.profile_edits"):
            with TConnectSyncSourceClient(paths, region=credentials.region, timezone=credentials.timezone, pump_serial=credentials.pump_serial) as client:
                written = backfill_profile_edits(client, credentials, paths)
        print(f"Wrote {len(written)} profile edit files; run build-dataset next.")
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


def _update_change_record(paths: ProjectPaths, analysis: AnalysisData, proposal, parts, as_of: pd.Timestamp, *, record: bool):
    """Sync the durable record with the pump, register the proposal, run due evaluations, and return the change log
    with the proposal (held, with the reason, when earlier changes are still being evaluated)."""
    record_path = paths.cloud_root / "data" / "records" / "setting_proposals.csv"
    records = load_records(record_path)
    pump, profile = analysis.pump, analysis.profile_id
    changes = detect_setting_changes(analysis.merged.profile_events)
    # Only the profile now running: edits to a profile that is not in use do not change the person's glucose.
    changes = changes[changes["pump_serial"].eq(pump) & changes["profile_id"].eq(profile)]
    parts_by_start = {float(part.start_minutes): part for part in parts}
    # Changes are looked at from 30 days back so a new record also picks up edits made since the last refresh.
    records = sync_with_pump(records, changes, parts_by_start, since=as_of - pd.Timedelta(days=30))
    records = evaluate_open(records, analysis.cgm, parts_by_start, now=as_of, excluded_days=analysis.excluded_days)
    updated, note = register_proposal(records, proposal, as_of=as_of)
    if proposal.status == "proposed" and note.startswith("not recorded"):
        entered = pd.to_datetime(records.loc[records["status"].eq("entered"), "entered_on"]).dropna()
        timing = f"the first can be judged on {(entered.min().normalize() + pd.Timedelta(days=EVALUATION_DAYS + 1)).date()}" if len(entered) else "an earlier proposal is awaiting entry"
        proposal = dataclasses.replace(
            proposal, status="held",
            reason=f"waiting: earlier changes are still being evaluated ({timing}). The evidence would otherwise support {proposal.current_value:.1f} to {proposal.proposed_value:.1f} g/U "
                   f"for the segment starting {int(proposal.segment_start_minutes) // 60:02d}:{int(proposal.segment_start_minutes) % 60:02d}",
        )
    if record:
        records = updated
        save_records(records, record_path)
    elif note == "recorded":
        note = "preview only"
    lines = ["## Change log", "", f"Record: `{record_path}` ({note}).", ""]
    if records.empty:
        lines.append("No setting changes recorded yet.")
    else:
        lines += ["| made | setting | segment | change | origin | other changes within 24 h | status | result |", "| --- | --- | --- | --- | --- | ---: | --- | --- |"]
        for row in records.sort_values("entered_on", na_position="last").itertuples(index=False):
            made = row.entered_on if pd.notna(row.entered_on) else f"proposed {row.proposed_on}"
            segment = f"{int(row.segment_start_minutes) // 60:02d}:{int(row.segment_start_minutes) % 60:02d}"
            lines.append(f"| {made} | {SETTING_FIELDS.get(row.setting, row.setting)} | {segment} | {row.old_value:g} -> {row.new_value:g} | {row.origin} | {int(row.concurrent_changes) if pd.notna(row.concurrent_changes) else 0} | {row.status} | {row.result if pd.notna(row.result) else ''} |")
    return "\n".join(lines) + "\n", proposal, records


def _run_refresh(args, paths: ProjectPaths, parser) -> int:
    """Collect, build, report, propose, record and publish; the run record says whether every step completed."""
    run_id = f"{pd.Timestamp.now():%Y-%m-%d}_{current_commit(paths.root)[:8]}"
    run_dir = paths.runs / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    record = {"run_id": run_id, "commit": current_commit(paths.root), "working_tree_clean": working_tree_is_clean(paths.root), "started": f"{pd.Timestamp.now():%Y-%m-%dT%H:%M:%S}", "steps": {}, "status": "running"}
    base = ["--root", str(args.root)]
    steps = []
    if args.collect:
        steps.append(("collect", ["collect", *(["--env-file", args.env_file] if args.env_file else [])]))
    steps += [("build-dataset", ["build-dataset"]), ("report-outcomes", ["report-outcomes"]), ("propose-breakfast-icr", ["propose-breakfast-icr", *(["--record"] if args.record else [])])]
    for name, argv in steps:
        code = main([*base, *argv])
        record["steps"][name] = code
        if code != 0:
            record["status"] = f"failed at {name}"
            (run_dir / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"Refresh stopped: {name} exited with {code}. Nothing was published.")
            return code or 1
    assemble_run(paths.reports, paths.cloud_raw, run_dir, record_path=paths.cloud_root / "data" / "records" / "setting_proposals.csv")
    write_front_page(run_dir)
    write_result_flow(run_dir, paths.cloud_raw)
    record["status"] = "complete"
    record["finished"] = f"{pd.Timestamp.now():%Y-%m-%dT%H:%M:%S}"
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    activate_current(run_dir, paths.current)
    print(f"Run {run_id} complete. Open {paths.current / 'index.html'}")
    if args.publish:
        destination = publish_run(run_dir, paths.cloud_output / f"{pd.Timestamp.now():%Y-%m-%d}", repo_root=paths.root)
        print(f"Published to {destination}")
    return 0


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
