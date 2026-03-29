from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from .evaluate import calibration_summary
from .acquisition import (
    ExportWindow,
    PlaywrightTandemSourceClient,
    backfill_tandem_exports,
    collect_tandem_exports,
    load_tandem_credentials,
    StepLogger,
)
from .features import FeatureConfig, build_feature_frame
from .ingest import build_export_manifest, load_tandem_exports, summarize_coverage, write_export_manifest
from .model import BayesianGlucoseModel
from .paths import ProjectPaths
from .recommend import recommend_setting_changes
from .report import build_run_summary, write_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bayesian-t1dm")
    parser.add_argument("--root", default=".", help="Project root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Load Tandem exports and write a coverage report")
    ingest.add_argument("--raw", default=None)
    ingest.add_argument("--report", default=None)
    ingest.add_argument("--manifest", default=None)

    run = subparsers.add_parser("run", help="Run the full forecasting and recommendation pipeline")
    run.add_argument("--raw", default=None)
    run.add_argument("--report", default=None)
    run.add_argument("--manifest", default=None)
    run.add_argument("--horizon", type=int, default=30)

    collect = subparsers.add_parser("collect", help="Export one Tandem Source window through the browser")
    collect.add_argument("--start-date", default=None, help="Requested window start date (YYYY-MM-DD)")
    collect.add_argument("--end-date", default=None, help="Requested window end date (YYYY-MM-DD)")
    collect.add_argument("--manifest", default=None)
    collect.add_argument("--report", default=None)
    collect.add_argument("--page-map", default=None)
    collect.add_argument("--env-file", default=None, help="Optional .env file with Tandem credentials")
    collect.add_argument("--login-url", default=None)
    collect.add_argument("--daily-timeline-url", default=None)
    collect.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    collect.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)

    backfill = subparsers.add_parser("backfill", help="Backfill Tandem Source export windows through the browser")
    backfill.add_argument("--start-date", required=True, help="Earliest requested date (YYYY-MM-DD)")
    backfill.add_argument("--end-date", required=True, help="Latest requested date (YYYY-MM-DD)")
    backfill.add_argument("--window-days", type=int, default=30)
    backfill.add_argument("--direction", choices=["backward", "forward"], default="backward")
    backfill.add_argument("--manifest", default=None)
    backfill.add_argument("--report", default=None)
    backfill.add_argument("--page-map", default=None)
    backfill.add_argument("--env-file", default=None, help="Optional .env file with Tandem credentials")
    backfill.add_argument("--login-url", default=None)
    backfill.add_argument("--daily-timeline-url", default=None)
    backfill.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    backfill.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    backfill.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    discover = subparsers.add_parser("discover", help="Discover and save Tandem Source selectors")
    discover.add_argument("--manifest", default=None)
    discover.add_argument("--report", default=None)
    discover.add_argument("--page-map", default=None)
    discover.add_argument("--env-file", default=None, help="Optional .env file with Tandem credentials")
    discover.add_argument("--login-url", default=None)
    discover.add_argument("--daily-timeline-url", default=None)
    discover.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = ProjectPaths.from_root(args.root).ensure()

    if args.command in {"ingest", "run"}:
        args.raw = args.raw or str(paths.cloud_raw)
        args.report = args.report or str(paths.cloud_output / ("coverage.md" if args.command == "ingest" else "run_summary.md"))
        args.manifest = args.manifest or str(paths.cloud_raw / "tandem_export_manifest.csv")

    if args.command == "ingest":
        data = load_tandem_exports(args.raw)
        write_export_manifest(build_export_manifest(data), args.manifest)
        coverage = summarize_coverage(data)
        summary = build_run_summary(coverage=coverage)
        write_markdown_report(summary, args.report)
        return 0

    if args.command == "run":
        data = load_tandem_exports(args.raw)
        write_export_manifest(build_export_manifest(data), args.manifest)
        coverage = summarize_coverage(data)
        features = build_feature_frame(data, FeatureConfig(horizon_minutes=args.horizon))
        model = BayesianGlucoseModel(draws=250, tune=250, chains=1)
        fit = model.fit(features)
        predictions = model.predict(fit, features.frame)
        calibration = calibration_summary(
            features.frame[features.target_column].to_numpy(),
            predictions["mean"].to_numpy(),
            predictions["lower"].to_numpy(),
            predictions["upper"].to_numpy(),
        )
        recommendations, _ = recommend_setting_changes(model, fit, features.frame)
        summary = build_run_summary(coverage=coverage, calibration=calibration, recommendations=recommendations)
        write_markdown_report(summary, args.report)
        return 0

    if args.command in {"collect", "backfill"}:
        credentials = load_tandem_credentials(args.root, args.env_file)
        manifest_path = args.manifest or str(paths.cloud_raw / "tandem_export_manifest.csv")
        report_path = args.report or str(paths.cloud_output / "tandem_acquisition_summary.md")
        page_map_path = args.page_map or str(paths.cloud_archive / "tandem_page_map.json")
        with PlaywrightTandemSourceClient(
            paths,
            page_map_path=page_map_path,
            login_url=args.login_url or "https://source.tandemdiabetes.com/",
            daily_timeline_url=args.daily_timeline_url,
            headless=args.headless,
        ) as client:
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

    if args.command == "discover":
        credentials = load_tandem_credentials(args.root, args.env_file)
        manifest_path = args.manifest or str(paths.cloud_raw / "tandem_export_manifest.csv")
        report_path = args.report or str(paths.cloud_output / "tandem_discovery_summary.md")
        page_map_path = args.page_map or str(paths.cloud_archive / "tandem_page_map.json")
        step_log = StepLogger(paths.runtime_logs / "tandem_discovery.log")
        with PlaywrightTandemSourceClient(
            paths,
            page_map_path=page_map_path,
            login_url=args.login_url or "https://source.tandemdiabetes.com/",
            daily_timeline_url=args.daily_timeline_url,
            headless=args.headless,
        ) as client:
            page_map = client.discover_page_map(credentials, step_log=step_log)
            client.save_page_map(page_map, page_map_path)
            manifest_status = f"selector_map_written: {page_map_path}"
            report_lines = [
                "# Tandem Source Discovery Summary",
                "",
                f"- page_map_path: {page_map_path}",
                f"- login_url: {page_map.login_url}",
                f"- daily_timeline_url: {page_map.daily_timeline_url or 'NA'}",
                f"- manifest_path: {manifest_path}",
                f"- status: {manifest_status}",
            ]
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(report_path).write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
