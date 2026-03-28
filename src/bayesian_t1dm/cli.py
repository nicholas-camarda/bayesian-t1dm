from __future__ import annotations

import argparse
from pathlib import Path

from .evaluate import calibration_summary
from .features import FeatureConfig, build_feature_frame
from .ingest import load_tandem_exports, summarize_coverage
from .model import BayesianGlucoseModel
from .paths import ProjectPaths
from .recommend import recommend_setting_changes
from .report import build_run_summary, write_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bayesian-t1dm")
    parser.add_argument("--root", default=".", help="Project root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Load Tandem exports and write a coverage report")
    ingest.add_argument("--raw", default="data/raw")
    ingest.add_argument("--report", default="output/coverage.md")

    run = subparsers.add_parser("run", help="Run the full forecasting and recommendation pipeline")
    run.add_argument("--raw", default="data/raw")
    run.add_argument("--report", default="output/run_summary.md")
    run.add_argument("--horizon", type=int, default=30)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = ProjectPaths.from_root(args.root).ensure()

    if args.command == "ingest":
        data = load_tandem_exports(args.raw)
        coverage = summarize_coverage(data)
        summary = build_run_summary(coverage=coverage)
        write_markdown_report(summary, args.report)
        return 0

    if args.command == "run":
        data = load_tandem_exports(args.raw)
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

    parser.error(f"Unknown command: {args.command}")
    return 2
