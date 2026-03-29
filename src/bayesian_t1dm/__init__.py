"""Bayesian T1DM forecasting and recommendation toolkit."""

from .acquisition import (
    AcquisitionError,
    AcquisitionRecord,
    ExportWindow,
    PlaywrightTandemSourceClient,
    SelectorConfig,
    StepLogger,
    TandemCredentials,
    backfill_tandem_exports,
    collect_tandem_exports,
    export_daily_timeline_window,
    generate_export_windows,
    load_local_env_file,
    load_tandem_credentials,
    login_tandem_source,
)
from .evaluate import CalibrationSummary, WalkForwardSplit, calibration_summary, walk_forward_splits
from .features import FeatureConfig, FeatureFrame, build_feature_frame
from .ingest import IngestedData, TandemCoverage, build_export_manifest, load_tandem_exports, summarize_coverage, summarize_export_manifest, write_export_manifest
from .insulin import insulin_action_curve, expand_bolus_to_grid
from .model import BayesianGlucoseModel, ModelFit, ScenarioForecast
from .recommend import Recommendation, Scenario, recommend_setting_changes
from .report import build_run_summary, write_markdown_report

__all__ = [
    "BayesianGlucoseModel",
    "AcquisitionError",
    "AcquisitionRecord",
    "CalibrationSummary",
    "FeatureConfig",
    "FeatureFrame",
    "ExportWindow",
    "IngestedData",
    "ModelFit",
    "PlaywrightTandemSourceClient",
    "Recommendation",
    "SelectorConfig",
    "StepLogger",
    "Scenario",
    "ScenarioForecast",
    "TandemCoverage",
    "TandemCredentials",
    "WalkForwardSplit",
    "backfill_tandem_exports",
    "collect_tandem_exports",
    "build_feature_frame",
    "build_export_manifest",
    "build_run_summary",
    "export_daily_timeline_window",
    "calibration_summary",
    "generate_export_windows",
    "expand_bolus_to_grid",
    "load_local_env_file",
    "insulin_action_curve",
    "load_tandem_exports",
    "load_tandem_credentials",
    "login_tandem_source",
    "recommend_setting_changes",
    "summarize_coverage",
    "summarize_export_manifest",
    "walk_forward_splits",
    "write_markdown_report",
    "write_export_manifest",
]
