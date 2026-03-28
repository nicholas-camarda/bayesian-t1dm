"""Bayesian T1DM forecasting and recommendation toolkit."""

from .evaluate import CalibrationSummary, WalkForwardSplit, calibration_summary, walk_forward_splits
from .features import FeatureConfig, FeatureFrame, build_feature_frame
from .ingest import IngestedData, TandemCoverage, load_tandem_exports, summarize_coverage
from .insulin import insulin_action_curve, expand_bolus_to_grid
from .model import BayesianGlucoseModel, ModelFit, ScenarioForecast
from .recommend import Recommendation, Scenario, recommend_setting_changes
from .report import build_run_summary, write_markdown_report

__all__ = [
    "BayesianGlucoseModel",
    "CalibrationSummary",
    "FeatureConfig",
    "FeatureFrame",
    "IngestedData",
    "ModelFit",
    "Recommendation",
    "Scenario",
    "ScenarioForecast",
    "TandemCoverage",
    "WalkForwardSplit",
    "build_feature_frame",
    "build_run_summary",
    "calibration_summary",
    "expand_bolus_to_grid",
    "insulin_action_curve",
    "load_tandem_exports",
    "recommend_setting_changes",
    "summarize_coverage",
    "walk_forward_splits",
    "write_markdown_report",
]
