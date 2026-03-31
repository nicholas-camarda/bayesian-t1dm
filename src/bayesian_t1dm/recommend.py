from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .evaluate import WalkForwardReport
from .model import BayesianGlucoseModel, FitDiagnostics, ModelFit, ScenarioForecast
from .features import recompute_scenario_features
from .quality import DataQualitySummary


@dataclass(frozen=True)
class Scenario:
    name: str
    basal_multiplier: float = 1.0
    bolus_multiplier: float = 1.0
    icr_multiplier: float = 1.0


@dataclass(frozen=True)
class Recommendation:
    setting: str
    direction: str
    change_percent: float
    expected_gain_mgdl: float
    posterior_probability_better: float
    rationale: str
    scenario_name: str
    confidence: str = "moderate"
    flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RecommendationPolicy:
    status: str
    reasons: list[str]
    validation_passed: bool
    sampler_passed: bool
    signal_passed: bool


def _apply_scenario(frame: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    out = frame.copy()
    if "basal_units_per_hour" in out.columns:
        out["basal_units_per_hour"] = out["basal_units_per_hour"] * scenario.basal_multiplier
    if "basal_units_delivered" in out.columns:
        out["basal_units_delivered"] = out["basal_units_delivered"] * scenario.basal_multiplier
    if "bolus_units" in out.columns:
        out["bolus_units"] = out["bolus_units"] * scenario.bolus_multiplier
        if scenario.icr_multiplier != 1.0:
            out["bolus_units"] = out["bolus_units"] * scenario.icr_multiplier
    return recompute_scenario_features(out)


def _scenario_setting(scenario: Scenario) -> str:
    if scenario.basal_multiplier != 1.0:
        return "basal"
    if scenario.bolus_multiplier != 1.0:
        return "bolus"
    return "I/C ratio"


def _dedupe_reasons(reasons: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason in seen:
            continue
        ordered.append(reason)
        seen.add(reason)
    return ordered


def build_recommendation_policy(
    *,
    walk_forward: WalkForwardReport | None,
    fit_diagnostics: FitDiagnostics | None,
    data_quality: DataQualitySummary | None,
    carbs_present: bool,
    activity_present: bool,
    skipped: bool = False,
) -> tuple[RecommendationPolicy, set[str]]:
    if skipped:
        return (
            RecommendationPolicy(
                status="skipped",
                reasons=["skipped_by_flag"],
                validation_passed=False,
                sampler_passed=False,
                signal_passed=False,
            ),
            set(),
        )

    reasons: list[str] = []
    validation_passed = True
    sampler_passed = True
    allowed_settings = {"basal", "bolus", "I/C ratio"}

    if walk_forward is None:
        validation_passed = False
        reasons.append("walk_forward_missing")
    else:
        if walk_forward.n_folds < 2:
            validation_passed = False
            reasons.append("walk_forward_insufficient_folds")
        if any(fold.n_test < 200 for fold in walk_forward.folds):
            validation_passed = False
            reasons.append("walk_forward_small_test_fold")
        aggregate = walk_forward.aggregate
        if aggregate.coverage < 0.70 or aggregate.coverage > 0.90:
            validation_passed = False
            reasons.append("coverage_out_of_range")
        if not np.isfinite(walk_forward.aggregate_persistence_mae) or aggregate.mae > 0.90 * walk_forward.aggregate_persistence_mae:
            validation_passed = False
            reasons.append("mae_not_meaningfully_better_than_persistence")
    if data_quality is not None and data_quality.status != "good":
        validation_passed = False
        reasons.append("data_incomplete")

    if fit_diagnostics is None:
        sampler_passed = False
        reasons.append("sampler_chains_lt_2")
    else:
        if fit_diagnostics.chains < 2:
            sampler_passed = False
            reasons.append("sampler_chains_lt_2")
        if fit_diagnostics.divergences > 0:
            sampler_passed = False
            reasons.append("sampler_divergences")
        if fit_diagnostics.max_tree_depth_hits > 0:
            sampler_passed = False
            reasons.append("sampler_max_treedepth_hits")
        if fit_diagnostics.rhat_max is not None and fit_diagnostics.rhat_max > 1.01:
            sampler_passed = False
            reasons.append("sampler_rhat_too_high")
        if (
            fit_diagnostics.ess_bulk_min is not None and fit_diagnostics.ess_bulk_min < 200
        ) or (
            fit_diagnostics.ess_tail_min is not None and fit_diagnostics.ess_tail_min < 200
        ):
            sampler_passed = False
            reasons.append("sampler_ess_too_low")

    if not carbs_present:
        allowed_settings.discard("bolus")
        allowed_settings.discard("I/C ratio")
        reasons.append("missing_carbs")
    if not activity_present:
        reasons.append("low_signal")

    signal_passed = bool(allowed_settings)
    status = "generated" if validation_passed and sampler_passed and signal_passed else "suppressed"
    return (
        RecommendationPolicy(
            status=status,
            reasons=_dedupe_reasons(reasons),
            validation_passed=validation_passed,
            sampler_passed=sampler_passed,
            signal_passed=signal_passed,
        ),
        allowed_settings,
    )


def _recommendation_flags(*, carbs_present: bool, activity_present: bool) -> list[str]:
    flags: list[str] = []
    if not activity_present:
        flags.append("low_signal")
    if not carbs_present:
        flags.append("missing_carbs")
    return flags


def _recommendation_confidence(
    posterior_probability_better: float,
    *,
    carbs_present: bool,
    activity_present: bool,
) -> str:
    if not carbs_present and not activity_present:
        return "low"
    if not activity_present:
        return "low"
    if posterior_probability_better >= 0.9:
        return "high"
    if posterior_probability_better >= 0.75:
        return "moderate"
    return "low"


def recommend_setting_changes(
    model: BayesianGlucoseModel,
    fit: ModelFit,
    feature_frame: pd.DataFrame,
    scenarios: Iterable[Scenario] | None = None,
    *,
    walk_forward: WalkForwardReport | None = None,
    data_quality: DataQualitySummary | None = None,
    carbs_present: bool = True,
    activity_present: bool = True,
    target_bg: float = 110.0,
    min_expected_gain_mgdl: float = 5.0,
) -> tuple[list[Recommendation], list[ScenarioForecast], RecommendationPolicy]:
    scenarios = list(scenarios or [
        Scenario("basal_minus_10", basal_multiplier=0.9),
        Scenario("basal_plus_10", basal_multiplier=1.1),
        Scenario("bolus_minus_10", bolus_multiplier=0.9),
        Scenario("bolus_plus_10", bolus_multiplier=1.1),
        Scenario("icr_plus_10", icr_multiplier=1.1),
        Scenario("icr_minus_10", icr_multiplier=0.9),
    ])
    policy, allowed_settings = build_recommendation_policy(
        walk_forward=walk_forward,
        fit_diagnostics=fit.diagnostics,
        data_quality=data_quality,
        carbs_present=carbs_present,
        activity_present=activity_present,
    )
    if policy.status != "generated":
        return [], [], policy

    scenarios = [scenario for scenario in scenarios if _scenario_setting(scenario) in allowed_settings]
    baseline_scenario = Scenario("current")
    forecast_inputs = [(baseline_scenario.name, _apply_scenario(feature_frame, baseline_scenario))]
    forecast_inputs.extend((scenario.name, _apply_scenario(feature_frame, scenario)) for scenario in scenarios)
    forecasts = model.scenario_forecasts(fit, forecast_inputs)
    baseline = forecasts[0].expected_loss if forecasts else np.nan
    recommendations: list[Recommendation] = []
    baseline_forecast = forecasts[0] if forecasts else None
    for scenario, forecast in zip(scenarios, forecasts[1:]):
        if not np.isfinite(forecast.expected_loss):
            continue
        gain = baseline - forecast.expected_loss
        if gain < min_expected_gain_mgdl:
            continue
        direction = "decrease" if scenario.basal_multiplier < 1 or scenario.bolus_multiplier < 1 or scenario.icr_multiplier < 1 else "increase"
        if scenario.basal_multiplier != 1.0:
            setting = "basal"
            change_percent = round(abs(scenario.basal_multiplier - 1.0) * 100.0, 6)
        elif scenario.bolus_multiplier != 1.0:
            setting = "bolus"
            change_percent = round(abs(scenario.bolus_multiplier - 1.0) * 100.0, 6)
        else:
            setting = "I/C ratio"
            change_percent = round(abs(scenario.icr_multiplier - 1.0) * 100.0, 6)
        if baseline_forecast is not None:
            current_loss = np.abs(baseline_forecast.mean - target_bg)
            candidate_loss = np.abs(forecast.mean - target_bg)
            posterior_probability_better = float(np.mean(candidate_loss < current_loss))
        else:
            posterior_probability_better = float("nan")
        flags = _recommendation_flags(carbs_present=carbs_present, activity_present=activity_present)
        recommendations.append(
            Recommendation(
                setting=setting,
                direction=direction,
                change_percent=change_percent,
                expected_gain_mgdl=gain,
                posterior_probability_better=posterior_probability_better,
                rationale=f"{scenario.name} lowers expected post-meal deviation from target.",
                scenario_name=scenario.name,
                confidence=_recommendation_confidence(
                    posterior_probability_better,
                    carbs_present=carbs_present,
                    activity_present=activity_present,
                ),
                flags=flags,
            )
        )
    recommendations.sort(key=lambda rec: rec.expected_gain_mgdl, reverse=True)
    return recommendations, forecasts, policy
