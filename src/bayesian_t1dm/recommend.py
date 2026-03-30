from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .model import BayesianGlucoseModel, ModelFit, ScenarioForecast
from .features import recompute_scenario_features


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


def recommend_setting_changes(
    model: BayesianGlucoseModel,
    fit: ModelFit,
    feature_frame: pd.DataFrame,
    scenarios: Iterable[Scenario] | None = None,
    *,
    target_bg: float = 110.0,
    min_expected_gain_mgdl: float = 5.0,
) -> tuple[list[Recommendation], list[ScenarioForecast]]:
    scenarios = list(scenarios or [
        Scenario("basal_minus_10", basal_multiplier=0.9),
        Scenario("basal_plus_10", basal_multiplier=1.1),
        Scenario("bolus_minus_10", bolus_multiplier=0.9),
        Scenario("bolus_plus_10", bolus_multiplier=1.1),
        Scenario("icr_plus_10", icr_multiplier=1.1),
        Scenario("icr_minus_10", icr_multiplier=0.9),
    ])
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
        recommendations.append(
            Recommendation(
                setting=setting,
                direction=direction,
                change_percent=change_percent,
                expected_gain_mgdl=gain,
                posterior_probability_better=posterior_probability_better,
                rationale=f"{scenario.name} lowers expected post-meal deviation from target.",
                scenario_name=scenario.name,
            )
        )
    recommendations.sort(key=lambda rec: rec.expected_gain_mgdl, reverse=True)
    return recommendations, forecasts
