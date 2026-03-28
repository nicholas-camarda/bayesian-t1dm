from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluate import CalibrationSummary
from .ingest import TandemCoverage
from .recommend import Recommendation


def build_run_summary(
    *,
    coverage: TandemCoverage,
    calibration: CalibrationSummary | None = None,
    recommendations: list[Recommendation] | None = None,
    model_name: str = "BayesianGlucoseModel",
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "coverage": asdict(coverage),
        "calibration": asdict(calibration) if calibration is not None else None,
        "recommendations": [asdict(rec) for rec in (recommendations or [])],
    }


def write_markdown_report(summary: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Bayesian T1DM Run Summary")
    lines.append("")
    lines.append("## Coverage")
    coverage = summary.get("coverage", {})
    for key, value in coverage.items():
        lines.append(f"- {key}: {value}")
    if summary.get("calibration"):
        lines.append("")
        lines.append("## Calibration")
        for key, value in summary["calibration"].items():
            lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Recommendations")
    for rec in summary.get("recommendations", []):
        lines.append(f"- {rec['setting']} {rec['direction']} by {rec['change_percent']:.1f}%: {rec['rationale']}")
    path.write_text("\n".join(lines) + "\n")
    return path
