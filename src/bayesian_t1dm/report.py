from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .ingest import TandemCoverage
from .quality import DataQualitySummary


def build_coverage_summary(
    *,
    coverage: TandemCoverage,
    data_quality: DataQualitySummary | None = None,
    review_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "coverage": asdict(coverage),
        "data_quality": asdict(data_quality) if data_quality is not None else None,
        "review_artifacts": dict(review_artifacts or {}),
    }


def _relative_link_target(markdown_path: Path, artifact_path: str | Path) -> str:
    try:
        return os.path.relpath(Path(artifact_path), start=markdown_path.parent)
    except Exception:
        return str(artifact_path)


def write_coverage_markdown(summary: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Source Coverage", "", "### Coverage"]
    for key, value in (summary.get("coverage") or {}).items():
        lines.append(f"- {key}: {value}")
    data_quality = summary.get("data_quality")
    if data_quality is not None:
        lines.append("")
        lines.append("### Data Quality")
        lines.append(f"- status: {data_quality.get('status')}")
        lines.append(f"- normalized_plausibility_status: {data_quality.get('normalized_plausibility_status')}")
        lines.append(f"- incomplete_window_count: {data_quality.get('incomplete_window_count')}")
        lines.append(f"- evaluation_touches_incomplete_windows: {data_quality.get('evaluation_touches_incomplete_windows')}")
        contributing = data_quality.get("contributing_window_ids") or []
        if contributing:
            lines.append(f"- contributing_window_ids: {', '.join(str(item) for item in contributing)}")
        reason_counts = data_quality.get("reason_counts") or {}
        if reason_counts:
            lines.append("- reason_counts: " + ", ".join(f"{key}={value}" for key, value in reason_counts.items()))
        plausibility_reasons = data_quality.get("normalized_plausibility_reasons") or {}
        if plausibility_reasons:
            lines.append("- normalized_plausibility_reasons: " + ", ".join(f"{key}={value}" for key, value in plausibility_reasons.items()))
    review_artifacts = summary.get("review_artifacts") or {}
    if review_artifacts:
        lines.append("")
        lines.append("### Review Artifacts")
        for name, artifact_path in sorted(review_artifacts.items()):
            lines.append(f"- [{name}]({_relative_link_target(path, artifact_path)})")
    path.write_text("\n".join(lines) + "\n")
    return path


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return str(value)
