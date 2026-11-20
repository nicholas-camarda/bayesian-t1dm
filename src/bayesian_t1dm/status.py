from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any


from .health_auto_export import ModelDataPreparationResult
from .paths import ProjectPaths
from .report import _to_jsonable


TOP_LEVEL_ENTRYPOINTS = {
    "current_status.html",
    "latent_meal",
    "outcomes",
    "replay",
    "replay_predictor",
    "outcomes_model",
    "data_audit",
    "forecast_harness",
    "proposals",
    "refresh",
    "fixture",
    "source",
    "prepare",
}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 2
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def migrate_legacy_output_logs(paths: ProjectPaths) -> list[dict[str, str]]:
    legacy_logs_root = paths.reports / "logs"
    if not legacy_logs_root.exists():
        return []
    migrated: list[dict[str, str]] = []
    for command_dir in sorted(legacy_logs_root.iterdir(), key=lambda candidate: candidate.name):
        destination_command_dir = paths.logs / command_dir.name
        if command_dir.is_dir():
            destination_command_dir.mkdir(parents=True, exist_ok=True)
            for child in sorted(command_dir.iterdir(), key=lambda candidate: candidate.name):
                destination = _next_available_path(destination_command_dir / child.name)
                shutil.move(str(child), str(destination))
                migrated.append({"source": str(child), "destination": str(destination)})
        else:
            destination = _next_available_path(paths.logs / command_dir.name)
            shutil.move(str(command_dir), str(destination))
            migrated.append({"source": str(command_dir), "destination": str(destination)})
    shutil.rmtree(legacy_logs_root)
    return migrated


def cleanup_legacy_top_level_output(paths: ProjectPaths, *, keep_names: set[str] | None = None) -> Path | None:
    migrated_logs = migrate_legacy_output_logs(paths)
    keep = set(keep_names or set()) | TOP_LEVEL_ENTRYPOINTS
    extras = [item for item in sorted(paths.reports.iterdir(), key=lambda candidate: candidate.name) if item.name not in keep]
    if not extras and not migrated_logs:
        return None
    legacy_root = paths.legacy_output_archive
    legacy_root.mkdir(parents=True, exist_ok=True)
    moved: list[dict[str, str]] = []
    for item in extras:
        destination = _next_available_path(legacy_root / item.name)
        shutil.move(str(item), str(destination))
        moved.append({"source": str(item), "destination": str(destination)})
    manifest_path = legacy_root / "legacy_cleanup_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "moved": moved,
                "migrated_logs": migrated_logs,
                "moved_at": _utc_timestamp(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def reset_output_directory(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _source_quality_blockers(data_quality: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    blockers: list[dict[str, str]] = []
    if str(data_quality.get("status") or "broken") != "good":
        blockers.append(
            {
                "code": "degraded_source_quality",
                "label": "Degraded source quality",
                "detail": f"Source quality status is {data_quality.get('status')}.",
            }
        )
    return blockers, {
        "data_quality_status": data_quality.get("status"),
        "incomplete_window_count": data_quality.get("incomplete_window_count"),
    }


def derive_current_status(
    *,
    preparation: ModelDataPreparationResult,
    data_quality: dict[str, Any],
    run_id: str,
    generated_at: str | None = None,
    artifact_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """State of the data behind the analyses: source quality and the prepared dataset. It says nothing about settings."""
    generated_at = generated_at or _utc_timestamp()
    primary_blockers, source_details = _source_quality_blockers(data_quality)
    if primary_blockers:
        overall_state = "blocked"
        headline = "The source data are degraded."
        summary = "Incomplete or degraded source windows must be repaired before the analyses are read."
        next_actions = ["Repair incomplete or degraded source windows, then rebuild the dataset."]
    else:
        overall_state = "data_ready"
        headline = "The source data are complete."
        summary = "No source-window blocker is open."
        next_actions = ["Run refresh after materially new data arrive."]

    return {
        "overall_state": overall_state,
        "headline": headline,
        "summary": summary,
        "primary_blockers": primary_blockers,
        "next_actions": next_actions,
        "source_quality": source_details,
        "data_prep": {
            "mode": preparation.dataset.mode,
            "apple_available": preparation.apple_available,
            "overlap_start": _to_jsonable(preparation.overlap_start),
            "overlap_end": _to_jsonable(preparation.overlap_end),
            "final_dataset_start": _to_jsonable(preparation.final_dataset_start),
            "final_dataset_end": _to_jsonable(preparation.final_dataset_end),
            "final_row_count": int(preparation.final_row_count),
            "backfill_status": preparation.backfill_status,
            "health_feature_count": int(len(preparation.dataset.health_feature_columns)),
        },
        "artifacts": dict(artifact_paths or {}),
        "run_id": run_id,
        "generated_at": generated_at,
    }


def write_status_json(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def finalize_status_logs(paths: ProjectPaths, log_run_dir: Path | None) -> Path | None:
    if log_run_dir is None or not log_run_dir.exists():
        return None
    return log_run_dir
