from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .evaluate import walk_forward_splits
from .features import FeatureConfig, build_feature_frame
from .ingest import IngestedData
from .io import read_table
from .paths import ProjectPaths


SUPPORTED_POINT_METRICS = {
    "step_count",
    "heart_rate_variability",
    "respiratory_rate",
    "resting_heart_rate",
    "weight_body_mass",
}
SUPPORTED_INTERVAL_METRICS = {"sleep_analysis"}
SUPPORTED_METRICS = SUPPORTED_POINT_METRICS | SUPPORTED_INTERVAL_METRICS | {"heart_rate"}
IGNORED_GLUCOSE_METRICS = {"blood_glucose"}
EXPORT_ID_PATTERN = re.compile(r"HealthAutoExport_(\d{14})$")
EXPORT_JSON_PATTERN = re.compile(r"HealthAutoExport-(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})\.json$")


@dataclass(frozen=True)
class HealthAutoExportBundle:
    export_id: str
    input_dir: Path
    json_path: Path
    gpx_paths: tuple[Path, ...]
    bundle_timestamp: pd.Timestamp | None
    source_json_filename: str
    covered_start_date: pd.Timestamp | None
    covered_end_date: pd.Timestamp | None


@dataclass(frozen=True)
class HealthAutoExportImportResult:
    export_id: str
    raw_json_path: Path
    raw_gpx_paths: tuple[Path, ...]
    normalized_paths: dict[str, str]
    row_counts: dict[str, int]
    ignored_metrics: tuple[str, ...]
    glucose_present: bool
    manifest_path: Path
    bundle_timestamp: pd.Timestamp | None
    source_json_filename: str
    covered_start_date: pd.Timestamp | None
    covered_end_date: pd.Timestamp | None


@dataclass(frozen=True)
class AnalysisReadyHealthDataset:
    frame: pd.DataFrame
    tandem_feature_columns: list[str]
    health_feature_columns: list[str]
    target_column: str
    config: FeatureConfig


@dataclass(frozen=True)
class HealthFeatureScreeningResult:
    scores: pd.DataFrame
    baseline_rmse_mean: float
    augmented_rmse_mean: float
    baseline_mae_mean: float
    augmented_mae_mean: float
    split_count: int


def _parse_timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if isinstance(parsed, pd.Timestamp) and parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return parsed


def _parse_series(series: pd.Series, *, floor: str | None = None) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    if floor is not None:
        parsed = parsed.dt.floor(floor)
    return parsed


def _safe_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _infer_bundle_timestamp(export_id: str) -> pd.Timestamp | None:
    match = EXPORT_ID_PATTERN.fullmatch(export_id)
    if not match:
        return None
    parsed = pd.to_datetime(match.group(1), format="%Y%m%d%H%M%S", errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _infer_covered_range(source_json_filename: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    match = EXPORT_JSON_PATTERN.fullmatch(source_json_filename)
    if not match:
        return None, None
    start = pd.to_datetime(match.group(1), errors="coerce")
    end = pd.to_datetime(match.group(2), errors="coerce")
    start_value = None if pd.isna(start) else pd.Timestamp(start)
    end_value = None if pd.isna(end) else pd.Timestamp(end)
    return start_value, end_value


def _discover_bundle_dirs(path: str | Path) -> list[Path]:
    input_path = Path(path).expanduser().resolve()
    if input_path.is_file():
        input_path = input_path.parent
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Health Auto Export input directory does not exist: {input_path}")

    direct_json_paths = sorted(input_path.glob("HealthAutoExport-*.json"))
    if direct_json_paths:
        return [input_path]

    child_dirs = sorted(
        child
        for child in input_path.iterdir()
        if child.is_dir() and child.name.startswith("HealthAutoExport_") and any(child.glob("HealthAutoExport-*.json"))
    )
    if child_dirs:
        return child_dirs
    raise ValueError(f"No HealthAutoExport bundle directories were found under {input_path}")


def discover_health_auto_export_bundle(path: str | Path) -> HealthAutoExportBundle:
    bundle_dirs = _discover_bundle_dirs(path)
    if len(bundle_dirs) != 1:
        raise ValueError(f"Expected exactly one Health Auto Export bundle directory, found {len(bundle_dirs)} under {path}")

    input_dir = bundle_dirs[0]
    json_paths = sorted(input_dir.glob("HealthAutoExport-*.json"))
    if len(json_paths) != 1:
        raise ValueError(f"Expected exactly one HealthAutoExport-*.json file in {input_dir}, found {len(json_paths)}")

    gpx_paths = tuple(sorted(input_dir.glob("*.gpx")))
    source_json_filename = json_paths[0].name
    covered_start_date, covered_end_date = _infer_covered_range(source_json_filename)
    return HealthAutoExportBundle(
        export_id=input_dir.name,
        input_dir=input_dir,
        json_path=json_paths[0],
        gpx_paths=gpx_paths,
        bundle_timestamp=_infer_bundle_timestamp(input_dir.name),
        source_json_filename=source_json_filename,
        covered_start_date=covered_start_date,
        covered_end_date=covered_end_date,
    )


def discover_health_auto_export_bundles(path: str | Path) -> tuple[HealthAutoExportBundle, ...]:
    return tuple(discover_health_auto_export_bundle(bundle_dir) for bundle_dir in _discover_bundle_dirs(path))


def _copy_artifact(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _normalize_point_metric(metric_name: str, units: str | None, rows: Iterable[dict[str, Any]], *, source_file: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        timestamp = _parse_timestamp(row.get("date"))
        value = _safe_float(row.get("qty"))
        if pd.isna(timestamp) or value is None:
            continue
        records.append(
            {
                "timestamp": timestamp,
                "metric": metric_name,
                "stat": "value",
                "value": value,
                "unit": units,
                "source_file": source_file,
                "source_device": str(row.get("source") or ""),
            }
        )
    return pd.DataFrame(records)


def _normalize_heart_rate_metric(rows: Iterable[dict[str, Any]], *, units: str | None, source_file: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        timestamp = _parse_timestamp(row.get("date"))
        if pd.isna(timestamp):
            continue
        for input_key, stat in [("Avg", "avg"), ("Min", "min"), ("Max", "max")]:
            value = _safe_float(row.get(input_key))
            if value is None:
                continue
            records.append(
                {
                    "timestamp": timestamp,
                    "metric": "heart_rate",
                    "stat": stat,
                    "value": value,
                    "unit": units,
                    "source_file": source_file,
                    "source_device": str(row.get("source") or ""),
                }
            )
    return pd.DataFrame(records)


def _normalize_sleep_metric(rows: Iterable[dict[str, Any]], *, source_file: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        date = _parse_timestamp(row.get("date"))
        sleep_start = _parse_timestamp(row.get("sleepStart"))
        sleep_end = _parse_timestamp(row.get("sleepEnd"))
        in_bed_start = _parse_timestamp(row.get("inBedStart"))
        in_bed_end = _parse_timestamp(row.get("inBedEnd"))
        if pd.isna(date):
            continue
        records.append(
            {
                "date": date.normalize(),
                "sleep_start": sleep_start,
                "sleep_end": sleep_end,
                "in_bed_start": in_bed_start,
                "in_bed_end": in_bed_end,
                "total_sleep_hours": _safe_float(row.get("totalSleep")),
                "in_bed_hours": _safe_float(row.get("inBed")),
                "core_hours": _safe_float(row.get("core")),
                "deep_hours": _safe_float(row.get("deep")),
                "rem_hours": _safe_float(row.get("rem")),
                "awake_hours": _safe_float(row.get("awake")),
                "source_file": source_file,
                "source_device": str(row.get("source") or ""),
            }
        )
    return pd.DataFrame(records)


def _normalize_activity_from_measurements(measurements: pd.DataFrame, *, source_file: str) -> pd.DataFrame:
    if measurements.empty:
        return pd.DataFrame(columns=["timestamp", "activity_value", "source_file", "source_device"])
    steps = measurements.loc[measurements["metric"] == "step_count", ["timestamp", "value", "source_device"]].copy()
    if steps.empty:
        return pd.DataFrame(columns=["timestamp", "activity_value", "source_file", "source_device"])
    steps["timestamp"] = pd.to_datetime(steps["timestamp"], errors="coerce").dt.floor("5min")
    steps["activity_value"] = pd.to_numeric(steps["value"], errors="coerce")
    steps["source_file"] = source_file
    return (
        steps.dropna(subset=["timestamp", "activity_value"])
        .groupby(["timestamp", "source_file", "source_device"], as_index=False)
        .agg(activity_value=("activity_value", "sum"))
    )


def _normalize_workouts(rows: Iterable[dict[str, Any]], *, source_file: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        distance = row.get("distance") if isinstance(row.get("distance"), dict) else {}
        energy = row.get("activeEnergyBurned") if isinstance(row.get("activeEnergyBurned"), dict) else {}
        avg_hr = row.get("avgHeartRate") if isinstance(row.get("avgHeartRate"), dict) else {}
        max_hr = row.get("maxHeartRate") if isinstance(row.get("maxHeartRate"), dict) else {}
        start = _parse_timestamp(row.get("start"))
        end = _parse_timestamp(row.get("end"))
        if pd.isna(start) or pd.isna(end):
            continue
        records.append(
            {
                "workout_id": str(row.get("id") or ""),
                "workout_type": str(row.get("name") or ""),
                "start_timestamp": start,
                "end_timestamp": end,
                "duration_seconds": _safe_float(row.get("duration")),
                "distance_value": _safe_float(distance.get("qty")),
                "distance_unit": distance.get("units"),
                "active_energy_value": _safe_float(energy.get("qty")),
                "active_energy_unit": energy.get("units"),
                "avg_heart_rate": _safe_float(avg_hr.get("qty")),
                "max_heart_rate": _safe_float(max_hr.get("qty")),
                "source_file": source_file,
            }
        )
    return pd.DataFrame(records)


def _bundle_metadata_columns() -> list[str]:
    return [
        "export_id",
        "export_timestamp",
        "source_json_filename",
        "covered_start_date",
        "covered_end_date",
    ]


def _empty_health_tables() -> dict[str, pd.DataFrame]:
    metadata_columns = _bundle_metadata_columns()
    return {
        "health_activity": pd.DataFrame(columns=["timestamp", "activity_value", "source_file", "source_device", *metadata_columns]),
        "health_measurements": pd.DataFrame(
            columns=["timestamp", "metric", "stat", "value", "unit", "source_file", "source_device", *metadata_columns]
        ),
        "sleep": pd.DataFrame(
            columns=[
                "date",
                "sleep_start",
                "sleep_end",
                "in_bed_start",
                "in_bed_end",
                "total_sleep_hours",
                "in_bed_hours",
                "core_hours",
                "deep_hours",
                "rem_hours",
                "awake_hours",
                "source_file",
                "source_device",
                *metadata_columns,
            ]
        ),
        "workouts": pd.DataFrame(
            columns=[
                "workout_id",
                "workout_type",
                "start_timestamp",
                "end_timestamp",
                "duration_seconds",
                "distance_value",
                "distance_unit",
                "active_energy_value",
                "active_energy_unit",
                "avg_heart_rate",
                "max_heart_rate",
                "source_file",
                *metadata_columns,
            ]
        ),
    }


def _load_payload(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _attach_bundle_metadata(frame: pd.DataFrame, bundle: HealthAutoExportBundle) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["export_id"] = bundle.export_id
    out["export_timestamp"] = bundle.bundle_timestamp
    out["source_json_filename"] = bundle.source_json_filename
    out["covered_start_date"] = bundle.covered_start_date
    out["covered_end_date"] = bundle.covered_end_date
    return out


def _normalize_health_payload(payload: dict[str, Any], *, source_file: str) -> tuple[dict[str, pd.DataFrame], tuple[str, ...], bool]:
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("Health Auto Export payload is missing a top-level data object")

    metrics = data.get("metrics")
    if not isinstance(metrics, list):
        raise ValueError("Health Auto Export payload metrics must be a list")

    tables = _empty_health_tables()
    measurement_frames: list[pd.DataFrame] = []
    sleep_frames: list[pd.DataFrame] = []
    ignored_metrics: list[str] = []
    glucose_present = False

    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        metric_name = str(metric.get("name") or "")
        units = metric.get("units")
        rows = metric.get("data")
        if not isinstance(rows, list):
            ignored_metrics.append(metric_name)
            continue
        if metric_name in IGNORED_GLUCOSE_METRICS:
            glucose_present = True
            continue
        if metric_name in SUPPORTED_POINT_METRICS:
            measurement_frames.append(_normalize_point_metric(metric_name, units, rows, source_file=source_file))
        elif metric_name == "heart_rate":
            measurement_frames.append(_normalize_heart_rate_metric(rows, units=units, source_file=source_file))
        elif metric_name == "sleep_analysis":
            sleep_frames.append(_normalize_sleep_metric(rows, source_file=source_file))
        else:
            ignored_metrics.append(metric_name)

    if measurement_frames:
        measurements = pd.concat(measurement_frames, ignore_index=True, sort=False)
        measurements["timestamp"] = _parse_series(measurements["timestamp"])
        measurements["value"] = pd.to_numeric(measurements["value"], errors="coerce")
        measurements = measurements.dropna(subset=["timestamp", "value"]).reset_index(drop=True)
        tables["health_measurements"] = measurements
        tables["health_activity"] = _normalize_activity_from_measurements(measurements, source_file=source_file)

    if sleep_frames:
        sleep = pd.concat(sleep_frames, ignore_index=True, sort=False)
        for column in ["date", "sleep_start", "sleep_end", "in_bed_start", "in_bed_end"]:
            sleep[column] = _parse_series(sleep[column])
        tables["sleep"] = sleep.reset_index(drop=True)

    workouts = data.get("workouts")
    if isinstance(workouts, list):
        tables["workouts"] = _normalize_workouts(workouts, source_file=source_file).reset_index(drop=True)

    return tables, tuple(sorted(set(metric for metric in ignored_metrics if metric))), glucose_present


def import_health_auto_export(path: str | Path, workspace: ProjectPaths) -> HealthAutoExportImportResult:
    bundle = discover_health_auto_export_bundle(path)
    archive_root = workspace.cloud_raw / "health_auto_export" / bundle.export_id
    raw_root = archive_root / "raw"
    normalized_root = archive_root / "normalized"
    raw_root.mkdir(parents=True, exist_ok=True)
    normalized_root.mkdir(parents=True, exist_ok=True)

    raw_json_path = _copy_artifact(bundle.json_path, raw_root / bundle.json_path.name)
    raw_gpx_paths = tuple(_copy_artifact(gpx_path, raw_root / gpx_path.name) for gpx_path in bundle.gpx_paths)

    payload = _load_payload(bundle.json_path)
    tables, ignored_metrics, glucose_present = _normalize_health_payload(payload, source_file=bundle.json_path.name)
    tables = {kind: _attach_bundle_metadata(frame, bundle) for kind, frame in tables.items()}

    normalized_paths: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for kind, frame in tables.items():
        if frame.empty:
            continue
        filename = "activity.csv" if kind == "health_activity" else f"{kind}.csv"
        output_path = normalized_root / filename
        frame.to_csv(output_path, index=False)
        normalized_paths[kind] = str(output_path)
        row_counts[kind] = int(len(frame))

    manifest_rows: list[dict[str, Any]] = [
        {
            "export_id": bundle.export_id,
            "kind": "raw_json",
            "raw_path": str(raw_json_path),
            "normalized_path": "",
            "row_count": 0,
            "ignored_metrics": json.dumps(list(ignored_metrics)),
            "glucose_present": glucose_present,
            "bundle_timestamp": bundle.bundle_timestamp,
            "source_json_filename": bundle.source_json_filename,
            "covered_start_date": bundle.covered_start_date,
            "covered_end_date": bundle.covered_end_date,
        }
    ]
    manifest_rows.extend(
        {
            "export_id": bundle.export_id,
            "kind": "raw_gpx",
            "raw_path": str(path),
            "normalized_path": "",
            "row_count": 0,
            "ignored_metrics": json.dumps(list(ignored_metrics)),
            "glucose_present": glucose_present,
            "bundle_timestamp": bundle.bundle_timestamp,
            "source_json_filename": bundle.source_json_filename,
            "covered_start_date": bundle.covered_start_date,
            "covered_end_date": bundle.covered_end_date,
        }
        for path in raw_gpx_paths
    )
    manifest_rows.extend(
        {
            "export_id": bundle.export_id,
            "kind": kind,
            "raw_path": str(raw_json_path),
            "normalized_path": normalized_path,
            "row_count": row_counts[kind],
            "ignored_metrics": json.dumps(list(ignored_metrics)),
            "glucose_present": glucose_present,
            "bundle_timestamp": bundle.bundle_timestamp,
            "source_json_filename": bundle.source_json_filename,
            "covered_start_date": bundle.covered_start_date,
            "covered_end_date": bundle.covered_end_date,
        }
        for kind, normalized_path in normalized_paths.items()
    )
    manifest_path = archive_root / "health_auto_export_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    return HealthAutoExportImportResult(
        export_id=bundle.export_id,
        raw_json_path=raw_json_path,
        raw_gpx_paths=raw_gpx_paths,
        normalized_paths=normalized_paths,
        row_counts=row_counts,
        ignored_metrics=ignored_metrics,
        glucose_present=glucose_present,
        manifest_path=manifest_path,
        bundle_timestamp=bundle.bundle_timestamp,
        source_json_filename=bundle.source_json_filename,
        covered_start_date=bundle.covered_start_date,
        covered_end_date=bundle.covered_end_date,
    )


def import_health_auto_export_batch(path: str | Path, workspace: ProjectPaths) -> tuple[HealthAutoExportImportResult, ...]:
    return tuple(import_health_auto_export(bundle.input_dir, workspace) for bundle in discover_health_auto_export_bundles(path))


def _coerce_manifest_timestamp(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.Timestamp) and parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return pd.Timestamp(parsed)


def _attach_manifest_metadata(frame: pd.DataFrame, row: dict[str, Any]) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    metadata_defaults = {
        "export_id": str(row.get("export_id") or ""),
        "export_timestamp": _coerce_manifest_timestamp(row.get("bundle_timestamp")),
        "source_json_filename": str(row.get("source_json_filename") or ""),
        "covered_start_date": _coerce_manifest_timestamp(row.get("covered_start_date")),
        "covered_end_date": _coerce_manifest_timestamp(row.get("covered_end_date")),
    }
    for column, value in metadata_defaults.items():
        if column not in out.columns:
            out[column] = value
        else:
            out[column] = out[column].where(out[column].notna(), value)
    return out


def _canonicalize_health_table_types(kind: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ["timestamp", "date", "sleep_start", "sleep_end", "in_bed_start", "in_bed_end", "start_timestamp", "end_timestamp", "export_timestamp", "covered_start_date", "covered_end_date"]:
        if column in out.columns:
            out[column] = _parse_series(out[column])
    for column in [
        "activity_value",
        "value",
        "total_sleep_hours",
        "in_bed_hours",
        "core_hours",
        "deep_hours",
        "rem_hours",
        "awake_hours",
        "duration_seconds",
        "distance_value",
        "active_energy_value",
        "avg_heart_rate",
        "max_heart_rate",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _dedupe_workouts(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["_workout_dedupe_key"] = np.where(
        out["workout_id"].astype(str).str.strip().ne(""),
        "id::" + out["workout_id"].astype(str),
        "fallback::"
        + out["start_timestamp"].astype(str)
        + "::"
        + out["workout_type"].astype(str)
        + "::"
        + out["source_file"].astype(str),
    )
    out = out.drop_duplicates(subset=["_workout_dedupe_key"], keep="last")
    return out.drop(columns=["_workout_dedupe_key"])


def _dedupe_canonical_health_table(kind: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "export_timestamp" in out.columns:
        out["export_timestamp"] = _parse_series(out["export_timestamp"])
    sort_columns = [column for column in ["export_timestamp", "export_id", "source_json_filename", "source_file"] if column in out.columns]
    if sort_columns:
        out = out.sort_values(sort_columns, kind="stable", na_position="first")
    if kind == "health_measurements":
        subset = ["timestamp", "metric", "stat", "source_device"]
        return out.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
    if kind == "health_activity":
        subset = ["timestamp", "source_device"]
        return out.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
    if kind == "sleep":
        subset = ["date", "source_device"]
        return out.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
    if kind == "workouts":
        return _dedupe_workouts(out).reset_index(drop=True)
    return out.reset_index(drop=True)


def load_unified_health_auto_export_tables(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    raw_path = Path(raw_dir)
    tables = _empty_health_tables()
    if not raw_path.exists():
        return tables

    manifest_files = sorted(raw_path.rglob("health_auto_export_manifest.csv"))
    collected: dict[str, list[pd.DataFrame]] = {key: [] for key in tables}
    for manifest_path in manifest_files:
        manifest = pd.read_csv(manifest_path)
        if "normalized_path" not in manifest.columns or "kind" not in manifest.columns:
            continue
        for row in manifest.to_dict(orient="records"):
            kind = str(row.get("kind") or "")
            normalized_path = str(row.get("normalized_path") or "").strip()
            if kind not in tables or not normalized_path:
                continue
            candidate = Path(normalized_path).expanduser()
            if not candidate.exists():
                continue
            frame = read_table(candidate)
            frame = _attach_manifest_metadata(frame, row)
            frame = _canonicalize_health_table_types(kind, frame)
            collected[kind].append(frame)

    for kind, frames in collected.items():
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True, sort=False)
        tables[kind] = _dedupe_canonical_health_table(kind, combined)
    return tables


def load_health_auto_export_tables(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    return load_unified_health_auto_export_tables(raw_dir)


def build_health_context_frame(base_frame: pd.DataFrame, data: IngestedData, *, freq: str = "5min") -> tuple[pd.DataFrame, list[str]]:
    out = base_frame.loc[:, ["timestamp"]].copy()
    out["timestamp"] = _parse_series(out["timestamp"])

    feature_columns: list[str] = []
    step_minutes = max(int(pd.Timedelta(freq).total_seconds() // 60), 1)

    if not data.health_activity.empty:
        activity = data.health_activity.copy()
        activity["timestamp"] = _parse_series(activity["timestamp"], floor=freq)
        activity["activity_value"] = pd.to_numeric(activity["activity_value"], errors="coerce")
        activity = activity.dropna(subset=["timestamp", "activity_value"])
        if not activity.empty:
            activity = activity.groupby("timestamp", as_index=False).agg(health_activity_value=("activity_value", "sum"))
            out = out.merge(activity, on="timestamp", how="left")
            out["health_activity_value"] = out["health_activity_value"].fillna(0.0)
            for window in (30, 60, 120):
                periods = max(int(window / step_minutes), 1)
                column = f"health_activity_roll_sum_{window}m"
                out[column] = out["health_activity_value"].rolling(periods, min_periods=1).sum()
                feature_columns.append(column)
            feature_columns.append("health_activity_value")

    if not data.sleep.empty:
        sleep = data.sleep.copy()
        sleep["date"] = _parse_series(sleep["date"]).dt.normalize()
        sleep["sleep_start"] = _parse_series(sleep["sleep_start"])
        sleep["sleep_end"] = _parse_series(sleep["sleep_end"])
        sleep["projection_date"] = sleep["sleep_end"].dt.normalize()
        sleep["projection_date"] = sleep["projection_date"].where(sleep["projection_date"].notna(), sleep["date"])
        out["date"] = out["timestamp"].dt.normalize()
        sleep_daily = sleep.loc[:, [
            "projection_date",
            "total_sleep_hours",
            "in_bed_hours",
            "core_hours",
            "deep_hours",
            "rem_hours",
            "awake_hours",
        ]].drop_duplicates(subset=["projection_date"], keep="last")
        sleep_daily = sleep_daily.rename(
            columns={
                "projection_date": "date",
                "total_sleep_hours": "prior_night_total_sleep_hours",
                "in_bed_hours": "prior_night_in_bed_hours",
                "core_hours": "prior_night_core_hours",
                "deep_hours": "prior_night_deep_hours",
                "rem_hours": "prior_night_rem_hours",
                "awake_hours": "prior_night_awake_hours",
            }
        )
        out = out.merge(sleep_daily, on="date", how="left")
        intervals = sleep.loc[:, ["sleep_start", "sleep_end"]].copy()
        out["in_sleep"] = 0
        for row in intervals.dropna(subset=["sleep_start", "sleep_end"]).itertuples(index=False):
            mask = (out["timestamp"] >= row.sleep_start) & (out["timestamp"] < row.sleep_end)
            out.loc[mask, "in_sleep"] = 1
        feature_columns.extend(
            [
                "in_sleep",
                "prior_night_total_sleep_hours",
                "prior_night_in_bed_hours",
                "prior_night_core_hours",
                "prior_night_deep_hours",
                "prior_night_rem_hours",
                "prior_night_awake_hours",
            ]
        )
        out = out.drop(columns=["date"])

    if not data.workouts.empty:
        workouts = data.workouts.copy()
        workouts["start_timestamp"] = _parse_series(workouts["start_timestamp"])
        workouts["end_timestamp"] = _parse_series(workouts["end_timestamp"])
        workouts["duration_seconds"] = pd.to_numeric(workouts["duration_seconds"], errors="coerce")
        workouts["distance_value"] = pd.to_numeric(workouts["distance_value"], errors="coerce")
        workouts["active_energy_value"] = pd.to_numeric(workouts["active_energy_value"], errors="coerce")
        workouts["avg_heart_rate"] = pd.to_numeric(workouts["avg_heart_rate"], errors="coerce")
        workouts["max_heart_rate"] = pd.to_numeric(workouts["max_heart_rate"], errors="coerce")
        workouts = workouts.dropna(subset=["end_timestamp"]).sort_values("end_timestamp")
        workout_times = workouts["end_timestamp"].to_numpy(dtype="datetime64[ns]")
        grid_times = out["timestamp"].to_numpy(dtype="datetime64[ns]")
        indices = np.searchsorted(workout_times, grid_times, side="right") - 1
        out["last_workout_duration_seconds"] = 0.0
        out["last_workout_active_energy_value"] = 0.0
        out["last_workout_distance_value"] = 0.0
        out["last_workout_avg_heart_rate"] = 0.0
        out["last_workout_max_heart_rate"] = 0.0
        out["minutes_since_last_workout"] = np.nan
        valid = indices >= 0
        if np.any(valid):
            valid_indices = indices[valid]
            matched = workouts.iloc[valid_indices]
            out.loc[valid, "last_workout_duration_seconds"] = matched["duration_seconds"].fillna(0.0).to_numpy()
            out.loc[valid, "last_workout_active_energy_value"] = matched["active_energy_value"].fillna(0.0).to_numpy()
            out.loc[valid, "last_workout_distance_value"] = matched["distance_value"].fillna(0.0).to_numpy()
            out.loc[valid, "last_workout_avg_heart_rate"] = matched["avg_heart_rate"].fillna(0.0).to_numpy()
            out.loc[valid, "last_workout_max_heart_rate"] = matched["max_heart_rate"].fillna(0.0).to_numpy()
            deltas = (out.loc[valid, "timestamp"].to_numpy(dtype="datetime64[ns]") - matched["end_timestamp"].to_numpy(dtype="datetime64[ns]")) / np.timedelta64(1, "m")
            out.loc[valid, "minutes_since_last_workout"] = deltas

        end_minutes = workouts["end_timestamp"].astype("int64").to_numpy()
        duration_values = workouts["duration_seconds"].fillna(0.0).to_numpy(dtype=float)
        energy_values = workouts["active_energy_value"].fillna(0.0).to_numpy(dtype=float)
        grid_minutes = out["timestamp"].astype("int64").to_numpy()
        if len(end_minutes):
            cumulative_duration = np.cumsum(duration_values)
            cumulative_energy = np.cumsum(energy_values)
            idx_6h = np.searchsorted(end_minutes, grid_minutes - int(pd.Timedelta(hours=6).value), side="left")
            idx_12h = np.searchsorted(end_minutes, grid_minutes - int(pd.Timedelta(hours=12).value), side="left")
            idx_24h = np.searchsorted(end_minutes, grid_minutes - int(pd.Timedelta(hours=24).value), side="left")
            current_idx = np.searchsorted(end_minutes, grid_minutes, side="right") - 1
            counts_6h = np.maximum(current_idx - idx_6h + 1, 0)
            counts_12h = np.maximum(current_idx - idx_12h + 1, 0)
            counts_24h = np.maximum(current_idx - idx_24h + 1, 0)
            out["recent_workout_6h"] = (counts_6h > 0).astype(int)
            out["recent_workout_12h"] = (counts_12h > 0).astype(int)
            out["workout_count_24h"] = counts_24h.astype(float)

            duration_before_24h = np.where(idx_24h > 0, cumulative_duration[idx_24h - 1], 0.0)
            energy_before_24h = np.where(idx_24h > 0, cumulative_energy[idx_24h - 1], 0.0)
            duration_up_to_current = np.where(current_idx >= 0, cumulative_duration[current_idx], 0.0)
            energy_up_to_current = np.where(current_idx >= 0, cumulative_energy[current_idx], 0.0)
            out["workout_duration_sum_24h"] = duration_up_to_current - duration_before_24h
            out["workout_active_energy_sum_24h"] = energy_up_to_current - energy_before_24h
        else:
            out["recent_workout_6h"] = 0
            out["recent_workout_12h"] = 0
            out["workout_count_24h"] = 0.0
            out["workout_duration_sum_24h"] = 0.0
            out["workout_active_energy_sum_24h"] = 0.0
        out["recent_workout_12h"] = out["minutes_since_last_workout"].le(12 * 60).fillna(False).astype(int)
        out["minutes_since_last_workout"] = out["minutes_since_last_workout"].fillna(1e6)
        feature_columns.extend(
            [
                "last_workout_duration_seconds",
                "last_workout_active_energy_value",
                "last_workout_distance_value",
                "last_workout_avg_heart_rate",
                "last_workout_max_heart_rate",
                "minutes_since_last_workout",
                "recent_workout_6h",
                "recent_workout_12h",
                "workout_count_24h",
                "workout_duration_sum_24h",
                "workout_active_energy_sum_24h",
            ]
        )

    if not data.health_measurements.empty:
        measurements = data.health_measurements.copy()
        measurements["timestamp"] = _parse_series(measurements["timestamp"], floor=freq)
        measurements["value"] = pd.to_numeric(measurements["value"], errors="coerce")
        measurements = measurements.dropna(subset=["timestamp", "metric", "stat", "value"])
        if not measurements.empty:
            heart_rate = measurements.loc[measurements["metric"] == "heart_rate"].copy()
            if not heart_rate.empty:
                heart_rate_pivot = (
                    heart_rate.assign(column=heart_rate["metric"].astype(str) + "__" + heart_rate["stat"].astype(str))
                    .pivot_table(index="timestamp", columns="column", values="value", aggfunc="mean")
                    .reset_index()
                )
                out = out.merge(heart_rate_pivot, on="timestamp", how="left")
                for stat in ["avg", "min", "max"]:
                    source_column = f"heart_rate__{stat}"
                    if source_column not in out.columns:
                        continue
                    latest_column = f"heart_rate_{stat}_latest"
                    out[latest_column] = out[source_column]
                    feature_columns.append(latest_column)
                    for window in (30, 60):
                        periods = max(int(window / step_minutes), 1)
                        roll_column = f"heart_rate_{stat}_roll_mean_{window}m"
                        out[roll_column] = out[source_column].rolling(periods, min_periods=1).mean()
                        feature_columns.append(roll_column)

            sparse = measurements.loc[
                measurements["metric"].isin(
                    ["heart_rate_variability", "respiratory_rate", "resting_heart_rate", "weight_body_mass"]
                )
            ].copy()
            if not sparse.empty:
                pivot = (
                    sparse.assign(column=sparse["metric"].astype(str) + "__" + sparse["stat"].astype(str))
                .pivot_table(index="timestamp", columns="column", values="value", aggfunc="mean")
                .reset_index()
                )
                out = out.merge(pivot, on="timestamp", how="left")
                fill_forward_columns = [column for column in out.columns if "__" in column and not column.startswith("heart_rate__")]
                if fill_forward_columns:
                    out[fill_forward_columns] = out[fill_forward_columns].ffill()

            for source_column, target_column, periods in [
                ("heart_rate_variability__value", "hrv_latest", None),
                ("heart_rate_variability__value", "hrv_roll_mean_24h", 288),
                ("respiratory_rate__value", "respiratory_rate_latest", None),
                ("respiratory_rate__value", "respiratory_rate_roll_mean_24h", 288),
                ("resting_heart_rate__value", "resting_heart_rate_latest", None),
                ("resting_heart_rate__value", "resting_heart_rate_roll_mean_24h", 288),
                ("weight_body_mass__value", "weight_latest", None),
                ("weight_body_mass__value", "weight_roll_mean_24h", 288),
            ]:
                if source_column not in out.columns:
                    continue
                if periods is None:
                    out[target_column] = out[source_column]
                else:
                    out[target_column] = out[source_column].rolling(periods, min_periods=1).mean()
                feature_columns.append(target_column)

    raw_measurement_columns = [column for column in out.columns if "__" in column]
    if raw_measurement_columns:
        out = out.drop(columns=raw_measurement_columns)

    feature_columns = [column for column in dict.fromkeys(feature_columns) if column in out.columns]
    if feature_columns:
        out[feature_columns] = out[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out, feature_columns


def build_analysis_ready_health_dataset(
    *,
    tandem_data: IngestedData,
    health_data: IngestedData,
    config: FeatureConfig | None = None,
) -> AnalysisReadyHealthDataset:
    feature_frame = build_feature_frame(tandem_data, config or FeatureConfig())
    if feature_frame.frame.empty:
        return AnalysisReadyHealthDataset(
            frame=feature_frame.frame.copy(),
            tandem_feature_columns=list(feature_frame.feature_columns),
            health_feature_columns=[],
            target_column=feature_frame.target_column,
            config=feature_frame.config,
        )
    context_frame, health_feature_columns = build_health_context_frame(
        feature_frame.frame,
        health_data,
        freq=feature_frame.config.freq,
    )
    combined = feature_frame.frame.merge(context_frame, on="timestamp", how="left")
    health_feature_columns = [column for column in dict.fromkeys(health_feature_columns) if column in combined.columns]
    if health_feature_columns:
        combined[health_feature_columns] = combined[health_feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return AnalysisReadyHealthDataset(
        frame=combined,
        tandem_feature_columns=list(feature_frame.feature_columns),
        health_feature_columns=health_feature_columns,
        target_column=feature_frame.target_column,
        config=feature_frame.config,
    )


def _ridge_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    means = X.mean(axis=0)
    scales = X.std(axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
    standardized = (X - means) / scales
    y_mean = float(np.mean(y))
    centered = y - y_mean
    gram = standardized.T @ standardized + alpha * np.eye(standardized.shape[1], dtype=float)
    coef = np.linalg.solve(gram, standardized.T @ centered)
    return coef, means, scales, y_mean


def _ridge_predict(X: np.ndarray, coef: np.ndarray, means: np.ndarray, scales: np.ndarray, y_mean: float) -> np.ndarray:
    standardized = (X - means) / scales
    return y_mean + standardized @ coef


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def screen_health_features(
    *,
    tandem_data: IngestedData,
    health_data: IngestedData,
    horizon_minutes: int = 30,
    alpha: float = 1.0,
) -> HealthFeatureScreeningResult:
    analysis_ready = build_analysis_ready_health_dataset(
        tandem_data=tandem_data,
        health_data=health_data,
        config=FeatureConfig(horizon_minutes=horizon_minutes),
    )
    if analysis_ready.frame.empty:
        raise ValueError("No Tandem feature rows are available for health feature screening")

    combined = analysis_ready.frame.copy()
    baseline_columns = list(analysis_ready.tandem_feature_columns)
    combined_health_columns = [column for column in analysis_ready.health_feature_columns if column in combined.columns]
    augmented_columns = baseline_columns + combined_health_columns
    combined = combined.dropna(subset=["target_glucose"]).reset_index(drop=True)
    if combined.empty:
        raise ValueError("No overlapping Tandem target rows are available for health feature screening")

    n_rows = len(combined)
    initial_train_size = max(int(n_rows * 0.5), 24)
    test_size = max(int(n_rows * 0.2), 12)
    if initial_train_size + test_size > n_rows:
        initial_train_size = max(int(n_rows * 0.6), 1)
        test_size = max(n_rows - initial_train_size, 1)
    splits = list(walk_forward_splits(n_rows, initial_train_size=initial_train_size, test_size=test_size))
    if not splits:
        splits = [type("Split", (), {"train_start": 0, "train_end": max(n_rows - 1, 1), "test_start": max(n_rows - 1, 1), "test_end": n_rows})()]

    baseline_rmse: list[float] = []
    augmented_rmse: list[float] = []
    baseline_mae: list[float] = []
    augmented_mae: list[float] = []
    coefficient_records: list[dict[str, Any]] = []

    y_all = combined["target_glucose"].to_numpy(dtype=float)
    base_matrix = combined.loc[:, baseline_columns].astype(float).to_numpy() if baseline_columns else np.zeros((n_rows, 0), dtype=float)
    augmented_matrix = combined.loc[:, augmented_columns].astype(float).to_numpy() if augmented_columns else np.zeros((n_rows, 0), dtype=float)

    for split_index, split in enumerate(splits, start=1):
        train_slice = slice(split.train_start, split.train_end)
        test_slice = slice(split.test_start, split.test_end)
        y_train = y_all[train_slice]
        y_test = y_all[test_slice]
        base_train = base_matrix[train_slice]
        base_test = base_matrix[test_slice]
        full_train = augmented_matrix[train_slice]
        full_test = augmented_matrix[test_slice]

        if base_train.shape[1] > 0:
            base_coef, base_means, base_scales, base_y_mean = _ridge_fit(base_train, y_train, alpha=alpha)
            base_pred = _ridge_predict(base_test, base_coef, base_means, base_scales, base_y_mean)
        else:
            base_pred = np.repeat(float(np.mean(y_train)), len(y_test))
        if full_train.shape[1] > 0:
            full_coef, full_means, full_scales, full_y_mean = _ridge_fit(full_train, y_train, alpha=alpha)
            full_pred = _ridge_predict(full_test, full_coef, full_means, full_scales, full_y_mean)
        else:
            full_coef = np.array([], dtype=float)
            full_pred = np.repeat(float(np.mean(y_train)), len(y_test))

        baseline_rmse.append(_rmse(y_test, base_pred))
        augmented_rmse.append(_rmse(y_test, full_pred))
        baseline_mae.append(_mae(y_test, base_pred))
        augmented_mae.append(_mae(y_test, full_pred))

        if full_coef.size and combined_health_columns:
            for column, coefficient in zip(augmented_columns, full_coef, strict=False):
                if column in combined_health_columns:
                    coefficient_records.append({"feature": column, "split": split_index, "coefficient": float(coefficient)})

    coefficient_frame = pd.DataFrame(coefficient_records)
    rows: list[dict[str, Any]] = []
    for column in combined_health_columns:
        availability = float(combined[column].notna().mean()) if column in combined else 0.0
        if coefficient_frame.empty:
            mean_abs = 0.0
            std_abs = 0.0
            split_present = 0
        else:
            feature_coefficients = coefficient_frame.loc[coefficient_frame["feature"] == column, "coefficient"].astype(float)
            mean_abs = float(feature_coefficients.abs().mean()) if not feature_coefficients.empty else 0.0
            std_abs = float(feature_coefficients.abs().std(ddof=0)) if len(feature_coefficients) > 1 else 0.0
            split_present = int(feature_coefficients.shape[0])
        rows.append(
            {
                "feature": column,
                "availability": availability,
                "mean_abs_coefficient": mean_abs,
                "std_abs_coefficient": std_abs,
                "split_count": split_present,
            }
        )
    scores = pd.DataFrame(rows).sort_values(["mean_abs_coefficient", "availability", "feature"], ascending=[False, False, True]).reset_index(drop=True)
    if not scores.empty:
        median_signal = float(scores["mean_abs_coefficient"].median())
        scores["recommended"] = (
            scores["availability"].ge(0.2)
            & scores["split_count"].ge(max(len(splits) // 2, 1))
            & scores["mean_abs_coefficient"].ge(median_signal)
        )
    else:
        scores["recommended"] = []

    return HealthFeatureScreeningResult(
        scores=scores,
        baseline_rmse_mean=float(np.mean(baseline_rmse)),
        augmented_rmse_mean=float(np.mean(augmented_rmse)),
        baseline_mae_mean=float(np.mean(baseline_mae)),
        augmented_mae_mean=float(np.mean(augmented_mae)),
        split_count=len(splits),
    )


def write_health_screening_report(result: HealthFeatureScreeningResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Health Feature Screening",
        "",
        f"- split_count: {result.split_count}",
        f"- baseline_rmse_mean: {result.baseline_rmse_mean:.4f}",
        f"- augmented_rmse_mean: {result.augmented_rmse_mean:.4f}",
        f"- baseline_mae_mean: {result.baseline_mae_mean:.4f}",
        f"- augmented_mae_mean: {result.augmented_mae_mean:.4f}",
        "",
        "## Recommended Features",
    ]
    if result.scores.empty:
        lines.append("")
        lines.append("No health context features were available.")
    else:
        recommended = result.scores.loc[result.scores["recommended"].fillna(False)]
        if recommended.empty:
            lines.append("")
            lines.append("No health features met the recommendation rule in this run.")
        else:
            lines.append("")
            for row in recommended.itertuples(index=False):
                lines.append(
                    f"- {row.feature}: availability={row.availability:.2f}, mean_abs_coefficient={row.mean_abs_coefficient:.4f}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
