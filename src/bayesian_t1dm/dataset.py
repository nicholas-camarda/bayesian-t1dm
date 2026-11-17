"""One merged, deduplicated, validated dataset built once per data update.

``build_merged_dataset`` fingerprints every raw input (Tandem window payloads, Apple Health
export JSONs and hand-kept carb logs), and only does work when a fingerprint changed: it
normalizes new Tandem windows, reads Tandem once, adds Apple Health, builds the 5-minute
dataset and writes everything to ``<Project Vault>/data/merged/``. Analyses call
``load_merged_dataset``, which refuses to return data whose raw inputs have changed since
the build, so a stale dataset is never used silently.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

from .acquisition import normalize_tconnectsync_archive
from .features import FeatureConfig
from .health_auto_export import (
    AnalysisReadyHealthDataset,
    ModelDataPreparationResult,
    build_prepared_model_dataset,
    discover_health_export_files,
    intersect_spans,
    summarize_apple_health_span,
)
from .ingest import IngestedData, TandemCoverage, attach_health_auto_export, load_tandem_exports, summarize_coverage, write_export_manifest
from .paths import ProjectPaths
from .quality import DataQualitySummary, assess_data_quality, assess_prepared_feature_plausibility
from .report_page import plain

# Bump when merge, cleaning or feature code changes what the build produces; a version change forces a rebuild.
DATASET_VERSION = 2
HORIZON_MINUTES = 30
CARB_LOG_NAMES = ("carbs.csv", "carb_log.csv")


class StaleDatasetError(RuntimeError):
    """Raw inputs changed since the merged dataset was built, or it was never built."""


@dataclass(frozen=True)
class InputFile:
    path: str  # relative to the raw data folder
    kind: str
    size: int
    sha256: str


@dataclass(frozen=True)
class MergedDataset:
    preparation: ModelDataPreparationResult
    data: IngestedData  # Tandem plus Apple Health, deduplicated
    tandem_data: IngestedData  # Tandem only
    coverage: TandemCoverage
    data_quality: DataQualitySummary
    quality_rows: pd.DataFrame
    manifest: dict[str, Any]


def merged_dir(raw_root: str | Path) -> Path:
    """The merged dataset sits next to the raw folder it was built from (``data/raw`` -> ``data/merged``)."""
    return Path(raw_root).expanduser().resolve().parent / "merged"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_input_files(raw_root: Path) -> list[tuple[Path, str]]:
    """The raw files the dataset is built from, with their kind. Derived ``normalized/`` files are excluded."""
    found: list[tuple[Path, str]] = []
    tandem_root = raw_root / "tconnectsync"
    if tandem_root.exists():
        found.extend((path, "tandem_raw") for path in sorted(tandem_root.glob("*/raw/*")) if path.is_file())
    found.extend((path, "apple_health_export") for path in discover_health_export_files(raw_root))
    for name in CARB_LOG_NAMES:
        found.extend(
            (path, "carb_log")
            for path in sorted(raw_root.rglob(name))
            if "normalized" not in path.relative_to(raw_root).parts
        )
    return found


def fingerprint_inputs(raw_root: Path) -> list[InputFile]:
    return [
        InputFile(path=path.relative_to(raw_root).as_posix(), kind=kind, size=path.stat().st_size, sha256=_sha256(path))
        for path, kind in raw_input_files(raw_root)
    ]


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _read_manifest(directory: Path) -> dict[str, Any] | None:
    path = directory / "dataset.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _input_diff(old: list[dict[str, Any]], new: list[InputFile]) -> dict[str, list[str]]:
    old_map = {item["path"]: item["sha256"] for item in old}
    new_map = {item.path: item.sha256 for item in new}
    return {
        "added": sorted(set(new_map) - set(old_map)),
        "removed": sorted(set(old_map) - set(new_map)),
        "changed": sorted(path for path in set(new_map) & set(old_map) if new_map[path] != old_map[path]),
    }


def _changed_tandem_windows(diff: dict[str, list[str]]) -> list[str]:
    windows = set()
    for path in diff["added"] + diff["changed"]:
        parts = Path(path).parts
        if len(parts) >= 3 and parts[0] == "tconnectsync" and parts[2] == "raw":
            windows.add(parts[1])
    return sorted(windows)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_TABLES = ("cgm", "bolus", "carbs", "basal", "activity", "health_activity", "health_measurements", "sleep", "workouts")


def _to_parquet(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == object:
            # Parquet needs one type per column; mixed object columns are stored as text.
            non_null = out[column].dropna()
            if not non_null.map(type).nunique() <= 1:
                out[column] = out[column].map(lambda value: None if pd.isna(value) else str(value))
    out.to_parquet(path, index=False)


def _write_tables(data: IngestedData, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in _TABLES:
        _to_parquet(getattr(data, name), directory / f"{name}.parquet")


def _read_tables(directory: Path) -> IngestedData:
    return IngestedData(**{name: pd.read_parquet(directory / f"{name}.parquet") for name in _TABLES})


def _timestamp_or_none(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _tandem_span(data: IngestedData) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    stamps = [pd.to_datetime(frame["timestamp"], errors="coerce") for frame in (data.cgm, data.bolus) if not frame.empty and "timestamp" in frame]
    combined = pd.concat(stamps, ignore_index=True).dropna() if stamps else pd.Series(dtype="datetime64[ns]")
    return (None, None) if combined.empty else (pd.Timestamp(combined.min()), pd.Timestamp(combined.max()))


def build_merged_dataset(paths: ProjectPaths, *, raw_root: str | Path | None = None, force: bool = False, log=print) -> tuple[MergedDataset, bool]:
    """Build the merged dataset if any raw input changed. Returns (dataset, rebuilt)."""
    raw_root = Path(raw_root or paths.cloud_raw).expanduser().resolve()
    out_dir = merged_dir(raw_root)
    log("Fingerprinting raw inputs...")
    inputs = fingerprint_inputs(raw_root)
    previous = _read_manifest(out_dir)
    diff = _input_diff(previous["inputs"] if previous else [], inputs)
    unchanged = previous is not None and previous.get("dataset_version") == DATASET_VERSION and not any(diff.values())
    if unchanged and not force:
        log(f"Merged dataset is up to date ({len(inputs)} raw files unchanged); nothing to do.")
        return load_merged_dataset(raw_root, inputs=inputs), False

    if previous is None:
        log(f"No merged dataset yet; building from {len(inputs)} raw files.")
    else:
        log(f"Inputs changed: {len(diff['added'])} added, {len(diff['changed'])} changed, {len(diff['removed'])} removed.")

    tandem_root = raw_root / "tconnectsync"
    if tandem_root.exists():
        missing = sorted(p.parent.name for p in tandem_root.glob("*/raw") if not (p.parent / "normalized").exists())
        windows = sorted(set(_changed_tandem_windows(diff) if previous else []) | set(missing))
        for window_id in windows:
            log(f"Normalizing Tandem window {window_id}...")
            normalize_tconnectsync_archive(
                paths,
                raw_root=tandem_root,
                window_id=window_id,
                force=True,
                report_path=paths.output_source / "normalize_raw_summary.md",
            )

    log("Reading Tandem data...")
    tandem = load_tandem_exports(raw_root, include_health_auto_export=False)
    log("Adding Apple Health...")
    merged = attach_health_auto_export(tandem, raw_root)
    write_export_manifest(merged.manifest, raw_root / "tandem_export_manifest.csv")

    log("Building the 5-minute dataset...")
    dataset = build_prepared_model_dataset(tandem_data=tandem, health_data=merged, config=FeatureConfig(horizon_minutes=HORIZON_MINUTES))
    data_quality, quality_rows = assess_data_quality(raw_root, export_manifest=merged.manifest)
    plausibility_status, plausibility_reasons = assess_prepared_feature_plausibility(dataset.frame)
    data_quality = DataQualitySummary(**{**asdict(data_quality), "normalized_plausibility_status": plausibility_status, "normalized_plausibility_reasons": plausibility_reasons})
    dataset = AnalysisReadyHealthDataset(
        **{
            **{f.name: getattr(dataset, f.name) for f in fields(dataset)},
            "source_quality_status": data_quality.status,
            "normalized_plausibility_status": plausibility_status,
        }
    )

    tandem_start, tandem_end = _tandem_span(tandem)
    apple_start, apple_end = summarize_apple_health_span(merged)
    overlap_start, overlap_end = intersect_spans(apple_start, apple_end, tandem_start, tandem_end)
    frame_ts = pd.to_datetime(dataset.frame["timestamp"], errors="coerce") if not dataset.frame.empty else pd.Series(dtype="datetime64[ns]")
    coverage = summarize_coverage(merged)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_tables(merged, out_dir / "tables")
    _to_parquet(tandem.cgm, out_dir / "tables" / "tandem_cgm.parquet")
    _to_parquet(dataset.frame, out_dir / "prepared_model_data_5min.parquet")
    _to_parquet(quality_rows, out_dir / "window_quality.parquet")

    manifest = {
        "dataset_version": DATASET_VERSION,
        "built_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "code_commit": _git_commit(paths.root),
        "horizon_minutes": HORIZON_MINUTES,
        "inputs": [asdict(item) for item in inputs],
        "input_changes": diff,
        "dataset": {
            "feature_columns": list(dataset.feature_columns),
            "tandem_feature_columns": list(dataset.tandem_feature_columns),
            "health_feature_columns": list(dataset.health_feature_columns),
            "target_column": dataset.target_column,
            "mode": dataset.mode,
            "apple_available": dataset.apple_available,
            "explicit_carb_source_available": dataset.explicit_carb_source_available,
            "source_quality_status": dataset.source_quality_status,
            "normalized_plausibility_status": dataset.normalized_plausibility_status,
        },
        "spans": {
            "tandem_start": tandem_start,
            "tandem_end": tandem_end,
            "apple_start": apple_start,
            "apple_end": apple_end,
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "dataset_start": None if frame_ts.dropna().empty else frame_ts.min(),
            "dataset_end": None if frame_ts.dropna().empty else frame_ts.max(),
        },
        "row_counts": {name: int(len(getattr(merged, name))) for name in _TABLES} | {"tandem_cgm": int(len(tandem.cgm)), "prepared_rows": int(len(dataset.frame))},
        "coverage": asdict(coverage),
        "data_quality": asdict(data_quality),
    }
    (out_dir / "dataset.json").write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    (out_dir / "validation_report.md").write_text(_validation_report(manifest), encoding="utf-8")
    log(f"Wrote the merged dataset to {out_dir}")
    return load_merged_dataset(raw_root, inputs=inputs), True


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_merged_dataset(raw_root: str | Path, *, inputs: list[InputFile] | None = None) -> MergedDataset:
    """Load the merged dataset built from ``raw_root``, refusing if the raw inputs changed since it was built."""
    raw_root = Path(raw_root).expanduser().resolve()
    out_dir = merged_dir(raw_root)
    manifest = _read_manifest(out_dir)
    if manifest is None:
        raise StaleDatasetError(f"No merged dataset in {out_dir}. Run `bayesian-t1dm build-dataset` first.")
    if manifest.get("dataset_version") != DATASET_VERSION:
        raise StaleDatasetError("The merged dataset was built by an older version of the code. Run `bayesian-t1dm build-dataset`.")
    current = inputs if inputs is not None else fingerprint_inputs(raw_root)
    diff = _input_diff(manifest["inputs"], current)
    if any(diff.values()):
        raise StaleDatasetError(
            f"Raw data changed since the merged dataset was built ({len(diff['added'])} added, {len(diff['changed'])} changed, "
            f"{len(diff['removed'])} removed). Run `bayesian-t1dm build-dataset`."
        )

    meta = manifest["dataset"]
    frame = pd.read_parquet(out_dir / "prepared_model_data_5min.parquet")
    dataset = AnalysisReadyHealthDataset(
        frame=frame,
        feature_columns=list(meta["feature_columns"]),
        tandem_feature_columns=list(meta["tandem_feature_columns"]),
        health_feature_columns=list(meta["health_feature_columns"]),
        target_column=meta["target_column"],
        horizon_minutes=int(manifest["horizon_minutes"]),
        config=FeatureConfig(horizon_minutes=int(manifest["horizon_minutes"])),
        mode=meta["mode"],
        apple_available=bool(meta["apple_available"]),
        explicit_carb_source_available=bool(meta["explicit_carb_source_available"]),
        source_quality_status=meta["source_quality_status"],
        normalized_plausibility_status=meta["normalized_plausibility_status"],
    )
    spans = {key: _timestamp_or_none(value) for key, value in manifest["spans"].items()}
    preparation = ModelDataPreparationResult(
        dataset=dataset,
        apple_available=dataset.apple_available,
        apple_span_start=spans["apple_start"],
        apple_span_end=spans["apple_end"],
        tandem_span_before_start=spans["tandem_start"],
        tandem_span_before_end=spans["tandem_end"],
        tandem_span_after_start=spans["tandem_start"],
        tandem_span_after_end=spans["tandem_end"],
        requested_tandem_start=None,
        requested_tandem_end=None,
        overlap_start=spans["overlap_start"],
        overlap_end=spans["overlap_end"],
        final_dataset_start=spans["dataset_start"],
        final_dataset_end=spans["dataset_end"],
        final_row_count=int(len(frame)),
        backfill_status="not_requested",
    )
    data = _read_tables(out_dir / "tables")
    tandem = IngestedData(**{**{name: getattr(data, name) for name in _TABLES}, "cgm": pd.read_parquet(out_dir / "tables" / "tandem_cgm.parquet")})
    tandem.health_activity = tandem.health_measurements = tandem.sleep = tandem.workouts = pd.DataFrame()
    coverage_fields = {**manifest["coverage"]}
    for key in ("first_timestamp", "last_timestamp"):
        coverage_fields[key] = _timestamp_or_none(coverage_fields.get(key))
    return MergedDataset(
        preparation=preparation,
        data=data,
        tandem_data=tandem,
        coverage=TandemCoverage(**coverage_fields),
        data_quality=DataQualitySummary(**manifest["data_quality"]),
        quality_rows=pd.read_parquet(out_dir / "window_quality.parquet"),
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


def _span_text(start: Any, end: Any) -> str:
    start_ts, end_ts = _timestamp_or_none(start), _timestamp_or_none(end)
    if start_ts is None or end_ts is None:
        return "none"
    return f"{start_ts:%b %d, %Y} to {end_ts:%b %d, %Y}"


def _validation_report(manifest: dict[str, Any]) -> str:
    spans = manifest["spans"]
    counts = manifest["row_counts"]
    quality = manifest["data_quality"]
    inputs = manifest["inputs"]
    kinds = pd.Series([item["kind"] for item in inputs]).value_counts().to_dict() if inputs else {}
    apple_files = [item for item in inputs if item["kind"] == "apple_health_export"]
    distinct_apple = len({item["sha256"] for item in apple_files})
    changes = manifest["input_changes"]
    plausibility = quality.get("normalized_plausibility_reasons") or {}
    lines = [
        "# Merged Dataset Validation",
        "",
        f"Built {manifest['built_at']} from code commit `{manifest['code_commit'][:12]}`.",
        "",
        "## What went in",
        "",
        f"- Tandem: {kinds.get('tandem_raw', 0)} raw download files, covering {_span_text(spans['tandem_start'], spans['tandem_end'])}.",
        f"- Apple Health: {len(apple_files)} export files ({distinct_apple} distinct; byte-identical copies are read once), "
        f"covering {_span_text(spans['apple_start'], spans['apple_end'])}.",
        f"- Hand-kept carb logs: {kinds.get('carb_log', 0)}.",
        f"- Changes since the last build: {len(changes['added'])} files added, {len(changes['changed'])} changed, {len(changes['removed'])} removed.",
        "",
        "## What came out",
        "",
        f"- 5-minute dataset: {counts['prepared_rows']:,} rows, {_span_text(spans['dataset_start'], spans['dataset_end'])}.",
        f"- Tandem and Apple Health overlap: {_span_text(spans['overlap_start'], spans['overlap_end'])}.",
        f"- CGM readings: {counts['tandem_cgm']:,} from Tandem; {counts['cgm']:,} after adding Apple Health readings that fill Tandem gaps "
        "(the same reading from both sources is kept once).",
        f"- Boluses {counts['bolus']:,}; carb entries {counts['carbs']:,}; basal segments {counts['basal']:,}.",
        f"- Apple Health: {counts['health_activity']:,} activity rows, {counts['health_measurements']:,} measurements, "
        f"{counts['sleep']:,} nights of sleep, {counts['workouts']:,} workouts.",
        "",
        "## Checks",
        "",
        "| check | result |",
        "| --- | --- |",
        f"| Tandem download windows complete | {plain(quality['status'])}; {quality['incomplete_window_count']} windows with gaps or other problems |",
    ]
    for reason, count in sorted((quality.get("reason_counts") or {}).items(), key=lambda kv: -kv[1]):
        lines.append(f"| windows where {plain(reason)} | {count} |")
    lines.append(f"| 5-minute data plausible | {plain(quality.get('normalized_plausibility_status'))} |")
    for reason, count in sorted(plausibility.items(), key=lambda kv: -kv[1]):
        lines.append(f"| rows with {plain(reason)} | {count:,} |")
    lines += [
        "",
        "Inputs are fingerprinted (SHA-256) in `dataset.json`. Analyses refuse to run if any raw file changes until the dataset is rebuilt.",
        "",
    ]
    return "\n".join(lines)
