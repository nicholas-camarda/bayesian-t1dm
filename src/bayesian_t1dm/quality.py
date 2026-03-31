from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REQUIRED_WINDOW_KINDS: tuple[str, ...] = ("cgm", "bolus", "basal")
OPTIONAL_WINDOW_KINDS: tuple[str, ...] = ("activity", "carbs")
QUALITY_REASON_ORDER: tuple[str, ...] = (
    "missing_kind",
    "starts_late",
    "ends_early",
    "internal_gap",
    "duplicates",
    "overlap",
)


@dataclass(frozen=True)
class DataQualitySummary:
    status: str
    contributing_window_ids: list[str]
    incomplete_window_count: int
    reason_counts: dict[str, int]
    evaluation_touches_incomplete_windows: bool


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _parse_timestamp_utc(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    try:
        return pd.to_datetime(value, errors="coerce", utc=True)
    except Exception:
        return pd.NaT


def _iso_or_none(value: Any) -> str | None:
    timestamp = _parse_timestamp_utc(value)
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).isoformat()


def ordered_completeness_reasons(
    *,
    requested_start: str | None,
    requested_end: str | None,
    observed_first_timestamp: Any,
    observed_last_timestamp: Any,
    row_count: int,
    required: bool,
    has_internal_gap: bool = False,
    has_duplicates: bool = False,
    has_overlap: bool = False,
) -> list[str]:
    reasons: list[str] = []
    requested_start_date = pd.Timestamp(requested_start).date() if requested_start else None
    requested_end_date = pd.Timestamp(requested_end).date() if requested_end else None
    observed_first = _parse_timestamp_utc(observed_first_timestamp)
    observed_last = _parse_timestamp_utc(observed_last_timestamp)

    if required and int(row_count) <= 0:
        reasons.append("missing_kind")
    if requested_start_date is not None and not pd.isna(observed_first) and observed_first.date() > requested_start_date:
        reasons.append("starts_late")
    if requested_end_date is not None and not pd.isna(observed_last) and observed_last.date() < requested_end_date:
        reasons.append("ends_early")
    if has_internal_gap:
        reasons.append("internal_gap")
    if has_duplicates:
        reasons.append("duplicates")
    if has_overlap:
        reasons.append("overlap")

    ordered: list[str] = []
    seen: set[str] = set()
    for reason in QUALITY_REASON_ORDER:
        if reason in reasons and reason not in seen:
            ordered.append(reason)
            seen.add(reason)
    for reason in reasons:
        if reason not in seen:
            ordered.append(reason)
            seen.add(reason)
    return ordered


def build_window_quality_row(
    *,
    window_id: str,
    kind: str,
    requested_start: str | None,
    requested_end: str | None,
    endpoint_family: str | None = None,
    source_label: str | None = None,
    raw_path: str | None = None,
    normalized_path: str | None = None,
    row_count: int = 0,
    observed_first_timestamp: Any = None,
    observed_last_timestamp: Any = None,
    has_internal_gap: bool = False,
    has_duplicates: bool = False,
    has_overlap: bool = False,
    payload_sha256: str | None = None,
    required: bool = True,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requested_days = None
    if requested_start and requested_end:
        requested_days = int((pd.Timestamp(requested_end).date() - pd.Timestamp(requested_start).date()).days + 1)

    observed_first = _parse_timestamp_utc(observed_first_timestamp)
    observed_last = _parse_timestamp_utc(observed_last_timestamp)
    observed_duration_days = None
    if not pd.isna(observed_first) and not pd.isna(observed_last):
        observed_duration_days = int((observed_last.date() - observed_first.date()).days + 1)

    coverage_fraction = 0.0
    if requested_days and requested_days > 0 and observed_duration_days is not None:
        coverage_fraction = float(max(min(observed_duration_days / requested_days, 1.0), 0.0))

    completeness_reasons = ordered_completeness_reasons(
        requested_start=requested_start,
        requested_end=requested_end,
        observed_first_timestamp=observed_first_timestamp,
        observed_last_timestamp=observed_last_timestamp,
        row_count=row_count,
        required=required,
        has_internal_gap=has_internal_gap,
        has_duplicates=has_duplicates,
        has_overlap=has_overlap,
    )

    row = {
        "window_id": window_id,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "endpoint_family": endpoint_family,
        "kind": kind,
        "source_label": source_label,
        "raw_path": raw_path,
        "normalized_path": normalized_path,
        "row_count": int(row_count),
        "first_timestamp": _iso_or_none(observed_first_timestamp),
        "last_timestamp": _iso_or_none(observed_last_timestamp),
        "observed_first_timestamp": _iso_or_none(observed_first_timestamp),
        "observed_last_timestamp": _iso_or_none(observed_last_timestamp),
        "observed_duration_days": observed_duration_days,
        "coverage_fraction": coverage_fraction,
        "is_complete_window": not completeness_reasons,
        "has_internal_gap": bool(has_internal_gap),
        "has_overlap": bool(has_overlap),
        "has_duplicates": bool(has_duplicates),
        "payload_sha256": payload_sha256,
        "completeness_reasons": json.dumps(completeness_reasons),
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def _discover_window_manifest_paths(raw_dir: str | Path) -> list[Path]:
    root = Path(raw_dir).expanduser()
    if not root.exists():
        return []
    direct = root / "window_manifest.csv"
    if direct.exists():
        return [direct]
    return sorted(path.resolve() for path in root.rglob("window_manifest.csv"))


def _read_window_quality_rows(raw_dir: str | Path) -> pd.DataFrame:
    manifest_paths = _discover_window_manifest_paths(raw_dir)
    if not manifest_paths:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        try:
            manifest = pd.read_csv(manifest_path)
        except Exception:
            continue
        if manifest.empty:
            continue

        window_id = str(manifest["window_id"].dropna().iloc[0]) if "window_id" in manifest.columns and manifest["window_id"].notna().any() else manifest_path.parent.name
        requested_start = str(manifest["requested_start"].dropna().iloc[0]) if "requested_start" in manifest.columns and manifest["requested_start"].notna().any() else None
        requested_end = str(manifest["requested_end"].dropna().iloc[0]) if "requested_end" in manifest.columns and manifest["requested_end"].notna().any() else None
        endpoint_family = str(manifest["endpoint_family"].dropna().iloc[0]) if "endpoint_family" in manifest.columns and manifest["endpoint_family"].notna().any() else None
        source_label = str(manifest["source_label"].dropna().iloc[0]) if "source_label" in manifest.columns and manifest["source_label"].notna().any() else None

        present_rows: dict[str, dict[str, Any]] = {}
        for row in manifest.to_dict(orient="records"):
            kind = str(row.get("kind") or "").strip()
            if not kind:
                continue
            present_rows[kind] = row

        expected_kinds = list(REQUIRED_WINDOW_KINDS)
        for kind in present_rows:
            if kind not in expected_kinds:
                expected_kinds.append(kind)

        for kind in expected_kinds:
            row = present_rows.get(kind, {})
            required = kind in REQUIRED_WINDOW_KINDS
            rows.append(
                build_window_quality_row(
                    window_id=window_id,
                    kind=kind,
                    requested_start=requested_start,
                    requested_end=requested_end,
                    endpoint_family=str(row.get("endpoint_family") or endpoint_family or ""),
                    source_label=str(row.get("source_label") or source_label or ""),
                    raw_path=None if pd.isna(row.get("raw_path")) else row.get("raw_path"),
                    normalized_path=None if pd.isna(row.get("normalized_path")) else row.get("normalized_path"),
                    row_count=int(row.get("row_count", 0) or 0),
                    observed_first_timestamp=row.get("observed_first_timestamp", row.get("first_timestamp")),
                    observed_last_timestamp=row.get("observed_last_timestamp", row.get("last_timestamp")),
                    has_internal_gap=_coerce_bool(row.get("has_internal_gap")),
                    has_duplicates=_coerce_bool(row.get("has_duplicates")),
                    has_overlap=_coerce_bool(row.get("has_overlap")),
                    payload_sha256=None if pd.isna(row.get("payload_sha256")) else row.get("payload_sha256"),
                    required=required,
                )
            )
    return pd.DataFrame(rows)


def _fallback_quality_rows_from_export_manifest(
    *,
    export_manifest: pd.DataFrame,
) -> pd.DataFrame:
    if export_manifest.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for row in export_manifest.to_dict(orient="records"):
        source_file = str(row.get("source_file") or "raw")
        reasons: list[str] = []
        if _coerce_bool(row.get("has_internal_gap")):
            reasons.append("internal_gap")
        if _coerce_bool(row.get("has_duplicates")):
            reasons.append("duplicates")
        if not _coerce_bool(row.get("is_complete_window", True)):
            if not reasons:
                reasons.append("ends_early")
        quality_row = {
            "window_id": source_file,
            "requested_start": _iso_or_none(row.get("first_timestamp")),
            "requested_end": _iso_or_none(row.get("last_timestamp")),
            "endpoint_family": None,
            "kind": str(row.get("kind") or ""),
            "source_label": source_file,
            "raw_path": None,
            "normalized_path": source_file,
            "row_count": int(row.get("rows", 0) or 0),
            "first_timestamp": _iso_or_none(row.get("first_timestamp")),
            "last_timestamp": _iso_or_none(row.get("last_timestamp")),
            "observed_first_timestamp": _iso_or_none(row.get("first_timestamp")),
            "observed_last_timestamp": _iso_or_none(row.get("last_timestamp")),
            "observed_duration_days": None,
            "coverage_fraction": 1.0,
            "is_complete_window": not reasons,
            "has_internal_gap": _coerce_bool(row.get("has_internal_gap")),
            "has_overlap": False,
            "has_duplicates": _coerce_bool(row.get("has_duplicates")),
            "payload_sha256": None,
            "completeness_reasons": json.dumps(reasons),
        }
        rows.append(quality_row)
    return pd.DataFrame(rows)


def assess_data_quality(
    raw_dir: str | Path,
    *,
    export_manifest: pd.DataFrame | None = None,
) -> tuple[DataQualitySummary, pd.DataFrame]:
    quality_rows = _read_window_quality_rows(raw_dir)
    if quality_rows.empty and export_manifest is not None:
        quality_rows = _fallback_quality_rows_from_export_manifest(export_manifest=export_manifest)

    if quality_rows.empty:
        return (
            DataQualitySummary(
                status="broken",
                contributing_window_ids=[],
                incomplete_window_count=0,
                reason_counts={},
                evaluation_touches_incomplete_windows=False,
            ),
            quality_rows,
        )

    incomplete_windows: set[str] = set()
    aggregated_reasons: Counter[str] = Counter()
    for window_id, window_rows in quality_rows.groupby("window_id", dropna=False):
        window_reasons: set[str] = set()
        for value in window_rows["completeness_reasons"].fillna("[]").astype(str):
            try:
                reasons = json.loads(value)
            except Exception:
                reasons = []
            for reason in reasons:
                window_reasons.add(str(reason))
        if window_reasons:
            incomplete_windows.add(str(window_id))
            aggregated_reasons.update(window_reasons)

    contributing_window_ids = sorted(quality_rows["window_id"].dropna().astype(str).unique().tolist())
    cgm_missing = (
        quality_rows["kind"].eq("cgm")
        & quality_rows["row_count"].fillna(0).astype(int).le(0)
    ).any()
    if cgm_missing or not contributing_window_ids:
        status = "broken"
    elif incomplete_windows:
        status = "degraded"
    else:
        status = "good"

    return (
        DataQualitySummary(
            status=status,
            contributing_window_ids=contributing_window_ids,
            incomplete_window_count=len(incomplete_windows),
            reason_counts={key: int(value) for key, value in sorted(aggregated_reasons.items())},
            evaluation_touches_incomplete_windows=bool(incomplete_windows),
        ),
        quality_rows.sort_values(["window_id", "kind"]).reset_index(drop=True),
    )
