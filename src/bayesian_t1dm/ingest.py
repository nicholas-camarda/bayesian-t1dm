from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .io import coalesce_columns, read_table


BROAD_CGM_COLUMNS = ["dateTime", "timestamp", "datetime", "eventdatetime", "date time"]
BOLUS_COLUMNS = ["completiondatetime", "datetime", "timestamp", "dateTime"]
CGM_GLUCOSE_COLUMNS = ["egv_estimatedGlucoseValue", "bg", "glucose", "readings (cgm / bgm)"]
CGM_GENERIC_GLUCOSE_COLUMNS = ["value"]
BOLUS_UNITS_COLUMNS = ["actualtotalbolusrequested", "bolus", "bolus_units", "insulin", "units"]
CARB_COLUMNS = ["carbsize", "carbs", "carbohydrates", "guessedcarbohydrate"]
BASAL_RATE_COLUMNS = ["basalrate", "rate", "units_per_hour", "units/hour"]
ACTIVITY_COLUMNS = ["value", "steps", "activity", "count"]
EXCLUDED_RAW_PATH_PARTS = {"health_auto_export", "archive data", "apple_health_data"}


@dataclass(frozen=True)
class TandemCoverage:
    source_files: int
    manifest_rows: int
    cgm_rows: int
    bolus_rows: int
    basal_rows: int
    activity_rows: int
    health_activity_rows: int
    health_measurement_rows: int
    sleep_rows: int
    workout_rows: int
    first_timestamp: pd.Timestamp | None
    last_timestamp: pd.Timestamp | None
    complete_windows: int
    incomplete_windows: int
    gap_count: int
    overlap_count: int
    duplicate_windows: int
    out_of_order_windows: int
    is_complete: bool


@dataclass
class IngestedData:
    cgm: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    bolus: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    carbs: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    basal: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    activity: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    health_activity: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    health_measurements: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    sleep: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    workouts: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    manifest: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    source_files: list[Path] = field(default_factory=list)

    def all_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "cgm": self.cgm,
            "bolus": self.bolus,
            "carbs": self.carbs,
            "basal": self.basal,
            "activity": self.activity,
            "health_activity": self.health_activity,
            "health_measurements": self.health_measurements,
            "sleep": self.sleep,
            "workouts": self.workouts,
        }


def _first_non_null(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def discover_source_files(raw_dir: str | Path) -> list[Path]:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return []
    discovered: list[Path] = []
    for root, dirs, files in os.walk(raw_path):
        dirs[:] = [directory for directory in dirs if directory not in EXCLUDED_RAW_PATH_PARTS]
        root_path = Path(root)
        for filename in files:
            path = root_path / filename
            if path.suffix.lower() not in {".csv", ".xlsx", ".xlsm", ".xls", ".parquet", ".pq"}:
                continue
            if "Rproj" in path.name:
                continue
            name_lower = path.name.lower()
            if "manifest" in name_lower or "coverage" in name_lower or "summary" in name_lower:
                continue
            discovered.append(path)
    return sorted(discovered)


def _parse_datetime(series: pd.Series) -> pd.Series:
    import os
    import re
    import warnings
    from zoneinfo import ZoneInfo

    raw_values = series.tolist()
    tz_offset_re = re.compile(r"(Z|[+-]\d{2}:?\d{2})$", re.IGNORECASE)
    has_tz_hint = False
    has_naive_hint = False
    parsed_values: list[pd.Timestamp | pd.NaT] = []
    for value in raw_values:
        if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
            parsed_values.append(pd.NaT)
            continue
        try:
            ts = pd.to_datetime(value, errors="coerce", utc=False)
        except Exception:
            ts = pd.NaT
        if pd.isna(ts):
            parsed_values.append(pd.NaT)
            continue
        stamp = pd.Timestamp(ts)
        if isinstance(value, str):
            if tz_offset_re.search(value.strip()):
                has_tz_hint = True
            else:
                has_naive_hint = True
        elif stamp.tzinfo is None:
            has_naive_hint = True
        else:
            has_tz_hint = True
        parsed_values.append(stamp)

    timezone_name = os.getenv("TIMEZONE_NAME") or "UTC"
    try:
        target_tz = ZoneInfo(timezone_name)
    except Exception:  # pragma: no cover - defensive guard
        warnings.warn(
            f"Invalid TIMEZONE_NAME='{timezone_name}'. Falling back to UTC.",
            UserWarning,
            stacklevel=2,
        )
        target_tz = ZoneInfo("UTC")
        timezone_name = "UTC"

    if has_tz_hint:
        if has_naive_hint:
            warnings.warn(
                "Mixed timezone-aware and timezone-naive timestamps detected. "
                f"Assuming naive timestamps are in TIMEZONE_NAME='{timezone_name}' and converting all timestamps "
                "to that timezone before dropping timezone info.",
                UserWarning,
                stacklevel=2,
            )
        normalized: list[pd.Timestamp] = []
        for stamp in parsed_values:
            if stamp is pd.NaT or pd.isna(stamp):
                normalized.append(pd.NaT)
                continue
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize(target_tz)
            else:
                stamp = stamp.tz_convert(target_tz)
            normalized.append(stamp.tz_localize(None))
        return pd.Series(normalized, index=series.index, dtype="datetime64[ns]")

    parsed = pd.Series(parsed_values, index=series.index)
    naive = pd.to_datetime(parsed, errors="coerce")
    return pd.Series(naive.to_numpy(dtype="datetime64[ns]"), index=series.index)

def _match_column(columns: Iterable[str], candidate: str) -> str | None:
    normalized = {str(col).strip().lower(): col for col in columns}
    return normalized.get(candidate.strip().lower())


def _infer_timestamp_series(kind: str, frame: pd.DataFrame) -> pd.Series:
    if kind in {"basal", "sleep", "workouts"}:
        timestamps: list[pd.Series] = []
        start_candidates = ["start_timestamp", "sleep_start"]
        end_candidates = ["end_timestamp", "sleep_end"]
        for column in start_candidates:
            if column in frame.columns:
                timestamps.append(pd.to_datetime(frame[column], errors="coerce"))
        for column in end_candidates:
            if column in frame.columns:
                timestamps.append(pd.to_datetime(frame[column], errors="coerce"))
        if not timestamps:
            return pd.Series(dtype="datetime64[ns]")
        return pd.concat(timestamps, ignore_index=True)
    if kind == "health_measurements":
        if "timestamp" not in frame.columns:
            return pd.Series(dtype="datetime64[ns]")
        return pd.to_datetime(frame["timestamp"], errors="coerce")
    if kind == "health_activity":
        if "timestamp" not in frame.columns:
            return pd.Series(dtype="datetime64[ns]")
        return pd.to_datetime(frame["timestamp"], errors="coerce")
    if "timestamp" not in frame.columns:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(frame["timestamp"], errors="coerce")


def _coerce_glucose_columns(columns: Iterable[str], *, allow_generic_value: bool = True) -> list[tuple[str, str]]:
    candidates = list(CGM_GLUCOSE_COLUMNS)
    selected: list[tuple[str, str]] = []
    for candidate in candidates:
        matched = _match_column(columns, candidate)
        if matched is not None and candidate not in {canonical for _, canonical in selected}:
            selected.append((matched, candidate))
    if allow_generic_value and not selected:
        generic_value = _match_column(columns, "value")
        event_hints = [
            "type",
            "description",
            "eventHistoryReportEventDesc",
            "eventTypeId",
            "deviceType",
        ]
        if generic_value is not None and any(_match_column(columns, hint) is not None for hint in event_hints):
            selected.append((generic_value, "value"))
    return selected


def _build_manifest_row(kind: str, frame: pd.DataFrame, source_order: int, source_file: str, source_sheet: str | None = None) -> dict[str, object]:
    timestamps = _infer_timestamp_series(kind, frame).dropna().sort_values()
    unique_timestamps = timestamps.drop_duplicates()
    first_timestamp = unique_timestamps.iloc[0] if not unique_timestamps.empty else pd.NaT
    last_timestamp = unique_timestamps.iloc[-1] if not unique_timestamps.empty else pd.NaT
    deltas = unique_timestamps.diff().dropna().dt.total_seconds().div(60.0) if len(unique_timestamps) > 1 else pd.Series(dtype=float)
    median_step_minutes = float(deltas.median()) if not deltas.empty else np.nan
    max_step_minutes = float(deltas.max()) if not deltas.empty else np.nan
    has_duplicates = bool(len(timestamps) != len(unique_timestamps))
    if kind == "cgm":
        has_internal_gap = bool(np.isfinite(max_step_minutes) and max_step_minutes > 7.5)
    else:
        has_internal_gap = False
    is_complete_window = bool(pd.notna(first_timestamp) and pd.notna(last_timestamp) and not has_internal_gap)
    return {
        "kind": kind,
        "source_order": int(source_order),
        "source_file": source_file,
        "source_sheet": source_sheet,
        "rows": int(len(frame)),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "duration_minutes": float((last_timestamp - first_timestamp).total_seconds() / 60.0) if pd.notna(first_timestamp) and pd.notna(last_timestamp) else np.nan,
        "median_step_minutes": median_step_minutes,
        "max_step_minutes": max_step_minutes,
        "has_internal_gap": has_internal_gap,
        "has_duplicates": has_duplicates,
        "is_complete_window": is_complete_window,
    }


def build_export_manifest(data: IngestedData) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_order_map = {Path(source).name: index for index, source in enumerate(data.source_files)}
    for kind, frame in data.all_tables().items():
        if frame.empty or "source_file" not in frame.columns:
            continue
        source_sheets = "source_sheet" in frame.columns
        group_cols = ["source_file"] + (["source_sheet"] if source_sheets else [])
        for group_key, group in frame.groupby(group_cols, dropna=False):
            if source_sheets:
                source_file, source_sheet = group_key if isinstance(group_key, tuple) else (group_key, None)
            else:
                source_file, source_sheet = (group_key if not isinstance(group_key, tuple) else group_key[0]), None
            rows.append(
                _build_manifest_row(
                    kind=kind,
                    frame=group,
                    source_order=source_order_map.get(str(source_file), len(source_order_map)),
                    source_file=str(source_file),
                    source_sheet=None if source_sheet is None or (isinstance(source_sheet, float) and np.isnan(source_sheet)) else str(source_sheet),
                )
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "kind",
                "source_order",
                "source_file",
                "source_sheet",
                "rows",
                "first_timestamp",
                "last_timestamp",
                "duration_minutes",
                "median_step_minutes",
                "max_step_minutes",
                "has_internal_gap",
                "has_duplicates",
                "is_complete_window",
            ]
        )
    manifest = pd.DataFrame(rows)
    return manifest.sort_values(["source_order", "kind", "source_sheet", "first_timestamp"], na_position="last").reset_index(drop=True)


def _manifest_declared_source_files(raw_dir: str | Path) -> list[Path]:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return []

    discovered: list[Path] = []
    manifest_files = sorted(
        path
        for path in raw_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".csv", ".json"}
        and any(token in path.name.lower() for token in ("window_manifest", "normalized_manifest", "tconnectsync_manifest"))
    )
    for manifest in manifest_files:
        try:
            if manifest.suffix.lower() == ".csv":
                frame = pd.read_csv(manifest)
                if "normalized_path" in frame.columns:
                    candidates = frame["normalized_path"].dropna().astype(str).tolist()
                elif "normalized_paths" in frame.columns:
                    candidates = []
                    for value in frame["normalized_paths"].dropna().astype(str):
                        candidates.extend([part.strip() for part in value.split(",") if part.strip()])
                else:
                    candidates = []
            else:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                candidates = []
                if isinstance(payload, dict):
                    for key in ("normalized_path", "normalized_paths"):
                        value = payload.get(key)
                        if isinstance(value, str):
                            candidates.append(value)
                        elif isinstance(value, list):
                            candidates.extend([str(item) for item in value])
                    windows = payload.get("windows")
                    if isinstance(windows, list):
                        for window in windows:
                            if not isinstance(window, dict):
                                continue
                            normalized_paths = window.get("normalized_paths")
                            if isinstance(normalized_paths, dict):
                                candidates.extend([str(path) for path in normalized_paths.values() if path])
                            normalized_path = window.get("normalized_path")
                            if isinstance(normalized_path, str):
                                candidates.append(normalized_path)
            for candidate in candidates:
                candidate_path = Path(candidate).expanduser()
                if candidate_path.exists():
                    discovered.append(candidate_path)
        except Exception:
            continue
    return sorted({path.resolve() for path in discovered})


def summarize_export_manifest(manifest: pd.DataFrame, *, required_kinds: Iterable[str] = ("cgm", "bolus")) -> dict[str, object]:
    if manifest.empty:
        return {
            "manifest_rows": 0,
            "complete_windows": 0,
            "incomplete_windows": 0,
            "gap_count": 0,
            "overlap_count": 0,
            "duplicate_windows": 0,
            "out_of_order_windows": 0,
            "present_kinds": (),
            "missing_kinds": tuple(required_kinds),
            "is_complete": False,
        }

    manifest = manifest.copy()
    manifest["first_timestamp"] = pd.to_datetime(manifest["first_timestamp"], errors="coerce")
    manifest["last_timestamp"] = pd.to_datetime(manifest["last_timestamp"], errors="coerce")
    present_kinds = tuple(sorted(manifest["kind"].dropna().astype(str).unique().tolist()))
    missing_kinds = tuple(kind for kind in required_kinds if kind not in present_kinds)

    complete_windows = int(manifest["is_complete_window"].fillna(False).sum()) if "is_complete_window" in manifest.columns else 0
    incomplete_windows = int(len(manifest) - complete_windows)
    duplicate_windows = int(
        manifest.duplicated(subset=["kind", "source_file", "source_sheet", "first_timestamp", "last_timestamp"], keep=False).sum()
    )

    gap_count = 0
    overlap_count = 0
    out_of_order_windows = 0
    for kind, kind_manifest in manifest.groupby("kind"):
        if kind == "cgm":
            ordered_by_source = kind_manifest.sort_values("source_order")
            source_times = ordered_by_source["first_timestamp"].dropna()
            if len(source_times) > 1:
                out_of_order_windows += int((source_times.diff().dt.total_seconds().fillna(0) < 0).sum())
            chronological = kind_manifest.sort_values(["first_timestamp", "last_timestamp", "source_order"])
            previous_last = None
            for row in chronological.itertuples(index=False):
                if pd.isna(row.first_timestamp) or pd.isna(row.last_timestamp):
                    continue
                if previous_last is not None:
                    gap_minutes = (row.first_timestamp - previous_last).total_seconds() / 60.0
                    if gap_minutes > 7.5:
                        gap_count += 1
                    if gap_minutes < 0:
                        overlap_count += 1
                previous_last = row.last_timestamp if previous_last is None else max(previous_last, row.last_timestamp)

    is_complete = not missing_kinds and gap_count == 0 and overlap_count == 0 and duplicate_windows == 0 and out_of_order_windows == 0 and incomplete_windows == 0
    return {
        "manifest_rows": int(len(manifest)),
        "complete_windows": complete_windows,
        "incomplete_windows": incomplete_windows,
        "gap_count": gap_count,
        "overlap_count": overlap_count,
        "duplicate_windows": duplicate_windows,
        "out_of_order_windows": out_of_order_windows,
        "present_kinds": present_kinds,
        "missing_kinds": missing_kinds,
        "is_complete": is_complete,
    }


def write_export_manifest(manifest: pd.DataFrame, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False, date_format="%Y-%m-%dT%H:%M:%S")
    return out_path


def _standardize_cgm(df: pd.DataFrame, source: Path, sheet: str | None = None, *, allow_generic_value: bool = True) -> pd.DataFrame:
    ts_col = coalesce_columns(df.columns, BROAD_CGM_COLUMNS)
    glucose_cols = _coerce_glucose_columns(df.columns, allow_generic_value=allow_generic_value)
    if ts_col is None or not glucose_cols:
        return pd.DataFrame()
    selected_cols = [ts_col, *[column for column, _ in glucose_cols]]
    rename_map = {ts_col: "timestamp"}
    for column, canonical in glucose_cols:
        rename_map[column] = canonical
    out = df.loc[:, selected_cols].copy()
    out = out.rename(columns=rename_map)
    out["timestamp"] = _parse_datetime(out["timestamp"])
    for _, column in glucose_cols:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["glucose"] = out[[column for _, column in glucose_cols]].bfill(axis=1).iloc[:, 0]
    source_priority = {
        "egv_estimatedGlucoseValue": 0,
        "bg": 1,
        "glucose": 2,
        "readings (cgm / bgm)": 3,
        "value": 4,
    }
    glucose_source = pd.Series(pd.NA, index=out.index, dtype="object")
    for _, column in glucose_cols:
        mask = glucose_source.isna() & out[column].notna()
        glucose_source.loc[mask] = column
    out["glucose_source"] = glucose_source
    out["_glucose_priority"] = out["glucose_source"].map(source_priority).fillna(len(source_priority)).astype(int)
    out = out.dropna(subset=["timestamp", "glucose"])
    out["timestamp"] = out["timestamp"].dt.floor("5min")
    out["source_file"] = source.name
    if sheet is not None:
        out["source_sheet"] = sheet
    group_cols = ["timestamp"]
    agg: dict[str, tuple[str, object]] = {
        "glucose": ("glucose", "first"),
        "source_file": ("source_file", "first"),
        "glucose_source": ("glucose_source", "first"),
    }
    for _, column in glucose_cols:
        agg[column] = (column, _first_non_null)
    if sheet is not None:
        agg["source_sheet"] = ("source_sheet", "first")
    out = out.sort_values(["timestamp", "_glucose_priority"])
    out = out.drop(columns=["_glucose_priority"])
    return out.groupby(group_cols, as_index=False).agg(**agg)


def summarize_tandem_raw_source(source: str | Path, *, allow_generic_value: bool = False) -> dict[str, object]:
    path = Path(source)
    frames = _load_frames(path)
    cgm_frames: list[pd.DataFrame] = []
    glucose_columns: list[str] = []
    sheet_names: list[str] = []
    for frame, sheet in frames:
        parsed = _standardize_cgm(frame, path, sheet, allow_generic_value=allow_generic_value)
        if parsed.empty:
            continue
        cgm_frames.append(parsed)
        if sheet is not None:
            sheet_names.append(sheet)
        for column in parsed.columns:
            if column in {"timestamp", "glucose", "glucose_source", "source_file", "source_sheet"}:
                continue
            if column not in glucose_columns:
                glucose_columns.append(column)

    if cgm_frames:
        combined = pd.concat(cgm_frames, ignore_index=True)
        priority_map = {
            "egv_estimatedGlucoseValue": 0,
            "bg": 1,
            "glucose": 2,
            "readings (cgm / bgm)": 3,
            "value": 4,
        }
        combined["_glucose_priority"] = combined["glucose_source"].map(priority_map).fillna(len(priority_map)).astype(int)
        combined = combined.sort_values(["timestamp", "_glucose_priority"], na_position="last")
        combined = combined.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
        combined = combined.drop(columns=["_glucose_priority"])
        primary_stream = combined.loc[combined["glucose_source"] == "egv_estimatedGlucoseValue"].copy()
        if primary_stream.empty:
            primary_stream = combined
        timestamps = primary_stream["timestamp"].dropna().sort_values().drop_duplicates()
        deltas = timestamps.diff().dropna().dt.total_seconds().div(60.0) if len(timestamps) > 1 else pd.Series(dtype=float)
        median_spacing = float(deltas.median()) if not deltas.empty else np.nan
        first_timestamp = timestamps.iloc[0] if not timestamps.empty else pd.NaT
        last_timestamp = timestamps.iloc[-1] if not timestamps.empty else pd.NaT
    else:
        combined = pd.DataFrame()
        primary_stream = combined
        median_spacing = np.nan
        first_timestamp = pd.NaT
        last_timestamp = pd.NaT

    has_dense_cgm_stream = bool(
        not primary_stream.empty and len(primary_stream) >= 3 and np.isfinite(median_spacing) and median_spacing <= 10.0
    )
    return {
        "source_file": path.name,
        "source_path": str(path),
        "source_suffix": path.suffix.lower(),
        "sheet_count": len(frames),
        "cgm_sheet_count": len(cgm_frames),
        "cgm_rows": int(len(primary_stream)),
        "all_cgm_rows": int(len(combined)),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "duration_minutes": float((last_timestamp - first_timestamp).total_seconds() / 60.0) if pd.notna(first_timestamp) and pd.notna(last_timestamp) else np.nan,
        "median_spacing_minutes": median_spacing,
        "has_dense_cgm_stream": has_dense_cgm_stream,
        "recognized_glucose_columns": tuple(glucose_columns),
        "recognized_sheets": tuple(sheet_names),
    }


def summarize_tandem_raw_dir(raw_dir: str | Path, *, allow_generic_value: bool = False) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    sources = discover_source_files(raw_path)
    summaries = [summarize_tandem_raw_source(source, allow_generic_value=allow_generic_value) for source in sources]
    if not summaries:
        return pd.DataFrame(
            columns=[
                "source_file",
                "source_path",
                "source_suffix",
                "sheet_count",
                "cgm_sheet_count",
                "cgm_rows",
                "all_cgm_rows",
                "first_timestamp",
                "last_timestamp",
                "duration_minutes",
                "median_spacing_minutes",
                "has_dense_cgm_stream",
                "recognized_glucose_columns",
                "recognized_sheets",
            ]
        )
    frame = pd.DataFrame(summaries)
    return frame.sort_values(["has_dense_cgm_stream", "cgm_rows", "source_file"], ascending=[False, False, True]).reset_index(drop=True)


def _standardize_bolus(df: pd.DataFrame, source: Path, sheet: str | None = None) -> pd.DataFrame:
    ts_col = coalesce_columns(df.columns, BOLUS_COLUMNS)
    units_col = coalesce_columns(df.columns, BOLUS_UNITS_COLUMNS)
    carb_col = coalesce_columns(df.columns, CARB_COLUMNS)
    if ts_col is None or units_col is None:
        return pd.DataFrame()
    cols = [ts_col, units_col]
    if carb_col is not None:
        cols.append(carb_col)
    out = df.loc[:, cols].copy()
    rename_map = {ts_col: "timestamp", units_col: "bolus_units"}
    if carb_col is not None:
        rename_map[carb_col] = "carb_grams"
    out = out.rename(columns=rename_map)
    out["timestamp"] = _parse_datetime(out["timestamp"])
    out["bolus_units"] = pd.to_numeric(out["bolus_units"], errors="coerce")
    if "carb_grams" in out.columns:
        out["carb_grams"] = pd.to_numeric(out["carb_grams"], errors="coerce")
    out = out.dropna(subset=["timestamp", "bolus_units"])
    out["timestamp"] = out["timestamp"].dt.floor("5min")
    out["source_file"] = source.name
    if sheet is not None:
        out["source_sheet"] = sheet
    group_cols = ["timestamp"]
    agg = {"bolus_units": ("bolus_units", "sum"), "source_file": ("source_file", "first")}
    if "carb_grams" in out.columns:
        agg["carb_grams"] = ("carb_grams", "sum")
    if sheet is not None:
        agg["source_sheet"] = ("source_sheet", "first")
    return out.groupby(group_cols, as_index=False).agg(**agg)


def _standardize_carbs(df: pd.DataFrame, source: Path, sheet: str | None = None, *, source_label: str | None = None) -> pd.DataFrame:
    timestamp_candidates = [
        "completiondatetime",
        "eventdatetime",
        "startdatetime",
        "datetime",
        "timestamp",
        "dateTime",
        "date time",
        "time",
        "startdate",
        "date",
    ]
    ts_col = coalesce_columns(df.columns, timestamp_candidates)
    carb_col = coalesce_columns(df.columns, CARB_COLUMNS)
    if ts_col is None or carb_col is None:
        return pd.DataFrame()
    out = df.loc[:, [ts_col, carb_col]].copy()
    out = out.rename(columns={ts_col: "timestamp", carb_col: "carb_grams"})
    out["timestamp"] = _parse_datetime(out["timestamp"])
    out["carb_grams"] = pd.to_numeric(out["carb_grams"], errors="coerce")
    out = out.dropna(subset=["timestamp", "carb_grams"])
    out["timestamp"] = out["timestamp"].dt.floor("5min")
    out["source_file"] = source.name
    if sheet is not None:
        out["source_sheet"] = sheet
    if source_label is not None:
        out["source_label"] = source_label
    group_cols = ["timestamp", "source_file"]
    if sheet is not None:
        group_cols.append("source_sheet")
    if source_label is not None:
        group_cols.append("source_label")
    agg: dict[str, tuple[str, object]] = {
        "carb_grams": ("carb_grams", "sum"),
    }
    if sheet is not None:
        agg["source_sheet"] = ("source_sheet", "first")
    if source_label is not None:
        agg["source_label"] = ("source_label", "first")
    return out.groupby(group_cols, as_index=False).agg(**agg)


def _standardize_basal(df: pd.DataFrame, source: Path, sheet: str | None = None) -> pd.DataFrame:
    start_col = coalesce_columns(df.columns, ["startdatetime", "startdate", "start"])
    end_col = coalesce_columns(df.columns, ["enddatetime", "enddate", "end"])
    rate_col = coalesce_columns(df.columns, BASAL_RATE_COLUMNS)
    if rate_col is None:
        return pd.DataFrame()
    cols = [rate_col]
    if start_col is not None:
        cols.append(start_col)
    if end_col is not None:
        cols.append(end_col)
    out = df.loc[:, list(dict.fromkeys(cols))].copy()
    rename_map = {rate_col: "basal_units_per_hour"}
    if start_col is not None:
        rename_map[start_col] = "start_timestamp"
    if end_col is not None:
        rename_map[end_col] = "end_timestamp"
    out = out.rename(columns=rename_map)
    if "start_timestamp" in out.columns:
        out["start_timestamp"] = _parse_datetime(out["start_timestamp"])
    if "end_timestamp" in out.columns:
        out["end_timestamp"] = _parse_datetime(out["end_timestamp"])
    out["basal_units_per_hour"] = pd.to_numeric(out["basal_units_per_hour"], errors="coerce")
    out = out.dropna(subset=["basal_units_per_hour"])
    out["source_file"] = source.name
    if sheet is not None:
        out["source_sheet"] = sheet
    return out


def _standardize_activity(df: pd.DataFrame, source: Path, sheet: str | None = None) -> pd.DataFrame:
    ts_col = coalesce_columns(df.columns, ["startdate", "timestamp", "datetime", "eventdatetime"])
    value_col = coalesce_columns(df.columns, ACTIVITY_COLUMNS)
    if ts_col is None or value_col is None:
        return pd.DataFrame()
    out = df.loc[:, [ts_col, value_col]].copy()
    out.columns = ["timestamp", "activity_value"]
    out["timestamp"] = _parse_datetime(out["timestamp"])
    out["activity_value"] = pd.to_numeric(out["activity_value"], errors="coerce")
    out = out.dropna(subset=["timestamp", "activity_value"])
    out["timestamp"] = out["timestamp"].dt.floor("5min")
    out["source_file"] = source.name
    if sheet is not None:
        out["source_sheet"] = sheet
    return out.groupby("timestamp", as_index=False).agg(activity_value=("activity_value", "sum"), source_file=("source_file", "first"), **({"source_sheet": ("source_sheet", "first")} if sheet is not None else {}))


def _load_frames(path: Path) -> list[tuple[pd.DataFrame, str | None]]:
    if path.suffix.lower() == ".csv":
        return [(read_table(path), None)]
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        workbook = pd.read_excel(path, sheet_name=None)
        return [(frame, sheet) for sheet, frame in workbook.items()]
    if path.suffix.lower() in {".parquet", ".pq"}:
        return [(pd.read_parquet(path), None)]
    return []


def _load_canonical_normalized_table(path: Path) -> tuple[str, pd.DataFrame] | None:
    if path.parent.name != "normalized":
        return None
    stem = path.stem.lower()
    frame = read_table(path)
    if frame.empty:
        return None

    if stem == "cgm" and {"timestamp", "glucose"} <= set(frame.columns):
        out = frame.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["glucose"] = pd.to_numeric(out["glucose"], errors="coerce")
        out = out.dropna(subset=["timestamp", "glucose"]).reset_index(drop=True)
        return "cgm", out

    if stem == "bolus" and {"timestamp", "bolus_units"} <= set(frame.columns):
        out = frame.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["bolus_units"] = pd.to_numeric(out["bolus_units"], errors="coerce")
        if "carb_grams" in out.columns:
            out["carb_grams"] = pd.to_numeric(out["carb_grams"], errors="coerce")
        out = out.dropna(subset=["timestamp", "bolus_units"]).reset_index(drop=True)
        return "bolus", out

    if stem == "basal" and {"start_timestamp", "end_timestamp", "basal_units_per_hour"} <= set(frame.columns):
        out = frame.copy()
        out["start_timestamp"] = pd.to_datetime(out["start_timestamp"], errors="coerce")
        out["end_timestamp"] = pd.to_datetime(out["end_timestamp"], errors="coerce")
        out["basal_units_per_hour"] = pd.to_numeric(out["basal_units_per_hour"], errors="coerce")
        out = out.dropna(subset=["start_timestamp", "end_timestamp", "basal_units_per_hour"]).reset_index(drop=True)
        return "basal", out

    if stem == "activity" and {"timestamp", "activity_value"} <= set(frame.columns):
        out = frame.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["activity_value"] = pd.to_numeric(out["activity_value"], errors="coerce")
        out = out.dropna(subset=["timestamp", "activity_value"]).reset_index(drop=True)
        return "activity", out

    if stem == "carbs" and {"timestamp", "carb_grams"} <= set(frame.columns):
        out = frame.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["carb_grams"] = pd.to_numeric(out["carb_grams"], errors="coerce")
        out = out.dropna(subset=["timestamp", "carb_grams"]).reset_index(drop=True)
        return "carbs", out

    return None


def load_tandem_exports(raw_dir: str | Path, *, include_health_auto_export: bool = False) -> IngestedData:
    raw_path = Path(raw_dir)
    source_files = sorted({path.resolve() for path in [*discover_source_files(raw_path), *_manifest_declared_source_files(raw_path)]})
    cgm_frames: list[pd.DataFrame] = []
    bolus_frames: list[pd.DataFrame] = []
    carb_frames: list[pd.DataFrame] = []
    basal_frames: list[pd.DataFrame] = []
    activity_frames: list[pd.DataFrame] = []

    for source in source_files:
        canonical = _load_canonical_normalized_table(source)
        if canonical is not None:
            kind, frame = canonical
            if kind == "cgm":
                cgm_frames.append(frame)
            elif kind == "bolus":
                bolus_frames.append(frame)
                if "carb_grams" in frame.columns:
                    carb_frame = frame.loc[:, [column for column in ["timestamp", "carb_grams", "source_file", "source_sheet", "source_label"] if column in frame.columns]].copy()
                    carb_frame["source_label"] = carb_frame.get("source_label", "bolus")
                    carb_frame["_carb_source_priority"] = 1
                    carb_frames.append(carb_frame)
            elif kind == "basal":
                basal_frames.append(frame)
            elif kind == "activity":
                activity_frames.append(frame)
            elif kind == "carbs":
                direct_carbs = frame.copy()
                if "source_label" not in direct_carbs.columns:
                    direct_carbs["source_label"] = source.stem
                direct_carbs["_carb_source_priority"] = 0
                carb_frames.append(direct_carbs)
            continue
        for frame, sheet in _load_frames(source):
            if frame.empty:
                continue
            normalized = frame.copy()
            parsed = _standardize_cgm(normalized, source, sheet)
            if not parsed.empty:
                cgm_frames.append(parsed)
            if any(col.lower() in {"actualtotalbolusrequested", "bolus", "bolus_units", "insulin", "units"} for col in normalized.columns.astype(str)) or any(col.lower() in {"carbsize", "carbs"} for col in normalized.columns.astype(str)):
                parsed = _standardize_bolus(normalized, source, sheet)
                if not parsed.empty:
                    bolus_frames.append(parsed)
                    if "carb_grams" in parsed.columns:
                        carb_frame = parsed.loc[:, [column for column in ["timestamp", "carb_grams", "source_file", "source_sheet"] if column in parsed.columns]].copy()
                        carb_frame["source_label"] = "bolus"
                        carb_frame["_carb_source_priority"] = 1
                        carb_frames.append(carb_frame)
            if any(col.lower() in {"carbsize", "carbs", "carbohydrates", "guessedcarbohydrate", "carb_grams"} for col in normalized.columns.astype(str)):
                parsed = _standardize_carbs(normalized, source, sheet, source_label=sheet or source.name)
                if not parsed.empty:
                    parsed["_carb_source_priority"] = 0
                    carb_frames.append(parsed)
            if any(col.lower() in {"basalrate", "basal_units_per_hour", "units_per_hour", "rate"} for col in normalized.columns.astype(str)):
                parsed = _standardize_basal(normalized, source, sheet)
                if not parsed.empty:
                    basal_frames.append(parsed)
            if any(col.lower() in {"steps", "activity", "activity_value", "value"} for col in normalized.columns.astype(str)) and ("step" in source.name.lower() or "activity" in source.name.lower() or "normalized" in source.name.lower()):
                parsed = _standardize_activity(normalized, source, sheet)
                if not parsed.empty:
                    activity_frames.append(parsed)

    ingested = IngestedData(
        cgm=pd.concat(cgm_frames, ignore_index=True) if cgm_frames else pd.DataFrame(columns=["timestamp", "glucose", "source_file"]),
        bolus=pd.concat(bolus_frames, ignore_index=True) if bolus_frames else pd.DataFrame(columns=["timestamp", "bolus_units", "source_file"]),
        carbs=pd.DataFrame(columns=["timestamp", "carb_grams", "source_file"]),
        basal=pd.concat(basal_frames, ignore_index=True) if basal_frames else pd.DataFrame(columns=["start_timestamp", "end_timestamp", "basal_units_per_hour", "source_file"]),
        activity=pd.concat(activity_frames, ignore_index=True) if activity_frames else pd.DataFrame(columns=["timestamp", "activity_value", "source_file"]),
        health_measurements=pd.DataFrame(
            columns=[
                "timestamp",
                "metric",
                "stat",
                "value",
                "unit",
                "source_file",
                "source_device",
                "export_id",
                "export_timestamp",
                "source_json_filename",
                "covered_start_date",
                "covered_end_date",
            ]
        ),
        health_activity=pd.DataFrame(
            columns=[
                "timestamp",
                "activity_value",
                "source_file",
                "source_device",
                "export_id",
                "export_timestamp",
                "source_json_filename",
                "covered_start_date",
                "covered_end_date",
            ]
        ),
        sleep=pd.DataFrame(
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
                "export_id",
                "export_timestamp",
                "source_json_filename",
                "covered_start_date",
                "covered_end_date",
            ]
        ),
        workouts=pd.DataFrame(
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
                "export_id",
                "export_timestamp",
                "source_json_filename",
                "covered_start_date",
                "covered_end_date",
            ]
        ),
        source_files=source_files,
    )
    if carb_frames:
        carbs = pd.concat(carb_frames, ignore_index=True, sort=False)
        if "timestamp" in carbs.columns:
            carbs["timestamp"] = pd.to_datetime(carbs["timestamp"], errors="coerce")
        if "carb_grams" in carbs.columns:
            carbs["carb_grams"] = pd.to_numeric(carbs["carb_grams"], errors="coerce")
        sort_columns = [column for column in ["timestamp", "_carb_source_priority", "source_file", "source_label"] if column in carbs.columns]
        if sort_columns:
            carbs = carbs.sort_values(sort_columns, na_position="last")
        dedupe_subset = [column for column in ["timestamp", "carb_grams"] if column in carbs.columns]
        if dedupe_subset:
            carbs = carbs.drop_duplicates(subset=dedupe_subset, keep="first")
        carbs = carbs.drop(columns=[column for column in ["_carb_source_priority"] if column in carbs.columns])
        ingested.carbs = carbs.reset_index(drop=True)
    if include_health_auto_export:
        from .health_auto_export import load_health_auto_export_tables

        health_tables = load_health_auto_export_tables(raw_path)
        ingested.health_activity = health_tables["health_activity"]
        ingested.health_measurements = health_tables["health_measurements"]
        ingested.sleep = health_tables["sleep"]
        ingested.workouts = health_tables["workouts"]
    ingested.manifest = build_export_manifest(ingested)
    return ingested


def summarize_coverage(data: IngestedData) -> TandemCoverage:
    manifest = data.manifest if not data.manifest.empty else build_export_manifest(data)
    manifest_summary = summarize_export_manifest(manifest)
    timestamps: list[pd.Series] = []
    for frame, column in [
        (data.cgm, "timestamp"),
        (data.bolus, "timestamp"),
        (data.activity, "timestamp"),
        (data.health_activity, "timestamp"),
        (data.health_measurements, "timestamp"),
    ]:
        if column in frame.columns and not frame.empty:
            timestamps.append(frame[column])
    if "start_timestamp" in data.basal.columns and not data.basal.empty:
        timestamps.append(data.basal["start_timestamp"])
    if "end_timestamp" in data.basal.columns and not data.basal.empty:
        timestamps.append(data.basal["end_timestamp"])
    for frame in [data.sleep, data.workouts]:
        if "sleep_start" in frame.columns and not frame.empty:
            timestamps.append(frame["sleep_start"])
        if "sleep_end" in frame.columns and not frame.empty:
            timestamps.append(frame["sleep_end"])
        if "start_timestamp" in frame.columns and not frame.empty:
            timestamps.append(frame["start_timestamp"])
        if "end_timestamp" in frame.columns and not frame.empty:
            timestamps.append(frame["end_timestamp"])
    combined = pd.concat(timestamps, ignore_index=True) if timestamps else pd.Series(dtype="datetime64[ns]")
    combined = pd.to_datetime(combined, errors="coerce").dropna()
    return TandemCoverage(
        source_files=len(data.source_files),
        manifest_rows=int(manifest_summary["manifest_rows"]),
        cgm_rows=int(len(data.cgm)),
        bolus_rows=int(len(data.bolus)),
        basal_rows=int(len(data.basal)),
        activity_rows=int(len(data.activity)),
        health_activity_rows=int(len(data.health_activity)),
        health_measurement_rows=int(len(data.health_measurements)),
        sleep_rows=int(len(data.sleep)),
        workout_rows=int(len(data.workouts)),
        first_timestamp=combined.min() if not combined.empty else None,
        last_timestamp=combined.max() if not combined.empty else None,
        complete_windows=int(manifest_summary["complete_windows"]),
        incomplete_windows=int(manifest_summary["incomplete_windows"]),
        gap_count=int(manifest_summary["gap_count"]),
        overlap_count=int(manifest_summary["overlap_count"]),
        duplicate_windows=int(manifest_summary["duplicate_windows"]),
        out_of_order_windows=int(manifest_summary["out_of_order_windows"]),
        is_complete=bool(manifest_summary["is_complete"]),
    )
