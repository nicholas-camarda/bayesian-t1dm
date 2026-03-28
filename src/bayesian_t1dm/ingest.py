from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .io import coalesce_columns, read_table


CGM_COLUMNS = ["dateTime", "timestamp", "datetime", "eventdatetime", "date time"]
BOLUS_COLUMNS = ["completiondatetime", "datetime", "timestamp", "dateTime"]
GLUCOSE_COLUMNS = ["bg", "glucose", "readings (cgm / bgm)", "value"]
BOLUS_UNITS_COLUMNS = ["actualtotalbolusrequested", "bolus", "insulin", "units"]
CARB_COLUMNS = ["carbsize", "carbs", "carbohydrates", "guessedcarbohydrate"]
BASAL_RATE_COLUMNS = ["basalrate", "rate", "units_per_hour", "units/hour"]
ACTIVITY_COLUMNS = ["value", "steps", "activity", "count"]


@dataclass(frozen=True)
class TandemCoverage:
    source_files: int
    cgm_rows: int
    bolus_rows: int
    basal_rows: int
    activity_rows: int
    first_timestamp: pd.Timestamp | None
    last_timestamp: pd.Timestamp | None


@dataclass
class IngestedData:
    cgm: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    bolus: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    basal: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    activity: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    source_files: list[Path] = field(default_factory=list)

    def all_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "cgm": self.cgm,
            "bolus": self.bolus,
            "basal": self.basal,
            "activity": self.activity,
        }


def discover_source_files(raw_dir: str | Path) -> list[Path]:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return []
    return sorted(
        path
        for path in raw_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".csv", ".xlsx", ".xlsm", ".xls", ".parquet", ".pq"}
        and "Rproj" not in path.name
    )


def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def _standardize_cgm(df: pd.DataFrame, source: Path, sheet: str | None = None) -> pd.DataFrame:
    ts_col = coalesce_columns(df.columns, CGM_COLUMNS)
    glucose_col = coalesce_columns(df.columns, GLUCOSE_COLUMNS)
    if ts_col is None or glucose_col is None:
        return pd.DataFrame()
    out = df.loc[:, [ts_col, glucose_col]].copy()
    out.columns = ["timestamp", "glucose"]
    out["timestamp"] = _parse_datetime(out["timestamp"])
    out["glucose"] = pd.to_numeric(out["glucose"], errors="coerce")
    out = out.dropna(subset=["timestamp", "glucose"])
    out["timestamp"] = out["timestamp"].dt.floor("5min")
    out["source_file"] = source.name
    if sheet is not None:
        out["source_sheet"] = sheet
    return out.groupby("timestamp", as_index=False).agg(glucose=("glucose", "mean"), source_file=("source_file", "first"), **({"source_sheet": ("source_sheet", "first")} if sheet is not None else {}))


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


def load_tandem_exports(raw_dir: str | Path) -> IngestedData:
    raw_path = Path(raw_dir)
    source_files = discover_source_files(raw_path)
    cgm_frames: list[pd.DataFrame] = []
    bolus_frames: list[pd.DataFrame] = []
    basal_frames: list[pd.DataFrame] = []
    activity_frames: list[pd.DataFrame] = []

    for source in source_files:
        for frame, sheet in _load_frames(source):
            if frame.empty:
                continue
            normalized = frame.copy()
            if any(col.lower() in {"bg", "glucose", "readings (cgm / bgm)"} for col in normalized.columns.astype(str)):
                parsed = _standardize_cgm(normalized, source, sheet)
                if not parsed.empty:
                    cgm_frames.append(parsed)
            if any(col.lower() in {"actualtotalbolusrequested", "bolus", "units"} for col in normalized.columns.astype(str)) or any(col.lower() in {"carbsize", "carbs"} for col in normalized.columns.astype(str)):
                parsed = _standardize_bolus(normalized, source, sheet)
                if not parsed.empty:
                    bolus_frames.append(parsed)
            if any(col.lower() in {"basalrate", "units_per_hour", "rate"} for col in normalized.columns.astype(str)):
                parsed = _standardize_basal(normalized, source, sheet)
                if not parsed.empty:
                    basal_frames.append(parsed)
            if any(col.lower() in {"steps", "activity", "value"} for col in normalized.columns.astype(str)) and ("step" in source.name.lower() or "activity" in source.name.lower()):
                parsed = _standardize_activity(normalized, source, sheet)
                if not parsed.empty:
                    activity_frames.append(parsed)

    return IngestedData(
        cgm=pd.concat(cgm_frames, ignore_index=True) if cgm_frames else pd.DataFrame(columns=["timestamp", "glucose", "source_file"]),
        bolus=pd.concat(bolus_frames, ignore_index=True) if bolus_frames else pd.DataFrame(columns=["timestamp", "bolus_units", "source_file"]),
        basal=pd.concat(basal_frames, ignore_index=True) if basal_frames else pd.DataFrame(columns=["start_timestamp", "end_timestamp", "basal_units_per_hour", "source_file"]),
        activity=pd.concat(activity_frames, ignore_index=True) if activity_frames else pd.DataFrame(columns=["timestamp", "activity_value", "source_file"]),
        source_files=source_files,
    )


def summarize_coverage(data: IngestedData) -> TandemCoverage:
    timestamps: list[pd.Series] = []
    for frame, column in [(data.cgm, "timestamp"), (data.bolus, "timestamp"), (data.activity, "timestamp")]:
        if column in frame.columns and not frame.empty:
            timestamps.append(frame[column])
    if "start_timestamp" in data.basal.columns and not data.basal.empty:
        timestamps.append(data.basal["start_timestamp"])
    if "end_timestamp" in data.basal.columns and not data.basal.empty:
        timestamps.append(data.basal["end_timestamp"])
    combined = pd.concat(timestamps, ignore_index=True) if timestamps else pd.Series(dtype="datetime64[ns]")
    combined = pd.to_datetime(combined, errors="coerce").dropna()
    return TandemCoverage(
        source_files=len(data.source_files),
        cgm_rows=int(len(data.cgm)),
        bolus_rows=int(len(data.bolus)),
        basal_rows=int(len(data.basal)),
        activity_rows=int(len(data.activity)),
        first_timestamp=combined.min() if not combined.empty else None,
        last_timestamp=combined.max() if not combined.empty else None,
    )
