from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import pandas as pd

from .io import read_table
from .ingest import IngestedData, build_export_manifest, load_tandem_exports, summarize_export_manifest
from .paths import ProjectPaths


@dataclass(frozen=True)
class TandemCredentials:
    email: str
    password: str
    region: str | None = None
    timezone: str | None = None
    pump_serial: str | None = None


@dataclass(frozen=True)
class ExportWindow:
    start_date: date
    end_date: date

    @property
    def window_id(self) -> str:
        return f'{self.start_date.isoformat()}__{self.end_date.isoformat()}'

    @property
    def label(self) -> str:
        return f'{self.start_date.isoformat()} to {self.end_date.isoformat()}'

    @property
    def days(self) -> int:
        return (self.end_date - self.start_date).days + 1


@dataclass(frozen=True)
class AcquisitionRecord:
    window_id: str
    requested_start: str
    requested_end: str
    source_file: str
    cloud_file: str
    status: str
    row_count: int
    timestamp_column: str | None
    observed_first_timestamp: str | None
    observed_last_timestamp: str | None
    observed_window_days: int | None
    is_complete_window: bool
    file_size_bytes: int
    sha256: str
    trace_path: str
    screenshot_path: str
    endpoint_family: str | None = None
    source_type: str = "browser"
    source_label: str | None = None
    raw_artifact_paths_json: str = ""
    normalized_paths_json: str = ""
    row_counts_json: str = ""
    completeness_json: str = ""
    timezone: str | None = None
    pump_serial: str | None = None
    notes: str = ''


@dataclass(frozen=True)
class ExportArtifact:
    kind: str
    path: Path
    metadata_path: Path | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RawApiArtifact:
    endpoint_family: str
    path: Path
    metadata_path: Path | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedWindowResult:
    window_id: str
    requested_start: str
    requested_end: str
    endpoint_family: str
    source_label: str
    raw_artifacts: tuple[RawApiArtifact, ...] = field(default_factory=tuple)
    normalized_paths: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    observed_first_timestamp: str | None = None
    observed_last_timestamp: str | None = None
    observed_window_days: int | None = None
    is_complete_window: bool = False
    has_internal_gap: bool = False
    has_overlap: bool = False
    has_duplicates: bool = False
    activity_present: bool = False
    notes: str = ''
    manifest_path: Path | None = None
    payload_hash: str | None = None


class AcquisitionError(RuntimeError):
    pass


class TandemSourceClient(Protocol):
    def login(self, credentials: TandemCredentials, step_log: StepLogger | None = None) -> None: ...

    def export_daily_timeline_window(
        self,
        window: ExportWindow,
        download_dir: Path,
        step_log: StepLogger | None = None,
    ) -> Path | ExportArtifact | NormalizedWindowResult: ...

    def capture_screenshot(self, path: Path) -> Path: ...

    def start_trace(self) -> None: ...

    def stop_trace(self, path: Path) -> Path: ...

    def capture_page_diagnostics(self, stem: str) -> Any: ...


class StepLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: object) -> None:
        record = {
            'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            'event': event,
            **fields,
        }
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, default=str) + '\n')


def load_local_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        values[key] = value
    return values


def load_tandem_credentials(repo_root: str | Path | None = None, env_file: str | Path | None = None) -> TandemCredentials:
    root = Path(repo_root or Path.cwd()).resolve()
    env_path = Path(env_file) if env_file is not None else root / '.env'
    if env_path.exists():
        env_values = load_local_env_file(env_path)
        for key, value in env_values.items():
            os.environ.setdefault(key, value)
    email = os.getenv('TANDEM_SOURCE_EMAIL') or os.getenv('TCONNECT_EMAIL')
    password = os.getenv('TANDEM_SOURCE_PASSWORD') or os.getenv('TCONNECT_PASSWORD')
    if not email or not password:
        raise AcquisitionError(
            f'Set TANDEM_SOURCE_EMAIL/TCONNECT_EMAIL and TANDEM_SOURCE_PASSWORD/TCONNECT_PASSWORD in {env_path} or your shell environment.',
        )
    region = os.getenv('TANDEM_SOURCE_REGION') or os.getenv('TCONNECT_REGION') or 'US'
    timezone = os.getenv('TANDEM_SOURCE_TIMEZONE') or os.getenv('TIMEZONE_NAME') or os.getenv('TZ')
    pump_serial = os.getenv('TANDEM_PUMP_SERIAL') or os.getenv('TANDEM_SOURCE_PUMP_SERIAL') or os.getenv('PUMP_SERIAL_NUMBER')
    return TandemCredentials(email=email, password=password, region=region, timezone=timezone, pump_serial=pump_serial)


def _parse_date(value: str | date | datetime) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return pd.Timestamp(value).date()


def generate_export_windows(
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    *,
    window_days: int = 30,
    direction: str = 'backward',
) -> list[ExportWindow]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start > end:
        raise AcquisitionError('start_date must be on or before end_date')
    if window_days < 1:
        raise AcquisitionError('window_days must be at least 1')

    windows: list[ExportWindow] = []
    if direction == 'backward':
        current_end = end
        while current_end >= start:
            current_start = max(start, current_end - timedelta(days=window_days - 1))
            windows.append(ExportWindow(current_start, current_end))
            current_end = current_start - timedelta(days=1)
    elif direction == 'forward':
        current_start = start
        while current_start <= end:
            current_end = min(end, current_start + timedelta(days=window_days - 1))
            windows.append(ExportWindow(current_start, current_end))
            current_start = current_end + timedelta(days=1)
    else:
        raise AcquisitionError("direction must be 'backward' or 'forward'")
    return windows


def _read_frames(path: Path) -> list[tuple[pd.DataFrame, str | None]]:
    suffix = path.suffix.lower()
    if suffix == '.csv':
        return [(read_table(path), None)]
    if suffix in {'.xlsx', '.xlsm', '.xls'}:
        workbook = pd.read_excel(path, sheet_name=None)
        return [(frame, sheet) for sheet, frame in workbook.items()]
    if suffix in {'.parquet', '.pq'}:
        return [(pd.read_parquet(path), None)]
    raise AcquisitionError(f'Unsupported export file type: {path.suffix}')


def _best_timestamp_series(frame: pd.DataFrame) -> tuple[str | None, pd.Series]:
    best_column: str | None = None
    best_series = pd.Series(dtype='datetime64[ns]')
    best_count = 0
    for column in frame.columns:
        parsed = pd.to_datetime(frame[column], errors='coerce')
        count = int(parsed.notna().sum())
        if count > best_count:
            best_column = str(column)
            best_series = parsed.dropna().sort_values()
            best_count = count
    return best_column, best_series


def _infer_export_span(path: Path) -> dict[str, object]:
    best_column: str | None = None
    best_series = pd.Series(dtype='datetime64[ns]')
    best_count = 0
    row_count = 0
    for frame, _sheet in _read_frames(path):
        row_count += len(frame)
        column, series = _best_timestamp_series(frame)
        count = int(series.notna().sum())
        if count > best_count:
            best_column = column
            best_series = series
            best_count = count
    if best_series.empty:
        return {
            'timestamp_column': None,
            'observed_first_timestamp': None,
            'observed_last_timestamp': None,
            'observed_window_days': None,
            'row_count': row_count,
        }
    first_timestamp = best_series.iloc[0]
    last_timestamp = best_series.iloc[-1]
    observed_days = int((last_timestamp.date() - first_timestamp.date()).days + 1)
    return {
        'timestamp_column': best_column,
        'observed_first_timestamp': pd.Timestamp(first_timestamp),
        'observed_last_timestamp': pd.Timestamp(last_timestamp),
        'observed_window_days': observed_days,
        'row_count': row_count,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def _hash_json_value(value: Any) -> str:
    return hashlib.sha256(_stable_json_dumps(value).encode('utf-8')).hexdigest()


def _ensure_datetime(value: Any, *, timezone: str | None = None) -> pd.Timestamp | pd.NaT:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    timestamp = pd.to_datetime(value, errors='coerce')
    if pd.isna(timestamp):
        return pd.NaT
    if timezone:
        try:
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(timezone)
            else:
                timestamp = timestamp.tz_convert(timezone)
        except Exception:
            pass
    return pd.Timestamp(timestamp)


def _first_existing_key(mapping: Mapping[str, Any], candidates: Sequence[str]) -> str | None:
    normalized = {str(key).strip().lower(): str(key) for key in mapping.keys()}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    for candidate in candidates:
        key = candidate.strip().lower()
        for normalized_key, original in normalized.items():
            if key in normalized_key:
                return original
    return None


def _payload_records(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, Mapping)]
    if isinstance(payload, Mapping):
        for key in ('data', 'items', 'rows', 'results', 'records', 'events', 'measurements', 'payload'):
            value = payload.get(key)
            if isinstance(value, list):
                records = [record for record in value if isinstance(record, Mapping)]
                if records:
                    return records
        obvious_keys = {
            'timestamp',
            'dateTime',
            'eventDateTime',
            'completionDateTime',
            'startDateTime',
            'startDate',
            'start',
            'endDateTime',
            'endDate',
            'end',
        }
        lowered = {str(key).strip().lower() for key in payload.keys()}
        if lowered & {key.lower() for key in obvious_keys}:
            return [dict(payload)]
    return []


def _source_label_for_window(window: ExportWindow, endpoint_family: str) -> str:
    return f'tconnectsync::{window.window_id}::{endpoint_family}'


def _window_archive_root(workspace: ProjectPaths, window: ExportWindow, endpoint_family: str) -> Path:
    return workspace.cloud_raw / 'tconnectsync' / window.window_id / endpoint_family


def _window_normalized_root(workspace: ProjectPaths, window: ExportWindow) -> Path:
    return workspace.cloud_raw / 'tconnectsync' / window.window_id / 'normalized'


def _window_manifest_path(workspace: ProjectPaths, window: ExportWindow) -> Path:
    return workspace.cloud_raw / 'tconnectsync' / window.window_id / 'window_manifest.csv'


def _normalize_cgm_records(
    records: Sequence[Mapping[str, Any]],
    *,
    source_label: str,
    endpoint_family: str,
    timezone: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    glucose_candidates = [
        'egv_estimatedGlucoseValue',
        'estimatedGlucoseValue',
        'sensorGlucose',
        'glucose',
        'bg',
        'value',
        'reading',
        'readingValue',
    ]
    timestamp_candidates = [
        'eventDateTime',
        'dateTime',
        'timestamp',
        'datetime',
        'date time',
        'time',
    ]
    for record in records:
        ts_key = _first_existing_key(record, timestamp_candidates)
        glucose_key = _first_existing_key(record, glucose_candidates)
        if ts_key is None or glucose_key is None:
            continue
        timestamp = _ensure_datetime(record.get(ts_key), timezone=timezone)
        glucose = pd.to_numeric(pd.Series([record.get(glucose_key)]), errors='coerce').iloc[0]
        if pd.isna(timestamp) or pd.isna(glucose):
            continue
        rows.append(
            {
                'timestamp': timestamp.floor('5min'),
                'glucose': float(glucose),
                'glucose_source': glucose_key,
                'source_file': source_label,
                'source_label': endpoint_family,
            }
        )
    if not rows:
        return pd.DataFrame(columns=['timestamp', 'glucose', 'glucose_source', 'source_file', 'source_label'])
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(['timestamp', 'glucose_source']).drop_duplicates(subset=['timestamp'], keep='first').reset_index(drop=True)
    return frame


def _normalize_bolus_records(
    records: Sequence[Mapping[str, Any]],
    *,
    source_label: str,
    endpoint_family: str,
    timezone: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    timestamp_candidates = [
        'completionDateTime',
        'completiondatetime',
        'eventDateTime',
        'dateTime',
        'datetime',
        'timestamp',
        'time',
    ]
    units_candidates = [
        'actualTotalBolusRequested',
        'actualtotalbolusrequested',
        'bolusUnits',
        'bolus',
        'insulin',
        'units',
        'requestedUnits',
        'totalUnits',
    ]
    carbs_candidates = ['carbSize', 'carbs', 'carbohydrates', 'carb_grams']
    for record in records:
        ts_key = _first_existing_key(record, timestamp_candidates)
        units_key = _first_existing_key(record, units_candidates)
        if ts_key is None or units_key is None:
            continue
        timestamp = _ensure_datetime(record.get(ts_key), timezone=timezone)
        units = pd.to_numeric(pd.Series([record.get(units_key)]), errors='coerce').iloc[0]
        if pd.isna(timestamp) or pd.isna(units):
            continue
        row: dict[str, Any] = {
            'timestamp': timestamp.floor('5min'),
            'bolus_units': float(units),
            'source_file': source_label,
            'source_label': endpoint_family,
        }
        carb_key = _first_existing_key(record, carbs_candidates)
        if carb_key is not None:
            carb_value = pd.to_numeric(pd.Series([record.get(carb_key)]), errors='coerce').iloc[0]
            if not pd.isna(carb_value):
                row['carb_grams'] = float(carb_value)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=['timestamp', 'bolus_units', 'source_file', 'source_label'])
    frame = pd.DataFrame(rows)
    group_cols = ['timestamp', 'source_file', 'source_label']
    aggregations: dict[str, tuple[str, Any]] = {
        'bolus_units': ('bolus_units', 'sum'),
    }
    if 'carb_grams' in frame.columns:
        aggregations['carb_grams'] = ('carb_grams', 'sum')
    return frame.groupby(group_cols, as_index=False).agg(**aggregations)


def _normalize_basal_records(
    records: Sequence[Mapping[str, Any]],
    *,
    source_label: str,
    endpoint_family: str,
    timezone: str | None,
    window: ExportWindow,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start_candidates = ['startDateTime', 'startdatetime', 'startDate', 'start', 'timestamp', 'dateTime']
    end_candidates = ['endDateTime', 'enddatetime', 'endDate', 'end', 'stopTime', 'stopDateTime']
    rate_candidates = ['basalRate', 'basalrate', 'rate', 'units_per_hour', 'units/hour', 'unitsPerHour']
    for record in records:
        rate_key = _first_existing_key(record, rate_candidates)
        if rate_key is None:
            continue
        rate = pd.to_numeric(pd.Series([record.get(rate_key)]), errors='coerce').iloc[0]
        if pd.isna(rate):
            continue
        start_key = _first_existing_key(record, start_candidates)
        end_key = _first_existing_key(record, end_candidates)
        start = _ensure_datetime(record.get(start_key), timezone=timezone) if start_key is not None else pd.NaT
        end = _ensure_datetime(record.get(end_key), timezone=timezone) if end_key is not None else pd.NaT
        rows.append(
            {
                'start_timestamp': start,
                'end_timestamp': end,
                'basal_units_per_hour': float(rate),
                'source_file': source_label,
                'source_label': endpoint_family,
            }
        )
    if not rows:
        return pd.DataFrame(columns=['start_timestamp', 'end_timestamp', 'basal_units_per_hour', 'source_file', 'source_label'])
    frame = pd.DataFrame(rows)
    frame['start_timestamp'] = pd.to_datetime(frame['start_timestamp'], errors='coerce')
    frame['end_timestamp'] = pd.to_datetime(frame['end_timestamp'], errors='coerce')
    frame = frame.sort_values(['start_timestamp', 'end_timestamp']).reset_index(drop=True)
    if frame['start_timestamp'].notna().any():
        next_starts = frame['start_timestamp'].shift(-1)
        missing_end = frame['end_timestamp'].isna()
        frame.loc[missing_end, 'end_timestamp'] = next_starts[missing_end]
        fallback_end = pd.Timestamp(window.end_date + timedelta(days=1))
        frame['end_timestamp'] = frame['end_timestamp'].fillna(fallback_end)
    return frame


def _normalize_activity_records(
    records: Sequence[Mapping[str, Any]],
    *,
    source_label: str,
    endpoint_family: str,
    timezone: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    timestamp_candidates = ['startDateTime', 'startdatetime', 'startDate', 'start', 'timestamp', 'dateTime', 'eventDateTime']
    value_candidates = ['steps', 'activity', 'value', 'count', 'activityValue']
    for record in records:
        ts_key = _first_existing_key(record, timestamp_candidates)
        value_key = _first_existing_key(record, value_candidates)
        if ts_key is None or value_key is None:
            continue
        timestamp = _ensure_datetime(record.get(ts_key), timezone=timezone)
        value = pd.to_numeric(pd.Series([record.get(value_key)]), errors='coerce').iloc[0]
        if pd.isna(timestamp) or pd.isna(value):
            continue
        rows.append(
            {
                'timestamp': timestamp.floor('5min'),
                'activity_value': float(value),
                'source_file': source_label,
                'source_label': endpoint_family,
            }
        )
    if not rows:
        return pd.DataFrame(columns=['timestamp', 'activity_value', 'source_file', 'source_label'])
    frame = pd.DataFrame(rows)
    return frame.groupby(['timestamp', 'source_file', 'source_label'], as_index=False).agg(activity_value=('activity_value', 'sum'))


def normalize_tconnectsync_payloads(
    payloads: Mapping[str, Any],
    *,
    window: ExportWindow,
    endpoint_family: str,
    timezone: str | None = None,
    payload_hash_source: Mapping[str, Any] | None = None,
) -> tuple[IngestedData, dict[str, Any]]:
    source_label = _source_label_for_window(window, endpoint_family)
    cgm_records = _payload_records(payloads.get('cgm'))
    bolus_records = _payload_records(payloads.get('bolus'))
    basal_records = _payload_records(payloads.get('basal'))
    activity_records = _payload_records(payloads.get('activity'))

    cgm = _normalize_cgm_records(cgm_records, source_label=source_label, endpoint_family=endpoint_family, timezone=timezone)
    bolus = _normalize_bolus_records(bolus_records, source_label=source_label, endpoint_family=endpoint_family, timezone=timezone)
    basal = _normalize_basal_records(basal_records, source_label=source_label, endpoint_family=endpoint_family, timezone=timezone, window=window)
    activity = _normalize_activity_records(activity_records, source_label=source_label, endpoint_family=endpoint_family, timezone=timezone)
    if activity.empty:
        activity = pd.DataFrame(columns=['timestamp', 'activity_value', 'source_file', 'source_label'])

    ingested = IngestedData(cgm=cgm, bolus=bolus, basal=basal, activity=activity)
    ingested.manifest = build_export_manifest(ingested)
    timestamps: list[pd.Series] = []
    for frame, column in [(cgm, 'timestamp'), (bolus, 'timestamp'), (activity, 'timestamp')]:
        if column in frame.columns and not frame.empty:
            timestamps.append(pd.to_datetime(frame[column], errors='coerce'))
    if not basal.empty:
        if 'start_timestamp' in basal.columns:
            timestamps.append(pd.to_datetime(basal['start_timestamp'], errors='coerce'))
        if 'end_timestamp' in basal.columns:
            timestamps.append(pd.to_datetime(basal['end_timestamp'], errors='coerce'))
    combined = pd.concat(timestamps, ignore_index=True).dropna() if timestamps else pd.Series(dtype='datetime64[ns]')
    observed_first = combined.min() if not combined.empty else None
    observed_last = combined.max() if not combined.empty else None
    if observed_first is not None and observed_last is not None:
        observed_window_days = int((observed_last.date() - observed_first.date()).days + 1)
    else:
        observed_window_days = None
    manifest_summary = summarize_export_manifest(ingested.manifest)
    payload_hash = _hash_json_value(payload_hash_source or payloads)
    summary = {
        'source_label': source_label,
        'endpoint_family': endpoint_family,
        'payload_hash': payload_hash,
        'observed_first_timestamp': observed_first.isoformat() if isinstance(observed_first, pd.Timestamp) else None,
        'observed_last_timestamp': observed_last.isoformat() if isinstance(observed_last, pd.Timestamp) else None,
        'observed_window_days': observed_window_days,
        'row_counts': {
            'cgm': int(len(cgm)),
            'bolus': int(len(bolus)),
            'basal': int(len(basal)),
            'activity': int(len(activity)),
        },
        'manifest_summary': manifest_summary,
        'activity_present': bool(not activity.empty),
    }
    return ingested, summary


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_stable_json_dumps(value) + '\n', encoding='utf-8')
    return path


def _write_tconnectsync_window_artifacts(
    *,
    workspace: ProjectPaths,
    window: ExportWindow,
    endpoint_family: str,
    payloads: Mapping[str, Any],
    timezone: str | None = None,
    pump_serial: str | None = None,
    extra_raw_artifacts: Sequence[RawApiArtifact] = (),
    payload_hash_source: Mapping[str, Any] | None = None,
) -> NormalizedWindowResult:
    archive_root = workspace.cloud_raw / 'tconnectsync' / window.window_id
    raw_root = archive_root / 'raw'
    normalized_root = archive_root / 'normalized'
    raw_root.mkdir(parents=True, exist_ok=True)
    normalized_root.mkdir(parents=True, exist_ok=True)

    ingested, summary = normalize_tconnectsync_payloads(
        payloads,
        window=window,
        endpoint_family=endpoint_family,
        timezone=timezone,
        payload_hash_source=payload_hash_source,
    )
    source_label = summary['source_label']

    raw_artifacts: list[RawApiArtifact] = []
    raw_metadata_rows: list[dict[str, Any]] = []
    for family, payload in payloads.items():
        body_path = raw_root / f'{family}.json'
        metadata_path = raw_root / f'{family}.meta.json'
        body_path.write_text(_stable_json_dumps(payload) + '\n', encoding='utf-8')
        metadata = {
            'window_id': window.window_id,
            'requested_start': window.start_date.isoformat(),
            'requested_end': window.end_date.isoformat(),
            'endpoint_family': family,
            'timezone': timezone,
            'pump_serial': pump_serial,
            'payload_sha256': _hash_json_value(payload),
            'record_count': len(_payload_records(payload)),
            'created_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            'path': str(body_path),
        }
        _write_json(metadata_path, metadata)
        raw_artifacts.append(RawApiArtifact(endpoint_family=family, path=body_path, metadata_path=metadata_path, metadata=metadata))
        raw_metadata_rows.append({
            'window_id': window.window_id,
            'requested_start': window.start_date.isoformat(),
            'requested_end': window.end_date.isoformat(),
            'endpoint_family': family,
            'raw_path': str(body_path),
            'metadata_path': str(metadata_path),
            'payload_sha256': metadata['payload_sha256'],
            'record_count': metadata['record_count'],
        })

    for artifact in extra_raw_artifacts:
        raw_artifacts.append(artifact)

    normalized_paths: dict[str, str] = {}
    for kind, frame in ingested.all_tables().items():
        if frame.empty:
            continue
        path = normalized_root / f'{kind}.csv'
        frame.to_csv(path, index=False)
        normalized_paths[kind] = str(path)

    manifest_path = _window_manifest_path(workspace, window)
    manifest_rows: list[dict[str, Any]] = []
    for kind, frame in ingested.all_tables().items():
        if frame.empty:
            continue
        if 'timestamp' in frame.columns:
            series = pd.to_datetime(frame['timestamp'], errors='coerce').dropna()
            first_ts = series.min() if not series.empty else pd.NaT
            last_ts = series.max() if not series.empty else pd.NaT
        elif 'start_timestamp' in frame.columns or 'end_timestamp' in frame.columns:
            start_series = pd.to_datetime(frame['start_timestamp'], errors='coerce').dropna() if 'start_timestamp' in frame.columns else pd.Series(dtype='datetime64[ns]')
            end_series = pd.to_datetime(frame['end_timestamp'], errors='coerce').dropna() if 'end_timestamp' in frame.columns else pd.Series(dtype='datetime64[ns]')
            first_ts = start_series.min() if not start_series.empty else end_series.min() if not end_series.empty else pd.NaT
            last_ts = end_series.max() if not end_series.empty else start_series.max() if not start_series.empty else pd.NaT
        else:
            first_ts = pd.NaT
            last_ts = pd.NaT
        manifest_rows.append(
            {
                'window_id': window.window_id,
                'requested_start': window.start_date.isoformat(),
                'requested_end': window.end_date.isoformat(),
                'endpoint_family': endpoint_family,
                'kind': kind,
                'source_label': source_label,
                'raw_path': str(raw_root / f'{kind}.json'),
                'normalized_path': normalized_paths.get(kind),
                'row_count': int(len(frame)),
                'first_timestamp': None if pd.isna(first_ts) else pd.Timestamp(first_ts).isoformat(),
                'last_timestamp': None if pd.isna(last_ts) else pd.Timestamp(last_ts).isoformat(),
                'is_complete_window': bool(summary['manifest_summary'].get('is_complete', False)),
                'has_internal_gap': bool(summary['manifest_summary'].get('gap_count', 0)),
                'has_overlap': bool(summary['manifest_summary'].get('overlap_count', 0)),
                'has_duplicates': bool(summary['manifest_summary'].get('duplicate_windows', 0)),
                'activity_present': bool(summary['activity_present']),
                'payload_sha256': summary['payload_hash'],
            }
        )
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    return NormalizedWindowResult(
        window_id=window.window_id,
        requested_start=window.start_date.isoformat(),
        requested_end=window.end_date.isoformat(),
        endpoint_family=endpoint_family,
        source_label=source_label,
        raw_artifacts=tuple(raw_artifacts),
        normalized_paths=normalized_paths,
        row_counts=summary['row_counts'],
        observed_first_timestamp=summary['observed_first_timestamp'],
        observed_last_timestamp=summary['observed_last_timestamp'],
        observed_window_days=summary['observed_window_days'],
        is_complete_window=bool(summary['manifest_summary'].get('is_complete', False)),
        has_internal_gap=bool(summary['manifest_summary'].get('gap_count', 0)),
        has_overlap=bool(summary['manifest_summary'].get('overlap_count', 0)),
        has_duplicates=bool(summary['manifest_summary'].get('duplicate_windows', 0)),
        activity_present=bool(summary['activity_present']),
        notes='raw API payloads archived and normalized',
        manifest_path=manifest_path,
        payload_hash=summary['payload_hash'],
    )


def _ensure_pkg_resources_module() -> None:
    try:
        import pkg_resources  # noqa: F401
    except Exception:
        import importlib.metadata
        import sys
        import types

        module = types.ModuleType('pkg_resources')

        class _Distribution:
            def __init__(self, name: str) -> None:
                self.version = importlib.metadata.version(name)

        def require(name: str) -> list[_Distribution]:
            return [_Distribution(name)]

        module.require = require  # type: ignore[attr-defined]
        sys.modules['pkg_resources'] = module


def _select_tconnectsync_pump(metadata: Sequence[Mapping[str, Any]], pump_serial: str | None) -> Mapping[str, Any]:
    if not metadata:
        raise AcquisitionError('tconnectsync returned no pump metadata')
    candidates = list(metadata)
    if pump_serial:
        serial = str(pump_serial)
        matched = [entry for entry in candidates if str(entry.get('serialNumber') or entry.get('serial_number') or '') == serial]
        if matched:
            candidates = matched

    def _sort_key(entry: Mapping[str, Any]) -> pd.Timestamp:
        for key in ('maxDateWithEvents', 'lastUpload', 'minDateWithEvents'):
            value = entry.get(key)
            if value:
                if isinstance(value, Mapping):
                    value = (
                        value.get('lastUploadedAt')
                        or value.get('uploadedAt')
                        or value.get('timestamp')
                        or value.get('dateTime')
                    )
                ts = pd.to_datetime(value, utc=True, errors='coerce')
                if not pd.isna(ts):
                    return ts
        return pd.Timestamp.min.tz_localize('UTC')

    return max(candidates, key=_sort_key)


def _shift_timestamp_text(value: Any, minutes: int) -> str | None:
    timestamp = _ensure_datetime(value)
    if pd.isna(timestamp):
        return None
    return (timestamp + pd.Timedelta(minutes=minutes)).isoformat()


def _tconnectsync_event_payloads(events: Sequence[Any]) -> dict[str, list[dict[str, Any]]]:
    payloads: dict[str, list[dict[str, Any]]] = {'cgm': [], 'bolus': [], 'basal': [], 'activity': []}
    for event in events:
        record = event.todict() if hasattr(event, 'todict') else {}
        name = str(record.get('name') or type(event).__name__).upper()
        event_timestamp = record.get('eventTimestamp') or getattr(event, 'eventTimestamp', None)
        if 'CGM' in name:
            glucose_value = record.get('currentglucosedisplayvalue')
            if glucose_value is None and record.get('egv') is not None:
                glucose_value = record.get('egv')
            payloads['cgm'].append(
                {
                    'eventDateTime': event_timestamp,
                    'egv_estimatedGlucoseValue': glucose_value,
                    'source_event_id': record.get('id'),
                    'source_event_name': name,
                }
            )
        elif name == 'LID_BASAL_DELIVERY':
            commanded_rate = pd.to_numeric(pd.Series([record.get('commandedRate')]), errors='coerce').iloc[0]
            payloads['basal'].append(
                {
                    'startDateTime': event_timestamp,
                    'endDateTime': _shift_timestamp_text(event_timestamp, 5),
                    'basalRate': None if pd.isna(commanded_rate) else float(commanded_rate) / 1000.0,
                    'source_event_id': record.get('id'),
                    'source_event_name': name,
                }
            )
        elif name == 'LID_BOLUS_DELIVERY':
            delivered_total = pd.to_numeric(pd.Series([record.get('deliveredTotal')]), errors='coerce').iloc[0]
            payloads['bolus'].append(
                {
                    'completionDateTime': event_timestamp,
                    'actualTotalBolusRequested': None if pd.isna(delivered_total) else float(delivered_total) / 1000.0,
                    'source_event_id': record.get('id'),
                    'source_event_name': name,
                }
            )
    payloads['activity'] = []
    return payloads


def _canonical_export_name(window: ExportWindow, source_path: Path) -> str:
    return f'tandem_daily_timeline_{window.window_id}{source_path.suffix.lower()}'


def _coerce_export_artifact(result: Path | str | ExportArtifact) -> ExportArtifact:
    if isinstance(result, ExportArtifact):
        return result
    return ExportArtifact(kind='download', path=Path(result))


def _response_metadata(response, body: bytes | None = None) -> dict[str, object]:
    headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    content_type = headers.get('content-type')
    content_length = headers.get('content-length')
    if body is None:
        try:
            body = response.body()
        except Exception:
            body = b''
    if body is None:
        body = b''
    metadata: dict[str, object] = {
        'url': response.url,
        'method': response.request.method,
        'status': response.status,
        'content_type': content_type,
        'content_length': int(content_length) if content_length and content_length.isdigit() else None,
        'headers': headers,
        'body_size': len(body),
        'body_sha256': hashlib.sha256(body).hexdigest() if body else None,
    }
    if body:
        try:
            metadata['body_text_preview'] = body.decode('utf-8')[:500]
        except Exception:
            metadata['body_text_preview'] = None
    else:
        metadata['body_text_preview'] = None
    return metadata


def _response_suffix(metadata: dict[str, object], body: bytes) -> str:
    content_type = str(metadata.get('content_type') or '').lower()
    if 'csv' in content_type:
        return '.csv'
    if 'json' in content_type:
        return '.json'
    if 'text/' in content_type:
        try:
            decoded = body.decode('utf-8')
        except Exception:
            return '.txt'
        first_line = next((line for line in decoded.splitlines() if line.strip()), '')
        if ',' in first_line and len(first_line.split(',')) > 1:
            return '.csv'
        return '.txt'
    return '.bin'


def _write_response_artifact(
    download_dir: Path,
    window: ExportWindow,
    response,
) -> ExportArtifact:
    try:
        body = response.body()
    except Exception:
        body = b''
    metadata = _response_metadata(response, body)
    suffix = _response_suffix(metadata, body)
    body_path = download_dir / f'{window.window_id}.export-response{suffix}'
    body_path.write_bytes(body)
    metadata['body_path'] = str(body_path)
    metadata['kind'] = 'response'
    metadata_path = download_dir / f'{window.window_id}.export-response.meta.json'
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return ExportArtifact(kind='response', path=body_path, metadata_path=metadata_path, metadata=metadata)


def _export_artifact_is_tabular(artifact: ExportArtifact) -> bool:
    return artifact.path.suffix.lower() in {'.csv', '.xlsx', '.xlsm', '.xls', '.parquet', '.pq'}


def _export_artifact_error_message(
    *,
    window: ExportWindow,
    artifact: ExportArtifact | None,
    observed_responses: int,
    confirm_clicked: bool,
) -> str:
    if artifact is None:
        if confirm_clicked:
            return f'No export response observed for {window.window_id} after confirm click (responses seen: {observed_responses})'
        return f'Export dialog was incomplete for {window.window_id}; no confirm button was available'
    metadata = artifact.metadata
    return (
        f'Export returned non-CSV payload for {window.window_id}: '
        f"{artifact.path.name} ({metadata.get('content_type') or 'unknown content-type'})"
    )


def _manifest_path(workspace: ProjectPaths) -> Path:
    return workspace.cloud_raw / 'tandem_export_manifest.csv'


def _page_map_path(workspace: ProjectPaths) -> Path:
    return workspace.cloud_archive / 'tandem_page_map.json'


def _load_existing_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _existing_complete_window_ids(manifest: pd.DataFrame) -> set[str]:
    if manifest.empty or 'window_id' not in manifest.columns:
        return set()
    if 'status' in manifest.columns:
        complete = manifest['status'].astype(str).str.lower().eq('complete')
    elif 'is_complete_window' in manifest.columns:
        complete = manifest['is_complete_window'].fillna(False).astype(bool)
    else:
        complete = pd.Series([True] * len(manifest))
    return set(manifest.loc[complete, 'window_id'].astype(str))


def _write_manifest(manifest_path: Path, records: Sequence[AcquisitionRecord]) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(record) for record in records])
    frame.to_csv(manifest_path, index=False)
    return manifest_path


def _write_report(report_path: Path, workspace: ProjectPaths, records: Sequence[AcquisitionRecord]) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# Tandem Source Acquisition Summary',
        '',
        f'- code_root: {workspace.root}',
        f'- runtime_root: {workspace.runtime}',
        f'- cloud_raw: {workspace.cloud_raw}',
        f'- cloud_output: {workspace.cloud_output}',
        f'- windows_collected: {len(records)}',
        '',
        '## Windows',
    ]
    for record in records:
        lines.append(
            f"- {record.window_id}: {record.status}, "
            f"{record.requested_start} to {record.requested_end}, "
            f"observed {record.observed_first_timestamp or 'NA'} to {record.observed_last_timestamp or 'NA'}",
        )
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report_path


def _record_from_download(
    *,
    window: ExportWindow,
    source_file: Path,
    cloud_file: Path,
    validation: dict[str, object],
    trace_path: Path,
    screenshot_path: Path,
    status: str,
    notes: str = '',
) -> AcquisitionRecord:
    first_timestamp = validation.get('observed_first_timestamp')
    last_timestamp = validation.get('observed_last_timestamp')
    if isinstance(first_timestamp, pd.Timestamp):
        first_timestamp_value = first_timestamp.isoformat()
    else:
        first_timestamp_value = None
    if isinstance(last_timestamp, pd.Timestamp):
        last_timestamp_value = last_timestamp.isoformat()
    else:
        last_timestamp_value = None
    return AcquisitionRecord(
        window_id=window.window_id,
        requested_start=window.start_date.isoformat(),
        requested_end=window.end_date.isoformat(),
        source_file=source_file.name,
        cloud_file=str(cloud_file),
        status=status,
        row_count=int(validation.get('row_count', 0) or 0),
        timestamp_column=validation.get('timestamp_column') if validation.get('timestamp_column') is not None else None,
        observed_first_timestamp=first_timestamp_value,
        observed_last_timestamp=last_timestamp_value,
        observed_window_days=int(validation.get('observed_window_days')) if validation.get('observed_window_days') is not None else None,
        is_complete_window=bool(validation.get('is_complete_window', False)),
        file_size_bytes=int(source_file.stat().st_size),
        sha256=_sha256(source_file),
        trace_path=str(trace_path),
        screenshot_path=str(screenshot_path),
        notes=notes,
    )


def _record_from_normalized_result(
    *,
    window: ExportWindow,
    result: NormalizedWindowResult,
    trace_path: Path,
    screenshot_path: Path,
    source_file: Path | None = None,
    notes: str = '',
) -> AcquisitionRecord:
    row_count = int(sum(int(value) for value in result.row_counts.values()))
    raw_artifact_paths = [str(artifact.path) for artifact in result.raw_artifacts]
    normalized_paths = dict(result.normalized_paths)
    completeness = {
        'is_complete_window': bool(result.is_complete_window),
        'has_internal_gap': bool(result.has_internal_gap),
        'has_overlap': bool(result.has_overlap),
        'has_duplicates': bool(result.has_duplicates),
        'activity_present': bool(result.activity_present),
        'observed_window_days': result.observed_window_days,
    }
    return AcquisitionRecord(
        window_id=window.window_id,
        requested_start=window.start_date.isoformat(),
        requested_end=window.end_date.isoformat(),
        source_file=(source_file or result.manifest_path or Path(result.source_label)).name if (source_file or result.manifest_path or result.source_label) else window.window_id,
        cloud_file=str(result.manifest_path) if result.manifest_path is not None else str(source_file or ''),
        status='complete' if result.is_complete_window else 'incomplete',
        row_count=row_count,
        timestamp_column='timestamp',
        observed_first_timestamp=result.observed_first_timestamp,
        observed_last_timestamp=result.observed_last_timestamp,
        observed_window_days=result.observed_window_days,
        is_complete_window=bool(result.is_complete_window),
        file_size_bytes=int((result.manifest_path.stat().st_size if result.manifest_path and result.manifest_path.exists() else 0)),
        sha256=result.payload_hash or '',
        trace_path=str(trace_path),
        screenshot_path=str(screenshot_path),
        endpoint_family=result.endpoint_family,
        source_type='api',
        source_label=result.source_label,
        raw_artifact_paths_json=_stable_json_dumps(raw_artifact_paths),
        normalized_paths_json=_stable_json_dumps(normalized_paths),
        row_counts_json=_stable_json_dumps(result.row_counts),
        completeness_json=_stable_json_dumps(completeness),
        timezone=None,
        pump_serial=None,
        notes=notes or result.notes,
    )


def login_tandem_source(
    client: TandemSourceClient,
    credentials: TandemCredentials,
    step_log: StepLogger | None = None,
) -> None:
    if step_log is not None:
        step_log.write('login.start', email=credentials.email)
    client.login(credentials, step_log)
    if step_log is not None:
        step_log.write('login.complete', email=credentials.email)


def export_daily_timeline_window(
    client: TandemSourceClient,
    window: ExportWindow,
    workspace: ProjectPaths,
    step_log: StepLogger | None = None,
    resume: bool = True,
    strict: bool = True,
) -> AcquisitionRecord:
    workspace.runtime_downloads.mkdir(parents=True, exist_ok=True)
    workspace.runtime_traces.mkdir(parents=True, exist_ok=True)
    workspace.runtime_logs.mkdir(parents=True, exist_ok=True)
    workspace.cloud_raw.mkdir(parents=True, exist_ok=True)

    temp_download = workspace.runtime_downloads / f'{window.window_id}.download'
    trace_path = workspace.runtime_traces / f'{window.window_id}.trace.zip'
    screenshot_path = workspace.runtime_logs / f'{window.window_id}.png'

    if step_log is not None:
        step_log.write('window.start', window_id=window.window_id, requested_start=window.start_date.isoformat(), requested_end=window.end_date.isoformat())

    client.start_trace()
    try:
        exported = client.export_daily_timeline_window(window, workspace.runtime_downloads, step_log)
        if isinstance(exported, NormalizedWindowResult):
            try:
                client.capture_screenshot(screenshot_path)
            except Exception:
                pass
            record = _record_from_normalized_result(
                window=window,
                result=exported,
                trace_path=trace_path,
                screenshot_path=screenshot_path,
                source_file=Path(exported.manifest_path) if exported.manifest_path is not None else None,
            )
            if step_log is not None:
                step_log.write('window.complete', **asdict(record))
            if strict and not record.is_complete_window:
                raise AcquisitionError(
                    f'Export for {window.window_id} does not cover the requested window: '
                    f'{record.observed_first_timestamp} to {record.observed_last_timestamp}',
                )
            return record

        exported = _coerce_export_artifact(exported)
        if not exported.path.exists():
            raise AcquisitionError(f'Download step did not create an export file for {window.window_id}')
        if not _export_artifact_is_tabular(exported):
            raise AcquisitionError(
                _export_artifact_error_message(
                    window=window,
                    artifact=exported,
                    observed_responses=0,
                    confirm_clicked=True,
                ),
            )
        temp_download = exported.path
        validation = _infer_export_span(temp_download)
        observed_first = validation.get('observed_first_timestamp')
        observed_last = validation.get('observed_last_timestamp')
        validation['is_complete_window'] = bool(
            isinstance(observed_first, pd.Timestamp)
            and isinstance(observed_last, pd.Timestamp)
            and observed_first.date() <= window.start_date
            and observed_last.date() >= window.end_date,
        )
        cloud_file = workspace.cloud_raw / _canonical_export_name(window, temp_download)
        shutil.copy2(temp_download, cloud_file)
        record = _record_from_download(
            window=window,
            source_file=temp_download,
            cloud_file=cloud_file,
            validation=validation,
            trace_path=trace_path,
            screenshot_path=screenshot_path,
            status='complete' if validation.get('is_complete_window') else 'incomplete',
        )
        try:
            client.capture_screenshot(screenshot_path)
        except Exception:
            pass
        if step_log is not None:
            step_log.write('window.complete', **asdict(record))
        if strict and not record.is_complete_window:
            raise AcquisitionError(
                f'Export for {window.window_id} does not cover the requested window: '
                f'{record.observed_first_timestamp} to {record.observed_last_timestamp}',
            )
        return record
    except Exception as exc:
        if hasattr(client, 'capture_page_diagnostics'):
            try:
                client.capture_page_diagnostics(f"{window.window_id}.failed")
            except Exception:
                pass
        if step_log is not None:
            step_log.write('window.error', window_id=window.window_id, error=str(exc))
        raise
    finally:
        try:
            client.stop_trace(trace_path)
        except Exception:
            pass
        if step_log is not None:
            step_log.write('window.end', window_id=window.window_id)


def collect_tandem_exports(
    client: TandemSourceClient,
    windows: Iterable[ExportWindow],
    workspace: ProjectPaths,
    credentials: TandemCredentials,
    *,
    manifest_path: str | Path | None = None,
    report_path: str | Path | None = None,
    resume: bool = True,
    strict: bool = True,
    step_log: StepLogger | None = None,
) -> list[AcquisitionRecord]:
    manifest_path = Path(manifest_path or _manifest_path(workspace))
    report_path = Path(report_path or (workspace.cloud_output / 'tandem_acquisition_summary.md'))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    previous_manifest = _load_existing_manifest(manifest_path) if resume else pd.DataFrame()
    completed_ids = _existing_complete_window_ids(previous_manifest)
    records: list[AcquisitionRecord] = []
    if previous_manifest.empty is False and resume and not previous_manifest.empty:
        for row in previous_manifest.to_dict(orient='records'):
            if row.get('status') == 'complete' or bool(row.get('is_complete_window')):
                records.append(
                    AcquisitionRecord(
                        window_id=str(row.get('window_id')),
                        requested_start=str(row.get('requested_start')),
                        requested_end=str(row.get('requested_end')),
                        source_file=str(row.get('source_file')),
                        cloud_file=str(row.get('cloud_file')),
                        status=str(row.get('status', 'complete')),
                        row_count=int(row.get('row_count', 0) or 0),
                        timestamp_column=row.get('timestamp_column'),
                        observed_first_timestamp=row.get('observed_first_timestamp'),
                        observed_last_timestamp=row.get('observed_last_timestamp'),
                        observed_window_days=int(row.get('observed_window_days')) if pd.notna(row.get('observed_window_days')) else None,
                        is_complete_window=bool(row.get('is_complete_window', True)),
                        file_size_bytes=int(row.get('file_size_bytes', 0) or 0),
                        sha256=str(row.get('sha256', '')),
                        trace_path=str(row.get('trace_path', '')),
                        screenshot_path=str(row.get('screenshot_path', '')),
                        notes=str(row.get('notes', '')),
                    ),
                )

    login_tandem_source(client, credentials, step_log)
    for window in windows:
        if resume and window.window_id in completed_ids:
            if step_log is not None:
                step_log.write('window.skip', window_id=window.window_id, reason='already_complete')
            continue
        record = export_daily_timeline_window(
            client,
            window,
            workspace,
            step_log,
            strict=strict,
        )
        records.append(record)
        _write_manifest(manifest_path, records)
    _write_manifest(manifest_path, records)
    _write_report(report_path, workspace, records)
    return records


def backfill_tandem_exports(
    client: TandemSourceClient,
    *,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    workspace: ProjectPaths,
    credentials: TandemCredentials,
    window_days: int = 30,
    direction: str = 'backward',
    manifest_path: str | Path | None = None,
    report_path: str | Path | None = None,
    resume: bool = True,
    strict: bool = True,
    step_log: StepLogger | None = None,
) -> list[AcquisitionRecord]:
    windows = generate_export_windows(start_date, end_date, window_days=window_days, direction=direction)
    return collect_tandem_exports(
        client,
        windows,
        workspace,
        credentials,
        manifest_path=manifest_path,
        report_path=report_path,
        resume=resume,
        strict=strict,
        step_log=step_log,
    )


class TConnectSyncSourceClient:
    def __init__(
        self,
        workspace: ProjectPaths,
        *,
        region: str | None = None,
        timezone: str | None = None,
        pump_serial: str | None = None,
        adapter: object | None = None,
        adapter_factory: object | None = None,
        endpoint_family: str = 'tconnectsync',
    ) -> None:
        self.workspace = workspace
        self.region = region
        self.timezone = timezone
        self.pump_serial = pump_serial
        self.endpoint_family = endpoint_family
        self._adapter = adapter
        self._adapter_factory = adapter_factory
        self._credentials: TandemCredentials | None = None
        self._trace_started = False

    def __enter__(self) -> TConnectSyncSourceClient:
        self.workspace.cloud_raw.mkdir(parents=True, exist_ok=True)
        self.workspace.runtime_downloads.mkdir(parents=True, exist_ok=True)
        self.workspace.runtime_traces.mkdir(parents=True, exist_ok=True)
        self.workspace.runtime_logs.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def _load_adapter(self) -> object:
        if self._adapter is not None:
            return self._adapter
        if self._adapter_factory is not None:
            factory = self._adapter_factory
            if callable(factory):
                try:
                    self._adapter = factory(timezone=self.timezone, pump_serial=self.pump_serial)
                except TypeError:
                    self._adapter = factory()
                return self._adapter
        if self._credentials is None:
            raise AcquisitionError('Credentials must be loaded before constructing the tconnectsync adapter')
        _ensure_pkg_resources_module()
        try:
            from tconnectsync.api import TConnectApi  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise AcquisitionError(
                'Install tconnectsync to use the Tandem API acquisition path: pip install tconnectsync',
            ) from exc

        region = self._credentials.region or self.region or os.getenv('TCONNECT_REGION') or 'US'
        self._adapter = TConnectApi(self._credentials.email, self._credentials.password, region=region).tandemsource
        self.region = region
        return self._adapter

    @staticmethod
    def _call_first_method(adapter: object, method_names: Sequence[str], *args: Any, **kwargs: Any) -> Any:
        for name in method_names:
            method = getattr(adapter, name, None)
            if callable(method):
                try:
                    return method(*args, **kwargs)
                except TypeError:
                    if args and isinstance(args[0], TandemCredentials):
                        credentials = args[0]
                        try:
                            return method(email=credentials.email, password=credentials.password, **kwargs)
                        except TypeError:
                            try:
                                return method(credentials.email, credentials.password, **kwargs)
                            except TypeError:
                                return method(credentials, **kwargs)
                    if 'window' in kwargs:
                        window = kwargs['window']
                        try:
                            return method(window=window, **{key: value for key, value in kwargs.items() if key != 'window'})
                        except TypeError:
                            try:
                                return method(window, **{key: value for key, value in kwargs.items() if key != 'window'})
                            except TypeError:
                                continue
                    continue
        raise AcquisitionError(f"Adapter does not expose any of the expected methods: {', '.join(method_names)}")

    @staticmethod
    def _coerce_payload_mapping(result: Any) -> Mapping[str, Any]:
        if isinstance(result, Mapping):
            return result
        payloads = getattr(result, 'payloads', None)
        if isinstance(payloads, Mapping):
            return payloads
        values = {}
        for key in ('cgm', 'bolus', 'basal', 'activity'):
            value = getattr(result, key, None)
            if value is not None:
                values[key] = value
        if values:
            return values
        if isinstance(result, Sequence) and result and isinstance(result[0], Mapping):
            return {'cgm': list(result)}
        raise AcquisitionError('tconnectsync adapter returned an unsupported payload shape')

    def login(self, credentials: TandemCredentials, step_log: StepLogger | None = None) -> None:
        self._credentials = credentials
        if step_log is not None:
            step_log.write(
                'api.login.start',
                email=credentials.email,
                region=credentials.region or self.region or os.getenv('TCONNECT_REGION') or 'US',
                timezone=credentials.timezone or self.timezone,
                pump_serial=credentials.pump_serial or self.pump_serial,
            )
        self._load_adapter()
        if step_log is not None:
            step_log.write('api.login.complete', email=credentials.email)

    def start_trace(self) -> None:
        self._trace_started = True

    def stop_trace(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._trace_started:
            path.write_text('tconnectsync acquisition trace placeholder\n', encoding='utf-8')
            self._trace_started = False
        else:
            path.write_text('tconnectsync acquisition trace placeholder\n', encoding='utf-8')
        return path

    def capture_screenshot(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('tconnectsync acquisition screenshot placeholder\n', encoding='utf-8')
        return path

    def capture_page_diagnostics(self, stem: str) -> None:
        diagnostics_dir = self.workspace.runtime_logs / 'tconnectsync'
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        (diagnostics_dir / f'{stem}.json').write_text(
            _stable_json_dumps(
                {
                    'stem': stem,
                    'timezone': self.timezone,
                    'pump_serial': self.pump_serial,
                    'created_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                }
            )
            + '\n',
            encoding='utf-8',
        )
        return None

    def export_daily_timeline_window(
        self,
        window: ExportWindow,
        download_dir: Path,
        step_log: StepLogger | None = None,
    ) -> NormalizedWindowResult:
        adapter = self._load_adapter()
        if step_log is not None:
            step_log.write(
                'api.export.start',
                window_id=window.window_id,
                requested_start=window.start_date.isoformat(),
                requested_end=window.end_date.isoformat(),
                endpoint_family=self.endpoint_family,
            )
        if hasattr(adapter, 'pump_event_metadata') and hasattr(adapter, 'pump_events_raw') and hasattr(adapter, 'pump_events'):
            metadata = adapter.pump_event_metadata()
            pump = _select_tconnectsync_pump(metadata, self.pump_serial or (self._credentials.pump_serial if self._credentials else None))
            device_id = pump.get('tconnectDeviceId')
            if not device_id:
                raise AcquisitionError('tconnectsync pump metadata did not include a tconnectDeviceId')
            requested_start = window.start_date.isoformat()
            requested_end = window.end_date.isoformat()
            selected_pump_serial = str(
                pump.get('serialNumber')
                or self.pump_serial
                or (self._credentials.pump_serial if self._credentials else '')
            )
            selected_timezone = self.timezone or (self._credentials.timezone if self._credentials else None)
            raw_metadata_payload = {
                'window_id': window.window_id,
                'requested_start': requested_start,
                'requested_end': requested_end,
                'endpoint_family': 'pump_event_metadata',
                'selected_pump': pump,
                'created_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            }
            raw_events_payload = adapter.pump_events_raw(device_id, min_date=requested_start, max_date=requested_end)
            decoded_events = list(adapter.pump_events(device_id, min_date=requested_start, max_date=requested_end, fetch_all_event_types=True))
            payloads = _tconnectsync_event_payloads(decoded_events)
            extra_raw_artifacts = (
                RawApiArtifact(
                    endpoint_family='pump_event_metadata',
                    path=(self.workspace.cloud_raw / 'tconnectsync' / window.window_id / 'raw' / 'pump_event_metadata.json'),
                    metadata_path=(self.workspace.cloud_raw / 'tconnectsync' / window.window_id / 'raw' / 'pump_event_metadata.meta.json'),
                    metadata={
                        **raw_metadata_payload,
                        'pump_serial': selected_pump_serial,
                        'pump_device_id': str(device_id),
                    },
                ),
                RawApiArtifact(
                    endpoint_family='pump_events_raw',
                    path=(self.workspace.cloud_raw / 'tconnectsync' / window.window_id / 'raw' / 'pump_events_raw.txt'),
                    metadata_path=(self.workspace.cloud_raw / 'tconnectsync' / window.window_id / 'raw' / 'pump_events_raw.meta.json'),
                    metadata={
                        'window_id': window.window_id,
                        'requested_start': requested_start,
                        'requested_end': requested_end,
                        'endpoint_family': 'pump_events_raw',
                        'pump_serial': selected_pump_serial,
                        'pump_device_id': str(device_id),
                        'payload_sha256': hashlib.sha256(raw_events_payload.encode('utf-8')).hexdigest(),
                        'record_count': len(decoded_events),
                        'created_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                    },
                ),
            )
            result = _write_tconnectsync_window_artifacts(
                workspace=self.workspace,
                window=window,
                endpoint_family=self.endpoint_family,
                payloads=payloads,
                timezone=selected_timezone,
                pump_serial=selected_pump_serial,
                extra_raw_artifacts=extra_raw_artifacts,
                payload_hash_source={
                    'pump_event_metadata': raw_metadata_payload,
                    'pump_events_raw': raw_events_payload,
                },
            )
            for artifact in extra_raw_artifacts:
                artifact.path.parent.mkdir(parents=True, exist_ok=True)
                if artifact.path.suffix == '.txt':
                    artifact.path.write_text(raw_events_payload + '\n', encoding='utf-8')
                else:
                    _write_json(artifact.metadata_path, artifact.metadata)
                    _write_json(artifact.path, raw_metadata_payload)
            if step_log is not None:
                step_log.write(
                    'api.export.complete',
                    window_id=window.window_id,
                    manifest_path=str(result.manifest_path) if result.manifest_path else None,
                    normalized_paths=_stable_json_dumps(result.normalized_paths),
                    row_counts=_stable_json_dumps(result.row_counts),
                    is_complete_window=result.is_complete_window,
                )
            return result

        payloads = self._call_first_method(
            adapter,
            (
                'fetch_window_payloads',
                'collect_window',
                'export_window',
                'download_window',
                'pull_window',
                'get_window_payloads',
            ),
            window=window,
            download_dir=download_dir,
            timezone=self.timezone,
            pump_serial=self.pump_serial,
        )
        payload_mapping = self._coerce_payload_mapping(payloads)
        result = _write_tconnectsync_window_artifacts(
            workspace=self.workspace,
            window=window,
            endpoint_family=self.endpoint_family,
            payloads=payload_mapping,
            timezone=self.timezone,
            pump_serial=self.pump_serial,
        )
        if step_log is not None:
            step_log.write(
                'api.export.complete',
                window_id=window.window_id,
                manifest_path=str(result.manifest_path) if result.manifest_path else None,
                normalized_paths=_stable_json_dumps(result.normalized_paths),
                row_counts=_stable_json_dumps(result.row_counts),
                is_complete_window=result.is_complete_window,
            )
        return result


class PlaywrightTandemSourceClient:
    def __init__(
        self,
        workspace: ProjectPaths,
        *,
        page_map_path: str | Path | None = None,
        login_url: str = DEFAULT_LOGIN_URL,
        headless: bool = False,
        slow_mo: int = 0,
        timeout_ms: int = 60_000,
        daily_timeline_url: str | None = None,
        page_map: TandemPageMap | None = None,
    ) -> None:
        self.workspace = workspace
        self.page_map_path = Path(page_map_path) if page_map_path is not None else _page_map_path(workspace)
        self.login_url = login_url
        self.daily_timeline_url = daily_timeline_url
        self.headless = headless
        self.slow_mo = slow_mo
        self.timeout_ms = timeout_ms
        self.page_map = page_map
        self._playwright = None
        self._context = None
        self._page = None
        self._tracing_active = False

    def __enter__(self) -> PlaywrightTandemSourceClient:
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # pragma: no cover - dependency guard
            raise AcquisitionError("Install Playwright to use Tandem web automation: pip install -e '.[automation]'") from exc

        self.workspace.runtime_browser.mkdir(parents=True, exist_ok=True)
        self.workspace.runtime_downloads.mkdir(parents=True, exist_ok=True)
        self.workspace.runtime_traces.mkdir(parents=True, exist_ok=True)
        self.workspace.runtime_logs.mkdir(parents=True, exist_ok=True)
        if self.page_map is None and self.page_map_path.exists():
            self.page_map = TandemPageMap.load(self.page_map_path)
        self._playwright = sync_playwright().start()
        browser_home = self.workspace.runtime_browser_home
        browser_cache = browser_home / "Library" / "Caches"
        browser_support = browser_home / "Library" / "Application Support"
        browser_tmp = self.workspace.runtime / "tmp"
        browser_home.mkdir(parents=True, exist_ok=True)
        browser_cache.mkdir(parents=True, exist_ok=True)
        browser_support.mkdir(parents=True, exist_ok=True)
        browser_tmp.mkdir(parents=True, exist_ok=True)
        browser_env = dict(os.environ)
        browser_env.update(
            {
                "HOME": str(browser_home),
                "USERPROFILE": str(browser_home),
                "TMPDIR": str(browser_tmp),
                "TEMP": str(browser_tmp),
                "TMP": str(browser_tmp),
                "XDG_CONFIG_HOME": str(browser_home / ".config"),
                "XDG_CACHE_HOME": str(browser_cache),
                "XDG_STATE_HOME": str(browser_home / ".local" / "state"),
            }
        )
        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.workspace.runtime_browser),
            headless=self.headless,
            slow_mo=self.slow_mo,
            accept_downloads=True,
            env=browser_env,
        )
        if self._context.pages:
            self._page = self._context.pages[0]
        else:
            self._page = self._context.new_page()
        self._page.set_default_timeout(self.timeout_ms)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._context is not None:
            self._context.close()
        if self._playwright is not None:
            self._playwright.stop()

    def capture_screenshot(self, path: Path) -> Path:
        page = self._page
        assert page is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(path), full_page=True)
        return path

    def capture_page_diagnostics(self, stem: str) -> PageDiagnostics:
        page = self._page
        assert page is not None
        return capture_page_diagnostics(page, self.workspace.runtime_logs, stem)

    def start_trace(self) -> None:
        if self._context is None or self._tracing_active:
            return
        self._context.tracing.start(screenshots=True, snapshots=True, sources=True)
        self._tracing_active = True

    def stop_trace(self, path: Path) -> Path:
        if self._context is None or not self._tracing_active:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._context.tracing.stop(path=str(path))
        self._tracing_active = False
        return path

    def load_page_map(self, path: str | Path | None = None, *, validate: bool = False) -> TandemPageMap:
        page_map_path = Path(path) if path is not None else self.page_map_path
        if not page_map_path.exists():
            raise AcquisitionError(
                f'No Tandem page map found at {page_map_path}. Run `bayesian-t1dm discover` first.',
            )
        self.page_map = TandemPageMap.load(page_map_path)
        if validate and not self.page_map.is_complete():
            raise AcquisitionError(
                f'Loaded Tandem page map at {page_map_path} is partial; run `bayesian-t1dm discover` to complete it.',
            )
        return self.page_map

    def save_page_map(self, page_map: TandemPageMap, path: str | Path | None = None, *, validate: bool = True) -> Path:
        destination = Path(path) if path is not None else self.page_map_path
        self.page_map = page_map
        return page_map.save(destination, validate=validate)

    def ensure_page_map(self, credentials: TandemCredentials | None = None, *, step_log: StepLogger | None = None, bootstrap: bool = False) -> TandemPageMap:
        if self.page_map is not None:
            if self.page_map.is_complete() or bootstrap:
                return self.page_map
            page = self._page
            if page is not None and _page_looks_authenticated(capture_control_inventory(page)):
                if step_log is not None:
                    step_log.write('browser.page_map.partial_authenticated', page_map_path=str(self.page_map_path))
                return self.page_map
            raise AcquisitionError(
                f'Loaded Tandem page map at {self.page_map_path} is partial; run `bayesian-t1dm discover` to complete it.',
            )
        if self.page_map_path.exists():
            return self.load_page_map(self.page_map_path, validate=not bootstrap)
        if bootstrap:
            if credentials is None:
                raise AcquisitionError('Credentials are required to discover the Tandem page map')
            return self.discover_page_map(credentials, step_log=step_log)
        raise AcquisitionError(
            f'No Tandem page map found at {self.page_map_path}. Run `bayesian-t1dm discover` first.',
        )

    def _wait_for_login_controls(self, step_log: StepLogger | None = None) -> list[dict[str, object]]:
        page = self._page
        assert page is not None
        deadline = datetime.utcnow() + timedelta(milliseconds=self.timeout_ms)
        if step_log is not None:
            step_log.write('browser.discover.login_wait_start', timeout_ms=self.timeout_ms, poll_ms=500)
        while datetime.utcnow() <= deadline:
            inventory = capture_control_inventory(page)
            if _page_looks_authenticated(inventory):
                if step_log is not None:
                    step_log.write('browser.discover.authenticated_ready', control_count=len(inventory))
                return inventory
            if _login_controls_ready_from_inventory(inventory):
                if step_log is not None:
                    step_log.write('browser.discover.login_ready', control_count=len(inventory))
                return inventory
            page.wait_for_timeout(500)
        try:
            self.capture_page_diagnostics('discover-login-timeout')
        except Exception:
            pass
        if step_log is not None:
            step_log.write('browser.discover.login_wait_timeout', timeout_ms=self.timeout_ms)
        raise AcquisitionError(
            'Tandem Source login never exposed interactive controls after hydration; '
            'the SPA may still be loading or the page structure may have changed.',
        )

    def _authenticated_page_map(self, *, step_log: StepLogger | None = None) -> TandemPageMap:
        page_map = self.page_map
        if page_map is not None:
            if step_log is not None:
                step_log.write('browser.discover.authenticated_page_map_existing', page_map_path=str(self.page_map_path))
            return page_map
        page_map = TandemPageMap(
            login_url=self.login_url,
            daily_timeline_url=self.daily_timeline_url,
            generated_at=datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            source='playwright-discovery-authenticated',
        )
        if step_log is not None:
            step_log.write('browser.discover.authenticated_page_map_created', login_url=self.login_url)
        return page_map

    def _bootstrap_login_locators(self, page_map: TandemPageMap | None, step_log: StepLogger | None = None) -> TandemPageMap | None:
        if not _page_map_login_complete(page_map):
            return None
        if step_log is not None and page_map is not None:
            step_log.write(
                'browser.discover.bootstrap_login_locators',
                login_email=page_map.login_email.describe() if page_map.login_email else '',
                login_password=page_map.login_password.describe() if page_map.login_password else '',
                login_submit=page_map.login_submit.describe() if page_map.login_submit else '',
            )
        return page_map

    def _discover_or_bootstrap_login_map(self, credentials: TandemCredentials, step_log: StepLogger | None = None) -> TandemPageMap:
        page = self._page
        assert page is not None
        bootstrap_map = self._bootstrap_login_locators(self.page_map, step_log=step_log)
        if bootstrap_map is not None:
            page_map = replace(bootstrap_map, daily_timeline_url=bootstrap_map.daily_timeline_url or self.daily_timeline_url)
            self._login_using_page_map(credentials, page_map, step_log=step_log)
            return page_map
        login_snapshot = capture_accessibility_snapshot(page)
        login_inventory = capture_control_inventory(page)
        login_email, login_password, login_submit = discover_login_controls_from_controls(
            accessibility_snapshot=login_snapshot,
            control_inventory=login_inventory,
        )
        if step_log is not None:
            step_log.write(
                'browser.discover.bootstrap_login_locators',
                login_email=login_email.describe(),
                login_password=login_password.describe(),
                login_submit=login_submit.describe(),
            )
        page_map = TandemPageMap(
            login_url=self.login_url,
            daily_timeline_url=self.daily_timeline_url,
            login_email=login_email,
            login_password=login_password,
            login_submit=login_submit,
            generated_at=datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            source='playwright-discovery',
        )
        self._login_using_page_map(credentials, page_map, step_log=step_log)
        return page_map

    def _login_using_page_map(self, credentials: TandemCredentials, page_map: TandemPageMap, step_log: StepLogger | None = None) -> None:
        page = self._page
        assert page is not None
        try:
            if step_log is not None:
                step_log.write('browser.login.goto', url=self.login_url)
            page.goto(self.login_url, wait_until='domcontentloaded')
            page_map.login_email.locate(page).first.fill(credentials.email)
            page_map.login_password.locate(page).first.fill(credentials.password)
            page_map.login_submit.locate(page).first.click()
            try:
                page.wait_for_load_state('networkidle', timeout=self.timeout_ms)
            except Exception:
                pass
            if page_map.daily_timeline_nav is not None:
                page_map.daily_timeline_nav.locate(page).first.wait_for(timeout=self.timeout_ms)
            else:
                page.wait_for_timeout(min(self.timeout_ms, 1_500))
        except Exception as exc:
            try:
                self.capture_page_diagnostics('login-error')
            except Exception:
                pass
            raise AcquisitionError('Tandem login did not expose the Daily Timeline navigation after submit') from exc

    def discover_page_map(self, credentials: TandemCredentials, step_log: StepLogger | None = None) -> TandemPageMap:
        page = self._page
        assert page is not None
        try:
            if step_log is not None:
                step_log.write('browser.discover.start', login_url=self.login_url)
            page.goto(self.login_url, wait_until='domcontentloaded')
            login_inventory = self._wait_for_login_controls(step_log=step_log)
            login_diagnostics = self.capture_page_diagnostics('discover-login')
            if step_log is not None:
                step_log.write('browser.discover.login_page', **{k: str(v) for k, v in asdict(login_diagnostics).items()})
            if _page_looks_authenticated(login_inventory):
                page_map = self._authenticated_page_map(step_log=step_log)
            else:
                page_map = self._discover_or_bootstrap_login_map(credentials, step_log=step_log)
            if step_log is not None:
                step_log.write('browser.discover.logged_in')
            post_login_diagnostics = self.capture_page_diagnostics('discover-post-login')
            if step_log is not None:
                step_log.write('browser.discover.post_login_page', **{k: str(v) for k, v in asdict(post_login_diagnostics).items()})
            daily_timeline_nav, derived_daily_url = self._discover_daily_timeline_control(page)
            if derived_daily_url and not page_map.daily_timeline_url:
                page_map = replace(page_map, daily_timeline_url=derived_daily_url)
            page_map = replace(page_map, daily_timeline_nav=daily_timeline_nav)
            self._open_daily_timeline(page_map)
            report_tab, report_tab_url = self._discover_report_timeline_tab(page)
            if report_tab_url:
                page_map = replace(page_map, daily_timeline_url=report_tab_url)
            page_map = replace(page_map, daily_timeline_nav=report_tab)
            self._open_daily_timeline(page_map)
            timeline_diagnostics = self.capture_page_diagnostics('discover-daily-timeline')
            if step_log is not None:
                step_log.write('browser.discover.timeline_page', **{k: str(v) for k, v in asdict(timeline_diagnostics).items()})
            timeline_snapshot = capture_accessibility_snapshot(page)
            timeline_inventory = capture_control_inventory(page)
            start_date, end_date, export_csv = self._discover_timeline_controls(timeline_snapshot, timeline_inventory)
            page_map = TandemPageMap(
                login_url=page_map.login_url,
                daily_timeline_url=page_map.daily_timeline_url,
                login_email=page_map.login_email,
                login_password=page_map.login_password,
                login_submit=page_map.login_submit,
                daily_timeline_nav=page_map.daily_timeline_nav,
                start_date=start_date,
                end_date=end_date,
                export_csv_launcher=export_csv,
                generated_at=page_map.generated_at,
                source=page_map.source,
            )
            if page_map.is_complete():
                page_map.validate()
                self.save_page_map(page_map, validate=True)
            else:
                self.save_page_map(page_map, validate=False)
            if step_log is not None:
                step_log.write('browser.discover.complete', page_map_path=str(self.page_map_path))
            return page_map
        except Exception as exc:
            try:
                self.capture_page_diagnostics('discover-error')
            except Exception:
                pass
            if isinstance(exc, AcquisitionError):
                raise
            raise AcquisitionError(f'Could not discover the Tandem page map: {exc}') from exc

    def _discover_login_controls(self, snapshot, inventory) -> tuple[LocatorSpec, LocatorSpec, LocatorSpec]:
        from .tandem_browser import discover_login_controls_from_controls

        return discover_login_controls_from_controls(accessibility_snapshot=snapshot, control_inventory=inventory)

    def _discover_daily_timeline_control(self, page) -> tuple[LocatorSpec, str | None]:
        page_snapshot = capture_accessibility_snapshot(page)
        page_inventory = capture_control_inventory(page)
        daily_timeline_nav = _discover_spec(
            snapshot=page_snapshot,
            inventory=page_inventory,
            roles=("button", "link", "tab"),
            keywords=("view my reports", "reports", "daily timeline", "timeline"),
        )
        derived_url: str | None = None
        try:
            locator = daily_timeline_nav.locate(page).first
            if locator.count() > 0:
                href = locator.get_attribute('href')
                if href:
                    from urllib.parse import urljoin

                    derived_url = urljoin(page.url, href)
        except Exception:
            pass
        return daily_timeline_nav, derived_url

    def _discover_report_timeline_tab(self, page) -> tuple[LocatorSpec, str | None]:
        page_snapshot = capture_accessibility_snapshot(page)
        page_inventory = capture_control_inventory(page)
        timeline_tab = _discover_spec(
            snapshot=page_snapshot,
            inventory=page_inventory,
            roles=("button", "link", "tab"),
            keywords=("daily timeline", "timeline"),
        )
        derived_url: str | None = None
        try:
            locator = timeline_tab.locate(page).first
            if locator.count() > 0:
                href = locator.get_attribute('href')
                if href:
                    from urllib.parse import urljoin

                    derived_url = urljoin(page.url, href)
        except Exception:
            pass
        return timeline_tab, derived_url

    def _discover_timeline_controls(self, snapshot, inventory) -> tuple[LocatorSpec, LocatorSpec, LocatorSpec]:
        from .tandem_browser import discover_timeline_controls_from_controls

        _daily_nav, range_combo, select_button, export_csv = discover_timeline_controls_from_controls(
            accessibility_snapshot=snapshot,
            control_inventory=inventory,
        )
        return range_combo, select_button, export_csv

    def _export_launcher_spec(self, page_map: TandemPageMap) -> LocatorSpec:
        launcher = page_map.export_csv_launcher or page_map.export_csv
        if launcher is None:
            raise AcquisitionError('Tandem page map does not include an export launcher locator')
        return launcher

    def _discover_export_confirm(self, page, step_log: StepLogger | None = None) -> LocatorSpec:
        snapshot = capture_accessibility_snapshot(page)
        inventory = capture_control_inventory(page)
        confirm = discover_export_confirm_from_controls(
            accessibility_snapshot=snapshot,
            control_inventory=inventory,
        )
        if step_log is not None:
            step_log.write('browser.export.confirm_discovered', confirm=confirm.describe(), control_count=len(inventory))
        return confirm

    @staticmethod
    def _response_is_candidate(response) -> bool:
        content_type = str(response.headers.get('content-type') or '').lower()
        resource_type = str(getattr(response.request, 'resource_type', '') or '').lower()
        url = str(response.url or '').lower()
        if resource_type in {'image', 'stylesheet', 'font', 'script', 'media'}:
            return False
        if content_type.startswith('text/html'):
            return False
        if any(token in url for token in ('export', 'csv', 'report', 'timeline')):
            return True
        if any(token in content_type for token in ('csv', 'json', 'octet-stream', 'text/plain')):
            return True
        return resource_type in {'xhr', 'fetch', 'document'}

    def _capture_export_candidate(
        self,
        responses: Sequence[object],
        *,
        download_dir: Path,
        window: ExportWindow,
    ) -> ExportArtifact | None:
        matched = [response for response in responses if self._response_is_candidate(response)]
        if not matched:
            return None
        # Prefer the most export-like response we saw after the confirm click.
        def score(response) -> tuple[int, int]:
            content_type = str(response.headers.get('content-type') or '').lower()
            url = str(response.url or '').lower()
            kind_score = 0
            if 'csv' in content_type:
                kind_score = 4
            elif 'json' in content_type:
                kind_score = 3
            elif 'octet-stream' in content_type:
                kind_score = 2
            elif 'text/plain' in content_type:
                kind_score = 1
            url_score = sum(1 for token in ('export', 'csv', 'report', 'timeline') if token in url)
            return kind_score, url_score

        chosen = sorted(matched, key=score, reverse=True)[0]
        return _write_response_artifact(download_dir, window, chosen)

    def _dismiss_browser_warning(self) -> None:
        page = self._page
        assert page is not None
        try:
            inventory = capture_control_inventory(page)
        except Exception:
            return
        warning_present = False
        continue_present = False
        for entry in inventory:
            haystack = ' '.join(
                [
                    str(entry.get('aria_label') or ''),
                    str(entry.get('placeholder') or ''),
                    str(entry.get('text') or ''),
                    str(entry.get('name') or ''),
                    str(entry.get('title') or ''),
                    str(entry.get('id') or ''),
                ]
            ).lower()
            if 'browser not supported' in haystack:
                warning_present = True
            if entry.get('visible', True) and 'continue' in haystack:
                continue_present = True
        if warning_present and continue_present:
            try:
                page.get_by_role('button', name='Continue').click()
                page.wait_for_timeout(500)
            except Exception:
                pass

    def _update_page_map_with_export_confirm(self, page_map: TandemPageMap, confirm: LocatorSpec, *, step_log: StepLogger | None = None) -> TandemPageMap:
        if page_map.export_csv_confirm == confirm:
            return page_map
        updated = replace(page_map, export_csv_confirm=confirm)
        if step_log is not None:
            step_log.write('browser.export.confirm_saved', page_map_path=str(self.page_map_path), confirm=confirm.describe())
        try:
            self.save_page_map(updated, validate=False)
        except Exception:
            pass
        return updated

    @staticmethod
    def _parse_report_range_label(label: str | None) -> tuple[date | None, date | None]:
        if not label:
            return None, None
        text = " ".join(str(label).split())
        match = re.search(
            r"\((?P<start_month>[A-Za-z]{3})\s+(?P<start_day>\d{1,2})\s+-\s+(?:(?P<end_month>[A-Za-z]{3})\s+)?(?P<end_day>\d{1,2}),\s+(?P<year>\d{4})\)",
            text,
        )
        if not match:
            return None, None
        start_month = match.group("start_month")
        end_month = match.group("end_month") or start_month
        try:
            start_month_num = list(month_abbr).index(start_month)
            end_month_num = list(month_abbr).index(end_month)
        except ValueError:
            return None, None
        year = int(match.group("year"))
        start = date(year, start_month_num, int(match.group("start_day")))
        end = date(year, end_month_num, int(match.group("end_day")))
        return start, end

    def _navigate_picker_months(self, month_delta: int) -> None:
        page = self._page
        assert page is not None
        if month_delta > 0:
            for _ in range(month_delta):
                page.get_by_role('button', name='Next month').click()
                page.wait_for_timeout(200)
        elif month_delta < 0:
            for _ in range(-month_delta):
                page.get_by_role('button', name='Previous month').click()
                page.wait_for_timeout(200)

    def _select_picker_day(self, day: int, *, occurrence: int = 0) -> None:
        page = self._page
        assert page is not None
        page.get_by_role('gridcell', name=str(day)).nth(occurrence).click()
        page.wait_for_timeout(200)

    def _set_custom_date_range(self, window: ExportWindow, page_map: TandemPageMap) -> None:
        page = self._page
        assert page is not None
        current_label = None
        try:
            inventory = capture_control_inventory(page)
        except Exception:
            inventory = []
        combobox_entries = [entry for entry in inventory if str(entry.get('role') or '').lower() == 'combobox' and bool(entry.get('visible', True))]
        preferred_entries = [
            entry
            for entry in combobox_entries
            if any(token in ' '.join(str(entry.get(key) or '') for key in ('text', 'aria_label', 'title', 'name')).lower() for token in ('week', 'custom', 'range', '('))
        ]
        if preferred_entries:
            current_label = ' '.join(str(preferred_entries[0].get(key) or '') for key in ('text', 'aria_label', 'title', 'name'))
        elif combobox_entries:
            current_label = ' '.join(str(combobox_entries[0].get(key) or '') for key in ('text', 'aria_label', 'title', 'name'))
        if not current_label:
            try:
                current_label = page_map.start_date.locate(page).first.text_content(timeout=min(self.timeout_ms, 5_000))
            except Exception:
                current_label = None
        current_start, _current_end = self._parse_report_range_label(current_label)
        if current_start is None:
            current_start = window.start_date
        month_delta = (window.start_date.year - current_start.year) * 12 + (window.start_date.month - current_start.month)
        page_map.start_date.locate(page).first.click()
        page.wait_for_timeout(250)
        page.get_by_role('option', name='Custom').click()
        page.wait_for_timeout(500)
        self._navigate_picker_months(month_delta)
        self._select_picker_day(window.start_date.day, occurrence=0)
        start_month_days = monthrange(window.start_date.year, window.start_date.month)[1]
        end_occurrence = 1 if window.start_date.month != window.end_date.month and window.end_date.day <= start_month_days else 0
        self._select_picker_day(window.end_date.day, occurrence=end_occurrence)
        try:
            page_map.end_date.locate(page).first.click()
        except Exception:
            page.get_by_role('button', name='Select').click()
        page.wait_for_timeout(500)

    def login(self, credentials: TandemCredentials, step_log: StepLogger | None = None) -> None:
        page = self._page
        assert page is not None
        current_inventory = capture_control_inventory(page)
        if _page_looks_authenticated(current_inventory):
            if self.page_map is None and self.page_map_path.exists():
                self.load_page_map(self.page_map_path, validate=False)
            elif self.page_map is None:
                self.page_map = self._authenticated_page_map(step_log=step_log)
            if step_log is not None:
                step_log.write('browser.login.complete', page_map_path=str(self.page_map_path), mode='authenticated-session')
            return
        if self.page_map is not None and not self.page_map.is_complete():
            if credentials is None:
                raise AcquisitionError(
                    f'Loaded Tandem page map at {self.page_map_path} is partial; run `bayesian-t1dm discover` to complete it.',
                )
            self.discover_page_map(credentials, step_log=step_log)
            if step_log is not None:
                step_log.write('browser.login.complete', page_map_path=str(self.page_map_path), mode='bootstrapped-partial-map')
            return
        if self.page_map is None and not self.page_map_path.exists():
            self.discover_page_map(credentials, step_log=step_log)
            if step_log is not None:
                step_log.write('browser.login.complete', page_map_path=str(self.page_map_path), mode='discovered')
            return
        page_map = self.ensure_page_map(credentials, step_log=step_log, bootstrap=False)
        self._login_using_page_map(credentials, page_map, step_log=step_log)
        if step_log is not None:
            step_log.write('browser.login.complete', page_map_path=str(self.page_map_path))

    def _open_daily_timeline(self, page_map: TandemPageMap) -> None:
        page = self._page
        assert page is not None
        if page_map.daily_timeline_url:
            page.goto(page_map.daily_timeline_url, wait_until='domcontentloaded')
            try:
                page.wait_for_load_state('networkidle', timeout=self.timeout_ms)
            except Exception:
                pass
            page.wait_for_timeout(min(self.timeout_ms, 1_500))
            self._dismiss_browser_warning()
            return
        if self.daily_timeline_url:
            page.goto(self.daily_timeline_url, wait_until='domcontentloaded')
            try:
                page.wait_for_load_state('networkidle', timeout=self.timeout_ms)
            except Exception:
                pass
            page.wait_for_timeout(min(self.timeout_ms, 1_500))
            self._dismiss_browser_warning()
            return
        page_map.daily_timeline_nav.locate(page).first.click()
        try:
            page.wait_for_load_state('networkidle', timeout=self.timeout_ms)
        except Exception:
            pass
        page.wait_for_timeout(min(self.timeout_ms, 1_500))
        self._dismiss_browser_warning()

    def export_daily_timeline_window(
        self,
        window: ExportWindow,
        download_dir: Path,
        step_log: StepLogger | None = None,
    ) -> Path:
        page = self._page
        assert page is not None
        download_dir.mkdir(parents=True, exist_ok=True)
        if step_log is not None:
            step_log.write('browser.export.goto', window_id=window.window_id)
        page_map = self.ensure_page_map()
        self._open_daily_timeline(page_map)
        try:
            self._set_custom_date_range(window, page_map)
            launcher = self._export_launcher_spec(page_map)
            launcher.locate(page).first.click()
            page.wait_for_timeout(min(self.timeout_ms, 1_500))
            try:
                confirm = page_map.export_csv_confirm or self._discover_export_confirm(page, step_log=step_log)
            except Exception as exc:
                try:
                    self.capture_page_diagnostics(f'{window.window_id}.export-dialog-incomplete')
                except Exception:
                    pass
                raise AcquisitionError(f'Export dialog was incomplete for {window.window_id}: {exc}') from exc
            page_map = self._update_page_map_with_export_confirm(page_map, confirm, step_log=step_log)
            responses: list[object] = []
            downloads: list[object] = []
            collecting = True

            def on_response(response) -> None:
                if collecting:
                    responses.append(response)

            def on_download(download) -> None:
                if collecting:
                    downloads.append(download)

            page.on('response', on_response)
            page.on('download', on_download)
            try:
                try:
                    confirm.locate(page).first.click()
                except Exception as exc:
                    try:
                        self.capture_page_diagnostics(f'{window.window_id}.export-confirm-error')
                    except Exception:
                        pass
                    raise AcquisitionError(f'Could not click export confirm for {window.window_id}: {exc}') from exc
                page.wait_for_timeout(min(self.timeout_ms, 3_000))
            finally:
                collecting = False
                remove_listener = getattr(page, 'remove_listener', None) or getattr(page, 'off', None)
                if callable(remove_listener):
                    try:
                        remove_listener('response', on_response)
                    except Exception:
                        pass
                    try:
                        remove_listener('download', on_download)
                    except Exception:
                        pass

            artifact: ExportArtifact | None = None
            if downloads:
                download = downloads[0]
                suggested_filename = getattr(download, 'suggested_filename', '') or ''
                suffix = Path(suggested_filename).suffix or '.csv'
                destination = download_dir / f'{window.window_id}{suffix}'
                download.save_as(str(destination))
                artifact = ExportArtifact(kind='download', path=destination, metadata={'suggested_filename': suggested_filename})
                if step_log is not None:
                    step_log.write('browser.export.downloaded', window_id=window.window_id, destination=str(destination))
            else:
                artifact = self._capture_export_candidate(responses, download_dir=download_dir, window=window)
                if artifact is not None and step_log is not None:
                    step_log.write(
                        'browser.export.response_captured',
                        window_id=window.window_id,
                        path=str(artifact.path),
                        metadata_path=str(artifact.metadata_path) if artifact.metadata_path else None,
                        content_type=artifact.metadata.get('content_type'),
                        status=artifact.metadata.get('status'),
                    )

            if artifact is None:
                try:
                    self.capture_page_diagnostics(f'{window.window_id}.export-missing-response')
                except Exception:
                    pass
                raise AcquisitionError(
                    _export_artifact_error_message(
                        window=window,
                        artifact=None,
                        observed_responses=len(responses),
                        confirm_clicked=True,
                    ),
                )

            if not _export_artifact_is_tabular(artifact):
                try:
                    self.capture_page_diagnostics(f'{window.window_id}.export-non-csv-response')
                except Exception:
                    pass
                raise AcquisitionError(
                    _export_artifact_error_message(
                        window=window,
                        artifact=artifact,
                        observed_responses=len(responses),
                        confirm_clicked=True,
                    ),
                )

            return artifact.path
        except Exception as exc:
            self.capture_page_diagnostics(f'{window.window_id}.export-error')
            if step_log is not None:
                step_log.write('browser.export.error', window_id=window.window_id, reason=str(exc))
            raise AcquisitionError(f'Could not export Tandem window {window.window_id}: {exc}') from exc


__all__ = [
    'AcquisitionError',
    'AcquisitionRecord',
    'ExportArtifact',
    'ExportWindow',
    'NormalizedWindowResult',
    'RawApiArtifact',
    'TConnectSyncSourceClient',
    'StepLogger',
    'TandemCredentials',
    'TandemSourceClient',
    'backfill_tandem_exports',
    'collect_tandem_exports',
    'export_daily_timeline_window',
    'generate_export_windows',
    'load_local_env_file',
    'load_tandem_exports',
    'load_tandem_credentials',
    'login_tandem_source',
    'normalize_tconnectsync_payloads',
]
