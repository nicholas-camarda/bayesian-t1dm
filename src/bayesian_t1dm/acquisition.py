from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field, replace
from calendar import month_abbr, monthrange
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import pandas as pd

from .io import read_table
from .paths import ProjectPaths
from .tandem_browser import (
    DEFAULT_LOGIN_URL,
    LocatorSpec,
    PageDiagnostics,
    TandemPageMap,
    capture_accessibility_snapshot,
    capture_control_inventory,
    capture_page_diagnostics,
    discover_export_confirm_from_controls,
    discover_login_controls_from_controls,
    _discover_spec,
    discover_timeline_controls_from_controls,
    discover_tandem_page_map_from_controls,
)


@dataclass(frozen=True)
class TandemCredentials:
    email: str
    password: str


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
    notes: str = ''


@dataclass(frozen=True)
class ExportArtifact:
    kind: str
    path: Path
    metadata_path: Path | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class AcquisitionError(RuntimeError):
    pass


class TandemSourceClient(Protocol):
    def login(self, credentials: TandemCredentials, step_log: StepLogger | None = None) -> None: ...

    def export_daily_timeline_window(
        self,
        window: ExportWindow,
        download_dir: Path,
        step_log: StepLogger | None = None,
    ) -> Path | ExportArtifact: ...

    def capture_screenshot(self, path: Path) -> Path: ...

    def start_trace(self) -> None: ...

    def stop_trace(self, path: Path) -> Path: ...

    def capture_page_diagnostics(self, stem: str) -> PageDiagnostics: ...


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
    email = os.getenv('TANDEM_SOURCE_EMAIL')
    password = os.getenv('TANDEM_SOURCE_PASSWORD')
    if not email or not password:
        raise AcquisitionError(
            f'Set TANDEM_SOURCE_EMAIL and TANDEM_SOURCE_PASSWORD in {env_path} or your shell environment.',
        )
    return TandemCredentials(email=email, password=password)


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


def _page_map_login_complete(page_map: TandemPageMap | None) -> bool:
    return bool(
        page_map
        and page_map.login_email is not None
        and page_map.login_password is not None
        and page_map.login_submit is not None
    )


def _login_controls_ready_from_inventory(inventory: Sequence[dict[str, object]]) -> bool:
    visible = [entry for entry in inventory if bool(entry.get('visible', True))]
    has_text_like = False
    has_password = False
    has_submit = False
    for entry in visible:
        tag = str(entry.get('tag') or '').lower()
        entry_type = str(entry.get('type') or '').lower()
        role = str(entry.get('role') or '').lower()
        autocomplete = str(entry.get('autocomplete') or '').lower()
        if entry_type == 'password' or 'password' in autocomplete:
            has_password = True
        if role in {'button', 'link'} or tag in {'button', 'a'} or entry_type in {'submit', 'button'}:
            has_submit = True
        if tag in {'textarea'} or role in {'textbox', 'combobox'}:
            if entry_type != 'password':
                has_text_like = True
        elif tag == 'input' and entry_type in {'text', 'email', 'search', 'tel', 'url', 'number', 'date'}:
            has_text_like = True
    return has_text_like and (has_password or has_submit)


def _page_looks_authenticated(inventory: Sequence[dict[str, object]]) -> bool:
    visible = [entry for entry in inventory if bool(entry.get('visible', True))]
    keywords = ('account settings', 'upload pump', 'reports', 'my pump', 'welcome')
    for entry in visible:
        haystack = ' '.join(
            [
                str(entry.get('aria_label') or ''),
                str(entry.get('placeholder') or ''),
                str(entry.get('text') or ''),
                str(entry.get('name') or ''),
                str(entry.get('title') or ''),
                str(entry.get('id') or ''),
                str(entry.get('autocomplete') or ''),
                str(entry.get('type') or ''),
            ]
        ).lower()
        if any(keyword in haystack for keyword in keywords):
            return True
    return False


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
        exported = _coerce_export_artifact(client.export_daily_timeline_window(window, workspace.runtime_downloads, step_log))
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
    'DEFAULT_LOGIN_URL',
    'AcquisitionError',
    'AcquisitionRecord',
    'ExportArtifact',
    'ExportWindow',
    'LocatorSpec',
    'PlaywrightTandemSourceClient',
    'TandemPageMap',
    'StepLogger',
    'TandemCredentials',
    'TandemSourceClient',
    'backfill_tandem_exports',
    'collect_tandem_exports',
    'capture_page_diagnostics',
    'discover_export_confirm_from_controls',
    'discover_login_controls_from_controls',
    'discover_timeline_controls_from_controls',
    'export_daily_timeline_window',
    'generate_export_windows',
    'load_local_env_file',
    'load_tandem_credentials',
    'login_tandem_source',
]
