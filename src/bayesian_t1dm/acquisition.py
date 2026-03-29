from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, replace
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
    discover_login_controls_from_controls,
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


class AcquisitionError(RuntimeError):
    pass


class TandemSourceClient(Protocol):
    def login(self, credentials: TandemCredentials, step_log: StepLogger | None = None) -> None: ...

    def export_daily_timeline_window(
        self,
        window: ExportWindow,
        download_dir: Path,
        step_log: StepLogger | None = None,
    ) -> Path: ...

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
        downloaded_path = Path(client.export_daily_timeline_window(window, workspace.runtime_downloads, step_log))
        if not downloaded_path.exists():
            raise AcquisitionError(f'Download step did not create an export file for {window.window_id}')
        temp_download = downloaded_path
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
                export_csv=export_csv,
                generated_at=page_map.generated_at,
                source=page_map.source,
            )
            page_map.validate()
            self.save_page_map(page_map, validate=True)
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
        from .tandem_browser import discover_timeline_controls_from_controls

        daily_timeline_nav, *_rest = discover_timeline_controls_from_controls(
            accessibility_snapshot=page_snapshot,
            control_inventory=page_inventory,
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

    def _discover_timeline_controls(self, snapshot, inventory) -> tuple[LocatorSpec, LocatorSpec, LocatorSpec]:
        from .tandem_browser import discover_timeline_controls_from_controls

        _daily_nav, start_date, end_date, export_csv = discover_timeline_controls_from_controls(
            accessibility_snapshot=snapshot,
            control_inventory=inventory,
        )
        return start_date, end_date, export_csv

    def login(self, credentials: TandemCredentials, step_log: StepLogger | None = None) -> None:
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
            return
        if self.daily_timeline_url:
            page.goto(self.daily_timeline_url, wait_until='domcontentloaded')
            return
        page_map.daily_timeline_nav.locate(page).first.click()

    def _fill_window_dates(self, window: ExportWindow) -> None:
        page = self._page
        assert page is not None
        start_value = window.start_date.isoformat()
        end_value = window.end_date.isoformat()
        page_map = self.ensure_page_map()
        page_map.start_date.locate(page).first.fill(start_value)
        page_map.end_date.locate(page).first.fill(end_value)
        try:
            page.keyboard.press('Tab')
        except Exception:
            pass

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
            self._fill_window_dates(window)
            with page.expect_download(timeout=self.timeout_ms) as download_info:
                page_map.export_csv.locate(page).first.click()
            download = download_info.value
            destination = download_dir / f"{window.window_id}{Path(download.suggested_filename).suffix or '.csv'}"
            download.save_as(str(destination))
            if step_log is not None:
                step_log.write('browser.export.downloaded', window_id=window.window_id, destination=str(destination))
            return destination
        except Exception as exc:
            self.capture_page_diagnostics(f'{window.window_id}.export-error')
            if step_log is not None:
                step_log.write('browser.export.error', window_id=window.window_id, reason=str(exc))
            raise AcquisitionError(f'Could not export Tandem window {window.window_id}: {exc}') from exc


__all__ = [
    'DEFAULT_LOGIN_URL',
    'AcquisitionError',
    'AcquisitionRecord',
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
    'discover_login_controls_from_controls',
    'discover_timeline_controls_from_controls',
    'export_daily_timeline_window',
    'generate_export_windows',
    'load_local_env_file',
    'load_tandem_credentials',
    'login_tandem_source',
]
