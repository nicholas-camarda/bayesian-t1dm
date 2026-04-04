from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterator, Mapping
import warnings

from .paths import ProjectPaths

LOGGER_NAME = "bayesian_t1dm"
REDACTED = "[REDACTED]"

_SENSITIVE_KEY_FRAGMENTS = (
    "password",
    "secret",
    "token",
    "auth",
    "cookie",
    "credential",
    "email",
    "serial",
)
_RAW_DATA_KEY_FRAGMENTS = (
    "payload",
    "body",
    "headers",
    "glucose",
    "insulin",
    "heart_rate",
    "steps",
    "values",
)
_SENSITIVE_OPTION_FRAGMENTS = ("password", "secret", "token", "auth")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SAFE_SUMMARY_SUFFIXES = ("_rows", "_count", "_counts", "_seconds", "_minutes", "_hours", "_days")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_timestamp(value: datetime | None = None) -> str:
    return (value or _utc_now()).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if isinstance(value, Mapping):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _key_has_fragment(key: str, fragments: tuple[str, ...]) -> bool:
    lower_key = key.lower()
    return any(fragment in lower_key for fragment in fragments)


def _looks_like_email(value: Any) -> bool:
    return isinstance(value, str) and bool(_EMAIL_RE.match(value.strip()))


def _is_safe_summary_field(key: str, value: Any) -> bool:
    lower_key = key.lower()
    return lower_key.endswith(_SAFE_SUMMARY_SUFFIXES) and isinstance(value, (int, float, bool))


def redact_value(key: str, value: Any, *, unsafe_debug_logging: bool = False) -> Any:
    if _key_has_fragment(key, _SENSITIVE_KEY_FRAGMENTS):
        return REDACTED
    if _looks_like_email(value):
        return REDACTED
    if _is_safe_summary_field(key, value):
        return _json_safe(value)
    if _key_has_fragment(key, _RAW_DATA_KEY_FRAGMENTS) and not unsafe_debug_logging:
        return REDACTED
    if isinstance(value, Mapping):
        return {str(inner_key): redact_value(str(inner_key), inner_value, unsafe_debug_logging=unsafe_debug_logging) for inner_key, inner_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        if not unsafe_debug_logging and (_key_has_fragment(key, _RAW_DATA_KEY_FRAGMENTS) or len(value) > 20):
            return REDACTED
        return [_json_safe(redact_value(key, item, unsafe_debug_logging=unsafe_debug_logging)) for item in value]
    return _json_safe(value)


def sanitize_fields(fields: Mapping[str, Any], *, unsafe_debug_logging: bool = False) -> dict[str, Any]:
    return {
        str(key): redact_value(str(key), value, unsafe_debug_logging=unsafe_debug_logging)
        for key, value in fields.items()
    }


def sanitize_argv(argv: list[str]) -> list[str]:
    sanitized: list[str] = []
    redact_next = False
    for arg in argv:
        if redact_next:
            sanitized.append(REDACTED)
            redact_next = False
            continue
        stripped = arg.lstrip("-").lower()
        if any(fragment in stripped for fragment in _SENSITIVE_OPTION_FRAGMENTS):
            sanitized.append(arg)
            redact_next = True
            continue
        sanitized.append(REDACTED if _looks_like_email(arg) else arg)
    return sanitized


class JsonlEventHandler(logging.Handler):
    def __init__(self, path: Path, *, command: str, run_id: str) -> None:
        super().__init__(level=logging.DEBUG)
        self._command = command
        self._run_id = run_id
        self._handle = path.open("a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        payload = {
            "ts": _utc_timestamp(datetime.fromtimestamp(record.created, tz=UTC)),
            "level": record.levelname,
            "event": getattr(record, "event_name", "log"),
            "logger": record.name,
            "command": self._command,
            "run_id": self._run_id,
            "stage": getattr(record, "stage", None),
            "status": getattr(record, "status", None),
            "message": record.getMessage(),
            "fields": _json_safe(getattr(record, "event_fields", {})),
        }
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        try:
            self._handle.close()
        finally:
            super().close()


class HumanReadableFormatter(logging.Formatter):
    def _format_stage(self, stage: str | None) -> str | None:
        if not stage:
            return None
        label = stage.split(".")[-1].replace("_", " ").strip()
        return label or None

    def _format_event(self, event_name: str | None) -> str | None:
        if not event_name:
            return None
        if event_name in {"command.start", "command.complete", "command.warning", "command.error"}:
            return None
        if event_name in {"command.stage.start", "command.stage.complete"}:
            return None
        return event_name.split(".")[-1].replace("_", " ").strip()

    def _format_scalar(self, value: Any) -> str:
        if value == REDACTED:
            return REDACTED
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            if value.is_integer():
                return f"{int(value):,}"
            return f"{value:,.3f}".rstrip("0").rstrip(".")
        if isinstance(value, str):
            text = value.replace(str(Path.home()), "~")
            return text
        return str(value)

    def _format_fields(self, fields: Mapping[str, Any]) -> str:
        if not fields:
            return ""
        preferred_keys = (
            "tandem_cgm_rows",
            "tandem_bolus_rows",
            "health_activity_rows",
            "health_measurement_rows",
            "sleep_rows",
            "workout_rows",
            "needs_backfill",
            "backfill_status",
            "apple_available",
            "final_row_count",
            "output_path",
            "report_path",
            "raw_root",
            "warning_source",
        )
        ordered_keys = [key for key in preferred_keys if key in fields]
        ordered_keys.extend(key for key in fields if key not in ordered_keys)
        rendered: list[str] = []
        for key in ordered_keys[:5]:
            value = fields[key]
            if isinstance(value, (dict, list, tuple, set)):
                continue
            rendered.append(f"{key}={self._format_scalar(value)}")
        return "  ".join(rendered)

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        stage = self._format_stage(getattr(record, "stage", None))
        status = getattr(record, "status", None)
        event_name = getattr(record, "event_name", None)
        fields = getattr(record, "event_fields", {})
        message = record.getMessage().replace(str(Path.home()), "~")
        event_label = self._format_event(event_name)
        parts = [timestamp, f"{record.levelname:<7}"]
        if stage:
            parts.append(f"{stage}:")
        if event_name == "command.stage.start":
            text = " ".join(parts + ["starting"])
        elif event_name == "command.stage.complete":
            text = " ".join(parts + ["done"])
        elif event_name == "command.start":
            text = " ".join(parts + [message])
        elif event_name == "command.complete":
            text = " ".join(parts + [message])
        elif event_name == "command.warning":
            text = " ".join(parts + ["warning", message])
        elif event_name == "command.error":
            text = " ".join(parts + ["error", message])
        elif event_label and message.strip() == (event_name or "").strip():
            text = " ".join(parts + [event_label])
        elif event_label and message.strip() == event_label:
            text = " ".join(parts + [message])
        else:
            text = " ".join(parts + [message])
        preview = self._format_fields(fields)
        if preview:
            text += f"  {preview}"
        if status and event_name in {"command.start", "command.complete"}:
            text += f"  status={status}"
        return text


@dataclass(frozen=True)
class RunLoggingContext:
    run_id: str
    command: str
    started_at: str
    pid: int
    cwd: str
    run_dir: Path | None
    event_log_path: Path | None
    text_log_path: Path | None
    meta_path: Path | None
    latest_path: Path | None
    unsafe_debug_logging: bool
    console_level: str
    file_level: str
    file_logging_enabled: bool


@dataclass(frozen=True)
class BoundLogger:
    session: "LoggingSession"
    logger_name: str
    context: dict[str, Any]

    def bind(self, **context: Any) -> "BoundLogger":
        merged = dict(self.context)
        merged.update(context)
        return BoundLogger(session=self.session, logger_name=self.logger_name, context=merged)

    def event(
        self,
        event: str,
        *,
        level: int | str = logging.INFO,
        message: str | None = None,
        stage: str | None = None,
        status: str | None = None,
        **fields: Any,
    ) -> None:
        merged = dict(self.context)
        merged.update(fields)
        self.session.log_event(
            event,
            level=level,
            message=message,
            stage=stage,
            status=status,
            logger_name=self.logger_name,
            **merged,
        )


class LoggingSession:
    def __init__(
        self,
        *,
        context: RunLoggingContext,
        argv: list[str],
        logger: logging.Logger,
        startup_warning: str | None = None,
    ) -> None:
        self.context = context
        self.argv = sanitize_argv(argv)
        self._logger = logger
        self._startup_warning = startup_warning
        self._stage_stack: list[str] = []
        self._warning_show = warnings.showwarning
        self._finalized = False
        self._error_logged = False

    @property
    def current_stage(self) -> str | None:
        return self._stage_stack[-1] if self._stage_stack else None

    @property
    def startup_warning(self) -> str | None:
        return self._startup_warning

    @property
    def error_logged(self) -> bool:
        return self._error_logged

    def get_logger(self, name: str, **context: Any) -> BoundLogger:
        return BoundLogger(self, f"{LOGGER_NAME}.{name}", dict(context))

    def _capture_warning(
        self,
        message: warnings.WarningMessage | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: str | None = None,
    ) -> None:
        text = str(message)
        self.log_event(
            "command.warning",
            level=logging.WARNING,
            message=text,
            stage=self.current_stage,
            warning_category=category.__name__,
            warning_source=f"{Path(filename).name}:{lineno}",
        )

    def start(self) -> None:
        warnings.showwarning = self._capture_warning

    def log_event(
        self,
        event: str,
        *,
        level: int | str = logging.INFO,
        message: str | None = None,
        stage: str | None = None,
        status: str | None = None,
        logger_name: str | None = None,
        **fields: Any,
    ) -> None:
        log_level = logging._nameToLevel.get(str(level).upper(), level) if isinstance(level, str) else level
        target = logging.getLogger(logger_name or LOGGER_NAME)
        sanitized_fields = sanitize_fields(fields, unsafe_debug_logging=self.context.unsafe_debug_logging)
        target.log(
            log_level,
            message or event,
            extra={
                "event_name": event,
                "event_fields": sanitized_fields,
                "stage": stage,
                "status": status,
            },
        )

    @contextmanager
    def stage(self, stage: str, *, message: str | None = None, **fields: Any) -> Iterator[None]:
        self._stage_stack.append(stage)
        self.log_event(
            "command.stage.start",
            message=message or f"{stage} started",
            stage=stage,
            status="started",
            **fields,
        )
        try:
            yield
        except Exception as exc:
            self.log_event(
                "command.error",
                level=logging.ERROR,
                message=f"{stage} failed: {exc}",
                stage=stage,
                status="failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            self._error_logged = True
            raise
        else:
            self.log_event(
                "command.stage.complete",
                message=message or f"{stage} completed",
                stage=stage,
                status="completed",
                **fields,
            )
        finally:
            self._stage_stack.pop()

    def finalize(self, *, exit_code: int, status: str) -> None:
        if self._finalized:
            return
        duration_seconds = (_utc_now() - datetime.fromisoformat(self.context.started_at.replace("Z", "+00:00"))).total_seconds()
        self.log_event(
            "command.complete",
            message=f"{self.context.command} completed",
            status=status,
            exit_code=exit_code,
            duration_seconds=round(duration_seconds, 3),
        )
        self._write_meta(exit_code=exit_code, status=status, duration_seconds=duration_seconds)
        warnings.showwarning = self._warning_show
        self._close_handlers()
        self._finalized = True

    def _write_meta(self, *, exit_code: int, status: str, duration_seconds: float) -> None:
        if self.context.meta_path is None:
            return
        payload = {
            "command": self.context.command,
            "run_id": self.context.run_id,
            "argv": self.argv,
            "started_at": self.context.started_at,
            "finished_at": _utc_timestamp(),
            "duration_seconds": round(duration_seconds, 3),
            "exit_code": exit_code,
            "status": status,
            "pid": self.context.pid,
            "cwd": self.context.cwd,
            "unsafe_debug_logging": self.context.unsafe_debug_logging,
            "file_logging_enabled": self.context.file_logging_enabled,
            "log_paths": {
                "run_dir": None if self.context.run_dir is None else str(self.context.run_dir),
                "events_jsonl": None if self.context.event_log_path is None else str(self.context.event_log_path),
                "run_log": None if self.context.text_log_path is None else str(self.context.text_log_path),
            },
        }
        self.context.meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if self.context.latest_path is not None:
            latest = {
                "command": self.context.command,
                "run_id": self.context.run_id,
                "run_dir": None if self.context.run_dir is None else str(self.context.run_dir),
                "meta_path": str(self.context.meta_path),
                "updated_at": _utc_timestamp(),
            }
            self.context.latest_path.write_text(json.dumps(latest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _close_handlers(self) -> None:
        for handler in list(self._logger.handlers):
            self._logger.removeHandler(handler)
            handler.close()


def _prepare_run_directory(base_dir: Path, run_id: str) -> Path:
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _reset_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    return logger


def setup_run_logging(
    paths: ProjectPaths,
    *,
    command: str,
    argv: list[str],
    log_level: str = "INFO",
    quiet: bool = False,
    unsafe_debug_logging: bool = False,
) -> LoggingSession:
    started_at = _utc_timestamp()
    timestamp = started_at.replace("-", "").replace(":", "").replace("Z", "Z").replace("T", "T")
    run_id = f"{timestamp}_{os.getpid()}"
    command_logs_dir = paths.logs / command
    console_level_name = "WARNING" if quiet else log_level.upper()
    file_level_name = log_level.upper()

    logger = _reset_logger()
    formatter = HumanReadableFormatter()
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging._nameToLevel.get(console_level_name, logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    run_dir: Path | None = None
    event_log_path: Path | None = None
    text_log_path: Path | None = None
    meta_path: Path | None = None
    latest_path: Path | None = None
    startup_warning: str | None = None

    try:
        run_dir = _prepare_run_directory(command_logs_dir, run_id)
        event_log_path = run_dir / "events.jsonl"
        text_log_path = run_dir / "run.log"
        meta_path = run_dir / "run_meta.json"
        latest_path = command_logs_dir / "latest.json"

        text_handler = logging.FileHandler(text_log_path, encoding="utf-8")
        text_handler.setLevel(logging._nameToLevel.get(file_level_name, logging.INFO))
        text_handler.setFormatter(formatter)
        logger.addHandler(text_handler)

        jsonl_handler = JsonlEventHandler(event_log_path, command=command, run_id=run_id)
        jsonl_handler.setLevel(logging._nameToLevel.get(file_level_name, logging.INFO))
        logger.addHandler(jsonl_handler)
    except OSError as exc:
        startup_warning = f"file logging disabled: {exc}"
        sys.stderr.write(f"Logging fallback: {startup_warning}\n")
        sys.stderr.flush()

    context = RunLoggingContext(
        run_id=run_id,
        command=command,
        started_at=started_at,
        pid=os.getpid(),
        cwd=str(paths.root),
        run_dir=run_dir,
        event_log_path=event_log_path,
        text_log_path=text_log_path,
        meta_path=meta_path,
        latest_path=latest_path,
        unsafe_debug_logging=unsafe_debug_logging,
        console_level=console_level_name,
        file_level=file_level_name,
        file_logging_enabled=event_log_path is not None and text_log_path is not None,
    )
    return LoggingSession(context=context, argv=argv, logger=logger, startup_warning=startup_warning)
