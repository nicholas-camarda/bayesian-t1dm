from __future__ import annotations

import json
import logging
from pathlib import Path
import warnings

from bayesian_t1dm.acquisition import StepLogger
from bayesian_t1dm.observability import REDACTED, HumanReadableFormatter, sanitize_fields, setup_run_logging
from bayesian_t1dm.paths import ProjectPaths


def _workspace(tmp_path: Path) -> ProjectPaths:
    root = tmp_path / "repo"
    root.mkdir()
    return ProjectPaths.from_root(
        root,
        runtime_root=tmp_path / "runtime",
        cloud_root=tmp_path / "cloud",
    ).ensure()


def test_sanitize_fields_redacts_sensitive_and_raw_values():
    fields = sanitize_fields(
        {
            "email": "me@example.com",
            "password": "secret",
            "payload": {"glucose": [101, 102]},
            "glucose_values": [101, 102],
            "row_count": 4,
        }
    )

    assert fields["email"] == REDACTED
    assert fields["password"] == REDACTED
    assert fields["payload"] == REDACTED
    assert fields["glucose_values"] == REDACTED
    assert fields["row_count"] == 4


def test_sanitize_fields_keeps_summary_counts_and_timestamps_readable():
    fields = sanitize_fields(
        {
            "health_measurement_rows": 832432,
            "health_activity_rows": 70054,
            "requested_tandem_start": "2025-04-25T00:00:00",
            "requested_tandem_end": "2025-05-24T00:00:00",
        }
    )

    assert fields["health_measurement_rows"] == 832432
    assert fields["health_activity_rows"] == 70054
    assert fields["requested_tandem_start"] == "2025-04-25T00:00:00"
    assert fields["requested_tandem_end"] == "2025-05-24T00:00:00"


def test_human_readable_formatter_is_compact_for_terminal_output():
    formatter = HumanReadableFormatter()
    record = logging.LogRecord(
        name="bayesian_t1dm",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Loading Tandem normalized inputs.",
        args=(),
        exc_info=None,
    )
    record.stage = "prepare_model_data.load_inputs"
    record.status = None
    record.event_name = "prepare_model_data.tandem_inputs.loading"
    record.event_fields = {}

    rendered = formatter.format(record)

    assert "load inputs:" in rendered
    assert "Loading Tandem normalized inputs." in rendered
    assert "prepare_model_data.tandem_inputs.loading" not in rendered
    assert "command.stage.start" not in rendered


def test_setup_run_logging_writes_run_bundle_and_captures_warnings(tmp_path):
    paths = _workspace(tmp_path)
    session = setup_run_logging(paths, command="ingest", argv=["ingest"])
    session.start()
    session.log_event("command.start", message="ingest started", status="started")
    warnings.warn("coverage degraded", UserWarning, stacklevel=1)
    session.finalize(exit_code=0, status="success")

    latest = json.loads((paths.logs / "ingest" / "latest.json").read_text(encoding="utf-8"))
    run_dir = Path(latest["run_dir"])
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "run.log").exists()
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "success"
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["event"] == "command.warning" and event["message"] == "coverage degraded" for event in events)
    assert any(event["event"] == "command.complete" and event["status"] == "success" for event in events)


def test_setup_run_logging_falls_back_when_run_directory_creation_fails(tmp_path, monkeypatch):
    paths = _workspace(tmp_path)
    monkeypatch.setattr("bayesian_t1dm.observability._prepare_run_directory", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))

    session = setup_run_logging(paths, command="ingest", argv=["ingest"])
    session.start()
    session.finalize(exit_code=0, status="success")

    assert session.context.file_logging_enabled is False
    assert session.startup_warning is not None
    assert "disk full" in session.startup_warning


def test_step_logger_redacts_sensitive_fields_in_legacy_file(tmp_path):
    log_path = tmp_path / "legacy.jsonl"
    step_log = StepLogger(log_path)

    step_log.write("login.start", email="me@example.com", payload={"foo": "bar"})

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["event"] == "login.start"
    assert record["email"] == REDACTED
    assert record["payload"] == REDACTED
