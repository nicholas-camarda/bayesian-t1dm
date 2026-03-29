from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

import bayesian_t1dm.timeline_pull as timeline_pull
from bayesian_t1dm.acquisition import ExportWindow, TandemCredentials
from bayesian_t1dm.paths import ProjectPaths


class FakeTimelinePage:
    def __init__(self) -> None:
        self.waits: list[int] = []
        self.current_window_id: str | None = None
        self.current_start: date | None = None
        self.current_end: date | None = None

    def goto(self, url: str, wait_until: str | None = None) -> None:
        parsed = parse_qs(urlparse(url).query)
        self.current_start = date.fromisoformat(parsed["startDate"][0])
        self.current_end = date.fromisoformat(parsed["endDate"][0])
        self.current_window_id = f"{self.current_start.isoformat()}__{self.current_end.isoformat()}"

    def wait_for_load_state(self, state: str, timeout: int | None = None) -> None:
        return None

    def wait_for_timeout(self, ms: int) -> None:
        self.waits.append(ms)


class FakeTimelineClient:
    def __init__(self) -> None:
        self._page = FakeTimelinePage()
        self.daily_timeline_url = "https://source.tandemdiabetes.com/reports/timeline"
        self.timeout_ms = 5_000
        self.login_calls: list[str] = []
        self.diagnostics: list[str] = []

    def login(self, credentials, step_log=None) -> None:
        self.login_calls.append(credentials.email)

    def capture_page_diagnostics(self, stem: str):
        self.diagnostics.append(stem)
        return None


def _workspace(tmp_path: Path) -> ProjectPaths:
    root = tmp_path / "bayesian-t1dm"
    root.mkdir()
    runtime_root = tmp_path / "runtime"
    cloud_root = tmp_path / "cloud"
    return ProjectPaths.from_root(root, runtime_root=runtime_root, cloud_root=cloud_root).ensure()


def _row_text(current_day: date, *, tir: int, avg: int, sd: int, tdi: float, carbs: float) -> str:
    weekday = current_day.strftime("%a")
    date_label = current_day.strftime("%b %d, %Y").replace(" 0", " ")
    return (
        f"{weekday} - {date_label} Time in Range: {tir} % "
        f"Avg: {avg} mg/dL SD: {sd} mg/dL TDI: {tdi} units Carbs: {carbs} g"
    )


def _make_inventory(start: date, end: date) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current = start
    index = 0
    while current <= end:
        rows.append(
            {
                "tag": "div",
                "role": "table",
                "id": f"row-{index}",
                "name": None,
                "type": None,
                "aria_label": None,
                "placeholder": None,
                "autocomplete": None,
                "text": _row_text(
                    current,
                    tir=70 + index,
                    avg=150 + index,
                    sd=40 + index,
                    tdi=30.0 + index,
                    carbs=100.0 + index,
                ),
                "title": None,
                "href": None,
                "data_testid": None,
                "visible": True,
            }
        )
        current += pd.Timedelta(days=1).to_pytimedelta()
        index += 1
    return rows


def test_collect_tandem_daily_timeline_range_retries_and_merges(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    client = FakeTimelineClient()
    credentials = TandemCredentials(email="me@example.com", password="secret")
    attempts: dict[str, int] = {}

    def fake_inventory(page):
        assert page.current_window_id is not None
        attempts[page.current_window_id] = attempts.get(page.current_window_id, 0) + 1
        if page.current_window_id == "2024-01-01__2024-01-03" and attempts[page.current_window_id] < 3:
            return []
        if page.current_window_id == "2024-01-04__2024-01-05" and attempts[page.current_window_id] < 2:
            return []
        return _make_inventory(page.current_start, page.current_end)

    monkeypatch.setattr(timeline_pull, "capture_control_inventory", fake_inventory)

    result = timeline_pull.collect_tandem_daily_timeline_range(
        client,
        start_date="2024-01-01",
        end_date="2024-01-05",
        workspace=workspace,
        credentials=credentials,
        output_path=workspace.cloud_raw / "tandem_daily_timeline_2024-01-01__2024-01-05.csv",
        summary_path=workspace.cloud_raw / "tandem_daily_timeline_2024-01-01__2024-01-05_summary.json",
        window_days=3,
        direction="forward",
        retries=4,
        poll_interval_ms=1,
    )

    df = pd.read_csv(result.csv_path)

    assert client.login_calls == ["me@example.com"]
    assert result.csv_path.exists()
    assert result.summary_path.exists()
    assert result.summary["row_count"] == 5
    assert result.summary["unique_days"] == 5
    assert result.summary["missing_dates"] == []
    assert list(df["date_iso"]) == ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    assert len(client._page.waits) >= 3
