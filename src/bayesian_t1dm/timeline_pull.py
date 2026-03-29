from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from .acquisition import (
    AcquisitionError,
    ExportWindow,
    PlaywrightTandemSourceClient,
    StepLogger,
    TandemCredentials,
    generate_export_windows,
    load_tandem_credentials,
)
from .paths import ProjectPaths
from .tandem_browser import capture_control_inventory

DEFAULT_TIMELINE_URL = "https://source.tandemdiabetes.com/reports/timeline"


@dataclass(frozen=True)
class TimelinePullResult:
    csv_path: Path
    summary_path: Path
    summary: dict[str, object]


def _parse_date(value: str | date | datetime) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return pd.Timestamp(value).date()


def _timeline_url(base_url: str, window: ExportWindow) -> str:
    root = base_url.split("?", 1)[0]
    return root + "?" + urlencode({"startDate": window.start_date.isoformat(), "endDate": window.end_date.isoformat()})


def _parse_timeline_row_text(text: str) -> dict[str, str] | None:
    collapsed = " ".join((text or "").split())
    prefix, sep, rest = collapsed.partition(" Time in Range: ")
    if not sep:
        return None
    try:
        weekday, date_part = prefix.split(" - ", 1)
        summary, sep2, carbs_tail = rest.partition(" Carbs: ")
        if not sep2:
            return None
        tir_part, rest_summary = summary.split(" Avg: ", 1)
        avg_part, rest_summary = rest_summary.split(" SD: ", 1)
        sd_part, tdi_part = rest_summary.split(" TDI: ", 1)
        carbs_part, _ = carbs_tail.split(" ", 1)
        date_iso = datetime.strptime(date_part.strip(), "%b %d, %Y").date().isoformat()
        return {
            "weekday": weekday.strip(),
            "date": date_part.strip(),
            "date_iso": date_iso,
            "time_in_range_pct": tir_part.strip().split(" %", 1)[0].strip(),
            "avg_mg_dl": avg_part.strip().split(" mg/dL", 1)[0].strip(),
            "sd_mg_dl": sd_part.strip().split(" mg/dL", 1)[0].strip(),
            "tdi_units": tdi_part.strip().split(" units", 1)[0].strip(),
            "carbs_g": carbs_part.strip(),
            "source_text": collapsed,
        }
    except Exception:
        return None


def _rows_from_inventory(inventory: list[dict[str, object]], window: ExportWindow) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for entry in inventory:
        if not bool(entry.get("visible", True)):
            continue
        if str(entry.get("role") or "").lower() != "table":
            continue
        text = str(entry.get("text") or "")
        if " Time in Range: " not in text:
            continue
        parsed = _parse_timeline_row_text(text)
        if parsed is None:
            continue
        parsed["window_id"] = window.window_id
        parsed["window_start"] = window.start_date.isoformat()
        parsed["window_end"] = window.end_date.isoformat()
        rows.append(parsed)
    return rows


def _summary_from_frame(df: pd.DataFrame, *, csv_path: Path, window_row_counts: list[tuple[str, int]]) -> dict[str, object]:
    numeric = df.copy()
    for column in ["time_in_range_pct", "avg_mg_dl", "sd_mg_dl", "tdi_units", "carbs_g"]:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    all_days = pd.date_range(numeric["date_iso"].min(), numeric["date_iso"].max(), freq="D").date.astype(str)
    present_days = set(numeric["date_iso"].astype(str))
    missing_dates = [day for day in all_days if day not in present_days]
    return {
        "source": "tandem daily timeline DOM fallback",
        "combined_path": str(csv_path),
        "window_count": len(window_row_counts),
        "window_row_counts": window_row_counts,
        "row_count": int(len(numeric)),
        "unique_days": int(numeric["date_iso"].nunique()),
        "date_start": str(numeric["date_iso"].min()),
        "date_end": str(numeric["date_iso"].max()),
        "time_in_range_pct_mean": float(numeric["time_in_range_pct"].mean()),
        "time_in_range_pct_median": float(numeric["time_in_range_pct"].median()),
        "avg_mg_dl_mean": float(numeric["avg_mg_dl"].mean()),
        "avg_mg_dl_median": float(numeric["avg_mg_dl"].median()),
        "sd_mg_dl_mean": float(numeric["sd_mg_dl"].mean()),
        "tdi_units_mean": float(numeric["tdi_units"].mean()),
        "tdi_units_total": float(numeric["tdi_units"].sum()),
        "carbs_g_mean": float(numeric["carbs_g"].mean()),
        "carbs_g_total": float(numeric["carbs_g"].sum()),
        "missing_dates": missing_dates,
    }


def _write_summary(summary_path: Path, summary: dict[str, object]) -> Path:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def collect_tandem_daily_timeline_range(
    client: PlaywrightTandemSourceClient,
    *,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    workspace: ProjectPaths,
    credentials: TandemCredentials,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    window_days: int = 14,
    direction: str = "forward",
    retries: int = 15,
    poll_interval_ms: int = 3_000,
    step_log: StepLogger | None = None,
) -> TimelinePullResult:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start > end:
        raise AcquisitionError("start_date must be on or before end_date")
    windows = generate_export_windows(start, end, window_days=window_days, direction=direction)
    workspace.cloud_raw.mkdir(parents=True, exist_ok=True)
    workspace.runtime_downloads.mkdir(parents=True, exist_ok=True)

    csv_path = Path(output_path) if output_path is not None else workspace.cloud_raw / f"tandem_daily_timeline_{start.isoformat()}__{end.isoformat()}.csv"
    summary_path = Path(summary_path) if summary_path is not None else csv_path.with_name(f"{csv_path.stem}_summary.json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    client.login(credentials, step_log=step_log)
    page = client._page
    assert page is not None

    base_url = client.daily_timeline_url or DEFAULT_TIMELINE_URL
    if step_log is not None:
        step_log.write(
            "timeline.pull.start",
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            window_days=window_days,
            direction=direction,
            window_count=len(windows),
        )

    rows: list[dict[str, str]] = []
    window_row_counts: list[tuple[str, int]] = []
    try:
        for window in windows:
            page.goto(_timeline_url(base_url, window), wait_until="domcontentloaded")
            parsed_rows: list[dict[str, str]] = []
            for attempt in range(1, retries + 1):
                try:
                    page.wait_for_load_state("networkidle", timeout=min(client.timeout_ms, 5_000))
                except Exception:
                    pass
                inventory = capture_control_inventory(page)
                parsed_rows = _rows_from_inventory(inventory, window)
                if parsed_rows:
                    if step_log is not None:
                        step_log.write(
                            "timeline.window.rows_ready",
                            window_id=window.window_id,
                            attempt=attempt,
                            row_count=len(parsed_rows),
                        )
                    break
                page.wait_for_timeout(poll_interval_ms)
            if not parsed_rows:
                raise AcquisitionError(
                    f"Could not read rendered timeline rows for {window.window_id} after {retries} attempts",
                )
            rows.extend(parsed_rows)
            window_row_counts.append((window.window_id, len(parsed_rows)))
    except Exception as exc:
        if hasattr(client, "capture_page_diagnostics"):
            try:
                client.capture_page_diagnostics("timeline-pull-error")
            except Exception:
                pass
        if step_log is not None:
            step_log.write("timeline.pull.error", error=str(exc))
        raise

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise AcquisitionError("No timeline rows were collected")
    frame["date_iso"] = frame["date_iso"].astype(str)
    frame = frame.sort_values(["date_iso", "window_id"]).drop_duplicates(subset=["date_iso"], keep="last").reset_index(drop=True)
    frame.to_csv(csv_path, index=False)

    summary = _summary_from_frame(frame, csv_path=csv_path, window_row_counts=window_row_counts)
    _write_summary(summary_path, summary)
    if step_log is not None:
        step_log.write("timeline.pull.complete", csv_path=str(csv_path), summary_path=str(summary_path), row_count=len(frame))
    return TimelinePullResult(csv_path=csv_path, summary_path=summary_path, summary=summary)


def pull_tandem_daily_timeline_range(
    *,
    root: str | Path | None = None,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    env_file: str | Path | None = None,
    page_map_path: str | Path | None = None,
    login_url: str = "https://source.tandemdiabetes.com/",
    daily_timeline_url: str | None = None,
    headless: bool = False,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    window_days: int = 14,
    direction: str = "forward",
    retries: int = 15,
    poll_interval_ms: int = 3_000,
    step_log: StepLogger | None = None,
) -> TimelinePullResult:
    paths = ProjectPaths.from_root(root).ensure()
    credentials = load_tandem_credentials(root, env_file)
    with PlaywrightTandemSourceClient(
        paths,
        page_map_path=page_map_path,
        login_url=login_url,
        daily_timeline_url=daily_timeline_url,
        headless=headless,
    ) as client:
        return collect_tandem_daily_timeline_range(
            client,
            start_date=start_date,
            end_date=end_date,
            workspace=paths,
            credentials=credentials,
            output_path=output_path,
            summary_path=summary_path,
            window_days=window_days,
            direction=direction,
            retries=retries,
            poll_interval_ms=poll_interval_ms,
            step_log=step_log,
        )


__all__ = [
    "DEFAULT_TIMELINE_URL",
    "TimelinePullResult",
    "collect_tandem_daily_timeline_range",
    "pull_tandem_daily_timeline_range",
]
