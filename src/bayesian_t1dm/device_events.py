"""Control-IQ user modes and infusion-site events from the pump archive, for the whole record.

The archive holds the same pump events in two forms: base64 binary windows (2.x downloads, 2024-01 on) and server-decoded
``pump_logs_*.json`` (3.x downloads, 2026-04-23 on). Both are read here, so a mode or a site change is known for the whole
record, not only the recent part. Binary records are decoded by tconnectsync's own event classes, the same ones that decode
the CGM, bolus and basal tables; no byte offsets are interpreted here.

- ``modes``: one row per mode change (``mode`` 0 standard, 1 sleep, 2 exercise) with the moment it began.
- ``sites``: one row per cartridge, cannula or tubing fill.
- ``coverage``: for each pump, the first and last date of the windows that hold pump events, so a missing event can be told
  apart from an unobserved period.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .pump_profiles import pump_key

MODE_EVENT = 229
SITE_EVENT_CODES = {33: "cartridge filled", 61: "cannula filled", 63: "tubing filled"}
MODE_COLUMNS = ["pump_serial", "sequence", "timestamp", "mode"]
SITE_COLUMNS = ["pump_serial", "sequence", "timestamp", "kind"]


def _wall_clock(value: Any) -> pd.Timestamp:
    """The pump's local wall-clock time, without the UTC offset the binary events carry."""
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp


def _binary_rows(path: Path, pump: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from tconnectsync.eventparser.generic import Events, decode_raw_events  # type: ignore
    from tconnectsync.eventparser.raw_event import EVENT_LEN  # type: ignore

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return [], []
    if text.startswith('"'):
        text = json.loads(text)
    data = decode_raw_events(text)
    records = np.frombuffer(data[: len(data) - len(data) % EVENT_LEN], dtype=np.uint8).reshape(-1, EVENT_LEN)
    event_id = ((records[:, 0].astype(np.int64) << 8) | records[:, 1]) & 0x0FFF
    wanted = records[np.isin(event_id, (MODE_EVENT, *SITE_EVENT_CODES))]
    modes: list[dict[str, Any]] = []
    sites: list[dict[str, Any]] = []
    for event in Events(wanted.tobytes()):
        record = event.todict()
        row = {"pump_serial": pump, "sequence": int(record["seqNum"]), "timestamp": _wall_clock(record["eventTimestamp"])}
        if record["id"] == MODE_EVENT:
            modes.append({**row, "mode": int(record["currentUserModeRaw"])})
        else:
            sites.append({**row, "kind": SITE_EVENT_CODES[int(record["id"])]})
    return modes, sites


def _json_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pump = pump_key(payload["pump_serial"])
    modes: list[dict[str, Any]] = []
    sites: list[dict[str, Any]] = []
    for chunk in payload["chunks"]:
        for event in chunk["response"].get("events") or []:
            code = event["eventCode"]
            row = {"pump_serial": pump, "sequence": int(event["sequenceNumber"]), "timestamp": _wall_clock(event["pumpDateTime"])}
            if code == MODE_EVENT:
                modes.append({**row, "mode": int(event["eventProperties"]["currentUserMode"])})
            elif code in SITE_EVENT_CODES:
                sites.append({**row, "kind": SITE_EVENT_CODES[code]})
    return modes, sites


def _window_dates(window_dir: Path) -> tuple[dt.date, dt.date]:
    start, end = window_dir.name.split("__", 1)
    return dt.date.fromisoformat(start), dt.date.fromisoformat(end)


def decode_device_events(tandem_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[dt.date, dt.date]]]:
    """Mode changes, site events and, per pump, the dates the archive's pump events span."""
    tandem_root = Path(tandem_root)
    modes: list[dict[str, Any]] = []
    sites: list[dict[str, Any]] = []
    spans: dict[str, list[dt.date]] = {}
    window_dirs = sorted(path for path in tandem_root.iterdir() if path.is_dir() and "__" in path.name) if tandem_root.exists() else []
    for window_dir in window_dirs:
        raw = window_dir / "raw"
        start, end = _window_dates(window_dir)
        for path in sorted(raw.glob("pump_events_raw_*.txt")):
            pump = pump_key(path.stem.removeprefix("pump_events_raw_").rpartition("_")[0])
            if pump in {"", "unknown"}:
                raise ValueError(f"{path} does not name its pump serial.")
            found = _binary_rows(path, pump)
            modes += found[0]
            sites += found[1]
            if path.stat().st_size:
                spans.setdefault(pump, []).extend([start, end])
        for path in sorted(raw.glob("pump_logs_*.json")):
            if path.name.endswith(".meta.json"):
                continue
            found = _json_rows(path)
            modes += found[0]
            sites += found[1]
            spans.setdefault(pump_key(json.loads(path.read_text(encoding="utf-8"))["pump_serial"]), []).extend([start, end])

    def frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
        out = pd.DataFrame(rows, columns=columns).drop_duplicates(["pump_serial", "sequence"])
        out["timestamp"] = pd.to_datetime(out["timestamp"]).astype("datetime64[ns]")
        return out.sort_values(["pump_serial", "timestamp"]).reset_index(drop=True)

    coverage = {pump: (min(dates), max(dates)) for pump, dates in sorted(spans.items())}

    def inside_archive(out: pd.DataFrame) -> pd.DataFrame:
        """Drop events stamped outside the archive's own span: a pump whose clock was not yet set logs a factory date (2020-01-15)."""
        first = out["pump_serial"].map(lambda pump: pd.Timestamp(coverage[pump][0]) if pump in coverage else pd.NaT)
        last = out["pump_serial"].map(lambda pump: pd.Timestamp(coverage[pump][1]) + pd.Timedelta(days=1) if pump in coverage else pd.NaT)
        return out[(out["timestamp"] >= first) & (out["timestamp"] < last)].reset_index(drop=True)

    return inside_archive(frame(modes, MODE_COLUMNS)), inside_archive(frame(sites, SITE_COLUMNS)), coverage
