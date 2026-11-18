"""Pump profile (IDP) history from the Tandem event log.

A Tandem pump logs every edit to a personal profile and every profile switch:

- ``LID_IDP_TD_SEG`` (event 68): one time-of-day segment of one profile after a change, with its start time,
  basal rate, correction factor (ISF), target and carb ratio. ``modificationType`` says what changed
  (0 segment created, 1 inserted, 2 deleted, 3 basal, 4 carb ratio, 5 correction factor, 6 target, 7 start
  time); ``status`` is a bitmask of the values that are set, 15 meaning all four.
- ``LID_IDP`` (event 69): an action on a whole profile; ``status`` 0 created, 1 copied (from ``sourceIdp``),
  2 deleted, 3 activated. ``LID_IDP_MSG2`` (57) carries the rest of the profile name.

tconnectsync does not decode event 68 or 69, so they are read here from the archived raw events. Tandem
Source serializes each 4-byte word of the pump's little-endian record in reverse byte order; the offsets
below are pumpX2's (``IdpTimeDependentSegmentHistoryLog``, ``IdpActionHistoryLog``) after that reversal.

The pump-logs JSON that tconnectsync 3.x downloads carries event 69 (decoded by the server) but not event 68,
so segment values are known only while the binary archive (tconnectsync 2.x windows) covers the pump; after
that, edits are invisible and the history is not used. Profile switches stay known.
"""

from __future__ import annotations

import base64
import bisect
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EVENT_LEN = 26
SEGMENT_EVENT = 68
PROFILE_EVENT = 69
PROFILE_NAME_EVENT = 57
PROFILE_ACTIVATED = 3
SEGMENT_INSERTS = (0, 1)
SEGMENT_DELETE = 2
SEGMENT_COMPLETE_STATUS = 15
_TANDEM_EPOCH = dt.datetime(2008, 1, 1)

PROFILE_EVENT_COLUMNS = (
    "pump_serial", "timestamp", "sequence", "event", "profile_id", "status", "source_profile_id", "profile_name",
    "modification", "segment_index", "start_minutes", "basal_units_per_hour", "isf_mgdl_per_unit", "target_bg_mgdl",
    "carb_ratio_g_per_unit",
)
# Columns attached to each carb entry: the profile active at the meal and, when its edits are visible in the
# archive, that profile's segment in force at the meal's time of day.
# ``profile_edits_visible`` is 1 while the archive shows that pump's profile edits and 0 after, when a profile's
# values may have changed unseen.
PROFILE_SETTING_COLUMNS = (
    "profile_id", "profile_edits_visible", "profile_carb_ratio_g_per_unit", "profile_isf_mgdl_per_unit", "profile_target_bg_mgdl",
)


def _word_reversed(payload: np.ndarray) -> np.ndarray:
    """Undo Tandem Source's per-word byte reversal of the 16 payload bytes (rows of shape (n, 16))."""
    return payload.reshape(-1, 4, 4)[:, :, ::-1].reshape(-1, 16)


def _u16le(payload: np.ndarray, offset: int) -> np.ndarray:
    return payload[:, offset].astype(np.int64) | (payload[:, offset + 1].astype(np.int64) << 8)


def _name(chars: Any) -> str:
    raw = bytes(int(c) for c in chars)
    return raw.split(b"\x00")[0].replace(b"\xff", b"").decode("latin1")


def _binary_events(path: Path, pump_serial: str) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith('"'):
        text = json.loads(text)
    data = base64.b64decode(text)
    records = np.frombuffer(data[: len(data) - len(data) % EVENT_LEN], dtype=np.uint8).reshape(-1, EVENT_LEN)
    event_id = ((records[:, 0].astype(np.int64) << 8) | records[:, 1]) & 0x0FFF
    records = records[np.isin(event_id, (SEGMENT_EVENT, PROFILE_EVENT))]
    event_id = ((records[:, 0].astype(np.int64) << 8) | records[:, 1]) & 0x0FFF
    seconds = records[:, 2:6].astype(np.int64) @ np.array([1 << 24, 1 << 16, 1 << 8, 1])
    sequence = records[:, 6:10].astype(np.int64) @ np.array([1 << 24, 1 << 16, 1 << 8, 1])
    payload = _word_reversed(records[:, 10:26])  # now in pumpX2 order: payload[k] is record offset 10 + k
    rows = []
    for k in range(len(records)):
        p = payload[k]
        row: dict[str, Any] = {
            "pump_serial": pump_serial,
            "timestamp": _TANDEM_EPOCH + dt.timedelta(seconds=int(seconds[k])),
            "sequence": int(sequence[k]),
            "event": int(event_id[k]),
            "profile_id": int(p[0]),
            "status": int(p[1]),
        }
        if event_id[k] == SEGMENT_EVENT:
            row |= {
                "segment_index": int(p[2]),
                "modification": int(p[3]),
                "start_minutes": int(_u16le(payload[k : k + 1], 4)[0]),
                "basal_units_per_hour": int(_u16le(payload[k : k + 1], 6)[0]) / 1000.0,
                "isf_mgdl_per_unit": int(_u16le(payload[k : k + 1], 8)[0]),
                "target_bg_mgdl": int(_u16le(payload[k : k + 1], 10)[0]),
                # pumpX2 reads a single byte at offset 24; the archive holds the ratio in mg/U at offsets 22-23
                # (7000 = 7.0 g/U), which reproduces the t:slim's own calculator records.
                "carb_ratio_g_per_unit": int(_u16le(payload[k : k + 1], 12)[0]) / 1000.0,
            }
        else:
            row |= {"source_profile_id": int(p[2]), "profile_name": _name(p[8:16])}
        rows.append(row)
    return rows


def _json_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pump_serial = str(payload["pump_serial"])
    rows = []
    for chunk in payload["chunks"]:
        for event in chunk["response"].get("events") or []:
            if event["eventCode"] not in (SEGMENT_EVENT, PROFILE_EVENT):
                continue
            if event["eventCode"] == SEGMENT_EVENT:
                # Not returned by the pump-logs endpoint so far; decoding it would need its server field names.
                raise ValueError(f"{path} contains event 68 records, which this decoder does not read from pump-logs JSON yet.")
            props = event["eventProperties"]
            rows.append(
                {
                    "pump_serial": pump_serial,
                    "timestamp": pd.Timestamp(event["pumpDateTime"]).to_pydatetime(),
                    "sequence": int(event["sequenceNumber"]),
                    "event": PROFILE_EVENT,
                    "profile_id": int(props["idp"]),
                    "status": int(props["status"]),
                    "source_profile_id": int(props["sourceIdp"]),
                    "profile_name": _name(props[f"name{i}"] for i in range(8)),
                }
            )
    return rows


def _window_dates(window_dir: Path) -> tuple[dt.date, dt.date]:
    start, end = window_dir.name.split("__", 1)
    return dt.date.fromisoformat(start), dt.date.fromisoformat(end)


def decode_profile_events(tandem_root: str | Path) -> tuple[pd.DataFrame, dict[str, tuple[dt.date, dt.date]]]:
    """All profile events in the archive, and for each pump the span over which its segment edits are visible.

    The visible span is the first unbroken run of windows holding that pump's binary events: before or after it,
    an edit may have happened that the archive does not show.
    """
    tandem_root = Path(tandem_root)
    rows: list[dict[str, Any]] = []
    binary_windows: dict[str, list[tuple[dt.date, dt.date]]] = {}
    window_dirs = sorted(path for path in tandem_root.iterdir() if path.is_dir() and "__" in path.name) if tandem_root.exists() else []
    for window_dir in window_dirs:
        raw = window_dir / "raw"
        for path in sorted(raw.glob("pump_events_raw_*.txt")):
            pump_serial = path.stem.removeprefix("pump_events_raw_").rpartition("_")[0]
            if pump_serial in {"", "unknown"}:
                raise ValueError(f"{path} does not name its pump serial; profile history needs it.")
            events = _binary_events(path, pump_serial)
            rows.extend(events)
            if path.stat().st_size:
                binary_windows.setdefault(pump_serial, []).append(_window_dates(window_dir))
        for path in sorted(raw.glob("pump_logs_*.json")):
            if not path.name.endswith(".meta.json"):
                rows.extend(_json_events(path))
    frame = pd.DataFrame(rows, columns=list(PROFILE_EVENT_COLUMNS))
    frame = frame.drop_duplicates(subset=["pump_serial", "sequence"]).sort_values(["pump_serial", "sequence"]).reset_index(drop=True)
    visible: dict[str, tuple[dt.date, dt.date]] = {}
    first_event = frame.groupby("pump_serial")["timestamp"].min()
    for pump_serial, spans in binary_windows.items():
        if pump_serial not in first_event:
            continue  # this pump logged no profile events
        spans.sort()
        start, end = spans[0]
        for span_start, span_end in spans[1:]:
            if span_start > end + dt.timedelta(days=1):
                break
            end = max(end, span_end)
        visible[pump_serial] = (max(start, first_event[pump_serial].date()), end)
    return frame, visible


@dataclass
class ProfileHistory:
    """Replays profile events into the profile active, and its segment in force, at any time on each pump."""

    events: pd.DataFrame
    visible: dict[str, tuple[dt.date, dt.date]]
    _tables: dict[tuple[str, int], tuple[list[dt.datetime], list[list[dict[str, float] | None]]]] = field(init=False, default_factory=dict)
    _active: dict[str, tuple[list[dt.datetime], list[int]]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        segments = self.events[self.events["event"].eq(SEGMENT_EVENT)]
        state: dict[tuple[str, int], list[dict[str, float] | None]] = {}
        for row in segments.itertuples(index=False):
            key = (row.pump_serial, int(row.profile_id))
            table = state.setdefault(key, [])
            index = int(row.segment_index)
            values = {
                "start_minutes": int(row.start_minutes),
                "carb_ratio_g_per_unit": float(row.carb_ratio_g_per_unit),
                "isf_mgdl_per_unit": float(row.isf_mgdl_per_unit),
                "target_bg_mgdl": float(row.target_bg_mgdl),
            }
            complete = int(row.status) == SEGMENT_COMPLETE_STATUS
            if int(row.modification) in SEGMENT_INSERTS:
                table.insert(index, values if complete else None)
            elif int(row.modification) == SEGMENT_DELETE:
                if index >= len(table):
                    raise ValueError(f"Profile history deletes segment {index} of profile {key} at {row.timestamp}, which it never saw.")
                table.pop(index)
            else:
                table.extend([None] * (index + 1 - len(table)))
                table[index] = values if complete else None
            times, tables = self._tables.setdefault(key, ([], []))
            times.append(row.timestamp)
            tables.append(list(table))
        activations = self.events[self.events["event"].eq(PROFILE_EVENT) & self.events["status"].eq(PROFILE_ACTIVATED)]
        for pump_serial, group in activations.sort_values("timestamp").groupby("pump_serial"):
            self._active[pump_serial] = (list(group["timestamp"]), [int(v) for v in group["profile_id"]])

    def active_profile(self, pump_serial: str, timestamp: dt.datetime) -> int | None:
        times, profiles = self._active.get(pump_serial, ([], []))
        k = bisect.bisect_right(times, timestamp) - 1
        return profiles[k] if k >= 0 else None

    def segment_in_force(self, pump_serial: str, profile_id: int, timestamp: dt.datetime) -> dict[str, float] | None:
        """The profile's segment covering ``timestamp``'s time of day, or None when the archive cannot show it."""
        span = self.visible.get(pump_serial)
        if span is None or not span[0] <= timestamp.date() <= span[1]:
            return None
        times, tables = self._tables.get((pump_serial, profile_id), ([], []))
        k = bisect.bisect_right(times, timestamp) - 1
        if k < 0:
            return None
        table = tables[k]
        minute = timestamp.hour * 60 + timestamp.minute
        covering = None
        for index, segment in enumerate(table):
            if segment is not None and segment["start_minutes"] <= minute and (covering is None or segment["start_minutes"] > table[covering]["start_minutes"]):
                covering = index
        # A slot not yet seen could start before this minute, so the covering segment is unknown.
        if covering is None or (covering + 1 < len(table) and table[covering + 1] is None):
            return None
        return table[covering]

    def settings_for(self, carbs: pd.DataFrame) -> pd.DataFrame:
        """``PROFILE_SETTING_COLUMNS`` for each carb entry (by ``pump_serial`` and ``timestamp``)."""
        out = pd.DataFrame(index=carbs.index, columns=list(PROFILE_SETTING_COLUMNS), dtype=float)
        if carbs.empty or "pump_serial" not in carbs.columns:
            return out  # carbs from hand-kept logs belong to no pump
        serials = carbs["pump_serial"].map(lambda v: "" if pd.isna(v) else str(v).removesuffix(".0"))
        for index, pump_serial, timestamp in zip(carbs.index, serials, pd.to_datetime(carbs["timestamp"])):
            if not pump_serial or pd.isna(timestamp):
                continue
            moment = timestamp.to_pydatetime()
            profile_id = self.active_profile(pump_serial, moment)
            if profile_id is None:
                continue
            out.at[index, "profile_id"] = profile_id
            span = self.visible.get(pump_serial)
            out.at[index, "profile_edits_visible"] = float(span is not None and span[0] <= moment.date() <= span[1])
            segment = self.segment_in_force(pump_serial, profile_id, moment)
            if segment is not None:
                out.at[index, "profile_carb_ratio_g_per_unit"] = segment["carb_ratio_g_per_unit"]
                out.at[index, "profile_isf_mgdl_per_unit"] = segment["isf_mgdl_per_unit"]
                out.at[index, "profile_target_bg_mgdl"] = segment["target_bg_mgdl"]
        return out
