"""Data-quality audit of the record before it is modelled: pure checks over the loaded streams and the hour-day table.

Every function here takes tables and returns tables; nothing reads or writes files (``data_audit_report`` does that, and
``main`` below only wires the two together). A check row has the columns ``CHECK_COLUMNS``: what was checked, the stream it
reads, the statistic, its value, the rule that sets the status, and the status with the reason. A check that needs an input
that was not supplied is not run and says so: ``not assessed`` is never a pass.

The checks cover, in order: what each stream holds (``inventory_coverage``); the CGM (``cgm_*``); the devices and the pump
clock (``pump_eras``, ``pump_change_break``, ``clock_jumps``, ``clock_alignment``); the settings the hour-day table
reconstructs (``unknown_settings``, ``calculator_agreement``, ``basal_log_agreement``, ``replay_event_crosscheck``);
meals and insulin (``meal_checks``, ``insulin_checks``); why hours are missing and whether the level of glucose drifts
(``missingness``, ``composition_test``, ``stationarity``); and whether each hour-day covariate would be known when a forecast
is made (``leakage_table``).
"""

from __future__ import annotations

import argparse
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .hour_day import GLUCOSE_COLUMNS, HOURS_PER_DAY, LOCAL_ZONE, SLOTS_PER_HOUR
from .outcomes import READING_MINUTES, READINGS_PER_DAY, MIN_DAY_COVERAGE
from .pump_profiles import pump_key

CHECK_COLUMNS = ["check", "source", "statistic", "result", "rule", "status", "reason"]
ISSUE_COLUMNS = ["step", "reason", "pump_serial", "start", "end", "n", "unit", "field", "value"]
WINDOW_DAYS = 14  # the consensus report window
ACTIVE_TIME_MIN = MIN_DAY_COVERAGE  # 70% of the readings a CGM would give in the window
FLAT_RUN_SLOTS = 12  # one hour of identical readings
LOW_MGDL = 70.0
VERY_LOW_MGDL = 54.0
SENSOR_FLOOR_MGDL = 40.0
SENSOR_CEILING_MGDL = 400.0
AGREEMENT_MIN = 0.99  # a reconstructed setting agrees with the pump's own record when 99% of compared moments match
COMPRESSION_FALL_MGDL = 30.0
COMPRESSION_NIGHT_HOURS = (23, 7)
HEAPING_MAX_SHARE = 0.50
BLOCK_DAYS = 28
DRAWS = 2000
SEED = 20261119
ALGORITHM_SOURCE = 7  # bolus source: Control-IQ automatic correction
BOLUS_CARB_BIT = 16
BOLUS_OVERRIDE_BIT = 4
DAILY_STREAMS = ("cgm", "basal")
MEAL_BOLUS_WINDOW = pd.Timedelta(minutes=10)


def _row(check: str, source: str, statistic: str, result: object, rule: str, status: str, reason: str) -> dict[str, object]:
    return {"check": check, "source": source, "statistic": statistic, "result": result, "rule": rule, "status": status, "reason": reason}


def _frame(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=CHECK_COLUMNS)


def _issues(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=ISSUE_COLUMNS)


def _pct(numerator: float, denominator: float) -> float:
    return float("nan") if not denominator else round(100.0 * float(numerator) / float(denominator), 2)


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Start and stop (exclusive) positions of each run of True in ``flags``."""
    padded = np.concatenate([[False], np.asarray(flags, dtype=bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def _run_issues(table: pd.DataFrame, flags: np.ndarray, *, step: str, reason: str | pd.Series, field: str, value: str) -> list[dict[str, object]]:
    """One issue row per run of consecutive hours in ``table`` where ``flags`` holds (and the pump and reason stay the same)."""
    reasons = reason if isinstance(reason, pd.Series) else pd.Series(reason, index=table.index)
    pumps = table["pump_serial"].astype(str).to_numpy()
    stamps = pd.to_datetime(table["hour_start"]).to_numpy()
    why = reasons.to_numpy()
    rows: list[dict[str, object]] = []
    for first, stop in _runs(flags):
        cut = [first] + [k for k in range(first + 1, stop) if pumps[k] != pumps[k - 1] or why[k] != why[k - 1]] + [stop]
        for a, b in zip(cut[:-1], cut[1:]):
            rows.append({"step": step, "reason": why[a], "pump_serial": pumps[a], "start": pd.Timestamp(stamps[a]), "end": pd.Timestamp(stamps[b - 1]) + pd.Timedelta(hours=1),
                         "n": b - a, "unit": "hours", "field": field, "value": value})
    return rows


# ---------------------------------------------------------------------------
# Inventory and coverage
# ---------------------------------------------------------------------------

def inventory_coverage(streams: dict[str, pd.Series], *, daily: Iterable[str] = DAILY_STREAMS) -> pd.DataFrame:
    """First and last date, rows, days with rows and the longest run of empty days for each stream.

    Args:
        streams: stream name to the timestamps of its rows.
        daily: the streams that should have rows every day (a whole empty day is a concern); the others are event streams,
            where days without rows are expected.

    Returns:
        One check row per stream.
    """
    rows = []
    for name, stamps in streams.items():
        stamps = pd.to_datetime(stamps).dropna()
        if stamps.empty:
            rows.append(_row(f"{name} rows and span", name, "rows", 0, "", "not assessed", "the stream has no rows"))
            continue
        days = pd.DatetimeIndex(stamps.dt.normalize().unique()).sort_values()
        calendar = int((days.max() - days.min()).days) + 1
        gaps = np.diff(days.to_numpy()) / np.timedelta64(1, "D")
        longest = int(gaps.max()) - 1 if len(gaps) else 0
        empty = calendar - len(days)
        per_day = stamps.groupby(stamps.dt.normalize()).size()
        result = f"{len(stamps)} rows, {days.min().date()} to {days.max().date()}; rows on {len(days)} of {calendar} days; median {per_day.median():.0f} rows on a day with rows; longest run of days without rows {longest}"
        if name in daily:
            status, reason = ("no concern", "every calendar day has rows") if empty == 0 else ("concern", f"{empty} calendar days have no rows (longest run {longest} days)")
            rule = "a stream that should have rows every day has no empty calendar day"
        else:
            status, reason, rule = "not applicable", "an event stream: days without rows are expected", "none"
        rows.append(_row(f"{name} rows and span", name, "rows, span, days with rows", result, rule, status, reason))
    return _frame(rows)


# ---------------------------------------------------------------------------
# CGM
# ---------------------------------------------------------------------------

def active_time_windows(table: pd.DataFrame, *, days: int = WINDOW_DAYS) -> pd.DataFrame:
    """Share of the possible 5-minute readings present in each consecutive ``days``-day window of the hour-day table.

    The last window is partial when the record does not divide into whole windows; it is marked ``full`` False and its
    share uses the days it holds.
    """
    frame = pd.DataFrame({"date": pd.to_datetime(table["date"]), "n_valid": table["n_valid"]})
    daily = frame.groupby("date")["n_valid"].sum()
    position = (daily.index - daily.index.min()).days // days
    grouped = daily.groupby(position)
    out = pd.DataFrame({"start": grouped.apply(lambda g: g.index.min()), "end": grouped.apply(lambda g: g.index.max()), "days": grouped.size(), "readings": grouped.sum()})
    out["active_pct"] = 100.0 * out["readings"] / (out["days"] * READINGS_PER_DAY)
    out["full"] = out["days"] == days
    return out.reset_index(drop=True)


def cgm_active_time(table: pd.DataFrame) -> pd.DataFrame:
    """Check rows for the share of readings present: overall, per 14-day window and per sensor model's era is in ``cgm_eras``."""
    windows = active_time_windows(table)
    full = windows[windows["full"]]
    below = full[full["active_pct"] < 100 * ACTIVE_TIME_MIN]
    overall = _pct(table["n_valid"].sum(), len(table) * SLOTS_PER_HOUR)
    rows = [
        _row("CGM active time, whole record", "hour-day table", "readings present / readings possible (%)", overall, "none", "not applicable", "descriptive"),
        _row("CGM active time per 14-day window", "hour-day table", "full windows with active time below 70%", f"{len(below)} of {len(full)} (lowest {full['active_pct'].min():.1f}%, median {full['active_pct'].median():.1f}%)",
             "the consensus asks for at least 70% active time in a 14-day report", "no concern" if below.empty else "concern",
             "every full window meets the standard" if below.empty else "windows starting " + ", ".join(str(d.date()) for d in below["start"].head(6)) + (" ..." if len(below) > 6 else "") + " fall short"),
    ]
    return _frame(rows)


def gap_distribution(cgm: pd.DataFrame) -> pd.DataFrame:
    """Runs of missing 5-minute slots between consecutive valid readings, binned by length, with the minutes they cover."""
    stamps = pd.to_datetime(cgm["timestamp"]).sort_values().to_numpy()
    missing_slots = (np.diff(stamps) / np.timedelta64(READING_MINUTES, "m")).astype(int) - 1
    missing_slots = missing_slots[missing_slots > 0]
    edges = [(1, 1, "5 min"), (2, 5, "10-25 min"), (6, 11, "30-55 min"), (12, 23, "1-2 h"), (24, 71, "2-6 h"), (72, 10**9, "6 h or more")]
    rows = []
    for low, high, label in edges:
        inside = missing_slots[(missing_slots >= low) & (missing_slots <= high)]
        rows.append({"gap_length": label, "gaps": len(inside), "missing_minutes": int(inside.sum()) * READING_MINUTES})
    out = pd.DataFrame(rows)
    out["share_of_missing_minutes_pct"] = 100.0 * out["missing_minutes"] / max(out["missing_minutes"].sum(), 1)
    return out


def cgm_gaps(cgm: pd.DataFrame) -> pd.DataFrame:
    """Check rows for the gaps in the valid CGM series: one row per gap-length bin and a row for the longest gap."""
    bins = gap_distribution(cgm)
    stamps = pd.to_datetime(cgm["timestamp"]).sort_values()
    longest = int((stamps.diff().max() / pd.Timedelta(minutes=READING_MINUTES)) - 1) * READING_MINUTES
    rows = [_row(f"CGM gaps of {row.gap_length}", "valid CGM readings", "gaps and the minutes they cover", f"{row.gaps} gaps, {row.missing_minutes} missing minutes ({row.share_of_missing_minutes_pct:.1f}% of all missing minutes)",
                 "none: gap length says whether a sensor stopped (long) or a reading dropped (short)", "not applicable", "descriptive") for row in bins.itertuples(index=False)]
    rows.append(_row("Longest CGM gap", "valid CGM readings", "minutes between consecutive valid readings, less one slot", f"{longest / 60:.1f} h", "none", "not applicable", "descriptive"))
    return _frame(rows)


def cgm_duplicates_and_values(raw: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Duplicate readings, impossible values, LOW and HIGH counts and flat runs.

    Args:
        raw: every Tandem CGM row of the merged dataset (``duplicate_count`` is the number of raw reports behind a slot).
        valid: the readings the analyses use (inside the sensor's reporting range).

    Returns:
        The check rows and the issue rows (one per impossible reading and per flat run).
    """
    glucose = pd.to_numeric(raw["glucose"], errors="coerce")
    below, ceiling = glucose < SENSOR_FLOOR_MGDL, glucose >= SENSOR_CEILING_MGDL
    low_total = int(((glucose < LOW_MGDL) & glucose.notna()).sum())
    dup = pd.to_numeric(raw["duplicate_count"], errors="coerce") if "duplicate_count" in raw else pd.Series(1, index=raw.index)
    stamps = pd.to_datetime(raw["timestamp"])
    rows = [
        _row("Duplicate timestamps", "Tandem CGM rows", "rows sharing a timestamp and pump", int(raw.duplicated(["timestamp", "pump_serial"]).sum()), "none expected after the merge", "no concern" if not raw.duplicated(["timestamp", "pump_serial"]).any() else "concern",
             "one reading per slot and pump" if not raw.duplicated(["timestamp", "pump_serial"]).any() else "duplicate slots remain after the merge"),
        _row("Raw reports merged into one slot", "Tandem CGM rows", "slots built from 2 or more raw reports", f"{_pct((dup > 1).sum(), len(raw))}% of slots (up to {int(dup.max())} reports)", "none", "not applicable",
             "a slot's reports come from overlapping downloads and backfill; they are one reading after the merge"),
        _row("Readings below the sensor floor (LOW)", "Tandem CGM rows", f"readings under {SENSOR_FLOOR_MGDL:.0f} mg/dL (of which exactly 0)", f"{int(below.sum())} ({int((glucose == 0).sum())} are 0); {_pct(below.sum(), low_total)}% of the {low_total} readings under {LOW_MGDL:.0f}",
             "readings under 40 are outside the reporting range and are left out of every analysis", "concern" if below.any() else "no concern",
             "a LOW reading is dropped, not counted as below 54: time below 54 is slightly understated" if below.any() else "no reading under 40"),
        _row("Readings at the sensor ceiling (HIGH)", "Tandem CGM rows", f"readings of {SENSOR_CEILING_MGDL:.0f} mg/dL or more", int(ceiling.sum()), "readings above 400 are outside the reporting range", "concern" if ceiling.any() else "no concern",
             "HIGH readings are dropped, so time above 250 is understated" if ceiling.any() else "none present; the maximum stored reading is " + f"{glucose.max():.0f} mg/dL"),
        _row("Non-integer or missing glucose", "Tandem CGM rows", "rows", int(glucose.isna().sum() + (glucose.dropna() % 1 != 0).sum()), "readings are whole mg/dL", "no concern" if not (glucose.isna().any() or (glucose.dropna() % 1 != 0).any()) else "concern", "units look like mg/dL"),
        _row("Units", "valid CGM readings", "median glucose (mg/dL)", f"{valid['glucose'].median():.0f}", "a median under 30 would mean mmol/L", "no concern" if valid["glucose"].median() > 30 else "concern",
             "values are on the mg/dL scale" if valid["glucose"].median() > 30 else "values look like mmol/L"),
        _row("Readings off the 5-minute grid", "Tandem CGM rows", "readings with a minute that is not a multiple of 5 or a non-zero second", int(((stamps.dt.minute % 5 != 0) | (stamps.dt.second != 0)).sum()), "every reading sits on its 5-minute slot", "no concern" if not ((stamps.dt.minute % 5 != 0) | (stamps.dt.second != 0)).any() else "concern", "readings are placed on 5-minute slots"),
    ]
    issues = [{"step": "CGM impossible value", "reason": "reading below 40 mg/dL" if low < SENSOR_FLOOR_MGDL else "reading at or above 400 mg/dL", "pump_serial": pump_key(pump), "start": ts, "end": ts, "n": 1, "unit": "readings", "field": "glucose_mgdl", "value": f"{low:.0f}"}
              for ts, pump, low in zip(raw.loc[below | ceiling, "timestamp"], raw.loc[below | ceiling, "pump_serial"], glucose[below | ceiling])]
    series = valid.sort_values("timestamp")
    values = series["glucose"].to_numpy()
    slot = (pd.to_datetime(series["timestamp"]).to_numpy() - np.datetime64("1970-01-01")) // np.timedelta64(READING_MINUTES, "m")
    same = (np.diff(values) == 0) & (np.diff(slot) == 1)
    flat = [(a, b + 1) for a, b in _runs(same) if b - a + 1 >= FLAT_RUN_SLOTS]
    flat_slots = sum(b - a for a, b in flat)
    longest = max([b - a for a, b in flat], default=0)
    for a, b in flat:
        issues.append({"step": "CGM flat run", "reason": f"{b - a} identical readings in a row", "pump_serial": pump_key(series["pump_serial"].iloc[a]), "start": pd.Timestamp(series["timestamp"].iloc[a]),
                       "end": pd.Timestamp(series["timestamp"].iloc[b - 1]), "n": b - a, "unit": "readings", "field": "glucose_mgdl", "value": f"{values[a]:.0f}"})
    rows.append(_row("Flat runs", "valid CGM readings", f"runs of {FLAT_RUN_SLOTS} or more identical readings in consecutive slots", f"{len(flat)} runs, {flat_slots} readings, longest {longest}", "a CGM rarely repeats one value for an hour", "no concern" if not flat else "concern",
                     "none" if not flat else "a stuck sensor value would be counted as a real hour of glucose"))
    return _frame(rows), _issues(issues)


def compression_lows(valid: pd.DataFrame) -> pd.DataFrame:
    """Check row for night-time lows that look like pressure on the sensor: an abrupt fall into a short low that recovers quickly.

    A candidate is a run of readings under 70 that starts between 23:00 and 07:00 after a fall of at least 30 mg/dL over the
    previous 15 minutes, lasts an hour or less and is followed by a rise of at least 30 mg/dL within 20 minutes. It is a
    pattern, not proof of a sensor artifact.
    """
    series = valid.sort_values("timestamp").reset_index(drop=True)
    values = series["glucose"].to_numpy(dtype=float)
    stamps = pd.to_datetime(series["timestamp"])
    slot = ((stamps - pd.Timestamp("1970-01-01")) // pd.Timedelta(minutes=READING_MINUTES)).to_numpy()
    low = values < LOW_MGDL
    episodes = [(a, b) for a, b in _runs(low) if slot[b - 1] - slot[a] == b - 1 - a]  # unbroken in time
    total = night = candidates = 0
    for a, b in episodes:
        total += 1
        hour = stamps.iloc[a].hour
        if not (hour >= COMPRESSION_NIGHT_HOURS[0] or hour < COMPRESSION_NIGHT_HOURS[1]):
            continue
        night += 1
        before = values[a - 3] if a >= 3 and slot[a] - slot[a - 3] == 3 else np.nan
        after = values[b + 3] if b + 3 < len(values) and slot[b + 3] - slot[b] == 3 else np.nan
        if values[a] and before - values[a] >= COMPRESSION_FALL_MGDL and (b - a) * READING_MINUTES <= 60 and after - values[b - 1] >= COMPRESSION_FALL_MGDL:
            candidates += 1
    share = candidates / night if night else 0.0
    return _frame([_row("Compression-low pattern", "valid CGM readings", "night low episodes with an abrupt fall and fast recovery", f"{candidates} of {night} night low episodes ({100 * share:.1f}%); {total} low episodes in all",
                        "a share above 20% of night lows would make night lows doubtful", "concern" if share > 0.2 else "no concern",
                        "a large share of night lows may be pressure on the sensor, not glucose" if share > 0.2 else "most night lows do not look like abrupt, fast-recovering dips")])


def cgm_eras(raw: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Readings, span and glucose outcomes by sensor model (the pump event that carried each reading), with a check row.

    Returns:
        The per-model table and one check row: more than 5% of readings from a model other than the latest is a concern.
    """
    names = raw["source_event_name"].astype(str)
    rows = []
    for name, group in raw.assign(model=names).groupby("model"):
        data = valid[valid["timestamp"].isin(group["timestamp"])]
        glucose = data["glucose"]
        rows.append({"sensor_model_event": name, "readings": len(group), "first": group["timestamp"].min(), "last": group["timestamp"].max(),
                     "time_in_range_pct": _pct(glucose.between(70, 180).sum(), len(glucose)), "time_below_70_pct": _pct((glucose < 70).sum(), len(glucose)), "mean_glucose_mgdl": round(float(glucose.mean()), 1)})
    table = pd.DataFrame(rows).sort_values("first").reset_index(drop=True)
    other = table["readings"].iloc[:-1].sum() / table["readings"].sum()
    check = _frame([_row("Sensor-model eras", "Tandem CGM rows", "share of readings from a sensor model other than the latest", f"{100 * other:.1f}% ({', '.join(table['sensor_model_event'].iloc[:-1])} before {table['first'].iloc[-1].date()})",
                         "more than 5% of readings from another model", "concern" if other > 0.05 else "no concern",
                         "earlier sensor model's readings differ in accuracy and warm-up" if other > 0.05 else "the earlier sensor model covers a small part of the record")])
    return table, check


def sensor_sessions(sessions: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Check rows for sensor sessions from the pump's join and stop events.

    Args:
        sessions: ``pump_serial``, ``timestamp`` and ``kind`` ("join" or "stop"), one row per sensor session event.
        valid: the readings the analyses use.

    Returns:
        Check rows: sessions, their length, and the minutes from a join to the first valid reading (the warm-up).
    """
    if sessions.empty:
        return _frame([_row("Sensor sessions", "pump event log", "sessions", 0, "", "not assessed", "no sensor session events were supplied")])
    joins = sessions[sessions["kind"].eq("join")].sort_values("timestamp")
    stops = sessions[sessions["kind"].eq("stop")].sort_values("timestamp")
    lengths = []
    for pump, group in stops.groupby("pump_serial"):
        starts = joins[joins["pump_serial"].eq(pump)]["timestamp"].to_numpy()
        for stop in group["timestamp"]:
            before = starts[starts <= stop.to_datetime64()]
            if len(before):
                lengths.append((stop - pd.Timestamp(before[-1])).total_seconds() / 86400)
    lengths = np.asarray(lengths)
    slots = np.sort(pd.to_datetime(valid["timestamp"]).to_numpy())
    first = np.searchsorted(slots, joins["timestamp"].to_numpy(dtype="datetime64[ns]"))
    inside = first < len(slots)
    waits = (slots[first[inside]] - joins["timestamp"].to_numpy(dtype="datetime64[ns]")[inside]) / np.timedelta64(1, "m")
    return _frame([
        _row("Sensor sessions", "pump event log", "join and stop events; matched session length (days)", f"{len(joins)} joins, {len(stops)} stops; length median {np.median(lengths):.1f} days (5th to 95th percentile {np.percentile(lengths, 5):.1f} to {np.percentile(lengths, 95):.1f}); {int((lengths < 1).sum())} of {len(lengths)} under a day",
             "none", "not applicable", "descriptive; a session under a day is a restart, not a sensor life"),
        _row("Warm-up after a sensor joins", "pump event log, valid CGM readings", "minutes from a join event to the first valid reading", f"median {np.median(waits):.0f} (5th to 95th percentile {np.percentile(waits, 5):.0f} to {np.percentile(waits, 95):.0f}); {int((waits > 120).sum())} of {len(waits)} joins wait more than 2 hours",
             "none", "not applicable", "descriptive; readings during warm-up are absent, so the first minutes of a session are not in the data"),
    ])


# ---------------------------------------------------------------------------
# Devices and clock
# ---------------------------------------------------------------------------

def pump_eras(raw: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Check rows for the pumps behind the record: spans, overlap and hours that have to be assigned to a pump without readings."""
    pumps = raw.assign(pump=raw["pump_serial"].map(pump_key)).groupby("pump")["timestamp"].agg(["min", "max", "count"]).sort_values("min")
    overlap = 0
    names = list(pumps.index)
    for first, second in zip(names[:-1], names[1:]):
        overlap += max(0, int((pumps.loc[first, "max"] - pumps.loc[second, "min"]) / pd.Timedelta(days=1)))
    empty = table[table["n_valid"].eq(0)]
    return _frame([
        _row("Pump eras", "Tandem CGM rows", "pumps, first and last reading, readings", "; ".join(f"pump {name}: {row['min'].date()} to {row['max'].date()}, {row['count']} readings" for name, row in pumps.iterrows()), "none", "not applicable", "descriptive"),
        _row("Pump eras overlap", "Tandem CGM rows", "days on which two pumps both report readings", overlap, "one pump per day", "no concern" if overlap == 0 else "concern", "the pumps' reading spans do not overlap" if overlap == 0 else "two pumps report at once, so settings may be read from the wrong one"),
        _row("Hours without readings assigned a pump", "hour-day table", "hours with no readings (pump carried from a neighbour)", len(empty), "none", "not applicable", "the settings of such an hour come from the carried pump; the hour is not modelled (no readings)"),
    ])


def pump_change_break(table: pd.DataFrame, *, block_days: int = BLOCK_DAYS) -> pd.DataFrame:
    """Does glucose step at the pump change by more than it does between other adjacent blocks of days?

    The change is the first hour whose pump differs from the one before. The step is the difference in time in range, time
    below 70 and mean glucose between the ``block_days`` days after and before it, set against the same step over every
    other pair of adjacent, non-overlapping blocks (counted back from the end of the record). It is a description of
    one change, and the pump change coincides with a new sensor and new settings, so a step cannot be attributed to the pump.
    """
    pumps = table["pump_serial"].astype(str).to_numpy()
    change = np.flatnonzero(pumps[1:] != pumps[:-1]) + 1
    if not len(change):
        return _frame([_row("Break at the pump change", "hour-day table", "step in glucose outcomes", "", "", "not applicable", "the record has one pump")])
    date = pd.to_datetime(table["date"])
    daily = table.assign(date=date).groupby("date")[GLUCOSE_COLUMNS].sum()
    daily["tir"] = daily["n_in_range"] / daily["n_valid"]
    daily["below70"] = daily["n_below_70"] / daily["n_valid"]
    daily["mean"] = daily["glucose_sum"] / daily["n_valid"]
    day = pd.Timestamp(table["date"].iloc[change[0]])
    rows = []
    for measure, scale, label in (("tir", 100, "time in range (points)"), ("below70", 100, "time below 70 (points)"), ("mean", 1, "mean glucose (mg/dL)")):
        blocks = []
        end = daily.index.max()
        while end - pd.Timedelta(days=block_days - 1) >= daily.index.min():
            start = end - pd.Timedelta(days=block_days - 1)
            window = daily.loc[start:end]
            blocks.append((start, end, _block_value(window, measure)))
            end = start - pd.Timedelta(days=1)
        blocks = blocks[::-1]
        steps = np.array([b[2] - a[2] for a, b in zip(blocks[:-1], blocks[1:])]) * scale
        before = _block_value(daily.loc[day - pd.Timedelta(days=block_days):day - pd.Timedelta(days=1)], measure)
        after = _block_value(daily.loc[day:day + pd.Timedelta(days=block_days - 1)], measure)
        step = (after - before) * scale
        percentile = 100 * float((np.abs(steps) < abs(step)).mean())
        rows.append(_row(f"Break at the pump change: {label}", "hour-day table", f"{block_days} days after minus {block_days} days before",
                         f"{step:+.1f} (larger in size than {percentile:.0f}% of the {len(steps)} steps between other adjacent {block_days}-day blocks; their 5th to 95th percentile {np.percentile(steps, 5):+.1f} to {np.percentile(steps, 95):+.1f})",
                         "the step is among the 5% largest absolute steps between adjacent blocks", "concern" if percentile >= 95 else "no concern",
                         "the step is unusually large; the pump change also brought a new sensor and settings, so it cannot be attributed" if percentile >= 95 else "the step is within what adjacent blocks of days do anyway"))
    return _frame(rows)


def _block_value(window: pd.DataFrame, measure: str) -> float:
    if measure == "mean":
        return float(window["glucose_sum"].sum() / window["n_valid"].sum())
    column = {"tir": "n_in_range", "below70": "n_below_70"}[measure]
    return float(window[column].sum() / window["n_valid"].sum())


ONE_HOUR_MAX_MINUTES = 75
ZONE_CHANGE_MAX_MINUTES = 210
DST_FOLLOW_DAYS = 7
CLOCK_HOURS_SHARE_MAX = 0.01
TRAVEL_DAYS_SHARE_MAX = 0.01
TIME_CHANGED_EVENT = 13


def _hours_since_clock_change(when: pd.Timestamp, zone: ZoneInfo) -> tuple[float, int] | None:
    """Hours from the latest local clock change (daylight-saving transition) before ``when`` within ``DST_FOLLOW_DAYS`` days, and its direction (+1 spring, -1 autumn)."""
    hours = pd.date_range(when.floor("h") - pd.Timedelta(days=DST_FOLLOW_DAYS), when.floor("h"), freq="h")
    offsets = np.array([stamp.to_pydatetime().replace(tzinfo=zone).utcoffset().total_seconds() for stamp in hours])
    moved = np.flatnonzero(np.diff(offsets) != 0)
    if not len(moved):
        return None
    index = int(moved[-1]) + 1
    return (when - hours[index]).total_seconds() / 3600.0, int(np.sign(offsets[index] - offsets[index - 1]))


def clock_jumps(jumps: pd.DataFrame, table: pd.DataFrame, *, zone: ZoneInfo = LOCAL_ZONE) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pump clock changes and timestamp inversions from the pump's own event sequence.

    A clock change is a time-changed event (13); its size is the jump in time from the event before it in the pump's sequence
    (accurate to the few minutes between two events). A one-hour change within a week after a local clock change is the person
    setting the pump to the new time; the hours between the transition and that change are hours on the wrong clock.

    Args:
        jumps: one row per pair of consecutive events (by sequence number) whose timestamps differ by more than 20 minutes or go
            backward: ``pump_serial``, ``sequence``, ``timestamp``, ``previous_timestamp``, ``event`` and ``jump_minutes``.
        table: the hour-day table (the span the checks count over).
        zone: the home time zone whose clock changes the pump is expected to follow.

    Returns:
        The table of time-changed events with a class, the check rows and the issue rows.
    """
    if jumps.empty:
        return jumps, _frame([_row("Pump clock changes", "pump event log", "clock changes", 0, "", "not assessed", "no event log was supplied")]), _issues([])
    first_hour, last_hour = pd.Timestamp(table["hour_start"].min()), pd.Timestamp(table["hour_start"].max()) + pd.Timedelta(hours=1)
    inside = jumps[(jumps["timestamp"] >= first_hour) & (jumps["timestamp"] <= last_hour)]
    changes = inside[inside["event"].eq(TIME_CHANGED_EVENT)].sort_values(["pump_serial", "sequence"]).copy()
    size = changes["jump_minutes"].abs()
    changes["class"] = np.where(size <= ONE_HOUR_MAX_MINUTES, "one hour", np.where(size <= ZONE_CHANGE_MAX_MINUTES, "time-zone change (2-3.5 hours)", "other jump"))
    follows, lag = [], []
    for _, row in changes.iterrows():
        near = _hours_since_clock_change(row["timestamp"], zone) if row["class"] == "one hour" else None
        ok = near is not None and np.sign(row["jump_minutes"]) == near[1]
        follows.append(ok)
        lag.append(near[0] if ok else np.nan)
    changes["follows_clock_change"], changes["hours_after_clock_change"] = follows, lag
    hours_in_record = len(table)
    one = changes[changes["class"].eq("one hour")]
    misaligned = float(np.nansum(one["hours_after_clock_change"]))
    zone_changes = changes[changes["class"].str.startswith("time-zone")]
    away = 0.0
    for _, group in changes[~changes["follows_clock_change"] & ~changes["class"].eq("other jump")].groupby("pump_serial"):
        offset, previous = 0.0, None  # minutes the pump clock is from home time, in half-hour steps; a change that follows a local clock change is the pump catching up
        for stamp, jump in zip(group["timestamp"], group["jump_minutes"]):
            if previous is not None and abs(offset) >= 90:
                away += (stamp - previous).total_seconds() / 86400
            offset += np.round(jump / 30.0) * 30.0
            previous = stamp
        if previous is not None and abs(offset) >= 90:
            away += max(0.0, (last_hour - previous).total_seconds() / 86400)
    other = changes[changes["class"].eq("other jump")]
    unexplained = inside[(inside["jump_minutes"] < -1) & ~inside["event"].eq(TIME_CHANGED_EVENT)]
    classes = changes["class"].value_counts()
    days = hours_in_record / 24
    rows = [
        _row("Pump clock changes", "pump event log", "time-changed events (13) in the record with a jump of more than 20 minutes", f"{len(changes)} ({'; '.join(f'{n} {k}' for k, n in classes.items())})", "none", "not applicable", "descriptive; the list is in Record_Issues"),
        _row("Clock changes at local clock-change dates", "pump event log", "one-hour changes within 7 days after a local clock change; hours on the wrong clock before them", f"{int(one['follows_clock_change'].sum())} of {len(one)} one-hour changes follow a clock change (median {np.nanmedian(one['hours_after_clock_change']) if one['follows_clock_change'].any() else float('nan'):.1f} h after it); {misaligned:.0f} hours on the wrong clock ({100 * misaligned / max(hours_in_record, 1):.2f}% of hours)",
             f"under {100 * CLOCK_HOURS_SHARE_MAX:.0f}% of hours", "no concern" if misaligned / max(hours_in_record, 1) < CLOCK_HOURS_SHARE_MAX else "concern",
             "the pump is set to the new time within hours, so few hours carry the wrong clock" if misaligned / max(hours_in_record, 1) < CLOCK_HOURS_SHARE_MAX else "the pump runs on the old clock for hours after each clock change"),
        _row("Time spent on a clock 2-3.5 hours from home", "pump event log", "days between a time-zone change out and back", f"{len(zone_changes)} changes; about {away:.0f} days ({100 * away / max(days, 1):.1f}% of the record)",
             f"under {100 * TRAVEL_DAYS_SHARE_MAX:.0f}% of days", "no concern" if away / max(days, 1) < TRAVEL_DAYS_SHARE_MAX else "concern",
             "few days are away from the home clock" if away / max(days, 1) < TRAVEL_DAYS_SHARE_MAX else "hour of day is the pump's clock: on these days it is not the home clock hour, which blurs hour-of-day patterns and the time-of-day settings"),
        _row("Other clock jumps", "pump event log", "time-changed events larger than 3.5 hours", f"{len(other)}" + (": " + ", ".join(str(t.date()) for t in other["timestamp"].head(8)) if len(other) else ""), "none", "no concern" if other.empty else "concern",
             "no large clock jumps" if other.empty else "the pump clock was far from the correct time on these dates, so pump-stamped events (boluses, basal, settings) may be off by hours there"),
        _row("Timestamp inversions", "pump event log", "events stamped earlier than the event before them, other than time-changed events", len(unexplained), "none expected", "no concern" if unexplained.empty else "concern",
             "event order and time agree" if unexplained.empty else "the pump logged events out of time order (a clock reset)"),
        _row("Shared clock of glucose, bolus and settings", "pump event log", "CGM reading times come from the sensor's own time, bolus and basal from the pump clock", "not testable on the stored tables", "the two clocks agree when the pump clock was right",
             "not assessed", "the stored CGM table keeps one time per reading; the dates of clock errors above are where the two could differ"),
    ]
    issues = [{"step": "pump clock jump", "reason": row["class"], "pump_serial": pump_key(row["pump_serial"]), "start": row["previous_timestamp"], "end": row["timestamp"], "n": 1, "unit": "events", "field": "jump_minutes", "value": f"{row['jump_minutes']:.0f}"}
              for _, row in changes.iterrows()]
    issues += [{"step": "timestamp inversion", "reason": "event stamped earlier than the one before it", "pump_serial": pump_key(row["pump_serial"]), "start": row["previous_timestamp"], "end": row["timestamp"], "n": 1, "unit": "events", "field": "jump_minutes", "value": f"{row['jump_minutes']:.0f}"}
               for _, row in unexplained.iterrows()]
    return changes, _frame(rows), _issues(issues)


def clock_alignment(table: pd.DataFrame, *, max_lag_hours: int = 3) -> pd.DataFrame:
    """Do the pump clock and the phone's clock keep the same offset throughout? Pump sleep mode against the phone's asleep share at lags of -3 to +3 hours, by pump.

    A person sets the pump's sleep mode around bedtime and the phone records sleep itself, so the two are not expected to
    line up at lag 0; what a clock error would change is the best lag between pumps. The check compares the best lag of each pump.
    """
    best, detail = {}, []
    for pump, group in table.groupby("pump_serial", sort=False):
        pump_values, phone_values = group["sleep_mode_share"].to_numpy(), group["apple_asleep_share"].to_numpy()
        if (np.isfinite(pump_values) & np.isfinite(phone_values)).sum() < 24 * 30:
            continue
        scores = {}
        for lag in range(-max_lag_hours, max_lag_hours + 1):
            a = pump_values[max(0, -lag): len(pump_values) - max(0, lag)]
            b = phone_values[max(0, lag): len(phone_values) - max(0, -lag)]
            keep = np.isfinite(a) & np.isfinite(b)
            scores[lag] = float(np.corrcoef(a[keep], b[keep])[0, 1]) if keep.sum() > 2 else float("nan")
        best[pump] = max(scores, key=lambda k: scores[k])
        detail.append(f"pump {pump}: best lag {best[pump]} h (correlation {scores[best[pump]]:.2f}, at 0 h {scores[0]:.2f})")
    if len(best) < 2:
        return _frame([_row("Pump clock against the phone's clock", "hour-day table", "best lag (hours) of pump sleep mode against the phone's sleep, by pump", "; ".join(detail), "needs two pumps with 30 days of both measures", "not assessed", "fewer than two pumps have both sleep measures")])
    spread = max(best.values()) - min(best.values())
    return _frame([_row("Pump clock against the phone's clock", "hour-day table", "best lag (hours) of pump sleep mode against the phone's sleep, by pump (lag k: pump at hour t against phone at hour t+k)", "; ".join(detail),
                        "the best lags of the pumps differ by at most 1 hour", "no concern" if spread <= 1 else "concern",
                        "the offset between the pump's sleep mode and the phone's sleep is the same on both pumps, so neither pump's clock is shifted against the phone's" if spread <= 1 else "the offset differs between pumps: one pump's clock may be shifted against the phone's")])


DISAGREEMENT_GAP_DAYS = 3


def _disagreement_runs(times: pd.Series, pumps: pd.Series, *, step: str, field: str, unit: str, hours: pd.Series | None = None) -> list[dict[str, object]]:
    """One issue row per stretch of days (gaps up to ``DISAGREEMENT_GAP_DAYS``) in which a reconstructed setting disagreed with the pump's own record, per pump."""
    frame = pd.DataFrame({"time": pd.to_datetime(times).to_numpy(), "pump": pumps.map(pump_key).to_numpy()})
    frame["date"] = frame["time"].dt.normalize()
    frame["hour"] = frame["time"].dt.hour
    rows = []
    for pump, group in frame.sort_values("time").groupby("pump"):
        days = group["date"].drop_duplicates().to_numpy()
        starts = [0] + [k for k in range(1, len(days)) if (days[k] - days[k - 1]) / np.timedelta64(1, "D") > DISAGREEMENT_GAP_DAYS] + [len(days)]
        for a, b in zip(starts[:-1], starts[1:]):
            part = group[(group["date"] >= days[a]) & (group["date"] <= days[b - 1])]
            clock = sorted(part["hour"].unique())
            rows.append({"step": step, "reason": "reconstructed value differs from the pump's own record", "pump_serial": pump, "start": pd.Timestamp(days[a]), "end": pd.Timestamp(days[b - 1]), "n": len(part), "unit": unit, "field": field,
                         "value": f"clock hours {clock[0]:02d}-{clock[-1]:02d} ({len(clock)} distinct)"})
    return rows


def _where(times: pd.Series) -> str:
    """The months and clock hours a set of disagreeing records fall in, in a few words."""
    stamps = pd.to_datetime(times)
    months = stamps.dt.to_period("M").astype(str).value_counts().sort_index()
    hours = stamps.dt.hour.value_counts()
    main = sorted(hours[hours >= 0.05 * hours.sum()].index)
    return f"{len(months)} months ({months.index[0]} to {months.index[-1]}); clock hours {', '.join(f'{h:02d}' for h in main)} hold at least 5% each"


# ---------------------------------------------------------------------------
# Settings reconstruction
# ---------------------------------------------------------------------------

def hour_unknown_reasons(history, table: pd.DataFrame) -> pd.Series:
    """The reason each hour's settings are unknown: the most frequent ``UNKNOWN_REASONS`` code among its twelve moments, ``""`` for a fully known hour.

    The reasons come from ``ProfileHistory.unknown_reasons``, the replay the hour-day table itself reads its settings from, so
    an hour has a reason exactly when ``settings_known_slots`` is below 12.
    """
    hours = pd.DatetimeIndex(table["hour_start"])
    offsets = pd.to_timedelta(np.arange(SLOTS_PER_HOUR) * READING_MINUTES, unit="min")
    moments = pd.DatetimeIndex((hours.to_numpy()[:, None] + offsets.to_numpy()[None, :]).ravel())
    pumps = pd.Series(np.repeat(table["pump_serial"].to_numpy(), SLOTS_PER_HOUR))
    codes = history.unknown_reasons(pumps, pd.Series(moments)).reshape(len(table), SLOTS_PER_HOUR)
    dominant = []
    for row in codes:
        names = [name for name in row if name]
        dominant.append(max(set(names), key=names.count) if names else "")
    return pd.Series(dominant, index=table.index, name="reason")


def unknown_settings(table: pd.DataFrame, reasons: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Why hours have unknown settings, by pump, clock hour and month, and whether those hours look different.

    Returns:
        The check rows, the detail table (pump, month, clock hour and reason with hour counts) and the issue rows (one per run of hours with readings).
    """
    unknown = reasons.ne("") & table["settings_known_slots"].lt(SLOTS_PER_HOUR)
    mismatch = int((reasons.ne("") != table["settings_known_slots"].lt(SLOTS_PER_HOUR)).sum())
    with_readings = unknown & table["n_valid"].gt(0)
    frame = table.assign(reason=reasons, month=pd.to_datetime(table["date"]).dt.to_period("M").astype(str), has_readings=table["n_valid"].gt(0))
    detail = frame[unknown].groupby(["pump_serial", "month", "hour", "reason"]).agg(hours=("hour", "size"), hours_with_readings=("has_readings", "sum")).reset_index()
    by_reason = frame[with_readings].groupby(["pump_serial", "reason"]).size()
    known = table["settings_known_slots"].eq(SLOTS_PER_HOUR) & table["n_valid"].gt(0)
    tir_known = _pct(table.loc[known, "n_in_range"].sum(), table.loc[known, "n_valid"].sum())
    tir_unknown = _pct(table.loc[with_readings, "n_in_range"].sum(), table.loc[with_readings, "n_valid"].sum())
    total = int(table["n_valid"].gt(0).sum())
    late = int((with_readings & table["hour"].ge(21)).sum())
    rows = [
        _row("Reasons cover every unknown hour", "ProfileHistory.unknown_reasons, hour-day table", "hours whose reason and settings_known_slots disagree", mismatch, "none", "no concern" if mismatch == 0 else "concern", "each unknown hour has exactly one dominant reason" if mismatch == 0 else "the reason replay and the table's replay differ"),
        _row("Hours with readings whose settings are not known for all 12 moments", "hour-day table", "hours with readings and unknown settings", f"{int(with_readings.sum())} of {total} ({_pct(with_readings.sum(), total)}%)", "none", "concern" if with_readings.any() else "no concern",
             "these hours cannot enter the outcomes model; " + "; ".join(f"pump {pump} {reason}: {n}" for (pump, reason), n in by_reason.items()) if with_readings.any() else "every hour with readings has its settings"),
        _row("Unknown hours late in the day", "hour-day table", "hours with readings and unknown settings at 21:00-23:59", late, "an open last slot was the largest known cause before the pump's rate log confirmed it", "no concern" if late <= max(1, 0.02 * total) else "concern",
             "the last profile slot no longer drives unknown hours" if late <= max(1, 0.02 * total) else "the last profile slot still leaves late-evening hours unknown"),
        _row("Time in range of hours with unknown settings against known", "hour-day table", "pooled time in range, unknown vs known settings (%)", f"{tir_unknown} vs {tir_known}", "see the composition test", "not applicable", "the two groups differ in clock hour and month; the standardised comparison is in Missingness_Stationarity"),
    ]
    issues = _run_issues(table, with_readings.to_numpy(), step="settings unknown (hours with readings)", reason=reasons, field="settings_known_slots", value="fewer than 12 of 12 moments")
    return _frame(rows), detail, _issues(issues)


def calculator_agreement(carbs: pd.DataFrame, history) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Does the reconstructed profile agree with the settings the pump's own bolus calculator logged at each meal?

    Compares the calculator's I:C, correction factor and target at each carbohydrate entry with ``values_at`` at the moment the
    pump recorded the entry (the profile in force at that moment, whichever hour it is).

    Returns:
        The check rows (one per setting) and the issue rows (one per stretch of days with disagreeing meals).
    """
    stamps = pd.to_datetime(carbs["event_time"]).fillna(pd.to_datetime(carbs["timestamp"]))
    pumps = carbs["pump_serial"].map(pump_key)
    rows, issues = [], []
    for column, field, label, tolerance in (("carb_ratio_g_per_unit", "carb_ratio_g_per_unit", "I:C (g/U)", 0.05), ("isf_mgdl_per_unit", "isf_mgdl_per_unit", "correction factor (mg/dL/U)", 0.5), ("target_bg_mgdl", "target_bg_mgdl", "target (mg/dL)", 0.5)):
        logged = pd.to_numeric(carbs[column], errors="coerce")
        rebuilt = pd.Series(history.values_at(pumps, stamps, field), index=carbs.index)
        usable = logged.notna() & (logged > 0)
        known = usable & rebuilt.notna()
        differs = known & ((logged - rebuilt).abs() > tolerance)
        late = known & stamps.dt.hour.ge(21)
        late_match = ((logged[late] - rebuilt[late]).abs() <= tolerance).mean() if late.any() else float("nan")
        matched = int(known.sum() - differs.sum())
        share = matched / known.sum() if known.any() else float("nan")
        ok = share >= AGREEMENT_MIN
        rows.append(_row(f"Pump calculator {label} against the reconstruction", "carbs table, ProfileHistory", "meals with the setting reconstructed whose logged value matches",
                         f"{matched} of {int(known.sum())} ({100 * share:.1f}%); {int((usable & ~rebuilt.notna()).sum())} of {int(usable.sum())} meals have no reconstruction; 21:00-23:59: {100 * late_match:.1f}% of {int(late.sum())}" + ("" if ok else f"; disagreements in {_where(stamps[differs])}"),
                         f"at least {100 * AGREEMENT_MIN:.0f}% match", "no concern" if ok else "concern",
                         "the reconstructed setting is what the pump used at meals" if ok else "the reconstruction disagrees with what the pump used at those meals, so the settings of the hour-day table are wrong there (they are not unknown)"))
        issues += _disagreement_runs(stamps[differs], pumps[differs], step=f"calculator {label} differs from the reconstruction", field=field, unit="meals")
    return _frame(rows), _issues(issues)


def basal_log_agreement(rates: pd.DataFrame, history, *, tolerance: float = 0.0006) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Does the reconstructed basal rate equal the rate the pump logs (event 279) as the active profile's, at every known moment?

    Returns:
        The check row and the issue rows (one per stretch of days with disagreeing records).
    """
    if rates.empty:
        return _frame([_row("Pump-logged profile basal rate against the reconstruction", "event 279", "agreement", "", "", "not assessed", "no event 279 records were supplied")]), _issues([])
    rebuilt = pd.Series(history.values_at(rates["pump_serial"], rates["timestamp"], "basal_units_per_hour"), index=rates.index)
    known = rebuilt.notna()
    logged = rates["rate"].astype(float)
    differs = known & ((logged - rebuilt).abs() > tolerance)
    late = known & pd.to_datetime(rates["timestamp"]).dt.hour.ge(21)
    late_agree = ((logged[late] - rebuilt[late]).abs() <= tolerance).mean() if late.any() else float("nan")
    matched = int(known.sum() - differs.sum())
    share = matched / known.sum() if known.any() else float("nan")
    ok = share >= AGREEMENT_MIN
    row = _row("Pump-logged profile basal rate against the reconstruction", "event 279 (profile basal rate), ProfileHistory", "records at known moments whose logged rate equals the reconstructed basal",
               f"{matched} of {int(known.sum())} ({100 * share:.2f}%); {int((~known).sum())} of {len(known)} records are at moments with unknown settings; 21:00-23:59: {100 * late_agree:.2f}% of {int(late.sum())}" + ("" if ok else f"; disagreements in {_where(rates.loc[differs, 'timestamp'])}"),
               f"at least {100 * AGREEMENT_MIN:.0f}% agree", "no concern" if ok else "concern",
               "the pump's own log confirms the reconstructed rate" if ok else "the reconstructed basal differs from the pump's own log at those moments, so the settings of the hour-day table are wrong there (they are not unknown)")
    return _frame([row]), _issues(_disagreement_runs(rates.loc[differs, "timestamp"], rates.loc[differs, "pump_serial"], step="pump-logged basal rate differs from the reconstruction", field="basal_units_per_hour", unit="records"))


def replay_event_crosscheck(table: pd.DataFrame, history, profile_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Is every setting-change event the replay finds a real edit or profile switch in the pump's raw record, and the reverse?

    Forward: each event (consecutive days holding a changed hour) must have a raw segment edit to the then-active profile, or a profile
    activation, dated between the day before its first day and its last day. Reverse: each raw edit that changed a value of the profile
    running at the time, inside the pump's visible span and not on the day the pump changed, must fall inside some event's days.
    The 15 isolated events are recomputed from the event dates (no other event within 14 days) and compared with the replay's own count.

    Returns:
        The per-event table, the check rows and the issue rows (events without a raw record, raw edits without an event).
    """
    from . import replay
    from .changes import detect_setting_changes
    from .pump_profiles import PROFILE_ACTIVATED, PROFILE_EVENT

    grid = replay.day_grid(table)
    events = replay.find_events(grid)
    _, isolated = replay.cluster_events(events)
    raw = detect_setting_changes(profile_events)
    raw = raw[raw["timestamp"].notna()].copy()
    raw["pump"] = raw["pump_serial"].map(pump_key)
    raw["date"] = pd.to_datetime(raw["timestamp"]).dt.normalize()
    raw["active"] = [history.active_profile(pump, ts.to_pydatetime()) == profile for pump, ts, profile in zip(raw["pump"], raw["timestamp"], raw["profile_id"])]
    visible = raw[[history.visible.get(pump) is not None and history.visible[pump][0] <= day.date() <= history.visible[pump][1] for pump, day in zip(raw["pump"], raw["date"])]]
    pumps = table["pump_serial"].astype(str).to_numpy()
    first_day = {pump: pd.Timestamp(table["date"][pumps == pump].min()) for pump in set(pumps)}
    visible = visible[[day > first_day.get(pump, pd.Timestamp.min) for pump, day in zip(visible["pump"], visible["date"])]]
    activations = profile_events[profile_events["event"].eq(PROFILE_EVENT) & profile_events["status"].eq(PROFILE_ACTIVATED)].copy()
    activations["date"] = pd.to_datetime(activations["timestamp"]).dt.normalize()
    rows = []
    for event in events:
        start, stop = grid.dates[event.first] - pd.Timedelta(days=1), grid.dates[event.last]
        edits = visible[visible["active"] & visible["date"].between(start, stop)]
        switched = activations[activations["date"].between(start, stop)]
        kind = "setting edit to the running profile" if len(edits) else "profile activation" if len(switched) else "none"
        rows.append({"event_id": event.event_id, "first_day": grid.dates[event.first].date(), "last_day": grid.dates[event.last].date(), "settings": event.settings, "direction": event.direction,
                     "raw_edits": len(edits), "raw_activations": len(switched), "matched_by": kind, "isolated_in_replay": event.event_id in isolated})
    table_out = pd.DataFrame(rows)
    covered = np.zeros(len(visible), dtype=bool)
    for event in events:
        start, stop = grid.dates[event.first] - pd.Timedelta(days=1), grid.dates[event.last]
        covered |= visible["date"].between(start, stop).to_numpy()
    running = visible[visible["active"]]
    missed = running[~covered[visible["active"].to_numpy()]]
    # Isolation recomputed from the dates alone: no other event starts within 14 days of this one's first or last day.
    recomputed = {e.event_id for e in events if not any(o.event_id != e.event_id and o.first - e.last <= replay.CLEARANCE_DAYS and e.first - o.last <= replay.CLEARANCE_DAYS for o in events)}
    unmatched = table_out[table_out["matched_by"].eq("none")]
    n_events = len(events)
    rows_out = [
        _row("Each replay event has a raw edit or profile switch", "hour-day table, pump profile events", "events with a raw segment edit to the running profile or a profile activation within a day of their days",
             f"{n_events - len(unmatched)} of {n_events} ({int((table_out['matched_by'].eq('setting edit to the running profile')).sum())} by an edit, {int(table_out['matched_by'].eq('profile activation').sum())} by an activation only)",
             "every event matches a raw record", "no concern" if unmatched.empty else "concern", "every event traces to a raw record" if unmatched.empty else f"{len(unmatched)} events have no raw edit or activation nearby (listed in Record_Issues)"),
        _row("Each raw edit of a running profile is inside a replay event", "pump profile events, hour-day table", "raw value changes to the running profile, inside the visible span, outside any event's days",
             f"{len(missed)} of {len(running)}", "none missed", "no concern" if missed.empty else "concern", "no raw edit is missing from the events" if missed.empty else "raw edits changed a running profile's value on days the table shows no change (the edit may not have changed an hour's setting, or the hours were unknown)"),
        _row("Isolated events recomputed from the dates", "replay.cluster_events", "events with no other event within 14 days", f"{len(recomputed)} recomputed, {len(isolated)} in the replay", "equal", "no concern" if recomputed == set(isolated) else "concern",
             "the replay's isolation matches an independent recomputation" if recomputed == set(isolated) else "isolation differs"),
    ]
    issues = [{"step": "replay event without a raw record", "reason": "no raw edit or activation within a day", "pump_serial": "", "start": pd.Timestamp(r["first_day"]), "end": pd.Timestamp(r["last_day"]), "n": 1, "unit": "events", "field": "settings", "value": r["settings"]} for _, r in unmatched.iterrows()]
    issues += [{"step": "raw edit outside any replay event", "reason": "a value of the running profile changed", "pump_serial": r["pump"], "start": r["timestamp"], "end": r["timestamp"], "n": 1, "unit": "events", "field": r["setting"], "value": f"{r['old_value']} to {r['new_value']}"} for _, r in missed.iterrows()]
    return table_out, _frame(rows_out), _issues(issues)


# ---------------------------------------------------------------------------
# Meals and insulin
# ---------------------------------------------------------------------------

def meal_checks(carbs: pd.DataFrame, bolus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carbohydrate entries and boluses against each other, rounding of the entries, and how the entries vary over 14-day windows.

    Returns:
        The check rows and the issue rows (entries without a bolus, carbohydrate boluses without an entry).
    """
    entries = carbs.assign(when=pd.to_datetime(carbs["event_time"]).fillna(pd.to_datetime(carbs["timestamp"])).dt.floor(f"{READING_MINUTES}min").astype("datetime64[ns]"), pump=carbs["pump_serial"].map(pump_key)).sort_values("when")
    boluses = bolus.assign(when=pd.to_datetime(bolus["timestamp"]).astype("datetime64[ns]"), pump=bolus["pump_serial"].map(pump_key)).sort_values("when")
    joined = pd.merge_asof(entries[["when", "pump"]], boluses[["when", "pump"]].rename(columns={"when": "bolus_when"}).assign(match=lambda f: f["bolus_when"]), left_on="when", right_on="bolus_when", by="pump", direction="nearest", tolerance=MEAL_BOLUS_WINDOW)
    without_bolus = int(joined["bolus_when"].isna().sum())
    nearest = pd.merge_asof(entries[["when", "pump"]], boluses[["when", "pump"]].rename(columns={"when": "bolus_when"}), left_on="when", right_on="bolus_when", by="pump", direction="nearest")
    lonely = joined["bolus_when"].isna().to_numpy()
    gap_minutes = ((nearest["when"] - nearest["bolus_when"]).abs().dt.total_seconds() / 60).to_numpy()[lonely]
    kind = pd.to_numeric(boluses["bolus_type_raw"], errors="coerce").fillna(0).astype(int)
    source = pd.to_numeric(boluses["bolus_source_raw"], errors="coerce")
    algorithm = source.eq(ALGORITHM_SOURCE)
    user = ~algorithm
    carb_flag = (kind & BOLUS_CARB_BIT) > 0
    pairs = pd.merge_asof(boluses[["when", "pump"]].assign(index=boluses.index), entries[["when", "pump"]].rename(columns={"when": "entry_when"}).assign(match=lambda f: f["entry_when"]), left_on="when", right_on="entry_when", by="pump", direction="nearest", tolerance=MEAL_BOLUS_WINDOW)
    has_entry = pd.Series(pairs["entry_when"].notna().to_numpy(), index=pairs["index"].to_numpy()).reindex(boluses.index)
    flagged_without = int((carb_flag & ~has_entry).sum())
    user_without_carb = int((user & ~carb_flag).sum())
    grams = pd.to_numeric(carbs["carb_grams"], errors="coerce").dropna()
    heaping = float((grams % 5 == 0).mean())
    windows = entries.assign(window=(entries["when"] - entries["when"].min()).dt.days // WINDOW_DAYS).groupby("window").agg(entries=("carb_grams", "size"), median_g=("carb_grams", "median"), sd_g=("carb_grams", "std"))
    thin = int((windows["entries"] < WINDOW_DAYS).sum())
    later = int((pd.to_numeric(boluses["requested_later_units"], errors="coerce").fillna(0) > 0).sum())
    override = int(((kind & BOLUS_OVERRIDE_BIT) > 0).sum())
    ordered = entries.reset_index(drop=True)
    entry_issues = [{"step": "carbohydrate entry without a bolus", "reason": "no delivered bolus within 10 minutes", "pump_serial": pump, "start": when, "end": when, "n": 1, "unit": "entries", "field": "carb_grams", "value": f"{grams_logged:.0f} g; nearest bolus {lag:.0f} min away"}
                    for (when, pump, grams_logged), lag in zip(ordered.loc[lonely, ["when", "pump", "carb_grams"]].itertuples(index=False), gap_minutes)]
    entry_issues += [{"step": "carbohydrate bolus without an entry", "reason": "carbohydrate bit set, no entry within 10 minutes", "pump_serial": pump, "start": when, "end": when, "n": 1, "unit": "boluses", "field": "bolus_units", "value": f"{units:.2f} U"}
                     for when, pump, units in zip(boluses.loc[carb_flag & ~has_entry, "when"], boluses.loc[carb_flag & ~has_entry, "pump"], pd.to_numeric(boluses.loc[carb_flag & ~has_entry, "bolus_units"], errors="coerce"))]
    return _frame([
        _row("Carbohydrate entries without a bolus", "carbs, bolus", f"entries with no delivered bolus within {int(MEAL_BOLUS_WINDOW.total_seconds() // 60)} minutes", f"{without_bolus} of {len(entries)}" + (f" (the nearest bolus is {gap_minutes.min():.0f} to {gap_minutes.max():.0f} minutes away)" if without_bolus else ""), "none: an entry is kept only for a delivered bolus", "no concern" if without_bolus == 0 else "concern",
             "every entry has its bolus" if without_bolus == 0 else "these entries were typed in the calculator but their bolus was delivered much later (or the request was changed), so the carbohydrate is not tied to its insulin in time"),
        _row("Boluses flagged as carbohydrate boluses without an entry", "bolus, carbs", "boluses with the carbohydrate bit and no entry nearby", f"{flagged_without} of {int(carb_flag.sum())}", "none", "no concern" if flagged_without == 0 else "concern", "the carbohydrate flag and the entries agree" if flagged_without == 0 else "some carbohydrate boluses have no stored entry"),
        _row("Boluses given by the person without carbohydrate", "bolus", "boluses not from the algorithm and without the carbohydrate bit", f"{user_without_carb} of {int(user.sum())} person-initiated boluses ({_pct(user_without_carb, user.sum())}%)", "none", "not applicable", "manual corrections or unflagged food; food eaten with such a bolus is not in the carbohydrate series"),
        _row("Automatic (Control-IQ) correction boluses", "bolus", "boluses from the algorithm", f"{int(algorithm.sum())} of {len(boluses)} ({_pct(algorithm.sum(), len(boluses))}%); {_pct(pd.to_numeric(boluses.loc[algorithm, 'bolus_units'], errors='coerce').sum(), pd.to_numeric(boluses['bolus_units'], errors='coerce').sum())}% of bolus insulin",
             "none", "not applicable", "automatic corrections are insulin the person did not choose, so bolus insulin is not a setting of the person's"),
        _row("Calculator overrides and extended boluses", "bolus", "boluses with the override bit; boluses with a later part", f"{override} overrides, {later} extended", "none", "not applicable", "an override changes the dose the settings would give; the settings still describe the calculator"),
        _row("Rounding of carbohydrate entries", "carbs", "share of entries that are multiples of 5 g", f"{100 * heaping:.1f}% of {len(grams)} entries (about 20% if grams were spread evenly)", f"at most {100 * HEAPING_MAX_SHARE:.0f}%", "no concern" if heaping <= HEAPING_MAX_SHARE else "concern",
             "entries are not heavily rounded" if heaping <= HEAPING_MAX_SHARE else "entries are heavily rounded to 5 g, so each is a coarse estimate of the meal"),
        _row("Carbohydrate entries per 14-day window", "carbs", "windows with fewer than 14 entries", f"{thin} of {len(windows)} (entries per window {windows['entries'].min()} to {windows['entries'].max()}; median grams per window {windows['median_g'].min():.0f} to {windows['median_g'].max():.0f})",
             "none below one entry a day", "no concern" if thin == 0 else "concern", "carbohydrate logging is steady across windows" if thin == 0 else "some windows hold few entries, so carbohydrate exposure there is poorly measured"),
    ]), _issues(entry_issues)


def insulin_checks(bolus: pd.DataFrame, basal: pd.DataFrame) -> pd.DataFrame:
    """Daily insulin from the delivered basal rate and boluses: implausible days and days with partial basal records."""
    segments = basal.assign(start=pd.to_datetime(basal["start_timestamp"]), end=pd.to_datetime(basal["end_timestamp"]))
    hours = (segments["end"] - segments["start"]).dt.total_seconds() / 3600
    segments = segments.assign(units=segments["basal_units_per_hour"] * hours, hours=hours, date=segments["start"].dt.normalize())
    daily_basal = segments.groupby("date").agg(basal_u=("units", "sum"), basal_hours=("hours", "sum"))
    daily_bolus = bolus.assign(date=pd.to_datetime(bolus["timestamp"]).dt.normalize()).groupby("date")["bolus_units"].sum().rename("bolus_u")
    days = daily_basal.join(daily_bolus, how="left").fillna({"bolus_u": 0.0})
    complete = days[days["basal_hours"] >= 0.9 * 24]
    total = complete["basal_u"] + complete["bolus_u"]
    median = float(total.median())
    odd = int(((total < 0.5 * median) | (total > 2 * median)).sum())
    no_bolus = int((complete["bolus_u"] == 0).sum())
    return _frame([
        _row("Total daily insulin", "basal, bolus", "days with basal records for 90% of the day: median daily units (5th to 95th percentile)", f"{median:.1f} U ({total.quantile(0.05):.1f} to {total.quantile(0.95):.1f}) over {len(complete)} days; {len(days) - len(complete)} days have basal records for under 90% of the day",
             "none", "not applicable", "descriptive"),
        _row("Implausible daily insulin", "basal, bolus", "complete days under half or over twice the median", f"{odd} of {len(complete)}", "at most 2% of days", "no concern" if odd <= 0.02 * len(complete) else "concern",
             "daily totals are steady" if odd <= 0.02 * len(complete) else "some days hold half or double the usual insulin: a missing record or an unusual day"),
        _row("Days without a bolus", "bolus", "complete days with no bolus", no_bolus, "none expected", "no concern" if no_bolus == 0 else "concern", "boluses are recorded every day" if no_bolus == 0 else "days with no bolus are either missing records or days without food"),
    ])


# ---------------------------------------------------------------------------
# Missingness, composition and stationarity
# ---------------------------------------------------------------------------

def missingness(table: pd.DataFrame) -> pd.DataFrame:
    """Which hours lack readings or settings (by clock hour, year and weekday), and whether the hours next to a gap look different."""
    frame = table.assign(no_readings=table["n_valid"].eq(0), partial=table["n_valid"].between(1, SLOTS_PER_HOUR - 1), unknown=table["settings_known_slots"].lt(SLOTS_PER_HOUR),
                         year=pd.to_datetime(table["date"]).dt.year, weekday=pd.to_datetime(table["date"]).dt.dayofweek)
    rows = []
    for column, label in (("no_readings", "no readings"), ("unknown", "settings not known for all 12 moments")):
        overall = float(frame[column].mean())
        by_hour = frame.groupby("hour")[column].mean()
        by_year = frame.groupby("year")[column].mean()
        by_weekday = frame.groupby("weekday")[column].mean()
        peak = by_hour.idxmax()
        ratio = by_hour.max() / overall if overall else float("nan")
        rows.append(_row(f"Hours with {label}", "hour-day table", "share of hours overall; highest clock hour; by year; by weekday (range)", f"{100 * overall:.2f}% overall; peak at {peak:02d}:00 ({100 * by_hour.max():.2f}%); by year {', '.join(f'{y}: {100 * v:.1f}%' for y, v in by_year.items())}; weekdays {100 * by_weekday.min():.2f} to {100 * by_weekday.max():.2f}%",
                         "the highest clock hour is under 3 times the overall share", "no concern" if not ratio > 3 else "concern", "missing hours are spread over the day" if not ratio > 3 else f"missing hours cluster at {peak:02d}:00"))
    usable = table["n_valid"].gt(0)
    # Hours next to an hour without readings (the reading may be absent because of what glucose was doing).
    missing = table["n_valid"].eq(0).to_numpy()
    adjacent = (np.roll(missing, 1) | np.roll(missing, -1)) & ~missing
    adjacent[[0, -1]] = False
    other = usable.to_numpy() & ~adjacent & ~missing
    gap = _pct(table.loc[adjacent, "n_in_range"].sum(), table.loc[adjacent, "n_valid"].sum())
    rest = _pct(table.loc[other, "n_in_range"].sum(), table.loc[other, "n_valid"].sum())
    partial = table["n_valid"].between(1, SLOTS_PER_HOUR - 1)
    full = table["n_valid"].eq(SLOTS_PER_HOUR)
    p_tir = _pct(table.loc[partial, "n_in_range"].sum(), table.loc[partial, "n_valid"].sum())
    f_tir = _pct(table.loc[full, "n_in_range"].sum(), table.loc[full, "n_valid"].sum())
    excl = table["excluded_day"]
    e_tir = _pct(table.loc[excl & usable, "n_in_range"].sum(), table.loc[excl & usable, "n_valid"].sum())
    n_tir = _pct(table.loc[~excl & usable, "n_in_range"].sum(), table.loc[~excl & usable, "n_valid"].sum())
    low_adjacent = _pct(table.loc[adjacent, "n_below_70"].sum(), table.loc[adjacent, "n_valid"].sum())
    low_rest = _pct(table.loc[other, "n_below_70"].sum(), table.loc[other, "n_valid"].sum())
    rows += [
        _row("Hours next to a gap in readings (is missing not at random?)", "hour-day table", "time in range and time below 70 of hours adjacent to an hour without readings, against other hours (%)", f"in range {gap} vs {rest}; below 70 {low_adjacent} vs {low_rest}",
             "time in range within 5 points and time below 70 within 2 points", "no concern" if abs(gap - rest) <= 5 and abs(low_adjacent - low_rest) <= 2 else "concern",
             "glucose next to missing hours looks like glucose elsewhere" if abs(gap - rest) <= 5 and abs(low_adjacent - low_rest) <= 2 else "glucose next to missing readings differs, so what is missing may depend on glucose"),
        _row("Partly covered hours against full hours", "hour-day table", "time in range of hours with 1 to 11 readings against 12 readings (%)", f"{p_tir} vs {f_tir}", "within 5 points", "no concern" if abs(p_tir - f_tir) <= 5 else "concern",
             "hours with some readings missing look like full hours" if abs(p_tir - f_tir) <= 5 else "hours with dropped readings differ from full hours"),
        _row("Context-event days against other days", "hour-day table", "time in range on days listed in context_events.csv against other days (%)", f"{e_tir} vs {n_tir} ({int(excl.sum() // HOURS_PER_DAY)} days left out)", "none: the days are left out on purpose", "not applicable",
             "left-out days are the days the person marked as unusual; their glucose is not used"),
    ]
    return _frame(rows)


def _cell_arrays(table: pd.DataFrame, mask: pd.Series, cells: pd.Series, n_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per day and cell: readings in range and valid readings (the arrays a day-level bootstrap resamples)."""
    days = pd.to_datetime(table["date"])[mask]
    codes = days.dt.normalize().factorize()[0]
    shape = (codes.max() + 1 if len(codes) else 0, n_cells)
    in_range = np.zeros(shape)
    valid = np.zeros(shape)
    np.add.at(in_range, (codes, cells[mask].to_numpy()), table.loc[mask, "n_in_range"].to_numpy())
    np.add.at(valid, (codes, cells[mask].to_numpy()), table.loc[mask, "n_valid"].to_numpy())
    return in_range, valid, codes


def composition_test(table: pd.DataFrame, *, draws: int = DRAWS, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Is the time in range of hours with unknown settings different from known-settings hours once the clock-hour and calendar-quarter mix is the same?

    Known hours are standardised to the unknown hours' mix: the known time in range of each cell (clock hour x calendar quarter) is
    weighted by the share of the unknown hours' readings in that cell. Cells with unknown hours and no known hours cannot be
    standardised and are reported. The 90% intervals resample days.

    Returns:
        The summary table and one check row.
    """
    known = table["settings_known_slots"].eq(SLOTS_PER_HOUR) & table["n_valid"].gt(0)
    unknown = table["settings_known_slots"].lt(SLOTS_PER_HOUR) & table["n_valid"].gt(0)
    if unknown.sum() == 0:
        return pd.DataFrame(), _frame([_row("Composition of unknown-settings hours", "hour-day table", "standardised time in range", "", "", "not applicable", "no hour with readings has unknown settings")])
    quarter = pd.to_datetime(table["date"]).dt.to_period("Q").astype(str)
    cell = (quarter + "_" + table["hour"].astype(str)).astype("category").cat.codes
    n_cells = int(cell.max()) + 1
    k_in, k_valid, _ = _cell_arrays(table, known, cell, n_cells)
    u_in, u_valid, _ = _cell_arrays(table, unknown, cell, n_cells)
    weights = u_valid.sum(axis=0)
    has_known = k_valid.sum(axis=0) > 0
    usable_weight = weights * has_known
    standard = usable_weight / usable_weight.sum()
    rng = np.random.default_rng(seed)

    def standardised(in_range: np.ndarray, valid: np.ndarray) -> float:
        rate = in_range.sum(axis=0) / np.maximum(valid.sum(axis=0), 1)
        return float(100 * (standard * rate).sum())

    crude_known = 100 * k_in.sum() / k_valid.sum()
    crude_unknown = 100 * u_in.sum() / u_valid.sum()
    std_known = standardised(k_in, k_valid)
    boot_std, boot_unknown, boot_crude = [], [], []
    for _ in range(draws):
        pick = rng.integers(0, len(k_in), len(k_in))
        boot_std.append(standardised(k_in[pick], k_valid[pick]))
        boot_crude.append(100 * k_in[pick].sum() / k_valid[pick].sum())
        pick = rng.integers(0, len(u_in), len(u_in))
        boot_unknown.append(100 * u_in[pick].sum() / u_valid[pick].sum())
    boot_std, boot_unknown, boot_crude = map(np.array, (boot_std, boot_unknown, boot_crude))
    gap_crude = boot_unknown - boot_crude
    gap_std = boot_unknown - boot_std
    interval = lambda x: f"{np.percentile(x, 5):+.1f} to {np.percentile(x, 95):+.1f}"  # noqa: E731
    outside = float(100 * weights[~has_known].sum() / weights.sum())
    summary = pd.DataFrame([
        {"group": "known settings, as observed", "time_in_range_pct": round(crude_known, 2), "interval_90": f"{np.percentile(boot_crude, 5):.1f} to {np.percentile(boot_crude, 95):.1f}", "readings": int(k_valid.sum())},
        {"group": "known settings, standardised to the unknown hours' clock-hour and quarter mix", "time_in_range_pct": round(std_known, 2), "interval_90": f"{np.percentile(boot_std, 5):.1f} to {np.percentile(boot_std, 95):.1f}", "readings": int(k_valid.sum())},
        {"group": "unknown settings", "time_in_range_pct": round(crude_unknown, 2), "interval_90": f"{np.percentile(boot_unknown, 5):.1f} to {np.percentile(boot_unknown, 95):.1f}", "readings": int(u_valid.sum())},
    ])
    gap_c, gap_s = crude_unknown - crude_known, crude_unknown - std_known
    explained = 100 * (1 - gap_s / gap_c) if abs(gap_c) > 1e-9 else float("nan")
    excludes_zero = np.percentile(gap_std, 5) > 0 or np.percentile(gap_std, 95) < 0
    check = _frame([_row("Composition of hours with unknown settings", "hour-day table", "time in range: unknown minus known, as observed and after standardising the known hours to the unknown hours' clock-hour x quarter mix (points, 90% interval over days)",
                         f"as observed {gap_c:+.1f} ({interval(gap_crude)}); standardised {gap_s:+.1f} ({interval(gap_std)}); composition explains {explained:.0f}% of the observed gap; {outside:.1f}% of unknown readings are in cells with no known hours (left out of the standardised comparison)",
                         "the standardised gap's 90% interval includes 0", "concern" if excludes_zero else "no concern",
                         "unknown hours differ from known hours beyond their clock-hour and quarter mix: dropping them selects a different kind of hour" if excludes_zero else f"after matching the clock-hour and quarter mix the gap is {gap_s:+.1f} points and its interval includes 0: the record cannot separate the unknown hours from known ones beyond when they fall")])
    return summary, check


def daily_series(table: pd.DataFrame) -> pd.DataFrame:
    """Per day: readings, time in range, time below 70 and mean glucose, for days holding at least 70% of the possible readings."""
    daily = table.assign(date=pd.to_datetime(table["date"])).groupby("date")[GLUCOSE_COLUMNS].sum()
    daily = daily[daily["n_valid"] >= ACTIVE_TIME_MIN * READINGS_PER_DAY].copy()
    daily["tir_pct"] = 100 * daily["n_in_range"] / daily["n_valid"]
    daily["below70_pct"] = 100 * daily["n_below_70"] / daily["n_valid"]
    daily["mean_mgdl"] = daily["glucose_sum"] / daily["n_valid"]
    return daily


def fortnight_series(table: pd.DataFrame, *, days: int = WINDOW_DAYS) -> pd.DataFrame:
    """Pooled time in range of consecutive ``days``-day windows counted back from the last day (so the last window is whole)."""
    frame = table.assign(date=pd.to_datetime(table["date"]))
    last = frame["date"].max()
    window = ((last - frame["date"]).dt.days // days)
    grouped = frame.groupby(window)
    out = pd.DataFrame({"start": grouped["date"].min(), "end": grouped["date"].max(), "readings": grouped["n_valid"].sum(), "in_range": grouped["n_in_range"].sum(), "below70": grouped["n_below_70"].sum(), "days": grouped["date"].nunique()})
    out = out[out["days"].eq(days)].copy()
    out["tir_pct"] = 100 * out["in_range"] / out["readings"]
    out["below70_pct"] = 100 * out["below70"] / out["readings"]
    return out.sort_values("start").reset_index(drop=True)


def monthly_table(table: pd.DataFrame) -> pd.DataFrame:
    """Per calendar month: hours, active time, time in range, below 70 and 54, the share of hours with settings known, and the hours with unknown settings."""
    frame = table.assign(month=pd.to_datetime(table["date"]).dt.to_period("M").astype(str))
    grouped = frame.groupby("month")
    out = pd.DataFrame({
        "hours": grouped.size(), "active_time_pct": 100 * grouped["n_valid"].sum() / (grouped.size() * SLOTS_PER_HOUR),
        "time_in_range_pct": 100 * grouped["n_in_range"].sum() / grouped["n_valid"].sum(), "time_below_70_pct": 100 * grouped["n_below_70"].sum() / grouped["n_valid"].sum(),
        "time_below_54_pct": 100 * grouped["n_below_54"].sum() / grouped["n_valid"].sum(), "mean_glucose_mgdl": grouped["glucose_sum"].sum() / grouped["n_valid"].sum(),
        "hours_settings_known_pct": 100 * grouped["settings_known_slots"].apply(lambda s: (s == SLOTS_PER_HOUR).mean()),
        "hours_with_readings_and_unknown_settings": grouped.apply(lambda g: int(((g["settings_known_slots"] < SLOTS_PER_HOUR) & (g["n_valid"] > 0)).sum())),
    }).round(2)
    return out.reset_index()


def stationarity(table: pd.DataFrame) -> pd.DataFrame:
    """Whether the level of glucose is steady: monthly trend, day-to-day autocorrelation, and where the latest fortnight sits among all fortnights."""
    daily = daily_series(table)
    months = monthly_table(table)
    tau, p = kendalltau(np.arange(len(months)), months["time_in_range_pct"])
    slope = float(np.median([(b - a) / (j - i) for i, a in enumerate(months["time_in_range_pct"]) for j, b in enumerate(months["time_in_range_pct"]) if j > i])) if len(months) > 1 else float("nan")
    full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    acf = {lag: float(full["tir_pct"].corr(full["tir_pct"].shift(lag))) for lag in (1, 7, WINDOW_DAYS)}
    fortnights = fortnight_series(table)
    latest = fortnights.iloc[-1]
    earlier = fortnights.iloc[:-1]
    rank = 100 * float((earlier["tir_pct"] < latest["tir_pct"]).mean())
    low, high = np.percentile(earlier["tir_pct"], [5, 95])
    previous = fortnights.iloc[-2] if len(fortnights) > 1 else latest
    return _frame([
        _row("Trend in monthly time in range", "hour-day table", "Kendall tau of monthly time in range against month; median pairwise slope (points per month)", f"tau {tau:.2f} (p {p:.2f}); slope {slope:+.2f} points/month over {len(months)} months",
             "|tau| below 0.3", "no concern" if abs(tau) < 0.3 else "concern", "no steady drift in the monthly level" if abs(tau) < 0.3 else "the monthly level drifts in one direction"),
        _row("Day-to-day dependence of time in range", "hour-day table", "autocorrelation of daily time in range at lags 1, 7 and 14 days", ", ".join(f"lag {k}: {v:.2f}" for k, v in acf.items()), "lag 14 below 0.2: day effects can be taken as independent two weeks apart",
             "no concern" if acf[WINDOW_DAYS] < 0.2 else "concern", "days two weeks apart are nearly independent" if acf[WINDOW_DAYS] < 0.2 else "days two weeks apart still resemble each other: independent day effects overstate the information in the record"),
        _row("Latest fortnight against earlier fortnights", "hour-day table", "time in range of the last 14 days; rank among earlier fortnights", f"{latest['tir_pct']:.1f}% (previous fortnight {previous['tir_pct']:.1f}%); higher than {rank:.0f}% of {len(earlier)} earlier fortnights (their 5th to 95th percentile {low:.1f} to {high:.1f}%)",
             "inside the earlier 5th to 95th percentile", "no concern" if low <= latest["tir_pct"] <= high else "concern",
             "the latest fortnight looks like earlier ones" if low <= latest["tir_pct"] <= high else "the latest fortnight is outside the range of earlier fortnights: a recent regime that the earlier record may not describe"),
        _row("Spread of 14-day time in range", "hour-day table", "standard deviation and range of fortnightly time in range (points)", f"{earlier['tir_pct'].std():.1f} (range {earlier['tir_pct'].min():.1f} to {earlier['tir_pct'].max():.1f}) over {len(earlier)} fortnights", "none", "not applicable", "the noise a 14-day aggregate has even without a change of settings"),
    ])


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

KNOWN_AT_ORIGIN = "known at the forecast origin"
CONCURRENT = "concurrent with the hour"
AFTER_THE_FACT = "after the fact"
SETTING_COLUMNS = ("icr", "cf", "basal", "target")
# column of the hour-day table, built from, when it is known, whether a forecast can use it, why
LEAKAGE = (
    ("hour, date, dst_day", "the calendar", KNOWN_AT_ORIGIN, "A date and its clock-change flag are fixed in advance."),
    ("icr, cf, basal, target", "the profile the pump holds at each moment", KNOWN_AT_ORIGIN, "A forecast is conditional on the settings in force over the forecast hours, which are decisions made at the origin; an edit after the origin makes the forecast describe settings that were not run."),
    ("carbs_prev3h_g", "carbohydrate entered in the hour and the three before it", CONCURRENT, "Meals in the forecast hours are eaten after the origin."),
    ("exercise_share", "workouts and exercise mode, from the start to four hours after the end", CONCURRENT, "Exercise in the forecast hours has not happened at the origin, and the four hours after it are built from the end of the workout."),
    ("sleep_mode_share", "the pump's sleep mode, switched by the person", CONCURRENT, "The mode is switched at bedtime; the share over a forecast hour is not known at the origin."),
    ("site_age_days", "days since the last cannula fill", AFTER_THE_FACT, "The age at a future hour depends on when the next cannula is filled, which is not known at the origin."),
    ("excluded_day", "days the person listed in context_events.csv", AFTER_THE_FACT, "A day is listed after it was seen to be unusual; dropping such days from the evaluation chooses days after the outcome is known."),
    ("apple_asleep_share", "sleep periods recorded by the phone", AFTER_THE_FACT, "A sleep period is recorded after it ends."),
    ("bolus_u, carbs_g", "delivered bolus and entered carbohydrate in the hour", CONCURRENT, "Insulin and food in the hour are consequences of the day, not inputs known at the origin; they lie on the path from settings to glucose."),
)
MODEL_COVARIATES = {"icr": "log_icr", "cf": "log_cf", "basal": "log_basal", "target": "target_10", "carbs_prev3h_g": "log_carbs_3h", "exercise_share": "exercise_share", "sleep_mode_share": "sleep_mode_share", "site_age_days": "site_age_capped"}


def leakage_table(table: pd.DataFrame, used_by_model: Iterable[str]) -> pd.DataFrame:
    """For each hour-day column: when it is known, whether the outcomes model uses it, and how often a setting differs from two weeks earlier.

    Args:
        table: the hour-day table.
        used_by_model: the covariate names the outcomes model reads (its modelling-scale names).

    Returns:
        One row per column group with ``known_when``, ``used_by_outcomes_model``, a ``status`` (concern when a column that is not known at the origin is used by the model or is used to drop rows) and a plain reason.
    """
    used = set(used_by_model)
    rows = []
    shift = 14 * HOURS_PER_DAY
    for names, built, when, why in LEAKAGE:
        columns = [name.strip() for name in names.split(",")]
        model_names = [MODEL_COVARIATES[c] for c in columns if c in MODEL_COVARIATES]
        in_model = [m for m in model_names if m in used]
        filter_only = "excluded_day" in columns
        statistic = ""
        if columns[0] in SETTING_COLUMNS:
            parts = []
            for name in SETTING_COLUMNS:
                values = table[name].to_numpy()
                same = np.isfinite(values[shift:]) & np.isfinite(values[:-shift])
                differs = (np.abs(values[shift:] - values[:-shift]) > 1e-9) & same
                parts.append(f"{name}: {100 * differs.sum() / max(same.sum(), 1):.1f}%")
            statistic = "hours whose value differs from the same hour 14 days earlier: " + ", ".join(parts)
        if when == KNOWN_AT_ORIGIN:
            status = "no concern"
            reason = why
        elif in_model or filter_only:
            status = "concern"
            reason = why + (" Used by the outcomes model." if in_model else " Used to drop days from the analysis.")
        else:
            status = "not applicable"
            reason = why + " Not used by the outcomes model; it would be a leak if added."
        rows.append({"column": names, "built_from": built, "known_when": when, "used_by_outcomes_model": "yes" if in_model else "no (row filter)" if filter_only else "no", "statistic": statistic, "status": status, "reason": reason})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """Run the audit on the real record: ``python -m bayesian_t1dm.data_audit``."""
    from .data_audit_report import run

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", default=None, help="Raw data root (default: the Project Vault data/raw)")
    parser.add_argument("--output", default=None, help="Output folder (default: runtime/output/data_audit)")
    args = parser.parse_args(argv)
    return run(args.raw, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
