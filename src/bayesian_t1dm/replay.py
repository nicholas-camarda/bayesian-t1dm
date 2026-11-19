"""Replay of past setting changes against placebo dates (exploratory, predictive).

For every change in the settings in force since 2024-01, a predictor sees only the days before the change and
predicts the time in range, time below 70 and below 54 and the mean glucose over the affected hours for the next 14
days. The prediction is scored against what happened, and against what the same predictor misses on placebo dates: days
with no change within 14 days, from the same season, whose recent time in range was close to the change's.

What is estimated. For each outcome, ``delta`` is the mean over change events of (observed minus predicted at the
change) minus the mean over each event's five matched placebo dates of (observed minus predicted there). The baseline
predictors know nothing about the settings, so ``delta`` is how far the 14 days after a change sit from the
pre-change level, over and above how far a no-change date sits from its own pre-level (regression to the mean and
season are in the placebo, not in ``delta``). It is an association in one person's record; intervals are 90% from a
bootstrap over clusters of changes that sit within 14 days of each other (with their placebos).

A change is a clock hour whose setting (I:C, correction factor, basal or target) differs from the same 5-minute moment of
the previous day (``hour_day.settings_changed``), so edits to a profile that was not running are not changes and a
profile switch is. Changes on consecutive days (an edit made mid-day reaches the hours already passed the next day) form
one event. The affected hours are the changed clock hours and the 4 after them.

A predictor is any function ``(history, n_days, rng) -> {outcome: draws}`` of the days before the event, so the
outcomes model plugs into the same replay.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .hour_day import GLUCOSE_COLUMNS, HOURS_PER_DAY, SLOTS_PER_HOUR, build_hour_day_table, hour_day_flow

LOGGER = logging.getLogger(__name__)

AFTER_DAYS = 14
BEFORE_DAYS = 28
EWMA_DAYS = 90
EWMA_HALF_LIFE_DAYS = 28
EXTRA_HOURS = 4
MIN_COVERAGE = 0.70
MIN_AFTER_DAYS = 7
MIN_BEFORE_DAYS = 14
PLACEBOS_PER_EVENT = 5
SEASON_DAYS = 90
CLEARANCE_DAYS = 14
DRAWS = 4000
SEED = 20261119
INTERVAL_PERCENT = (5.0, 95.0)
LOW_LIMIT_PERCENT = 4.0
VERY_LOW_LIMIT_PERCENT = 1.0
OUTCOMES = ("tir", "below70", "below54", "mean_glucose")
OUTCOME_LABELS = {"tir": "time in range 70-180 (%)", "below70": "time below 70 (%)", "below54": "time below 54 (%)", "mean_glucose": "mean glucose (mg/dL)"}
SETTING_LABELS = {"icr": "I:C", "cf": "CF", "basal": "basal", "target": "target"}
# A larger value of these settings gives less insulin (a larger I:C or CF, a higher target); a larger basal gives more.
INSULIN_SIGN = {"icr": -1, "cf": -1, "basal": 1, "target": -1}
DIRECTION_SIGN = {"more insulin": 1, "less insulin": -1}


@dataclass(frozen=True)
class DayGrid:
    """The hour-day table as arrays: ``counts[name]`` is (days, 24); one row per consecutive calendar day."""

    dates: pd.DatetimeIndex
    counts: dict[str, np.ndarray]
    excluded: np.ndarray
    changed: np.ndarray
    delta: dict[str, np.ndarray]

    @property
    def n_days(self) -> int:
        return len(self.dates)


@dataclass(frozen=True)
class Event:
    """A change in the settings in force: ``first`` to ``last`` are day positions (consecutive days holding changed hours)."""

    event_id: int
    first: int
    last: int
    changed_hours: np.ndarray
    affected_hours: np.ndarray
    direction: str
    settings: str

    @property
    def span(self) -> int:
        return self.last - self.first + 1


@dataclass(frozen=True)
class History:
    """The days before an event that qualify (enough readings in the affected hours), with their age in days and counts."""

    age_days: np.ndarray
    counts: dict[str, np.ndarray]


Predictor = Callable[[History, int, np.random.Generator], dict[str, np.ndarray]]


def day_grid(table: pd.DataFrame) -> DayGrid:
    """Arrange the hour-day table (consecutive hours from midnight) as day x hour arrays."""
    if len(table) % HOURS_PER_DAY or not (pd.to_datetime(table["hour_start"]).diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("The hour-day table must hold every clock hour of whole consecutive days")
    shape = (len(table) // HOURS_PER_DAY, HOURS_PER_DAY)
    excluded = table["excluded_day"].to_numpy(dtype=bool).reshape(shape)[:, 0]
    return DayGrid(
        dates=pd.DatetimeIndex(table["date"].to_numpy()[::HOURS_PER_DAY]),
        counts={name: table[name].to_numpy(dtype=float).reshape(shape) for name in GLUCOSE_COLUMNS},
        excluded=excluded,
        changed=table["settings_changed"].to_numpy(dtype=bool).reshape(shape),
        delta={name: table["d_" + name].to_numpy(dtype=float).reshape(shape) for name in INSULIN_SIGN},
    )


def _affected(changed_hours: np.ndarray) -> np.ndarray:
    affected = changed_hours.copy()
    for shift in range(1, EXTRA_HOURS + 1):
        affected |= np.roll(changed_hours, shift)
    return affected


def find_events(grid: DayGrid) -> list[Event]:
    """Group the days that hold a changed hour into events (consecutive days are one event)."""
    flagged = np.flatnonzero(grid.changed.any(axis=1))
    events: list[Event] = []
    start = 0
    for index in range(1, len(flagged) + 1):
        if index < len(flagged) and flagged[index] == flagged[index - 1] + 1:
            continue
        days = flagged[start:index]
        changed_hours = grid.changed[days].any(axis=0)
        cells = grid.changed[days]
        signs, names = [], []
        for name, sign in INSULIN_SIGN.items():
            net = np.nansum(np.where(cells, grid.delta[name][days], 0.0))
            if abs(net) > 1e-9:
                signs.append(sign * int(np.sign(net)))
                names.append(SETTING_LABELS[name])
        direction = "more insulin" if signs and all(sign > 0 for sign in signs) else "less insulin" if signs and all(sign < 0 for sign in signs) else "mixed"
        events.append(Event(len(events), int(days[0]), int(days[-1]), changed_hours, _affected(changed_hours), direction, ", ".join(names) or "start time"))
        start = index
    return events


def cluster_events(events: list[Event]) -> tuple[dict[int, int], set[int]]:
    """Cluster id per event (events within 14 days of each other chain together) and the isolated events (no other change within 14 days)."""
    cluster: dict[int, int] = {}
    isolated: set[int] = set()
    current = -1
    for index, event in enumerate(events):
        previous_gap = event.first - events[index - 1].last if index else np.inf
        next_gap = events[index + 1].first - event.last if index + 1 < len(events) else np.inf
        if previous_gap > CLEARANCE_DAYS:
            current += 1
        cluster[event.event_id] = current
        if previous_gap > CLEARANCE_DAYS and next_gap > CLEARANCE_DAYS:
            isolated.add(event.event_id)
    return cluster, isolated


def _window(grid: DayGrid, start: int, stop: int, hours: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The days in ``[start, stop)`` with enough readings in ``hours`` (at least 70% of the expected, not a context-event day), and their summed counts."""
    positions = np.arange(max(start, 0), min(stop, grid.n_days))
    sums = {name: grid.counts[name][positions][:, hours].sum(axis=1) for name in GLUCOSE_COLUMNS}
    counted = (sums["n_valid"] >= MIN_COVERAGE * SLOTS_PER_HOUR * hours.sum()) & ~grid.excluded[positions]
    return positions[counted], {name: values[counted] for name, values in sums.items()}


def _outcomes(counts: dict[str, np.ndarray], *, axis: int | None = None) -> dict[str, np.ndarray]:
    valid = counts["n_valid"].sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "tir": 100 * counts["n_in_range"].sum(axis=axis) / valid,
            "below70": 100 * counts["n_below_70"].sum(axis=axis) / valid,
            "below54": 100 * counts["n_below_54"].sum(axis=axis) / valid,
            "mean_glucose": counts["glucose_sum"].sum(axis=axis) / valid,
        }


def history_before(grid: DayGrid, first: int, hours: np.ndarray) -> History:
    """The qualifying days of the 90 days before day position ``first``; nothing from ``first`` on is read."""
    positions, counts = _window(grid, first - EWMA_DAYS, first, hours)
    return History(age_days=first - positions, counts=counts)


def _bootstrap_draws(history: History, n_days: int, weights: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, np.ndarray]:
    """Pool ``n_days`` days drawn with replacement from the history, ``draws`` times."""
    picks = rng.choice(len(weights), size=(draws, n_days), p=weights / weights.sum())
    pooled = {name: values[picks].sum(axis=1) for name, values in history.counts.items()}
    return _outcomes({name: value[:, None] for name, value in pooled.items()}, axis=1)


def last28_bootstrap(history: History, n_days: int, rng: np.random.Generator, draws: int = DRAWS) -> dict[str, np.ndarray]:
    """The last 28 days, resampled: what the next ``n_days`` days look like if they behave like the last 28."""
    keep = history.age_days <= BEFORE_DAYS
    recent = History(history.age_days[keep], {name: values[keep] for name, values in history.counts.items()})
    return _bootstrap_draws(recent, n_days, np.ones(keep.sum()), rng, draws)


def ewma_bootstrap(history: History, n_days: int, rng: np.random.Generator, draws: int = DRAWS) -> dict[str, np.ndarray]:
    """The last 90 days with weights halving every 28 days."""
    return _bootstrap_draws(history, n_days, np.exp2(-history.age_days / EWMA_HALF_LIFE_DAYS), rng, draws)


PREDICTORS: dict[str, Predictor] = {"last 28 days": last28_bootstrap, "90 days, half-life 28": ewma_bootstrap}
PRIMARY = "last 28 days"


def crps(draws: np.ndarray, observed: float) -> float:
    """Empirical continuous ranked probability score of the draws against one observation (lower is better)."""
    x = np.sort(draws)
    n = len(x)
    spread = 2.0 * np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n**2
    return float(np.mean(np.abs(x - observed)) - 0.5 * spread)


def _score(draws: dict[str, np.ndarray], observed: dict[str, float]) -> dict[str, float]:
    row: dict[str, float] = {}
    for name in OUTCOMES:
        low, high = np.percentile(draws[name], INTERVAL_PERCENT)
        mean = float(draws[name].mean())
        row[f"{name}_predicted"] = mean
        row[f"{name}_lower"] = float(low)
        row[f"{name}_upper"] = float(high)
        row[f"{name}_observed"] = float(observed[name])
        row[f"{name}_residual"] = float(observed[name]) - mean
        row[f"{name}_covered"] = float(low <= observed[name] <= high)
        row[f"{name}_crps"] = crps(draws[name], observed[name])
    over = (draws["below70"] > LOW_LIMIT_PERCENT) | (draws["below54"] > VERY_LOW_LIMIT_PERCENT)
    row["low_limit_probability"] = float(over.mean())
    row["low_limit_exceeded"] = float(observed["below70"] > LOW_LIMIT_PERCENT or observed["below54"] > VERY_LOW_LIMIT_PERCENT)
    return row


def observe(grid: DayGrid, day: int, span: int, hours: np.ndarray) -> tuple[dict[str, float], int] | None:
    """What happened over ``hours`` on the 14 days after the event (``day`` to ``day + span - 1`` are left out), or None when too few days qualify or the window is not complete yet."""
    start = day + span
    if start + AFTER_DAYS > grid.n_days:
        return None
    positions, counts = _window(grid, start, start + AFTER_DAYS, hours)
    if len(positions) < MIN_AFTER_DAYS:
        return None
    return {name: float(value) for name, value in _outcomes(counts).items()}, int(len(positions))


def _before_ok(grid: DayGrid, first: int, hours: np.ndarray) -> bool:
    positions, _ = _window(grid, first - BEFORE_DAYS, first, hours)
    return len(positions) >= MIN_BEFORE_DAYS


def _pre_tir(grid: DayGrid, first: int, hours: np.ndarray) -> float:
    _, counts = _window(grid, first - BEFORE_DAYS, first, hours)
    return float(_outcomes(counts)["tir"])


def score_target(grid: DayGrid, day: int, span: int, hours: np.ndarray, *, key: tuple[int, ...]) -> dict[str, dict[str, float]] | None:
    """Predict from the days before ``day`` and score against the 14 days after it, for each predictor; None when the window cannot be scored."""
    if not _before_ok(grid, day, hours):
        return None
    observed = observe(grid, day, span, hours)
    if observed is None:
        return None
    values, n_days = observed
    history = history_before(grid, day, hours)
    scores: dict[str, dict[str, float]] = {}
    for index, (name, predictor) in enumerate(PREDICTORS.items()):
        rng = np.random.default_rng([SEED, *key, index])
        scores[name] = {**_score(predictor(history, n_days, rng), values), "after_days": float(n_days)}
    return scores


def placebo_candidates(grid: DayGrid, event: Event, flagged: np.ndarray) -> np.ndarray:
    """Day positions where ``event``'s own windows could be placed with no change inside them and the same data sufficiency, from its season."""
    n = grid.n_days
    near = np.concatenate([[0], np.cumsum(flagged)])
    candidates = []
    for day in range(BEFORE_DAYS, n - event.span - AFTER_DAYS + 1):
        low, high = max(day - CLEARANCE_DAYS, 0), min(day + event.span + AFTER_DAYS + CLEARANCE_DAYS, n)
        if near[high] - near[low] > 0:
            continue
        seasonal = abs((grid.dates[day].dayofyear - grid.dates[event.first].dayofyear + 183) % 365 - 183)
        if seasonal > SEASON_DAYS:
            continue
        if _before_ok(grid, day, event.affected_hours) and observe(grid, day, event.span, event.affected_hours) is not None:
            candidates.append(day)
    return np.array(candidates, dtype=int)


def _choose_placebos(grid: DayGrid, event: Event, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Five dates matched on the pre-change time in range, and five drawn at random (a check that matching is doing something)."""
    target = _pre_tir(grid, event.first, event.affected_hours)
    gaps = np.array([abs(_pre_tir(grid, int(day), event.affected_hours) - target) for day in candidates])
    matched = candidates[np.lexsort((candidates, gaps))[:PLACEBOS_PER_EVENT]]
    rng = np.random.default_rng([SEED, event.event_id, 99])
    random = rng.choice(candidates, size=PLACEBOS_PER_EVENT, replace=False)
    return matched, random


def run_replay(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Find every change, score it and its placebos, and tabulate the flow, the scores and the cluster-bootstrap estimates."""
    grid = day_grid(table)
    events = find_events(grid)
    clusters, isolated = cluster_events(events)
    flagged = grid.changed.any(axis=1)
    flow = [("days holding a changed hour", int(flagged.sum())), ("changes (consecutive days merged)", len(events))]
    issues: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    complete, sufficient, placed = [], [], []
    for event in events:
        identity = {"event_id": event.event_id, "change_date": grid.dates[event.first].date(), "change_days": event.span}
        if event.first + event.span + AFTER_DAYS > grid.n_days:
            issues.append({**identity, "step": "after window complete", "reason": "fewer than 14 days after the change in the record", "field": "days after", "value": grid.n_days - event.first - event.span})
            continue
        complete.append(event)
        if not _before_ok(grid, event.first, event.affected_hours):
            positions, _ = _window(grid, event.first - BEFORE_DAYS, event.first, event.affected_hours)
            issues.append({**identity, "step": "enough data", "reason": f"fewer than {MIN_BEFORE_DAYS} of the 28 days before have enough readings in the affected hours", "field": "qualifying days before", "value": len(positions)})
            continue
        if observe(grid, event.first, event.span, event.affected_hours) is None:
            _, counts = _window(grid, event.first + event.span, event.first + event.span + AFTER_DAYS, event.affected_hours)
            issues.append({**identity, "step": "enough data", "reason": f"fewer than {MIN_AFTER_DAYS} of the 14 days after have enough readings in the affected hours", "field": "qualifying days after", "value": len(counts["n_valid"])})
            continue
        sufficient.append(event)
        candidates = placebo_candidates(grid, event, flagged)
        if len(candidates) < PLACEBOS_PER_EVENT:
            issues.append({**identity, "step": "placebo dates", "reason": f"fewer than {PLACEBOS_PER_EVENT} eligible placebo dates", "field": "eligible placebo dates", "value": len(candidates)})
            continue
        placed.append(event)
        matched, random = _choose_placebos(grid, event, candidates)
        event_scores = score_target(grid, event.first, event.span, event.affected_hours, key=(event.event_id, 0))
        base = {
            "event_id": event.event_id, "cluster_id": clusters[event.event_id], "isolated": event.event_id in isolated, "direction": event.direction,
            "settings": event.settings, "affected_hours": int(event.affected_hours.sum()), "changed_hours": int(event.changed_hours.sum()),
        }
        for predictor, score in event_scores.items():
            rows.append({**base, "kind": "change", "date": grid.dates[event.first].date(), "predictor": predictor, "pre_tir": _pre_tir(grid, event.first, event.affected_hours), **score})
        for kind, days in (("matched placebo", matched), ("random placebo", random)):
            for order, day in enumerate(days):
                scores = score_target(grid, int(day), event.span, event.affected_hours, key=(event.event_id, 1 + order + (PLACEBOS_PER_EVENT if kind == "random placebo" else 0)))
                for predictor, score in scores.items():
                    rows.append({**base, "kind": kind, "date": grid.dates[int(day)].date(), "predictor": predictor, "pre_tir": _pre_tir(grid, int(day), event.affected_hours), **score})
    flow += [
        ("with the 14 days after observed in the record", len(complete)),
        ("with enough readings before and after in the affected hours", len(sufficient)),
        ("with five placebo dates (scored)", len(placed)),
        ("of which isolated (no other change within 14 days)", sum(event.event_id in isolated for event in placed)),
    ]
    if any(count == 0 for _, count in flow):
        raise RuntimeError(f"The replay has no events left at a stage: {flow}")
    scores = pd.DataFrame(rows)
    flow_table = pd.DataFrame(flow, columns=["stage", "n"])
    return {"scores": scores, "flow": flow_table, "issues": pd.DataFrame(issues), "summary": summarize(scores), "calibration": calibration(scores)}


def _agreement(frame: pd.DataFrame, outcome: str) -> np.ndarray:
    """1 when the sign of the residual matches the direction the change should push the outcome, 0 when not, NaN for a mixed change."""
    sign = frame["direction"].map(DIRECTION_SIGN)
    expected = -sign if outcome == "mean_glucose" else sign
    return np.where(sign.isna(), np.nan, (np.sign(frame[f"{outcome}_residual"]) == expected).astype(float))


def _contrasts(scores: pd.DataFrame, predictor: str, placebo: str, measure: str) -> pd.DataFrame:
    """One row per event: its value of ``measure`` minus the mean over its placebos of the same measure."""
    frame = scores[scores["predictor"].eq(predictor)].copy()
    if measure in ("agree_glucose", "agree_below70"):
        frame[measure] = _agreement(frame, "mean_glucose" if measure == "agree_glucose" else "below70")
        column = measure
    else:
        column = f"{measure}_residual"
    events = frame[frame["kind"].eq("change")].set_index("event_id")
    controls = frame[frame["kind"].eq(placebo)].groupby("event_id")[column].mean()
    out = pd.DataFrame({"event": events[column], "placebo": controls, "cluster_id": events["cluster_id"], "isolated": events["isolated"], "direction": events["direction"]})
    out["contrast"] = out["event"] - out["placebo"]
    return out.dropna(subset=["contrast"])


def _cluster_bootstrap(frame: pd.DataFrame, column: str = "contrast", *, draws: int = DRAWS) -> tuple[float, float, float]:
    """Mean of ``column`` with a 90% interval from resampling clusters of events (events within 14 days of each other) as units."""
    values = [group[column].to_numpy() for _, group in frame.groupby("cluster_id")]
    rng = np.random.default_rng(SEED + 7)
    means = np.empty(draws)
    for draw in range(draws):
        picked = rng.integers(0, len(values), len(values))
        means[draw] = np.concatenate([values[i] for i in picked]).mean()
    low, high = np.percentile(means, INTERVAL_PERCENT)
    return float(frame[column].mean()), float(low), float(high)


GROUPS = {
    "all changes": lambda frame: frame,
    "isolated changes": lambda frame: frame[frame["isolated"]],
    "changes giving more insulin": lambda frame: frame[frame["direction"].eq("more insulin")],
    "changes giving less insulin": lambda frame: frame[frame["direction"].eq("less insulin")],
}
MEASURES = (*OUTCOMES, "agree_glucose", "agree_below70")


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """The estimates: per group, predictor and measure, the change residual, the matched-placebo residual, their difference (delta) with a 90% interval, and the same against random placebos."""
    rows = []
    for predictor in PREDICTORS:
        for measure in MEASURES:
            matched = _contrasts(scores, predictor, "matched placebo", measure)
            random = _contrasts(scores, predictor, "random placebo", measure)
            for group, select in GROUPS.items():
                subset, subset_random = select(matched), select(random)
                if len(subset) < 3 or subset["cluster_id"].nunique() < 2:
                    continue
                delta, low, high = _cluster_bootstrap(subset)
                random_delta, random_low, random_high = _cluster_bootstrap(subset_random)
                rows.append({
                    "group": group, "predictor": predictor, "measure": measure, "n_changes": len(subset), "n_clusters": int(subset["cluster_id"].nunique()),
                    "change_mean": float(subset["event"].mean()), "matched_placebo_mean": float(subset["placebo"].mean()),
                    "delta": delta, "delta_lower_90": low, "delta_upper_90": high,
                    "delta_vs_random": random_delta, "delta_vs_random_lower_90": random_low, "delta_vs_random_upper_90": random_high,
                })
    return pd.DataFrame(rows)


def calibration(scores: pd.DataFrame) -> pd.DataFrame:
    """How the predictors do on changes and on placebo dates: coverage of the 90% interval (nominal 0.90), mean CRPS and the low-limit Brier score."""
    rows = []
    for (predictor, kind), group in scores.groupby(["predictor", "kind"]):
        row = {"predictor": predictor, "kind": kind, "n": len(group), "low_limit_brier": float(((group["low_limit_probability"] - group["low_limit_exceeded"]) ** 2).mean())}
        for name in OUTCOMES:
            row[f"{name}_coverage_90"] = float(group[f"{name}_covered"].mean())
            row[f"{name}_mean_crps"] = float(group[f"{name}_crps"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    from .analysis_data import load_analysis_data
    from .paths import ProjectPaths
    from .replay_report import write_replay_outputs

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", default=None, help="Raw data root (default: the Project Vault data/raw)")
    parser.add_argument("--output", default=None, help="Output folder (default: runtime/output/replay)")
    args = parser.parse_args(argv)
    paths = ProjectPaths.from_root(Path.cwd())
    raw = args.raw or str(paths.cloud_raw)
    output = Path(args.output or paths.output_replay)
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(output / f"replay_run_{run_id}.log")])
    started = time.time()
    LOGGER.info("replay started run_id=%s raw=%s", run_id, raw)
    analysis = load_analysis_data(raw)
    table = build_hour_day_table(analysis)
    LOGGER.info("hour-day table %s", json.dumps(hour_day_flow(table).__dict__))
    results = run_replay(table)
    LOGGER.info("replay flow\n%s", results["flow"].to_string(index=False))
    write_replay_outputs(results, table, output, run_id=run_id, raw=raw, seconds=time.time() - started)
    LOGGER.info("replay finished in %.1f s", time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
