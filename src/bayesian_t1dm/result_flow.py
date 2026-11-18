"""Private denominator checks for the dated dashboard run."""

from pathlib import Path

import numpy as np
import pandas as pd

from .context_rules import valid_cgm
from .dataset import merged_dir


def sleep_flow(sleep: pd.DataFrame, grid: pd.DataFrame) -> dict[str, int]:
    """Check each displayed duration against the most recent completed night within 36 hours."""
    nights = sleep.dropna(subset=["sleep_end"]).drop_duplicates("sleep_end", keep="last").sort_values("sleep_end")
    stamps = pd.DatetimeIndex(grid["timestamp"])
    ends = pd.DatetimeIndex(nights["sleep_end"])
    positions = ends.searchsorted(stamps, side="right") - 1
    expected = np.full(len(stamps), np.nan)
    if len(ends):
        age = stamps - ends[positions.clip(min=0)]
        valid = (positions >= 0) & (age <= pd.Timedelta(hours=36))
        expected[valid] = nights.total_sleep_hours.to_numpy()[positions[valid]]
    known = np.isfinite(expected)
    if not np.array_equal(grid["prior_night_sleep_known"].to_numpy(), known):
        raise ValueError("Completed sleep records and the grid's known-sleep flag disagree")
    if not np.allclose(grid.loc[known, "prior_night_total_sleep_hours"], expected[known]):
        raise ValueError("Grid duration differs from the most recent completed sleep record")
    zero = sleep.in_bed_hours.eq(0)
    longer = sleep.total_sleep_hours.gt(sleep.in_bed_hours)
    return {"sleep records": len(sleep), "zero in-bed hours": int(zero.sum()), "sleep exceeds in-bed": int(longer.sum()),
            "both flags": int((zero & longer).sum()), "sleep exceeds positive in-bed": int((longer & sleep.in_bed_hours.gt(0)).sum()),
            "known sleep grid rows": int(known.sum()), "unknown sleep grid rows": int((~known).sum())}


def write_result_flow(run: Path, raw_root: Path) -> None:
    """Read the artifacts that will be served by this dashboard and reconcile their denominators."""
    root = merged_dir(raw_root)
    sleep = pd.read_parquet(root / "tables/sleep.parquet")
    grid = pd.read_parquet(root / "prepared_model_data_5min.parquet")
    rows = []

    def add(goal, source, dates, unit, count, command, artifact):
        dates = pd.to_datetime(pd.Series(dates)).dropna()
        rows.append(dict(goal=goal, source=source, first=dates.min(), last=dates.max(), unit=unit,
                         count=count, command=command, artifact=artifact))

    if len(sleep):
        for name, count in sleep_flow(sleep, grid).items():
            dates = grid.timestamp if "grid rows" in name else sleep.sleep_end
            add("Sleep", "Apple Health completed sleep records", dates, name, count, "build-dataset", "prepared_model_data_5min.parquet")
    path = "A_outcomes/supporting/outcomes_baseline.csv"
    cgm = pd.read_parquet(root / "tables/tandem_cgm.parquet", columns=["timestamp", "glucose"])
    end = valid_cgm(cgm)["timestamp"].max().normalize()
    for row in pd.read_csv(run / path).itertuples():
        add("A", "Tandem sensor readings, qualifying days: " + row.window,
            [end - pd.Timedelta(days=row.days_in_window), end - pd.Timedelta(days=1)], "readings", row.readings, "report-outcomes", path)
    path = "B_icr/supporting/effective_icr_evidence.csv"
    evidence = pd.read_csv(run / path)
    for row in evidence[evidence.analysis.eq("primary")].itertuples():
        add("B/E", f"Accepted logged meals, {row.segment}, {row.programmed_icr:g} g/U", [row.period_start,row.period_end],
            "meals with an observation", row.n_meals, "propose-breakfast-icr", path)
    path = "F_basal/supporting/controller_daily.csv"
    if (run / path).exists():
        for part, group in pd.read_csv(run / path).groupby("part"):
            for unit, value in [("eligible segment-days", len(group)), ("eligible hours", group.clean_hours.sum())]:
                add("F", "Pump rates and context exclusions, " + part, group.date, unit, value, "propose-breakfast-icr", path)
    path = "H_episodes/episodes.csv"
    for kind, group in pd.read_csv(run / path).groupby("kind"):
        add("H", "Tandem CGM, episode rules", list(group.start) + list(group.end), kind + " episodes", len(group), "propose-breakfast-icr", path)
    path = "C_changes/setting_proposals.csv"
    if (run / path).exists():
        for status, group in pd.read_csv(run / path).groupby("status"):
            add("C", "Change record", group.entered_on.fillna(group.proposed_on), status + " changes", len(group), "refresh", path)
    pd.DataFrame(rows).to_csv(run / "00_General/source_to_result.csv", index=False)
