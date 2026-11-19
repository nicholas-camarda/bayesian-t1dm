"""Read-only paired analysis of historical basal and snapshot support for I:C evidence."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import therapy_research as meals
from .analysis_data import load_analysis_data
from .outcomes import day_part_table
from .proposals import _assess_segment, segment_ends_from_parts, segment_summary
from .pump_profiles import pump_key


def programmed_units_by_slot(analysis):
    """Integrate in-force programmed U/h over exactly the logged delivery exposure in each five-minute bin."""
    basal = analysis.merged.tandem_data.basal
    totals = []
    for pump, group in basal.groupby(basal.pump_serial.map(pump_key)):
        starts = pd.DatetimeIndex(group.start_timestamp)
        ends = pd.DatetimeIndex(group.end_timestamp)
        usable = starts.notna() & ends.notna() & (ends > starts)
        starts, ends = starts[usable], ends[usable]
        if not len(starts):
            continue
        slots = pd.date_range(starts.min().floor("5min"), ends.max().ceil("5min"), freq="5min")
        edits = analysis.history.events.loc[analysis.history.events.pump_serial.eq(pump)]
        offsets = edits.start_minutes.dropna().astype(int).unique()
        days = pd.date_range(slots.min().normalize(), slots.max().normalize())
        edges = slots.union(starts.unique()).union(ends.unique()).union(pd.DatetimeIndex(edits.timestamp.unique()))
        for minute in offsets[offsets % 5 != 0]:
            edges = edges.union(days + pd.Timedelta(minutes=int(minute)))
        edges = edges.sort_values()
        left = edges[:-1]
        multiplicity = starts.sort_values().searchsorted(left, side="right") - ends.sort_values().searchsorted(left, side="right")
        observed = multiplicity > 0
        times = pd.Series(left[observed])
        hours = (edges[1:] - left).total_seconds().to_numpy()[observed] / 3600
        rate = analysis.history.basal_rates(pd.Series(pump, index=times.index), times)
        totals.append(pd.DataFrame({"slot": times.dt.floor("5min"), "units": rate * hours * multiplicity[observed],
                                    "unknown": ~np.isfinite(rate), "pump": pump}))
    if not totals:
        return pd.Series(dtype=float)
    groups = pd.concat(totals).groupby("slot")
    units = groups.units.sum()
    return units.mask(groups.unknown.any() | groups.pump.nunique().gt(1))


def compare(raw: Path, output: Path):
    a = load_analysis_data(raw)
    segments = meals.parse_therapy_segments()
    settings = meals.extract_latent_meal_therapy_settings_from_tandem_raw(raw, segments=segments)
    frame = meals.build_therapy_research_frame(a.merged.preparation.dataset, segments=segments,
                                               meal_proxy_mode="strict", excluded_periods=meals.load_excluded_periods(raw))
    registry = meals.build_meal_event_registry(frame)
    clean = meals.build_first_meal_clean_window_registry(frame, unlogged=a.unlogged, exercise=a.exercise,
                                                         meal_segments=("morning", "afternoon", "evening"))
    metrics = meals._accepted_first_meal_window_metrics(frame, registry, meals.build_latent_meal_tiered_window_registry(clean))
    metrics = meals._attach_logged_food(metrics, a.merged.data.nutrition)
    scale, pairs = meals._logged_food_scale(frame, a.merged.data.nutrition)
    median = meals._first_meal_window_trajectories(frame, metrics)
    exact = meals._first_meal_window_trajectories(frame, metrics, programmed_units=programmed_units_by_slot(a))
    sources = {}
    for _, group in metrics.groupby(meals._pool_label(metrics)):
        ordered = list(group.sort_values("timestamp").itertuples(index=False))
        resolved = meals._resolve_meal_settings(ordered, str(group.segment.astype(str).mode().iloc[0]), settings)
        sources.update({str(row.meal_id): source for row, (_, _, source) in zip(ordered, resolved)})
    paired = set(median) & set(exact)
    sources_frame = metrics[["meal_id", "timestamp", "segment", "profile_segment_start_minutes", "stated_carbs"]].copy()
    ids = sources_frame.meal_id.astype(str)
    sources_frame["settings_source"] = ids.map(sources)
    sources_frame["median_trajectory"] = ids.isin(median)
    sources_frame["paired_trajectory"] = ids.isin(paired)
    output.mkdir(parents=True, exist_ok=True)
    sources_frame.to_csv(output / "meal_support.csv", index=False)
    parts, _ = a.day_parts()
    ends = segment_ends_from_parts(parts)
    current = {minute: item["icr"] for minute, item in a.segment_settings().items()}
    summaries = []
    displays = []
    as_of = a.cgm.timestamp.max().normalize()
    outcomes = [day_part_table(a.cgm, parts, as_of=as_of, days=d) for d in (90, 14)]
    for name, trajectories, keep in [("original", median, set(median)), ("median_paired", median, paired),
                                      ("exact_paired", exact, paired), ("exact_supported_settings", exact,
                                       {key for key in paired if sources[key] != "pump profile"})]:
        evidence = meals._effective_icr_evidence(metrics, {key: trajectories[key] for key in keep},
                                                 settings=settings, bevel_scale=scale, bevel_pairs=pairs)
        evidence.to_csv(output / f"{name}.csv", index=False)
        displays.append(segment_summary(parts, a.segment_settings(), *outcomes, evidence, pump_serial=a.pump,
                                        profile_id=a.profile_id).assign(comparison=name))
        current_rows = evidence.loc[evidence.analysis.eq("primary") & evidence.is_current_setting &
                                    evidence.pump_serial.eq(a.pump) & evidence.profile_id.eq(a.profile_id)]
        for _, row in current_rows.iterrows():
            minute = row.pump_segment_start_minutes
            decision = _assess_segment(evidence, row, minute, a.cgm, ends, a.cgm.timestamp.max().normalize(), current)
            summaries.append({"comparison": name, **decision.to_row(), "evidence_setting": row.programmed_icr})
    pd.concat(displays, ignore_index=True).to_csv(output / "segment_comparison.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "comparison.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.raw, args.output)
