"""The tree a person opens: ``runtime/current/index.html`` and goal-lettered folders beside it.

Commands write to their own working folders under ``runtime/output/``; a refresh run gathers what matters into one
folder organised by the goal it answers (the letters in ``docs/technical.md``), each with a ``supporting/`` folder for
diagnostics, and copies the latest complete run to ``runtime/current/``. Runs live under ``runtime/runs/``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .dataset import merged_dir

# destination in the run folder -> source under runtime/output (or the merged dataset folder for "merged/...")
RUN_LAYOUT: dict[str, str] = {
    "index.html": "proposals/dashboard.html",
    "A_outcomes/outcomes_report.html": "outcomes/outcomes_report.html",
    "A_outcomes/supporting/outcomes_baseline.csv": "outcomes/outcomes_baseline.csv",
    "A_outcomes/supporting/outcomes_day_parts.csv": "outcomes/outcomes_day_parts.csv",
    "A_outcomes/supporting/outcomes_daily.csv": "outcomes/outcomes_daily.csv",
    "B_icr/icr_proposal.md": "proposals/breakfast_proposal.md",
    "B_icr/segment_summary.csv": "proposals/segment_summary.csv",
    "B_icr/supporting/icr_support.csv": "proposals/icr_support.csv",
    "B_icr/supporting/effective_icr_evidence.csv": "proposals/effective_icr_evidence.csv",
    "B_icr/supporting/proposal_row.csv": "proposals/breakfast_proposal.csv",
    "C_changes/change_log.md": "proposals/change_log.md",
    "F_basal/supporting/controller_flow.csv": "proposals/controller_flow.csv",
    "F_basal/supporting/basal_signal_strict.csv": "proposals/basal_signal_strict.csv",
    "F_basal/basal_signal.csv": "proposals/basal_signal.csv",
    "F_basal/supporting/controller_daily.csv": "proposals/controller_daily.csv",
    "H_episodes/episodes.csv": "proposals/episodes.csv",
    "00_General/validation_report.md": "merged/validation_report.md",
}
REQUIRED = ("index.html", "A_outcomes/outcomes_report.html", "B_icr/icr_proposal.md", "B_icr/segment_summary.csv")


def assemble_run(output_root: Path, raw_root: str | Path, run_dir: Path, *, record_path: Path | None = None) -> list[str]:
    """Copy each source into the run folder at its place in the goal layout; returns the destinations written.

    A source that does not exist is skipped only when it is optional (not in ``REQUIRED``); a missing required one raises.
    """
    written = []
    for destination, source in RUN_LAYOUT.items():
        path = merged_dir(raw_root) / source.removeprefix("merged/") if source.startswith("merged/") else output_root / source
        if not path.exists():
            if destination in REQUIRED:
                raise FileNotFoundError(f"The run needs {source}, which was not written.")
            continue
        target = run_dir / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        written.append(destination)
    if record_path is not None and record_path.exists():
        target = run_dir / "C_changes" / "setting_proposals.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record_path, target)
        written.append("C_changes/setting_proposals.csv")
    return written


def activate_current(run_dir: Path, current_dir: Path) -> None:
    """Make ``current_dir`` a copy of the run, replacing the previous one only once the copy has succeeded."""
    staging = current_dir.with_name(current_dir.name + ".new")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(run_dir, staging)
    if current_dir.exists():
        shutil.rmtree(current_dir)
    staging.rename(current_dir)
