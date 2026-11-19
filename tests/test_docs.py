from __future__ import annotations

from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "technical.md"

REQUIRED_SECTIONS = (
    "## Goals",
    "## Design overview",
    "## Breakfast I:C proposals (goal B)",
    "## Lunch and dinner (goal E)",
    "## Food the pump did not log (goal D1)",
    "## Refresh, change record and evaluation (goal C)",
    "## Basal from the controller pattern (goal F)",
    "## Problem episodes and responses (goal H)",
    "## The front page and the runtime tree",
    "## Planned: probabilities for every outcome, and the Bayesian basal model",
    "## Outcome metrics (goal A)",
    "## Hour-day table and settings in force",
    "## Replay of past setting changes (exploratory)",
    "## Hour-day outcomes model",
)


def test_every_analysis_is_documented_in_the_methods_document():
    """A method that is built must be described: an edit that silently failed to insert a section would otherwise go unnoticed."""
    headings = {line.strip() for line in DOC.read_text(encoding="utf-8").splitlines() if line.startswith("## ")}
    missing = [section for section in REQUIRED_SECTIONS if section not in headings]
    assert not missing, f"technical.md lacks: {missing}"


def test_goal_table_lists_every_lettered_goal():
    text = DOC.read_text(encoding="utf-8")
    for letter in "ABCDEFGH":
        assert f"| {letter} |" in text
