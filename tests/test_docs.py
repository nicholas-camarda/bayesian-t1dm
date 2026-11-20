from __future__ import annotations

import re
from pathlib import Path

from bayesian_t1dm import cli

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "technical.md"
METHODS = ROOT / "docs" / "methods.md"
README = ROOT / "README.md"

REQUIRED_SECTIONS = (
    "## Contents",
    "## Goals",
    "## Design overview",
    "## Breakfast I:C proposals (goal B)",
    "## Lunch and dinner (goal E)",
    "## Food the pump did not log (goal D1)",
    "## Refresh, change record and evaluation (goal C)",
    "## Basal from the controller pattern (goal F)",
    "## Problem episodes and responses (goal H)",
    "## The front page and the runtime tree",
    "## Outcome metrics (goal A)",
    "## Hour-day table and settings in force",
    "## Replay of past setting changes (exploratory)",
    "## Hour-day outcomes model",
    "## Data-quality audit (step A)",
    "## Forecast evaluation harness (step 2)",
    "## Replay as predictor (step 4)",
)


def test_every_analysis_is_documented_in_the_methods_document():
    """A method that is built must be described: an edit that silently failed to insert a section would otherwise go unnoticed."""
    headings = {line.strip() for line in DOC.read_text(encoding="utf-8").splitlines() if line.startswith("## ")}
    missing = [section for section in REQUIRED_SECTIONS if section not in headings]
    assert not missing, f"technical.md lacks: {missing}"


def test_goal_table_lists_every_lettered_goal():
    """The goals and lettered subgoals are defined once, at the top of methods.md."""
    text = METHODS.read_text(encoding="utf-8")
    for letter in "ABCDEFGHIJ":
        assert re.search(rf"^\| {letter} \|.*\b{letter}1\b", text, flags=re.MULTILINE), f"goal {letter} or its first subgoal is missing"
    assert text.index("## Goals") < text.index("## Data processing") < text.index("## Statistical methods")


def _links(path: Path) -> list[str]:
    return re.findall(r"\]\(([^)\s]+)\)", path.read_text(encoding="utf-8"))


def _anchors(path: Path) -> set[str]:
    anchors = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            slug = re.sub(r"[^\w\- ]", "", line.lstrip("#").strip().lower()).replace(" ", "-")
            anchors.add(slug)
    return anchors


def test_every_relative_link_and_anchor_in_the_documents_resolves():
    for document in (README, METHODS, DOC):
        for link in _links(document):
            if link.startswith(("http://", "https://", "mailto:")):
                continue
            target, _, anchor = link.partition("#")
            resolved = (document.parent / target).resolve() if target else document
            assert resolved.exists(), f"{document.name} links to a missing file: {link}"
            if anchor:
                assert anchor in _anchors(resolved), f"{document.name} links to a missing heading: {link}"


def test_every_command_the_readme_names_is_registered():
    """A README command that the CLI does not know is a stale instruction for a reader starting from a clone."""
    parser = cli.build_parser()
    registered = set(parser._subparsers._group_actions[0].choices)
    text = README.read_text(encoding="utf-8")
    invoked = set(re.findall(r"bayesian-t1dm (?!--)([a-z][a-z-]+)", text))
    assert invoked <= registered, f"README invokes unregistered commands: {invoked - registered}"
    tabled = set(re.findall(r"`([a-z][a-z-]+)[` ]", text)) & registered
    assert {"backfill", "build-dataset", "status", "refresh", "front-page", "report-outcomes", "propose-breakfast-icr"} <= invoked | tabled
    for module in re.findall(r"python -m bayesian_t1dm\.([a-z_]+)", text):
        assert (ROOT / "src" / "bayesian_t1dm" / f"{module}.py").exists(), f"README names a missing module {module}"
