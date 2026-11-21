"""The tracked documents describe the code as it is: every command, module and heading they name exists."""

from __future__ import annotations

import re
from pathlib import Path

from bayesian_t1dm import cli

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "technical.md"
METHODS = ROOT / "docs" / "methods.md"
README = ROOT / "README.md"
DOCUMENTS = (README, METHODS, DOC)

REQUIRED_SECTIONS = (
    "## Pipeline at a glance",
    "## Download from Tandem Source",
    "## Normalization and timestamps",
    "## Merged dataset",
    "## Settings in force",
    "## Shared definitions",
    "## Outcome metrics",
    "## Insulin action kernel",
    "## Effective I:C from meals",
    "## Basal from the controller pattern",
    "## Personal readout",
    "## Assumptions and failure modes",
)


def _registered() -> set[str]:
    return set(cli.build_parser()._subparsers._group_actions[0].choices)


def test_every_pipeline_step_is_documented():
    """A step that is built must be described: an edit that silently dropped a section would otherwise go unnoticed."""
    headings = {line.strip() for line in DOC.read_text(encoding="utf-8").splitlines() if line.startswith("## ")}
    missing = [section for section in REQUIRED_SECTIONS if section not in headings]
    assert not missing, f"technical.md lacks: {missing}"


def test_goal_table_lists_every_lettered_goal():
    """The goals and lettered subgoals are defined once, at the top of methods.md, and every goal row names its first subgoal."""
    text = METHODS.read_text(encoding="utf-8")
    letters = re.findall(r"^\| ([A-Z]) \|", text.split("## Data processing")[0], flags=re.MULTILINE)
    assert letters == ["A", "B", "C", "D"]
    for letter in letters:
        assert re.search(rf"^\| {letter} \|.*\b{letter}1\b", text, flags=re.MULTILINE), f"goal {letter} has no first subgoal"
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
    """Links into the code (for example ``../src/bayesian_t1dm/readout.py``) fail here when a module is removed or renamed."""
    for document in DOCUMENTS:
        for link in _links(document):
            if link.startswith(("http://", "https://", "mailto:")):
                continue
            target, _, anchor = link.partition("#")
            resolved = (document.parent / target).resolve() if target else document
            assert resolved.exists(), f"{document.name} links to a missing file: {link}"
            if anchor:
                assert anchor in _anchors(resolved), f"{document.name} links to a missing heading: {link}"


def test_every_command_a_document_names_is_registered_and_every_command_is_documented():
    registered = _registered()
    for document in DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        invoked = set(re.findall(r"bayesian-t1dm (?!--)([a-z][a-z-]+)", text))
        assert invoked <= registered, f"{document.name} invokes unregistered commands: {invoked - registered}"
        named = set(re.findall(r"`([a-z]+(?:-[a-z]+)+|collect|backfill|readout|refresh)[` ]", text))
        commandlike = {name for name in named if name in registered or "-" in name and not name.endswith((".py", ".md"))}
        stale = {name for name in commandlike if name not in registered and name.replace("-", "_") not in text}
        assert not stale, f"{document.name} names commands that do not exist: {stale}"
    readme = README.read_text(encoding="utf-8")
    undocumented = {command for command in registered if f"`{command}" not in readme}
    assert not undocumented, f"README does not mention: {undocumented}"


def test_every_module_a_document_names_exists():
    """``name.py`` and ``module.function`` references must point at code that is still there."""
    package = ROOT / "src" / "bayesian_t1dm"
    modules = {path.stem for path in package.glob("*.py")}
    for document in DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        for name in re.findall(r"`([a-z_]+)\.py`", text):
            assert name in modules, f"{document.name} names a missing module {name}.py"
        for module, attribute in re.findall(r"`([a-z_]+)\.([a-z_][a-z0-9_]*)`", text):
            if module in modules and attribute not in {"py", "md", "html", "xlsx", "csv", "json", "parquet", "log"}:
                source = (package / f"{module}.py").read_text(encoding="utf-8")
                assert re.search(rf"^(def|class) {attribute}\b|^{attribute} =", source, flags=re.MULTILINE), f"{document.name} names {module}.{attribute}, which does not exist"
