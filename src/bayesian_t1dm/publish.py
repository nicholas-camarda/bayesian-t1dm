"""Publish a complete refresh run to the Project Vault, byte-identical, with a SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REQUIRED_RUN_FILES = ("run_record.json", "index.html", "A_outcomes/outcomes_report.html", "B_icr/icr_proposal.md", "B_icr/segment_summary.csv")
MANIFEST_NAME = "MANIFEST.sha256"


class PublishRefused(RuntimeError):
    """The run is incomplete, started from a dirty tree, or would overwrite an earlier publish."""


def working_tree_is_clean(repo_root: str | Path) -> bool:
    result = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True)
    return result.stdout.strip() == ""


def current_commit(repo_root: str | Path) -> str:
    return subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_run(run_dir: str | Path, destination: str | Path, *, repo_root: str | Path) -> Path:
    """Copy ``run_dir`` to ``destination`` (``outputs/<date>/``) and write a manifest of every file's SHA-256.

    Refuses a run that lacks a required file or whose record is not marked complete, a run started from a dirty
    working tree (the commit in its record must be the current one and the tree clean), and a destination that exists.
    """
    run_dir, destination = Path(run_dir), Path(destination)
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).exists()]
    if missing:
        raise PublishRefused(f"The run is incomplete; missing {', '.join(missing)}.")
    record = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
    if record.get("status") != "complete":
        raise PublishRefused(f"The run record says {record.get('status')!r}, not 'complete'.")
    if record.get("working_tree_clean") is not True:
        raise PublishRefused("The run started from a dirty working tree.")
    if not working_tree_is_clean(repo_root) or record.get("commit") != current_commit(repo_root):
        raise PublishRefused("The working tree has changed since the run; rerun from a clean checkout of the same commit.")
    if destination.exists():
        raise PublishRefused(f"{destination} already exists; an earlier publish is never overwritten.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, destination)
    files = sorted(path for path in destination.rglob("*") if path.is_file() and path.name != MANIFEST_NAME)
    lines = [f"{_sha256(path)}  {path.relative_to(destination).as_posix()}" for path in files]
    (destination / MANIFEST_NAME).write_text(f"# run {record['run_id']} commit {record['commit']}\n" + "\n".join(lines) + "\n", encoding="utf-8")
    for path in files:  # the copy must match the source byte for byte
        if _sha256(path) != _sha256(run_dir / path.relative_to(destination)):
            raise PublishRefused(f"{path} differs from the run after copying.")
    return destination
