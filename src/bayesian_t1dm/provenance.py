"""The commit a result was built from and whether the working tree was clean, for the provenance sheet of a workbook."""

from __future__ import annotations

import subprocess
from pathlib import Path


def working_tree_is_clean(repo_root: str | Path) -> bool:
    """True when tracked files have no uncommitted changes."""
    result = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True)
    return result.stdout.strip() == ""


def current_commit(repo_root: str | Path) -> str:
    """The commit checked out at ``repo_root``."""
    return subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
