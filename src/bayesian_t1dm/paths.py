from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw: Path
    processed: Path
    reports: Path
    archive: Path

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> "ProjectPaths":
        root_path = Path(root or Path.cwd()).resolve()
        return cls(
            root=root_path,
            raw=root_path / "data" / "raw",
            processed=root_path / "data" / "processed",
            reports=root_path / "output",
            archive=root_path / "archive",
        )

    def ensure(self) -> "ProjectPaths":
        for path in [self.raw, self.processed, self.reports, self.archive]:
            path.mkdir(parents=True, exist_ok=True)
        return self
