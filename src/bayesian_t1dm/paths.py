from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw: Path
    processed: Path
    reports: Path
    archive: Path
    runtime: Path
    cloud_root: Path
    cloud_raw: Path
    cloud_output: Path
    cloud_archive: Path

    @classmethod
    def from_root(
        cls,
        root: str | Path | None = None,
        *,
        runtime_root: str | Path | None = None,
        cloud_root: str | Path | None = None,
    ) -> "ProjectPaths":
        root_path = Path(root or Path.cwd()).resolve()
        slug = root_path.name
        runtime_env = runtime_root or os.getenv("BAYESIAN_T1DM_RUNTIME_ROOT")
        cloud_env = cloud_root or os.getenv("BAYESIAN_T1DM_CLOUD_ROOT")
        runtime_path = Path(runtime_env).expanduser().resolve() if runtime_env else Path.home() / "ProjectsRuntime" / slug
        cloud_path = Path(cloud_env).expanduser().resolve() if cloud_env else Path.home() / "Library" / "CloudStorage" / "OneDrive-Personal" / "SideProjects" / slug
        return cls(
            root=root_path,
            raw=root_path / "data" / "raw",
            processed=root_path / "data" / "processed",
            reports=root_path / "output",
            archive=root_path / "archive",
            runtime=runtime_path,
            cloud_root=cloud_path,
            cloud_raw=cloud_path / "data" / "raw",
            cloud_output=cloud_path / "output",
            cloud_archive=cloud_path / "archive",
        )

    def ensure(self) -> "ProjectPaths":
        for path in [
            self.raw,
            self.processed,
            self.reports,
            self.archive,
            self.runtime,
            self.cloud_root,
            self.cloud_raw,
            self.cloud_output,
            self.cloud_archive,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        for path in [
            self.runtime_browser,
            self.runtime_browser_home,
            self.runtime_downloads,
            self.runtime_traces,
            self.runtime_logs,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def runtime_browser(self) -> Path:
        return self.runtime / "browser-profile"

    @property
    def runtime_browser_home(self) -> Path:
        return self.runtime / "browser-home"

    @property
    def runtime_downloads(self) -> Path:
        return self.runtime / "downloads"

    @property
    def runtime_traces(self) -> Path:
        return self.runtime / "traces"

    @property
    def runtime_logs(self) -> Path:
        return self.runtime / "logs"
