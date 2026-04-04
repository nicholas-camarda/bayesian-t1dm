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
    output_forecast: Path
    output_therapy: Path
    output_latent_meal: Path
    output_fixture: Path
    output_source: Path
    output_prepare: Path
    cache: Path
    cache_prepared: Path
    cache_analysis_ready: Path
    cache_latent_meal: Path
    cache_prepare: Path
    cache_forecast: Path
    cache_status: Path
    logs: Path
    runtime_archive: Path
    legacy_output_archive: Path
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
            reports=runtime_path / "output",
            output_forecast=runtime_path / "output" / "forecast",
            output_therapy=runtime_path / "output" / "therapy",
            output_latent_meal=runtime_path / "output" / "latent_meal",
            output_fixture=runtime_path / "output" / "fixture",
            output_source=runtime_path / "output" / "source",
            output_prepare=runtime_path / "output" / "prepare",
            cache=runtime_path / "cache",
            cache_prepared=runtime_path / "cache" / "prepared",
            cache_analysis_ready=runtime_path / "cache" / "analysis_ready",
            cache_latent_meal=runtime_path / "cache" / "latent_meal",
            cache_prepare=runtime_path / "cache" / "prepare",
            cache_forecast=runtime_path / "cache" / "forecast",
            cache_status=runtime_path / "cache" / "status",
            logs=runtime_path / "logs",
            runtime_archive=runtime_path / "archive",
            legacy_output_archive=runtime_path / "archive" / "legacy_output",
            archive=root_path / "archive",
            runtime=runtime_path,
            cloud_root=cloud_path,
            cloud_raw=cloud_path / "data" / "raw",
            cloud_output=cloud_path / "output",
            cloud_archive=cloud_path / "archive",
        )

    def ensure(self) -> "ProjectPaths":
        for path in [
            self.runtime,
            self.reports,
            self.output_forecast,
            self.output_therapy,
            self.output_latent_meal,
            self.output_fixture,
            self.output_source,
            self.output_prepare,
            self.cache,
            self.cache_prepared,
            self.cache_analysis_ready,
            self.cache_latent_meal,
            self.cache_prepare,
            self.cache_forecast,
            self.cache_status,
            self.logs,
            self.runtime_archive,
            self.legacy_output_archive,
            self.cloud_root,
            self.cloud_raw,
            self.cloud_output,
            self.cloud_archive,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return self
