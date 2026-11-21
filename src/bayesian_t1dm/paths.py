from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


PROJECT_SLUG = "bayesian-t1dm"


def analysis_timezone() -> str:
    """The timezone all timestamps are stored in (local wall-clock time), from TIMEZONE_NAME.

    Tandem pump times and Apple Health times are both local; storing one of them in UTC shifts it by
    four or five hours against the other. There is deliberately no default.
    """
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    name = os.getenv("TIMEZONE_NAME", "").strip()
    if not name:
        raise ValueError("Set TIMEZONE_NAME (for example America/New_York) in .env or the environment; timestamps are stored in that local time.")
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"TIMEZONE_NAME={name!r} is not a valid IANA timezone name.") from exc
    return name


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw: Path
    processed: Path
    reports: Path
    output_latent_meal: Path
    output_outcomes: Path
    output_replay: Path
    output_replay_predictor: Path
    output_outcomes_model: Path
    output_data_audit: Path
    output_forecast_harness: Path
    output_proposals: Path
    output_readout: Path
    current: Path
    runs: Path
    output_fixture: Path
    output_source: Path
    output_prepare: Path
    cache: Path
    cache_prepared: Path
    cache_analysis_ready: Path
    cache_latent_meal: Path
    cache_prepare: Path
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
    cloud_readout: Path

    @classmethod
    def from_root(
        cls,
        root: str | Path | None = None,
        *,
        runtime_root: str | Path | None = None,
        cloud_root: str | Path | None = None,
    ) -> "ProjectPaths":
        root_path = Path(root or Path.cwd()).resolve()
        runtime_env = runtime_root or os.getenv("BAYESIAN_T1DM_RUNTIME_ROOT")
        cloud_env = cloud_root or os.getenv("BAYESIAN_T1DM_CLOUD_ROOT")
        runtime_path = (
            Path(runtime_env).expanduser().resolve()
            if runtime_env
            else Path.home() / "Workspaces" / PROJECT_SLUG / "runtime"
        )
        cloud_path = (
            Path(cloud_env).expanduser().resolve()
            if cloud_env
            else Path.home()
            / "Library"
            / "CloudStorage"
            / "OneDrive-Personal"
            / "Project Vault"
            / "Personal Projects"
            / PROJECT_SLUG
        )
        return cls(
            root=root_path,
            raw=root_path / "data" / "raw",
            processed=root_path / "data" / "processed",
            reports=runtime_path / "output",
            output_latent_meal=runtime_path / "output" / "latent_meal",
            output_outcomes=runtime_path / "output" / "outcomes",
            output_replay=runtime_path / "output" / "replay",
            output_replay_predictor=runtime_path / "output" / "replay_predictor",
            output_outcomes_model=runtime_path / "output" / "outcomes_model",
            output_data_audit=runtime_path / "output" / "data_audit",
            output_forecast_harness=runtime_path / "output" / "forecast_harness",
            output_proposals=runtime_path / "output" / "proposals",
            output_readout=runtime_path / "output" / "readout",
            current=runtime_path / "current",
            runs=runtime_path / "runs",
            output_fixture=runtime_path / "output" / "fixture",
            output_source=runtime_path / "output" / "source",
            output_prepare=runtime_path / "output" / "prepare",
            cache=runtime_path / "cache",
            cache_prepared=runtime_path / "cache" / "prepared",
            cache_analysis_ready=runtime_path / "cache" / "analysis_ready",
            cache_latent_meal=runtime_path / "cache" / "latent_meal",
            cache_prepare=runtime_path / "cache" / "prepare",
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
            cloud_readout=cloud_path / "readout",
        )

    def ensure(self) -> "ProjectPaths":
        for path in [
            self.runtime,
            self.reports,
            self.output_latent_meal,
            self.output_outcomes,
            self.output_replay,
            self.output_replay_predictor,
            self.output_outcomes_model,
            self.output_data_audit,
            self.output_forecast_harness,
            self.output_proposals,
            self.output_readout,
            self.runs,
            self.output_fixture,
            self.output_source,
            self.output_prepare,
            self.cache,
            self.cache_prepared,
            self.cache_analysis_ready,
            self.cache_latent_meal,
            self.cache_prepare,
            self.cache_status,
            self.logs,
            self.runtime_archive,
            self.legacy_output_archive,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return self
