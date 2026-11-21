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
    """Where code, generated work and the durable record live.

    ``runtime`` holds generated work (``output/`` with ``source/`` for acquisition summaries and ``readout/`` for the
    readout; ``logs/``). ``cloud_root`` is the Project Vault folder: raw data under ``data/raw`` (the merged dataset sits
    beside it in ``data/merged``) and the readout page in ``readout/``.
    """

    root: Path
    runtime: Path
    reports: Path
    output_source: Path
    output_readout: Path
    logs: Path
    cloud_root: Path
    cloud_raw: Path
    cloud_readout: Path

    @classmethod
    def from_root(cls, root: str | Path | None = None, *, runtime_root: str | Path | None = None, cloud_root: str | Path | None = None) -> "ProjectPaths":
        root_path = Path(root or Path.cwd()).resolve()
        runtime_env = runtime_root or os.getenv("BAYESIAN_T1DM_RUNTIME_ROOT")
        cloud_env = cloud_root or os.getenv("BAYESIAN_T1DM_CLOUD_ROOT")
        runtime_path = Path(runtime_env).expanduser().resolve() if runtime_env else Path.home() / "Workspaces" / PROJECT_SLUG / "runtime"
        cloud_path = (Path(cloud_env).expanduser().resolve() if cloud_env
                      else Path.home() / "Library" / "CloudStorage" / "OneDrive-Personal" / "Project Vault" / "Personal Projects" / PROJECT_SLUG)
        return cls(
            root=root_path, runtime=runtime_path, reports=runtime_path / "output", output_source=runtime_path / "output" / "source",
            output_readout=runtime_path / "output" / "readout", logs=runtime_path / "logs",
            cloud_root=cloud_path, cloud_raw=cloud_path / "data" / "raw", cloud_readout=cloud_path / "readout",
        )

    def ensure(self) -> "ProjectPaths":
        for path in (self.runtime, self.reports, self.output_source, self.output_readout, self.logs):
            path.mkdir(parents=True, exist_ok=True)
        return self
