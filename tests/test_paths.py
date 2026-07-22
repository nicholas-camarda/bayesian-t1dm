from __future__ import annotations

import os
from pathlib import Path

from bayesian_t1dm.paths import ProjectPaths


def test_pytest_process_unconditionally_isolates_home_runtime_and_cloud():
    test_home = Path(os.environ["HOME"])

    assert test_home.name.startswith("bayesian_t1dm_pytest_")
    assert Path(os.environ["USERPROFILE"]) == test_home
    assert Path(os.environ["XDG_CACHE_HOME"]) == test_home / "cache"
    assert Path(os.environ["MPLCONFIGDIR"]) == test_home / "matplotlib"
    assert Path(os.environ["BAYESIAN_T1DM_RUNTIME_ROOT"]) == test_home / "runtime"
    assert Path(os.environ["BAYESIAN_T1DM_CLOUD_ROOT"]) == test_home / "cloud"

    paths = ProjectPaths.from_root(Path.cwd())

    assert paths.runtime == (test_home / "runtime").resolve()
    assert paths.cloud_root == (test_home / "cloud").resolve()


def test_project_paths_ensure_creates_runtime_paths_without_creating_cloud_paths(tmp_path):
    root = tmp_path / "bayesian-t1dm"
    root.mkdir()

    paths = ProjectPaths.from_root(
        root,
        runtime_root=tmp_path / "runtime",
        cloud_root=tmp_path / "cloud",
    ).ensure()

    assert paths.reports == tmp_path / "runtime" / "output"
    assert paths.cloud_raw == tmp_path / "cloud" / "data" / "raw"
    assert paths.cloud_output == tmp_path / "cloud" / "output"
    assert paths.reports.exists()
    assert not paths.cloud_root.exists()


def test_project_paths_default_slug_survives_workspace_source_checkout(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("BAYESIAN_T1DM_RUNTIME_ROOT", raising=False)
    monkeypatch.delenv("BAYESIAN_T1DM_CLOUD_ROOT", raising=False)
    source_root = tmp_path / "Workspaces" / "bayesian-t1dm" / "source"
    source_root.mkdir(parents=True)

    paths = ProjectPaths.from_root(source_root)

    assert paths.root == source_root
    assert paths.runtime == Path.home() / "Workspaces" / "bayesian-t1dm" / "runtime"
    assert paths.cloud_root == (
        Path.home()
        / "Library"
        / "CloudStorage"
        / "OneDrive-Personal"
        / "SideProjects"
        / "bayesian-t1dm"
    )
