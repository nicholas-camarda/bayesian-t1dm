from __future__ import annotations

from bayesian_t1dm.paths import ProjectPaths


def test_project_paths_default_runtime_reports_and_cloud_raw(tmp_path):
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
    assert paths.cloud_raw.exists()
