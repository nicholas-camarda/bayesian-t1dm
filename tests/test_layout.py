from __future__ import annotations

import pytest

from bayesian_t1dm import layout


def _output(tmp_path, skip=()):
    out = tmp_path / ("output_" + "_".join(sorted(x.replace("/", "-") for x in skip)))
    for destination, source in layout.RUN_LAYOUT.items():
        if source.startswith("merged/") or source in skip:
            continue
        path = out / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return out


def test_a_run_is_organised_by_goal_with_supporting_files_beside_them(tmp_path):
    raw = tmp_path / "data" / "raw"
    (tmp_path / "data" / "merged").mkdir(parents=True)
    (tmp_path / "data" / "merged" / "validation_report.md").write_text("v")
    record = tmp_path / "setting_proposals.csv"
    record.write_text("r")
    run = tmp_path / "run"
    written = layout.assemble_run(_output(tmp_path), raw, run, record_path=record)
    assert (run / "dashboard.html").read_text() == "proposals/dashboard.html" and not (run / "index.html").exists()
    assert (run / "B_icr" / "supporting" / "effective_icr_evidence.csv").exists() and (run / "C_changes" / "setting_proposals.csv").exists()
    assert (run / "00_General" / "validation_report.md").exists() and "A_outcomes/outcomes_report.html" in written


def test_a_missing_required_file_stops_the_run_but_an_optional_one_does_not(tmp_path):
    raw = tmp_path / "data" / "raw"
    with pytest.raises(FileNotFoundError):
        layout.assemble_run(_output(tmp_path, skip=("proposals/dashboard.html",)), raw, tmp_path / "run1")
    written = layout.assemble_run(_output(tmp_path, skip=("proposals/change_log.md",)), raw, tmp_path / "run2")
    assert "C_changes/change_log.md" not in written and "dashboard.html" in written


def test_current_is_replaced_only_after_the_copy_succeeds(tmp_path):
    run, current = tmp_path / "run", tmp_path / "current"
    run.mkdir()
    (run / "index.html").write_text("new")
    current.mkdir()
    (current / "index.html").write_text("old")
    layout.activate_current(run, current)
    assert (current / "index.html").read_text() == "new" and not (tmp_path / "current.new").exists()


def test_every_file_the_front_page_links_is_one_a_run_copies_in():
    destinations = set(layout.RUN_LAYOUT) | {"C_changes/setting_proposals.csv"}
    linked = [path for _, _, entries in layout.GOAL_LINKS for path, _ in entries]
    assert linked and set(linked) <= destinations
    assert all(path.startswith(folder + "/") for folder, _, entries in layout.GOAL_LINKS for path, _ in entries)
