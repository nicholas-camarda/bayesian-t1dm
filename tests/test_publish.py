from __future__ import annotations

import json
import subprocess

import pytest

from bayesian_t1dm.publish import MANIFEST_NAME, REQUIRED_RUN_FILES, PublishRefused, current_commit, publish_run


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for cmd in (["init", "-q"], ["config", "user.email", "t@example.com"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True)
    (repo / "a.txt").write_text("a")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    return repo


def _run(tmp_path, repo, **record):
    run = tmp_path / "run"
    for name in REQUIRED_RUN_FILES:
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text(name)
    base = {"run_id": "r1", "status": "complete", "working_tree_clean": True, "commit": current_commit(repo)}
    (run / "run_record.json").write_text(json.dumps({**base, **record}))
    return run


def test_a_complete_clean_run_is_copied_byte_identical_with_a_manifest(tmp_path):
    repo = _repo(tmp_path)
    run = _run(tmp_path, repo)
    destination = publish_run(run, tmp_path / "vault" / "outputs" / "2026-11-18", repo_root=repo)
    manifest = (destination / MANIFEST_NAME).read_text()
    assert "B_icr/icr_proposal.md" in manifest and "commit" in manifest.splitlines()[0]
    assert (destination / "A_outcomes/outcomes_report.html").read_bytes() == (run / "A_outcomes/outcomes_report.html").read_bytes()


@pytest.mark.parametrize("problem", ["incomplete", "status", "dirty_record", "dirty_tree", "exists", "moved"])
def test_a_run_is_refused_when_incomplete_dirty_or_would_overwrite(tmp_path, problem):
    repo = _repo(tmp_path)
    run = _run(tmp_path, repo, **({"status": "failed"} if problem == "status" else {"working_tree_clean": False} if problem == "dirty_record" else {}))
    destination = tmp_path / "vault" / "2026-11-18"
    if problem == "incomplete":
        (run / REQUIRED_RUN_FILES[1]).unlink()
    if problem == "dirty_tree":
        (repo / "a.txt").write_text("changed")
    if problem == "exists":
        destination.mkdir(parents=True)
    if problem == "moved":
        (repo / "b.txt").write_text("b")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "next"], check=True)
    with pytest.raises(PublishRefused):
        publish_run(run, destination, repo_root=repo)
