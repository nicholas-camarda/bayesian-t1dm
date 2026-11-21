from __future__ import annotations

import subprocess

from bayesian_t1dm.provenance import current_commit, working_tree_is_clean


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for cmd in (["init", "-q"], ["config", "user.email", "t@example.com"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True)
    (repo / "a.txt").write_text("a")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    return repo


def test_commit_and_clean_state_are_read_from_git(tmp_path):
    repo = _repo(tmp_path)
    assert len(current_commit(repo)) == 40
    assert working_tree_is_clean(repo)
    (repo / "a.txt").write_text("changed")
    assert not working_tree_is_clean(repo), "an edit to a tracked file makes the tree dirty"
