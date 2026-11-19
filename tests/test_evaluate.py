from __future__ import annotations

from bayesian_t1dm.evaluate import walk_forward_splits


def test_walk_forward_splits_progresses_forward():
    splits = list(walk_forward_splits(20, initial_train_size=8, test_size=4, step_size=4))

    assert splits[0].train_end == 8
    assert splits[0].test_start == 8
    assert splits[1].train_end == 12
    assert splits[1].test_end == 16
