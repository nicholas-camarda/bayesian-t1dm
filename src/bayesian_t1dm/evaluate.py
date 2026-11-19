"""Time-ordered train/test splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def walk_forward_splits(
    n_rows: int,
    *,
    initial_train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> Iterator[WalkForwardSplit]:
    if initial_train_size <= 0 or test_size <= 0:
        raise ValueError("initial_train_size and test_size must be positive")
    if initial_train_size + test_size > n_rows:
        return
    step = step_size or test_size
    train_end = initial_train_size
    while train_end + test_size <= n_rows:
        yield WalkForwardSplit(0, train_end, train_end, train_end + test_size)
        train_end += step
