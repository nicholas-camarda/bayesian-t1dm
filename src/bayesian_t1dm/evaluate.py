from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class CalibrationSummary:
    mae: float
    rmse: float
    coverage: float
    interval_width: float


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


def calibration_summary(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    lower: np.ndarray | pd.Series,
    upper: np.ndarray | pd.Series,
) -> CalibrationSummary:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    err = yt - yp
    return CalibrationSummary(
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err**2))),
        coverage=float(np.mean((yt >= lo) & (yt <= hi))),
        interval_width=float(np.mean(hi - lo)),
    )
