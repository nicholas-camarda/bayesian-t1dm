from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from .model import FitDiagnostics


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


@dataclass(frozen=True)
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    model_mae: float
    model_rmse: float
    model_coverage: float
    persistence_mae: float
    fit_diagnostics: FitDiagnostics | None = None
    prediction_trace: dict[str, list[object]] | None = None


@dataclass(frozen=True)
class WalkForwardReport:
    folds: list[FoldResult]
    aggregate: CalibrationSummary
    aggregate_persistence_mae: float

    @property
    def n_folds(self) -> int:
        return len(self.folds)


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


def persistence_mae(y_true: np.ndarray | pd.Series, horizon_steps: int = 1) -> float:
    """MAE of the naive persistence forecast: predict glucose stays constant for horizon_steps ahead."""
    yt = np.asarray(y_true, dtype=float)
    if len(yt) <= horizon_steps:
        return float("nan")
    errors = yt[horizon_steps:] - yt[:-horizon_steps]
    return float(np.nanmean(np.abs(errors)))


def run_walk_forward(
    feature_frame,
    model,
    *,
    n_folds: int = 4,
    min_test_rows: int = 200,
) -> WalkForwardReport | None:
    """
    Run walk-forward cross-validation on a FeatureFrame.

    Returns a WalkForwardReport, or None if there is insufficient data.
    Each fold trains on all data up to the fold boundary and evaluates on the next test window.
    """
    frame = feature_frame.frame
    n = len(frame)
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    test_size = max(n // (n_folds + 1), 1)
    initial_train_size = n - n_folds * test_size

    if initial_train_size <= 0 or initial_train_size + test_size > n:
        warnings.warn(
            "Insufficient data for walk-forward calibration — skipping evaluation.",
            UserWarning,
            stacklevel=2,
        )
        return None

    splits = list(walk_forward_splits(n, initial_train_size=initial_train_size, test_size=test_size))

    if not splits:
        warnings.warn(
            "Insufficient data for walk-forward calibration — skipping evaluation.",
            UserWarning,
            stacklevel=2,
        )
        return None

    # Compute horizon_steps for persistence baseline and to avoid label leakage across folds.
    step_minutes = int(pd.Timedelta(feature_frame.config.freq).total_seconds() // 60)
    horizon_steps = max(int(feature_frame.horizon_minutes / step_minutes), 1)

    fold_results: list[FoldResult] = []
    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    all_lower: list[np.ndarray] = []
    all_upper: list[np.ndarray] = []
    all_persistence_abs_errors: list[np.ndarray] = []

    for fold_idx, split in enumerate(splits):
        # Avoid label leakage: target_glucose at time t depends on glucose at t + horizon_steps.
        # If the training window ends at the test boundary, the last horizon_steps training rows
        # would have targets inside the test period.
        train_end = split.train_end - horizon_steps
        if train_end <= split.train_start:
            warnings.warn(
                "Insufficient data for walk-forward calibration — skipping evaluation.",
                UserWarning,
                stacklevel=2,
            )
            return None

        train_frame = frame.iloc[split.train_start:train_end]
        test_frame = frame.iloc[split.test_start:split.test_end]

        if len(test_frame) < min_test_rows:
            warnings.warn(
                f"Fold {fold_idx + 1} test window has only {len(test_frame)} rows "
                f"(minimum {min_test_rows}); calibration metrics may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        from .features import FeatureFrame
        train_ff = FeatureFrame(
            frame=train_frame,
            feature_columns=feature_frame.feature_columns,
            target_column=feature_frame.target_column,
            horizon_minutes=feature_frame.horizon_minutes,
            config=feature_frame.config,
        )
        fit = model.fit(train_ff)
        preds = model.predict(fit, test_frame)

        y_true = test_frame[feature_frame.target_column].to_numpy()
        y_pred = preds["mean"].to_numpy()
        lower = preds["lower"].to_numpy()
        upper = preds["upper"].to_numpy()

        fold_cal = calibration_summary(y_true, y_pred, lower, upper)
        # Persistence baseline aligned to evaluated rows: predict target_glucose ~= current glucose.
        persistence_abs = np.abs(y_true - test_frame["glucose"].to_numpy(dtype=float))
        p_mae = float(np.nanmean(persistence_abs))

        fold_results.append(FoldResult(
            fold=fold_idx + 1,
            n_train=len(train_frame),
            n_test=len(test_frame),
            model_mae=fold_cal.mae,
            model_rmse=fold_cal.rmse,
            model_coverage=fold_cal.coverage,
            persistence_mae=p_mae,
            fit_diagnostics=getattr(fit, "diagnostics", None),
            prediction_trace={
                "timestamps": [timestamp.isoformat() for timestamp in pd.to_datetime(test_frame["timestamp"], errors="coerce")],
                "actual": y_true.tolist(),
                "predicted": y_pred.tolist(),
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "interval_hit": ((y_true >= lower) & (y_true <= upper)).astype(int).tolist(),
            } if "timestamp" in test_frame.columns else None,
        ))

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        all_lower.append(lower)
        all_upper.append(upper)
        all_persistence_abs_errors.append(persistence_abs)

    agg_y_true = np.concatenate(all_y_true)
    agg_y_pred = np.concatenate(all_y_pred)
    agg_lower = np.concatenate(all_lower)
    agg_upper = np.concatenate(all_upper)
    agg_cal = calibration_summary(agg_y_true, agg_y_pred, agg_lower, agg_upper)

    agg_persistence_abs = np.concatenate(all_persistence_abs_errors)
    agg_p_mae = float(np.nanmean(agg_persistence_abs)) if len(agg_persistence_abs) else float("nan")

    return WalkForwardReport(
        folds=fold_results,
        aggregate=agg_cal,
        aggregate_persistence_mae=agg_p_mae,
    )
