from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision_macro": precision_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, predictions, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, predictions, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, predictions, average="weighted", zero_division=0
        ),
        "auc": roc_auc_score(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, predictions),
        "predictions": predictions,
    }


def find_best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Find the 0.40-0.60 threshold that maximizes macro F1."""
    best_threshold = 0.50
    best_f1 = -1.0
    for threshold in np.arange(0.40, 0.61, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, predictions, average="macro", zero_division=0)
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold, best_f1


def average_history_curve(histories: list[dict], key: str) -> np.ndarray:
    max_length = max(len(history[key]) for history in histories)
    curves = []
    for history in histories:
        values = np.asarray(history[key], dtype=float)
        padded = np.full(max_length, np.nan)
        padded[: len(values)] = values
        curves.append(padded)
    return np.nanmean(np.vstack(curves), axis=0)


def average_history_until_best(
    histories: list[dict],
    best_epochs: list[int],
    key: str,
) -> np.ndarray:
    trimmed = [
        np.asarray(history[key][:best_epoch], dtype=float)
        for history, best_epoch in zip(histories, best_epochs)
    ]
    max_length = max(len(values) for values in trimmed)
    curves = []
    for values in trimmed:
        padded = np.full(max_length, np.nan)
        padded[: len(values)] = values
        curves.append(padded)
    return np.nanmean(np.vstack(curves), axis=0)


def smooth_curve(values: np.ndarray, window: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window:
        return values
    smoothed = np.convolve(values, np.ones(window) / window, mode="same")
    smoothed[0] = np.mean(values[:2])
    smoothed[-1] = np.mean(values[-2:])
    return smoothed

