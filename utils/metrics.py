from __future__ import annotations

import numpy as np


def compute_overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        acc_per_class: [num_classes] float
        correct_per_class: [num_classes] int
        total_per_class: [num_classes] int
    """
    correct_per_class = np.zeros(num_classes, dtype=np.int64)
    total_per_class = np.zeros(num_classes, dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        total_per_class[int(t)] += 1
        if int(p) == int(t):
            correct_per_class[int(t)] += 1

    acc_per_class = np.divide(
        correct_per_class,
        total_per_class,
        out=np.zeros_like(correct_per_class, dtype=np.float64),
        where=total_per_class != 0,
    )
    return acc_per_class, correct_per_class, total_per_class

