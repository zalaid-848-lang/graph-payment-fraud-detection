"""Ranking and threshold metrics aligned with investigation capacity."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _validated_arrays(target: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(target) != len(scores) or len(target) == 0:
        raise ValueError("target and scores must be non-empty and equally sized")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("target must be binary")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return target, scores


def average_precision(target: np.ndarray, scores: np.ndarray) -> float:
    """Compute non-interpolated PR-AUC, grouping equal-score thresholds."""
    target, scores = _validated_arrays(target, scores)
    positives = int(target.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    sorted_target = target[order]
    sorted_scores = scores[order]
    distinct_end = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    true_positives = np.cumsum(sorted_target)[distinct_end]
    false_positives = (distinct_end + 1) - true_positives
    recall = true_positives / positives
    precision = true_positives / (true_positives + false_positives)
    recall_change = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_change * precision))


def _top_fraction(target: np.ndarray, scores: np.ndarray, fraction: float) -> dict[str, Any]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    reviewed = max(1, math.ceil(len(target) * fraction))
    selected = np.argsort(-scores, kind="stable")[:reviewed]
    true_positives = int(target[selected].sum())
    false_positives = reviewed - true_positives
    positives = int(target.sum())
    negatives = len(target) - positives
    return {
        "fraction": fraction,
        "alert_count": reviewed,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "precision": true_positives / reviewed,
        "recall": true_positives / positives if positives else 0.0,
        "false_positive_rate": false_positives / negatives if negatives else 0.0,
    }


def select_threshold_at_capacity(scores: np.ndarray, capacity_fraction: float) -> float:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0 or not 0 < capacity_fraction <= 1:
        raise ValueError("scores must be non-empty and capacity must be in (0, 1]")
    reviewed = max(1, math.ceil(len(scores) * capacity_fraction))
    return float(np.sort(scores)[::-1][reviewed - 1])


def threshold_metrics(target: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    target, scores = _validated_arrays(target, scores)
    selected = scores >= threshold
    true_positives = int(target[selected].sum())
    false_positives = int(selected.sum()) - true_positives
    positives = int(target.sum())
    negatives = len(target) - positives
    return {
        "threshold": float(threshold),
        "alert_count": int(selected.sum()),
        "alert_fraction": float(selected.mean()),
        "precision": true_positives / int(selected.sum()) if selected.any() else 0.0,
        "recall": true_positives / positives if positives else 0.0,
        "false_positive_rate": false_positives / negatives if negatives else 0.0,
    }


def minimum_alerts_for_recall(
    target: np.ndarray, scores: np.ndarray, target_recall: float
) -> dict[str, Any]:
    """Return the smallest exact top-k review set that reaches a target recall."""
    target, scores = _validated_arrays(target, scores)
    if not 0 <= target_recall <= 1:
        raise ValueError("target_recall must be in [0, 1]")
    positives = int(target.sum())
    required_positives = math.ceil(positives * target_recall)
    if required_positives == 0:
        return {"target_recall": target_recall, "alert_count": 0, "alert_fraction": 0.0}
    sorted_target = target[np.argsort(-scores, kind="stable")]
    positive_positions = np.flatnonzero(sorted_target == 1)
    alert_count = int(positive_positions[required_positives - 1] + 1)
    return {
        "target_recall": target_recall,
        "alert_count": alert_count,
        "alert_fraction": alert_count / len(target),
    }


def evaluate_ranking(
    target: np.ndarray, scores: np.ndarray, *, capacity_fraction: float
) -> dict[str, Any]:
    """Evaluate one score vector without fitting or selecting a threshold."""
    target, scores = _validated_arrays(target, scores)
    return {
        "pr_auc": average_precision(target, scores),
        "at_investigation_capacity": _top_fraction(target, scores, capacity_fraction),
        "top_1_percent": _top_fraction(target, scores, 0.01),
        "top_5_percent": _top_fraction(target, scores, 0.05),
    }
