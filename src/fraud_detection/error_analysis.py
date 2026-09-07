"""Post-evaluation error slices for an investigator-prioritization ranking."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _slice_summary(
    frame: pd.DataFrame,
    scores: np.ndarray,
    alerts: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    target = frame["is_fraud"].to_numpy(dtype=int)
    slice_target = target[mask]
    slice_alerts = alerts[mask]
    rows = int(mask.sum())
    fraud_labels = int(slice_target.sum())
    alert_count = int(slice_alerts.sum())
    true_positives = int((slice_alerts & (slice_target == 1)).sum())
    false_positives = int((slice_alerts & (slice_target == 0)).sum())
    false_negatives = fraud_labels - true_positives
    return {
        "rows": rows,
        "fraud_labels": fraud_labels,
        "fraud_prevalence": fraud_labels / rows if rows else 0.0,
        "mean_score": float(scores[mask].mean()) if rows else 0.0,
        "alert_count": alert_count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": true_positives / alert_count if alert_count else 0.0,
        "recall_within_slice": true_positives / fraud_labels if fraud_labels else None,
    }


def _case_records(
    frame: pd.DataFrame,
    scores: np.ndarray,
    alerts: np.ndarray,
    *,
    false_positive: bool,
    limit: int = 5,
) -> list[dict[str, Any]]:
    target = frame["is_fraud"].to_numpy(dtype=int)
    mask = (alerts & (target == 0)) if false_positive else (~alerts & (target == 1))
    candidates = np.flatnonzero(mask)
    ordered = candidates[np.argsort(-scores[candidates], kind="stable")][:limit]
    records = []
    for index in ordered:
        row = frame.iloc[index]
        record = {
            "transaction_id": str(row["transaction_id"]),
            "amount": float(row["amount"]),
            "product_code": str(row["product_code"]),
            "specific_shared_transaction_count": int(row["specific_shared_transaction_count"]),
            "specific_known_entity_count": int(row["specific_known_entity_count"]),
            "matured_neighbor_label_count": int(row["matured_neighbor_label_count"]),
            "matured_neighbor_fraud_ratio": float(row["matured_neighbor_fraud_ratio"]),
            "score": float(scores[index]),
        }
        records.append(record)
    return records


def build_error_analysis(
    train: pd.DataFrame,
    test: pd.DataFrame,
    scores: np.ndarray,
    *,
    capacity_fraction: float,
) -> dict[str, Any]:
    """Create exhaustive post-test slices without changing the fitted model."""

    required = {
        "transaction_id",
        "is_fraud",
        "amount",
        "product_code",
        "specific_shared_transaction_count",
        "specific_known_entity_count",
        "matured_neighbor_label_count",
        "matured_neighbor_fraud_count",
        "matured_neighbor_fraud_ratio",
    }
    missing = sorted((required - set(train.columns)) | (required - set(test.columns)))
    if missing:
        raise ValueError(f"Missing error-analysis columns: {', '.join(missing)}")
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(test) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and match the test row count")
    if not 0 < capacity_fraction <= 1:
        raise ValueError("capacity_fraction must be in (0, 1]")

    review_count = max(1, math.ceil(len(test) * capacity_fraction))
    alerts = np.zeros(len(test), dtype=bool)
    alerts[np.argsort(-scores, kind="stable")[:review_count]] = True
    target = test["is_fraud"].to_numpy(dtype=int)
    if not np.isin(target, [0, 1]).all():
        raise ValueError("Test target must be binary")

    amount_median = float(train["amount"].median())
    amount_ninetieth = float(train["amount"].quantile(0.9))
    connectivity_median = float(train["specific_shared_transaction_count"].median())
    connectivity_ninetieth = float(train["specific_shared_transaction_count"].quantile(0.9))
    amount = test["amount"].to_numpy(dtype=float)
    shared = test["specific_shared_transaction_count"].to_numpy(dtype=float)
    known = test["specific_known_entity_count"].to_numpy(dtype=float)
    label_count = test["matured_neighbor_label_count"].to_numpy(dtype=float)
    fraud_count = test["matured_neighbor_fraud_count"].to_numpy(dtype=float)
    product = test["product_code"].astype("string").fillna("<missing>")

    masks: dict[str, dict[str, np.ndarray]] = {
        "amount_band": {
            "at_or_below_train_median": amount <= amount_median,
            "train_median_to_p90": (amount > amount_median) & (amount <= amount_ninetieth),
            "above_train_p90": amount > amount_ninetieth,
        },
        "specific_connectivity": {
            "at_or_below_train_median": shared <= connectivity_median,
            "train_median_to_p90": (shared > connectivity_median)
            & (shared <= connectivity_ninetieth),
            "above_train_p90": shared > connectivity_ninetieth,
        },
        "specific_entity_history": {
            "all_specific_entities_new": known == 0,
            "at_least_one_known_specific_entity": known > 0,
        },
        "matured_label_history": {
            "no_matured_neighbour": label_count == 0,
            "matured_neighbours_no_fraud": (label_count > 0) & (fraud_count == 0),
            "matured_fraud_neighbour": fraud_count > 0,
        },
        "product_code": {
            str(value): product.eq(value).to_numpy() for value in sorted(product.unique())
        },
    }
    families = {
        family: {
            name: _slice_summary(test, scores, alerts, mask) for name, mask in family_masks.items()
        }
        for family, family_masks in masks.items()
    }
    total_fraud = int(target.sum())
    gates = {
        "selected_alert_count_matches_capacity": int(alerts.sum()) == review_count,
        "slice_rows_are_exhaustive": all(
            sum(value["rows"] for value in slices.values()) == len(test)
            for slices in families.values()
        ),
        "slice_alerts_are_exhaustive": all(
            sum(value["alert_count"] for value in slices.values()) == review_count
            for slices in families.values()
        ),
        "slice_fraud_labels_are_exhaustive": all(
            sum(value["fraud_labels"] for value in slices.values()) == total_fraud
            for slices in families.values()
        ),
    }
    true_positives = int((alerts & (target == 1)).sum())
    false_positives = int((alerts & (target == 0)).sum())
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "capacity_fraction": capacity_fraction,
        "review_count": review_count,
        "overall": {
            "fraud_labels": total_fraud,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": total_fraud - true_positives,
        },
        "train_derived_amount_cutoffs": {
            "median": amount_median,
            "p90": amount_ninetieth,
        },
        "train_derived_connectivity_cutoffs": {
            "median": connectivity_median,
            "p90": connectivity_ninetieth,
        },
        "families": families,
        "example_cases": {
            "highest_scored_false_positives": _case_records(
                test, scores, alerts, false_positive=True
            ),
            "highest_scored_missed_fraud": _case_records(
                test, scores, alerts, false_positive=False
            ),
        },
    }
