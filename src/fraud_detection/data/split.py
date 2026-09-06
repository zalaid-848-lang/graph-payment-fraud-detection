"""Chronological holdout assignment with explicit purge windows."""

from __future__ import annotations

import math

import pandas as pd


def assign_temporal_splits(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    purge_seconds: int = 86_400,
) -> pd.DataFrame:
    """Assign train, validation, test, and boundary-purge labels.

    Boundaries are chosen by ordered row position, then moved to include all rows
    sharing the boundary time on the earlier side. Purge rows remain visible.
    """
    if frame.empty:
        raise ValueError("Cannot split an empty dataset")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    if purge_seconds < 0:
        raise ValueError("purge_seconds cannot be negative")

    result = frame.sort_values(
        ["transaction_time", "transaction_id"], kind="stable"
    ).reset_index(drop=True).copy()
    row_count = len(result)
    train_position = max(0, min(row_count - 1, math.ceil(row_count * train_fraction) - 1))
    validation_position = max(
        train_position,
        min(row_count - 1, math.ceil(row_count * (train_fraction + validation_fraction)) - 1),
    )
    train_end = int(result.loc[train_position, "transaction_time"])
    validation_end = int(result.loc[validation_position, "transaction_time"])

    time = result["transaction_time"]
    split = pd.Series("test", index=result.index, dtype="string")
    split.loc[time <= train_end] = "train"
    split.loc[(time > train_end) & (time <= train_end + purge_seconds)] = (
        "purge_train_validation"
    )
    split.loc[(time > train_end + purge_seconds) & (time <= validation_end)] = "validation"
    split.loc[(time > validation_end) & (time <= validation_end + purge_seconds)] = (
        "purge_validation_test"
    )
    result["split"] = split

    eligible = set(result["split"])
    required = {"train", "validation", "test"}
    if not required.issubset(eligible):
        raise ValueError(
            "Temporal split produced an empty train, validation, or test period. "
            "Use more time coverage or a shorter purge window."
        )
    return result

