"""Canonical schema and validation for transaction-level project data."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

CANONICAL_COLUMNS = (
    "transaction_id",
    "transaction_time",
    "amount",
    "product_code",
    "card_id",
    "customer_id",
    "device_id",
    "address_id",
    "payer_email_domain",
    "recipient_email_domain",
    "recipient_id",
    "is_fraud",
    "source",
)

ENTITY_COLUMNS = (
    "card_id",
    "customer_id",
    "device_id",
    "address_id",
    "payer_email_domain",
    "recipient_email_domain",
    "recipient_id",
)


class DataValidationError(ValueError):
    """Raised when prepared transactions violate the canonical contract."""


def _format_values(values: Iterable[object], limit: int = 5) -> str:
    return ", ".join(map(str, list(values)[:limit]))


def validate_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a canonical transaction frame.

    Returns a copy sorted in causal order. This function intentionally validates
    only fields common to all sources; optional ``raw_`` columns pass through.
    """
    missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")

    result = frame.copy()
    result["transaction_id"] = result["transaction_id"].astype("string")
    if result["transaction_id"].isna().any() or (result["transaction_id"].str.len() == 0).any():
        raise DataValidationError("transaction_id must be non-empty")

    duplicate_ids = result.loc[
        result["transaction_id"].duplicated(keep=False), "transaction_id"
    ].unique()
    if len(duplicate_ids):
        raise DataValidationError(
            f"transaction_id must be unique; duplicates: {_format_values(duplicate_ids)}"
        )

    result["transaction_time"] = pd.to_numeric(result["transaction_time"], errors="coerce")
    if result["transaction_time"].isna().any() or (result["transaction_time"] < 0).any():
        raise DataValidationError("transaction_time must contain non-negative elapsed seconds")
    result["transaction_time"] = result["transaction_time"].astype("int64")

    result["amount"] = pd.to_numeric(result["amount"], errors="coerce")
    if result["amount"].isna().any() or (result["amount"] < 0).any():
        raise DataValidationError("amount must be numeric, non-null, and non-negative")

    result["is_fraud"] = pd.to_numeric(result["is_fraud"], errors="coerce")
    invalid_labels = sorted(set(result["is_fraud"].dropna()) - {0, 1})
    if result["is_fraud"].isna().any() or invalid_labels:
        raise DataValidationError("is_fraud must contain only 0 or 1")
    result["is_fraud"] = result["is_fraud"].astype("int8")

    for column in (*ENTITY_COLUMNS, "product_code", "source"):
        result[column] = result[column].astype("string")

    if result[list(ENTITY_COLUMNS)].isna().all(axis=1).any():
        bad_ids = result.loc[
            result[list(ENTITY_COLUMNS)].isna().all(axis=1), "transaction_id"
        ].tolist()
        raise DataValidationError(
            "Each transaction needs at least one graph entity; affected IDs: "
            f"{_format_values(bad_ids)}"
        )

    return result.sort_values(["transaction_time", "transaction_id"], kind="stable").reset_index(
        drop=True
    )

