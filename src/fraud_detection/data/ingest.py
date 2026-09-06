"""IEEE-CIS adapter for the canonical project schema."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detection.data.schema import CANONICAL_COLUMNS, validate_transactions

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"

TRANSACTION_KEY_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
]
IDENTITY_KEY_COLUMNS = ["TransactionID", "DeviceType", "DeviceInfo", "id_30", "id_31"]


def ieee_files_available(raw_dir: str | Path) -> bool:
    """Return whether both required IEEE-CIS training files exist."""
    path = Path(raw_dir)
    return (path / TRANSACTION_FILE).is_file() and (path / IDENTITY_FILE).is_file()


def _read_available_columns(path: Path, desired: list[str], *, nrows: int | None) -> pd.DataFrame:
    available = pd.read_csv(path, nrows=0).columns.tolist()
    selected = [column for column in desired if column in available]
    return pd.read_csv(path, usecols=selected, nrows=nrows, low_memory=False)


def _normalise_part(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip().lower()
    return text or None


def _entity_key(prefix: str, *parts: Any) -> str | pd.NA:
    values = [_normalise_part(part) for part in parts]
    present = [value for value in values if value is not None]
    if not present:
        return pd.NA
    payload = "|".join(value if value is not None else "<missing>" for value in values)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def _keys(frame: pd.DataFrame, prefix: str, columns: list[str]) -> pd.Series:
    values = [_column(frame, column) for column in columns]
    return pd.Series(
        (_entity_key(prefix, *parts) for parts in zip(*values, strict=True)),
        index=frame.index,
        dtype="string",
    )


def load_ieee_cis(raw_dir: str | Path, *, max_rows: int | None = None) -> pd.DataFrame:
    """Load a chronological IEEE-CIS slice and map it to canonical fields.

    Only fields needed for Day 1 entity/schema work are loaded, keeping this adapter
    usable on modest machines. Broader tabular feature ingestion is added on Day 2.
    """
    raw_path = Path(raw_dir)
    transaction_path = raw_path / TRANSACTION_FILE
    identity_path = raw_path / IDENTITY_FILE
    missing = [str(path) for path in (transaction_path, identity_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing IEEE-CIS files: " + ", ".join(missing) + ". See README.md for placement."
        )
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")

    transactions = _read_available_columns(
        transaction_path, TRANSACTION_KEY_COLUMNS, nrows=max_rows
    )
    required = {"TransactionID", "isFraud", "TransactionDT", "TransactionAmt"}
    missing_required = sorted(required - set(transactions.columns))
    if missing_required:
        raise ValueError(f"IEEE transaction file lacks: {', '.join(missing_required)}")

    identities = _read_available_columns(identity_path, IDENTITY_KEY_COLUMNS, nrows=None)
    identities = identities.loc[identities["TransactionID"].isin(transactions["TransactionID"])]
    if identities["TransactionID"].duplicated().any():
        raise ValueError("IEEE identity file has duplicate TransactionID values")
    merged = transactions.merge(identities, on="TransactionID", how="left", validate="one_to_one")

    canonical = pd.DataFrame(index=merged.index)
    canonical["transaction_id"] = merged["TransactionID"].map(lambda value: f"ieee:{value}")
    canonical["transaction_time"] = merged["TransactionDT"]
    canonical["amount"] = merged["TransactionAmt"]
    canonical["product_code"] = _column(merged, "ProductCD").fillna("unknown")
    card_columns = [f"card{index}" for index in range(1, 7)]
    canonical["card_id"] = _keys(merged, "card", card_columns)
    canonical["customer_id"] = pd.Series(
        (
            _entity_key("customer_proxy", card, address, email)
            for card, address, email in zip(
                canonical["card_id"],
                _column(merged, "addr1"),
                _column(merged, "P_emaildomain"),
                strict=True,
            )
        ),
        dtype="string",
    )
    canonical["device_id"] = _keys(
        merged, "device", ["DeviceType", "DeviceInfo", "id_30", "id_31"]
    )
    canonical["address_id"] = _keys(merged, "address", ["addr1", "addr2"])
    canonical["payer_email_domain"] = (
        _column(merged, "P_emaildomain").astype("string").str.strip().str.lower()
    )
    canonical["recipient_email_domain"] = (
        _column(merged, "R_emaildomain").astype("string").str.strip().str.lower()
    )
    canonical["recipient_id"] = pd.Series(
        (
            _entity_key("recipient_proxy", product, email)
            for product, email in zip(
                canonical["product_code"], canonical["recipient_email_domain"], strict=True
            )
        ),
        dtype="string",
    )
    canonical["is_fraud"] = merged["isFraud"]
    canonical["source"] = "ieee_cis"

    optional_raw = [
        column
        for column in merged.columns
        if column not in {"TransactionID", "isFraud", "TransactionDT", "TransactionAmt"}
    ]
    raw = merged[optional_raw].rename(columns=lambda name: f"raw_{name}")
    prepared = pd.concat([canonical[list(CANONICAL_COLUMNS)], raw], axis=1)
    return validate_transactions(prepared)

