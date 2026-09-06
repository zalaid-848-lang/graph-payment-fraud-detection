"""Data-quality profiling and auditable gates for prepared transactions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fraud_detection.data.schema import ENTITY_COLUMNS


def _split_is_ordered(frame: pd.DataFrame, earlier: str, later: str) -> bool:
    earlier_rows = frame[frame["split"] == earlier]
    later_rows = frame[frame["split"] == later]
    if earlier_rows.empty or later_rows.empty:
        return False
    return bool(
        earlier_rows["transaction_time"].max() < later_rows["transaction_time"].min()
    )


def profile_transactions(frame: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-serializable quality statistics and pass/fail gates."""
    required_for_profile = {"transaction_id", "transaction_time", "amount", "is_fraud", "split"}
    missing = sorted(required_for_profile - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot profile data; missing columns: {', '.join(missing)}")

    split_summary: dict[str, dict[str, int | float | None]] = {}
    for split_name, group in frame.groupby("split", sort=False):
        split_summary[str(split_name)] = {
            "rows": int(len(group)),
            "fraud_count": int(group["is_fraud"].sum()),
            "fraud_rate": round(float(group["is_fraud"].mean()), 8),
            "min_transaction_time": int(group["transaction_time"].min()),
            "max_transaction_time": int(group["transaction_time"].max()),
        }

    entity_summary = {
        column: {
            "missing_count": int(frame[column].isna().sum()),
            "missing_rate": round(float(frame[column].isna().mean()), 8),
            "unique_count": int(frame[column].nunique(dropna=True)),
        }
        for column in ENTITY_COLUMNS
    }
    gates = {
        "transaction_ids_unique": not frame["transaction_id"].duplicated().any(),
        "labels_binary": bool(frame["is_fraud"].isin([0, 1]).all()),
        "amounts_non_negative": bool(frame["amount"].ge(0).all()),
        "time_non_decreasing": bool(frame["transaction_time"].is_monotonic_increasing),
        "every_row_has_entity": bool(frame[list(ENTITY_COLUMNS)].notna().any(axis=1).all()),
        "train_precedes_validation": _split_is_ordered(frame, "train", "validation"),
        "validation_precedes_test": _split_is_ordered(frame, "validation", "test"),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "duplicate_transaction_ids": int(frame["transaction_id"].duplicated().sum()),
        "overall_fraud_count": int(frame["is_fraud"].sum()),
        "overall_fraud_rate": round(float(frame["is_fraud"].mean()), 8),
        "split_summary": split_summary,
        "entity_summary": entity_summary,
        "gates": gates,
    }


def render_quality_markdown(profile: dict[str, Any]) -> str:
    """Render a compact quality profile for the experiment report."""
    lines = [
        f"**Quality status:** {profile['status']}",
        "",
        f"Rows: {profile['row_count']:,}; columns: {profile['column_count']}; "
        f"fraud labels: {profile['overall_fraud_count']:,} "
        f"({profile['overall_fraud_rate']:.2%}).",
        "",
        "| Split | Rows | Fraud labels | Fraud rate | Time range (elapsed seconds) |",
        "|---|---:|---:|---:|---:|",
    ]
    for split_name, summary in profile["split_summary"].items():
        lines.append(
            f"| {split_name} | {summary['rows']:,} | {summary['fraud_count']:,} | "
            f"{summary['fraud_rate']:.2%} | {summary['min_transaction_time']:,}–"
            f"{summary['max_transaction_time']:,} |"
        )
    lines.extend(["", "Quality gates:", ""])
    for gate, passed in profile["gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{gate}`")
    return "\n".join(lines)

