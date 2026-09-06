"""Data generation, ingestion, schema validation, and splitting."""

from fraud_detection.data.ingest import load_ieee_cis
from fraud_detection.data.schema import DataValidationError, validate_transactions
from fraud_detection.data.split import assign_temporal_splits
from fraud_detection.data.synthetic import generate_synthetic_transactions

__all__ = [
    "DataValidationError",
    "assign_temporal_splits",
    "generate_synthetic_transactions",
    "load_ieee_cis",
    "validate_transactions",
]

