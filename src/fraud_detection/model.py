"""Leakage-safe tabular encoding and a scikit-learn logistic baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

TARGET_ALIASES = {"is_fraud", "isFraud", "target", "label"}
IDENTIFIER_COLUMNS = {
    "transaction_id",
    "card_id",
    "customer_id",
    "device_id",
    "address_id",
    "recipient_id",
}
NUMERIC_FEATURES = ("amount",)
CATEGORICAL_FEATURES = ("product_code", "payer_email_domain", "recipient_email_domain")
AVAILABILITY_FEATURES = (
    "card_id",
    "customer_id",
    "device_id",
    "address_id",
    "payer_email_domain",
    "recipient_email_domain",
    "recipient_id",
)


def audit_feature_names(feature_names: list[str]) -> None:
    """Fail loudly if a target alias or direct identifier reaches the model."""
    forbidden = [
        name
        for name in feature_names
        if name in TARGET_ALIASES or name in IDENTIFIER_COLUMNS
    ]
    if forbidden:
        raise ValueError(f"Forbidden model features: {', '.join(sorted(forbidden))}")


class TabularEncoder(BaseEstimator, TransformerMixin):
    """Fit numeric statistics and categorical vocabulary on training rows only."""

    def fit(self, frame: pd.DataFrame, target: pd.Series | None = None) -> TabularEncoder:
        required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + AVAILABILITY_FEATURES)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing tabular input columns: {', '.join(missing)}")
        self.amount_median_ = float(frame["amount"].median())
        log_amount = np.log1p(frame["amount"].fillna(self.amount_median_).to_numpy(dtype=float))
        self.log_amount_mean_ = float(log_amount.mean())
        self.log_amount_scale_ = float(log_amount.std()) or 1.0
        self.categories_ = {
            column: sorted(frame[column].astype("string").fillna("<missing>").unique().tolist())
            for column in CATEGORICAL_FEATURES
        }
        names = ["log_amount_standardized"]
        names.extend(f"has_{column.removesuffix('_id')}" for column in AVAILABILITY_FEATURES)
        for column, categories in self.categories_.items():
            names.extend(f"{column}={category}" for category in categories)
        audit_feature_names(names)
        self.feature_names_ = names
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "feature_names_"):
            raise RuntimeError("TabularEncoder must be fitted before transform")
        amount = frame["amount"].fillna(self.amount_median_).to_numpy(dtype=float)
        log_amount = ((np.log1p(amount) - self.log_amount_mean_) / self.log_amount_scale_).reshape(
            -1, 1
        )
        availability = np.column_stack(
            [frame[column].notna().to_numpy(dtype=float) for column in AVAILABILITY_FEATURES]
        )
        encoded_parts = [log_amount, availability]
        for column in CATEGORICAL_FEATURES:
            values = frame[column].astype("string").fillna("<missing>")
            encoded_parts.append(
                np.column_stack(
                    [
                        values.eq(category).to_numpy(dtype=float)
                        for category in self.categories_[column]
                    ]
                )
            )
        return np.column_stack(encoded_parts)


@dataclass
class TabularBaseline:
    """Train-only encoder plus a class-balanced scikit-learn classifier."""

    l2_strength: float = 0.10
    max_iterations: int = 5000
    tolerance: float = 1e-8
    random_seed: int = 42

    def fit(self, frame: pd.DataFrame) -> TabularBaseline:
        if self.l2_strength <= 0 or self.max_iterations <= 0 or self.tolerance <= 0:
            raise ValueError("Logistic-regression settings must be positive")
        target = frame["is_fraud"].to_numpy(dtype=int)
        if set(np.unique(target)) != {0, 1}:
            raise ValueError("Training target must contain both 0 and 1")

        self.pipeline_ = Pipeline(
            steps=[
                ("encoder", TabularEncoder()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0 / self.l2_strength,
                        class_weight="balanced",
                        max_iter=self.max_iterations,
                        random_state=self.random_seed,
                        solver="lbfgs",
                        tol=self.tolerance,
                    ),
                ),
            ]
        )
        self.pipeline_.fit(frame, target)
        self.encoder_ = self.pipeline_.named_steps["encoder"]
        self.classifier_ = self.pipeline_.named_steps["classifier"]
        self.iterations_ = int(self.classifier_.n_iter_[0])
        self.converged_ = self.iterations_ < self.max_iterations

        positives = int(target.sum())
        negatives = len(target) - positives
        sample_weight = np.where(
            target == 1,
            len(target) / (2.0 * positives),
            len(target) / (2.0 * negatives),
        )
        self.training_loss_ = float(
            log_loss(target, self.score(frame), sample_weight=sample_weight, labels=[0, 1])
        )
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("TabularBaseline must be fitted before scoring")
        return self.pipeline_.predict_proba(frame)[:, 1]

    def strongest_coefficients(self, limit: int = 8) -> list[tuple[str, float]]:
        coefficients = self.classifier_.coef_[0]
        order = np.argsort(np.abs(coefficients))[::-1][:limit]
        return [
            (self.encoder_.feature_names_[index], float(coefficients[index])) for index in order
        ]
