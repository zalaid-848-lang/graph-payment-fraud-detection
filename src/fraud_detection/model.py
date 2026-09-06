"""Leakage-safe tabular encoding and a transparent NumPy logistic baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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


class TabularEncoder:
    """Fit numeric statistics and categorical vocabulary on training rows only."""

    def fit(self, frame: pd.DataFrame) -> "TabularEncoder":
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
                    [values.eq(category).to_numpy(dtype=float) for category in self.categories_[column]]
                )
            )
        return np.column_stack(encoded_parts)


@dataclass
class NumpyLogisticRegression:
    """Small deterministic L2-regularized logistic regression for the baseline."""

    l2_strength: float = 0.10
    learning_rate: float = 0.10
    max_iterations: int = 5000
    tolerance: float = 1e-8

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def fit(self, features: np.ndarray, target: np.ndarray) -> "NumpyLogisticRegression":
        target = np.asarray(target, dtype=float)
        classes, counts = np.unique(target, return_counts=True)
        if set(classes) != {0.0, 1.0}:
            raise ValueError("Training target must contain both 0 and 1")
        if self.l2_strength < 0 or self.learning_rate <= 0 or self.max_iterations <= 0:
            raise ValueError("Invalid logistic-regression optimization settings")

        design = np.column_stack([np.ones(len(features)), features])
        class_weight = {label: len(target) / (2.0 * count) for label, count in zip(classes, counts)}
        sample_weight = np.array([class_weight[value] for value in target])
        weight_total = sample_weight.sum()
        coefficients = np.zeros(design.shape[1], dtype=float)
        previous_loss = np.inf
        converged = False

        for iteration in range(self.max_iterations):
            probability = self._sigmoid(design @ coefficients)
            gradient = design.T @ (sample_weight * (probability - target)) / weight_total
            gradient[1:] += self.l2_strength * coefficients[1:] / design.shape[1]
            coefficients -= self.learning_rate * gradient

            if iteration % 20 == 0 or iteration == self.max_iterations - 1:
                logits = design @ coefficients
                loss = float(
                    np.average(np.logaddexp(0.0, logits) - target * logits, weights=sample_weight)
                    + 0.5 * self.l2_strength * np.square(coefficients[1:]).sum()
                    / design.shape[1]
                )
                if abs(previous_loss - loss) < self.tolerance:
                    converged = True
                    break
                previous_loss = loss

        self.coefficients_ = coefficients
        self.iterations_ = iteration + 1
        self.training_loss_ = previous_loss
        self.converged_ = converged
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("NumpyLogisticRegression must be fitted before prediction")
        design = np.column_stack([np.ones(len(features)), features])
        return self._sigmoid(design @ self.coefficients_)


@dataclass
class TabularBaseline:
    """Train-only encoder plus regularized logistic regression."""

    l2_strength: float = 0.10
    learning_rate: float = 0.10
    max_iterations: int = 5000
    tolerance: float = 1e-8

    def fit(self, frame: pd.DataFrame) -> "TabularBaseline":
        self.encoder_ = TabularEncoder().fit(frame)
        features = self.encoder_.transform(frame)
        self.model_ = NumpyLogisticRegression(
            l2_strength=self.l2_strength,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        ).fit(features, frame["is_fraud"].to_numpy())
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("TabularBaseline must be fitted before scoring")
        return self.model_.predict_proba(self.encoder_.transform(frame))

    def strongest_coefficients(self, limit: int = 8) -> list[tuple[str, float]]:
        coefficients = self.model_.coefficients_[1:]
        order = np.argsort(np.abs(coefficients))[::-1][:limit]
        return [
            (self.encoder_.feature_names_[index], float(coefficients[index])) for index in order
        ]
