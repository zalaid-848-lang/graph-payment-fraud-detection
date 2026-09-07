"""Fidelity-checked SHAP attribution and investigator-facing reason codes."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap

from fraud_detection.model import LightGBMGraphModel, TabularGraphBaseline


@dataclass
class ShapAttribution:
    values: np.ndarray
    base_values: np.ndarray
    scores: np.ndarray
    feature_names: list[str]
    audit: dict[str, Any]


class LinearShapAttributor:
    """Explain logistic log-odds using a training-only SHAP background."""

    def fit(
        self,
        model: TabularGraphBaseline,
        background: pd.DataFrame,
    ) -> LinearShapAttributor:
        if not hasattr(model, "classifier_"):
            raise ValueError("The graph-logistic model must be fitted before explanation")
        self.model_ = model
        self.feature_names_ = list(model.encoder_.feature_names_)
        background_matrix = model.encoder_.transform(background)
        masker = shap.maskers.Independent(background_matrix, max_samples=len(background_matrix))
        self.explainer_ = shap.LinearExplainer(model.classifier_, masker)
        self.background_rows_ = len(background)
        return self

    def explain(self, frame: pd.DataFrame) -> ShapAttribution:
        if not hasattr(self, "explainer_"):
            raise RuntimeError("LinearShapAttributor must be fitted before explain")
        matrix = self.model_.encoder_.transform(frame)
        explanation = self.explainer_(matrix)
        values = np.asarray(explanation.values, dtype=float)
        base_values = np.asarray(explanation.base_values, dtype=float)
        if base_values.ndim == 0:
            base_values = np.repeat(base_values, len(frame))
        scores = self.model_.score(frame)
        clipped_scores = np.clip(scores, 1e-12, 1 - 1e-12)
        model_log_odds = np.log(clipped_scores / (1.0 - clipped_scores))
        reconstructed = base_values + values.sum(axis=1)
        maximum_error = float(np.max(np.abs(reconstructed - model_log_odds)))
        audit = {
            "status": "PASS" if maximum_error <= 1e-8 else "FAIL",
            "background_rows": self.background_rows_,
            "explained_rows": len(frame),
            "feature_count": len(self.feature_names_),
            "maximum_log_odds_reconstruction_error": maximum_error,
            "all_values_finite": bool(np.isfinite(values).all()),
        }
        if audit["status"] != "PASS" or not audit["all_values_finite"]:
            raise RuntimeError("SHAP fidelity audit failed")
        return ShapAttribution(
            values=values,
            base_values=base_values,
            scores=scores,
            feature_names=self.feature_names_,
            audit=audit,
        )


class TreeShapAttributor:
    """Explain LightGBM raw scores using a complete training-only background."""

    def fit(
        self,
        model: LightGBMGraphModel,
        background: pd.DataFrame,
    ) -> TreeShapAttributor:
        if not hasattr(model, "classifier_"):
            raise ValueError("The LightGBM graph model must be fitted before explanation")
        self.model_ = model
        self.feature_names_ = list(model.encoder_.feature_names_)
        background_matrix = model.encoder_.transform(background)
        masker = shap.maskers.Independent(background_matrix, max_samples=len(background_matrix))
        self.explainer_ = shap.TreeExplainer(
            model.classifier_,
            data=masker,
            feature_perturbation="interventional",
            model_output="raw",
        )
        self.background_rows_ = len(background)
        return self

    def explain(self, frame: pd.DataFrame) -> ShapAttribution:
        if not hasattr(self, "explainer_"):
            raise RuntimeError("TreeShapAttributor must be fitted before explain")
        matrix = self.model_.encoder_.transform(frame)
        explanation = self.explainer_(matrix, check_additivity=True)
        values = np.asarray(explanation.values, dtype=float)
        base_values = np.asarray(explanation.base_values, dtype=float)
        if base_values.ndim == 0:
            base_values = np.repeat(base_values, len(frame))
        raw_scores = np.asarray(
            self.model_.classifier_.predict(
                matrix,
                raw_score=True,
                num_iteration=self.model_.best_iteration_,
            ),
            dtype=float,
        )
        reconstructed = base_values + values.sum(axis=1)
        maximum_error = float(np.max(np.abs(reconstructed - raw_scores)))
        audit = {
            "status": "PASS" if maximum_error <= 1e-8 else "FAIL",
            "explanation_type": "interventional_tree_shap_raw_score",
            "background_rows": self.background_rows_,
            "explained_rows": len(frame),
            "feature_count": len(self.feature_names_),
            "maximum_raw_score_reconstruction_error": maximum_error,
            "all_values_finite": bool(np.isfinite(values).all()),
        }
        if audit["status"] != "PASS" or not audit["all_values_finite"]:
            raise RuntimeError("SHAP fidelity audit failed")
        return ShapAttribution(
            values=values,
            base_values=base_values,
            scores=self.model_.score(frame),
            feature_names=self.feature_names_,
            audit=audit,
        )


REASON_CATALOG = {
    "MATURED_FRAUD_NEIGHBOUR": "Previously resolved fraud-labelled neighbour",
    "HIGH_SHARED_ENTITY_ACTIVITY": "High shared-entity activity",
    "HIGH_ENTITY_REUSE": "High historical entity reuse",
    "LARGE_CONNECTED_GROUP": "Large prior connected group",
    "CENTRAL_GRAPH_POSITION": "Central graph position",
    "DENSE_ENTITY_NEIGHBOURHOOD": "Dense entity neighbourhood",
    "UNUSUAL_AMOUNT_PATTERN": "High amount relative to training history",
    "NEW_ENTITY_PATTERN": "Multiple new graph entities",
    "LIMITED_RESOLVED_HISTORY": "Limited resolved-neighbour history",
    "TRANSACTION_ATTRIBUTE_PATTERN": "Model-associated transaction attributes",
    "COMBINED_MODEL_SIGNAL": "Combined model signal",
}


class InvestigatorReasonBuilder:
    """Turn positive SHAP contributions into factual, evidence-gated reasons."""

    def fit(self, train: pd.DataFrame) -> InvestigatorReasonBuilder:
        required = {
            "amount",
            "specific_shared_transaction_count",
            "specific_max_entity_degree",
            "specific_max_component_transactions",
            "specific_max_pagerank",
            "specific_max_clustering",
            "graph_new_entity_count",
            "matured_neighbor_label_count",
        }
        missing = sorted(required - set(train.columns))
        if missing:
            raise ValueError(f"Missing reason-threshold columns: {', '.join(missing)}")
        self.thresholds_ = {
            "amount_p95": float(train["amount"].quantile(0.95)),
            "shared_p90": float(train["specific_shared_transaction_count"].quantile(0.9)),
            "degree_p90": float(train["specific_max_entity_degree"].quantile(0.9)),
            "component_p90": float(train["specific_max_component_transactions"].quantile(0.9)),
            "pagerank_p90": float(train["specific_max_pagerank"].quantile(0.9)),
            "clustering_p90": float(train["specific_max_clustering"].quantile(0.9)),
            "new_entity_p90": float(train["graph_new_entity_count"].quantile(0.9)),
            "label_count_median": float(train["matured_neighbor_label_count"].median()),
        }
        self.training_rows_ = len(train)
        return self

    @staticmethod
    def _positive_sum(
        contributions: np.ndarray,
        feature_names: list[str],
        *,
        exact: tuple[str, ...] = (),
        prefixes: tuple[str, ...] = (),
    ) -> float:
        indices = [
            index
            for index, name in enumerate(feature_names)
            if name in exact or any(name.startswith(prefix) for prefix in prefixes)
        ]
        return float(np.maximum(contributions[indices], 0).sum()) if indices else 0.0

    def _reasons_for_row(
        self,
        row: pd.Series,
        contributions: np.ndarray,
        feature_names: list[str],
        max_reasons: int,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []

        def add(
            condition: bool,
            code: str,
            contribution: float,
            detail: str,
            evidence: dict[str, Any],
        ) -> None:
            if condition and contribution > 0:
                candidates.append(
                    {
                        "code": code,
                        "title": REASON_CATALOG[code],
                        "detail": detail,
                        "positive_log_odds_contribution": contribution,
                        "evidence": evidence,
                    }
                )

        fraud_count = int(row["matured_neighbor_fraud_count"])
        fraud_ratio = float(row["matured_neighbor_fraud_ratio"])
        contribution = self._positive_sum(
            contributions,
            feature_names,
            exact=(
                "log1p_standardized_matured_neighbor_fraud_count",
                "log1p_standardized_matured_neighbor_fraud_ratio",
            ),
        )
        add(
            fraud_count > 0,
            "MATURED_FRAUD_NEIGHBOUR",
            contribution,
            "Shares a specific entity with earlier, matured fraud-labelled training activity.",
            {"fraud_labelled_neighbours": fraud_count, "fraud_neighbour_ratio": fraud_ratio},
        )

        shared_count = int(row["specific_shared_transaction_count"])
        contribution = self._positive_sum(
            contributions,
            feature_names,
            exact=("log1p_standardized_specific_shared_transaction_count",),
        )
        add(
            shared_count > self.thresholds_["shared_p90"],
            "HIGH_SHARED_ENTITY_ACTIVITY",
            contribution,
            "Earlier transactions sharing specific entities exceed the training 90th percentile.",
            {"earlier_shared_transactions": shared_count},
        )

        max_degree = int(row["specific_max_entity_degree"])
        contribution = self._positive_sum(
            contributions,
            feature_names,
            exact=("log1p_standardized_specific_max_entity_degree",),
            prefixes=("log1p_standardized_historical_",),
        )
        add(
            max_degree > self.thresholds_["degree_p90"],
            "HIGH_ENTITY_REUSE",
            contribution,
            "At least one linked specific entity has unusually high historical reuse.",
            {"maximum_specific_entity_degree": max_degree},
        )

        component_size = int(row["specific_max_component_transactions"])
        contribution = self._positive_sum(
            contributions,
            feature_names,
            exact=(
                "log1p_standardized_specific_max_component_transactions",
                "log1p_standardized_specific_max_component_nodes",
            ),
        )
        add(
            component_size > self.thresholds_["component_p90"],
            "LARGE_CONNECTED_GROUP",
            contribution,
            "The transaction touches a prior component larger than the training 90th percentile.",
            {"prior_component_transactions": component_size},
        )

        pagerank = float(row["specific_max_pagerank"])
        contribution = self._positive_sum(
            contributions,
            feature_names,
            exact=(
                "log1p_standardized_specific_max_pagerank",
                "log1p_standardized_specific_mean_pagerank",
            ),
        )
        add(
            pagerank > self.thresholds_["pagerank_p90"],
            "CENTRAL_GRAPH_POSITION",
            contribution,
            "A linked specific entity is above the training 90th percentile for PageRank.",
            {"maximum_specific_pagerank": pagerank},
        )

        clustering = float(row["specific_max_clustering"])
        contribution = self._positive_sum(
            contributions,
            feature_names,
            exact=(
                "log1p_standardized_specific_max_clustering",
                "log1p_standardized_specific_mean_clustering",
            ),
        )
        add(
            clustering > self.thresholds_["clustering_p90"],
            "DENSE_ENTITY_NEIGHBOURHOOD",
            contribution,
            "A linked entity has clustering above the training 90th percentile.",
            {"maximum_specific_clustering": clustering},
        )

        amount = float(row["amount"])
        contribution = self._positive_sum(
            contributions,
            feature_names,
            exact=("log_amount_standardized",),
        )
        add(
            amount > self.thresholds_["amount_p95"],
            "UNUSUAL_AMOUNT_PATTERN",
            contribution,
            "Transaction amount exceeds the training-period 95th percentile.",
            {"amount": amount},
        )

        new_entities = int(row["graph_new_entity_count"])
        contribution = self._positive_sum(
            contributions,
            feature_names,
            exact=("log1p_standardized_graph_new_entity_count",),
        )
        add(
            new_entities > 0 and new_entities >= self.thresholds_["new_entity_p90"],
            "NEW_ENTITY_PATTERN",
            contribution,
            "The transaction contains multiple entities not present in earlier graph history.",
            {"new_graph_entities": new_entities},
        )

        label_count = int(row["matured_neighbor_label_count"])
        contribution = self._positive_sum(
            contributions,
            feature_names,
            exact=("log1p_standardized_matured_neighbor_label_count",),
        )
        add(
            label_count <= self.thresholds_["label_count_median"],
            "LIMITED_RESOLVED_HISTORY",
            contribution,
            "Resolved neighbouring outcomes are limited relative to training history.",
            {"matured_labelled_neighbours": label_count},
        )

        contribution = self._positive_sum(
            contributions,
            feature_names,
            prefixes=(
                "product_code=",
                "payer_email_domain=",
                "recipient_email_domain=",
                "has_",
            ),
        )
        add(
            True,
            "TRANSACTION_ATTRIBUTE_PATTERN",
            contribution,
            "The fitted model associates this combination of available transaction attributes "
            "with higher risk in the training sample.",
            {
                "product_code": str(row["product_code"]),
                "payer_email_domain": str(row["payer_email_domain"]),
                "recipient_email_domain": str(row["recipient_email_domain"]),
            },
        )

        candidates.sort(key=lambda item: (-item["positive_log_odds_contribution"], item["code"]))
        if not candidates:
            positive_total = float(np.maximum(contributions, 0).sum())
            candidates.append(
                {
                    "code": "COMBINED_MODEL_SIGNAL",
                    "title": REASON_CATALOG["COMBINED_MODEL_SIGNAL"],
                    "detail": "The combined scoring pattern raises risk; inspect linked entities "
                    "before making a decision.",
                    "positive_log_odds_contribution": positive_total,
                    "evidence": {},
                }
            )
        return candidates[:max_reasons]

    def build(
        self,
        frame: pd.DataFrame,
        attribution: ShapAttribution,
        *,
        capacity_fraction: float,
        max_reasons: int = 3,
    ) -> dict[str, Any]:
        if not hasattr(self, "thresholds_"):
            raise RuntimeError("InvestigatorReasonBuilder must be fitted before build")
        if len(frame) != len(attribution.scores) or len(frame) != len(attribution.values):
            raise ValueError("Attributions must align exactly with scored rows")
        if not 0 < capacity_fraction <= 1 or max_reasons <= 0:
            raise ValueError("Capacity and max_reasons must be positive")
        review_count = max(1, math.ceil(len(frame) * capacity_fraction))
        order = np.argsort(-attribution.scores, kind="stable")[:review_count]
        cutoff_score = float(attribution.scores[order[-1]])
        rows_above_cutoff = int((attribution.scores > cutoff_score).sum())
        rows_at_cutoff = int((attribution.scores == cutoff_score).sum())
        selected_at_cutoff = review_count - rows_above_cutoff
        alerts = []
        for rank, index in enumerate(order, start=1):
            row = frame.iloc[index]
            alerts.append(
                {
                    "transaction_id": str(row["transaction_id"]),
                    "rank": rank,
                    "risk_score": float(attribution.scores[index]),
                    "recommended_action": "INVESTIGATOR_REVIEW",
                    "reasons": self._reasons_for_row(
                        row,
                        attribution.values[index],
                        attribution.feature_names,
                        max_reasons,
                    ),
                }
            )
        codes = [reason["code"] for alert in alerts for reason in alert["reasons"]]
        gates = {
            "alert_count_matches_capacity": len(alerts) == review_count,
            "transaction_ids_unique": len({item["transaction_id"] for item in alerts})
            == len(alerts),
            "every_alert_has_reason": all(item["reasons"] for item in alerts),
            "reason_codes_known": all(code in REASON_CATALOG for code in codes),
            "positive_contributions_only": all(
                reason["positive_log_odds_contribution"] >= 0
                for alert in alerts
                for reason in alert["reasons"]
            ),
            "target_fields_absent": all(
                not {"is_fraud", "label", "target"}.intersection(item) for item in alerts
            ),
            "review_only_action": all(
                item["recommended_action"] == "INVESTIGATOR_REVIEW" for item in alerts
            ),
        }
        return {
            "status": "PASS" if all(gates.values()) else "FAIL",
            "gates": gates,
            "model": "tabular_graph_lightgbm",
            "capacity_fraction": capacity_fraction,
            "review_count": review_count,
            "max_reasons_per_alert": max_reasons,
            "capacity_tie_audit": {
                "cutoff_score": cutoff_score,
                "rows_above_cutoff": rows_above_cutoff,
                "rows_at_cutoff": rows_at_cutoff,
                "selected_from_cutoff_tie": selected_at_cutoff,
                "tie_break_required": rows_at_cutoff > selected_at_cutoff,
                "tie_break_rule": "stable chronological transaction order",
            },
            "reason_code_counts": dict(sorted(Counter(codes).items())),
            "thresholds_learned_from_training": self.thresholds_,
            "alerts": alerts,
            "disclaimer": (
                "Reasons describe model signals for investigator review. They do not establish "
                "fraud-ring membership and do not authorize automatic blocking."
            ),
        }


def global_shap_importance(
    attribution: ShapAttribution, limit: int = 15
) -> list[tuple[str, float]]:
    """Return normalized mean absolute SHAP importance."""

    if limit <= 0:
        raise ValueError("Global-importance limit must be positive")
    mean_absolute = np.abs(attribution.values).mean(axis=0)
    total = float(mean_absolute.sum())
    normalized = mean_absolute / total if total else mean_absolute
    order = np.argsort(normalized)[::-1][:limit]
    return [(attribution.feature_names[index], float(normalized[index])) for index in order]
