import math
import unittest

import numpy as np

from fraud_detection.data.split import assign_temporal_splits
from fraud_detection.data.synthetic import generate_synthetic_transactions
from fraud_detection.explainability import (
    InvestigatorReasonBuilder,
    ShapAttribution,
    TreeShapAttributor,
    global_shap_importance,
)
from fraud_detection.features import (
    CausalGraphFeatureBuilder,
    MaturedLabelFeatureBuilder,
    attach_graph_features,
)
from fraud_detection.model import LightGBMGraphModel


class Day7ExplainabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        frame = assign_temporal_splits(generate_synthetic_transactions(240))
        structural = CausalGraphFeatureBuilder(pagerank_refresh_batches=10).build(frame)
        labels = MaturedLabelFeatureBuilder().build(frame)
        combined = attach_graph_features(frame, structural, labels)
        cls.train = combined[combined["split"] == "train"].copy()
        cls.validation = combined[combined["split"] == "validation"].copy()
        cls.test = combined[combined["split"] == "test"].copy()
        cls.model = LightGBMGraphModel(
            n_estimators=80,
            early_stopping_rounds=10,
            min_child_samples=5,
        ).fit(cls.train, cls.validation)
        cls.attribution = TreeShapAttributor().fit(cls.model, cls.train).explain(cls.test)

    def test_shap_reconstructs_lightgbm_raw_scores(self) -> None:
        self.assertEqual(self.attribution.audit["status"], "PASS")
        self.assertLessEqual(self.attribution.audit["maximum_raw_score_reconstruction_error"], 1e-8)
        self.assertTrue(np.isfinite(self.attribution.values).all())

    def test_shap_attribution_is_deterministic(self) -> None:
        repeated = TreeShapAttributor().fit(self.model, self.train).explain(self.test)

        np.testing.assert_array_equal(self.attribution.values, repeated.values)
        np.testing.assert_array_equal(self.attribution.base_values, repeated.base_values)

    def test_reason_output_is_review_only_and_target_free(self) -> None:
        result = (
            InvestigatorReasonBuilder()
            .fit(self.train)
            .build(
                self.test,
                self.attribution,
                capacity_fraction=0.05,
                max_reasons=3,
            )
        )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["review_count"], math.ceil(len(self.test) * 0.05))
        self.assertTrue(all(result["gates"].values()))
        self.assertTrue(all(len(alert["reasons"]) <= 3 for alert in result["alerts"]))
        self.assertNotIn("block", str(result["alerts"]).lower())
        tie_audit = result["capacity_tie_audit"]
        self.assertGreaterEqual(tie_audit["rows_at_cutoff"], tie_audit["selected_from_cutoff_tie"])

    def test_fraud_neighbour_reason_has_factual_evidence(self) -> None:
        eligible_index = int(
            np.flatnonzero(self.test["matured_neighbor_fraud_count"].to_numpy() > 0)[0]
        )
        values = np.zeros_like(self.attribution.values)
        fraud_feature = self.attribution.feature_names.index(
            "log1p_standardized_matured_neighbor_fraud_ratio"
        )
        values[eligible_index, fraud_feature] = 1.0
        scores = np.zeros(len(self.test))
        scores[eligible_index] = 1.0
        controlled = ShapAttribution(
            values=values,
            base_values=np.zeros(len(self.test)),
            scores=scores,
            feature_names=self.attribution.feature_names,
            audit={"status": "PASS"},
        )
        result = (
            InvestigatorReasonBuilder()
            .fit(self.train)
            .build(
                self.test,
                controlled,
                capacity_fraction=1 / len(self.test),
            )
        )
        matching = [
            reason
            for alert in result["alerts"]
            for reason in alert["reasons"]
            if reason["code"] == "MATURED_FRAUD_NEIGHBOUR"
        ]

        self.assertEqual(len(matching), 1)
        self.assertGreater(matching[0]["evidence"]["fraud_labelled_neighbours"], 0)

    def test_global_importance_is_valid(self) -> None:
        importance = global_shap_importance(self.attribution, limit=10)

        self.assertEqual(len(importance), 10)
        self.assertTrue(all(value >= 0 for _, value in importance))
        self.assertLessEqual(sum(value for _, value in importance), 1.0 + 1e-12)


if __name__ == "__main__":
    unittest.main()
