import unittest

import numpy as np

from fraud_detection.data.split import assign_temporal_splits
from fraud_detection.data.synthetic import generate_synthetic_transactions
from fraud_detection.error_analysis import build_error_analysis
from fraud_detection.features import (
    CausalGraphFeatureBuilder,
    MaturedLabelFeatureBuilder,
    attach_graph_features,
)
from fraud_detection.model import LightGBMGraphModel


class Day6BoostingAndErrorAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        frame = assign_temporal_splits(generate_synthetic_transactions(240))
        structural = CausalGraphFeatureBuilder(pagerank_refresh_batches=10).build(frame)
        label_history = MaturedLabelFeatureBuilder().build(frame)
        cls.combined = attach_graph_features(frame, structural, label_history)
        cls.train = cls.combined[cls.combined["split"] == "train"].copy()
        cls.validation = cls.combined[cls.combined["split"] == "validation"].copy()
        cls.test = cls.combined[cls.combined["split"] == "test"].copy()

    def test_lightgbm_scores_and_importance_are_valid(self) -> None:
        model = LightGBMGraphModel(
            n_estimators=80,
            early_stopping_rounds=10,
            min_child_samples=5,
        ).fit(self.train, self.validation)
        scores = model.score(self.test)
        importance = model.feature_importance(limit=8)

        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        self.assertGreaterEqual(model.best_iteration_, 1)
        self.assertLessEqual(model.best_iteration_, 80)
        self.assertEqual(len(importance), 8)
        self.assertLessEqual(sum(value for _, value in importance), 1.0 + 1e-12)

    def test_lightgbm_training_is_deterministic(self) -> None:
        settings = {
            "n_estimators": 50,
            "early_stopping_rounds": 8,
            "min_child_samples": 5,
        }
        first = LightGBMGraphModel(**settings).fit(self.train, self.validation)
        second = LightGBMGraphModel(**settings).fit(self.train, self.validation)

        np.testing.assert_array_equal(first.score(self.test), second.score(self.test))
        self.assertEqual(first.best_iteration_, second.best_iteration_)

    def test_error_slice_families_are_exhaustive(self) -> None:
        scores = np.linspace(0.0, 1.0, len(self.test))
        analysis = build_error_analysis(
            self.train,
            self.test,
            scores,
            capacity_fraction=0.05,
        )

        self.assertEqual(analysis["status"], "PASS")
        self.assertTrue(all(analysis["gates"].values()))
        for slices in analysis["families"].values():
            self.assertEqual(sum(value["rows"] for value in slices.values()), len(self.test))

    def test_error_analysis_rejects_misaligned_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "match the test row count"):
            build_error_analysis(
                self.train,
                self.test,
                np.ones(len(self.test) - 1),
                capacity_fraction=0.05,
            )


if __name__ == "__main__":
    unittest.main()
