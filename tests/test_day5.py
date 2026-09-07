import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fraud_detection.data.split import assign_temporal_splits
from fraud_detection.data.synthetic import generate_synthetic_transactions
from fraud_detection.features import (
    LABEL_HISTORY_FEATURE_COLUMNS,
    MODEL_GRAPH_FEATURE_COLUMNS,
    CausalGraphFeatureBuilder,
    MaturedLabelFeatureBuilder,
    attach_graph_features,
)
from fraud_detection.model import GraphFeatureEncoder, TabularGraphBaseline


class Day5FeatureAndModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = assign_temporal_splits(generate_synthetic_transactions(180))
        cls.structural = CausalGraphFeatureBuilder(pagerank_refresh_batches=10).build(cls.frame)
        cls.label_history = MaturedLabelFeatureBuilder(label_maturity_seconds=86_400).build(
            cls.frame
        )
        cls.combined = attach_graph_features(cls.frame, cls.structural, cls.label_history)

    def test_validation_and_test_labels_never_change_label_history_features(self) -> None:
        changed = self.frame.copy()
        non_train = changed["split"] != "train"
        changed.loc[non_train, "is_fraud"] = 1 - changed.loc[non_train, "is_fraud"]
        changed_features = MaturedLabelFeatureBuilder(label_maturity_seconds=86_400).build(changed)

        pd.testing.assert_frame_equal(self.label_history, changed_features)

    def test_label_delay_and_same_time_batch_are_respected(self) -> None:
        frame = self.frame.iloc[:4].copy()
        frame.loc[:, "split"] = "train"
        frame.loc[:, "device_id"] = "device:shared"
        frame["is_fraud"] = np.array([1, 0, 0, 0], dtype=np.int8)
        frame.loc[:, "transaction_time"] = [100, 100, 149, 150]
        features = MaturedLabelFeatureBuilder(label_maturity_seconds=50).build(frame)

        self.assertEqual(features.loc[0, "matured_neighbor_label_count"], 0)
        self.assertEqual(features.loc[1, "matured_neighbor_label_count"], 0)
        self.assertEqual(features.loc[2, "matured_neighbor_label_count"], 0)
        self.assertEqual(features.loc[3, "matured_neighbor_label_count"], 2)
        self.assertEqual(features.loc[3, "matured_neighbor_fraud_count"], 1)
        self.assertEqual(features.loc[3, "matured_neighbor_fraud_ratio"], 0.5)

    def test_feature_attachment_requires_exact_ids_and_times(self) -> None:
        attached = attach_graph_features(self.frame, self.structural, self.label_history)
        self.assertEqual(len(attached), len(self.frame))
        self.assertFalse(attached[list(MODEL_GRAPH_FEATURE_COLUMNS)].isna().any().any())

        shifted = self.structural.copy()
        shifted.loc[0, "transaction_time"] += 1
        with self.assertRaisesRegex(ValueError, "times do not match"):
            attach_graph_features(self.frame, shifted, self.label_history)

    def test_feature_attachment_rejects_target_columns(self) -> None:
        contaminated = self.structural.assign(is_fraud=0)
        with self.assertRaisesRegex(ValueError, "Target columns"):
            attach_graph_features(self.frame, contaminated, self.label_history)

    def test_graph_scaling_statistics_are_learned_from_training_only(self) -> None:
        train = self.combined[self.combined["split"] == "train"]
        encoder = GraphFeatureEncoder().fit(train)
        expected = np.log1p(train[list(MODEL_GRAPH_FEATURE_COLUMNS)].to_numpy(dtype=float)).mean(
            axis=0
        )

        np.testing.assert_allclose(encoder.means_, expected)
        self.assertEqual(len(encoder.feature_names_), len(MODEL_GRAPH_FEATURE_COLUMNS))

    def test_combined_model_returns_finite_probabilities(self) -> None:
        train = self.combined[self.combined["split"] == "train"]
        test = self.combined[self.combined["split"] == "test"]
        model = TabularGraphBaseline(max_iterations=2000).fit(train)
        scores = model.score(test)

        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        self.assertIsInstance(model.classifier_, LogisticRegression)
        self.assertTrue(model.converged_)
        self.assertEqual(
            len(model.encoder_.graph_encoder_.feature_names_),
            len(MODEL_GRAPH_FEATURE_COLUMNS),
        )
        self.assertEqual(len(LABEL_HISTORY_FEATURE_COLUMNS), 3)


if __name__ == "__main__":
    unittest.main()
