import unittest

import numpy as np

from fraud_detection.data.split import assign_temporal_splits
from fraud_detection.data.synthetic import generate_synthetic_transactions
from fraud_detection.evaluation import average_precision
from fraud_detection.model import TabularBaseline, audit_feature_names
from fraud_detection.rules import RulesBaseline


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        frame = assign_temporal_splits(generate_synthetic_transactions(500))
        self.train = frame[frame["split"] == "train"].copy()
        self.test = frame[frame["split"] == "test"].copy()

    def test_rules_fit_does_not_read_labels(self) -> None:
        original = RulesBaseline().fit(self.train)
        changed = self.train.copy()
        changed["is_fraud"] = 1 - changed["is_fraud"]
        changed_labels = RulesBaseline().fit(changed)
        np.testing.assert_array_equal(original.score(self.test), changed_labels.score(self.test))

    def test_tabular_model_scores_are_finite_and_in_range(self) -> None:
        model = TabularBaseline(max_iterations=1000).fit(self.train)
        scores = model.score(self.test)

        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        self.assertGreater(average_precision(self.test["is_fraud"].to_numpy(), scores), 0.0)
        self.assertFalse(any(name.endswith("_id") for name in model.encoder_.feature_names_))

    def test_feature_audit_rejects_targets_and_identifiers(self) -> None:
        with self.assertRaisesRegex(ValueError, "Forbidden model features"):
            audit_feature_names(["amount", "is_fraud", "device_id"])


if __name__ == "__main__":
    unittest.main()

