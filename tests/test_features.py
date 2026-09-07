import unittest

import numpy as np
import pandas as pd

from fraud_detection.data.synthetic import generate_synthetic_transactions
from fraud_detection.features import (
    STRUCTURAL_FEATURE_COLUMNS,
    CausalGraphFeatureBuilder,
)


class CausalGraphFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = generate_synthetic_transactions(120)
        cls.features = CausalGraphFeatureBuilder(pagerank_refresh_batches=10).build(cls.frame)

    def test_output_is_complete_finite_non_negative_and_target_free(self) -> None:
        matrix = self.features[list(STRUCTURAL_FEATURE_COLUMNS)].to_numpy(dtype=float)

        self.assertEqual(len(self.features), len(self.frame))
        self.assertEqual(len(STRUCTURAL_FEATURE_COLUMNS), 26)
        self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue((matrix >= 0).all())
        self.assertNotIn("is_fraud", self.features.columns)

    def test_label_changes_do_not_change_features(self) -> None:
        changed = self.frame.copy()
        changed["is_fraud"] = 1 - changed["is_fraud"]
        changed_features = CausalGraphFeatureBuilder(pagerank_refresh_batches=10).build(changed)

        pd.testing.assert_frame_equal(self.features, changed_features)

    def test_prefix_features_do_not_change_when_future_rows_are_added(self) -> None:
        prefix = self.frame.iloc[:60]
        prefix_features = CausalGraphFeatureBuilder(pagerank_refresh_batches=10).build(prefix)

        pd.testing.assert_frame_equal(
            self.features.iloc[:60].reset_index(drop=True), prefix_features
        )

    def test_same_time_rows_cannot_see_each_other(self) -> None:
        frame = self.frame.iloc[:3].copy()
        frame.loc[:, "device_id"] = "device:batch_shared"
        frame.loc[1, "transaction_time"] = frame.loc[0, "transaction_time"]
        frame.loc[2, "transaction_time"] = frame.loc[0, "transaction_time"] + 1
        features = CausalGraphFeatureBuilder(pagerank_refresh_batches=1).build(frame)

        self.assertEqual(features.loc[0, "historical_device_degree"], 0)
        self.assertEqual(features.loc[1, "historical_device_degree"], 0)
        self.assertEqual(features.loc[2, "historical_device_degree"], 2)

    def test_email_hubs_are_excluded_from_specific_shared_count(self) -> None:
        frame = self.frame.iloc[:3].copy()
        for index in frame.index:
            frame.loc[index, "card_id"] = f"card:unique_{index}"
            frame.loc[index, "customer_id"] = f"customer:unique_{index}"
            frame.loc[index, "device_id"] = f"device:unique_{index}"
            frame.loc[index, "address_id"] = f"address:unique_{index}"
            frame.loc[index, "recipient_id"] = f"recipient:unique_{index}"
        frame.loc[:, "payer_email_domain"] = "common.example"
        frame.loc[:, "recipient_email_domain"] = "merchant.example"
        features = CausalGraphFeatureBuilder(pagerank_refresh_batches=1).build(frame)

        self.assertGreaterEqual(features.loc[2, "graph_shared_transaction_count"], 2)
        self.assertEqual(features.loc[2, "specific_shared_transaction_count"], 0)

    def test_projection_features_become_nonzero_from_history(self) -> None:
        features = CausalGraphFeatureBuilder(pagerank_refresh_batches=1).build(self.frame.iloc[:20])

        self.assertGreater(features["specific_max_pagerank"].max(), 0)
        self.assertGreater(features["specific_max_clustering"].max(), 0)

    def test_pagerank_snapshot_age_is_bounded_by_refresh_interval(self) -> None:
        self.assertLess(self.features["pagerank_snapshot_age_batches"].max(), 10)


if __name__ == "__main__":
    unittest.main()
