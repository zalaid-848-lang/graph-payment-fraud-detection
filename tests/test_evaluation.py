import unittest

import numpy as np

from fraud_detection.evaluation import (
    average_precision,
    evaluate_ranking,
    minimum_alerts_for_recall,
    select_threshold_at_capacity,
    threshold_metrics,
)


class EvaluationTests(unittest.TestCase):
    def test_average_precision_matches_hand_calculation(self) -> None:
        target = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.1])
        self.assertAlmostEqual(average_precision(target, scores), (1.0 + 2.0 / 3.0) / 2.0)

    def test_capacity_metrics_review_exact_number_of_rows(self) -> None:
        result = evaluate_ranking(
            np.array([1, 0, 1, 0]),
            np.array([0.9, 0.8, 0.7, 0.1]),
            capacity_fraction=0.5,
        )
        capacity = result["at_investigation_capacity"]
        self.assertEqual(capacity["alert_count"], 2)
        self.assertEqual(capacity["precision"], 0.5)
        self.assertEqual(capacity["recall"], 0.5)

    def test_threshold_metrics_keep_score_ties_visible(self) -> None:
        target = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.8, 0.8, 0.1])
        threshold = select_threshold_at_capacity(scores, 0.5)
        result = threshold_metrics(target, scores, threshold)
        self.assertEqual(threshold, 0.8)
        self.assertEqual(result["alert_count"], 3)

    def test_minimum_alerts_for_target_recall(self) -> None:
        result = minimum_alerts_for_recall(
            np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1]), 1.0
        )
        self.assertEqual(result["alert_count"], 3)
        self.assertEqual(result["alert_fraction"], 0.75)


if __name__ == "__main__":
    unittest.main()
