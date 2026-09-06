import unittest

from fraud_detection.data.split import assign_temporal_splits
from fraud_detection.data.synthetic import generate_synthetic_transactions
from fraud_detection.quality import profile_transactions


class QualityTests(unittest.TestCase):
    def test_clean_synthetic_data_passes_all_quality_gates(self) -> None:
        frame = assign_temporal_splits(generate_synthetic_transactions(500))
        profile = profile_transactions(frame)

        self.assertEqual(profile["status"], "PASS")
        self.assertTrue(all(profile["gates"].values()))
        self.assertEqual(profile["row_count"], 500)


if __name__ == "__main__":
    unittest.main()

