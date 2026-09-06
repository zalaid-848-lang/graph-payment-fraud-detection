import unittest

import pandas as pd

from fraud_detection.data.schema import DataValidationError, validate_transactions
from fraud_detection.data.synthetic import generate_synthetic_transactions


class SchemaTests(unittest.TestCase):
    def test_synthetic_generation_is_deterministic_and_contains_shared_pattern(self) -> None:
        first = generate_synthetic_transactions(200, seed=42)
        second = generate_synthetic_transactions(200, seed=42)
        pd.testing.assert_frame_equal(first, second)

        suspicious = first[first["device_id"] == "device:ring_shared"]
        self.assertGreaterEqual(len(suspicious), 18)
        self.assertEqual(suspicious["card_id"].nunique(), 3)
        self.assertTrue(suspicious["is_fraud"].eq(1).all())

    def test_schema_rejects_duplicate_transaction_ids(self) -> None:
        frame = generate_synthetic_transactions(100)
        frame.loc[1, "transaction_id"] = frame.loc[0, "transaction_id"]

        with self.assertRaisesRegex(DataValidationError, "must be unique"):
            validate_transactions(frame)

    def test_schema_rejects_non_binary_labels(self) -> None:
        frame = generate_synthetic_transactions(100)
        frame.loc[0, "is_fraud"] = 2

        with self.assertRaisesRegex(DataValidationError, "only 0 or 1"):
            validate_transactions(frame)


if __name__ == "__main__":
    unittest.main()
