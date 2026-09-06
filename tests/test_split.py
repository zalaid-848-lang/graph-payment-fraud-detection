import unittest

from fraud_detection.data.split import assign_temporal_splits
from fraud_detection.data.synthetic import generate_synthetic_transactions


class TemporalSplitTests(unittest.TestCase):
    def test_temporal_splits_are_ordered_and_purged(self) -> None:
        purge_seconds = 86_400
        result = assign_temporal_splits(
            generate_synthetic_transactions(500), purge_seconds=purge_seconds
        )

        train = result[result["split"] == "train"]
        validation = result[result["split"] == "validation"]
        test = result[result["split"] == "test"]

        self.assertTrue({"train", "validation", "test"}.issubset(set(result["split"])))
        self.assertLess(
            train["transaction_time"].max() + purge_seconds,
            validation["transaction_time"].min(),
        )
        self.assertLess(
            validation["transaction_time"].max() + purge_seconds,
            test["transaction_time"].min(),
        )

    def test_same_time_stays_on_earlier_side(self) -> None:
        frame = generate_synthetic_transactions(100)
        boundary_time = frame.loc[59, "transaction_time"]
        frame.loc[60, "transaction_time"] = boundary_time
        result = assign_temporal_splits(frame, purge_seconds=0)

        at_boundary = result[result["transaction_time"] == boundary_time]
        self.assertTrue(at_boundary["split"].eq("train").all())


if __name__ == "__main__":
    unittest.main()
