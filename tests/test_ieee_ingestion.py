import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fraud_detection.data.ingest import load_ieee_cis


class IeeeIngestionTests(unittest.TestCase):
    def test_ieee_adapter_joins_identity_and_builds_stable_keys(self) -> None:
        transactions = pd.DataFrame(
            {
                "TransactionID": [1001, 1002],
                "isFraud": [0, 1],
                "TransactionDT": [10, 20],
                "TransactionAmt": [25.0, 90.0],
                "ProductCD": ["W", "C"],
                "card1": [1111, 2222],
                "card2": [100, 200],
                "addr1": [10, 20],
                "addr2": [87, 87],
                "P_emaildomain": ["GMAIL.COM", "yahoo.com"],
                "R_emaildomain": ["merchant.example", "shop.example"],
            }
        )
        identities = pd.DataFrame(
            {
                "TransactionID": [1002],
                "DeviceType": ["mobile"],
                "DeviceInfo": ["device-x"],
                "id_31": ["browser-y"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            temp_path = Path(directory)
            transactions.to_csv(temp_path / "train_transaction.csv", index=False)
            identities.to_csv(temp_path / "train_identity.csv", index=False)
            result = load_ieee_cis(temp_path)

        self.assertEqual(result["transaction_id"].tolist(), ["ieee:1001", "ieee:1002"])
        self.assertEqual(result.loc[0, "payer_email_domain"], "gmail.com")
        self.assertTrue(pd.isna(result.loc[0, "device_id"]))
        self.assertTrue(result.loc[1, "device_id"].startswith("device:"))
        self.assertTrue(result["card_id"].str.startswith("card:").all())
        self.assertTrue(result["customer_id"].str.startswith("customer_proxy:").all())
        self.assertTrue(result["recipient_id"].str.startswith("recipient_proxy:").all())
        self.assertNotIn("isFraud", result.columns)


if __name__ == "__main__":
    unittest.main()
