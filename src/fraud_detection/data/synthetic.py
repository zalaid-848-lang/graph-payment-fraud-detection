"""Deterministic development data with a connected suspicious pattern."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.data.schema import validate_transactions


def generate_synthetic_transactions(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create a small, reproducible transaction table for pipeline development.

    The timeline contains a deliberately shared device/address/recipient pattern.
    Labels are synthetic and provide no evidence about real fraud behavior.
    """
    if n_rows < 100:
        raise ValueError("n_rows must be at least 100 so every temporal split is useful")

    rng = np.random.default_rng(seed)
    event_gaps = rng.integers(2_400, 9_000, size=n_rows, endpoint=False)
    transaction_time = np.cumsum(event_gaps).astype("int64")
    customer_number = rng.integers(1, max(30, n_rows // 4), size=n_rows)
    card_number = customer_number * 2 + rng.integers(0, 2, size=n_rows)

    frame = pd.DataFrame(
        {
            "transaction_id": [f"syn_{index:06d}" for index in range(n_rows)],
            "transaction_time": transaction_time,
            "amount": np.round(rng.lognormal(mean=3.7, sigma=0.85, size=n_rows), 2),
            "product_code": rng.choice(["W", "C", "R", "H", "S"], size=n_rows),
            "card_id": [f"card:{value:04d}" for value in card_number],
            "customer_id": [f"customer:{value:04d}" for value in customer_number],
            "device_id": [f"device:{value:03d}" for value in rng.integers(1, 90, size=n_rows)],
            "address_id": [f"address:{value:03d}" for value in rng.integers(1, 120, size=n_rows)],
            "payer_email_domain": rng.choice(
                ["gmail.com", "yahoo.com", "outlook.com", "example.net"], size=n_rows
            ),
            "recipient_email_domain": rng.choice(
                ["merchant.example", "market.example", "services.example"], size=n_rows
            ),
            "recipient_id": [
                f"recipient:{value:03d}" for value in rng.integers(1, 45, size=n_rows)
            ],
            "is_fraud": rng.binomial(1, 0.025, size=n_rows),
            "source": "synthetic",
        }
    )

    ring_size = max(18, n_rows // 14)
    ring_indices = np.linspace(n_rows // 8, n_rows - 2, ring_size, dtype=int)
    ring_cards = np.array(["card:ring_a", "card:ring_b", "card:ring_c"])
    frame.loc[ring_indices, "card_id"] = np.resize(ring_cards, ring_size)
    frame.loc[ring_indices, "customer_id"] = [
        f"customer:ring_{index % 5}" for index in range(ring_size)
    ]
    frame.loc[ring_indices, "device_id"] = "device:ring_shared"
    frame.loc[ring_indices, "address_id"] = "address:ring_shared"
    frame.loc[ring_indices, "recipient_id"] = "recipient:ring_shared"
    frame.loc[ring_indices, "recipient_email_domain"] = "ring-merchant.example"
    frame.loc[ring_indices, "amount"] = np.round(
        rng.uniform(350.0, 1_250.0, size=ring_size), 2
    )
    frame.loc[ring_indices, "is_fraud"] = 1

    return validate_transactions(frame)
