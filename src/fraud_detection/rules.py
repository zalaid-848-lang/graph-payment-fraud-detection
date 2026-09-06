"""A deliberately simple, investigator-readable rules baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RulesBaseline:
    """Score amount outliers and entities unseen in the training period.

    Thresholds and reference entity sets are fitted without reading the target.
    The rules are benchmarks, not claims about real bank policy.
    """

    high_amount_quantile: float = 0.95
    very_high_amount_quantile: float = 0.99

    def fit(self, frame: pd.DataFrame) -> RulesBaseline:
        if not 0 < self.high_amount_quantile < self.very_high_amount_quantile < 1:
            raise ValueError("Amount quantiles must satisfy 0 < high < very_high < 1")
        self.high_amount_threshold_ = float(frame["amount"].quantile(self.high_amount_quantile))
        self.very_high_amount_threshold_ = float(
            frame["amount"].quantile(self.very_high_amount_quantile)
        )
        self.known_devices_ = set(frame["device_id"].dropna().astype(str))
        self.known_recipients_ = set(frame["recipient_id"].dropna().astype(str))
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "high_amount_threshold_"):
            raise RuntimeError("RulesBaseline must be fitted before scoring")

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Return an additive risk score; higher scores are reviewed first."""
        self._check_fitted()
        high_amount = frame["amount"].ge(self.high_amount_threshold_).to_numpy(dtype=float)
        very_high_amount = frame["amount"].ge(
            self.very_high_amount_threshold_
        ).to_numpy(dtype=float)
        device = frame["device_id"].astype("string")
        recipient = frame["recipient_id"].astype("string")
        unseen_device = (device.notna() & ~device.isin(self.known_devices_)).to_numpy(dtype=float)
        unseen_recipient = (
            recipient.notna() & ~recipient.isin(self.known_recipients_)
        ).to_numpy(dtype=float)
        return high_amount + very_high_amount + unseen_device + unseen_recipient

    def reasons(self, frame: pd.DataFrame) -> list[str]:
        """Return semicolon-delimited reason codes for investigator display."""
        self._check_fitted()
        results: list[str] = []
        for row in frame.itertuples(index=False):
            reasons: list[str] = []
            if row.amount >= self.very_high_amount_threshold_:
                reasons.append("very_high_amount")
            elif row.amount >= self.high_amount_threshold_:
                reasons.append("high_amount")
            if pd.notna(row.device_id) and str(row.device_id) not in self.known_devices_:
                reasons.append("device_unseen_in_training")
            if pd.notna(row.recipient_id) and str(row.recipient_id) not in self.known_recipients_:
                reasons.append("recipient_unseen_in_training")
            results.append(";".join(reasons) if reasons else "no_rule_triggered")
        return results
