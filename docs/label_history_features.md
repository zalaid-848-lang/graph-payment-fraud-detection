# Leakage-safe label-history features

Neighbouring-fraud features require labels and therefore follow a stricter contract
than structural graph features. The Day 5 implementation uses only matured labels
from the training split. Validation, test, and purge labels are never eligible.

For a transaction at time *t*, a training transaction at time *s* can contribute
only when both conditions hold:

1. *s* is strictly earlier than *t*, so another transaction in the same timestamp
   batch cannot reveal its outcome;
2. *s* plus the configured label-maturity delay is at or before *t*.

The default delay is 86,400 seconds (24 hours). This is an explicit development
scenario, not a claim about an issuer, bank, merchant, or the IEEE-CIS labelling
process.

## Features

| Feature | Definition |
|---|---|
| `matured_neighbor_label_count` | Distinct matured training transactions sharing a card, customer, device, address, or recipient proxy |
| `matured_neighbor_fraud_count` | Those neighbouring transactions with a fraud label |
| `matured_neighbor_fraud_ratio` | Fraud count divided by labelled-neighbour count; zero when no matured neighbour exists |

Email domains are excluded from this lookup because common providers and merchant
domains can join unrelated activity. A historical transaction that shares multiple
entities with the current transaction is counted once.

## Evaluation boundary

The training-period feature rows use only earlier matured training labels. During
validation and test scoring, the eligible label pool remains restricted to the
training split. This is conservative compared with a live system that may receive
newly resolved outcomes, but it prevents validation or test labels from feeding
back into their own evaluation period.
