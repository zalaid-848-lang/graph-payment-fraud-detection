# Error-analysis contract

Day 6 error analysis is performed only after the model, validation-selected
boosting iteration, and investigation capacity are fixed. It describes the held-out
test result and does not feed back into model tuning.

## Slice families

- **Amount band:** at or below the training median, between the training median
  and 90th percentile, and above the training 90th percentile. No test-derived
  amount boundary is used.
- **Specific connectivity:** at or below the training median, between the training
  median and 90th percentile, or above the training 90th percentile for earlier
  transactions sharing a card, customer, device, address, or recipient proxy.
- **Specific entity history:** all specific entities are new, or at least one has
  appeared previously.
- **Matured label history:** no matured neighbour, matured neighbours with no fraud
  label, or at least one matured fraud-labelled neighbour.
- **Product code:** one slice per anonymized product category present in the test
  period.

Amount and connectivity boundaries are derived from the training period only.
Every family must independently account for all test rows, all globally selected
alerts, and all test fraud labels. The pipeline fails if any family is incomplete
or overlapping.

## Reported counts

Each slice reports rows, fraud labels, alerts, true positives, false positives,
false negatives, precision, and within-slice recall. The fixed-capacity selection
is global; the system does not allocate a separate quota to each slice.

Case-level examples are written only to ignored experiment artifacts. The tracked
report contains aggregate results so that later use with IEEE-CIS data does not
encourage committing transaction-level records.

These diagnostics reveal where a ranking makes mistakes; they do not prove bias,
causality, criminal coordination, or readiness for production use.
