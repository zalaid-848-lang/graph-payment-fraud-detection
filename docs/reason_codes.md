# Investigator explanation and reason-code contract

Day 7 explains the validation-selected LightGBM graph ranking with exact Tree SHAP values.
The complete training split is the explanation background. The sum of the SHAP
values and base value must reconstruct each model raw score within
a strict numerical tolerance.

The displayed risk score comes from a class-balanced classifier and is used for
ranking. It is not presented as a calibrated probability of fraud.

## Reason-code rules

A reason can appear only when:

1. its mapped model features have a positive SHAP contribution for the current
   transaction; and
2. the raw scoring-time feature row satisfies a factual evidence gate learned
   from the training period where a threshold is required.

| Code | Meaning and evidence gate |
|---|---|
| `MATURED_FRAUD_NEIGHBOUR` | At least one earlier, maturity-delayed, fraud-labelled training transaction shares a card, customer, device, address, or recipient proxy |
| `HIGH_SHARED_ENTITY_ACTIVITY` | Earlier shared-specific-entity transaction count exceeds the training 90th percentile |
| `HIGH_ENTITY_REUSE` | Maximum specific-entity degree exceeds the training 90th percentile |
| `LARGE_CONNECTED_GROUP` | Prior component transaction count exceeds the training 90th percentile |
| `CENTRAL_GRAPH_POSITION` | Specific-entity PageRank exceeds the training 90th percentile |
| `DENSE_ENTITY_NEIGHBOURHOOD` | Specific-entity clustering exceeds the training 90th percentile |
| `UNUSUAL_AMOUNT_PATTERN` | Amount exceeds the training 95th percentile |
| `NEW_ENTITY_PATTERN` | New-entity count is positive and at or above the training 90th percentile |
| `LIMITED_RESOLVED_HISTORY` | Matured-neighbour count is at or below the training median and its SHAP contribution raises risk |
| `TRANSACTION_ATTRIBUTE_PATTERN` | Available product/email-domain/category features contribute positively |
| `COMBINED_MODEL_SIGNAL` | Safe fallback when positive contributions do not meet a more specific factual gate |

Thresholds are fitted on training rows only. Reason codes never inspect validation
or test labels, direct card/device/address identifiers, or future graph state.

## Investigator-facing output

The output contains a transaction reference, ranking score, rank, review-only
action, and up to three positive reasons. Evaluation labels are deliberately
omitted. Evidence values are counts or model inputs available at scoring time;
they are not assertions of criminal intent.

When multiple transactions share the capacity-cutoff score, the artifact records
the size of the tied group and the number selected from it. Stable chronological
transaction order provides reproducibility inside a tie but is not presented as a
model-measured difference in risk.

The only supported action is `INVESTIGATOR_REVIEW`. No reason code authorizes
account blocking, adverse customer action, or designation of a connected component
as a confirmed fraud ring.
