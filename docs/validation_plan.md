# Leakage-safe validation plan

## Objective

Estimate how the system would prioritize newly arriving transactions when only past transactions, past relationships, and legitimately available historical outcomes are known.

## Primary holdout

Transactions are sorted by `transaction_time`, with `transaction_id` as a deterministic tie-breaker. Default boundaries are:

- earliest 60%: training;
- next 20%: validation;
- latest 20%: test;
- 24-hour purge window immediately after the training period and immediately after the validation period.

Rows in purge windows are retained and named in the prepared data but are not used for fitting, threshold selection, or final metrics. This creates a visible buffer around boundaries and makes accidental reuse testable.

If many transactions share the exact boundary time, all transactions at that time remain on the earlier side. The manifest records actual counts after purging.

## Feature availability rules

For a transaction at elapsed time *t*:

1. Transaction attributes must be available at scoring time.
2. Structural graph features may use only edges from transactions with time less than *t*. The current transaction's own entity links may be used to query that historical state, but future and later same-time transactions may not be used.
3. Degree, component size, PageRank, clustering coefficient, and shared-entity counts are snapshots of historical graph state—not values from a graph built over the full dataset.
4. Label-derived features, including neighbouring-fraud ratios, use only matured historical labels. A configurable label-delay assumption will be applied. Validation/test labels never enter features for validation/test rows.
5. Imputers, encoders, scaling, feature selection, class weighting, calibration, and model fitting are learned on training data only.
6. Alert thresholds and investigation capacity are selected on validation data only. Test data is evaluated once after decisions are fixed.

## Model comparison

All three approaches receive the same eligible rows and chronological holdout:

1. Rules-only detection
2. Tabular model
3. Tabular model plus graph features

The primary ranking metric is PR-AUC because fraud is rare. Operational metrics are precision and recall at a stated daily investigation capacity, recall among the top 1% and top 5% of ranked transactions, false-positive rate, and alert-volume reduction relative to the rules baseline.

## Additional checks

- Report fraud prevalence and date/time span by split.
- Check entity overlap across splits; overlap is expected and is the point of graph history, but identifiers are never direct model inputs.
- Assert maximum training time is earlier than minimum validation time, and likewise for validation/test, after purge rows are excluded.
- Run a feature audit that fails if target aliases (`isFraud`, `is_fraud`) or full-data graph statistics enter a model matrix.
- Use temporal cross-validation inside the training/validation period for consequential tuning if dataset size permits.

## Known realism limits

IEEE-CIS does not publish the operational label delay, investigation capacity, verified ring membership, or true customer/recipient identifiers. Those assumptions will therefore be explicit scenario parameters rather than claims about ICICI Bank or any other institution.

