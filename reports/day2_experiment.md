# Day 2 experiment report

## Executive summary

This report establishes two pre-graph benchmarks: a transparent rules score and a regularized scikit-learn logistic-regression score. All fitting uses only the training period; validation selects operating thresholds; test labels are used only for final evaluation.

> **Development-data warning:** These results use deliberately enriched synthetic fraud patterns. They verify code and evaluation logic; they are not evidence of real-world fraud performance.

## Data quality

**Quality status:** PASS

Rows: 500; columns: 14; fraud labels: 49 (9.80%).

| Split | Rows | Fraud labels | Fraud rate | Time range (elapsed seconds) |
|---|---:|---:|---:|---:|
| train | 300 | 29 | 9.67% | 2,989–1,708,216 |
| purge_train_validation | 16 | 1 | 6.25% | 1,713,016–1,790,836 |
| validation | 84 | 8 | 9.52% | 1,798,055–2,271,038 |
| purge_validation_test | 16 | 3 | 18.75% | 2,280,024–2,356,317 |
| test | 84 | 8 | 9.52% | 2,365,114–2,847,050 |

Quality gates:

- PASS — `transaction_ids_unique`
- PASS — `labels_binary`
- PASS — `amounts_non_negative`
- PASS — `time_non_decreasing`
- PASS — `every_row_has_entity`
- PASS — `train_precedes_validation`
- PASS — `validation_precedes_test`

## Experimental design

- Investigation capacity: `5.00%` of transactions
- Rules: high/very-high amount plus device or recipient unseen in training
- Tabular pipeline: train-fitted encoding, class-balanced scikit-learn logistic regression, and deterministic configuration
- Tabular features: log amount, entity-availability flags, product code, and payer/recipient email-domain categories
- Direct transaction/entity identifiers and target aliases are excluded
- Purge-window rows are excluded from training, threshold selection, and testing

## Untouched test-period results

### Rules baseline

- PR-AUC: `0.2976`
- Precision at capacity: `60.00%`
- Recall at capacity: `37.50%`
- False-positive rate at capacity: `2.63%`
- Recall in top 1%: `12.50%`
- Recall in top 5%: `37.50%`
- Test alerts at validation-selected threshold: `8` (`9.52%` of test rows)

### Tabular logistic baseline

- PR-AUC: `0.4989`
- Precision at capacity: `60.00%`
- Recall at capacity: `37.50%`
- False-positive rate at capacity: `2.63%`
- Recall in top 1%: `12.50%`
- Recall in top 5%: `37.50%`
- Test alerts at validation-selected threshold: `3` (`3.57%` of test rows)

The tabular model improves overall PR-AUC, but at the exact investigation-capacity cutoff it finds `3` fraud labels versus `3` for the rules. This is why model selection must consider the operating point rather than PR-AUC alone.

## Investigator-volume comparison

At their separate validation-selected thresholds, the tabular model emits `62.50%` fewer test alerts than the rules baseline. Its recall at that point is `25.00%`, versus `50.00%` for the rules, so the raw volume change must not be presented as an efficiency gain.

At matched recall (`50.00%`), the tabular ranking needs `8` alerts versus `8` rule alerts: an alert-volume reduction of `0.00%`. A negative value means the model needs more reviews to match the rules.
These are project-scenario measurements, not a bank policy recommendation.

## Strongest tabular coefficients

Positive coefficients increase the model score and negative coefficients decrease it. They are associations in this development sample, not causal explanations.

| Feature | Coefficient |
|---|---:|
| `log_amount_standardized` | 1.1054 |
| `product_code=R` | -0.7607 |
| `product_code=W` | 0.6512 |
| `recipient_email_domain=services.example` | -0.6435 |
| `recipient_email_domain=market.example` | 0.4066 |
| `recipient_email_domain=merchant.example` | 0.2369 |
| `payer_email_domain=yahoo.com` | 0.1944 |
| `payer_email_domain=gmail.com` | -0.1765 |

## Day 2 conclusion

The project now has a fair, reproducible pre-graph benchmark. Day 3 will construct the historical entity-link graph and test that missing values never create false links. Only after those integrity checks will graph features be compared against these baselines.

## Limitations

- Labels describe transactions, not verified fraud rings.
- Customer and recipient fields are proxies for IEEE-CIS data.
- Shared or previously unseen entities can be legitimate.
- Rules and thresholds are project scenarios, not ICICI Bank practices.
- Investigator review remains required; no automated blocking action is recommended.
