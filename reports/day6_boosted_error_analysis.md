# Day 6 boosted-model comparison and error analysis

## Executive summary

A deterministic LightGBM candidate was added to the frozen Day 5 comparison. Its encoder is fitted on training rows, validation labels select the boosting iteration and operating threshold, and test labels are opened only for final metrics and error slices.

> **Development-data warning:** These results use a small, deliberately patterned synthetic dataset. They validate the workflow and are not expected bank performance.

## Untouched test-period comparison

| Model | PR-AUC | Precision @ capacity | Recall @ capacity | FPR @ capacity | Recall top 1% | Recall top 5% |
|---|---:|---:|---:|---:|---:|---:|
| Rules only | 0.2976 | 60.00% | 37.50% | 2.63% | 12.50% | 37.50% |
| Tabular logistic | 0.4989 | 60.00% | 37.50% | 2.63% | 12.50% | 37.50% |
| Tabular + graph logistic | 0.8935 | 100.00% | 62.50% | 0.00% | 12.50% | 62.50% |
| Tabular + graph LightGBM | 0.8869 | 100.00% | 62.50% | 0.00% | 12.50% | 62.50% |

Relative to the graph-logistic ablation, LightGBM reduced PR-AUC by `0.0066` and changed recall at capacity by `+0.00%`.

## LightGBM training contract

- Library: LightGBM `4.7.0`
- Validation-selected best iteration: `3` of `400`
- Encoded feature count: `48`
- Class balancing is learned from training labels only.
- Deterministic, single-threaded CPU settings are used for reproducibility.
- PageRank snapshot age is audit metadata and is excluded from model inputs.

## Error analysis at fixed investigation capacity

The global top-5 review set contains `5` fraud labels and `0` false positives; `3` fraud labels fall outside capacity.

Slices are descriptive post-test diagnostics. They do not change the model, threshold, or capacity. Numeric boundaries come only from the training period.

### Amount Band

| Slice | Rows | Fraud labels | Alerts | False positives | False negatives | Recall within slice |
|---|---:|---:|---:|---:|---:|---:|
| `at_or_below_train_median` | 49 | 1 | 0 | 0 | 1 | 0.00% |
| `train_median_to_p90` | 21 | 1 | 1 | 0 | 0 | 100.00% |
| `above_train_p90` | 14 | 6 | 4 | 0 | 2 | 66.67% |

### Specific Connectivity

| Slice | Rows | Fraud labels | Alerts | False positives | False negatives | Recall within slice |
|---|---:|---:|---:|---:|---:|---:|
| `at_or_below_train_median` | 0 | 0 | 0 | 0 | 0 | n/a |
| `train_median_to_p90` | 4 | 0 | 0 | 0 | 0 | n/a |
| `above_train_p90` | 80 | 8 | 5 | 0 | 3 | 62.50% |

### Specific Entity History

| Slice | Rows | Fraud labels | Alerts | False positives | False negatives | Recall within slice |
|---|---:|---:|---:|---:|---:|---:|
| `all_specific_entities_new` | 0 | 0 | 0 | 0 | 0 | n/a |
| `at_least_one_known_specific_entity` | 84 | 8 | 5 | 0 | 3 | 62.50% |

### Matured Label History

| Slice | Rows | Fraud labels | Alerts | False positives | False negatives | Recall within slice |
|---|---:|---:|---:|---:|---:|---:|
| `no_matured_neighbour` | 0 | 0 | 0 | 0 | 0 | n/a |
| `matured_neighbours_no_fraud` | 52 | 1 | 0 | 0 | 1 | 0.00% |
| `matured_fraud_neighbour` | 32 | 7 | 5 | 0 | 2 | 71.43% |

### Product Code

| Slice | Rows | Fraud labels | Alerts | False positives | False negatives | Recall within slice |
|---|---:|---:|---:|---:|---:|---:|
| `C` | 9 | 2 | 0 | 0 | 2 | 0.00% |
| `H` | 23 | 2 | 1 | 0 | 1 | 50.00% |
| `R` | 22 | 2 | 2 | 0 | 0 | 100.00% |
| `S` | 17 | 0 | 0 | 0 | 0 | n/a |
| `W` | 13 | 2 | 2 | 0 | 0 | 100.00% |

## Temporal accumulation finding

`80` of `84` test transactions exceed the training-period 90th percentile of historical specific connectivity. Graph counts naturally accumulate over time, so later work should test rolling windows or age-normalized features on the real dataset.

## Gain-based feature importance

Gain measures how much fitted splits used a feature. It is neither a causal explanation nor an investigator reason code.

| Feature | Normalized gain |
|---|---:|
| `log1p_standardized_matured_neighbor_fraud_ratio` | 0.6535 |
| `log1p_standardized_specific_mean_pagerank` | 0.0855 |
| `log1p_standardized_specific_max_pagerank` | 0.0600 |
| `log1p_standardized_specific_max_component_nodes` | 0.0436 |
| `log1p_standardized_historical_payer_email_domain_degree` | 0.0347 |
| `log1p_standardized_graph_sum_entity_degree` | 0.0276 |
| `log_amount_standardized` | 0.0250 |
| `product_code=W` | 0.0210 |
| `log1p_standardized_specific_mean_clustering` | 0.0201 |
| `recipient_email_domain=merchant.example` | 0.0119 |
| `log1p_standardized_graph_shared_transaction_count` | 0.0093 |
| `log1p_standardized_specific_max_entity_degree` | 0.0078 |
| `log1p_standardized_historical_device_degree` | 0.0000 |
| `log1p_standardized_historical_address_degree` | 0.0000 |
| `log1p_standardized_matured_neighbor_label_count` | 0.0000 |

## Interpretation boundaries

- Fraud labels apply to transactions, not verified fraud rings.
- Slice differences on this small synthetic test set are unstable.
- Shared entities may represent legitimate households, offices, or infrastructure.
- The 5% capacity and 24-hour label delay are project scenarios, not bank policies.
- Scores prioritize human review and never authorize automatic account blocking.

## Day 7 handoff

Add SHAP-based local explanations and stable investigator reason codes, then verify that every displayed explanation uses information available at scoring time.
