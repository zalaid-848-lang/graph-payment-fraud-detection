# Day 7 SHAP explanations and investigator reason codes

## Executive summary

The graph LightGBM model is selected using validation PR-AUC before final test reporting. Exact Tree SHAP values are converted into factual, evidence-gated reasons for the capacity-ranked alerts.

> The displayed risk score is a ranking score from a class-balanced model, not a calibrated probability of fraud. Explanations support investigator review; they do not confirm fraud-ring membership or authorize automatic blocking.

## Model selection

| Candidate | Validation PR-AUC | Frozen test PR-AUC | Test recall @ capacity | Selected |
|---|---:|---:|---:|---:|
| `tabular_graph_logistic` | 0.9250 | 0.8935 | 62.50% | no |
| `tabular_graph_lightgbm` | 0.9519 | 0.8869 | 62.50% | yes |

Selection uses validation PR-AUC. The test comparison is frozen from Day 6 and is reported only after selection; explanations do not trigger model retuning.

## SHAP fidelity audit

- SHAP version: `0.52.0`
- Background: `300` training rows
- Explained rows: `84` test rows
- Explained features: `48`
- Maximum raw-score reconstruction error: `3.331e-09`
- Fidelity status: **PASS**

SHAP values sum with the base value to reproduce the fitted LightGBM raw score. This verifies model fidelity, not causal validity.

## Investigator reason-code coverage

Generated explanations for `5` capacity-selected alerts, with at most `3` positive reasons per alert.

| Reason code | Investigator label | Selected-alert count |
|---|---|---:|
| `MATURED_FRAUD_NEIGHBOUR` | Previously resolved fraud-labelled neighbour | 5 |
| `TRANSACTION_ATTRIBUTE_PATTERN` | Model-associated transaction attributes | 3 |

All reasons require both a positive SHAP contribution and factual evidence in the transaction's scoring-time feature row. For example, the matured-fraud-neighbour reason cannot appear unless the matured neighbour count is greater than zero.

## Capacity-cutoff tie audit

`7` transactions share the cutoff score, while `5` positions remain at that score. The artifact retains stable chronological transaction order for reproducibility, but that order must not be interpreted as a model-measured risk difference.

## Global mean absolute SHAP importance

Global importance summarizes model reliance across the test period. It does not replace transaction-level reasons.

| Feature | Normalized mean absolute SHAP |
|---|---:|
| `log1p_standardized_specific_max_component_nodes` | 0.2413 |
| `log1p_standardized_matured_neighbor_fraud_ratio` | 0.2369 |
| `log1p_standardized_specific_mean_pagerank` | 0.2062 |
| `log1p_standardized_specific_max_pagerank` | 0.0759 |
| `log1p_standardized_graph_sum_entity_degree` | 0.0524 |
| `log1p_standardized_historical_payer_email_domain_degree` | 0.0430 |
| `log_amount_standardized` | 0.0416 |
| `recipient_email_domain=merchant.example` | 0.0269 |
| `log1p_standardized_specific_mean_clustering` | 0.0265 |
| `log1p_standardized_specific_max_entity_degree` | 0.0195 |
| `product_code=W` | 0.0153 |
| `log1p_standardized_graph_shared_transaction_count` | 0.0147 |
| `log1p_standardized_historical_device_degree` | 0.0000 |
| `log1p_standardized_historical_address_degree` | 0.0000 |
| `log1p_standardized_matured_neighbor_label_count` | 0.0000 |

## Safety and interpretation boundaries

- Only features available at scoring time are explained.
- Validation, test, and purge labels never enter neighbour-label features.
- Positive SHAP values explain the model score; they do not prove misconduct.
- Shared entities may have legitimate explanations.
- Investigator-facing records omit the transaction's evaluation label.
- Recommendations remain human-reviewed and never automate account blocking.

## Day 8 handoff

Build the Streamlit investigation dashboard around the audited alert artifact, with ranked alerts, local reasons, and a focused connection view for suspected fraud rings.
