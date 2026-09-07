# Day 5 tabular-plus-graph model comparison

## Executive summary

A controlled ablation compares rules, tabular logistic regression, and the same logistic classifier with causal graph features. Keeping the classifier fixed isolates the incremental value of graph context.

> **Development-data warning:** Results use a deliberately patterned synthetic sample. They verify the experimental design and cannot be presented as expected bank performance.

Connected activity is described as a **suspected fraud ring**, never a confirmed criminal network. Scores support investigator review and never trigger automatic blocking.

## Leakage-safe design

- Training labels mature after `86400` seconds before they may contribute to neighbour-label features.
- Only labels from the training split are eligible; validation, test, and purge labels are ignored by the label-history builder.
- Structural features use only earlier transaction-time batches.
- Graph scaling, tabular encoding, class weighting, and model fitting use training rows only.
- PageRank snapshot age is retained for audit but excluded from the model because refresh cadence is not a fraud behaviour.
- Validation selects thresholds; test labels are used only for this final comparison.
- Direct entity identifiers are excluded from every model matrix.

## Untouched test-period comparison

| Model | PR-AUC | Precision @ capacity | Recall @ capacity | FPR @ capacity | Recall top 1% | Recall top 5% |
|---|---:|---:|---:|---:|---:|---:|
| Rules only | 0.2976 | 60.00% | 37.50% | 2.63% | 12.50% | 37.50% |
| Tabular logistic | 0.4989 | 60.00% | 37.50% | 2.63% | 12.50% | 37.50% |
| Tabular + graph logistic | 0.8935 | 100.00% | 62.50% | 0.00% | 12.50% | 62.50% |

Adding graph context improved PR-AUC by `0.3946` and changed recall at the fixed capacity by `+25.00%` relative to the tabular model.

## Alert-volume diagnostic at matched recall

Target recall is the rules model's recall at the fixed investigation capacity: `37.50%`.

| Model | Minimum test alerts | Alert fraction | Reduction vs rules |
|---|---:|---:|---:|
| Rules only | 4 | 4.76% | 0.00% |
| Tabular logistic | 5 | 5.95% | -25.00% |
| Tabular + graph logistic | 3 | 3.57% | 25.00% |

This is a retrospective ranking diagnostic calculated with test labels, not an operating threshold or a staffing recommendation.

## Validation-selected thresholds

The following alert counts result when each validation-selected threshold is applied unchanged to the test period. Distribution shift and tied rule scores can move volume away from the nominal capacity.

| Model | Test alerts | Alert fraction | Precision | Recall |
|---|---:|---:|---:|---:|
| Rules only | 8 | 9.52% | 50.00% | 50.00% |
| Tabular logistic | 3 | 3.57% | 66.67% | 25.00% |
| Tabular + graph logistic | 3 | 3.57% | 100.00% | 37.50% |

## Strongest combined-model coefficients

Coefficients are associations after preprocessing, not causal explanations. Correlated graph features can redistribute coefficient magnitude.

| Feature | Coefficient |
|---|---:|
| `log1p_standardized_historical_recipient_degree` | 3.3361 |
| `log1p_standardized_specific_max_entity_degree` | -3.2232 |
| `log1p_standardized_matured_neighbor_label_count` | -2.8328 |
| `log1p_standardized_historical_recipient_email_domain_degree` | 2.2415 |
| `log1p_standardized_specific_max_component_nodes` | -1.8105 |
| `recipient_email_domain=services.example` | -1.6742 |
| `log1p_standardized_graph_known_entity_count` | 1.6508 |
| `log1p_standardized_historical_address_degree` | 1.5253 |
| `log1p_standardized_historical_payer_email_domain_degree` | -1.3409 |
| `log1p_standardized_specific_known_entity_count` | 1.0276 |
| `product_code=H` | 1.0268 |
| `log1p_standardized_specific_mean_pagerank` | 0.9434 |

## Limitations and decision boundary

- IEEE-CIS provides transaction labels, not verified fraud-ring labels.
- A shared device, address, card proxy, or recipient proxy can be legitimate.
- The 24-hour label delay is an explicit project scenario, not a claimed bank process.
- Synthetic performance is useful only for pipeline verification.
- Every escalation remains investigator-reviewed; the project does not block accounts.

## Day 6 handoff

Add a boosted-tree candidate and perform slice-based error analysis while preserving this logistic comparison as the controlled graph-feature ablation.
