# Day 4 causal graph-feature report

## Executive summary

Generated `26` structural features for `500` transactions. Every feature was materialized before its equal-time transaction batch entered graph history. The output contains no target column and all feature-audit gates pass.

> These features support ranking for investigator review. They do not establish that a connected group is a confirmed fraud ring.

## Leakage and quality gates

Overall status: **PASS**

- PASS — `row_count_matches_source`
- PASS — `transaction_ids_unique`
- PASS — `transaction_ids_match_source`
- PASS — `target_columns_absent`
- PASS — `all_features_finite`
- PASS — `all_features_non_negative`
- PASS — `pagerank_bounded`
- PASS — `clustering_bounded`

## Feature families

- Historical degree for each entity type
- Known/new and shared-entity transaction counts
- Email-hub-excluded component size and transaction count
- Lagged weighted PageRank on the specific-entity projection
- Historical clustering coefficient on the specific-entity projection

PageRank was refreshed `20` times across `500` chronological batches, every `25` batches. Snapshot age is emitted explicitly.

## Hub-conscious design

Payer and recipient email domains remain in the full graph and retain degree features. They are excluded from the specific projection and component features because common domains otherwise merge unrelated activity into a giant component.

## Post-generation label diagnostic

Labels were joined only after feature generation for this diagnostic; they were not available to the feature builder.

| Structural feature | Non-fraud mean | Fraud mean |
|---|---:|---:|
| `specific_shared_transaction_count` | 18.434 | 25.938 |
| `specific_max_entity_degree` | 8.559 | 24.750 |
| `specific_max_component_transactions` | 380.072 | 70.438 |

Synthetic label differences validate that the connected pattern reaches the feature table; they do not estimate real-world performance.

## Reproducibility and outputs

- `data/processed/graph_features.csv`: deterministic transaction feature table
- `artifacts/day4/feature_audit.json`: machine-readable audit and summaries
- `docs/graph_feature_dictionary.md`: feature definitions and scaling limitation

## Day 5 handoff

Join these structural features to the existing tabular inputs, train the first tabular-plus-graph model, and compare it fairly with Day 2. Neighbouring-fraud ratios will be added only with a documented label-maturation delay and training-history-only labels.
