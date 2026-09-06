# Day 3 graph construction and integrity report

## Executive summary

The project now constructs an undirected, typed transaction–entity graph with NetworkX. Transaction labels are deliberately absent from node and edge attributes. Chronological equal-time batches provide the safe update boundary required for Day 4 historical graph features.

> Connected groups are potential investigation leads only. They are described as **suspected fraud rings**, never confirmed criminal networks.

## Training-period graph

- Transactions: `300`
- Entity nodes: `524`
- Edges: `2,100`
- Connected components: `1`
- Largest component: `824` nodes (`100.00%` of graph nodes)
- Isolated transactions: `0`

| Node type | Count |
|---|---:|
| `address` | 108 |
| `card` | 163 |
| `customer` | 113 |
| `device` | 88 |
| `payer_email_domain` | 4 |
| `recipient` | 45 |
| `recipient_email_domain` | 3 |
| `transaction` | 300 |

## Integrity gates

Overall status: **PASS**

- PASS — `undirected_simple_graph`
- PASS — `node_types_valid`
- PASS — `entity_values_non_missing`
- PASS — `target_attributes_absent`
- PASS — `edges_are_transaction_entity_only`
- PASS — `edge_relations_match_entity_types`
- PASS — `edge_times_match_transactions`
- PASS — `transaction_nodes_match_source`
- PASS — `edge_count_matches_available_entities`
- PASS — `transactions_have_entity_edges`

## Causal replay audit

- Chronological batches processed: `500`
- Equal-time multi-transaction batches: `0`
- Ordering or pre-existence violations: `0`
- Full replay transactions: `500`
- Full replay edges: `3,500`

For every batch at time *t*, consumers first inspect the graph containing only times less than *t*. All transactions at *t* are then added together. This prevents an arbitrary row order from leaking relationships between simultaneous transactions.

The full replay is an integrity diagnostic, not a model feature table. Day 4 will materialize each transaction's features before its batch is added.

## Important topology finding

Common email-domain hubs create a very large connected component. Therefore, raw component membership alone is not a credible suspected-ring signal. Day 4 features will remain type-aware and will distinguish high-frequency infrastructure hubs from more specific shared devices, cards, addresses, customers, and recipients.

## Files produced

- `artifacts/day3/training_graph.json`: label-free training topology
- `artifacts/day3/graph_integrity.json`: machine-readable gates and statistics
- `docs/graph_schema.md`: versioned node, edge, and leakage contract

## Limitations and investigator controls

- IEEE-CIS customer, card, address, device, and recipient nodes are anonymized proxies.
- Shared infrastructure can connect unrelated legitimate transactions.
- Connectivity does not verify intent, identity, or criminal coordination.
- Graph outputs prioritize human review and never automate account blocking.

## Day 4 handoff

Compute causal node degree, shared-entity counts, historical component size, PageRank, and clustering features before each time batch is added. Label-derived neighbour features remain deferred until an explicit historical label-delay policy is enforced.
