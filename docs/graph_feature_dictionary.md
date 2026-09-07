# Causal graph feature dictionary

Every feature is calculated immediately before the transaction's equal-time batch
is added to graph history. Values therefore use only relationships with an earlier
`transaction_time`. No transaction label is read or emitted.

“Specific” features exclude payer and recipient email-domain nodes because common
domains can create infrastructure hubs. Those domains remain present in the full
graph and retain their own historical degree features.

| Feature | Meaning before the current time batch |
|---|---|
| `graph_entity_count` | Number of available entity links on the transaction |
| `graph_known_entity_count` | Available entities already present in full history |
| `graph_new_entity_count` | Available entities not previously observed |
| `graph_sum_entity_degree` | Sum of earlier transaction counts across all linked entities |
| `graph_max_entity_degree` | Largest earlier transaction count for any linked entity |
| `graph_mean_entity_degree` | Mean earlier transaction count across linked entities |
| `graph_shared_transaction_count` | Distinct earlier transactions sharing at least one entity, including email domains |
| `specific_entity_count` | Available non-email-domain entity links |
| `specific_known_entity_count` | Non-email-domain entities already in history |
| `specific_shared_transaction_count` | Distinct earlier transactions sharing a card, customer, device, address, or recipient |
| `specific_max_entity_degree` | Largest earlier transaction count among specific entities |
| `specific_mean_entity_degree` | Mean earlier transaction count among specific entities |
| `specific_max_component_nodes` | Largest prior component, in transaction and specific-entity nodes, touched by the transaction |
| `specific_max_component_transactions` | Largest prior transaction count in a touched specific component |
| `specific_max_pagerank` | Maximum lagged PageRank among linked specific entities |
| `specific_mean_pagerank` | Mean lagged PageRank among linked specific entities |
| `specific_max_clustering` | Maximum prior unweighted clustering coefficient in the specific-entity projection |
| `specific_mean_clustering` | Mean prior clustering coefficient in the specific-entity projection |
| `pagerank_snapshot_age_batches` | Number of chronological batches since the PageRank snapshot was refreshed |
| `historical_card_degree` | Earlier transactions using this card proxy |
| `historical_customer_degree` | Earlier transactions using this customer proxy |
| `historical_device_degree` | Earlier transactions using this device proxy |
| `historical_address_degree` | Earlier transactions using this address proxy |
| `historical_payer_email_domain_degree` | Earlier transactions using this payer email domain |
| `historical_recipient_email_domain_degree` | Earlier transactions using this recipient email domain |
| `historical_recipient_degree` | Earlier transactions using this recipient proxy |

## PageRank and clustering graph

PageRank and clustering are calculated on a projected graph containing specific
entity nodes. Two entities are connected when an earlier transaction used both;
edge weight counts repeated co-occurrence. PageRank is refreshed periodically for
runtime control, and snapshot age is retained as an auditable feature. Each
snapshot is based only on history available before its refresh batch.

## Scaling note

The current implementation is an exact, clear development reference suitable for
the synthetic data and IEEE-CIS samples. Full-dataset production would replace
per-batch connected-component recomputation with incremental union-find state and
would checkpoint centrality calculations. This is an engineering limitation, not
a claim that the current implementation is ready for real-time banking traffic.
