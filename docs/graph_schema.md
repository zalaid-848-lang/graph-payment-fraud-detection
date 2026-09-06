# Graph schema and construction rules

## Purpose

The entity-link graph represents relationships visible in payment data. It supports
structural feature engineering and investigator navigation; it does not assert that
any connected component is a criminal network.

## Node types

| Node type | Source field | Interpretation |
|---|---|---|
| `transaction` | `transaction_id` | One observed payment transaction |
| `card` | `card_id` | Card or IEEE-CIS card-attribute proxy |
| `customer` | `customer_id` | Customer or documented customer proxy |
| `device` | `device_id` | Device/browser proxy |
| `address` | `address_id` | Address proxy |
| `payer_email_domain` | `payer_email_domain` | Normalized payer email domain |
| `recipient_email_domain` | `recipient_email_domain` | Normalized recipient email domain |
| `recipient` | `recipient_id` | Recipient or documented recipient proxy |

Internal node identifiers are `(node_type, value)` tuples. Type is therefore part
of identity: a device and address with the same raw value remain different nodes.

## Edge types

Every edge connects a transaction to one available entity. Edge relation names are
`uses_card`, `uses_customer`, `uses_device`, `uses_address`,
`uses_payer_email_domain`, `uses_recipient_email_domain`, and `uses_recipient`.
Edges record the transaction's elapsed event time.

The graph is undirected and bipartite by construction. There are no direct
transaction-to-transaction or entity-to-entity edges. Such relationships can be
derived for analysis without duplicating the source of truth.

## Leakage and integrity rules

1. `is_fraud` and all target aliases are absent from graph node and edge attributes.
2. Null, empty, `nan`, `none`, and similar placeholder values never create nodes.
3. Transactions sharing a missing field are never connected.
4. Data is replayed in chronological batches. All transactions at the same elapsed
   time are queried against the same prior graph state, then added together.
5. A Day 4 feature for time *t* will query only edges with time less than *t*.
6. Purge-period edges may enter later unlabelled structural history because they
   would have been observed operationally. Their labels remain unavailable.
7. Validation and test labels never influence graph construction or structural
   features.

## Investigation language

A large or unusually interconnected component may be escalated as a **suspected
fraud ring** only after a documented scoring rule or model ranks it for human
review. Connectivity alone can reflect legitimate households, shared networks,
business devices, or common service providers.
