# Data dictionary

## Canonical transaction table

The prepared table has one row per transaction. Columns beginning with `raw_` are optional source fields retained for later tabular modelling. Canonical columns have stable meaning across synthetic and IEEE-CIS inputs.

| Column | Type | Required | Meaning | Leakage treatment |
|---|---|---:|---|---|
| `transaction_id` | string | yes | Unique transaction key | Identifier only; never a model feature |
| `transaction_time` | integer | yes | Elapsed seconds from a source-specific reference | Split/order field; not interpreted as a real date |
| `amount` | float | yes | Transaction amount in source units | Known at authorization time |
| `product_code` | string | yes | Product/category code | Known transaction attribute |
| `card_id` | string/nullable | yes | Stable hashed card entity key | Graph join key; excluded as a direct model feature |
| `customer_id` | string/nullable | yes | Customer key or documented customer proxy | Graph join key; not a verified identity in IEEE-CIS |
| `device_id` | string/nullable | yes | Stable hashed device entity key | Graph join key |
| `address_id` | string/nullable | yes | Stable hashed address entity key | Graph join key; may be shared legitimately |
| `payer_email_domain` | string/nullable | yes | Normalized payer email domain | Graph node/category; no email local part |
| `recipient_email_domain` | string/nullable | yes | Normalized recipient email domain | Graph node/category |
| `recipient_id` | string/nullable | yes | Recipient key or documented recipient proxy | Graph join key; not a verified merchant identity in IEEE-CIS |
| `is_fraud` | integer {0,1} | yes | Transaction-level target | Never used to create contemporaneous or future features |
| `source` | string | yes | `synthetic` or `ieee_cis` | Provenance only |
| `split` | category | after preparation | `train`, `purge_train_validation`, `validation`, `purge_validation_test`, or `test` | Assigned strictly by time |

## IEEE-CIS field mapping

IEEE-CIS is anonymized and does not directly expose all business entities in the project title. These are analytical proxies and are named precisely in reporting.

| Canonical field | IEEE-CIS source | Interpretation |
|---|---|---|
| `transaction_id` | `TransactionID` | Dataset transaction key |
| `transaction_time` | `TransactionDT` | Elapsed seconds from an undisclosed reference |
| `amount` | `TransactionAmt` | Transaction amount |
| `product_code` | `ProductCD` | Anonymized product code |
| `card_id` | hash of available `card1`…`card6` | Card-attribute combination proxy |
| `customer_id` | hash of card proxy, `addr1`, and `P_emaildomain` | Customer/account proxy; not a known person |
| `device_id` | hash of `DeviceType`, `DeviceInfo`, `id_30`, and `id_31` | Device/browser proxy |
| `address_id` | hash of `addr1` and `addr2` | Billing-address proxy |
| `payer_email_domain` | `P_emaildomain` | Normalized domain |
| `recipient_email_domain` | `R_emaildomain` | Normalized domain |
| `recipient_id` | hash of `ProductCD` and `R_emaildomain` | Coarse payment-destination proxy, not a verified recipient |
| `is_fraud` | `isFraud` | Transaction-level target |

Hashes are deterministic SHA-256 prefixes used to create repeatable keys without presenting anonymized combinations as identities. A missing combination remains null; all missing values are never collapsed into a single fake entity.

## Graph representation planned for Day 3

- Transaction nodes connect to each available entity node.
- Entity types remain separate through prefixes (`card:`, `device:`, and so on), preventing accidental cross-type collisions.
- Transactions are never linked because they merely share a missing value.
- Structural features will be calculated from historical graph state.
- A suspected group is a connected subgraph that meets a documented risk threshold; it is not a confirmed fraud ring.

