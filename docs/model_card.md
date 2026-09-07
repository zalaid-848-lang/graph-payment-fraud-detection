# Model card: graph-feature LightGBM alert ranker

## Model summary

| Field | Value |
|---|---|
| Model | Class-balanced LightGBM binary classifier used as an alert ranker |
| Project version | `0.1.0`, Day 9 documentation milestone |
| Selected candidate | `tabular_graph_lightgbm` |
| Selection rule | Highest validation PR-AUC among the graph logistic and graph LightGBM candidates |
| Implementation | Python, pandas, NetworkX, scikit-learn preprocessing, LightGBM |
| Random seed | `42` |
| Output | Ranking score between 0 and 1; not a calibrated fraud probability |
| Decision owner | Human investigator; the model does not automatically block an account |

## Intended use

The model prioritises transaction alerts when investigators can review only a limited fraction of payment traffic. It combines transaction attributes with historical connectivity and matured training-label context. The dashboard helps an investigator inspect understandable reasons and earlier shared-entity activity.

Connected groups must be described as **suspected fraud rings**. IEEE-CIS labels transactions and does not provide verified fraud-ring labels. A model score or connection is a lead for human review, not proof of fraud, collusion, identity, or criminal conduct.

## Out-of-scope and prohibited uses

- Automatically blocking, declining, freezing, or closing an account or transaction.
- Treating the score as a calibrated probability or financial-loss estimate.
- Treating graph components as confirmed criminal networks.
- Identifying real customers from anonymized proxy identifiers.
- Making legal, regulatory, employment, credit, or law-enforcement decisions.
- Claiming performance at ICICI Bank or another institution from synthetic results.

## Training and evaluation data

The current evidence uses 500 deterministic synthetic transactions created to exercise shared devices, addresses, recipients, and other entity relationships. The temporal assignment contains 300 training rows, 84 validation rows, 84 test rows, and two 16-row purge intervals. The frozen test period contains 8 fraud-labelled transactions.

The synthetic data proves that the software and evaluation protocol work; it is not representative evidence about real fraud prevalence, graph density, investigator workload, or future model performance. The repository includes a documented adapter for IEEE-CIS, but real files are not included in Git.

## Features

The fitted pipeline exposes 48 transformed inputs:

- Transaction amount, product code, payer/recipient email-domain attributes, and entity-presence
  flags.
- Historical degree and shared-transaction counts.
- Hub-conscious connected-component sizes.
- PageRank and clustering coefficients from the specific-entity projection.
- Matured-neighbour label counts and fraud ratios.

There are 28 model graph features: 25 structural features after excluding the PageRank snapshot-age diagnostic, plus 3 matured-label features. Raw entity identifiers are never model inputs. Common email domains are excluded from specific-entity components even though their historical degree can remain a broad feature.

## Leakage controls

- Chronological train/validation/test periods are separated by 24-hour purge gaps.
- Preprocessing and rule thresholds are fitted on training rows only.
- Structural graph features use strictly earlier relationship history.
- Transactions with the same event time are scored before the batch updates graph state.
- Neighbour-label features use only matured labels from the training split with a 24-hour delay.
- Validation, test, and purge labels never enter neighbour-label state.
- Model selection uses validation PR-AUC; test performance is reported after selection.

## Selection and performance

LightGBM was selected because its validation PR-AUC was 0.9519 versus 0.9250 for graph logistic regression. On the frozen synthetic test set:

| Metric | Selected LightGBM |
|---|---:|
| PR-AUC | 0.8869 |
| Precision at top 5% capacity | 100.00% |
| Recall at top 5% capacity | 62.50% |
| Recall in top 1% | 12.50% |
| Recall in top 5% | 62.50% |
| False-positive rate at capacity | 0.00% |
| True positives / false positives / false negatives | 5 / 0 / 3 |

Graph logistic regression achieved a slightly higher frozen-test PR-AUC of 0.8935. The project does not switch candidates after observing that result. This difference should be treated as uncertainty in a very small synthetic test, not as evidence that one algorithm will dominate on real data.

At a matched 37.50% recall, LightGBM and graph logistic regression each required 3 alerts versus 4 for rules, a 25% alert-volume reduction on this synthetic sample.

## Threshold and capacity policy

The primary policy uses a strict top-k queue equal to 5% of test-period traffic, rounded up to 5 alerts. Seven transactions share the selected model's cutoff score while only five positions remain. Stable chronological order resolves the capacity tie for reproducibility; it is not a measured risk difference.

The validation-selected numerical threshold yields 7 test alerts because of ties and is therefore reported separately from the hard-capacity result. A real deployment would need an explicit business policy for ties, queue carryover, and service-level targets.

## Explainability

Exact Tree SHAP values were produced for all 84 test rows across 48 transformed features. The maximum reconstruction error in raw-score space was `3.331e-09`. Alert reasons require a positive model contribution and corresponding factual feature evidence.

SHAP describes how the fitted model produced a score. It does not show that a feature caused fraud or establish suspected fraud-ring membership.

## Known limitations and risks

- Synthetic sample size and deliberately injected structure can exaggerate graph-feature value.
- Test metrics have high uncertainty because there are only 84 rows and 8 positive labels.
- Early stopping selected iteration 3 of 400 configured estimators, showing how little real tuning evidence is available.
- IEEE-CIS customer and recipient nodes are proxies rather than verified identities.
- Shared cards, devices, addresses, or recipients can reflect legitimate households, offices, merchants, or infrastructure.
- No probability calibration, protected-group fairness assessment, loss-based thresholding, concept-drift study, adversarial testing, or production latency benchmark has been completed.
- Investigator reason usefulness has not been evaluated with real investigators.

## Monitoring recommendations before production use

These are future controls, not implemented claims:

- Monitor schema validity, missingness, unseen categories, graph degree, component size, and score distributions.
- Track precision, recall, alert volume, investigation yield, and time-to-review after labels mature.
- Recheck calibration and capacity thresholds by time period and relevant approved cohorts.
- Review false positives, false negatives, large hubs, and reason-code stability.
- Require versioned approval, rollback capability, access control, audit logs, and documented human escalation paths.

## Reproducibility

Configuration lives in `configs/project.toml`. Run `prepare_data.py`, Days 2–9 in order, and the test suite from the documented D-drive environment. Generated raw data and artifacts remain outside Git.
