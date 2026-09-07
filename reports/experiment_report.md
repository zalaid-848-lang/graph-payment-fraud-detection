# Consolidated experiment report

## Executive summary

This project tests whether scoring-time graph context improves payment-fraud alert prioritisation beyond individual-transaction rules and tabular attributes. On the deterministic synthetic development dataset, adding leakage-safe graph features to logistic regression increased test PR-AUC from `0.4989` to `0.8935` and recall within the top 5% of alerts from `37.50%` to `62.50%`.

These figures demonstrate pipeline behaviour on synthetic data; they do not estimate performance at ICICI Bank or any other institution. Connected groups are suspected fraud rings for investigation, not verified criminal networks.

## Business question and decision

The decision is which transactions should enter a limited investigator queue. The primary operating point ranks the top `5.00%` of test-period transactions. A model score prioritises review and is not a calibrated fraud probability. The system recommends human investigation only and never blocks an account automatically.

## Data and validation design

- Source: deterministic `synthetic` development data with seed `42`.
- Modelled rows: `300` train, `84` validation, and `84` frozen test rows, plus two 24-hour purge intervals.
- Test positives: `8` transaction-level fraud labels.
- Ordering: chronological splits; preprocessing is fitted on training rows only.
- Structural graph state: strictly earlier relationship observations; transactions sharing an event time are scored as one batch before that batch updates history.
- Label-history state: only matured labels from the training split; validation, test, and purge labels are never used as neighbour outcomes.
- Entity identifiers: excluded from model inputs and used only to construct links.

The synthetic generator deliberately contains connected suspicious activity so the complete pipeline can be developed before IEEE-CIS is added. It is not a benchmark.

## Compared approaches

1. Rules only: high-amount and unseen-entity indicators learned from training data.
2. Tabular logistic regression: class-balanced scikit-learn pipeline using amount, product, and email-domain attributes.
3. Tabular + graph logistic regression: the same classifier family with 28 graph features, isolating the incremental value of graph context.
4. Tabular + graph LightGBM: a nonlinear candidate using the same 48 transformed inputs, with early stopping on validation data.

## Frozen test results

| Approach | PR-AUC | Precision @ 5% | Recall @ 5% | Recall top 1% | Recall top 5% | FPR @ 5% |
|---|---:|---:|---:|---:|---:|---:|
| Rules only | 0.2976 | 60.00% | 37.50% | 12.50% | 37.50% | 2.63% |
| Tabular logistic | 0.4989 | 60.00% | 37.50% | 12.50% | 37.50% | 2.63% |
| Tabular + graph logistic | 0.8935 | 100.00% | 62.50% | 12.50% | 62.50% | 0.00% |
| Tabular + graph LightGBM | 0.8869 | 100.00% | 62.50% | 12.50% | 62.50% | 0.00% |

PR-AUC is the primary ranking metric because fraud labels are imbalanced. Capacity metrics answer the operational question: what is recovered when investigators can review only a small fraction of transactions?

## Alert-volume comparison at matched recall

At a matched recall target of `37.50%`:

| Approach | Alerts required | Reduction versus rules |
|---|---:|---:|
| Rules only | 4 | 0.00% |
| Tabular logistic | 5 | -25.00% |
| Tabular + graph logistic | 3 | 25.00% |
| Tabular + graph LightGBM | 3 | 25.00% |

Both graph models required 3 alerts versus 4 for rules, a 25% alert-volume reduction on this synthetic test set. The tabular-only model required 5 alerts.

## Model selection without test-set tuning

The selected model is `tabular_graph_lightgbm` because validation PR-AUC was `0.9519`, compared with `0.9250` for graph logistic regression. After selection, the frozen test showed graph logistic regression slightly ahead on PR-AUC (`0.8935` versus `0.8869`). The project does not switch models after seeing that test result; it reports the result as sampling uncertainty and a reason to validate on real data.

LightGBM stopped at iteration `3` of `400` configured estimators. That very early stopping point is another warning against over-interpreting the small synthetic experiment.

## Selected-model errors and explanations

At the fixed top-5% capacity, the selected model produced `5` true positives, `0` false positives, and `3` false negatives among `8` fraud-labelled test transactions.

Tree SHAP explained `84` test rows across `48` transformed features. The maximum raw-score reconstruction error was `3.331e-09`. SHAP explains the fitted model's score; it does not establish causality or misconduct.

## Capacity-tie interpretation

`7` transactions shared the selected model's cutoff score, while `5` positions remained at that score. Stable chronological order makes the top-k queue reproducible, but investigators must not interpret the order among tied transactions as a measured risk difference. Applying the validation threshold without a hard capacity cap would produce a different alert count, so threshold and top-k results are reported separately.

## Conclusion

The controlled ablation supports the project hypothesis on synthetic data: graph context materially improves prioritisation over tabular-only scoring. The correct next evidence step is to rerun the unchanged protocol on IEEE-CIS, inspect temporal stability and graph density, and calibrate operational choices with investigators. A GraphSAGE or attention model remains optional until this core baseline is validated.
