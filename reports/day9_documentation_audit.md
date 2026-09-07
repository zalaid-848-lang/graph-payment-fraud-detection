# Day 9 interview documentation audit

## Outcome

Day 9 status: **PASS**. The experiment report, model card, architecture, and limitations are tied to the verified Day 5–8 artifacts and preserve the project's responsible-use boundaries.

## Evidence summary

- Data source: `synthetic`
- Selected model: `tabular_graph_lightgbm`
- Validation PR-AUC: `0.9519`
- Frozen test PR-AUC: `0.8869`
- Recall at 5% investigation capacity: `62.50%`
- Test fraud-labelled transactions: `8`

These are synthetic development results, not expected performance at a bank.

## Documents

| Document | Path | Characters |
|---|---|---:|
| Readme | `README.md` | 11,039 |
| Architecture | `docs/architecture.md` | 5,255 |
| Model Card | `docs/model_card.md` | 7,365 |
| Limitations | `docs/limitations.md` | 6,821 |
| Experiment Report | `reports/experiment_report.md` | 5,518 |

## Documentation gates

- `all_documents_nonempty`: **PASS**
- `architecture_is_visual`: **PASS**
- `readme_marks_day9_complete`: **PASS**
- `synthetic_scope_disclosed`: **PASS**
- `suspected_ring_language_present`: **PASS**
- `human_review_boundary_present`: **PASS**
- `score_not_probability_disclosed`: **PASS**
- `selected_model_reported`: **PASS**
- `core_metrics_match_evidence`: **PASS**
- `frozen_test_selection_disclosed`: **PASS**
- `ieee_ring_label_limit_disclosed`: **PASS**
- `unsupported_assertions_absent`: **PASS**

## Interview narrative

The business contribution is not merely a higher model score. The project shows how to build entity connections with scoring-time history, test graph features through a controlled logistic-regression ablation, select a nonlinear candidate without test tuning, evaluate within investigation capacity, and expose review-only evidence with clear limitations.

## Day 10 handoff

Create the interview presentation and demo script, then run the final clean-clone-style reproducibility check. GraphSAGE remains optional and should not displace the polished core delivery.
