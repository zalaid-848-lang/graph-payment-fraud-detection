# Limitations, assumptions, and responsible-use boundaries

This document separates what the prototype demonstrates from what it does not establish. It should be discussed alongside the experiment report and model card.

## Data and label limitations

| Limitation | Why it matters | Mitigation or next evidence step |
|---|---|---|
| Current results use deterministic synthetic data | The generator deliberately includes shared-entity suspicious activity, so performance may be optimistic | Rerun the unchanged pipeline on IEEE-CIS and report the new results separately |
| IEEE-CIS labels transactions, not groups | Connected clusters cannot be evaluated as true fraud rings | Use the term **suspected fraud rings** and evaluate only transaction labels |
| IEEE-CIS does not provide verified fraud-ring labels | Cluster purity, ring precision, and ring recall cannot be claimed | Seek separately governed case-link or investigator-confirmed group labels if available |
| Customer and recipient identities are proxies | Different real entities can collide, and one real entity can fragment across proxies | Document the mapping, test sensitivity, and avoid identity claims |
| Synthetic test set has 84 rows and 8 positives | Point estimates have large sampling uncertainty | Use a larger chronological holdout and confidence intervals on real data |
| Fraud labels may mature with delay | Recent unresolved outcomes can be mistaken for negatives | Preserve the maturity-delay contract and evaluate only after an appropriate observation window |

The project does not claim that the synthetic fraud prevalence, entity reuse, or transaction patterns match ICICI Bank or any other financial institution.

## Graph-representation limitations

- A shared card, device, address, recipient, customer proxy, or email domain is an association, not proof of common control or collusion.
- Legitimate households, offices, public networks, merchants, payment processors, and reused devices can create dense components.
- Common email domains can create weakly informative hubs. They are excluded from specific-entity components and the dashboard graph, but broad historical degree features still require monitoring.
- Connected-component size can become dominated by a high-degree entity. Production work should evaluate hub thresholds, edge weighting, time decay, and component-size caps.
- Current edges are field-equality links. They do not encode transaction direction, monetary flow, ownership, device confidence, or semantic relationship strength.
- The focused dashboard shows at most 20 earlier connected transactions. This improves readability but is not a complete case graph.

## Validation and statistical limitations

- A single chronological split is more realistic than random splitting but does not measure variation across multiple time periods.
- The 24-hour purge and label-maturity windows are project assumptions, not banking-policy claims. They must be aligned with actual data availability before real use.
- Earlier unlabelled validation or test transaction attributes may enter structural state for later transactions, matching streaming availability. Their outcome labels never enter state.
- The selected LightGBM model won validation PR-AUC but was slightly behind graph logistic regression on frozen-test PR-AUC. The project correctly avoids post-test model switching; additional time-based evaluation is needed.
- LightGBM stopped after only 3 boosting iterations on the synthetic validation set, limiting what can be inferred about nonlinear modelling.
- The class-balanced model score is not a calibrated probability. Probability-dependent decisions require held-out calibration and calibration-drift monitoring.
- A zero false-positive rate at top-5% capacity is an observed count on eight positive and 76 negative test examples, not a guarantee.
- Capacity metrics depend on a deterministic rule for score ties. Seven transactions share the cutoff score in the current test.

## Explainability limitations

- SHAP values explain the fitted model locally; they do not establish causality.
- Evidence-gated reason codes improve factual consistency but have not been tested for investigator comprehension, usefulness, or decision bias.
- A positive contribution means a feature raised the model score relative to its SHAP baseline; it does not mean that feature is inherently suspicious in every context.
- Displayed entity references are masked, but the local prototype is not a complete privacy or access-control solution.

## Operational and responsible-use boundaries

The intended output is a prioritised queue for human review. The prototype does not automatically block, decline, freeze, close, report, or accuse. It must not be used as the sole basis for adverse action.

Before any production consideration, an institution would need to define investigation procedures, decision authority, escalation routes, label governance, data access, retention, security, audit logging, monitoring, incident response, and independent validation. These are implementation gaps, not assertions about a particular bank's obligations or current systems.

No protected demographic attributes are available in the current data, so group-fairness analysis has not been performed. Their absence does not demonstrate fairness: graph proxies can encode indirect socioeconomic or geographic patterns. Any future fairness evaluation must use lawfully and appropriately governed data and institution-approved definitions.

## Interview-safe claims

Supported statements:

- The synthetic experiment demonstrates a reproducible, leakage-aware graph-feature pipeline.
- Graph features improved ranking over the tabular baseline on the current synthetic test.
- The dashboard presents suspected connections and reasons for investigator review.
- The model selection rule was fixed before frozen-test comparison.

Unsupported statements:

- The model will achieve these metrics at ICICI Bank.
- The displayed clusters are verified fraud rings or confirmed criminal networks.
- A shared entity proves coordinated fraud.
- The score is a calibrated probability or expected loss.
- The prototype is production-ready, compliant, unbiased, or safe for automatic blocking.

## Priority next steps

1. Add the real IEEE-CIS files locally and rerun the full time-ordered protocol.
2. Measure graph density, missingness, runtime, and memory on a representative data volume.
3. Add rolling or repeated temporal backtests and uncertainty intervals.
4. Compare calibration, stable capacity policies, tie handling, and time-decayed graph features.
5. Conduct investigator review of alert reasons and focused graph usability.
6. Consider GraphSAGE or graph attention only after the graph-feature baseline remains sound on real data.
