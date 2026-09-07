# Likely interview questions and concise answers

These answers are phrased for a campus-placement discussion. Adapt the depth to the interviewer and never imply that synthetic results represent ICICI Bank performance.

## 1. Why use a graph if you already have a fraud classifier?

A row-level classifier asks whether one transaction looks unusual. The graph adds whether it shares devices, cards, addresses, customer proxies, or recipient proxies with earlier activity. That relationship context can reveal coordination that is weak in any single row. The controlled logistic-regression ablation isolates this incremental signal.

## 2. What exactly is a “fraud ring” in this project?

It is a suspected connected group that may deserve investigation. IEEE-CIS labels individual transactions and does not verify criminal rings, so the project never treats a connected component as proof of collusion or criminal conduct.

## 3. How did you prevent target leakage?

Splits are chronological. Structural graph features use only relationships available before the scoring batch. Label-derived neighbour features use only matured labels from the training period; validation, test, and purge labels never enter graph state. Preprocessors fit on training rows, and target or post-outcome columns are excluded from model inputs.

## 4. Why batch transactions with the same timestamp?

If equal-time rows update the graph one by one, later rows in arbitrary file order can see information from nominally simultaneous events. The implementation first scores every row in the same-time batch, then updates structural state. This removes row-order leakage.

## 5. Why use a 60/20/20 temporal split and purge gaps?

Temporal order approximates deployment: train on earlier behaviour, select on a later validation period, and report once on the latest test. The 24-hour gaps reduce boundary contamination from rapidly repeated entities. The exact fractions are configurable; the principle is time-respecting evaluation.

## 6. What graph features did you build?

The model uses type-aware historical degree and repeated-entity counts, specific connected-component size, shared-entity counts, PageRank, clustering-related measures, and maturity-delayed neighbouring-fraud counts and ratios. One freshness diagnostic is audited but excluded from model inputs. In total, 28 graph features enter the model.

## 7. How did you handle high-degree hubs?

Entity types are not treated as equally specific. Common email domains can connect many unrelated transactions, so they remain useful for broad degree features but are excluded from specific components and dashboard graphs. Missing entities are also not converted into shared placeholder nodes.

## 8. Why compare logistic regression before LightGBM?

Using the same logistic classifier with and without graph features makes the ablation interpretable: the performance difference comes from relationship context rather than a more flexible algorithm. LightGBM is a second-stage candidate for nonlinear interactions after the graph contribution is established.

## 9. Why was LightGBM selected if graph logistic did better on test?

Model selection happened on validation data. LightGBM validation PR-AUC was 0.9519 versus 0.9250 for graph logistic, so it was selected before the test was inspected. On the frozen test, graph logistic reached 0.8935 and LightGBM 0.8869. Switching afterward would tune to the test, so the project reports the reversal honestly.

## 10. What does early stopping at iteration 3 of 400 tell you?

It suggests the small synthetic validation set supports only a very simple boosted model and that model ordering is uncertain. I treat it as a warning against over-interpreting the LightGBM result, not as evidence that three trees will be optimal on real data.

## 11. Why PR-AUC rather than accuracy or only ROC-AUC?

Fraud labels are imbalanced, so accuracy can look high while missing most fraud. PR-AUC focuses on precision-recall ranking for the positive class. I also report precision, recall, and false-positive rate at investigator capacity because operational usefulness depends on the review queue, not one global score.

## 12. How is “top 5%” implemented when scores tie?

The queue uses an exact top-k capacity with a stable chronological tie-break, so repeated runs select the same rows. The dashboard discloses ties at the cutoff because order within identical scores is not a measured risk difference. Threshold-based alert volume is reported separately from hard-capacity results.

## 13. What does the 25% alert-volume reduction mean?

At a matched synthetic recall target of 37.5%, the graph models needed 3 alerts while the rules baseline needed 4. That is a 25% reduction in this small worked example. It is not a bank staffing estimate and must be re-measured on real temporal data.

## 14. Are the model scores fraud probabilities?

No. They are ranking scores. Probability calibration has not been validated, so the dashboard does not present them as probabilities. The immediate use is prioritisation within a capacity-constrained queue.

## 15. How do SHAP explanations help, and what do they not prove?

Tree SHAP decomposes the selected model’s raw score into feature contributions. A fidelity audit checks that the contributions reconstruct the score. Reason codes are shown only after that gate passes. SHAP explains model behaviour; it does not prove causality, intent, or wrongdoing.

## 16. How would you add the real IEEE-CIS files?

Place `train_transaction.csv` and `train_identity.csv` in `data/raw/ieee-cis/`, which is ignored by Git. Run `python scripts/prepare_data.py --mode ieee --max-rows 100000` for a time-ordered development slice, then omit `--max-rows` when memory allows. The unchanged Day 2–10 pipeline should be rerun and the real-data results compared with the synthetic evidence.

## 17. What would you monitor in production?

I would monitor schema and missingness, score and alert-volume drift, entity-degree and component-size drift, data latency, explanation availability, investigator outcomes, precision at capacity, and subgroup/error slices where lawful and appropriate. I would also version graph snapshots, label-maturity rules, and thresholds so decisions are reproducible.

## 18. Would you block an account automatically?

No. This project recommends investigator review only. Shared entities can have legitimate explanations, labels can be delayed or noisy, and ring membership is not verified. Any operational action would require institution-specific policy, governance, validation, and human oversight beyond this project.

## 19. Why did you defer GraphSAGE or graph attention?

The first objective was a complete, auditable graph-feature system with fair baselines, temporal validation, capacity metrics, explanations, tests, and a dashboard. A graph neural network would add complexity before the real-data value of the simpler graph signals is established. It is a stretch experiment, not a prerequisite.

## 20. What is the most important limitation?

The current evidence is synthetic. It proves that the pipeline, leakage controls, evaluation, and interface work together; it does not establish real-world performance. The next step is an unchanged IEEE-CIS temporal rerun followed by stability and error analysis.
