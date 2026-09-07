# Interview presentation and dashboard demo script

Target duration: 8 minutes for the presentation, followed by a 2-minute dashboard demo. The slide deck contains matching speaker notes, so this document is both a rehearsal guide and a fallback if Presenter View is unavailable.

## Minute 0:00–0:40 — Slide 1: one-sentence framing

“This project asks whether relationship context can improve payment-fraud alert prioritisation beyond transaction-only scoring. I built a reproducible pipeline that links transactions to entity proxies, engineers leakage-safe historical graph features, compares controlled baselines, and sends only evidence-backed recommendations to a human investigator.”

State the boundary immediately: the current metrics are synthetic development results. Connected groups are suspected fraud rings, not confirmed criminal networks.

## Minute 0:40–1:20 — Slide 2: business problem

Explain that four individually ordinary-looking payments may share a device, card proxy, address, or recipient proxy. A tabular row cannot directly express this coordinated pattern. The graph turns shared infrastructure into investigation leads, while recognising that shared entities may also have legitimate explanations.

## Minute 1:20–2:05 — Slide 3: graph design

Describe the heterogeneous entity-link graph: transaction nodes connect to card, customer, device, address, and recipient proxies. IEEE-CIS does not expose verified customers or recipients, so those nodes are explicitly documented proxies. Email domains contribute broad degree signals but are excluded from specific connected components and dashboard views because common domains can become misleading hubs.

## Minute 2:05–3:05 — Slide 4: leakage-safe validation

Walk from left to right through the chronological 60/20/20 split and two 24-hour purge gaps. Emphasise three controls:

1. Preprocessing and decision thresholds learn only from training or validation as appropriate.
2. Each transaction receives graph features from strictly earlier relationship history.
3. Neighbour-fraud features use only matured training labels. Validation and test labels never update label history.

Mention same-time batching: all transactions at the same timestamp are scored before that batch changes the graph, preventing row-order leakage.

## Minute 3:05–3:50 — Slide 5: controlled comparison

Explain why the logistic comparison matters. Rules provide an operational reference; tabular logistic regression is the conventional baseline; the same logistic classifier plus graph features isolates the incremental value of relationship context. LightGBM is introduced only afterward to test nonlinear interactions. The core model was completed before considering GraphSAGE or attention.

## Minute 3:50–4:45 — Slide 6: ranking result

Lead with the result: on the frozen synthetic test, PR-AUC rose from 0.4989 for tabular logistic regression to 0.8935 after adding graph features. The selected graph LightGBM candidate reached 0.8869. Rules achieved 0.2976. Explain that PR-AUC is suitable for imbalanced fraud labels and that these numbers demonstrate pipeline behaviour, not expected bank performance.

## Minute 4:45–5:35 — Slide 7: investigator capacity

Translate ranking into workload. At a hard 5% capacity, both graph models recovered 5 of 8 fraud-labelled test transactions: 62.5% recall, with no false positives in this small synthetic holdout. At matched 37.5% recall, graph models needed 3 alerts versus 4 for rules, a 25% reduction. Call this a worked operational example, not a staffing forecast.

## Minute 5:35–6:25 — Slide 8: honest model selection

LightGBM was selected before test inspection because validation PR-AUC was 0.9519 versus 0.9250 for graph logistic regression. The frozen test then ranked graph logistic slightly higher: 0.8935 versus 0.8869. I did not switch models after seeing the test result. LightGBM also stopped at iteration 3 of 400, which reinforces that the synthetic holdout is too small for strong model-choice claims.

## Minute 6:25–7:20 — Slide 9: investigator workflow

Describe the product path: capacity-limited queue, evidence-gated SHAP reasons, focused earlier-connection view, and human review. Model scores are ranking scores rather than calibrated fraud probabilities. The dashboard masks entity values, excludes email hubs, caps neighbour display, and never exposes evaluation labels. The recommendation is `INVESTIGATOR_REVIEW`; there is no automatic blocking.

## Minute 7:20–8:00 — Slide 10: close with evidence discipline

“The project supports three claims: the pipeline is reproducible and leakage-aware; graph context improved synthetic ranking over the tabular baseline; and model evidence is connected to a review-only workflow. It does not yet prove IEEE-CIS performance, ring-level accuracy, probability calibration, or production utility. The next evidence step is to rerun the unchanged temporal protocol on IEEE-CIS—not to add a deep graph model for novelty.”

Pause on the questions slide.

## Minute 8:00–10:00 — Dashboard demo

Before the interview, run:

```powershell
. .\scripts\activate_windows.ps1
python -m streamlit run dashboard/app.py
```

Demo path:

1. Start at the queue and point out the fixed capacity, stable rank, risk score, and `INVESTIGATOR_REVIEW` recommendation.
2. Select a high-priority transaction and read two reason codes. Explain that reasons appear only when SHAP fidelity and data contracts have passed.
3. Open the focused connection view. Show that transaction neighbours are strictly earlier, entity references are masked, and the view is capped at 20 neighbours.
4. Point to the limitation panel: suspected fraud-ring connections are investigation leads, not verified collusion.
5. Close by saying that investigators decide the outcome and that labels are absent from the dashboard data layer.

## If the live dashboard fails

Do not spend interview time debugging. Return to slide 9 and describe the same four-step workflow. State that the dashboard audit is reproducible through `python scripts/run_day8.py` and the complete delivery audit through `python scripts/run_day10.py`.

## Numbers to memorise

- Tabular logistic test PR-AUC: 0.4989
- Graph logistic test PR-AUC: 0.8935
- Graph LightGBM test PR-AUC: 0.8869
- Graph logistic validation PR-AUC: 0.9250
- Graph LightGBM validation PR-AUC: 0.9519
- Recall at 5% capacity for both graph models: 62.5%, or 5 of 8 fraud-labelled transactions
- Matched-recall alert reduction versus rules: 25%, or 3 alerts instead of 4

Every number above is synthetic development evidence.
