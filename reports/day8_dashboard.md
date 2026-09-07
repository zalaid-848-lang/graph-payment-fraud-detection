# Day 8 investigator dashboard

## Executive summary

The Streamlit dashboard turns the audited Day 7 output into a ranked human-review queue, understandable escalation reasons, and a focused graph of strictly earlier transactions connected through specific entities.

> The dashboard describes connected activity as suspected fraud-ring evidence only. It does not confirm a criminal network, display evaluation labels, or automate account blocking.

## Verification result

- Audit status: **PASS**
- Alerts audited: `5`
- Connection views rendered: `5`
- Maximum nodes in a focused view: `26`
- Maximum earlier transactions displayed: `20`
- Investigation capacity: `5.00%` of test traffic

All dashboard data and connection views passed these gates:

- `day7_explanations_audited`: **PASS**
- `evaluation_targets_absent`: **PASS**
- `strict_historical_connections`: **PASS**
- `email_domain_hubs_excluded`: **PASS**
- `focused_views_capped`: **PASS**
- `figures_serialized`: **PASS**
- `review_only_recommendations`: **PASS**

## Investigator workflow

1. Review the capacity-limited queue in risk-score order.
2. Filter alerts by an evidence-gated reason code.
3. Open one alert to inspect its amount, product, recommended review action, and plain-language reasons.
4. Inspect the focused connection view for earlier transactions sharing a masked card, customer, device, address, or recipient proxy.
5. Use the evidence as a lead for manual investigation, not as an automatic decision.

## Leakage and visual-scope controls

- Transaction-level fraud labels are removed before data reaches the dashboard.
- Neighbour transactions must have a strictly earlier event time than the selected transaction; equal-time and future activity are excluded.
- Payer and recipient email-domain nodes are omitted from the local view because common domains can create visually misleading hubs.
- The local graph is capped at 20 earlier transactions to preserve readability.
- Entity values are masked in node labels and hover text.

## Capacity-cutoff tie

`7` transactions share the cutoff score while `5` positions were available at that score. Stable chronological order makes the result reproducible; it is not evidence of a risk difference among tied transactions.

## Limitations

- Current dashboard views and metrics use deterministic synthetic development data.
- IEEE-CIS provides transaction labels, not verified fraud-ring labels.
- Shared entities can have legitimate explanations, including households or shared infrastructure.
- The displayed score is a ranking score from a class-balanced model, not a calibrated probability.
- A production version would require access controls, audit logging, monitored data quality, model governance, and an approved investigator workflow.

## Day 9 handoff

Consolidate the experiment report, model card, limitations, and architecture into a polished interview-ready documentation set.
