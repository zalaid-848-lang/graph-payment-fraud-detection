# Investigator dashboard guide

The Streamlit dashboard presents the audited Day 7 alert queue for investigator review. It is a demonstration interface, not a production case-management system.

## Start the dashboard

From the repository root on drive D:

```powershell
. .\scripts\activate_windows.ps1
python scripts\run_day8.py
python -m streamlit run dashboard\app.py
```

Open the local address displayed by Streamlit, normally `http://localhost:8501`.

If the dashboard says its inputs are not ready, run the preparation script followed by the Day 4–7 scripts. Generated data and model artifacts remain outside Git.

## What the investigator sees

- **Capacity-limited alert queue:** transactions selected within the configured 5% investigation capacity, ordered by model score.
- **Reason filter:** restricts the queue to alerts carrying a particular evidence-gated reason code.
- **Alert detail:** transaction amount, product, ranking score, and the recommendation `INVESTIGATOR_REVIEW`.
- **Why it was escalated:** plain-language reasons backed by positive SHAP contribution and factual scoring-time evidence.
- **Historical connection view:** the selected transaction, its specific entities, and up to 20 strictly earlier transactions sharing those entities.

The focused graph uses card, customer, device, address, and recipient proxies. Email-domain nodes are deliberately excluded because common domains can create large, weakly informative hubs. Entity values are masked in labels and hover text.

## Interpretation boundaries

- A high score means higher review priority within this model; it is not a calibrated probability of fraud.
- A shared entity is a lead to investigate, not proof that the transactions belong to a criminal network.
- Transaction labels are used only for offline evaluation and leakage-safe matured-history features. They are not exposed in the dashboard data layer.
- When multiple transactions share the capacity cutoff score, stable chronological ordering makes selection reproducible but does not imply a score difference.
- No account is blocked and no case is closed automatically. The only recommendation is investigator review.

Use the term **suspected fraud ring** in discussion. IEEE-CIS does not contain verified fraud-ring labels.

## Real-data and production considerations

The current interface is verified with deterministic synthetic data. After adding IEEE-CIS files, rerun the full pipeline and inspect scale, missingness, and graph density before demonstrating results. A production deployment would additionally need authentication, role-based access, audit logging, encryption, retention controls, monitored data quality, model monitoring, and an institution-approved investigation workflow. Those controls are outside this interview prototype and are not claimed as implemented.

## Interview walkthrough

1. Explain that ranking is constrained by investigator capacity.
2. Point out the warning that suspected connections are not confirmed fraud rings.
3. Open one alert and connect its reason code to the visible historical structure.
4. Explain that only strictly earlier neighbours are displayed, matching scoring-time information.
5. Close with the human-review boundary and the limitations of synthetic and anonymized data.
