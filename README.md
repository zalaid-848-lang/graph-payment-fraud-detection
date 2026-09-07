# Graph-Based Payment Fraud Ring Detection and Investigator Prioritization

An interview-ready data-science project that identifies suspicious relationships among payment transactions and helps an investigator decide which alerts to review first.

> **Scope and language:** Transaction labels indicate fraud at the transaction level. Connected groups are described only as **suspected fraud rings** because IEEE-CIS does not provide verified fraud-ring labels. The system recommends investigator review; it does not automate account blocking or make legal conclusions.

## Business problem

Transaction-only models can miss coordinated activity spread across cards, devices, addresses, email domains, customers, and recipients. This project combines conventional transaction features with graph signals to answer two questions:

1. How risky is this transaction?
2. Which connected suspicious activity should an investigator examine first, and why?

The core delivery is deliberately achievable in 10–12 focused days: a rules baseline, a tabular baseline, a leakage-safe graph-feature model, a fair comparison, and an investigator-facing Streamlit dashboard. GraphSAGE or GAT remains a stretch goal.

## Planned architecture

```mermaid
flowchart LR
    A[IEEE-CIS CSV files or deterministic synthetic sample] --> B[Schema checks and canonical entity keys]
    B --> C[Leakage-safe temporal splits]
    C --> D[Rules baseline]
    C --> E[Tabular model]
    C --> F[Historical graph builder]
    F --> G[Graph features]
    E --> H[Tabular + graph model]
    G --> H
    D --> I[Capacity-aware evaluation]
    E --> I
    H --> I
    H --> J[Investigator dashboard]
    J --> K[Human review decision]
```

## Implementation status

- [x] Independent local Git repository and project structure
- [x] Canonical transaction/entity schema and data dictionary
- [x] Deterministic synthetic generator with a deliberately connected suspicious pattern
- [x] IEEE-CIS ingestion adapter for `train_transaction.csv` and `train_identity.csv`
- [x] Chronological train/validation/test assignment with 24-hour purge gaps
- [x] Dataset manifest with row counts, fraud prevalence, split counts, and SHA-256 checksum
- [x] Focused tests for schema, deterministic generation, IEEE mapping, and temporal separation
- [x] Data-quality report, rules baseline, and leakage-safe tabular baseline (Day 2)
- [x] Capacity-aware evaluation and reproducible Day 2 experiment report
- [x] Typed entity-link graph construction and integrity audit (Day 3)
- [x] Causal same-time batching and missing-entity safeguards
- [x] Historical, type-aware graph feature engineering (Day 4)
- [x] Hub-conscious components, PageRank, and clustering features
- [x] Maturity-delayed, training-label-only neighbour features (Day 5)
- [x] Fair rules/tabular/tabular-plus-graph model comparison
- [x] Deterministic LightGBM graph-feature candidate (Day 6)
- [x] Exhaustive post-test error slices at fixed investigation capacity
- [ ] SHAP explanations and investigator reason codes (Day 7)

## Repository layout

```text
graph-payment-fraud-detection/
├── configs/project.toml
├── data/{raw,interim,processed}/
├── docs/
│   ├── data_dictionary.md
│   ├── graph_feature_dictionary.md
│   ├── graph_schema.md
│   ├── error_analysis.md
│   ├── label_history_features.md
│   └── validation_plan.md
├── reports/
│   ├── day2_experiment.md
│   ├── day3_graph_construction.md
│   ├── day4_graph_features.md
│   ├── day5_model_comparison.md
│   └── day6_boosted_error_analysis.md
├── scripts/
│   ├── prepare_data.py
│   ├── run_day2.py
│   ├── run_day3.py
│   ├── run_day4.py
│   ├── run_day5.py
│   └── run_day6.py
├── src/fraud_detection/
├── tests/
├── .gitignore
├── pyproject.toml
└── README.md
```

## Quick start with synthetic data

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
. .\scripts\activate_windows.ps1
python -m pip install -e ".[dev]"
python scripts/prepare_data.py --mode synthetic
python scripts/run_day2.py
python scripts/run_day3.py
python scripts/run_day4.py
python scripts/run_day5.py
python scripts/run_day6.py
python -m pytest
```

On Windows, `activate_windows.ps1` activates the project virtual environment and
redirects temporary installation files and the pip cache into ignored directories
inside the project. This keeps dependency installation on the same drive as the
repository when drive C has limited free space.

The preparation command writes:

- `data/processed/transactions.csv` — canonical, chronologically ordered transactions with a split column;
- `data/processed/manifest.json` — provenance and validation summary.

All generated and raw data are ignored by Git.

The Day 2 experiment writes machine-readable outputs under `artifacts/day2/` and a
versioned, interview-friendly summary at
[`reports/day2_experiment.md`](reports/day2_experiment.md). The current tabular
baseline is a transparent, class-balanced logistic regression implemented as a
scikit-learn `Pipeline`, with preprocessing fitted only on the training period.
Day 5 keeps the logistic classifier fixed for a controlled graph-feature ablation.
A boosted-tree candidate will be introduced next, after the incremental graph signal
has been measured without changing the classifier family.

The controlled logistic ablation and maturity-delayed neighbour-label contract are
documented in the [Day 5 model comparison](reports/day5_model_comparison.md). The
[Day 6 report](reports/day6_boosted_error_analysis.md) adds a deterministic LightGBM
candidate and post-test error slices. Reported numbers use synthetic development data
and demonstrate pipeline behaviour—not expected performance at ICICI Bank or any other
institution.

## Add the real IEEE-CIS data later

1. Download the IEEE-CIS Fraud Detection competition data from Kaggle after accepting its terms.
2. Place these exact files in `data/raw/ieee-cis/`:
   - `train_transaction.csv`
   - `train_identity.csv`
3. Do **not** add either file to Git. Confirm with `git status --short`.
4. Prepare a development-sized, time-ordered slice:

```powershell
python scripts/prepare_data.py --mode ieee --max-rows 100000
```

5. When memory permits, omit `--max-rows` to prepare the full training data:

```powershell
python scripts/prepare_data.py --mode ieee
```

`TransactionDT` is retained as elapsed seconds from the dataset's undisclosed reference point. No calendar date is invented. The adapter derives stable, hashed entity keys from anonymized fields. IEEE-CIS has no explicit verified customer or payment-recipient identifier, so `customer_id` and `recipient_id` are documented proxies—not real identities.

## Validation and leakage controls

The default holdout is chronological: earliest 60% for training, next 20% for validation, latest 20% for test, with a 24-hour purge gap after each earlier period. Full details are in [the validation plan](docs/validation_plan.md).

The key graph rule is: a feature for transaction time *t* may use only relationships observed at or before *t*, and any label-derived feature may use only labels that would have been available before *t*. Validation and test labels never contribute to their own features. Entity identifiers, raw target aliases, post-outcome fields, and fitted encoders are excluded or fitted within the training boundary.

## Planned evaluation

Models will be compared on PR-AUC, precision and recall at a fixed investigation capacity, recall in the top 1% and 5% of alerts, false-positive rate, and alert-volume reduction versus the rules baseline. Thresholds are selected on validation data and reported once on the untouched test period.

## Reproducibility

- Project seed: `42`
- Configuration: `configs/project.toml`
- Dependencies and test settings: `pyproject.toml`
- Every prepared dataset receives a checksum and provenance manifest
- Coherent milestones are committed only after tests pass

## Limitations already known

- IEEE-CIS labels individual transactions, not fraud rings.
- Anonymized fields make customer and recipient nodes approximate.
- Shared entities can reflect legitimate households, offices, or infrastructure.
- The synthetic data proves the pipeline, not model performance.
- A useful risk score supports investigation; it is not evidence of criminal conduct.

## 10–12 day delivery plan

| Day | Outcome |
|---|---|
| 1 | Repository, schema, ingestion/sample pipeline, leakage-safe split, tests |
| 2 | Data quality report, rules baseline, tabular preprocessing and baseline |
| 3 | Entity-link graph construction and graph integrity tests |
| 4 | Causal structural graph features |
| 5 | Leakage-safe label-history features and controlled graph-feature model comparison |
| 6 | Boosted-tree candidate, capacity-aware evaluation and error analysis |
| 7 | Explainability and investigator reason codes |
| 8 | Streamlit investigation dashboard |
| 9 | Experiment report, model card, limitations and architecture polish |
| 10 | Interview presentation, demo script and end-to-end reproducibility check |
| 11–12 | Buffer; optional GraphSAGE/GAT only if the core is complete |
