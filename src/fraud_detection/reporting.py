"""Evidence-backed Day 9 reporting and documentation audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODEL_ORDER = (
    "rules",
    "tabular_logistic",
    "tabular_graph_logistic",
    "tabular_graph_lightgbm",
)
MODEL_NAMES = {
    "rules": "Rules only",
    "tabular_logistic": "Tabular logistic",
    "tabular_graph_logistic": "Tabular + graph logistic",
    "tabular_graph_lightgbm": "Tabular + graph LightGBM",
}


def load_day9_sources(project_root: Path) -> dict[str, dict[str, Any]]:
    """Load the verified artifacts used as Day 9's source of truth."""

    paths = {
        "day5": project_root / "artifacts" / "day5" / "metrics.json",
        "day6": project_root / "artifacts" / "day6" / "metrics_and_errors.json",
        "day7": project_root / "artifacts" / "day7" / "explanation_audit.json",
        "day8": project_root / "artifacts" / "day8" / "dashboard_audit.json",
    }
    missing = [str(path.relative_to(project_root)) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Day 9 evidence sources: {', '.join(missing)}")
    return {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}


def build_evidence_snapshot(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Extract a compact, stable evidence contract from verified experiment artifacts."""

    missing = sorted({"day5", "day6", "day7", "day8"} - set(sources))
    if missing:
        raise ValueError(f"Missing evidence source groups: {', '.join(missing)}")
    day5 = sources["day5"]
    day6 = sources["day6"]
    day7 = sources["day7"]
    day8 = sources["day8"]
    models = {}
    for name in MODEL_ORDER:
        model = day6["models"][name]
        ranking = model["ranking"]
        capacity = ranking["at_investigation_capacity"]
        matched = day6["matched_recall"]["models"][name]
        models[name] = {
            "pr_auc": float(ranking["pr_auc"]),
            "precision_at_capacity": float(capacity["precision"]),
            "recall_at_capacity": float(capacity["recall"]),
            "false_positive_rate_at_capacity": float(capacity["false_positive_rate"]),
            "recall_top_1_percent": float(ranking["top_1_percent"]["recall"]),
            "recall_top_5_percent": float(ranking["top_5_percent"]["recall"]),
            "matched_recall_alert_count": int(matched["alert_count"]),
            "matched_recall_alert_volume_reduction_vs_rules": float(
                matched["alert_volume_reduction_vs_rules"]
            ),
        }

    selected_name = str(day7["model_selection"]["selected_model"])
    selection_candidates = day7["model_selection"]["candidates"]
    gates = {
        "label_history_audit_passed": day5["label_history_audit"]["status"] == "PASS"
        and all(day5["label_history_audit"]["gates"].values()),
        "error_analysis_audit_passed": day6["error_analysis"]["status"] == "PASS"
        and all(day6["error_analysis"]["gates"].values()),
        "shap_fidelity_audit_passed": day7["shap_audit"]["status"] == "PASS",
        "reason_code_audit_passed": day7["reason_code_summary"]["status"] == "PASS"
        and all(day7["reason_code_summary"]["gates"].values()),
        "dashboard_audit_passed": day8["status"] == "PASS" and all(day8["gates"].values()),
    }
    snapshot = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "source": str(day6["source"]),
        "project_seed": 42,
        "rows": {name: int(value) for name, value in day6["rows"].items()},
        "test_fraud_labels": int(day6["error_analysis"]["overall"]["fraud_labels"]),
        "capacity_fraction": float(day6["capacity_fraction"]),
        "matched_recall_target": float(day6["matched_recall"]["target_recall"]),
        "model_graph_feature_count": int(day5["graph_feature_count"]),
        "model_total_transformed_feature_count": int(day6["boosted_training"]["feature_count"]),
        "models": models,
        "model_selection": {
            "selected_model": selected_name,
            "selection_metric": str(day7["model_selection"]["selection_metric"]),
            "validation_pr_auc": {
                name: float(values["validation_pr_auc"])
                for name, values in selection_candidates.items()
            },
        },
        "boosted_training": {
            "library": str(day6["boosted_training"]["library"]),
            "library_version": str(day6["boosted_training"]["library_version"]),
            "best_iteration": int(day6["boosted_training"]["best_iteration"]),
            "configured_estimators": int(day6["boosted_training"]["configured_estimators"]),
        },
        "selected_model_capacity_outcomes": {
            key: int(value) for key, value in day6["error_analysis"]["overall"].items()
        },
        "shap": {
            "version": str(day7["shap_version"]),
            "explained_rows": int(day7["shap_audit"]["explained_rows"]),
            "feature_count": int(day7["shap_audit"]["feature_count"]),
            "maximum_raw_score_reconstruction_error": float(
                day7["shap_audit"]["maximum_raw_score_reconstruction_error"]
            ),
        },
        "dashboard": {
            "alert_count": int(day8["alert_count"]),
            "connection_views_rendered": int(day8["connection_views_rendered"]),
            "maximum_displayed_historical_neighbours": int(
                day8["maximum_displayed_historical_neighbours"]
            ),
        },
        "capacity_tie_audit": day8["capacity_tie_audit"],
        "gates": gates,
    }
    if snapshot["status"] != "PASS":
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"Evidence source audits failed: {', '.join(failed)}")
    return snapshot


def _percent(value: float) -> str:
    return f"{value:.2%}"


def render_experiment_report(evidence: dict[str, Any]) -> str:
    """Render the consolidated interview-ready experiment report."""

    rows = evidence["rows"]
    models = evidence["models"]
    selected = evidence["model_selection"]["selected_model"]
    selection_scores = evidence["model_selection"]["validation_pr_auc"]
    outcomes = evidence["selected_model_capacity_outcomes"]
    tie = evidence["capacity_tie_audit"]
    table_rows = []
    for name in MODEL_ORDER:
        metrics = models[name]
        table_rows.append(
            f"| {MODEL_NAMES[name]} | {metrics['pr_auc']:.4f} | "
            f"{_percent(metrics['precision_at_capacity'])} | "
            f"{_percent(metrics['recall_at_capacity'])} | "
            f"{_percent(metrics['recall_top_1_percent'])} | "
            f"{_percent(metrics['recall_top_5_percent'])} | "
            f"{_percent(metrics['false_positive_rate_at_capacity'])} |"
        )
    matched_rows = []
    for name in MODEL_ORDER:
        metrics = models[name]
        matched_rows.append(
            f"| {MODEL_NAMES[name]} | {metrics['matched_recall_alert_count']} | "
            f"{_percent(metrics['matched_recall_alert_volume_reduction_vs_rules'])} |"
        )

    return "\n".join(
        [
            "# Consolidated experiment report",
            "",
            "## Executive summary",
            "",
            "This project tests whether scoring-time graph context improves payment-fraud alert "
            "prioritisation beyond individual-transaction rules and tabular attributes. On the "
            "deterministic synthetic development dataset, adding leakage-safe graph features to "
            "logistic regression increased test PR-AUC from "
            f"`{models['tabular_logistic']['pr_auc']:.4f}` to "
            f"`{models['tabular_graph_logistic']['pr_auc']:.4f}` and recall within the top 5% "
            f"of alerts from `{_percent(models['tabular_logistic']['recall_at_capacity'])}` to "
            f"`{_percent(models['tabular_graph_logistic']['recall_at_capacity'])}`.",
            "",
            "These figures demonstrate pipeline behaviour on synthetic data; they do not estimate "
            "performance at ICICI Bank or any other institution. Connected groups are suspected "
            "fraud rings for investigation, not verified criminal networks.",
            "",
            "## Business question and decision",
            "",
            "The decision is which transactions should enter a limited investigator queue. The "
            f"primary operating point ranks the top `{_percent(evidence['capacity_fraction'])}` "
            "of test-period transactions. A model score prioritises review and is not a calibrated "
            "fraud probability. The system recommends human investigation only and never blocks "
            "an account automatically.",
            "",
            "## Data and validation design",
            "",
            f"- Source: deterministic `{evidence['source']}` development data with seed "
            f"`{evidence['project_seed']}`.",
            f"- Modelled rows: `{rows['train']}` train, `{rows['validation']}` validation, and "
            f"`{rows['test']}` frozen test rows, plus two 24-hour purge intervals.",
            f"- Test positives: `{evidence['test_fraud_labels']}` transaction-level fraud labels.",
            "- Ordering: chronological splits; preprocessing is fitted on training rows only.",
            "- Structural graph state: strictly earlier relationship observations; transactions "
            "sharing an event time are scored as one batch before that batch updates history.",
            "- Label-history state: only matured labels from the training split; validation, test, "
            "and purge labels are never used as neighbour outcomes.",
            "- Entity identifiers: excluded from model inputs and used only to construct links.",
            "",
            "The synthetic generator deliberately contains connected suspicious activity so the "
            "complete pipeline can be developed before IEEE-CIS is added. It is not a benchmark.",
            "",
            "## Compared approaches",
            "",
            "1. Rules only: high-amount and unseen-entity indicators learned from training data.",
            "2. Tabular logistic regression: class-balanced scikit-learn pipeline using amount, "
            "product, and email-domain attributes.",
            "3. Tabular + graph logistic regression: the same classifier family with 28 graph "
            "features, isolating the incremental value of graph context.",
            "4. Tabular + graph LightGBM: a nonlinear candidate using the same 48 transformed "
            "inputs, with early stopping on validation data.",
            "",
            "## Frozen test results",
            "",
            "| Approach | PR-AUC | Precision @ 5% | Recall @ 5% | Recall top 1% | "
            "Recall top 5% | FPR @ 5% |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *table_rows,
            "",
            "PR-AUC is the primary ranking metric because fraud labels are imbalanced. Capacity "
            "metrics answer the operational question: what is recovered when investigators can "
            "review only a small fraction of transactions?",
            "",
            "## Alert-volume comparison at matched recall",
            "",
            f"At a matched recall target of `{_percent(evidence['matched_recall_target'])}`:",
            "",
            "| Approach | Alerts required | Reduction versus rules |",
            "|---|---:|---:|",
            *matched_rows,
            "",
            "Both graph models required 3 alerts versus 4 for rules, a 25% alert-volume "
            "reduction on this synthetic test set. The tabular-only model required 5 alerts.",
            "",
            "## Model selection without test-set tuning",
            "",
            f"The selected model is `{selected}` because validation PR-AUC was "
            f"`{selection_scores[selected]:.4f}`, compared with "
            f"`{selection_scores['tabular_graph_logistic']:.4f}` for graph logistic regression. "
            "After selection, the frozen test showed graph logistic regression slightly ahead on "
            f"PR-AUC (`{models['tabular_graph_logistic']['pr_auc']:.4f}` versus "
            f"`{models[selected]['pr_auc']:.4f}`). The project does not switch models after seeing "
            "that test result; it reports the result as sampling uncertainty and a reason to "
            "validate on real data.",
            "",
            f"LightGBM stopped at iteration `{evidence['boosted_training']['best_iteration']}` of "
            f"`{evidence['boosted_training']['configured_estimators']}` configured estimators. "
            "That very early stopping point is another warning against over-interpreting the small "
            "synthetic experiment.",
            "",
            "## Selected-model errors and explanations",
            "",
            f"At the fixed top-5% capacity, the selected model produced "
            f"`{outcomes['true_positives']}` true positives, "
            f"`{outcomes['false_positives']}` false positives, and "
            f"`{outcomes['false_negatives']}` false negatives among "
            f"`{evidence['test_fraud_labels']}` fraud-labelled test transactions.",
            "",
            f"Tree SHAP explained `{evidence['shap']['explained_rows']}` test rows across "
            f"`{evidence['shap']['feature_count']}` transformed features. The maximum raw-score "
            "reconstruction error was "
            f"`{evidence['shap']['maximum_raw_score_reconstruction_error']:.3e}`. SHAP explains "
            "the fitted model's score; it does not establish causality or misconduct.",
            "",
            "## Capacity-tie interpretation",
            "",
            f"`{tie['rows_at_cutoff']}` transactions shared the selected model's cutoff score, "
            f"while `{tie['selected_from_cutoff_tie']}` positions remained at that score. Stable "
            "chronological order makes the top-k queue reproducible, but investigators must not "
            "interpret the order among tied transactions as a measured risk difference. Applying "
            "the validation threshold without a hard capacity cap would produce a different alert "
            "count, so threshold and top-k results are reported separately.",
            "",
            "## Conclusion",
            "",
            "The controlled ablation supports the project hypothesis on synthetic data: graph "
            "context materially improves prioritisation over tabular-only scoring. The correct "
            "next evidence step is to rerun the unchanged protocol on IEEE-CIS, inspect temporal "
            "stability and graph density, and calibrate operational choices with investigators. "
            "A GraphSAGE or attention model remains optional until this core baseline is "
            "validated.",
            "",
        ]
    )


def audit_documentation(project_root: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    """Check that the interview documents exist and preserve key evidence boundaries."""

    paths = {
        "readme": project_root / "README.md",
        "architecture": project_root / "docs" / "architecture.md",
        "model_card": project_root / "docs" / "model_card.md",
        "limitations": project_root / "docs" / "limitations.md",
        "experiment_report": project_root / "reports" / "experiment_report.md",
    }
    missing = [str(path.relative_to(project_root)) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Day 9 documents: {', '.join(missing)}")
    documents = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    combined = "\n".join(documents.values()).lower()
    model_metrics = evidence["models"]
    gates = {
        "all_documents_nonempty": all(len(text.strip()) >= 500 for text in documents.values()),
        "architecture_is_visual": "```mermaid" in documents["architecture"],
        "readme_marks_day9_complete": "(Day 9)" in documents["readme"]
        and "run_day9.py" in documents["readme"],
        "synthetic_scope_disclosed": all(
            "synthetic" in documents[name].lower()
            for name in ("model_card", "limitations", "experiment_report")
        ),
        "suspected_ring_language_present": "suspected fraud ring" in combined,
        "human_review_boundary_present": "human review" in combined and "automatically" in combined,
        "score_not_probability_disclosed": "not a calibrated" in combined,
        "selected_model_reported": evidence["model_selection"]["selected_model"]
        in documents["experiment_report"],
        "core_metrics_match_evidence": all(
            f"{model_metrics[name]['pr_auc']:.4f}" in documents["experiment_report"]
            for name in MODEL_ORDER
        ),
        "frozen_test_selection_disclosed": "does not switch models"
        in documents["experiment_report"],
        "ieee_ring_label_limit_disclosed": "does not provide verified fraud-ring labels"
        in combined,
    }
    forbidden_assertions = (
        "these clusters are confirmed criminal networks",
        "the system automatically blocks accounts",
        "proven performance at icici bank",
    )
    gates["unsupported_assertions_absent"] = not any(
        claim in combined for claim in forbidden_assertions
    )
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "documents": {
            name: {
                "path": str(path.relative_to(project_root)).replace("\\", "/"),
                "characters": len(documents[name]),
            }
            for name, path in paths.items()
        },
        "evidence_status": evidence["status"],
    }
