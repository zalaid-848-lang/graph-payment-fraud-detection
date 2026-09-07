"""Run Day 7 SHAP fidelity checks and investigator reason-code generation."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.data.schema import validate_transactions  # noqa: E402
from fraud_detection.evaluation import average_precision  # noqa: E402
from fraud_detection.explainability import (  # noqa: E402
    REASON_CATALOG,
    InvestigatorReasonBuilder,
    TreeShapAttributor,
    global_shap_importance,
)
from fraud_detection.features import (  # noqa: E402
    MaturedLabelFeatureBuilder,
    attach_graph_features,
)
from fraud_detection.model import (  # noqa: E402
    LightGBMGraphModel,
    TabularGraphBaseline,
)


def _render_report(result: dict[str, Any]) -> str:
    selection = result["model_selection"]
    shap_audit = result["shap_audit"]
    reasons = result["reason_code_summary"]
    tie_audit = reasons["capacity_tie_audit"]
    lines = [
        "# Day 7 SHAP explanations and investigator reason codes",
        "",
        "## Executive summary",
        "",
        "The graph LightGBM model is selected using validation PR-AUC before final test reporting. "
        "Exact Tree SHAP values are converted into factual, evidence-gated reasons for the "
        "capacity-ranked alerts.",
        "",
        "> The displayed risk score is a ranking score from a class-balanced model, not a "
        "calibrated probability of fraud. Explanations support investigator review; they do not "
        "confirm fraud-ring membership or authorize automatic blocking.",
        "",
        "## Model selection",
        "",
        "| Candidate | Validation PR-AUC | Frozen test PR-AUC | "
        "Test recall @ capacity | Selected |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, candidate in selection["candidates"].items():
        lines.append(
            f"| `{name}` | {candidate['validation_pr_auc']:.4f} | "
            f"{candidate['test_pr_auc']:.4f} | "
            f"{candidate['test_recall_at_capacity']:.2%} | "
            f"{'yes' if name == selection['selected_model'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Selection uses validation PR-AUC. The test comparison is frozen from Day 6 and is "
            "reported only after selection; explanations do not trigger model retuning.",
            "",
            "## SHAP fidelity audit",
            "",
            f"- SHAP version: `{result['shap_version']}`",
            f"- Background: `{shap_audit['background_rows']}` training rows",
            f"- Explained rows: `{shap_audit['explained_rows']}` test rows",
            f"- Explained features: `{shap_audit['feature_count']}`",
            "- Maximum raw-score reconstruction error: "
            f"`{shap_audit['maximum_raw_score_reconstruction_error']:.3e}`",
            f"- Fidelity status: **{shap_audit['status']}**",
            "",
            "SHAP values sum with the base value to reproduce the fitted LightGBM raw score. "
            "This verifies model fidelity, not causal validity.",
            "",
            "## Investigator reason-code coverage",
            "",
            f"Generated explanations for `{reasons['review_count']}` capacity-selected alerts, "
            f"with at most `{reasons['max_reasons_per_alert']}` positive reasons per alert.",
            "",
            "| Reason code | Investigator label | Selected-alert count |",
            "|---|---|---:|",
        ]
    )
    for code, count in reasons["reason_code_counts"].items():
        lines.append(f"| `{code}` | {REASON_CATALOG[code]} | {count} |")
    lines.extend(
        [
            "",
            "All reasons require both a positive SHAP contribution and factual evidence in the "
            "transaction's scoring-time feature row. For example, the matured-fraud-neighbour "
            "reason cannot appear unless the matured neighbour count is greater than zero.",
            "",
            "## Capacity-cutoff tie audit",
            "",
            f"`{tie_audit['rows_at_cutoff']}` transactions share the cutoff score, while "
            f"`{tie_audit['selected_from_cutoff_tie']}` positions remain at that score. "
            "The artifact retains stable chronological transaction order for reproducibility, "
            "but that order must not be interpreted as a model-measured risk difference.",
            "",
            "## Global mean absolute SHAP importance",
            "",
            "Global importance summarizes model reliance across the test period. It does not "
            "replace transaction-level reasons.",
            "",
            "| Feature | Normalized mean absolute SHAP |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| `{name}` | {importance:.4f} |" for name, importance in result["global_shap_importance"]
    )
    lines.extend(
        [
            "",
            "## Safety and interpretation boundaries",
            "",
            "- Only features available at scoring time are explained.",
            "- Validation, test, and purge labels never enter neighbour-label features.",
            "- Positive SHAP values explain the model score; they do not prove misconduct.",
            "- Shared entities may have legitimate explanations.",
            "- Investigator-facing records omit the transaction's evaluation label.",
            "- Recommendations remain human-reviewed and never automate account blocking.",
            "",
            "## Day 8 handoff",
            "",
            "Build the Streamlit investigation dashboard around the audited alert artifact, with "
            "ranked alerts, local reasons, and a focused connection view for suspected fraud "
            "rings.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> tuple[dict[str, Any], Path]:
    with (PROJECT_ROOT / "configs" / "project.toml").open("rb") as file:
        config = tomllib.load(file)
    data_path = PROJECT_ROOT / config["data"]["processed_file"]
    graph_path = PROJECT_ROOT / "data" / "processed" / "graph_features.csv"
    day6_scores_path = PROJECT_ROOT / "artifacts" / "day6" / "test_scores.csv"
    day6_metrics_path = PROJECT_ROOT / "artifacts" / "day6" / "metrics_and_errors.json"
    required_paths = (data_path, graph_path, day6_scores_path, day6_metrics_path)
    if not all(path.is_file() for path in required_paths):
        raise FileNotFoundError("Day 7 requires prepared data and completed Day 4–6 artifacts")

    frame = validate_transactions(pd.read_csv(data_path))
    structural = pd.read_csv(graph_path)
    frozen_scores = pd.read_csv(day6_scores_path)
    day6_metrics = json.loads(day6_metrics_path.read_text(encoding="utf-8"))

    label_builder = MaturedLabelFeatureBuilder(
        label_maturity_seconds=int(config["graph"]["label_maturity_seconds"]),
        eligible_label_split="train",
    )
    labels = label_builder.build(frame)
    combined = attach_graph_features(frame, structural, labels)
    train = combined[combined["split"] == "train"].copy()
    validation = combined[combined["split"] == "validation"].copy()
    test = combined[combined["split"] == "test"].copy()
    logistic = TabularGraphBaseline(**config["model"]).fit(train)
    boosted = LightGBMGraphModel(**config["boosted_model"]).fit(train, validation)
    validation_target = validation["is_fraud"].to_numpy()
    validation_pr_auc = {
        "tabular_graph_logistic": average_precision(validation_target, logistic.score(validation)),
        "tabular_graph_lightgbm": average_precision(validation_target, boosted.score(validation)),
    }
    candidate_names = ("tabular_graph_logistic", "tabular_graph_lightgbm")
    candidates = {
        name: {
            "validation_pr_auc": validation_pr_auc[name],
            "test_pr_auc": day6_metrics["models"][name]["ranking"]["pr_auc"],
            "test_recall_at_capacity": day6_metrics["models"][name]["ranking"][
                "at_investigation_capacity"
            ]["recall"],
        }
        for name in candidate_names
    }
    selected_model = str(config["explainability"]["selected_model"])
    highest_validation_model = max(
        candidates, key=lambda name: candidates[name]["validation_pr_auc"]
    )
    if selected_model != highest_validation_model or selected_model != "tabular_graph_lightgbm":
        raise RuntimeError("Configured explanation model does not match validation selection")

    expected_scores = (
        frozen_scores.set_index("transaction_id")
        .loc[test["transaction_id"], "tabular_graph_lightgbm_score"]
        .to_numpy(dtype=float)
    )
    reproduced_scores = boosted.score(test)
    maximum_score_difference = float(np.max(np.abs(reproduced_scores - expected_scores)))
    if maximum_score_difference > 1e-12:
        raise RuntimeError("Day 7 model scores do not reproduce the frozen Day 6 scores")

    attribution = TreeShapAttributor().fit(boosted, train).explain(test)
    reason_output = (
        InvestigatorReasonBuilder()
        .fit(train)
        .build(
            test,
            attribution,
            capacity_fraction=float(config["evaluation"]["investigation_capacity_fraction"]),
            max_reasons=int(config["explainability"]["max_reasons"]),
        )
    )
    if reason_output["status"] != "PASS":
        raise RuntimeError("Investigator reason-code audit failed")

    reason_summary = {key: value for key, value in reason_output.items() if key != "alerts"}
    result: dict[str, Any] = {
        "source": str(frame["source"].iloc[0]),
        "model_selection": {
            "selected_model": selected_model,
            "selection_metric": "validation_pr_auc",
            "candidates": candidates,
        },
        "score_reproduction": {
            "status": "PASS",
            "maximum_absolute_difference": maximum_score_difference,
        },
        "shap_version": shap.__version__,
        "shap_audit": attribution.audit,
        "reason_code_summary": reason_summary,
        "global_shap_importance": global_shap_importance(attribution, limit=15),
    }

    artifact_dir = PROJECT_ROOT / "artifacts" / "day7"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "explanation_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (artifact_dir / "investigator_explanations.json").write_text(
        json.dumps(reason_output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path = PROJECT_ROOT / "reports" / "day7_explainability.md"
    report_path.write_text(_render_report(result), encoding="utf-8")
    return result, report_path


def main() -> None:
    result, report_path = run()
    audit = result["shap_audit"]
    reasons = result["reason_code_summary"]
    print(
        f"SHAP: {audit['explained_rows']} rows, {audit['feature_count']} features, "
        f"max_error={audit['maximum_raw_score_reconstruction_error']:.3e}"
    )
    print(f"Investigator reasons: {reasons['review_count']} alerts, status={reasons['status']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
