"""Run the Day 5 tabular-plus-graph model comparison."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.data.schema import validate_transactions  # noqa: E402
from fraud_detection.evaluation import (  # noqa: E402
    evaluate_ranking,
    minimum_alerts_for_recall,
    select_threshold_at_capacity,
    threshold_metrics,
)
from fraud_detection.features import (  # noqa: E402
    MODEL_GRAPH_FEATURE_COLUMNS,
    MaturedLabelFeatureBuilder,
    attach_graph_features,
)
from fraud_detection.model import TabularBaseline, TabularGraphBaseline  # noqa: E402
from fraud_detection.rules import RulesBaseline  # noqa: E402


def _percent(value: float) -> str:
    return f"{value:.2%}"


def _evaluate_model(
    target: pd.Series,
    validation_scores: Any,
    test_scores: Any,
    capacity: float,
) -> dict[str, Any]:
    threshold = select_threshold_at_capacity(validation_scores, capacity)
    return {
        "validation_selected_threshold": threshold,
        "ranking": evaluate_ranking(target.to_numpy(), test_scores, capacity_fraction=capacity),
        "threshold_operating_point": threshold_metrics(target.to_numpy(), test_scores, threshold),
    }


def _render_report(result: dict[str, Any]) -> str:
    models = result["models"]
    increment = result["graph_increment_vs_tabular"]
    direction = "improved" if increment["pr_auc_delta"] >= 0 else "reduced"
    lines = [
        "# Day 5 tabular-plus-graph model comparison",
        "",
        "## Executive summary",
        "",
        "A controlled ablation compares rules, tabular logistic regression, and the same "
        "logistic classifier with causal graph features. Keeping the classifier fixed isolates "
        "the incremental value of graph context.",
        "",
        "> **Development-data warning:** Results use a deliberately patterned synthetic sample. "
        "They verify the experimental design and cannot be presented as expected bank performance.",
        "",
        "Connected activity is described as a **suspected fraud ring**, never a confirmed "
        "criminal network. Scores support investigator review and never trigger automatic "
        "blocking.",
        "",
        "## Leakage-safe design",
        "",
        "- Training labels mature after "
        f"`{result['label_history_audit']['label_maturity_seconds']}` "
        "seconds before they may contribute to neighbour-label features.",
        "- Only labels from the training split are eligible; validation, test, and purge labels "
        "are ignored by the label-history builder.",
        "- Structural features use only earlier transaction-time batches.",
        "- Graph scaling, tabular encoding, class weighting, and model fitting use training "
        "rows only.",
        "- Validation selects thresholds; test labels are used only for this final comparison.",
        "- Direct entity identifiers are excluded from every model matrix.",
        "",
        "## Untouched test-period comparison",
        "",
        "| Model | PR-AUC | Precision @ capacity | Recall @ capacity | FPR @ capacity | "
        "Recall top 1% | Recall top 5% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    display_names = {
        "rules": "Rules only",
        "tabular_logistic": "Tabular logistic",
        "tabular_graph_logistic": "Tabular + graph logistic",
    }
    for name, display_name in display_names.items():
        ranking = models[name]["ranking"]
        capacity = ranking["at_investigation_capacity"]
        lines.append(
            f"| {display_name} | {ranking['pr_auc']:.4f} | "
            f"{_percent(capacity['precision'])} | {_percent(capacity['recall'])} | "
            f"{_percent(capacity['false_positive_rate'])} | "
            f"{_percent(ranking['top_1_percent']['recall'])} | "
            f"{_percent(ranking['top_5_percent']['recall'])} |"
        )
    lines.extend(
        [
            "",
            f"Adding graph context {direction} PR-AUC by "
            f"`{abs(increment['pr_auc_delta']):.4f}` and changed recall at the fixed capacity "
            f"by `{increment['capacity_recall_delta']:+.2%}` relative to the tabular model.",
            "",
            "## Alert-volume diagnostic at matched recall",
            "",
            f"Target recall is the rules model's recall at the fixed investigation capacity: "
            f"`{_percent(result['matched_recall']['target_recall'])}`.",
            "",
            "| Model | Minimum test alerts | Alert fraction | Reduction vs rules |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, display_name in display_names.items():
        matched = result["matched_recall"]["models"][name]
        lines.append(
            f"| {display_name} | {matched['alert_count']} | "
            f"{_percent(matched['alert_fraction'])} | "
            f"{_percent(matched['alert_volume_reduction_vs_rules'])} |"
        )
    lines.extend(
        [
            "",
            "This is a retrospective ranking diagnostic calculated with test labels, not an "
            "operating threshold or a staffing recommendation.",
            "",
            "## Validation-selected thresholds",
            "",
            "The following alert counts result when each validation-selected threshold is applied "
            "unchanged to the test period. Distribution shift and tied rule scores can move volume "
            "away from the nominal capacity.",
            "",
            "| Model | Test alerts | Alert fraction | Precision | Recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, display_name in display_names.items():
        threshold = models[name]["threshold_operating_point"]
        lines.append(
            f"| {display_name} | {threshold['alert_count']} | "
            f"{_percent(threshold['alert_fraction'])} | "
            f"{_percent(threshold['precision'])} | {_percent(threshold['recall'])} |"
        )
    lines.extend(
        [
            "",
            "## Strongest combined-model coefficients",
            "",
            "Coefficients are associations after preprocessing, not causal explanations. "
            "Correlated "
            "graph features can redistribute coefficient magnitude.",
            "",
            "| Feature | Coefficient |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| `{name}` | {coefficient:.4f} |"
        for name, coefficient in result["strongest_combined_coefficients"]
    )
    lines.extend(
        [
            "",
            "## Limitations and decision boundary",
            "",
            "- IEEE-CIS provides transaction labels, not verified fraud-ring labels.",
            "- A shared device, address, card proxy, or recipient proxy can be legitimate.",
            "- The 24-hour label delay is an explicit project scenario, not a claimed bank "
            "process.",
            "- Synthetic performance is useful only for pipeline verification.",
            "- Every escalation remains investigator-reviewed; the project does not block "
            "accounts.",
            "",
            "## Day 6 handoff",
            "",
            "Add a boosted-tree candidate and perform slice-based error analysis while preserving "
            "this logistic comparison as the controlled graph-feature ablation.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> tuple[dict[str, Any], Path]:
    with (PROJECT_ROOT / "configs" / "project.toml").open("rb") as file:
        config = tomllib.load(file)
    data_path = PROJECT_ROOT / config["data"]["processed_file"]
    graph_path = PROJECT_ROOT / "data" / "processed" / "graph_features.csv"
    if not data_path.is_file() or not graph_path.is_file():
        raise FileNotFoundError(
            "Prepared transactions and Day 4 graph features are required. Run Days 1 and 4 first."
        )

    frame = validate_transactions(pd.read_csv(data_path))
    structural = pd.read_csv(graph_path)
    label_builder = MaturedLabelFeatureBuilder(
        label_maturity_seconds=int(config["graph"]["label_maturity_seconds"]),
        eligible_label_split="train",
    )
    label_history = label_builder.build(frame)
    if label_builder.audit_["status"] != "PASS":
        raise RuntimeError("Label-history feature audit failed")
    combined = attach_graph_features(frame, structural, label_history)

    train = combined[combined["split"] == "train"].copy()
    validation = combined[combined["split"] == "validation"].copy()
    test = combined[combined["split"] == "test"].copy()
    capacity = float(config["evaluation"]["investigation_capacity_fraction"])

    rules = RulesBaseline(**config["rules"]).fit(train)
    tabular = TabularBaseline(**config["model"]).fit(train)
    combined_model = TabularGraphBaseline(**config["model"]).fit(train)
    validation_scores = {
        "rules": rules.score(validation),
        "tabular_logistic": tabular.score(validation),
        "tabular_graph_logistic": combined_model.score(validation),
    }
    test_scores = {
        "rules": rules.score(test),
        "tabular_logistic": tabular.score(test),
        "tabular_graph_logistic": combined_model.score(test),
    }
    models = {
        name: _evaluate_model(
            test["is_fraud"], validation_scores[name], test_scores[name], capacity
        )
        for name in validation_scores
    }

    reference_recall = models["rules"]["ranking"]["at_investigation_capacity"]["recall"]
    matched_models = {
        name: minimum_alerts_for_recall(test["is_fraud"].to_numpy(), scores, reference_recall)
        for name, scores in test_scores.items()
    }
    rules_alert_count = matched_models["rules"]["alert_count"]
    for matched in matched_models.values():
        matched["alert_volume_reduction_vs_rules"] = (
            1.0 - matched["alert_count"] / rules_alert_count if rules_alert_count else 0.0
        )

    tabular_ranking = models["tabular_logistic"]["ranking"]
    graph_ranking = models["tabular_graph_logistic"]["ranking"]
    result: dict[str, Any] = {
        "source": str(frame["source"].iloc[0]),
        "capacity_fraction": capacity,
        "rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "graph_feature_count": len(MODEL_GRAPH_FEATURE_COLUMNS),
        "label_history_audit": label_builder.audit_,
        "models": models,
        "graph_increment_vs_tabular": {
            "pr_auc_delta": graph_ranking["pr_auc"] - tabular_ranking["pr_auc"],
            "capacity_recall_delta": (
                graph_ranking["at_investigation_capacity"]["recall"]
                - tabular_ranking["at_investigation_capacity"]["recall"]
            ),
        },
        "matched_recall": {
            "target_recall": reference_recall,
            "models": matched_models,
        },
        "training": {
            "tabular_logistic": {
                "iterations": tabular.iterations_,
                "converged": tabular.converged_,
                "weighted_log_loss": tabular.training_loss_,
                "feature_count": len(tabular.encoder_.feature_names_),
            },
            "tabular_graph_logistic": {
                "iterations": combined_model.iterations_,
                "converged": combined_model.converged_,
                "weighted_log_loss": combined_model.training_loss_,
                "feature_count": len(combined_model.encoder_.feature_names_),
            },
        },
        "strongest_combined_coefficients": combined_model.strongest_coefficients(limit=12),
    }

    output_path = PROJECT_ROOT / "data" / "processed" / "label_history_features.csv"
    label_history.to_csv(output_path, index=False, lineterminator="\n")
    artifact_dir = PROJECT_ROOT / "artifacts" / "day5"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    score_rows = test[["transaction_id", "transaction_time", "split", "is_fraud"]].copy()
    for name, scores in test_scores.items():
        score_rows[f"{name}_score"] = scores
    score_rows["rule_reasons"] = rules.reasons(test)
    score_rows.to_csv(artifact_dir / "test_scores.csv", index=False, lineterminator="\n")

    report_path = PROJECT_ROOT / "reports" / "day5_model_comparison.md"
    report_path.write_text(_render_report(result), encoding="utf-8")
    return result, report_path


def main() -> None:
    result, report_path = run()
    for name, metrics in result["models"].items():
        capacity = metrics["ranking"]["at_investigation_capacity"]
        print(
            f"{name}: PR-AUC={metrics['ranking']['pr_auc']:.4f}, "
            f"precision@capacity={capacity['precision']:.2%}, "
            f"recall@capacity={capacity['recall']:.2%}"
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
