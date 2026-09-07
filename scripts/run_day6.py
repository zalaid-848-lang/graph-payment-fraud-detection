"""Run the Day 6 LightGBM comparison and post-test error analysis."""

from __future__ import annotations

import json
import sys
import tomllib
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import lightgbm
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.data.schema import validate_transactions  # noqa: E402
from fraud_detection.error_analysis import build_error_analysis  # noqa: E402
from fraud_detection.evaluation import (  # noqa: E402
    evaluate_ranking,
    minimum_alerts_for_recall,
    select_threshold_at_capacity,
    threshold_metrics,
)
from fraud_detection.features import (  # noqa: E402
    MaturedLabelFeatureBuilder,
    attach_graph_features,
)
from fraud_detection.model import LightGBMGraphModel  # noqa: E402

DISPLAY_NAMES = {
    "rules": "Rules only",
    "tabular_logistic": "Tabular logistic",
    "tabular_graph_logistic": "Tabular + graph logistic",
    "tabular_graph_lightgbm": "Tabular + graph LightGBM",
}


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


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


def _slice_lines(error_analysis: dict[str, Any]) -> list[str]:
    lines = []
    for family, slices in error_analysis["families"].items():
        lines.extend(
            [
                f"### {family.replace('_', ' ').title()}",
                "",
                "| Slice | Rows | Fraud labels | Alerts | False positives | "
                "False negatives | Recall within slice |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, summary in slices.items():
            lines.append(
                f"| `{name}` | {summary['rows']} | {summary['fraud_labels']} | "
                f"{summary['alert_count']} | {summary['false_positives']} | "
                f"{summary['false_negatives']} | "
                f"{_percent(summary['recall_within_slice'])} |"
            )
        lines.append("")
    return lines


def _render_report(result: dict[str, Any]) -> str:
    models = result["models"]
    increment = result["boosted_increment_vs_graph_logistic"]
    error = result["error_analysis"]
    overall = error["overall"]
    high_connectivity = error["families"]["specific_connectivity"]["above_train_p90"]
    connectivity_rows = sum(
        value["rows"] for value in error["families"]["specific_connectivity"].values()
    )
    direction = "improved" if increment["pr_auc_delta"] >= 0 else "reduced"
    lines = [
        "# Day 6 boosted-model comparison and error analysis",
        "",
        "## Executive summary",
        "",
        "A deterministic LightGBM candidate was added to the frozen Day 5 comparison. Its "
        "encoder is fitted on training rows, validation labels select the boosting iteration and "
        "operating threshold, and test labels are opened only for final metrics and error slices.",
        "",
        "> **Development-data warning:** These results use a small, deliberately patterned "
        "synthetic dataset. They validate the workflow and are not expected bank performance.",
        "",
        "## Untouched test-period comparison",
        "",
        "| Model | PR-AUC | Precision @ capacity | Recall @ capacity | FPR @ capacity | "
        "Recall top 1% | Recall top 5% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, display_name in DISPLAY_NAMES.items():
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
            f"Relative to the graph-logistic ablation, LightGBM {direction} PR-AUC by "
            f"`{abs(increment['pr_auc_delta']):.4f}` and changed recall at capacity by "
            f"`{increment['capacity_recall_delta']:+.2%}`.",
            "",
            "## LightGBM training contract",
            "",
            f"- Library: LightGBM `{result['boosted_training']['library_version']}`",
            f"- Validation-selected best iteration: "
            f"`{result['boosted_training']['best_iteration']}` of "
            f"`{result['boosted_training']['configured_estimators']}`",
            f"- Encoded feature count: `{result['boosted_training']['feature_count']}`",
            "- Class balancing is learned from training labels only.",
            "- Deterministic, single-threaded CPU settings are used for reproducibility.",
            "- PageRank snapshot age is audit metadata and is excluded from model inputs.",
            "",
            "## Error analysis at fixed investigation capacity",
            "",
            f"The global top-{error['review_count']} review set contains "
            f"`{overall['true_positives']}` fraud labels and "
            f"`{overall['false_positives']}` false positives; "
            f"`{overall['false_negatives']}` fraud labels fall outside capacity.",
            "",
            "Slices are descriptive post-test diagnostics. They do not change the model, "
            "threshold, or capacity. Numeric boundaries come only from the training period.",
            "",
        ]
    )
    lines.extend(_slice_lines(error))
    lines.extend(
        [
            "## Temporal accumulation finding",
            "",
            f"`{high_connectivity['rows']}` of `{connectivity_rows}` "
            "test transactions exceed the training-period 90th percentile of historical "
            "specific connectivity. Graph counts naturally accumulate over time, so later work "
            "should test rolling windows or age-normalized features on the real dataset.",
            "",
            "## Gain-based feature importance",
            "",
            "Gain measures how much fitted splits used a feature. It is neither a causal "
            "explanation nor an investigator reason code.",
            "",
            "| Feature | Normalized gain |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| `{name}` | {importance:.4f} |"
        for name, importance in result["boosted_feature_importance"]
    )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- Fraud labels apply to transactions, not verified fraud rings.",
            "- Slice differences on this small synthetic test set are unstable.",
            "- Shared entities may represent legitimate households, offices, or infrastructure.",
            "- The 5% capacity and 24-hour label delay are project scenarios, not bank policies.",
            "- Scores prioritize human review and never authorize automatic account blocking.",
            "",
            "## Day 7 handoff",
            "",
            "Add SHAP-based local explanations and stable investigator reason codes, then verify "
            "that every displayed explanation uses information available at scoring time.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> tuple[dict[str, Any], Path]:
    with (PROJECT_ROOT / "configs" / "project.toml").open("rb") as file:
        config = tomllib.load(file)
    data_path = PROJECT_ROOT / config["data"]["processed_file"]
    graph_path = PROJECT_ROOT / "data" / "processed" / "graph_features.csv"
    day5_metrics_path = PROJECT_ROOT / "artifacts" / "day5" / "metrics.json"
    required_paths = (data_path, graph_path, day5_metrics_path)
    if not all(path.is_file() for path in required_paths):
        raise FileNotFoundError("Day 6 requires prepared data and completed Day 4–5 artifacts")

    frame = validate_transactions(pd.read_csv(data_path))
    structural = pd.read_csv(graph_path)
    baseline_metrics = json.loads(day5_metrics_path.read_text(encoding="utf-8"))
    label_builder = MaturedLabelFeatureBuilder(
        label_maturity_seconds=int(config["graph"]["label_maturity_seconds"]),
        eligible_label_split="train",
    )
    label_history = label_builder.build(frame)
    combined = attach_graph_features(frame, structural, label_history)
    train = combined[combined["split"] == "train"].copy()
    validation = combined[combined["split"] == "validation"].copy()
    test = combined[combined["split"] == "test"].copy()
    capacity = float(config["evaluation"]["investigation_capacity_fraction"])
    if (
        baseline_metrics["rows"]
        != {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        }
        or baseline_metrics["capacity_fraction"] != capacity
    ):
        raise RuntimeError("Day 5 metrics do not match the current dataset or capacity")

    boosted = LightGBMGraphModel(**config["boosted_model"]).fit(train, validation)
    validation_scores = boosted.score(validation)
    test_scores = boosted.score(test)
    boosted_metrics = _evaluate_model(test["is_fraud"], validation_scores, test_scores, capacity)
    models = deepcopy(baseline_metrics["models"])
    models["tabular_graph_lightgbm"] = boosted_metrics

    reference_recall = models["rules"]["ranking"]["at_investigation_capacity"]["recall"]
    boosted_matched = minimum_alerts_for_recall(
        test["is_fraud"].to_numpy(), test_scores, reference_recall
    )
    matched_recall = deepcopy(baseline_metrics["matched_recall"])
    rule_alerts = matched_recall["models"]["rules"]["alert_count"]
    boosted_matched["alert_volume_reduction_vs_rules"] = (
        1.0 - boosted_matched["alert_count"] / rule_alerts if rule_alerts else 0.0
    )
    matched_recall["models"]["tabular_graph_lightgbm"] = boosted_matched

    error_analysis = build_error_analysis(
        train,
        test,
        test_scores,
        capacity_fraction=capacity,
    )
    if error_analysis["status"] != "PASS":
        raise RuntimeError("Day 6 error-analysis audit failed")

    logistic_ranking = models["tabular_graph_logistic"]["ranking"]
    boosted_ranking = boosted_metrics["ranking"]
    result: dict[str, Any] = {
        "source": str(frame["source"].iloc[0]),
        "capacity_fraction": capacity,
        "rows": baseline_metrics["rows"],
        "models": models,
        "matched_recall": matched_recall,
        "boosted_increment_vs_graph_logistic": {
            "pr_auc_delta": boosted_ranking["pr_auc"] - logistic_ranking["pr_auc"],
            "capacity_recall_delta": (
                boosted_ranking["at_investigation_capacity"]["recall"]
                - logistic_ranking["at_investigation_capacity"]["recall"]
            ),
        },
        "boosted_training": {
            "library": "lightgbm",
            "library_version": lightgbm.__version__,
            "best_iteration": boosted.best_iteration_,
            "configured_estimators": boosted.n_estimators,
            "feature_count": boosted.feature_count_,
            "parameters": asdict(boosted),
        },
        "boosted_feature_importance": boosted.feature_importance(limit=15),
        "error_analysis": error_analysis,
    }

    artifact_dir = PROJECT_ROOT / "artifacts" / "day6"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metrics_and_errors.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    score_rows = test[["transaction_id", "transaction_time", "split", "is_fraud"]].copy()
    score_rows["tabular_graph_lightgbm_score"] = test_scores
    score_rows.to_csv(artifact_dir / "test_scores.csv", index=False, lineterminator="\n")
    report_path = PROJECT_ROOT / "reports" / "day6_boosted_error_analysis.md"
    report_path.write_text(_render_report(result), encoding="utf-8")
    return result, report_path


def main() -> None:
    result, report_path = run()
    metrics = result["models"]["tabular_graph_lightgbm"]["ranking"]
    capacity = metrics["at_investigation_capacity"]
    print(
        f"lightgbm: PR-AUC={metrics['pr_auc']:.4f}, "
        f"precision@capacity={capacity['precision']:.2%}, "
        f"recall@capacity={capacity['recall']:.2%}"
    )
    print(
        f"Error audit: {result['error_analysis']['status']}; "
        f"best_iteration={result['boosted_training']['best_iteration']}"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
