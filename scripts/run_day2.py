"""Run Day 2 quality checks, baselines, and capacity-aware evaluation."""

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
from fraud_detection.model import TabularBaseline  # noqa: E402
from fraud_detection.quality import profile_transactions, render_quality_markdown  # noqa: E402
from fraud_detection.rules import RulesBaseline  # noqa: E402


def _percent(value: float) -> str:
    return f"{value:.2%}"


def _model_section(name: str, metrics: dict[str, Any]) -> list[str]:
    capacity = metrics["ranking"]["at_investigation_capacity"]
    threshold = metrics["threshold_operating_point"]
    return [
        f"### {name}",
        "",
        f"- PR-AUC: `{metrics['ranking']['pr_auc']:.4f}`",
        f"- Precision at capacity: `{_percent(capacity['precision'])}`",
        f"- Recall at capacity: `{_percent(capacity['recall'])}`",
        f"- False-positive rate at capacity: `{_percent(capacity['false_positive_rate'])}`",
        f"- Recall in top 1%: `{_percent(metrics['ranking']['top_1_percent']['recall'])}`",
        f"- Recall in top 5%: `{_percent(metrics['ranking']['top_5_percent']['recall'])}`",
        f"- Test alerts at validation-selected threshold: `{threshold['alert_count']}` "
        f"(`{_percent(threshold['alert_fraction'])}` of test rows)",
        "",
    ]


def _render_report(
    *,
    source: str,
    quality: dict[str, Any],
    metrics: dict[str, Any],
    coefficients: list[tuple[str, float]],
) -> str:
    rules = metrics["models"]["rules"]
    tabular = metrics["models"]["tabular_logistic"]
    lines = [
        "# Day 2 experiment report",
        "",
        "## Executive summary",
        "",
        "This report establishes two pre-graph benchmarks: a transparent rules score and a "
        "regularized tabular logistic-regression score. All fitting uses only the training period; "
        "validation selects operating thresholds; test labels are used only for final evaluation.",
        "",
    ]
    if source == "synthetic":
        lines.extend(
            [
                "> **Development-data warning:** These results use deliberately enriched synthetic "
                "fraud patterns. They verify code and evaluation logic; they are not evidence of "
                "real-world fraud performance.",
                "",
            ]
        )
    lines.extend(["## Data quality", "", render_quality_markdown(quality), ""])
    lines.extend(
        [
            "## Experimental design",
            "",
            f"- Investigation capacity: `{_percent(metrics['capacity_fraction'])}` of transactions",
            "- Rules: high/very-high amount plus device or recipient unseen in training",
            "- Tabular features: log amount, entity-availability flags, product code, and payer/recipient "
            "email-domain categories",
            "- Direct transaction/entity identifiers and target aliases are excluded",
            "- Purge-window rows are excluded from training, threshold selection, and testing",
            "",
            "## Untouched test-period results",
            "",
        ]
    )
    lines.extend(_model_section("Rules baseline", rules))
    lines.extend(_model_section("Tabular logistic baseline", tabular))
    rules_capacity = rules["ranking"]["at_investigation_capacity"]
    tabular_capacity = tabular["ranking"]["at_investigation_capacity"]
    lines.extend(
        [
            "The tabular model improves overall PR-AUC, but at the exact investigation-capacity "
            f"cutoff it finds `{tabular_capacity['true_positives']}` fraud labels versus "
            f"`{rules_capacity['true_positives']}` for the rules. This is why model selection must "
            "consider the operating point rather than PR-AUC alone.",
            "",
        ]
    )
    threshold_change = metrics["threshold_alert_volume_change_vs_rules"]
    matched = metrics["matched_recall_comparison"]
    lines.extend(
        [
            "## Investigator-volume comparison",
            "",
            f"At their separate validation-selected thresholds, the tabular model emits "
            f"`{_percent(threshold_change)}` fewer test alerts than the rules baseline. Its recall at "
            f"that point is `{_percent(tabular['threshold_operating_point']['recall'])}`, versus "
            f"`{_percent(rules['threshold_operating_point']['recall'])}` for the rules, so the raw "
            "volume change must not be presented as an efficiency gain.",
            "",
            f"At matched recall (`{_percent(matched['target_recall'])}`), the tabular ranking needs "
            f"`{matched['tabular_alert_count']}` alerts versus `{matched['rules_alert_count']}` rule "
            f"alerts: an alert-volume reduction of `{_percent(matched['alert_volume_reduction'])}`. "
            "A negative value means the model needs more reviews to match the rules.",
            "These are project-scenario measurements, not a bank policy recommendation.",
            "",
            "## Strongest tabular coefficients",
            "",
            "Positive coefficients increase the model score and negative coefficients decrease it. "
            "They are associations in this development sample, not causal explanations.",
            "",
            "| Feature | Coefficient |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| `{name}` | {value:.4f} |" for name, value in coefficients)
    lines.extend(
        [
            "",
            "## Day 2 conclusion",
            "",
            "The project now has a fair, reproducible pre-graph benchmark. Day 3 will construct the "
            "historical entity-link graph and test that missing values never create false links. "
            "Only after those integrity checks will graph features be compared against these baselines.",
            "",
            "## Limitations",
            "",
            "- Labels describe transactions, not verified fraud rings.",
            "- Customer and recipient fields are proxies for IEEE-CIS data.",
            "- Shared or previously unseen entities can be legitimate.",
            "- Rules and thresholds are project scenarios, not ICICI Bank practices.",
            "- Investigator review remains required; no automated blocking action is recommended.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> tuple[dict[str, Any], Path]:
    config_path = PROJECT_ROOT / "configs" / "project.toml"
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    data_path = PROJECT_ROOT / config["data"]["processed_file"]
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Prepared data not found at {data_path}. Run python scripts/prepare_data.py first."
        )

    frame = validate_transactions(pd.read_csv(data_path))
    quality = profile_transactions(frame)
    if quality["status"] != "PASS":
        raise RuntimeError("Data-quality gates failed; inspect the generated quality profile")

    train = frame[frame["split"] == "train"].copy()
    validation = frame[frame["split"] == "validation"].copy()
    test = frame[frame["split"] == "test"].copy()
    capacity = float(config["evaluation"]["investigation_capacity_fraction"])

    rules = RulesBaseline(**config["rules"]).fit(train)
    rules_validation_score = rules.score(validation)
    rules_test_score = rules.score(test)

    tabular = TabularBaseline(**config["model"]).fit(train)
    tabular_validation_score = tabular.score(validation)
    tabular_test_score = tabular.score(test)

    model_scores = {
        "rules": (rules_validation_score, rules_test_score),
        "tabular_logistic": (tabular_validation_score, tabular_test_score),
    }
    model_metrics: dict[str, Any] = {}
    for name, (validation_score, test_score) in model_scores.items():
        threshold = select_threshold_at_capacity(validation_score, capacity)
        model_metrics[name] = {
            "validation_selected_threshold": threshold,
            "ranking": evaluate_ranking(
                test["is_fraud"].to_numpy(), test_score, capacity_fraction=capacity
            ),
            "threshold_operating_point": threshold_metrics(
                test["is_fraud"].to_numpy(), test_score, threshold
            ),
        }

    rule_alerts = model_metrics["rules"]["threshold_operating_point"]["alert_count"]
    tabular_alerts = model_metrics["tabular_logistic"]["threshold_operating_point"][
        "alert_count"
    ]
    threshold_alert_change = 1.0 - (tabular_alerts / rule_alerts) if rule_alerts else 0.0
    rules_recall = model_metrics["rules"]["threshold_operating_point"]["recall"]
    tabular_matched = minimum_alerts_for_recall(
        test["is_fraud"].to_numpy(), tabular_test_score, rules_recall
    )
    matched_reduction = (
        1.0 - tabular_matched["alert_count"] / rule_alerts if rule_alerts else 0.0
    )
    metrics: dict[str, Any] = {
        "source": str(frame["source"].iloc[0]),
        "capacity_fraction": capacity,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "models": model_metrics,
        "threshold_alert_volume_change_vs_rules": threshold_alert_change,
        "matched_recall_comparison": {
            "target_recall": rules_recall,
            "rules_alert_count": rule_alerts,
            "tabular_alert_count": tabular_matched["alert_count"],
            "alert_volume_reduction": matched_reduction,
        },
        "tabular_training": {
            "iterations": tabular.model_.iterations_,
            "converged": tabular.model_.converged_,
            "training_loss": tabular.model_.training_loss_,
            "feature_count": len(tabular.encoder_.feature_names_),
        },
        "strongest_coefficients": tabular.strongest_coefficients(),
    }

    artifact_dir = PROJECT_ROOT / "artifacts" / "day2"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "data_quality.json").write_text(
        json.dumps(quality, indent=2) + "\n", encoding="utf-8"
    )
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    score_rows = pd.concat([validation, test], ignore_index=True)[
        ["transaction_id", "transaction_time", "split", "is_fraud"]
    ].copy()
    score_rows["rules_score"] = pd.concat(
        [pd.Series(rules_validation_score), pd.Series(rules_test_score)], ignore_index=True
    )
    score_rows["tabular_score"] = pd.concat(
        [pd.Series(tabular_validation_score), pd.Series(tabular_test_score)], ignore_index=True
    )
    score_rows["rule_reasons"] = rules.reasons(pd.concat([validation, test], ignore_index=True))
    score_rows.to_csv(artifact_dir / "scores.csv", index=False, lineterminator="\n")

    report_path = PROJECT_ROOT / "reports" / "day2_experiment.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(
            source=metrics["source"],
            quality=quality,
            metrics=metrics,
            coefficients=metrics["strongest_coefficients"],
        ),
        encoding="utf-8",
    )
    return metrics, report_path


def main() -> None:
    metrics, report_path = run()
    for name, result in metrics["models"].items():
        capacity = result["ranking"]["at_investigation_capacity"]
        print(
            f"{name}: PR-AUC={result['ranking']['pr_auc']:.4f}, "
            f"precision@capacity={capacity['precision']:.2%}, "
            f"recall@capacity={capacity['recall']:.2%}"
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
