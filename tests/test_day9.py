import tempfile
import unittest
from pathlib import Path

from fraud_detection.reporting import (
    MODEL_ORDER,
    audit_documentation,
    build_evidence_snapshot,
    load_day9_sources,
    render_experiment_report,
)


def evidence_sources() -> dict:
    model_metrics = {}
    for index, name in enumerate(MODEL_ORDER):
        value = 0.30 + index * 0.10
        model_metrics[name] = {
            "ranking": {
                "pr_auc": value,
                "at_investigation_capacity": {
                    "precision": 0.5,
                    "recall": 0.4,
                    "false_positive_rate": 0.02,
                },
                "top_1_percent": {"recall": 0.1},
                "top_5_percent": {"recall": 0.4},
            }
        }
    matched = {
        name: {
            "alert_count": 4 - min(index, 1),
            "alert_volume_reduction_vs_rules": 0.0 if index == 0 else 0.25,
        }
        for index, name in enumerate(MODEL_ORDER)
    }
    return {
        "day5": {
            "graph_feature_count": 28,
            "label_history_audit": {
                "status": "PASS",
                "gates": {"target_columns_absent": True},
            },
        },
        "day6": {
            "source": "synthetic",
            "rows": {"train": 300, "validation": 84, "test": 84},
            "capacity_fraction": 0.05,
            "models": model_metrics,
            "matched_recall": {"target_recall": 0.4, "models": matched},
            "boosted_training": {
                "feature_count": 48,
                "library": "lightgbm",
                "library_version": "4.7.0",
                "best_iteration": 3,
                "configured_estimators": 400,
            },
            "error_analysis": {
                "status": "PASS",
                "gates": {"slice_rows_are_exhaustive": True},
                "overall": {
                    "fraud_labels": 8,
                    "true_positives": 5,
                    "false_positives": 0,
                    "false_negatives": 3,
                },
            },
        },
        "day7": {
            "model_selection": {
                "selected_model": "tabular_graph_lightgbm",
                "selection_metric": "validation_pr_auc",
                "candidates": {
                    "tabular_graph_logistic": {"validation_pr_auc": 0.70},
                    "tabular_graph_lightgbm": {"validation_pr_auc": 0.75},
                },
            },
            "shap_version": "0.52.0",
            "shap_audit": {
                "status": "PASS",
                "explained_rows": 84,
                "feature_count": 48,
                "maximum_raw_score_reconstruction_error": 1e-9,
            },
            "reason_code_summary": {
                "status": "PASS",
                "gates": {"review_only_action": True},
            },
        },
        "day8": {
            "status": "PASS",
            "gates": {"evaluation_targets_absent": True},
            "alert_count": 5,
            "connection_views_rendered": 5,
            "maximum_displayed_historical_neighbours": 20,
            "capacity_tie_audit": {
                "rows_at_cutoff": 7,
                "selected_from_cutoff_tie": 5,
            },
        },
    }


class Day9ReportingTests(unittest.TestCase):
    def test_evidence_snapshot_extracts_audited_metrics(self) -> None:
        snapshot = build_evidence_snapshot(evidence_sources())

        self.assertEqual(snapshot["status"], "PASS")
        self.assertEqual(snapshot["model_graph_feature_count"], 28)
        self.assertEqual(snapshot["model_total_transformed_feature_count"], 48)
        self.assertEqual(
            snapshot["model_selection"]["selected_model"],
            "tabular_graph_lightgbm",
        )
        self.assertTrue(all(snapshot["gates"].values()))

    def test_report_contains_comparison_and_test_selection_boundary(self) -> None:
        snapshot = build_evidence_snapshot(evidence_sources())
        report = render_experiment_report(snapshot)

        for name in MODEL_ORDER:
            self.assertIn(f"{snapshot['models'][name]['pr_auc']:.4f}", report)
        self.assertIn("does not switch models", report)
        self.assertIn("synthetic", report)
        self.assertIn("not a calibrated", report)
        self.assertNotIn("confirmed criminal networks", report)

    def test_documentation_audit_checks_scope_and_evidence(self) -> None:
        snapshot = build_evidence_snapshot(evidence_sources())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "reports").mkdir()
            filler = " Evidence remains subject to human review." * 30
            (root / "README.md").write_text(
                "Day 9 documentation complete (Day 9). Run scripts/run_day9.py. " + filler,
                encoding="utf-8",
            )
            (root / "docs" / "architecture.md").write_text(
                "```mermaid\ngraph LR\nA-->B\n```\nSuspected fraud ring. " + filler,
                encoding="utf-8",
            )
            (root / "docs" / "model_card.md").write_text(
                "Synthetic model; score is not a calibrated probability. It does not "
                "automatically block an account. Suspected fraud ring. " + filler,
                encoding="utf-8",
            )
            (root / "docs" / "limitations.md").write_text(
                "Synthetic evidence. IEEE-CIS does not provide verified fraud-ring labels. "
                + filler,
                encoding="utf-8",
            )
            (root / "reports" / "experiment_report.md").write_text(
                render_experiment_report(snapshot), encoding="utf-8"
            )

            audit = audit_documentation(root, snapshot)

        self.assertEqual(audit["status"], "PASS")
        self.assertTrue(all(audit["gates"].values()))

    def test_source_loader_reports_missing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Missing Day 9 evidence"):
                load_day9_sources(Path(directory))


if __name__ == "__main__":
    unittest.main()
