import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from fraud_detection.dashboard import (
    build_local_connection_graph,
    connection_figure,
    load_dashboard_bundle,
)
from fraud_detection.data.split import assign_temporal_splits
from fraud_detection.data.synthetic import generate_synthetic_transactions
from fraud_detection.features import CausalGraphFeatureBuilder, MaturedLabelFeatureBuilder


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary_directory.name)
        processed = self.project_root / "data" / "processed"
        artifact_dir = self.project_root / "artifacts" / "day7"
        processed.mkdir(parents=True)
        artifact_dir.mkdir(parents=True)

        frame = assign_temporal_splits(generate_synthetic_transactions(180))
        structural = CausalGraphFeatureBuilder(pagerank_refresh_batches=10).build(frame)
        label_history = MaturedLabelFeatureBuilder().build(frame)
        selected = frame[frame["split"] == "test"].iloc[-1]
        transaction_id = str(selected["transaction_id"])
        frame.to_csv(processed / "transactions.csv", index=False)
        structural.to_csv(processed / "graph_features.csv", index=False)
        label_history.to_csv(processed / "label_history_features.csv", index=False)
        artifact = {
            "status": "PASS",
            "gates": {"target_fields_absent": True, "review_only_actions": True},
            "model": "tabular_graph_lightgbm",
            "capacity_fraction": 0.05,
            "review_count": 1,
            "capacity_tie_audit": {
                "cutoff_score": 0.81,
                "rows_above_cutoff": 0,
                "rows_at_cutoff": 2,
                "selected_from_cutoff_tie": 1,
                "tie_break_required": True,
                "tie_break_rule": "stable chronological transaction order",
            },
            "disclaimer": (
                "Reasons support investigator review and do not establish fraud-ring membership."
            ),
            "alerts": [
                {
                    "rank": 1,
                    "transaction_id": transaction_id,
                    "risk_score": 0.81,
                    "recommended_action": "INVESTIGATOR_REVIEW",
                    "reasons": [
                        {
                            "code": "SHARED_SPECIFIC_ENTITY",
                            "title": "Repeated specific entity",
                            "detail": "Shares a specific entity with earlier activity.",
                            "evidence": {"earlier_connected_transactions": 2},
                        }
                    ],
                }
            ],
        }
        (artifact_dir / "investigator_explanations.json").write_text(
            json.dumps(artifact), encoding="utf-8"
        )
        self.selected_transaction = transaction_id

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_loader_strips_evaluation_target(self) -> None:
        bundle = load_dashboard_bundle(self.project_root)

        self.assertNotIn("is_fraud", bundle.transactions.columns)
        self.assertNotIn("is_fraud", bundle.alerts.columns)
        self.assertEqual(bundle.alerts.loc[0, "recommended_action"], "INVESTIGATOR_REVIEW")

    def test_connection_view_uses_strict_history_and_excludes_email_hubs(self) -> None:
        bundle = load_dashboard_bundle(self.project_root)
        selected_time = int(
            bundle.transactions.loc[
                bundle.transactions["transaction_id"] == self.selected_transaction,
                "transaction_time",
            ].iloc[0]
        )
        equal_time = bundle.transactions.iloc[[0]].copy()
        equal_time["transaction_id"] = "equal_time_peer"
        equal_time["transaction_time"] = selected_time
        graph, summary = build_local_connection_graph(
            pd.concat([bundle.transactions, equal_time], ignore_index=True),
            self.selected_transaction,
        )

        historical_times = [
            values["transaction_time"]
            for _, values in graph.nodes(data=True)
            if values["node_type"] == "historical_transaction"
        ]
        node_types = {values["node_type"] for _, values in graph.nodes(data=True)}
        node_identifiers = {str(node) for node in graph.nodes}
        self.assertTrue(summary["strictly_historical"])
        self.assertTrue(all(value < selected_time for value in historical_times))
        self.assertNotIn("equal_time_peer", " ".join(node_identifiers))
        self.assertNotIn("payer_email_domain", node_types)
        self.assertNotIn("recipient_email_domain", node_types)
        for node, values in graph.nodes(data=True):
            if values["node_type"] in {"card", "customer", "device", "address", "recipient"}:
                self.assertNotIn(str(node[1]), values["display_label"])
                self.assertNotIn(str(node[1]), values["hover_text"])

    def test_loader_rejects_target_hidden_in_reason_evidence(self) -> None:
        artifact_path = self.project_root / "artifacts" / "day7" / "investigator_explanations.json"
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["alerts"][0]["reasons"][0]["evidence"]["label"] = 1
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Evaluation target"):
            load_dashboard_bundle(self.project_root)

    def test_connection_figure_is_deterministic(self) -> None:
        bundle = load_dashboard_bundle(self.project_root)
        graph, _ = build_local_connection_graph(
            bundle.transactions,
            self.selected_transaction,
        )

        self.assertEqual(connection_figure(graph).to_json(), connection_figure(graph).to_json())

    def test_streamlit_app_smoke_test(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
        with patch.dict(os.environ, {"FRAUD_DASHBOARD_PROJECT_ROOT": str(self.project_root)}):
            app = AppTest.from_file(str(app_path)).run(timeout=30)

        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Fraud investigation queue")
        self.assertIn("not confirmed fraud rings", app.info[0].value)
        self.assertEqual(app.selectbox[0].label, "Filter by reason")
        self.assertEqual(app.selectbox[1].label, "Open alert")
        self.assertIn(
            "Historical connection view",
            [element.value for element in app.subheader],
        )
        self.assertGreaterEqual(len(app.metric), 7)

        with patch.dict(
            os.environ,
            {"FRAUD_DASHBOARD_PROJECT_ROOT": str(self.project_root)},
        ):
            app.selectbox[0].select("SHARED_SPECIFIC_ENTITY").run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox[0].value, "SHARED_SPECIFIC_ENTITY")
        self.assertEqual(app.selectbox[1].value, self.selected_transaction)


if __name__ == "__main__":
    unittest.main()
