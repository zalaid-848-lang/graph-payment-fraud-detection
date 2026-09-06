import unittest

import networkx as nx
import pandas as pd

from fraud_detection.data.split import assign_temporal_splits
from fraud_detection.data.synthetic import generate_synthetic_transactions
from fraud_detection.graph import (
    EntityLinkGraphBuilder,
    GraphBuildError,
    build_entity_link_graph,
    iter_time_batches,
    serialize_graph,
    validate_graph_integrity,
)


class EntityLinkGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = generate_synthetic_transactions(100)

    def test_graph_is_typed_bipartite_and_target_free(self) -> None:
        graph = build_entity_link_graph(self.frame)
        integrity = validate_graph_integrity(graph, self.frame)

        self.assertEqual(integrity["status"], "PASS")
        self.assertTrue(nx.is_bipartite(graph))
        self.assertEqual(integrity["statistics"]["transaction_node_count"], 100)
        self.assertTrue(
            all("is_fraud" not in attributes for _, attributes in graph.nodes(data=True))
        )

    def test_missing_entities_never_become_nodes_or_edges(self) -> None:
        frame = self.frame.copy()
        frame.loc[0, "device_id"] = pd.NA
        frame.loc[0, "address_id"] = ""
        graph = build_entity_link_graph(frame)
        integrity = validate_graph_integrity(graph, frame)

        self.assertEqual(integrity["status"], "PASS")
        self.assertFalse(any(node[1].lower() in {"", "<na>"} for node in graph.nodes))
        self.assertEqual(graph.degree(("transaction", frame.loc[0, "transaction_id"])), 5)

    def test_same_value_in_different_entity_types_does_not_collide(self) -> None:
        frame = self.frame.copy()
        frame.loc[0, "device_id"] = "shared-value"
        frame.loc[0, "address_id"] = "shared-value"
        graph = build_entity_link_graph(frame)

        self.assertIn(("device", "shared-value"), graph)
        self.assertIn(("address", "shared-value"), graph)
        self.assertNotEqual(("device", "shared-value"), ("address", "shared-value"))

    def test_graph_is_identical_when_labels_change(self) -> None:
        changed = self.frame.copy()
        changed["is_fraud"] = 1 - changed["is_fraud"]

        self.assertEqual(
            serialize_graph(build_entity_link_graph(self.frame)),
            serialize_graph(build_entity_link_graph(changed)),
        )

    def test_builder_rejects_time_reversal(self) -> None:
        batches = list(iter_time_batches(self.frame.iloc[:2]))
        builder = EntityLinkGraphBuilder()
        builder.add_batch(batches[1][1])

        with self.assertRaisesRegex(GraphBuildError, "strictly increasing"):
            builder.add_batch(batches[0][1])

    def test_equal_time_rows_are_returned_as_one_batch(self) -> None:
        frame = self.frame.iloc[:3].copy()
        frame.loc[1, "transaction_time"] = frame.loc[0, "transaction_time"]
        batches = list(iter_time_batches(frame))

        self.assertEqual(len(batches[0][1]), 2)
        self.assertEqual(batches[0][0], frame.loc[0, "transaction_time"])

    def test_synthetic_shared_device_connects_multiple_ring_transactions(self) -> None:
        frame = assign_temporal_splits(generate_synthetic_transactions(500))
        training = frame[frame["split"] == "train"]
        graph = build_entity_link_graph(training)
        ring_device = ("device", "device:ring_shared")

        self.assertIn(ring_device, graph)
        self.assertGreaterEqual(graph.degree(ring_device), 10)
        component = nx.node_connected_component(graph, ring_device)
        ring_cards = {node for node in component if node[0] == "card" and "ring_" in node[1]}
        self.assertEqual(len(ring_cards), 3)

    def test_integrity_audit_detects_target_attribute(self) -> None:
        graph = build_entity_link_graph(self.frame)
        transaction = next(node for node in graph if node[0] == "transaction")
        graph.nodes[transaction]["is_fraud"] = 1
        integrity = validate_graph_integrity(graph, self.frame)

        self.assertEqual(integrity["status"], "FAIL")
        self.assertFalse(integrity["gates"]["target_attributes_absent"])


if __name__ == "__main__":
    unittest.main()
