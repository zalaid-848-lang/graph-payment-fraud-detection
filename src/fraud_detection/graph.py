"""Typed, label-free entity-link graph construction and integrity checks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import networkx as nx
import pandas as pd

NodeId: TypeAlias = tuple[str, str]

ENTITY_COLUMN_TYPES = {
    "card_id": "card",
    "customer_id": "customer",
    "device_id": "device",
    "address_id": "address",
    "payer_email_domain": "payer_email_domain",
    "recipient_email_domain": "recipient_email_domain",
    "recipient_id": "recipient",
}
TARGET_ATTRIBUTE_NAMES = {"is_fraud", "isFraud", "target", "label"}
MISSING_TEXT_VALUES = {"", "<na>", "nan", "none", "null"}


class GraphBuildError(ValueError):
    """Raised when transactions cannot be added without breaking graph invariants."""


def clean_entity_value(value: Any) -> str | None:
    """Normalize an entity value without turning missingness into a shared node."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in MISSING_TEXT_VALUES:
        return None
    return text


def transaction_node_id(transaction_id: Any) -> NodeId:
    """Return a namespaced transaction node identifier."""
    value = clean_entity_value(transaction_id)
    if value is None:
        raise GraphBuildError("transaction_id must be non-empty")
    return ("transaction", value)


def entity_node_id(entity_type: str, value: Any) -> NodeId | None:
    """Return a typed entity node identifier, or None for missing values."""
    cleaned = clean_entity_value(value)
    return None if cleaned is None else (entity_type, cleaned)


def iter_time_batches(frame: pd.DataFrame) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield stable chronological batches so equal-time rows share prior state."""
    required = {"transaction_id", "transaction_time"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise GraphBuildError(f"Missing graph ordering columns: {', '.join(missing)}")
    ordered = frame.sort_values(
        ["transaction_time", "transaction_id"], kind="stable"
    ).reset_index(drop=True)
    for event_time, batch in ordered.groupby("transaction_time", sort=False):
        yield int(event_time), batch.reset_index(drop=True)


@dataclass
class EntityLinkGraphBuilder:
    """Incrementally build a graph using strictly increasing time batches."""

    graph: nx.Graph = field(default_factory=nx.Graph)
    last_time: int | None = None
    entity_column_types: dict[str, str] = field(
        default_factory=lambda: dict(ENTITY_COLUMN_TYPES)
    )
    allow_entityless_transactions: bool = False

    def add_batch(self, batch: pd.DataFrame) -> None:
        """Add one equal-time batch after consumers have queried prior state."""
        missing = sorted(
            ({"transaction_id", "transaction_time", "amount", "product_code"}
            | set(self.entity_column_types))
            - set(batch.columns)
        )
        if missing:
            raise GraphBuildError(f"Missing graph columns: {', '.join(missing)}")
        event_times = pd.to_numeric(batch["transaction_time"], errors="coerce").unique()
        if len(event_times) != 1 or pd.isna(event_times[0]):
            raise GraphBuildError("add_batch requires exactly one valid transaction_time")
        event_time = int(event_times[0])
        if self.last_time is not None and event_time <= self.last_time:
            raise GraphBuildError("Batches must be added in strictly increasing time order")

        for row in batch.itertuples(index=False):
            transaction_id = transaction_node_id(row.transaction_id)
            if transaction_id in self.graph:
                raise GraphBuildError(f"Duplicate transaction node: {row.transaction_id}")
            entity_nodes: list[tuple[str, NodeId]] = []
            for column, entity_type in self.entity_column_types.items():
                node_id = entity_node_id(entity_type, getattr(row, column))
                if node_id is not None:
                    entity_nodes.append((column, node_id))
            if not entity_nodes and not self.allow_entityless_transactions:
                raise GraphBuildError(
                    f"Transaction {row.transaction_id} has no usable graph entities"
                )

            self.graph.add_node(
                transaction_id,
                node_type="transaction",
                entity_value=str(row.transaction_id),
                transaction_time=event_time,
                amount=float(row.amount),
                product_code=str(row.product_code),
            )
            for source_column, node_id in entity_nodes:
                entity_type, entity_value = node_id
                self.graph.add_node(
                    node_id,
                    node_type=entity_type,
                    entity_value=entity_value,
                    source_column=source_column,
                )
                self.graph.add_edge(
                    transaction_id,
                    node_id,
                    relation=f"uses_{entity_type}",
                    observed_time=event_time,
                )
        self.last_time = event_time

    def add_transactions(self, frame: pd.DataFrame) -> None:
        """Add a transaction frame in causal equal-time batches."""
        for _, batch in iter_time_batches(frame):
            self.add_batch(batch)


def build_entity_link_graph(
    frame: pd.DataFrame, *, cutoff_time: int | None = None
) -> nx.Graph:
    """Build a label-free graph, optionally through an inclusive time cutoff."""
    selected = frame
    if cutoff_time is not None:
        selected = frame[frame["transaction_time"] <= cutoff_time]
    builder = EntityLinkGraphBuilder()
    if not selected.empty:
        builder.add_transactions(selected)
    return builder.graph


def graph_statistics(graph: nx.Graph) -> dict[str, Any]:
    """Return deterministic, JSON-ready topology statistics."""
    node_types = Counter(
        attributes.get("node_type", "missing")
        for _, attributes in graph.nodes(data=True)
    )
    relation_counts = Counter(
        attributes.get("relation", "missing") for _, _, attributes in graph.edges(data=True)
    )
    component_sizes = sorted(
        (len(component) for component in nx.connected_components(graph)), reverse=True
    )
    transaction_nodes = [
        node
        for node, attributes in graph.nodes(data=True)
        if attributes.get("node_type") == "transaction"
    ]
    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "transaction_node_count": len(transaction_nodes),
        "entity_node_count": graph.number_of_nodes() - len(transaction_nodes),
        "nodes_by_type": dict(sorted(node_types.items())),
        "edges_by_relation": dict(sorted(relation_counts.items())),
        "connected_component_count": len(component_sizes),
        "largest_component_size": component_sizes[0] if component_sizes else 0,
        "isolated_transaction_count": sum(graph.degree(node) == 0 for node in transaction_nodes),
    }


def validate_graph_integrity(graph: nx.Graph, source_frame: pd.DataFrame) -> dict[str, Any]:
    """Audit graph structure against its source transactions without using labels."""
    expected_transactions = {
        transaction_node_id(value) for value in source_frame["transaction_id"]
    }
    actual_transactions = {
        node
        for node, attributes in graph.nodes(data=True)
        if attributes.get("node_type") == "transaction"
    }
    allowed_types = {"transaction", *ENTITY_COLUMN_TYPES.values()}

    node_types_valid = all(
        isinstance(node, tuple)
        and len(node) == 2
        and attributes.get("node_type") == node[0]
        and node[0] in allowed_types
        for node, attributes in graph.nodes(data=True)
    )
    entity_values_valid = all(
        node[0] == "transaction" or clean_entity_value(node[1]) is not None
        for node in graph.nodes
    )
    target_free = all(
        TARGET_ATTRIBUTE_NAMES.isdisjoint(attributes)
        for _, attributes in graph.nodes(data=True)
    ) and all(
        TARGET_ATTRIBUTE_NAMES.isdisjoint(attributes)
        for _, _, attributes in graph.edges(data=True)
    )

    edges_bipartite = True
    edge_times_valid = True
    relations_valid = True
    for left, right, attributes in graph.edges(data=True):
        transaction = left if left[0] == "transaction" else right
        entity = right if left[0] == "transaction" else left
        if transaction[0] != "transaction" or entity[0] == "transaction":
            edges_bipartite = False
            continue
        if attributes.get("relation") != f"uses_{entity[0]}":
            relations_valid = False
        transaction_time = graph.nodes[transaction].get("transaction_time")
        if attributes.get("observed_time") != transaction_time:
            edge_times_valid = False

    expected_edge_count = 0
    for row in source_frame.itertuples(index=False):
        expected_edge_count += sum(
            entity_node_id(entity_type, getattr(row, column)) is not None
            for column, entity_type in ENTITY_COLUMN_TYPES.items()
        )
    gates = {
        "undirected_simple_graph": not graph.is_directed() and not graph.is_multigraph(),
        "node_types_valid": node_types_valid,
        "entity_values_non_missing": entity_values_valid,
        "target_attributes_absent": target_free,
        "edges_are_transaction_entity_only": edges_bipartite,
        "edge_relations_match_entity_types": relations_valid,
        "edge_times_match_transactions": edge_times_valid,
        "transaction_nodes_match_source": actual_transactions == expected_transactions,
        "edge_count_matches_available_entities": graph.number_of_edges() == expected_edge_count,
        "transactions_have_entity_edges": all(
            graph.degree(node) > 0 for node in actual_transactions
        ),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "expected_transaction_count": len(expected_transactions),
        "expected_edge_count": expected_edge_count,
        "statistics": graph_statistics(graph),
    }


def serialize_graph(graph: nx.Graph) -> dict[str, Any]:
    """Create a deterministic, portable node/edge payload for investigation tools."""
    nodes = [
        {"id": f"{node[0]}::{node[1]}", **attributes}
        for node, attributes in sorted(graph.nodes(data=True), key=lambda item: item[0])
    ]
    edges = []
    for left, right, attributes in graph.edges(data=True):
        source, target = sorted((left, right))
        edges.append(
            {
                "source": f"{source[0]}::{source[1]}",
                "target": f"{target[0]}::{target[1]}",
                **attributes,
            }
        )
    edges.sort(key=lambda edge: (edge["source"], edge["target"], edge["relation"]))
    return {"directed": False, "multigraph": False, "nodes": nodes, "edges": edges}
