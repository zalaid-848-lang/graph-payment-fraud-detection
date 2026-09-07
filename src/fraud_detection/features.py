"""Causal structural graph features computed before each transaction-time batch."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from fraud_detection.graph import (
    ENTITY_COLUMN_TYPES,
    EntityLinkGraphBuilder,
    NodeId,
    entity_node_id,
    iter_time_batches,
)

HUB_PRONE_ENTITY_TYPES = {"payer_email_domain", "recipient_email_domain"}
SPECIFIC_ENTITY_COLUMN_TYPES = {
    column: entity_type
    for column, entity_type in ENTITY_COLUMN_TYPES.items()
    if entity_type not in HUB_PRONE_ENTITY_TYPES
}

PER_TYPE_DEGREE_FEATURES = tuple(
    f"historical_{entity_type}_degree" for entity_type in ENTITY_COLUMN_TYPES.values()
)
STRUCTURAL_FEATURE_COLUMNS = (
    "graph_entity_count",
    "graph_known_entity_count",
    "graph_new_entity_count",
    "graph_sum_entity_degree",
    "graph_max_entity_degree",
    "graph_mean_entity_degree",
    "graph_shared_transaction_count",
    "specific_entity_count",
    "specific_known_entity_count",
    "specific_shared_transaction_count",
    "specific_max_entity_degree",
    "specific_mean_entity_degree",
    "specific_max_component_nodes",
    "specific_max_component_transactions",
    "specific_max_pagerank",
    "specific_mean_pagerank",
    "specific_max_clustering",
    "specific_mean_clustering",
    "pagerank_snapshot_age_batches",
    *PER_TYPE_DEGREE_FEATURES,
)


def _entity_nodes(row: Any, mapping: dict[str, str]) -> list[NodeId]:
    nodes = []
    for column, entity_type in mapping.items():
        node = entity_node_id(entity_type, getattr(row, column))
        if node is not None:
            nodes.append(node)
    return nodes


def _degrees(graph: nx.Graph, nodes: Iterable[NodeId]) -> list[int]:
    return [int(graph.degree(node)) if node in graph else 0 for node in nodes]


def _shared_transactions(graph: nx.Graph, nodes: Iterable[NodeId]) -> int:
    transactions: set[NodeId] = set()
    for node in nodes:
        if node in graph:
            transactions.update(
                neighbour for neighbour in graph.neighbors(node) if neighbour[0] == "transaction"
            )
    return len(transactions)


def _component_lookup(graph: nx.Graph) -> dict[NodeId, tuple[int, int]]:
    lookup: dict[NodeId, tuple[int, int]] = {}
    for component in nx.connected_components(graph):
        transaction_count = sum(node[0] == "transaction" for node in component)
        summary = (len(component), transaction_count)
        for node in component:
            if node[0] != "transaction":
                lookup[node] = summary
    return lookup


def _update_projection(projection: nx.Graph, batch: pd.DataFrame, event_time: int) -> None:
    for row in batch.itertuples(index=False):
        nodes = _entity_nodes(row, SPECIFIC_ENTITY_COLUMN_TYPES)
        projection.add_nodes_from(nodes)
        for left, right in combinations(nodes, 2):
            if projection.has_edge(left, right):
                projection[left][right]["weight"] += 1
                projection[left][right]["last_observed_time"] = event_time
            else:
                projection.add_edge(
                    left,
                    right,
                    weight=1,
                    first_observed_time=event_time,
                    last_observed_time=event_time,
                )


@dataclass
class CausalGraphFeatureBuilder:
    """Build type-aware structural features from strictly earlier graph history."""

    pagerank_refresh_batches: int = 25
    pagerank_alpha: float = 0.85

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.pagerank_refresh_batches <= 0:
            raise ValueError("pagerank_refresh_batches must be positive")
        if not 0 < self.pagerank_alpha < 1:
            raise ValueError("pagerank_alpha must be between 0 and 1")

        full_history = EntityLinkGraphBuilder()
        specific_history = EntityLinkGraphBuilder(
            entity_column_types=dict(SPECIFIC_ENTITY_COLUMN_TYPES),
            allow_entityless_transactions=True,
        )
        projection = nx.Graph()
        pagerank: dict[NodeId, float] = {}
        pagerank_refresh_count = 0
        last_pagerank_refresh_batch = 0
        records: list[dict[str, Any]] = []

        for batch_number, (event_time, batch) in enumerate(iter_time_batches(frame)):
            if batch_number % self.pagerank_refresh_batches == 0:
                pagerank = (
                    nx.pagerank(
                        projection,
                        alpha=self.pagerank_alpha,
                        weight="weight",
                    )
                    if projection.number_of_nodes()
                    else {}
                )
                pagerank_refresh_count += 1
                last_pagerank_refresh_batch = batch_number

            component_lookup = _component_lookup(specific_history.graph)
            current_specific_nodes = {
                node
                for row in batch.itertuples(index=False)
                for node in _entity_nodes(row, SPECIFIC_ENTITY_COLUMN_TYPES)
                if node in projection
            }
            clustering = (
                nx.clustering(projection, nodes=current_specific_nodes, weight=None)
                if current_specific_nodes
                else {}
            )

            for row in batch.itertuples(index=False):
                all_nodes = _entity_nodes(row, ENTITY_COLUMN_TYPES)
                specific_nodes = _entity_nodes(row, SPECIFIC_ENTITY_COLUMN_TYPES)
                all_degrees = _degrees(full_history.graph, all_nodes)
                specific_degrees = _degrees(specific_history.graph, specific_nodes)
                known_count = sum(degree > 0 for degree in all_degrees)
                specific_known_count = sum(degree > 0 for degree in specific_degrees)
                component_summaries = [
                    component_lookup[node] for node in specific_nodes if node in component_lookup
                ]
                pagerank_values = [pagerank.get(node, 0.0) for node in specific_nodes]
                clustering_values = [clustering.get(node, 0.0) for node in specific_nodes]
                degree_by_type = dict(
                    zip((node[0] for node in all_nodes), all_degrees, strict=True)
                )

                record: dict[str, Any] = {
                    "transaction_id": str(row.transaction_id),
                    "transaction_time": event_time,
                    "graph_entity_count": len(all_nodes),
                    "graph_known_entity_count": known_count,
                    "graph_new_entity_count": len(all_nodes) - known_count,
                    "graph_sum_entity_degree": sum(all_degrees),
                    "graph_max_entity_degree": max(all_degrees, default=0),
                    "graph_mean_entity_degree": float(np.mean(all_degrees)) if all_degrees else 0.0,
                    "graph_shared_transaction_count": _shared_transactions(
                        full_history.graph, all_nodes
                    ),
                    "specific_entity_count": len(specific_nodes),
                    "specific_known_entity_count": specific_known_count,
                    "specific_shared_transaction_count": _shared_transactions(
                        specific_history.graph, specific_nodes
                    ),
                    "specific_max_entity_degree": max(specific_degrees, default=0),
                    "specific_mean_entity_degree": (
                        float(np.mean(specific_degrees)) if specific_degrees else 0.0
                    ),
                    "specific_max_component_nodes": max(
                        (summary[0] for summary in component_summaries), default=0
                    ),
                    "specific_max_component_transactions": max(
                        (summary[1] for summary in component_summaries), default=0
                    ),
                    "specific_max_pagerank": max(pagerank_values, default=0.0),
                    "specific_mean_pagerank": (
                        float(np.mean(pagerank_values)) if pagerank_values else 0.0
                    ),
                    "specific_max_clustering": max(clustering_values, default=0.0),
                    "specific_mean_clustering": (
                        float(np.mean(clustering_values)) if clustering_values else 0.0
                    ),
                    "pagerank_snapshot_age_batches": (batch_number - last_pagerank_refresh_batch),
                }
                record.update(
                    {
                        f"historical_{entity_type}_degree": degree_by_type.get(entity_type, 0)
                        for entity_type in ENTITY_COLUMN_TYPES.values()
                    }
                )
                records.append(record)

            full_history.add_batch(batch)
            specific_history.add_batch(batch)
            _update_projection(projection, batch, event_time)

        result = pd.DataFrame.from_records(records)
        expected_columns = [
            "transaction_id",
            "transaction_time",
            *STRUCTURAL_FEATURE_COLUMNS,
        ]
        result = result.reindex(columns=expected_columns)
        self.audit_ = {
            "row_count": len(result),
            "feature_count": len(STRUCTURAL_FEATURE_COLUMNS),
            "time_batch_count": sum(1 for _ in iter_time_batches(frame)),
            "pagerank_refresh_count": pagerank_refresh_count,
            "pagerank_refresh_batches": self.pagerank_refresh_batches,
            "hub_prone_types_excluded_from_specific_features": sorted(HUB_PRONE_ENTITY_TYPES),
            "final_full_history_nodes": full_history.graph.number_of_nodes(),
            "final_specific_history_nodes": specific_history.graph.number_of_nodes(),
            "final_projection_nodes": projection.number_of_nodes(),
            "final_projection_edges": projection.number_of_edges(),
        }
        return result
