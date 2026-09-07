"""Target-free dashboard data and focused scoring-time connection views."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from fraud_detection.data.schema import validate_transactions
from fraud_detection.features import SPECIFIC_ENTITY_COLUMN_TYPES
from fraud_detection.graph import NodeId, entity_node_id, transaction_node_id

TARGET_ALIASES = {"is_fraud", "isFraud", "target", "label"}


@dataclass
class DashboardBundle:
    alerts: pd.DataFrame
    transactions: pd.DataFrame
    explanation_artifact: dict[str, Any]


def _masked_reference(value: Any, *, visible: int = 8) -> str:
    """Return a stable opaque reference without exposing the source value."""

    digest = sha256(str(value).encode("utf-8")).hexdigest()
    return f"ref-{digest[:visible]}"


def _contains_target_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(TARGET_ALIASES.intersection(map(str, value))) or any(
            _contains_target_key(item) for item in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_target_key(item) for item in value)
    return False


def load_dashboard_bundle(project_root: Path) -> DashboardBundle:
    """Load audited explanation data and strip evaluation labels from the UI layer."""

    transaction_path = project_root / "data" / "processed" / "transactions.csv"
    graph_feature_path = project_root / "data" / "processed" / "graph_features.csv"
    label_feature_path = project_root / "data" / "processed" / "label_history_features.csv"
    explanation_path = project_root / "artifacts" / "day7" / "investigator_explanations.json"
    missing = [
        str(path.relative_to(project_root))
        for path in (
            transaction_path,
            graph_feature_path,
            label_feature_path,
            explanation_path,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing dashboard inputs: {', '.join(missing)}")

    transactions = validate_transactions(pd.read_csv(transaction_path))
    graph_features = pd.read_csv(graph_feature_path)
    label_features = pd.read_csv(label_feature_path)
    artifact = json.loads(explanation_path.read_text(encoding="utf-8"))
    artifact_gates = artifact.get("gates")
    if (
        artifact.get("status") != "PASS"
        or not isinstance(artifact_gates, dict)
        or not artifact_gates
        or not all(artifact_gates.values())
    ):
        raise ValueError("Investigator explanation artifact has not passed its audit")
    required_artifact_keys = {
        "alerts",
        "capacity_fraction",
        "capacity_tie_audit",
        "disclaimer",
        "gates",
        "model",
        "review_count",
    }
    if not required_artifact_keys.issubset(artifact):
        raise ValueError("Investigator explanation artifact is incomplete")

    safe_transactions = transactions.drop(columns=["is_fraud"], errors="ignore")
    forbidden_transaction_fields = TARGET_ALIASES.intersection(safe_transactions.columns)
    if forbidden_transaction_fields:
        raise ValueError(
            "Evaluation target reached the dashboard transaction layer: "
            f"{', '.join(sorted(forbidden_transaction_fields))}"
        )
    graph_columns = [
        "transaction_id",
        "specific_shared_transaction_count",
        "specific_max_component_transactions",
        "specific_max_entity_degree",
    ]
    label_columns = ["transaction_id", "matured_neighbor_fraud_ratio"]
    missing_graph_columns = sorted(set(graph_columns) - set(graph_features.columns))
    missing_label_columns = sorted(set(label_columns) - set(label_features.columns))
    if missing_graph_columns or missing_label_columns:
        raise ValueError(
            "Dashboard feature tables are incomplete: "
            f"graph={missing_graph_columns}, label_history={missing_label_columns}"
        )
    enriched = safe_transactions.merge(
        graph_features[graph_columns],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    ).merge(
        label_features[label_columns],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )
    feature_columns = [*graph_columns[1:], *label_columns[1:]]
    if enriched[feature_columns].isna().any().any():
        raise ValueError("Dashboard feature tables do not align with prepared transactions")

    records = []
    for alert in artifact["alerts"]:
        if _contains_target_key(alert):
            raise ValueError("Evaluation target found in investigator alert")
        if alert.get("recommended_action") != "INVESTIGATOR_REVIEW":
            raise ValueError("Dashboard recommendations must remain investigator-review only")
        risk_score = float(alert["risk_score"])
        if not 0 <= risk_score <= 1:
            raise ValueError("Dashboard risk scores must be between zero and one")
        reasons = alert.get("reasons", [])
        if not reasons:
            raise ValueError("Every investigator alert must have at least one reason")
        records.append(
            {
                "transaction_id": str(alert["transaction_id"]),
                "rank": int(alert["rank"]),
                "risk_score": risk_score,
                "recommended_action": str(alert["recommended_action"]),
                "primary_reason": str(reasons[0]["title"]),
                "reason_codes": ", ".join(reason["code"] for reason in reasons),
                "reasons": reasons,
            }
        )
    alerts = pd.DataFrame.from_records(records).merge(
        enriched,
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )
    if len(alerts) != int(artifact["review_count"]) or alerts["amount"].isna().any():
        raise ValueError("Investigator alerts do not align with prepared transactions")
    forbidden = TARGET_ALIASES.intersection(alerts.columns)
    if forbidden:
        raise ValueError(f"Target fields reached dashboard alerts: {', '.join(sorted(forbidden))}")
    return DashboardBundle(
        alerts=alerts.sort_values("rank").reset_index(drop=True),
        transactions=enriched,
        explanation_artifact=artifact,
    )


def _specific_nodes(row: Any) -> dict[NodeId, str]:
    nodes = {}
    for column, entity_type in SPECIFIC_ENTITY_COLUMN_TYPES.items():
        node = entity_node_id(entity_type, getattr(row, column))
        if node is not None:
            nodes[node] = column
    return nodes


def build_local_connection_graph(
    transactions: pd.DataFrame,
    transaction_id: str,
    *,
    max_neighbour_transactions: int = 20,
) -> tuple[nx.Graph, dict[str, Any]]:
    """Build a compact graph from current entities and strictly earlier shared activity."""

    if max_neighbour_transactions <= 0:
        raise ValueError("max_neighbour_transactions must be positive")
    forbidden = TARGET_ALIASES.intersection(transactions.columns)
    if forbidden:
        raise ValueError(
            "Evaluation labels must be removed before building dashboard graphs: "
            f"{', '.join(sorted(forbidden))}"
        )
    matches = transactions[transactions["transaction_id"].astype(str) == str(transaction_id)]
    if len(matches) != 1:
        raise ValueError(f"Expected one selected transaction, found {len(matches)}")
    selected = next(matches.itertuples(index=False))
    selected_time = int(selected.transaction_time)
    selected_entities = _specific_nodes(selected)

    neighbours = []
    history = transactions[transactions["transaction_time"] < selected_time]
    for row in history.itertuples(index=False):
        shared_nodes = set(selected_entities).intersection(_specific_nodes(row))
        if shared_nodes:
            neighbours.append((row, shared_nodes))
    neighbours.sort(
        key=lambda item: (
            -len(item[1]),
            -int(item[0].transaction_time),
            str(item[0].transaction_id),
        )
    )
    displayed = neighbours[:max_neighbour_transactions]

    graph = nx.Graph()
    selected_node = transaction_node_id(transaction_id)
    graph.add_node(
        selected_node,
        node_type="selected_transaction",
        display_label=f"Selected\n{_masked_reference(transaction_id, visible=12)}",
        hover_text=f"Selected transaction {transaction_id}<br>Elapsed time: {selected_time:,}",
        transaction_time=selected_time,
    )
    for entity_node, source_column in selected_entities.items():
        entity_type, entity_value = entity_node
        graph.add_node(
            entity_node,
            node_type=entity_type,
            display_label=f"{entity_type.replace('_', ' ').title()}\n"
            f"{_masked_reference(entity_value)}",
            hover_text=f"{entity_type.replace('_', ' ').title()}: "
            f"{_masked_reference(entity_value)}",
        )
        graph.add_edge(selected_node, entity_node, relation=source_column)

    for row, shared_nodes in displayed:
        transaction_node = transaction_node_id(row.transaction_id)
        graph.add_node(
            transaction_node,
            node_type="historical_transaction",
            display_label=f"Earlier txn\n{_masked_reference(row.transaction_id, visible=12)}",
            hover_text=f"Earlier transaction {_masked_reference(row.transaction_id, visible=12)}"
            f"<br>Elapsed time: {int(row.transaction_time):,}",
            transaction_time=int(row.transaction_time),
        )
        for entity_node in shared_nodes:
            graph.add_edge(transaction_node, entity_node, relation=selected_entities[entity_node])

    summary = {
        "selected_transaction_id": str(transaction_id),
        "selected_transaction_time": selected_time,
        "available_specific_entities": len(selected_entities),
        "historical_neighbour_count": len(neighbours),
        "displayed_historical_neighbours": len(displayed),
        "truncated": len(neighbours) > len(displayed),
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "strictly_historical": all(
            attributes.get("node_type") != "historical_transaction"
            or int(attributes["transaction_time"]) < selected_time
            for _, attributes in graph.nodes(data=True)
        ),
    }
    if not summary["strictly_historical"]:
        raise RuntimeError("Dashboard graph contains non-historical neighbour activity")
    return graph, summary


NODE_STYLE = {
    "selected_transaction": ("#B42318", 25, "Selected transaction"),
    "historical_transaction": ("#F79009", 14, "Earlier transaction"),
    "card": ("#175CD3", 18, "Card proxy"),
    "customer": ("#0E9384", 18, "Customer proxy"),
    "device": ("#6938EF", 18, "Device"),
    "address": ("#039855", 18, "Address proxy"),
    "recipient": ("#DC6803", 18, "Recipient proxy"),
}


def connection_figure(graph: nx.Graph, *, seed: int = 42) -> go.Figure:
    """Render a deterministic, interactive Plotly figure for a local connection graph."""

    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot render an empty connection graph")
    positions = nx.spring_layout(graph, seed=seed, k=0.9)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for left, right in graph.edges:
        x0, y0 = positions[left]
        x1, y1 = positions[right]
        edge_x.extend([float(x0), float(x1), None])
        edge_y.extend([float(y0), float(y1), None])
    traces: list[go.Scatter] = [
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 1.2, "color": "#D0D5DD"},
            hoverinfo="skip",
            showlegend=False,
        )
    ]
    node_types = sorted({attributes["node_type"] for _, attributes in graph.nodes(data=True)})
    for node_type in node_types:
        nodes = [
            node
            for node, attributes in graph.nodes(data=True)
            if attributes["node_type"] == node_type
        ]
        color, size, legend = NODE_STYLE[node_type]
        traces.append(
            go.Scatter(
                x=[float(positions[node][0]) for node in nodes],
                y=[float(positions[node][1]) for node in nodes],
                mode="markers+text",
                name=legend,
                text=[graph.nodes[node]["display_label"] for node in nodes],
                textposition="top center",
                hovertext=[graph.nodes[node]["hover_text"] for node in nodes],
                hoverinfo="text",
                marker={
                    "size": size,
                    "color": color,
                    "line": {"width": 1.5, "color": "white"},
                },
            )
        )
    return go.Figure(
        data=traces,
        layout=go.Layout(
            height=560,
            margin={"l": 10, "r": 10, "t": 20, "b": 10},
            paper_bgcolor="white",
            plot_bgcolor="white",
            hovermode="closest",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
            xaxis={"showgrid": False, "zeroline": False, "visible": False},
            yaxis={"showgrid": False, "zeroline": False, "visible": False},
        ),
    )
