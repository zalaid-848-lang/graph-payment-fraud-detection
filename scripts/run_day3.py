"""Run Day 3 graph construction, causal replay, and integrity reporting."""

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
from fraud_detection.graph import (  # noqa: E402
    EntityLinkGraphBuilder,
    build_entity_link_graph,
    iter_time_batches,
    serialize_graph,
    transaction_node_id,
    validate_graph_integrity,
)


def _render_type_table(statistics: dict[str, Any]) -> list[str]:
    lines = ["| Node type | Count |", "|---|---:|"]
    lines.extend(
        f"| `{node_type}` | {count:,} |"
        for node_type, count in statistics["nodes_by_type"].items()
    )
    return lines


def _render_report(result: dict[str, Any]) -> str:
    training = result["training_graph"]
    full = result["full_structural_replay"]
    training_stats = training["statistics"]
    full_stats = full["statistics"]
    largest_share = (
        training_stats["largest_component_size"] / training_stats["node_count"]
        if training_stats["node_count"]
        else 0.0
    )
    lines = [
        "# Day 3 graph construction and integrity report",
        "",
        "## Executive summary",
        "",
        "The project now constructs an undirected, typed transaction–entity graph with "
        "NetworkX. Transaction labels are deliberately absent from node and edge attributes. "
        "Chronological equal-time batches provide the safe update boundary required for Day 4 "
        "historical graph features.",
        "",
        "> Connected groups are potential investigation leads only. They are described as "
        "**suspected fraud rings**, never confirmed criminal networks.",
        "",
        "## Training-period graph",
        "",
        f"- Transactions: `{training_stats['transaction_node_count']:,}`",
        f"- Entity nodes: `{training_stats['entity_node_count']:,}`",
        f"- Edges: `{training_stats['edge_count']:,}`",
        f"- Connected components: `{training_stats['connected_component_count']:,}`",
        f"- Largest component: `{training_stats['largest_component_size']:,}` nodes "
        f"(`{largest_share:.2%}` of graph nodes)",
        f"- Isolated transactions: `{training_stats['isolated_transaction_count']:,}`",
        "",
    ]
    lines.extend(_render_type_table(training_stats))
    lines.extend(
        [
            "",
            "## Integrity gates",
            "",
            f"Overall status: **{training['status']}**",
            "",
        ]
    )
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — `{gate}`"
        for gate, passed in training["gates"].items()
    )
    lines.extend(
        [
            "",
            "## Causal replay audit",
            "",
            f"- Chronological batches processed: `{result['causal_audit']['batch_count']:,}`",
            f"- Equal-time multi-transaction batches: "
            f"`{result['causal_audit']['multi_transaction_batch_count']:,}`",
            f"- Ordering or pre-existence violations: "
            f"`{result['causal_audit']['violation_count']:,}`",
            f"- Full replay transactions: `{full_stats['transaction_node_count']:,}`",
            f"- Full replay edges: `{full_stats['edge_count']:,}`",
            "",
            "For every batch at time *t*, consumers first inspect the graph containing only times "
            "less than *t*. All transactions at *t* are then added together. This prevents an "
            "arbitrary row order from leaking relationships between simultaneous transactions.",
            "",
            "The full replay is an integrity diagnostic, not a model feature table. Day 4 will "
            "materialize each transaction's features before its batch is added.",
            "",
            "## Important topology finding",
            "",
        ]
    )
    if largest_share >= 0.5:
        lines.append(
            "Common email-domain hubs create a very large connected component. Therefore, raw "
            "component membership alone is not a credible suspected-ring signal. Day 4 features "
            "will remain type-aware and will distinguish high-frequency infrastructure hubs from "
            "more specific shared devices, cards, addresses, customers, and recipients."
        )
    else:
        lines.append(
            "No single training component contains most graph nodes. Component size will still be "
            "treated as a prioritization feature rather than proof of coordinated fraud."
        )
    lines.extend(
        [
            "",
            "## Files produced",
            "",
            "- `artifacts/day3/training_graph.json`: label-free training topology",
            "- `artifacts/day3/graph_integrity.json`: machine-readable gates and statistics",
            "- `docs/graph_schema.md`: versioned node, edge, and leakage contract",
            "",
            "## Limitations and investigator controls",
            "",
            "- IEEE-CIS customer, card, address, device, and recipient nodes are "
            "anonymized proxies.",
            "- Shared infrastructure can connect unrelated legitimate transactions.",
            "- Connectivity does not verify intent, identity, or criminal coordination.",
            "- Graph outputs prioritize human review and never automate account blocking.",
            "",
            "## Day 4 handoff",
            "",
            "Compute causal node degree, shared-entity counts, historical component size, "
            "PageRank, "
            "and clustering features before each time batch is added. Label-derived neighbour "
            "features remain deferred until an explicit historical label-delay policy is enforced.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> tuple[dict[str, Any], Path]:
    with (PROJECT_ROOT / "configs" / "project.toml").open("rb") as file:
        config = tomllib.load(file)
    data_path = PROJECT_ROOT / config["data"]["processed_file"]
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Prepared data not found at {data_path}. Run python scripts/prepare_data.py first."
        )

    frame = validate_transactions(pd.read_csv(data_path))
    training_frame = frame[frame["split"] == "train"].copy()
    training_graph = build_entity_link_graph(training_frame)
    training_integrity = validate_graph_integrity(training_graph, training_frame)
    if training_integrity["status"] != "PASS":
        raise RuntimeError("Training graph integrity checks failed")

    replay = EntityLinkGraphBuilder()
    violations: list[str] = []
    batch_count = 0
    multi_transaction_batch_count = 0
    for event_time, batch in iter_time_batches(frame):
        batch_count += 1
        multi_transaction_batch_count += int(len(batch) > 1)
        if replay.last_time is not None and replay.last_time >= event_time:
            violations.append(f"non_increasing_time:{event_time}")
        for transaction_id in batch["transaction_id"]:
            if transaction_node_id(transaction_id) in replay.graph:
                violations.append(f"transaction_pre_exists:{transaction_id}")
        replay.add_batch(batch)

    full_integrity = validate_graph_integrity(replay.graph, frame)
    if full_integrity["status"] != "PASS" or violations:
        raise RuntimeError("Full causal graph replay integrity checks failed")

    result: dict[str, Any] = {
        "source": str(frame["source"].iloc[0]),
        "training_graph": training_integrity,
        "full_structural_replay": full_integrity,
        "causal_audit": {
            "batch_count": batch_count,
            "multi_transaction_batch_count": multi_transaction_batch_count,
            "violation_count": len(violations),
            "violations": violations,
        },
        "configuration": config["graph"],
    }

    artifact_dir = PROJECT_ROOT / "artifacts" / "day3"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "training_graph.json").write_text(
        json.dumps(serialize_graph(training_graph), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "graph_integrity.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report_path = PROJECT_ROOT / "reports" / "day3_graph_construction.md"
    report_path.write_text(_render_report(result), encoding="utf-8")
    return result, report_path


def main() -> None:
    result, report_path = run()
    statistics = result["training_graph"]["statistics"]
    print(
        f"Training graph: {statistics['node_count']} nodes, "
        f"{statistics['edge_count']} edges, "
        f"{statistics['connected_component_count']} components"
    )
    print(f"Integrity: {result['training_graph']['status']}")
    print(f"Causal replay violations: {result['causal_audit']['violation_count']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
