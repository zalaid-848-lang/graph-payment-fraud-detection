"""Audit the Day 8 investigator dashboard inputs and focused connection views."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.dashboard import (  # noqa: E402
    build_local_connection_graph,
    connection_figure,
    load_dashboard_bundle,
)

TARGET_ALIASES = {"is_fraud", "isFraud", "target", "label"}
EMAIL_NODE_TYPES = {"payer_email_domain", "recipient_email_domain"}


def _render_report(result: dict[str, Any]) -> str:
    tie = result["capacity_tie_audit"]
    return "\n".join(
        [
            "# Day 8 investigator dashboard",
            "",
            "## Executive summary",
            "",
            "The Streamlit dashboard turns the audited Day 7 output into a ranked human-review "
            "queue, understandable escalation reasons, and a focused graph of strictly earlier "
            "transactions connected through specific entities.",
            "",
            "> The dashboard describes connected activity as suspected fraud-ring evidence only. "
            "It does not confirm a criminal network, display evaluation labels, or automate "
            "account blocking.",
            "",
            "## Verification result",
            "",
            f"- Audit status: **{result['status']}**",
            f"- Alerts audited: `{result['alert_count']}`",
            f"- Connection views rendered: `{result['connection_views_rendered']}`",
            f"- Maximum nodes in a focused view: `{result['maximum_view_nodes']}`",
            f"- Maximum earlier transactions displayed: "
            f"`{result['maximum_displayed_historical_neighbours']}`",
            f"- Investigation capacity: `{result['capacity_fraction']:.2%}` of test traffic",
            "",
            "All dashboard data and connection views passed these gates:",
            "",
            *[f"- `{name}`: **PASS**" for name, passed in result["gates"].items() if passed],
            "",
            "## Investigator workflow",
            "",
            "1. Review the capacity-limited queue in risk-score order.",
            "2. Filter alerts by an evidence-gated reason code.",
            "3. Open one alert to inspect its amount, product, recommended review action, and "
            "plain-language reasons.",
            "4. Inspect the focused connection view for earlier transactions sharing a masked "
            "card, customer, device, address, or recipient proxy.",
            "5. Use the evidence as a lead for manual investigation, not as an automatic decision.",
            "",
            "## Leakage and visual-scope controls",
            "",
            "- Transaction-level fraud labels are removed before data reaches the dashboard.",
            "- Neighbour transactions must have a strictly earlier event time than the selected "
            "transaction; equal-time and future activity are excluded.",
            "- Payer and recipient email-domain nodes are omitted from the local view because "
            "common domains can create visually misleading hubs.",
            "- The local graph is capped at 20 earlier transactions to preserve readability.",
            "- Entity values are masked in node labels and hover text.",
            "",
            "## Capacity-cutoff tie",
            "",
            f"`{tie['rows_at_cutoff']}` transactions share the cutoff score while "
            f"`{tie['selected_from_cutoff_tie']}` positions were available at that score. Stable "
            "chronological order makes the result reproducible; it is not evidence of a risk "
            "difference among tied transactions.",
            "",
            "## Limitations",
            "",
            "- Current dashboard views and metrics use deterministic synthetic development data.",
            "- IEEE-CIS provides transaction labels, not verified fraud-ring labels.",
            "- Shared entities can have legitimate explanations, including households or shared "
            "infrastructure.",
            "- The displayed score is a ranking score from a class-balanced model, not a "
            "calibrated probability.",
            "- A production version would require access controls, audit logging, monitored data "
            "quality, model governance, and an approved investigator workflow.",
            "",
            "## Day 9 handoff",
            "",
            "Consolidate the experiment report, model card, limitations, and architecture into "
            "a polished interview-ready documentation set.",
            "",
        ]
    )


def run() -> tuple[dict[str, Any], Path]:
    bundle = load_dashboard_bundle(PROJECT_ROOT)
    graph_summaries = []
    graph_node_types: set[str] = set()
    figures_serialized = True
    for transaction_id in bundle.alerts["transaction_id"].astype(str):
        graph, summary = build_local_connection_graph(
            bundle.transactions,
            transaction_id,
            max_neighbour_transactions=20,
        )
        graph_summaries.append(summary)
        graph_node_types.update(values["node_type"] for _, values in graph.nodes(data=True))
        figure_json = connection_figure(graph).to_json()
        figures_serialized = figures_serialized and bool(figure_json)

    actions = set(bundle.alerts["recommended_action"])
    gates = {
        "day7_explanations_audited": bundle.explanation_artifact["status"] == "PASS"
        and all(bundle.explanation_artifact["gates"].values()),
        "evaluation_targets_absent": not TARGET_ALIASES.intersection(bundle.alerts.columns)
        and not TARGET_ALIASES.intersection(bundle.transactions.columns),
        "strict_historical_connections": all(
            summary["strictly_historical"] for summary in graph_summaries
        ),
        "email_domain_hubs_excluded": not EMAIL_NODE_TYPES.intersection(graph_node_types),
        "focused_views_capped": all(
            summary["displayed_historical_neighbours"] <= 20 for summary in graph_summaries
        ),
        "figures_serialized": figures_serialized,
        "review_only_recommendations": actions == {"INVESTIGATOR_REVIEW"},
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "model": bundle.explanation_artifact["model"],
        "capacity_fraction": bundle.explanation_artifact["capacity_fraction"],
        "capacity_tie_audit": bundle.explanation_artifact["capacity_tie_audit"],
        "alert_count": len(bundle.alerts),
        "connection_views_rendered": len(graph_summaries),
        "maximum_view_nodes": max(summary["node_count"] for summary in graph_summaries),
        "maximum_displayed_historical_neighbours": max(
            summary["displayed_historical_neighbours"] for summary in graph_summaries
        ),
        "node_types_rendered": sorted(graph_node_types),
    }
    if result["status"] != "PASS":
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"Day 8 dashboard audit failed: {', '.join(failed)}")

    artifact_dir = PROJECT_ROOT / "artifacts" / "day8"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "dashboard_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path = PROJECT_ROOT / "reports" / "day8_dashboard.md"
    report_path.write_text(_render_report(result), encoding="utf-8")
    return result, report_path


def main() -> None:
    result, report_path = run()
    print(
        f"Dashboard audit: {result['status']} | {result['alert_count']} alerts | "
        f"{result['connection_views_rendered']} views"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
