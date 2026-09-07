"""Investigator-facing Streamlit dashboard for the audited Day 7 alert queue."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from fraud_detection.dashboard import (  # noqa: E402
    DashboardBundle,
    build_local_connection_graph,
    connection_figure,
    load_dashboard_bundle,
)

DATA_ROOT = Path(os.environ.get("FRAUD_DASHBOARD_PROJECT_ROOT", CODE_ROOT))

st.set_page_config(
    page_title="Fraud investigation queue",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(project_root: str) -> DashboardBundle:
    return load_dashboard_bundle(Path(project_root))


def _render_reason(reason: dict[str, object]) -> None:
    evidence = reason.get("evidence", {})
    evidence_text = " · ".join(
        f"{str(key).replace('_', ' ').title()}: {value}" for key, value in evidence.items()
    )
    with st.container(border=True):
        st.caption(str(reason["code"]))
        st.markdown(f"**{reason['title']}**")
        st.write(str(reason["detail"]))
        if evidence_text:
            st.caption(evidence_text)


try:
    bundle = load_data(str(DATA_ROOT))
except (FileNotFoundError, ValueError) as error:
    st.error("The verified dashboard inputs are not ready.")
    st.code(str(error))
    st.info("Run the data preparation and Day 4–7 scripts, then reload this page.")
    st.stop()

artifact = bundle.explanation_artifact
tie_audit = artifact["capacity_tie_audit"]

st.title("Fraud investigation queue")
st.caption("Prioritised transactions and historical connections for human review")
st.info(
    "Connections are investigation leads—not confirmed fraud rings. Risk scores rank review "
    "priority; they are not calibrated fraud probabilities. No account action is automated."
)

metric_columns = st.columns(4)
metric_columns[0].metric("Alerts to review", f"{len(bundle.alerts):,}")
metric_columns[1].metric("Investigation capacity", f"{artifact['capacity_fraction']:.1%}")
metric_columns[2].metric("Selected model", "Graph LightGBM")
metric_columns[3].metric("Transactions at cutoff", tie_audit["rows_at_cutoff"])

if tie_audit["rows_at_cutoff"] > tie_audit["selected_from_cutoff_tie"]:
    st.warning(
        f"{tie_audit['rows_at_cutoff']} transactions share the cutoff score, but only "
        f"{tie_audit['selected_from_cutoff_tie']} fit within capacity. Their internal order "
        "is deterministic and does not imply a measured risk difference."
    )

st.subheader("Ranked alert queue")
reason_options = sorted(
    {code.strip() for codes in bundle.alerts["reason_codes"] for code in codes.split(",")}
)
selected_reason = st.selectbox("Filter by reason", ["All reasons", *reason_options])
queue = bundle.alerts
if selected_reason != "All reasons":
    queue = queue[queue["reason_codes"].str.contains(selected_reason, regex=False)]

display_queue = queue[
    ["rank", "transaction_id", "risk_score", "amount", "product_code", "primary_reason"]
].rename(
    columns={
        "rank": "Rank",
        "transaction_id": "Transaction",
        "risk_score": "Risk score",
        "amount": "Amount",
        "product_code": "Product",
        "primary_reason": "Primary reason",
    }
)
st.dataframe(
    display_queue,
    hide_index=True,
    width="stretch",
    column_config={
        "Risk score": st.column_config.ProgressColumn(
            min_value=0.0,
            max_value=1.0,
            format="%.3f",
        ),
        "Amount": st.column_config.NumberColumn(format="%.2f"),
    },
)

if queue.empty:
    st.warning("No alerts match this reason filter.")
    st.stop()

alert_options = queue["transaction_id"].astype(str).tolist()
alert_lookup = bundle.alerts.set_index("transaction_id")
selected_transaction = st.selectbox(
    "Open alert",
    alert_options,
    format_func=lambda value: f"#{int(alert_lookup.loc[value, 'rank'])} · {value}",
)
alert = alert_lookup.loc[selected_transaction]

left, right = st.columns([0.9, 1.6], gap="large")
with left:
    st.subheader(f"Alert #{int(alert['rank'])}")
    st.metric("Risk ranking score", f"{alert['risk_score']:.3f}")
    st.write(f"**Transaction:** `{selected_transaction}`")
    st.write(f"**Amount:** {alert['amount']:,.2f}")
    st.write(f"**Product:** {alert['product_code']}")
    st.write(f"**Recommended action:** {alert['recommended_action']}")
    st.markdown("#### Why it was escalated")
    for reason in alert["reasons"]:
        _render_reason(reason)

with right:
    st.subheader("Historical connection view")
    graph, summary = build_local_connection_graph(
        bundle.transactions,
        selected_transaction,
        max_neighbour_transactions=20,
    )
    st.plotly_chart(connection_figure(graph), width="stretch", config={"displaylogo": False})
    graph_metrics = st.columns(3)
    graph_metrics[0].metric("Specific entities", summary["available_specific_entities"])
    graph_metrics[1].metric("Earlier connected txns", summary["historical_neighbour_count"])
    graph_metrics[2].metric("Shown", summary["displayed_historical_neighbours"])
    if summary["truncated"]:
        st.caption("The view is capped at the 20 most connected earlier transactions.")

with st.expander("How to interpret this view"):
    st.markdown(
        """
        - The red node is the selected transaction; orange nodes are strictly earlier transactions.
        - Entity nodes show masked card, customer, device, address, and recipient proxies.
        - Email-domain nodes are excluded because common domains can create misleading hubs.
        - A connection is a reason to inspect context, not proof that participants form a
          criminal network.
        - Evaluation labels are intentionally absent from the dashboard data layer.
        """
    )

st.divider()
st.caption(artifact["disclaimer"])
