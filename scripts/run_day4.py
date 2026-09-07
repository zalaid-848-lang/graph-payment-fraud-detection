"""Run Day 4 causal structural graph-feature generation and auditing."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.data.schema import validate_transactions  # noqa: E402
from fraud_detection.features import (  # noqa: E402
    STRUCTURAL_FEATURE_COLUMNS,
    CausalGraphFeatureBuilder,
)


def _feature_audit(frame: pd.DataFrame, features: pd.DataFrame) -> dict[str, Any]:
    matrix = features[list(STRUCTURAL_FEATURE_COLUMNS)].to_numpy(dtype=float)
    gates = {
        "row_count_matches_source": len(features) == len(frame),
        "transaction_ids_unique": not features["transaction_id"].duplicated().any(),
        "transaction_ids_match_source": set(features["transaction_id"])
        == set(frame["transaction_id"].astype(str)),
        "target_columns_absent": not {"is_fraud", "isFraud", "target", "label"}.intersection(
            features.columns
        ),
        "all_features_finite": bool(np.isfinite(matrix).all()),
        "all_features_non_negative": bool((matrix >= 0).all()),
        "pagerank_bounded": bool(
            features[["specific_max_pagerank", "specific_mean_pagerank"]].le(1).all().all()
        ),
        "clustering_bounded": bool(
            features[["specific_max_clustering", "specific_mean_clustering"]].le(1).all().all()
        ),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "row_count": len(features),
        "feature_count": len(STRUCTURAL_FEATURE_COLUMNS),
    }


def _split_summaries(merged: pd.DataFrame) -> dict[str, Any]:
    columns = [
        "graph_shared_transaction_count",
        "specific_shared_transaction_count",
        "specific_max_entity_degree",
        "specific_max_component_transactions",
        "specific_max_pagerank",
        "specific_max_clustering",
    ]
    summaries: dict[str, Any] = {}
    for split, group in merged.groupby("split", sort=False):
        summaries[str(split)] = {
            "rows": len(group),
            "feature_means": {column: float(group[column].mean()) for column in columns},
        }
    return summaries


def _label_diagnostic(merged: pd.DataFrame) -> dict[str, Any]:
    eligible = merged[merged["split"].isin(["validation", "test"])]
    columns = [
        "specific_shared_transaction_count",
        "specific_max_entity_degree",
        "specific_max_component_transactions",
    ]
    return {
        str(label): {
            "rows": len(group),
            "means": {column: float(group[column].mean()) for column in columns},
        }
        for label, group in eligible.groupby("is_fraud")
    }


def _render_report(result: dict[str, Any]) -> str:
    audit = result["feature_audit"]
    builder = result["builder_audit"]
    labels = result["post_generation_label_diagnostic"]
    non_fraud = labels.get("0", {}).get("means", {})
    fraud = labels.get("1", {}).get("means", {})
    lines = [
        "# Day 4 causal graph-feature report",
        "",
        "## Executive summary",
        "",
        f"Generated `{audit['feature_count']}` structural features for "
        f"`{audit['row_count']:,}` transactions. Every feature was materialized before its "
        "equal-time transaction batch entered graph history. The output contains no target "
        "column and all feature-audit gates pass.",
        "",
        "> These features support ranking for investigator review. They do not establish that a "
        "connected group is a confirmed fraud ring.",
        "",
        "## Leakage and quality gates",
        "",
        f"Overall status: **{audit['status']}**",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — `{gate}`" for gate, passed in audit["gates"].items()
    )
    lines.extend(
        [
            "",
            "## Feature families",
            "",
            "- Historical degree for each entity type",
            "- Known/new and shared-entity transaction counts",
            "- Email-hub-excluded component size and transaction count",
            "- Lagged weighted PageRank on the specific-entity projection",
            "- Historical clustering coefficient on the specific-entity projection",
            "",
            f"PageRank was refreshed `{builder['pagerank_refresh_count']}` times across "
            f"`{builder['time_batch_count']}` chronological batches, every "
            f"`{builder['pagerank_refresh_batches']}` batches. Snapshot age is emitted explicitly.",
            "",
            "## Hub-conscious design",
            "",
            "Payer and recipient email domains remain in the full graph and retain degree "
            "features. They are excluded from the specific projection and component features "
            "because common domains otherwise merge unrelated activity into a giant component.",
            "",
            "## Post-generation label diagnostic",
            "",
            "Labels were joined only after feature generation for this diagnostic; they were not "
            "available to the feature builder.",
            "",
            "| Structural feature | Non-fraud mean | Fraud mean |",
            "|---|---:|---:|",
        ]
    )
    for column in (
        "specific_shared_transaction_count",
        "specific_max_entity_degree",
        "specific_max_component_transactions",
    ):
        lines.append(
            f"| `{column}` | {non_fraud.get(column, 0.0):.3f} | {fraud.get(column, 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "Synthetic label differences validate that the connected pattern reaches the feature "
            "table; they do not estimate real-world performance.",
            "",
            "## Reproducibility and outputs",
            "",
            "- `data/processed/graph_features.csv`: deterministic transaction feature table",
            "- `artifacts/day4/feature_audit.json`: machine-readable audit and summaries",
            "- `docs/graph_feature_dictionary.md`: feature definitions and scaling limitation",
            "",
            "## Day 5 handoff",
            "",
            "Join these structural features to the existing tabular inputs, train the first "
            "tabular-plus-graph model, and compare it fairly with Day 2. Neighbouring-fraud ratios "
            "will be added only with a documented label-maturation delay and training-history-only "
            "labels.",
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
    graph_config = config["graph"]
    builder = CausalGraphFeatureBuilder(
        pagerank_refresh_batches=graph_config["pagerank_refresh_batches"],
        pagerank_alpha=graph_config["pagerank_alpha"],
    )
    features = builder.build(frame)
    audit = _feature_audit(frame, features)
    if audit["status"] != "PASS":
        raise RuntimeError("Causal graph feature audit failed")

    merged = frame[["transaction_id", "split", "is_fraud"]].merge(
        features, on="transaction_id", how="left", validate="one_to_one"
    )
    result = {
        "source": str(frame["source"].iloc[0]),
        "feature_audit": audit,
        "builder_audit": builder.audit_,
        "split_summaries": _split_summaries(merged),
        "post_generation_label_diagnostic": _label_diagnostic(merged),
    }

    output_path = PROJECT_ROOT / "data" / "processed" / "graph_features.csv"
    features.to_csv(output_path, index=False, lineterminator="\n")
    artifact_dir = PROJECT_ROOT / "artifacts" / "day4"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "feature_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path = PROJECT_ROOT / "reports" / "day4_graph_features.md"
    report_path.write_text(_render_report(result), encoding="utf-8")
    return result, report_path


def main() -> None:
    result, report_path = run()
    audit = result["feature_audit"]
    print(
        f"Graph features: {audit['row_count']} rows, "
        f"{audit['feature_count']} features, status={audit['status']}"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
