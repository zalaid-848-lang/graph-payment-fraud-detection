"""Build and audit the consolidated Day 9 interview documentation."""

from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.reporting import (  # noqa: E402
    audit_documentation,
    build_evidence_snapshot,
    load_day9_sources,
    render_experiment_report,
)


def _source_checksums() -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "artifacts" / "day5" / "metrics.json",
        PROJECT_ROOT / "artifacts" / "day6" / "metrics_and_errors.json",
        PROJECT_ROOT / "artifacts" / "day7" / "explanation_audit.json",
        PROJECT_ROOT / "artifacts" / "day8" / "dashboard_audit.json",
    )
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }


def _render_audit_report(result: dict[str, Any]) -> str:
    evidence = result["evidence_summary"]
    audit = result["documentation_audit"]
    documents = audit["documents"]
    return "\n".join(
        [
            "# Day 9 interview documentation audit",
            "",
            "## Outcome",
            "",
            f"Day 9 status: **{result['status']}**. The experiment report, model card, "
            "architecture, and limitations are tied to the verified Day 5–8 artifacts and "
            "preserve the project's responsible-use boundaries.",
            "",
            "## Evidence summary",
            "",
            f"- Data source: `{evidence['source']}`",
            f"- Selected model: `{evidence['selected_model']}`",
            f"- Validation PR-AUC: `{evidence['selected_validation_pr_auc']:.4f}`",
            f"- Frozen test PR-AUC: `{evidence['selected_test_pr_auc']:.4f}`",
            f"- Recall at 5% investigation capacity: "
            f"`{evidence['selected_recall_at_capacity']:.2%}`",
            f"- Test fraud-labelled transactions: `{evidence['test_fraud_labels']}`",
            "",
            "These are synthetic development results, not expected performance at a bank.",
            "",
            "## Documents",
            "",
            "| Document | Path | Characters |",
            "|---|---|---:|",
            *[
                f"| {name.replace('_', ' ').title()} | `{values['path']}` | "
                f"{values['characters']:,} |"
                for name, values in documents.items()
            ],
            "",
            "## Documentation gates",
            "",
            *[
                f"- `{name}`: **{'PASS' if passed else 'FAIL'}**"
                for name, passed in audit["gates"].items()
            ],
            "",
            "## Interview narrative",
            "",
            "The business contribution is not merely a higher model score. The project shows how "
            "to build entity connections with scoring-time history, test graph features through a "
            "controlled logistic-regression ablation, select a nonlinear candidate without test "
            "tuning, evaluate within investigation capacity, and expose review-only evidence with "
            "clear limitations.",
            "",
            "## Day 10 handoff",
            "",
            "Create the interview presentation and demo script, then run the final "
            "clean-clone-style reproducibility check. GraphSAGE remains optional and should not "
            "displace the polished core delivery.",
            "",
        ]
    )


def run() -> tuple[dict[str, Any], Path]:
    sources = load_day9_sources(PROJECT_ROOT)
    evidence = build_evidence_snapshot(sources)
    experiment_path = PROJECT_ROOT / "reports" / "experiment_report.md"
    experiment_path.write_text(render_experiment_report(evidence), encoding="utf-8")
    documentation_audit = audit_documentation(PROJECT_ROOT, evidence)
    result = {
        "status": "PASS"
        if evidence["status"] == "PASS" and documentation_audit["status"] == "PASS"
        else "FAIL",
        "source_checksums": _source_checksums(),
        "evidence_summary": {
            "source": evidence["source"],
            "selected_model": evidence["model_selection"]["selected_model"],
            "selected_validation_pr_auc": evidence["model_selection"]["validation_pr_auc"][
                evidence["model_selection"]["selected_model"]
            ],
            "selected_test_pr_auc": evidence["models"][
                evidence["model_selection"]["selected_model"]
            ]["pr_auc"],
            "selected_recall_at_capacity": evidence["models"][
                evidence["model_selection"]["selected_model"]
            ]["recall_at_capacity"],
            "test_fraud_labels": evidence["test_fraud_labels"],
        },
        "documentation_audit": documentation_audit,
    }
    if result["status"] != "PASS":
        failed = [name for name, passed in documentation_audit["gates"].items() if not passed]
        raise RuntimeError(f"Day 9 documentation audit failed: {', '.join(failed)}")

    artifact_dir = PROJECT_ROOT / "artifacts" / "day9"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "evidence_snapshot.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (artifact_dir / "documentation_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit_path = PROJECT_ROOT / "reports" / "day9_documentation_audit.md"
    audit_path.write_text(_render_audit_report(result), encoding="utf-8")
    return result, audit_path


def main() -> None:
    result, audit_path = run()
    summary = result["evidence_summary"]
    print(
        f"Day 9 documentation: {result['status']} | "
        f"selected={summary['selected_model']} | "
        f"test_pr_auc={summary['selected_test_pr_auc']:.4f}"
    )
    print(f"Audit report: {audit_path}")


if __name__ == "__main__":
    main()
