"""Audit the final presentation and interview delivery package."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.delivery import audit_delivery, render_delivery_report  # noqa: E402


def run() -> tuple[dict, Path]:
    result = audit_delivery(PROJECT_ROOT)
    if result["status"] != "PASS":
        failed = [name for name, passed in result["gates"].items() if not passed]
        raise RuntimeError(f"Day 10 delivery audit failed: {', '.join(failed)}")

    artifact_dir = PROJECT_ROOT / "artifacts" / "day10"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "final_delivery_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path = PROJECT_ROOT / "reports" / "day10_final_delivery.md"
    report_path.write_text(render_delivery_report(result), encoding="utf-8")
    return result, report_path


def main() -> None:
    result, report_path = run()
    deck = result["presentation"]
    print(
        f"Day 10 delivery: {result['status']} | slides={deck['slide_count']} | "
        f"charts={deck['chart_count']} | notes={deck['notes_count']}"
    )
    print(f"Audit report: {report_path}")


if __name__ == "__main__":
    main()
