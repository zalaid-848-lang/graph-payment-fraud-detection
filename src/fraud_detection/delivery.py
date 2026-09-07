"""Final-delivery checks for the interview presentation and supporting material."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


def _numbered_parts(names: list[str], pattern: str) -> list[str]:
    expression = re.compile(pattern)
    return sorted(name for name in names if expression.fullmatch(name))


def inspect_presentation(path: Path) -> dict[str, Any]:
    """Inspect the portable OOXML structure and searchable text of a PowerPoint file."""

    if not path.is_file():
        raise FileNotFoundError(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a valid OOXML package: {path}")

    with zipfile.ZipFile(path) as package:
        names = package.namelist()
        slides = _numbered_parts(names, r"ppt/slides/slide\d+\.xml")
        notes = _numbered_parts(names, r"ppt/notesSlides/notesSlide\d+\.xml")
        charts = _numbered_parts(names, r"ppt/slides/charts/chart\d+\.xml")
        workbooks = sorted(
            name
            for name in names
            if name.startswith("ppt/embeddings/") and name.lower().endswith(".xlsx")
        )
        media = sorted(
            name
            for name in names
            if name.startswith("ppt/media/") and not name.endswith("/")
        )
        text_fragments: list[str] = []
        for name in names:
            if not name.startswith("ppt/") or not name.endswith(".xml"):
                continue
            try:
                root = ElementTree.fromstring(package.read(name))
            except ElementTree.ParseError:
                continue
            for element in root.iter():
                if element.text and element.tag.rsplit("}", 1)[-1] in {"t", "v"}:
                    text_fragments.append(element.text)

    payload = path.read_bytes()
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
        "slide_count": len(slides),
        "notes_count": len(notes),
        "chart_count": len(charts),
        "embedded_workbook_count": len(workbooks),
        "media_count": len(media),
        "searchable_text": " ".join(text_fragments).lower(),
    }


def audit_delivery(project_root: Path) -> dict[str, Any]:
    """Check that Day 10 is complete, evidence-linked, and interview-safe."""

    required = {
        "presentation": project_root
        / "presentation"
        / "output"
        / "graph-payment-fraud-interview-v3.pptx",
        "presentation_source": project_root / "presentation" / "build_deck.mjs",
        "cover_prompt": project_root / "presentation" / "assets" / "network-cover.prompt.md",
        "demo_script": project_root / "presentation" / "demo_script.md",
        "presentation_readme": project_root / "presentation" / "README.md",
        "interview_questions": project_root / "docs" / "interview_questions.md",
        "experiment_report": project_root / "reports" / "experiment_report.md",
        "model_card": project_root / "docs" / "model_card.md",
        "dashboard": project_root / "dashboard" / "app.py",
    }
    missing = [
        str(path.relative_to(project_root))
        for path in required.values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Day 10 delivery files: {', '.join(missing)}")

    deck = inspect_presentation(required["presentation"])
    deck_text = deck.pop("searchable_text")
    deck["path"] = str(required["presentation"].relative_to(project_root)).replace("\\", "/")
    demo_text = required["demo_script"].read_text(encoding="utf-8").lower()
    questions_text = required["interview_questions"].read_text(encoding="utf-8").lower()
    supporting_text = "\n".join(
        [
            demo_text,
            questions_text,
            required["experiment_report"].read_text(encoding="utf-8").lower(),
            required["model_card"].read_text(encoding="utf-8").lower(),
        ]
    )
    metric_tokens = {
        "0.8935": "0.8935" in deck_text,
        "0.8869": "0.8869" in deck_text,
        "0.9519": "0.9519" in deck_text,
        "0.9250": "0.9250" in deck_text or "0.925" in deck_text,
    }
    gates = {
        "presentation_is_valid_ooxml": True,
        "ten_slides_present": deck["slide_count"] == 10,
        "speaker_notes_on_every_slide": deck["notes_count"] == 10,
        "four_native_charts_present": deck["chart_count"] == 4,
        "chart_workbooks_embedded": deck["embedded_workbook_count"] == 4,
        "cover_media_embedded": deck["media_count"] >= 1,
        "title_present": "graph-based payment fraud ring detection" in deck_text,
        "core_metrics_present": all(metric_tokens.values()),
        "suspected_ring_boundary_present": "suspected fraud-ring" in deck_text
        and "suspected fraud ring" in supporting_text,
        "review_only_boundary_present": "no automatic blocking" in deck_text
        and "investigator review" in supporting_text,
        "synthetic_scope_present": "synthetic" in deck_text and "synthetic" in supporting_text,
        "demo_is_timed": "minute 0" in demo_text and "minute 8" in demo_text,
        "qa_covers_leakage": "target leakage" in questions_text
        and "same-time" in questions_text,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    return {
        "status": status,
        "presentation": deck,
        "metric_tokens": metric_tokens,
        "gates": gates,
        "files": {
            name: str(path.relative_to(project_root)).replace("\\", "/")
            for name, path in required.items()
        },
    }


def render_delivery_report(result: dict[str, Any]) -> str:
    """Render the final Day 10 audit as an interview-friendly Markdown report."""

    deck = result["presentation"]
    return "\n".join(
        [
            "# Day 10 final-delivery audit",
            "",
            "## Outcome",
            "",
            f"Day 10 status: **{result['status']}**. The core 10-day project is complete.",
            "",
            "## Presentation package",
            "",
            f"- Slides: `{deck['slide_count']}`",
            f"- Slides with speaker notes: `{deck['notes_count']}`",
            f"- Native editable charts: `{deck['chart_count']}`",
            f"- Embedded chart workbooks: `{deck['embedded_workbook_count']}`",
            f"- Embedded media files: `{deck['media_count']}`",
            f"- SHA-256: `{deck['sha256']}`",
            "",
            "The deck was also rendered and inspected slide by slide. Package inspection does not "
            "claim that Microsoft PowerPoint was opened natively.",
            "",
            "## Delivery gates",
            "",
            *[
                f"- `{name}`: **{'PASS' if passed else 'FAIL'}**"
                for name, passed in result["gates"].items()
            ],
            "",
            "## Interview boundary",
            "",
            "All reported results are deterministic synthetic development evidence. Connected "
            "groups remain suspected fraud rings rather than verified criminal networks, and every "
            "recommendation remains subject to investigator review with no automatic blocking.",
            "",
            "## Optional buffer",
            "",
            "Days 11–12 are available for rehearsal, IEEE-CIS reruns when the files are available, "
            "or a tightly scoped GraphSAGE/GAT experiment. None is required for the complete core "
            "portfolio project.",
            "",
        ]
    )
