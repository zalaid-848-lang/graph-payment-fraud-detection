import tempfile
import unittest
import zipfile
from pathlib import Path

from fraud_detection.delivery import audit_delivery, inspect_presentation, render_delivery_report


def write_fake_deck(path: Path) -> None:
    slide_text = (
        "Graph-Based Payment Fraud Ring Detection suspected fraud-ring synthetic "
        "No automatic blocking 0.8935 0.8869 0.9519 0.9250"
    )
    xml = f'<root xmlns:a="urn:test"><a:t>{slide_text}</a:t></root>'
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("[Content_Types].xml", "<Types/>")
        for number in range(1, 11):
            package.writestr(f"ppt/slides/slide{number}.xml", xml)
            package.writestr(f"ppt/notesSlides/notesSlide{number}.xml", xml)
        for number in range(1, 5):
            package.writestr(f"ppt/slides/charts/chart{number}.xml", xml)
            package.writestr(f"ppt/embeddings/chart{number}.xlsx", b"workbook")
        package.writestr("ppt/media/image1.png", b"image")


class Day10DeliveryTests(unittest.TestCase):
    def test_presentation_inspection_counts_editable_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            deck_path = Path(directory) / "deck.pptx"
            write_fake_deck(deck_path)
            result = inspect_presentation(deck_path)

        self.assertEqual(result["slide_count"], 10)
        self.assertEqual(result["notes_count"], 10)
        self.assertEqual(result["chart_count"], 4)
        self.assertEqual(result["embedded_workbook_count"], 4)
        self.assertEqual(result["media_count"], 1)
        self.assertEqual(len(result["sha256"]), 64)

    def test_delivery_audit_enforces_scope_and_safety_language(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                root / "presentation" / "assets" / "network-cover.prompt.md",
                root / "presentation" / "demo_script.md",
                root / "presentation" / "README.md",
                root / "presentation" / "build_deck.mjs",
                root / "docs" / "interview_questions.md",
                root / "docs" / "model_card.md",
                root / "reports" / "experiment_report.md",
                root / "dashboard" / "app.py",
            ]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    "Synthetic suspected fraud ring. Investigator review only. "
                    "No automatic blocking. Target leakage and same-time batching. "
                    "Minute 0 through Minute 8.",
                    encoding="utf-8",
                )
            deck_path = (
                root
                / "presentation"
                / "output"
                / "graph-payment-fraud-interview-v3.pptx"
            )
            deck_path.parent.mkdir(parents=True)
            write_fake_deck(deck_path)

            result = audit_delivery(root)
            report = render_delivery_report(result)

        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(result["gates"].values()))
        self.assertIn("core 10-day project is complete", report)

    def test_invalid_package_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pptx"
            path.write_text("not a PowerPoint", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Not a valid OOXML"):
                inspect_presentation(path)


if __name__ == "__main__":
    unittest.main()
