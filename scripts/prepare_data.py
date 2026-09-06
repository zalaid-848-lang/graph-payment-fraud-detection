"""Prepare deterministic synthetic data or locally supplied IEEE-CIS files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.data.ingest import ieee_files_available, load_ieee_cis  # noqa: E402
from fraud_detection.data.split import assign_temporal_splits  # noqa: E402
from fraud_detection.data.synthetic import generate_synthetic_transactions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("auto", "synthetic", "ieee"), default="auto")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ieee-cis"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/transactions.csv")
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--synthetic-rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--purge-seconds", type=int, default=86_400)
    return parser.parse_args()


def _resolve_from_project(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare(args: argparse.Namespace) -> tuple[Path, Path, dict[str, object]]:
    raw_dir = _resolve_from_project(args.raw_dir)
    output = _resolve_from_project(args.output)
    mode = args.mode
    if mode == "auto":
        mode = "ieee" if ieee_files_available(raw_dir) else "synthetic"

    if mode == "ieee":
        frame = load_ieee_cis(raw_dir, max_rows=args.max_rows)
    else:
        frame = generate_synthetic_transactions(args.synthetic_rows, args.seed)

    frame = assign_temporal_splits(
        frame,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        purge_seconds=args.purge_seconds,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, lineterminator="\n")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    split_counts = {str(key): int(value) for key, value in frame["split"].value_counts().items()}
    manifest: dict[str, object] = {
        "source": mode,
        "seed": args.seed if mode == "synthetic" else None,
        "row_count": len(frame),
        "fraud_count": int(frame["is_fraud"].sum()),
        "fraud_rate": round(float(frame["is_fraud"].mean()), 8),
        "min_transaction_time": int(frame["transaction_time"].min()),
        "max_transaction_time": int(frame["transaction_time"].max()),
        "split_counts": split_counts,
        "validation": {
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "purge_seconds": args.purge_seconds,
        },
        "output_file": output.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": checksum,
    }
    manifest_path = output.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output, manifest_path, manifest


def main() -> None:
    output, manifest_path, manifest = prepare(parse_args())
    print(f"Prepared {manifest['row_count']} {manifest['source']} rows at {output}")
    print(f"Fraud labels: {manifest['fraud_count']} ({manifest['fraud_rate']:.2%})")
    print(f"Split counts: {manifest['split_counts']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
