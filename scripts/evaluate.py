import argparse
import json
from pathlib import Path

from halludetect.dataset import load_dataset
from halludetect.evaluate import evaluate_model
from halludetect.model import save_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hallucination detector.")
    parser.add_argument(
        "--train-path",
        default="data/train.csv",
        help="Path to training data.",
    )
    parser.add_argument(
        "--test-path",
        default="data/test.csv",
        help="Path to test data.",
    )
    parser.add_argument(
        "--model-path",
        default="models/halludetect.joblib",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--metrics-path",
        default="reports/metrics.json",
        help="Path to write evaluation metrics.",
    )
    parser.add_argument(
        "--analysis-path",
        default="reports/error_analysis.md",
        help="Path to write error analysis.",
    )
    return parser.parse_args()


def _write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _write_error_analysis(path: Path, errors: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    false_positives = errors.get("false_positives", [])
    false_negatives = errors.get("false_negatives", [])
    category_counts = errors.get("category_counts", {})

    lines = [
        "# Error analysis",
        "",
        f"- False positives: {len(false_positives)}",
        f"- False negatives: {len(false_negatives)}",
        "",
        "## Category counts",
    ]
    for category, count in sorted(category_counts.items()):
        lines.append(f"- {category}: {count}")

    def _format_examples(title: str, items: list[dict]) -> list[str]:
        section = [f"## {title}"]
        if not items:
            section.append("- None")
            return section
        for item in items[:5]:
            section.extend(
                [
                    "- Example",
                    f"  - Response: {item['response']}",
                    f"  - Context: {item['context'] or 'N/A'}",
                    f"  - Expected: {item['expected']}",
                    f"  - Predicted: {item['predicted']}",
                ]
            )
        return section

    lines.extend(_format_examples("False positives", false_positives))
    lines.extend(_format_examples("False negatives", false_negatives))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_samples = load_dataset(args.train_path)
    test_samples = load_dataset(args.test_path)
    model = train_model(train_samples)
    save_model(model, Path(args.model_path))
    metrics, errors = evaluate_model(model, test_samples)

    _write_metrics(Path(args.metrics_path), metrics)
    _write_error_analysis(Path(args.analysis_path), errors)
    print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
