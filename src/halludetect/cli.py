import argparse
import json
from pathlib import Path

from .config import DEFAULT_MODEL_PATH
from .dataset import Sample
from .heuristics import heuristic_judgement
from .model import load_model, predict_labels, predict_proba


def _load_model_or_none(path: Path):
    try:
        return load_model(path)
    except FileNotFoundError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hallucination detector CLI")
    parser.add_argument("--response", help="Model response to analyze.")
    parser.add_argument("--context", default="", help="Optional conversation context.")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to a trained model.",
    )
    parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Use only heuristic detector.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output a JSON payload.",
    )
    return parser.parse_args()


def _prompt_if_missing(value: str | None, label: str) -> str:
    if value:
        return value
    return input(f"{label}: ").strip()


def main() -> None:
    args = parse_args()
    response = _prompt_if_missing(args.response, "Response")
    context = args.context or ""

    if args.heuristic_only:
        result = heuristic_judgement(response, context)
        payload = {
            "label": result.label,
            "score": result.score,
            "reason": result.reason,
            "source": "heuristic",
        }
    else:
        model_path = Path(args.model_path)
        model = _load_model_or_none(model_path)
        if model is None:
            result = heuristic_judgement(response, context)
            payload = {
                "label": result.label,
                "score": result.score,
                "reason": result.reason,
                "source": "heuristic",
            }
        else:
            sample = [Sample(response=response, context=context)]
            label = predict_labels(model, sample)[0]
            score = predict_proba(model, sample)[0]
            payload = {
                "label": label,
                "score": score,
                "source": "model",
            }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Label: {payload['label']}")
        print(f"Score: {payload['score']:.2f}")
        if payload.get("reason"):
            print(f"Reason: {payload['reason']}")
        print(f"Source: {payload['source']}")


if __name__ == "__main__":
    main()
