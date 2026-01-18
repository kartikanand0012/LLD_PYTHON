import argparse
from pathlib import Path

from halludetect.dataset import load_dataset
from halludetect.model import save_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hallucination detector.")
    parser.add_argument(
        "--train-path",
        default="data/train.csv",
        help="Path to training data.",
    )
    parser.add_argument(
        "--model-path",
        default="models/halludetect.joblib",
        help="Path to save the trained model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_samples = load_dataset(args.train_path)
    model = train_model(train_samples)
    model_path = save_model(model, Path(args.model_path))
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
