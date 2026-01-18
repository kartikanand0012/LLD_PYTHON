from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import DEFAULT_MODEL_PATH, LABEL_NAMES, LABELS
from .dataset import labels_from_samples, samples_to_payload
from .features import build_feature_extractor


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("features", build_feature_extractor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )


def _encode_labels(labels: list[str]) -> list[int]:
    return [LABELS[label] for label in labels]


def _decode_labels(predictions: list[int]) -> list[str]:
    return [LABEL_NAMES[prediction] for prediction in predictions]


def train_model(samples: list[object]) -> Pipeline:
    payload = samples_to_payload(samples)
    labels = labels_from_samples(samples)
    model = build_model()
    model.fit(payload, _encode_labels(labels))
    return model


def save_model(model: Pipeline, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def predict_labels(model: Pipeline, samples: list[object]) -> list[str]:
    payload = samples_to_payload(samples)
    predictions = model.predict(payload)
    return _decode_labels(predictions)


def predict_proba(model: Pipeline, samples: list[object]) -> list[float]:
    payload = samples_to_payload(samples)
    probas = model.predict_proba(payload)
    return [float(score[1]) for score in probas]
