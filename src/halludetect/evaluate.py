from collections import Counter

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .config import LABELS
from .dataset import Sample, labels_from_samples
from .features import tokenize
from .model import predict_labels


def _encode_labels(labels: list[str]) -> list[int]:
    return [LABELS[label] for label in labels]


def evaluate_model(model, samples: list[Sample]) -> tuple[dict, dict]:
    y_true_labels = labels_from_samples(samples)
    y_true = _encode_labels(y_true_labels)
    y_pred_labels = predict_labels(model, samples)
    y_pred = _encode_labels(y_pred_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=LABELS["hallucination"]
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    errors = build_error_report(samples, y_true_labels, y_pred_labels)
    return metrics, errors


def _overlap_ratio(response: str, context: str) -> float:
    response_tokens = set(tokenize(response))
    context_tokens = set(tokenize(context))
    if not response_tokens:
        return 0.0
    return len(response_tokens & context_tokens) / max(len(response_tokens), 1)


def _categorize_error(sample: Sample) -> list[str]:
    categories = []
    response = sample.response.lower()
    context = (sample.context or "").lower()
    if any(token in response for token in ("not", "no", "never")):
        categories.append("negation")
    if any(char.isdigit() for char in response):
        categories.append("numeric")
    if not context:
        categories.append("empty_context")
    if _overlap_ratio(response, context) < 0.2:
        categories.append("low_overlap")
    return categories or ["uncategorized"]


def build_error_report(
    samples: list[Sample], y_true: list[str], y_pred: list[str]
) -> dict:
    false_positives = []
    false_negatives = []
    categories = Counter()
    for sample, truth, predicted in zip(samples, y_true, y_pred):
        if truth == predicted:
            continue
        error = {
            "response": sample.response,
            "context": sample.context,
            "expected": truth,
            "predicted": predicted,
        }
        if truth == "non-hallucination" and predicted == "hallucination":
            false_positives.append(error)
        elif truth == "hallucination" and predicted == "non-hallucination":
            false_negatives.append(error)
        categories.update(_categorize_error(sample))

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "category_counts": dict(categories),
    }
