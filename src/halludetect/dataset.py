import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Sample:
    response: str
    context: str
    label: str | None = None


def load_dataset(path: str | Path) -> list[Sample]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        samples: list[Sample] = []
        for row in reader:
            response = (row.get("response") or "").strip()
            context = (row.get("context") or "").strip()
            label = (row.get("label") or "").strip() or None
            samples.append(Sample(response=response, context=context, label=label))
    return samples


def samples_to_payload(samples: Iterable[Sample]) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for sample in samples:
        payload.append(
            {"response": sample.response, "context": sample.context or ""}
        )
    return payload


def labels_from_samples(samples: Iterable[Sample]) -> list[str]:
    labels: list[str] = []
    for sample in samples:
        if sample.label is None:
            raise ValueError("Sample is missing a label.")
        labels.append(sample.label)
    return labels
