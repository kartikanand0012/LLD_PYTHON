import re
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

UNCERTAINTY_PHRASES = (
    "not enough context",
    "cannot verify",
    "can't verify",
    "not specified",
    "not provided",
    "unknown",
    "unclear",
)

SOURCE_CLAIM_PHRASES = (
    "according to",
    "the report",
    "the document",
    "the context",
    "as stated",
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _get_value(sample, field: str) -> str:
    if isinstance(sample, dict):
        return str(sample.get(field, "") or "")
    if hasattr(sample, field):
        return str(getattr(sample, field) or "")
    return ""


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def select_text(samples: Iterable[object]) -> list[str]:
    combined = []
    for sample in samples:
        response = _get_value(sample, "response").strip()
        context = _get_value(sample, "context").strip()
        merged = f"{response} {context}".strip()
        combined.append(merged)
    return combined


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> int:
    lower = text.lower()
    return int(any(phrase in lower for phrase in phrases))


def extract_heuristics(samples: Iterable[object]) -> csr_matrix:
    rows = []
    for sample in samples:
        response = _get_value(sample, "response")
        context = _get_value(sample, "context")
        response_tokens = tokenize(response)
        context_tokens = tokenize(context)
        response_len = len(response_tokens)
        context_len = len(context_tokens)
        overlap = (
            len(set(response_tokens) & set(context_tokens)) / max(response_len, 1)
        )
        novelty = 1.0 - overlap
        has_number = int(bool(re.search(r"\d", response)))
        has_uncertainty = _contains_phrase(response, UNCERTAINTY_PHRASES)
        has_source_claim = _contains_phrase(response, SOURCE_CLAIM_PHRASES)
        context_empty = int(context_len == 0)
        response_context_ratio = response_len / max(context_len, 1)
        rows.append(
            [
                response_len,
                context_len,
                overlap,
                novelty,
                has_number,
                has_uncertainty,
                has_source_claim,
                context_empty,
                response_context_ratio,
            ]
        )
    features = np.array(rows, dtype=float)
    return csr_matrix(features)


def build_feature_extractor(max_features: int = 2000) -> FeatureUnion:
    text_pipeline = Pipeline(
        steps=[
            ("selector", FunctionTransformer(select_text, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=max_features,
                    min_df=1,
                ),
            ),
        ]
    )
    heuristic_pipeline = Pipeline(
        steps=[
            ("selector", FunctionTransformer(extract_heuristics, validate=False)),
        ]
    )
    return FeatureUnion([("tfidf", text_pipeline), ("heuristics", heuristic_pipeline)])
