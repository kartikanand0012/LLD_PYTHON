from dataclasses import dataclass

from .features import SOURCE_CLAIM_PHRASES, UNCERTAINTY_PHRASES, tokenize


@dataclass(frozen=True)
class HeuristicResult:
    label: str
    score: float
    reason: str


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in phrases)


def _overlap_ratio(response: str, context: str) -> float:
    response_tokens = set(tokenize(response))
    context_tokens = set(tokenize(context))
    if not response_tokens:
        return 0.0
    return len(response_tokens & context_tokens) / max(len(response_tokens), 1)


def heuristic_judgement(response: str, context: str) -> HeuristicResult:
    response = response or ""
    context = context or ""
    response_lower = response.lower()
    context_lower = context.lower()
    response_tokens = tokenize(response_lower)
    overlap = _overlap_ratio(response_lower, context_lower)

    if _contains_phrase(response_lower, UNCERTAINTY_PHRASES):
        return HeuristicResult(
            label="non-hallucination",
            score=0.15,
            reason="Response signals uncertainty or missing context.",
        )
    if not context_lower and any(char.isdigit() for char in response_lower):
        return HeuristicResult(
            label="hallucination",
            score=0.85,
            reason="Specific numeric claim without context.",
        )
    if not context_lower and len(response_tokens) >= 6:
        return HeuristicResult(
            label="hallucination",
            score=0.75,
            reason="Detailed claim without supporting context.",
        )
    if _contains_phrase(response_lower, SOURCE_CLAIM_PHRASES) and overlap < 0.2:
        return HeuristicResult(
            label="hallucination",
            score=0.7,
            reason="Source claim with low overlap to context.",
        )
    if overlap < 0.2 and len(response_tokens) >= 6:
        return HeuristicResult(
            label="hallucination",
            score=0.65,
            reason="Low lexical overlap with context.",
        )
    return HeuristicResult(
        label="non-hallucination",
        score=0.35,
        reason="No strong hallucination signal detected.",
    )
