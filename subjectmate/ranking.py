"""Final passage selection shared by local retrieval and the deployed API.

Kept free of heavy dependencies so the Vercel function can import it.
"""

from typing import Any, Sequence, TypeVar

from subjectmate.settings import Settings

P = TypeVar("P")


def is_course_material(metadata: dict[str, Any]) -> bool:
    """Lecture material ships inside the course ZIP archives; reference books and docs don't."""
    return "!/" in str(metadata.get("source_path", ""))


def select_passages(
    candidates: Sequence[tuple[P, float]],
    settings: Settings,
    rerank_scores: Sequence[float] | None = None,
) -> list[tuple[P, float]]:
    """Choose up to RETRIEVAL_K passages, best first, or none for an off-topic question.

    Without re-ranker scores, candidates keep their cosine similarity and the best one
    must reach MIN_TOP_SCORE. With them, the best re-ranker score must reach
    MIN_RERANK_SCORE, passages below RERANK_PASSAGE_FLOOR are dropped, and COURSE_BOOST
    favours lecture material when ordering (the returned scores are unboosted).
    """
    if not candidates:
        return []
    if rerank_scores is None:
        ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
        return list(ranked[: settings.retrieval_k]) if ranked[0][1] >= settings.min_top_score else []

    scored = [(passage, float(score)) for (passage, _), score in zip(candidates, rerank_scores)]
    if max(score for _, score in scored) < settings.min_rerank_score:
        return []
    kept = [item for item in scored if item[1] >= settings.rerank_passage_floor]
    kept.sort(
        key=lambda item: item[1]
        + (settings.course_boost if is_course_material(item[0].metadata) else 0),
        reverse=True,
    )
    return kept[: settings.retrieval_k]
