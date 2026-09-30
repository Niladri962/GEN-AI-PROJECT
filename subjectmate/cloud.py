"""Retrieval for the deployed (serverless) app: fastembed + Qdrant's HTTP API.

Mirrors `vectorstore.retrieve` without PyTorch or LangChain, so it fits a Vercel
function. fastembed runs the same BAAI/bge-small-en-v1.5 model as ONNX and produces the
same vectors as the sentence-transformers model used to build the index.
"""

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import httpx

from subjectmate.rag import Retriever
from subjectmate.ranking import select_passages
from subjectmate.settings import Settings

# Directory with pre-downloaded fastembed models (bundled with the deployment); fastembed
# downloads into it on first use when a model is missing.
MODEL_CACHE = os.getenv("FASTEMBED_CACHE_PATH") or None


@dataclass
class CloudPassage:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=1)
def _embedder(model_name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name, cache_dir=MODEL_CACHE)


@lru_cache(maxsize=1)
def _reranker(model_name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name, cache_dir=MODEL_CACHE)


def embed_query(question: str, settings: Settings) -> list[float]:
    [vector] = list(_embedder(settings.embedding_model).embed([settings.embedding_query_prompt + question]))
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [float(value) / norm for value in vector]


def rerank_scores(question: str, passages: list[CloudPassage], settings: Settings) -> list[float]:
    """Cross-encoder probabilities in [0, 1] (fastembed returns logits)."""
    logits = _reranker(settings.reranker_model).rerank(
        question, [passage.page_content for passage in passages]
    )
    return [1 / (1 + math.exp(-float(logit))) for logit in logits]


def _qdrant(settings: Settings) -> httpx.Client:
    headers = {"api-key": settings.qdrant_api_key} if settings.qdrant_api_key else {}
    return httpx.Client(base_url=settings.qdrant_url.rstrip("/"), headers=headers, timeout=30)


def search(question: str, settings: Settings) -> list[tuple[CloudPassage, float]]:
    """Diverse candidates by cosine similarity (Qdrant MMR), best first."""
    count = settings.rerank_candidates if settings.reranker_model else settings.retrieval_k
    with _qdrant(settings) as client:
        response = client.post(
            f"/collections/{settings.qdrant_collection}/points/query",
            json={
                "query": {
                    "nearest": embed_query(question, settings),
                    "mmr": {
                        "diversity": settings.mmr_diversity,
                        "candidates_limit": max(settings.retrieval_fetch_k, 2 * count),
                    },
                },
                "limit": count,
                "score_threshold": settings.min_relevance_score,
                "with_payload": True,
            },
        )
        response.raise_for_status()
    return [
        (
            CloudPassage(point["payload"]["page_content"], point["payload"].get("metadata", {})),
            float(point["score"]),
        )
        for point in response.json()["result"]["points"]
    ]


def retrieve(question: str, settings: Settings) -> list[tuple[CloudPassage, float]]:
    candidates = search(question, settings)
    if not candidates or not settings.reranker_model:
        return select_passages(candidates, settings)
    scores = rerank_scores(question, [passage for passage, _ in candidates], settings)
    return select_passages(candidates, settings, scores)


def make_retriever(settings: Settings) -> Retriever:
    return lambda query: retrieve(query, settings)


def passage_count(settings: Settings) -> int | None:
    """Points in the collection, or None if Qdrant can't be reached."""
    try:
        with _qdrant(settings) as client:
            response = client.get(f"/collections/{settings.qdrant_collection}")
            response.raise_for_status()
        return response.json()["result"]["points_count"]
    except (httpx.HTTPError, KeyError):
        return None
