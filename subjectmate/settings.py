from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    pdf_dir: Path = PROJECT_ROOT / "data" / "course_pdfs"
    material_dir: Path = PROJECT_ROOT / "all data"
    manifest_path: Path = PROJECT_ROOT / "storage" / "index_manifest.json"
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "subjectmate")
    evaluation_file: Path = PROJECT_ROOT / "data" / "evaluation" / "questions.json"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    # Instruction prepended to questions (not passages); BGE v1.5 retrieves better with it.
    embedding_query_prompt: str = os.getenv(
        "EMBEDDING_QUERY_PROMPT", "Represent this sentence for searching relevant passages: "
    )
    llm_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    # Optional; "none" turns off thinking for reasoning models served by Ollama (qwen3).
    llm_reasoning_effort: str | None = os.getenv("LLM_REASONING_EFFORT") or None
    # Optional second OpenAI-compatible endpoint (e.g. local Ollama), used when the primary
    # one is unreachable, rejects the key, or has no quota left.
    fallback_base_url: str | None = os.getenv("FALLBACK_OPENAI_BASE_URL") or None
    fallback_model: str | None = os.getenv("FALLBACK_OPENAI_MODEL") or None
    fallback_api_key: str = os.getenv("FALLBACK_OPENAI_API_KEY", "ollama")
    fallback_reasoning_effort: str | None = os.getenv("FALLBACK_LLM_REASONING_EFFORT") or None
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "850"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    min_section_chars: int = int(os.getenv("MIN_SECTION_CHARS", "200"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "6"))
    retrieval_fetch_k: int = int(os.getenv("RETRIEVAL_FETCH_K", "24"))
    mmr_diversity: float = float(os.getenv("MMR_DIVERSITY", "0.3"))
    # Passage floor: supporting passages below this cosine similarity are never used.
    min_relevance_score: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.5"))
    # Question gate: answer only if the best passage reaches this similarity. With
    # bge-small, off-topic questions peaked at 0.64 and course questions started at 0.69.
    min_top_score: float = float(os.getenv("MIN_TOP_SCORE", "0.66"))
    # Cross-encoder that re-scores candidates by reading question and passage together.
    # Empty disables re-ranking (cosine similarity and MIN_TOP_SCORE are used instead).
    reranker_model: str = os.getenv("RERANKER_MODEL", "")
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "20"))
    # With a re-ranker: answer only if the best passage's re-ranker score reaches this,
    # and drop passages scoring below the floor.
    min_rerank_score: float = float(os.getenv("MIN_RERANK_SCORE", "0.5"))
    rerank_passage_floor: float = float(os.getenv("RERANK_PASSAGE_FLOOR", "0.05"))
    # Added to lecture material's re-ranker score when ordering passages (not for the gate).
    course_boost: float = float(os.getenv("COURSE_BOOST", "0"))

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_api_key or (self.fallback_base_url and self.fallback_model))

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be a positive integer")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE")
        if self.retrieval_k <= 0:
            raise ValueError("RETRIEVAL_K must be a positive integer")
        if self.retrieval_fetch_k < self.retrieval_k:
            raise ValueError("RETRIEVAL_FETCH_K must be at least RETRIEVAL_K")
        if not 0 <= self.mmr_diversity <= 1:
            raise ValueError("MMR_DIVERSITY must be between 0 and 1")
        if self.min_section_chars < 0:
            raise ValueError("MIN_SECTION_CHARS must be non-negative")
        if not 0 <= self.min_relevance_score <= 1:
            raise ValueError("MIN_RELEVANCE_SCORE must be between 0 and 1")
        if not 0 <= self.min_top_score <= 1:
            raise ValueError("MIN_TOP_SCORE must be between 0 and 1")
        if self.rerank_candidates < self.retrieval_k:
            raise ValueError("RERANK_CANDIDATES must be at least RETRIEVAL_K")
        for name in ("min_rerank_score", "rerank_passage_floor", "course_boost"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name.upper()} must be between 0 and 1")
