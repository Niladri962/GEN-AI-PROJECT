import hashlib
import json
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import CrossEncoder
from qdrant_client.models import Distance, VectorParams

from subjectmate.documents import (
    deduplicate_chunks,
    discover_source_files,
    load_course_documents,
    merge_short_sections,
    split_page_documents,
)
from subjectmate.ranking import is_course_material, select_passages  # noqa: F401 (re-exported)
from subjectmate.settings import Settings

UPSERT_BATCH_SIZE = 256
# Bump when chunk preparation changes so existing indexes are reported as stale.
INDEX_FORMAT = 2
SUBJECT_KEY = "metadata.subject"


def create_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    query_kwargs: dict[str, Any] = {"normalize_embeddings": True}
    if settings.embedding_query_prompt:
        query_kwargs["prompt"] = settings.embedding_query_prompt
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs=query_kwargs,
    )


def create_client(settings: Settings) -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=120,
    )


# SHA-256 per (path, size, modification time); a file is re-hashed only after it changes.
_DIGESTS: dict[tuple[str, int, int], str] = {}


def _file_digest(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _DIGESTS:
        digest = hashlib.sha256()
        with path.open("rb") as source_file:
            for block in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(block)
        _DIGESTS[key] = digest.hexdigest()
    return _DIGESTS[key]


def corpus_manifest(settings: Settings) -> dict[str, Any]:
    files = [
        {"filename": display_path, "sha256": _file_digest(path)}
        for path, display_path in discover_source_files([settings.pdf_dir, settings.material_dir])
    ]
    return {
        "files": files,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "min_section_chars": settings.min_section_chars,
        "index_format": INDEX_FORMAT,
        "qdrant_collection": settings.qdrant_collection,
    }


def added_files(settings: Settings, expected: dict[str, Any]) -> list[str] | None:
    """Display paths of files added since the stored index was built.

    Returns None when the index can't simply be extended: nothing stored yet, settings
    changed, files removed or modified, or Qdrant does not hold what the manifest says.
    """
    stored = read_index_manifest(settings.manifest_path)
    if not stored or any(
        stored.get(key) != value for key, value in expected.items() if key != "files"
    ):
        return None
    stored_files = {(item["filename"], item["sha256"]) for item in stored["files"]}
    current_files = {(item["filename"], item["sha256"]) for item in expected["files"]}
    if not stored_files <= current_files:
        return None
    if collection_point_count(settings) != stored.get("chunk_count"):
        return None
    return sorted(filename for filename, _ in current_files - stored_files)


def build_index(
    settings: Settings,
    progress: Callable[[str], None] | None = None,
    full: bool = False,
) -> dict[str, Any]:
    """Embed the course files into Qdrant.

    Newly added files are embedded into the existing collection; anything else (first
    build, changed settings, removed or modified files, or `full=True`) rebuilds it.
    """
    report = progress or (lambda message: None)
    settings.validate()
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest = corpus_manifest(settings)
    if not manifest["files"]:
        raise FileNotFoundError(
            f"Add course files or ZIP archives to {settings.material_dir} or {settings.pdf_dir}"
        )

    new_files = None if full else added_files(settings, manifest)
    if new_files is not None and not new_files:
        report("Index is already up to date")
        return read_index_manifest(settings.manifest_path) or manifest
    incremental = new_files is not None
    stored = read_index_manifest(settings.manifest_path) if incremental else None

    if incremental:
        report(f"Adding {len(new_files)} new file(s): {', '.join(new_files)}")
    else:
        report(f"Extracting text from {len(manifest['files'])} files/archives...")
    documents = merge_short_sections(
        load_course_documents(
            [settings.pdf_dir, settings.material_dir],
            only=set(new_files) if incremental else None,
        ),
        settings.min_section_chars,
    )
    chunks = deduplicate_chunks(
        split_page_documents(documents, settings.chunk_size, settings.chunk_overlap)
    )
    report(f"Extracted {len(documents)} sections into {len(chunks)} unique chunks")

    embeddings = create_embeddings(settings)
    client = create_client(settings)
    # Remove the manifest first so an interrupted build is never reported as current.
    settings.manifest_path.unlink(missing_ok=True)
    if not incremental:
        dimension = len(embeddings.embed_query("dimension probe"))
        if client.collection_exists(settings.qdrant_collection):
            client.delete_collection(settings.qdrant_collection)
        client.create_collection(
            settings.qdrant_collection,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )
    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[start : start + UPSERT_BATCH_SIZE]
        vectorstore.add_texts(
            [chunk.page_content for chunk in batch],
            [chunk.metadata for chunk in batch],
            ids=[str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.metadata["chunk_id"])) for chunk in batch],
            batch_size=UPSERT_BATCH_SIZE,
        )
        report(f"Embedded {min(start + UPSERT_BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")
    # Keyword index so subject-filtered searches stay fast; creating it again is harmless.
    client.create_payload_index(
        settings.qdrant_collection, SUBJECT_KEY, models.PayloadSchemaType.KEYWORD
    )

    previous = stored or {}
    manifest["document_count"] = previous.get("document_count", 0) + len(documents)
    manifest["page_count"] = previous.get("page_count", 0) + sum(
        "page" in document.metadata for document in documents
    )
    # Count what Qdrant actually holds: re-added identical chunks share an id and overwrite.
    manifest["chunk_count"] = client.count(settings.qdrant_collection, exact=True).count
    manifest["source_count"] = len(manifest["files"])
    settings.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    settings.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_index_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def collection_point_count(settings: Settings) -> int | None:
    """Return the number of stored vectors, or None if Qdrant or the collection is unavailable."""
    try:
        client = create_client(settings)
        if not client.collection_exists(settings.qdrant_collection):
            return None
        return client.count(settings.qdrant_collection, exact=True).count
    except Exception:
        return None


def index_is_current(settings: Settings) -> bool:
    stored = read_index_manifest(settings.manifest_path)
    expected = corpus_manifest(settings)
    if not (
        stored
        and expected["files"]
        and all(stored.get(key) == value for key, value in expected.items())
    ):
        return False
    return collection_point_count(settings) == stored.get("chunk_count")


def load_index(settings: Settings) -> QdrantVectorStore:
    if not index_is_current(settings):
        raise ValueError("The course index is missing or stale. Build it from the app first.")
    return QdrantVectorStore(
        client=create_client(settings),
        collection_name=settings.qdrant_collection,
        embedding=create_embeddings(settings),
    )


def retrieve(
    question: str,
    vectorstore: QdrantVectorStore,
    settings: Settings,
    subjects: list[str] | None = None,
) -> list[tuple[Document, float]]:
    """Return up to RETRIEVAL_K relevant, mutually diverse passages, best first.

    Vector search (cosine similarity, candidates below MIN_RELEVANCE_SCORE excluded, MMR
    for diversity) finds candidates. With a RERANKER_MODEL, a cross-encoder re-scores
    RERANK_CANDIDATES of them; nothing is returned unless the best re-ranker score reaches
    MIN_RERANK_SCORE (the question is about the course), and COURSE_BOOST favours lecture
    material when ordering. Without one, MIN_TOP_SCORE gates on cosine similarity.
    Returned scores are re-ranker probabilities or cosine similarities respectively.
    `subjects` limits the search to chunks whose subject metadata is one of these values.
    """
    search_filter = (
        models.Filter(
            must=[models.FieldCondition(key=SUBJECT_KEY, match=models.MatchAny(any=subjects))]
        )
        if subjects
        else None
    )
    candidate_count = settings.rerank_candidates if settings.reranker_model else settings.retrieval_k
    results = vectorstore.max_marginal_relevance_search_with_score_by_vector(
        vectorstore.embeddings.embed_query(question),
        k=candidate_count,
        fetch_k=max(settings.retrieval_fetch_k, 2 * candidate_count),
        lambda_mult=settings.mmr_diversity,
        score_threshold=settings.min_relevance_score,
        filter=search_filter,
    )
    if not results or not settings.reranker_model:
        return select_passages(results, settings)
    scores = load_reranker(settings.reranker_model).predict(
        [(question, document.page_content) for document, _ in results]
    )
    return select_passages(results, settings, scores)


@lru_cache(maxsize=2)
def load_reranker(model_name: str) -> CrossEncoder:
    # Single-label cross-encoders apply a sigmoid, so scores are probabilities in [0, 1].
    return CrossEncoder(model_name, max_length=512)
