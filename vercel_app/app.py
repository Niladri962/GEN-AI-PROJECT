"""SubjectMate API and web page, deployed as a Vercel Python function.

Build first with `python scripts/build_vercel.py`, which copies the `subjectmate` modules
this file needs and bundles the embedding model into `models/`.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
if (HERE / "models").is_dir():
    # Use the bundled models; never download at request time.
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(HERE / "models"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from subjectmate.cloud import make_retriever, passage_count  # noqa: E402
from subjectmate.rag import Turn, answer_question  # noqa: E402
from subjectmate.settings import Settings  # noqa: E402

settings = Settings()
app = FastAPI(title="SubjectMate", docs_url=None, redoc_url=None)


class HistoryTurn(BaseModel):
    question: str = Field(max_length=2000)
    answer: str = Field(max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=20)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(HERE / "public" / "index.html")


@app.get("/api/health")
def health() -> dict:
    count = passage_count(settings)
    return {
        "ready": bool(count) and settings.llm_configured,
        "passages": count,
        "model": settings.llm_model if settings.openai_api_key else None,
    }


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    if not settings.llm_configured:
        raise HTTPException(503, "The answer model is not configured (set OPENAI_API_KEY).")
    history = [Turn(turn.question, turn.answer) for turn in request.history]
    try:
        result = answer_question(request.question, make_retriever(settings), settings, history)
    except Exception as exc:  # surface provider errors (quota, auth, network) to the page
        raise HTTPException(502, f"Unable to answer right now: {exc}") from exc
    return {
        "answer": result.answer,
        "abstained": result.abstained,
        "search_query": result.search_query,
        "model_used": result.model_used,
        "sources": [
            {key: source[key] for key in ("label", "filename", "location", "score")}
            for source in result.sources
        ],
    }
