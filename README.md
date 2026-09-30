# SubjectMate

SubjectMate is a local-corpus RAG app for course-specific questions. It retrieves passages from course PDFs, Word documents, PowerPoint presentations, and Excel workbooks, including files inside ZIP archives, then generates grounded answers through an OpenAI-compatible chat API. Students cannot upload files, and the app does not use web search.

## Requirements

- Python 3.10+
- An API key for the course LLM endpoint; the app supports OpenAI-compatible chat-completions APIs
- Course PDFs with extractable text; scanned PDFs need OCR before indexing
- Course files in PDF, DOCX, PPTX, or XLSX format
- Docker, for the Qdrant vector database

## Setup

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`, `OPENAI_MODEL`, and, if required by your provider, `OPENAI_BASE_URL`. Put course files or ZIP archives in `all data/`; nested ZIP archives are also read. Loose PDFs in `data/course_pdfs/` remain supported.

## Start Qdrant

```powershell
docker run -d --name subjectmate-qdrant -p 127.0.0.1:6333:6333 -v "${PWD}\storage\qdrant:/qdrant/storage" qdrant/qdrant:latest
```

If the container already exists, start it with `docker start subjectmate-qdrant`. Vectors persist in `storage/qdrant/`. Set `QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_COLLECTION` in `.env` to use a different Qdrant server or collection.

## Run

```powershell
streamlit run app.py
```

Build the index with **Build / rebuild index** in the sidebar, or run `python -m subjectmate.ingest`. Indexing downloads/loads the configured sentence-transformers model and may take a few minutes the first time. Chunks and their embeddings are stored in the Qdrant collection `subjectmate`. When files have only been added since the last build, just those files are embedded into the existing collection; if files were removed or changed, or indexing settings changed, the collection is rebuilt. Force a full rebuild with `python -m subjectmate.ingest --full`.

Loose files in a subfolder of `all data/` take the folder name as their subject, e.g. `all data/python programming/Think Python 2e - Downey.pdf`. The `all data/<subject>/` folders hold freely licensed or author-hosted reference textbooks (Think Python, Think Stats, Jeff Erickson's Algorithms, Open Data Structures, Foundations of Databases, Jurafsky & Martin's Speech and Language Processing draft, Linear Algebra Done Right 4e, Mathematics for Machine Learning, Grinstead & Snell's Introduction to Probability, ISLP, and A Course in Machine Learning). Several are licensed for non-commercial use only; check each book's license before serving the app publicly. `storage/index_manifest.json` records which files were indexed, so the app reports when source files or indexing settings change and a rebuild is needed. Embeddings are generated locally; the API key is used only when answering questions.

## Evaluation

Populate `data/evaluation/questions.json` with cases following this shape:

```json
[
  {
    "id": "q001",
    "question": "What is ...?",
    "expected_outcome": "answer",
    "reference_answer": "...",
    "acceptable_points": ["..."],
    "supporting_sources": [
      {"filename": "Lecture_1.pdf", "pages": [2, 3]}
    ]
  },
  {
    "id": "q002",
    "question": "A question not answered by the PDFs",
    "expected_outcome": "abstain",
    "reference_answer": "",
    "acceptable_points": [],
    "supporting_sources": []
  }
]
```

Run retrieval evaluation after building the index:

```powershell
python -m subjectmate.evaluate --k 6
```

This reports Recall@k for the gold supporting pages and writes retrieved passages to `storage/evaluation/retrieval_report.json` for inspection. Answer correctness, citation support, and abstention should be reviewed against the reference fields; the benchmark is intentionally not scored by an LLM judge alone.

## Design notes

- Extraction preserves PDF page, PowerPoint slide, and Excel sheet locations in source citations.
- Pages and slides shorter than `MIN_SECTION_CHARS` (default 200) are merged into the next page or slide of the same file, so title slides and bare examples are indexed with their context; citations then show a range such as "slides 40-41".
- Chunks start at 850 characters with 120 characters of overlap; tune against the evaluation set. Identical chunks (for example, the same PDF included twice) and chunks with almost no words are dropped before embedding.
- Embeddings use `BAAI/bge-small-en-v1.5`, with its retrieval instruction (`EMBEDDING_QUERY_PROMPT`) prepended to questions only. They are normalized and stored in Qdrant with cosine distance.
- Retrieval considers the `RETRIEVAL_FETCH_K` most similar chunks at or above `MIN_RELEVANCE_SCORE` (cosine similarity), then picks `RETRIEVAL_K` of them with maximal marginal relevance (`MMR_DIVERSITY`) so near-duplicate passages do not crowd out other evidence. Calibrate the threshold against both answerable and unanswerable benchmark items.
- To answer with a local model, point the OpenAI settings at Ollama (`OPENAI_BASE_URL=http://localhost:11434/v1`, any non-empty `OPENAI_API_KEY`). For reasoning models such as `qwen3:8b`, set `LLM_REASONING_EFFORT=none` to skip their slow thinking phase; `<think>` blocks are stripped from answers either way.
- For a provider that is not OpenAI-compatible, replace the OpenAI SDK call in `subjectmate/rag.py` with that provider's API client.

## Tests

```powershell
.\test.ps1 -q
```

The test script always uses `.venv`, even if the current PowerShell session has not activated it. To run pytest directly, activate the environment first with `.\.venv\Scripts\Activate.ps1`, or invoke `.\.venv\Scripts\python.exe -m pytest -q`.
