import json
from types import SimpleNamespace

import pytest

from subjectmate import vectorstore
from subjectmate.settings import Settings
from subjectmate.vectorstore import corpus_manifest, index_is_current, retrieve


def test_only_added_files_allow_incremental_indexing(tmp_path, monkeypatch):
    material_dir = tmp_path / "materials"
    (material_dir / "python").mkdir(parents=True)
    manifest_path = tmp_path / "index_manifest.json"
    lecture = material_dir / "python" / "Lecture.pdf"
    lecture.write_bytes(b"lecture v1")
    settings = Settings(
        pdf_dir=tmp_path / "pdfs", material_dir=material_dir, manifest_path=manifest_path
    )
    manifest_path.write_text(
        json.dumps(corpus_manifest(settings) | {"chunk_count": 5}), encoding="utf-8"
    )
    monkeypatch.setattr(vectorstore, "collection_point_count", lambda settings: 5)

    assert vectorstore.added_files(settings, corpus_manifest(settings)) == []

    (material_dir / "python" / "Book.pdf").write_bytes(b"book")
    assert vectorstore.added_files(settings, corpus_manifest(settings)) == [
        "materials/python/Book.pdf"
    ]

    lecture.write_bytes(b"lecture v2")  # a changed file needs a full rebuild
    assert vectorstore.added_files(settings, corpus_manifest(settings)) is None


class FakeStore:
    def __init__(self, results):
        self.results = results
        self.embeddings = SimpleNamespace(embed_query=lambda text: [0.0])

    def max_marginal_relevance_search_with_score_by_vector(self, embedding, **kwargs):
        self.filter = kwargs["filter"]
        return [item for item in self.results if item[1] >= kwargs["score_threshold"]]


def test_retrieve_filters_by_subject_values():
    store = FakeStore([("doc", 0.9)])

    retrieve("question", store, Settings(), subjects=["machine learning", "ML 2 - Supervised"])

    [condition] = store.filter.must
    assert condition.key == "metadata.subject"
    assert condition.match.any == ["machine learning", "ML 2 - Supervised"]
    retrieve("question", store, Settings())
    assert store.filter is None


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([0.55, 0.64], []),  # off-topic: best passage below the question gate
        ([0.52, 0.71, 0.45], [0.71, 0.52]),  # on-topic: best first, floor applied
    ],
)
def test_retrieve_gates_off_topic_questions(scores, expected):
    store = FakeStore([(f"doc{index}", score) for index, score in enumerate(scores)])
    settings = Settings(min_relevance_score=0.5, min_top_score=0.66)

    assert [score for _, score in retrieve("question", store, settings)] == expected


def test_index_manifest_allows_build_metadata_and_detects_changed_pdf(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "pdfs"
    material_dir = tmp_path / "materials"
    manifest_path = tmp_path / "storage" / "index_manifest.json"
    pdf_dir.mkdir()
    material_dir.mkdir()
    manifest_path.parent.mkdir()
    source_pdf = pdf_dir / "Lecture.pdf"
    source_pdf.write_bytes(b"course material v1")
    point_count = {"value": 1}
    monkeypatch.setattr(vectorstore, "collection_point_count", lambda settings: point_count["value"])

    settings = Settings(pdf_dir=pdf_dir, material_dir=material_dir, manifest_path=manifest_path)
    stored = corpus_manifest(settings) | {"page_count": 1, "chunk_count": 1}
    manifest_path.write_text(json.dumps(stored), encoding="utf-8")

    assert index_is_current(settings)

    point_count["value"] = None
    assert not index_is_current(settings)

    point_count["value"] = 1
    source_pdf.write_bytes(b"course material v2")
    assert not index_is_current(settings)
