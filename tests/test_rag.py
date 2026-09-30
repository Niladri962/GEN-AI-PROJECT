from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from subjectmate import rag
from subjectmate.rag import Turn, answer_question, format_location
from subjectmate.settings import Settings


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"location_type": "page", "location": "12"}, "p. 12"),
        ({"location_type": "page", "location": "4-6"}, "pp. 4-6"),
        ({"location_type": "slide", "location": "3-4"}, "slides 3-4"),
        ({"location_type": "sheet", "location": "Results"}, "sheet Results"),
        ({"location_type": "document", "location": "document"}, "whole document"),
    ],
)
def test_format_location(metadata, expected):
    assert format_location(metadata) == expected


class FakeClient:
    """Records chat requests and replies with queued answers."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        message = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_follow_up_is_rewritten_before_retrieval_and_history_is_sent(monkeypatch):
    client = FakeClient(
        ["<think>resolve it</think>What are the limitations of PCA?", "PCA is linear [S1]."]
    )
    searched = []
    passage = Document(
        page_content="PCA only captures linear structure.",
        metadata={"source": "pca.pptx", "location_type": "slide", "location": "9"},
    )
    monkeypatch.setattr(rag, "OpenAI", lambda **kwargs: client)
    history = [Turn("What is PCA?", "PCA reduces dimensionality [S1].")]

    result = answer_question(
        "What are its limitations?",
        lambda query: searched.append(query) or [(passage, 0.8)],
        Settings(openai_api_key="test"),
        history,
    )

    assert searched == ["What are the limitations of PCA?"]
    assert result.search_query == "What are the limitations of PCA?"
    assert result.answer == "PCA is linear [S1]."
    assert result.sources[0]["location"] == "slide 9"
    answer_messages = client.requests[1]["messages"]
    assert [message["role"] for message in answer_messages] == [
        "system", "user", "assistant", "user"
    ]
    assert "[S1]" not in answer_messages[2]["content"]
    assert "What are its limitations?" in answer_messages[3]["content"]


def test_falls_back_when_primary_has_no_quota(monkeypatch):
    import httpx
    import openai

    class NoQuotaClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **request):
            response = httpx.Response(429, request=httpx.Request("POST", "https://api.test"))
            raise openai.RateLimitError("no credits", response=response, body=None)

    fallback = FakeClient(["PCA reduces dimensions [S1]."])
    clients = iter([NoQuotaClient(), fallback])
    monkeypatch.setattr(rag, "OpenAI", lambda **kwargs: next(clients))
    passage = Document(page_content="PCA", metadata={"location_type": "page", "location": "3"})
    settings = Settings(
        openai_api_key="test",
        llm_model="gpt-4o-mini",
        fallback_base_url="http://localhost:11434/v1",
        fallback_model="llama3.2:3b",
    )

    result = answer_question("What is PCA?", lambda query: [(passage, 0.9)], settings)

    assert result.answer == "PCA reduces dimensions [S1]."
    assert result.model_used == "llama3.2:3b (fallback)"
    assert fallback.requests[0]["model"] == "llama3.2:3b"


def test_uncited_answer_is_asked_for_citations_once(monkeypatch):
    client = FakeClient(["Dijkstra repeatedly settles the closest vertex.", "It settles the closest vertex [S1]."])
    monkeypatch.setattr(rag, "OpenAI", lambda **kwargs: client)
    passage = Document(page_content="Dijkstra...", metadata={"location_type": "page", "location": "3"})

    result = answer_question("How does Dijkstra work?", lambda query: [(passage, 0.9)], Settings(openai_api_key="test"))

    assert result.answer == "It settles the closest vertex [S1]."
    assert not result.abstained
    repair = client.requests[1]["messages"]
    assert repair[-2] == {"role": "assistant", "content": "Dijkstra repeatedly settles the closest vertex."}
    assert repair[-1]["content"] == rag.CITE_REQUEST


def test_answer_still_uncited_after_repair_is_withheld(monkeypatch):
    client = FakeClient(["An unsupported claim.", "Still no labels."])
    monkeypatch.setattr(rag, "OpenAI", lambda **kwargs: client)
    passage = Document(page_content="text", metadata={})

    result = answer_question("Question?", lambda query: [(passage, 0.9)], Settings(openai_api_key="test"))

    assert result.abstained
    assert result.answer == rag.ABSTENTION


def test_first_question_skips_rewriting(monkeypatch):
    client = FakeClient(["No evidence covers this."])
    monkeypatch.setattr(rag, "OpenAI", lambda **kwargs: client)

    result = answer_question("What is PCA?", lambda query: [], Settings(openai_api_key="test"))

    assert result.abstained
    assert result.search_query == "What is PCA?"
    assert client.requests == []
