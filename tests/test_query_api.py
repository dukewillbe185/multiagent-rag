from types import SimpleNamespace

import pytest

from src.api import routes
from src.api.models import QueryRequest
from src.guardrails.content_safety import STATUS_DETECTED


class FakeConfig:
    def __init__(self, input_enabled=False, relevance_enabled=True):
        self.input_guardrail_enabled = input_enabled
        self.output_guardrail_enabled = False
        self.relevance_guardrail_enabled = relevance_enabled

    def get_optional_env(self, key, default):
        values = {
            "SESSION_TIMEOUT_MINUTES": "30",
            "GUARDRAIL_ENABLED": str(self.relevance_guardrail_enabled).lower(),
            "GUARDRAIL_STRICTNESS": "medium",
        }
        return values.get(key, default)


class FakeSessionManager:
    def get_session(self, session_id):
        return None

    def update_session(self, **kwargs):
        return None


def install_query_fakes(
    monkeypatch,
    graph_result,
    config=None,
    guardrail=None,
    completed_calls=None,
):
    config = config or FakeConfig()
    guardrail = guardrail or SimpleNamespace()

    class FakeRAG:
        def __init__(self, **kwargs):
            self.graph = SimpleNamespace(invoke=lambda state: graph_result)

    monkeypatch.setattr(routes, "get_config", lambda: config)
    monkeypatch.setattr(
        routes,
        "get_session_manager",
        lambda **kwargs: FakeSessionManager(),
    )
    monkeypatch.setattr(routes, "get_guardrail", lambda: guardrail)
    monkeypatch.setattr(routes, "MultiAgentRAG", FakeRAG)
    monkeypatch.setattr(
        routes,
        "track_event",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        routes,
        "track_metric",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        routes,
        "log_request_start",
        lambda *args, **kwargs: None,
    )
    if completed_calls is None:
        monkeypatch.setattr(
            routes,
            "log_request_complete",
            lambda *args, **kwargs: None,
        )
    else:
        monkeypatch.setattr(
            routes,
            "log_request_complete",
            lambda *args, **kwargs: completed_calls.append(kwargs),
        )


def make_request(question="test question"):
    return QueryRequest(
        question=question,
        session_id="test-session",
        top_k=5,
    )


@pytest.mark.asyncio
async def test_relevance_rejection_returns_guardrail_reason(monkeypatch):
    reason = "The retrieved chunks do not address weather."
    install_query_fakes(
        monkeypatch,
        {
            "guardrail_passed": False,
            "guardrail_reason": reason,
            "retrieved_chunks": [],
            "retrieved_metadata": [],
            "intent": "",
            "answer": "",
        },
    )

    response = await routes.query_documents.__wrapped__(
        make_request("What is the weather?")
    )

    assert response.guardrail_reason == reason


@pytest.mark.asyncio
async def test_success_returns_guardrail_reason(monkeypatch):
    reason = "The retrieved passage directly answers the TAL claims question."
    completed_calls = []
    install_query_fakes(
        monkeypatch,
        {
            "guardrail_passed": True,
            "guardrail_reason": reason,
            "retrieved_chunks": [
                "TAL paid over $4.2 billion to over 50,128 customers."
            ],
            "retrieved_metadata": [],
            "intent": "factual",
            "answer": ("TAL paid over $4.2 billion to over 50,128 customers."),
        },
        completed_calls=completed_calls,
    )

    response = await routes.query_documents.__wrapped__(make_request())

    assert response.guardrail_reason == reason
    assert completed_calls[0]["agents_executed"] == [
        "SupervisorRetrievalAgent",
        "GuardrailAgent",
        "IntentIdentifierAgent",
        "AnswerGeneratorAgent",
    ]


@pytest.mark.asyncio
async def test_input_safety_rejection_returns_trigger_reason(monkeypatch):
    safety_result = SimpleNamespace(
        detected=True,
        status=STATUS_DETECTED,
        triggered=["Violence=4 (>= 4)"],
        jailbreak=False,
        to_details=lambda: {"triggered": ["Violence=4 (>= 4)"]},
    )
    guardrail = SimpleNamespace(check_input=lambda question: safety_result)
    install_query_fakes(
        monkeypatch,
        graph_result={},
        config=FakeConfig(input_enabled=True),
        guardrail=guardrail,
    )

    response = await routes.query_documents.__wrapped__(make_request())

    assert "Violence=4 (>= 4)" in response.guardrail_reason


@pytest.mark.asyncio
async def test_disabled_relevance_guardrail_does_not_reject(monkeypatch):
    completed_calls = []
    install_query_fakes(
        monkeypatch,
        {
            "guardrail_passed": False,
            "guardrail_reason": "",
            "retrieved_chunks": ["Retrieved evidence"],
            "retrieved_metadata": [],
            "intent": "factual",
            "answer": "Answer generated without relevance filtering.",
        },
        config=FakeConfig(relevance_enabled=False),
        completed_calls=completed_calls,
    )

    response = await routes.query_documents.__wrapped__(make_request())

    assert response.success is True
    assert response.guardrail_passed is True
    assert response.guardrail_reason == "Relevance guardrail disabled."
    assert completed_calls[0]["agents_executed"] == [
        "SupervisorRetrievalAgent",
        "IntentIdentifierAgent",
        "AnswerGeneratorAgent",
    ]
