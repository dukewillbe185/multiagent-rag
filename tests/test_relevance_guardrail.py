from types import SimpleNamespace

import pytest

from src.agents import base_agent, rag_agent
from src.agents.base_agent import BaseAgent
from src.agents.rag_agent import (
    AnswerGeneratorAgent,
    GuardrailAgent,
    IntentIdentifierAgent,
    MultiAgentRAG,
    SupervisorRetrievalAgent,
)
from src.retrieval.retriever import AzureSearchRetriever


def make_state(**overrides):
    state = {
        "session_id": "test-session",
        "user_id": "",
        "current_question": (
            "In 2024 how much did TAL Group paid and over how many customers?"
        ),
        "previous_question": "",
        "previous_answer": "",
        "retrieved_chunks": [],
        "retrieved_metadata": [],
        "retrieval_error": "",
        "intent": "",
        "answer": "",
        "conversation_turn": 1,
        "guardrail_passed": False,
        "guardrail_reason": "",
    }
    state.update(overrides)
    return state


def test_guardrail_uses_supported_seeded_sampling(monkeypatch):
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(base_agent, "AzureChatOpenAI", fake_chat_openai)
    monkeypatch.setattr(
        base_agent,
        "get_config",
        lambda: SimpleNamespace(
            ai_foundry_gpt4_endpoint="https://example.openai.azure.com",
            ai_foundry_gpt4_deployment="gpt-test",
            ai_foundry_gpt4_key="test-key",
            ai_foundry_gpt4_api_version="2025-01-01-preview",
            llm_temperature=1.0,
        ),
    )

    GuardrailAgent(strictness="medium")

    assert captured["temperature"] == 1.0
    assert captured["model_kwargs"] == {"seed": 0}


def test_guardrail_prompt_uses_retrieved_evidence(monkeypatch):
    passage = (
        "In the year to 31 March 2024, the TAL Group paid over $4.2 billion "
        "in claims to over 50,128 customers and their families."
    )

    class RecordingLLM:
        prompt = ""

        def invoke(self, messages):
            self.prompt = messages[-1].content
            return SimpleNamespace(
                content=(
                    '{"decision":"relevant","reason":'
                    '"The retrieved passage directly answers the question."}'
                )
            )

    llm = RecordingLLM()
    agent_log = SimpleNamespace(
        log_action=lambda *args, **kwargs: None,
        log_decision=lambda *args, **kwargs: None,
        log_complete=lambda **kwargs: setattr(
            agent_log,
            "next_agent",
            kwargs["next_agent"],
        ),
    )

    class LogContext:
        def __enter__(self):
            return agent_log

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(BaseAgent, "_initialize_llm", lambda self: llm)
    monkeypatch.setattr(
        rag_agent,
        "log_agent_execution",
        lambda *args, **kwargs: LogContext(),
    )
    agent = GuardrailAgent(strictness="medium")

    result = agent.execute(
        make_state(
            retrieved_chunks=[passage],
            retrieved_metadata=[{"source_file": "doc.pdf", "chunk_index": 12}],
        )
    )

    assert result["guardrail_passed"] is True
    assert passage in llm.prompt
    assert "doc.pdf" in llm.prompt
    assert "medium" in llm.prompt
    assert result["current_question"] in llm.prompt
    assert agent_log.next_agent == "IntentIdentifierAgent"


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '{"decision": null, "reason": "Missing decision"}',
        '{"decision": "relevant", "reason": null}',
        '{"decision": "unknown", "reason": "Unexpected decision"}',
    ],
)
def test_guardrail_rejects_schema_invalid_json(monkeypatch, payload):
    class InvalidSchemaLLM:
        def invoke(self, messages):
            return SimpleNamespace(content=payload)

    agent_log = SimpleNamespace(
        log_action=lambda *args, **kwargs: None,
        log_decision=lambda *args, **kwargs: None,
        log_complete=lambda **kwargs: None,
    )

    class LogContext:
        def __enter__(self):
            return agent_log

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(
        BaseAgent,
        "_initialize_llm",
        lambda self: InvalidSchemaLLM(),
    )
    monkeypatch.setattr(
        rag_agent,
        "log_agent_execution",
        lambda *args, **kwargs: LogContext(),
    )

    result = GuardrailAgent().execute(
        make_state(retrieved_chunks=["Retrieved evidence"])
    )

    assert result["guardrail_passed"] is False
    assert result["guardrail_reason"] == "Invalid guardrail response schema."


@pytest.mark.parametrize(
    "payload",
    [
        "not relevant",
        "The question is not relevant to the documents",
    ],
)
def test_guardrail_rejects_non_json_negative_phrases(monkeypatch, payload):
    agent = GuardrailAgent.__new__(GuardrailAgent)
    agent.agent_name = "Guardrail"

    decision, reason = agent._parse_decision(payload)

    assert decision == "irrelevant"
    assert reason == "Failed to parse guardrail response"


def test_guardrail_fails_open_when_retrieval_failed(monkeypatch):
    llm_called = False

    class RecordingLLM:
        def invoke(self, messages):
            nonlocal llm_called
            llm_called = True
            return SimpleNamespace(
                content='{"decision":"irrelevant","reason":"No evidence"}'
            )

    agent_log = SimpleNamespace(
        log_action=lambda *args, **kwargs: None,
        log_decision=lambda *args, **kwargs: None,
        log_complete=lambda **kwargs: None,
    )

    class LogContext:
        def __enter__(self):
            return agent_log

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(
        BaseAgent,
        "_initialize_llm",
        lambda self: RecordingLLM(),
    )
    monkeypatch.setattr(
        rag_agent,
        "log_agent_execution",
        lambda *args, **kwargs: LogContext(),
    )

    result = GuardrailAgent().execute(
        make_state(retrieval_error="Azure Search unavailable")
    )

    assert llm_called is False
    assert result["guardrail_passed"] is True
    assert result["guardrail_reason"] == (
        "Relevance check skipped because document retrieval failed."
    )


def test_guardrail_invocation_failure_remains_fail_open(monkeypatch):
    class FailingLLM:
        def invoke(self, messages):
            raise RuntimeError("model unavailable")

    agent_log = SimpleNamespace(
        log_action=lambda *args, **kwargs: None,
        log_decision=lambda *args, **kwargs: None,
        log_complete=lambda **kwargs: None,
    )

    class LogContext:
        def __enter__(self):
            return agent_log

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(
        BaseAgent,
        "_initialize_llm",
        lambda self: FailingLLM(),
    )
    monkeypatch.setattr(
        rag_agent,
        "log_agent_execution",
        lambda *args, **kwargs: LogContext(),
    )

    result = GuardrailAgent().execute(
        make_state(retrieved_chunks=["Retrieved evidence"])
    )

    assert result["guardrail_passed"] is True
    assert result["guardrail_reason"] == (
        "Relevance classifier unavailable; check skipped (fail-open)."
    )


def test_workflow_retrieves_before_guardrail_and_stops_on_rejection(
    monkeypatch,
):
    calls = []

    def fake_base_init(
        self,
        agent_name,
        system_prompt=None,
        temperature=None,
        seed=None,
    ):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.seed = seed
        self.llm = object()

    def retrieve(self, state):
        calls.append("retrieval")
        state["retrieved_chunks"] = ["Unrelated indexed content"]
        state["retrieved_metadata"] = [{"source_file": "doc.pdf"}]
        return state

    def reject(self, state):
        calls.append("guardrail")
        assert state["retrieved_chunks"] == ["Unrelated indexed content"]
        state["guardrail_passed"] = False
        state["guardrail_reason"] = "Retrieved evidence does not address the question."
        return state

    def identify(self, state):
        calls.append("intent")
        return state

    def answer(self, state):
        calls.append("answer")
        return state

    monkeypatch.setattr(BaseAgent, "__init__", fake_base_init)
    monkeypatch.setattr(
        AzureSearchRetriever,
        "__init__",
        lambda self, **kwargs: None,
    )
    monkeypatch.setattr(SupervisorRetrievalAgent, "execute", retrieve)
    monkeypatch.setattr(GuardrailAgent, "execute", reject)
    monkeypatch.setattr(IntentIdentifierAgent, "execute", identify)
    monkeypatch.setattr(AnswerGeneratorAgent, "execute", answer)

    result = MultiAgentRAG().graph.invoke(make_state())

    assert calls == ["retrieval", "guardrail"]
    assert result["guardrail_passed"] is False


@pytest.mark.parametrize(
    ("guardrail_enabled", "expected_next_agent"),
    [
        (True, "GuardrailAgent"),
        (False, "IntentIdentifierAgent"),
    ],
)
def test_retrieval_agent_logs_actual_handoff(
    monkeypatch,
    guardrail_enabled,
    expected_next_agent,
):
    agent_log = SimpleNamespace(
        log_action=lambda *args, **kwargs: None,
        log_complete=lambda **kwargs: setattr(
            agent_log,
            "next_agent",
            kwargs["next_agent"],
        ),
    )

    class LogContext:
        def __enter__(self):
            return agent_log

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(BaseAgent, "_initialize_llm", lambda self: object())
    monkeypatch.setattr(
        AzureSearchRetriever,
        "__init__",
        lambda self, **kwargs: None,
    )
    monkeypatch.setattr(
        AzureSearchRetriever,
        "retrieve",
        lambda self, question: [
            {
                "id": "chunk-1",
                "content": "Retrieved evidence",
                "source_file": "doc.pdf",
                "chunk_index": 1,
                "score": 0.03,
            }
        ],
    )
    monkeypatch.setattr(
        rag_agent,
        "log_agent_execution",
        lambda *args, **kwargs: LogContext(),
    )

    SupervisorRetrievalAgent(
        top_k=5,
        guardrail_enabled=guardrail_enabled,
    ).execute(make_state())

    assert agent_log.next_agent == expected_next_agent


def test_retrieval_agent_records_failure(monkeypatch):
    agent_log = SimpleNamespace(
        log_action=lambda *args, **kwargs: None,
        log_complete=lambda **kwargs: None,
    )

    class LogContext:
        def __enter__(self):
            return agent_log

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(BaseAgent, "_initialize_llm", lambda self: object())
    monkeypatch.setattr(
        AzureSearchRetriever,
        "__init__",
        lambda self, **kwargs: None,
    )
    monkeypatch.setattr(
        AzureSearchRetriever,
        "retrieve",
        lambda self, question: (_ for _ in ()).throw(
            RuntimeError("Azure Search unavailable")
        ),
    )
    monkeypatch.setattr(
        rag_agent,
        "log_agent_execution",
        lambda *args, **kwargs: LogContext(),
    )

    result = SupervisorRetrievalAgent(top_k=5).execute(make_state())

    assert result["retrieved_chunks"] == []
    assert result["retrieved_metadata"] == []
    assert result["retrieval_error"] == "Azure Search unavailable"


def test_disabled_guardrail_skips_classifier(monkeypatch):
    calls = []

    def fake_base_init(self, agent_name, **kwargs):
        self.agent_name = agent_name
        self.system_prompt = kwargs.get("system_prompt")
        self.llm = object()

    def retrieve(self, state):
        calls.append("retrieval")
        state["retrieved_chunks"] = ["Retrieved evidence"]
        return state

    def guardrail(self, state):
        calls.append("guardrail")
        return state

    def identify(self, state):
        calls.append("intent")
        state["intent"] = "factual"
        return state

    def answer(self, state):
        calls.append("answer")
        state["answer"] = "Answer"
        return state

    monkeypatch.setattr(BaseAgent, "__init__", fake_base_init)
    monkeypatch.setattr(
        AzureSearchRetriever,
        "__init__",
        lambda self, **kwargs: None,
    )
    monkeypatch.setattr(SupervisorRetrievalAgent, "execute", retrieve)
    monkeypatch.setattr(GuardrailAgent, "execute", guardrail)
    monkeypatch.setattr(IntentIdentifierAgent, "execute", identify)
    monkeypatch.setattr(AnswerGeneratorAgent, "execute", answer)

    result = MultiAgentRAG(guardrail_enabled=False).graph.invoke(make_state())

    assert calls == ["retrieval", "intent", "answer"]
    assert result["answer"] == "Answer"


def test_cli_wrapper_respects_disabled_guardrail():
    rag = MultiAgentRAG.__new__(MultiAgentRAG)
    rag.guardrail_enabled = False
    rag.graph = SimpleNamespace(
        invoke=lambda state: {
            **state,
            "answer": "Answer",
            "intent": "factual",
            "retrieved_chunks": ["Retrieved evidence"],
            "retrieved_metadata": [{"source_file": "doc.pdf"}],
        }
    )

    response = rag.query("Question")

    assert response["guardrail_passed"] is True
    assert response["guardrail_reason"] == "Relevance guardrail disabled."
    assert response["answer"] == "Answer"
