from types import SimpleNamespace

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


def test_retrieval_agent_logs_guardrail_handoff(monkeypatch):
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

    SupervisorRetrievalAgent(top_k=5).execute(make_state())

    assert agent_log.next_agent == "GuardrailAgent"
