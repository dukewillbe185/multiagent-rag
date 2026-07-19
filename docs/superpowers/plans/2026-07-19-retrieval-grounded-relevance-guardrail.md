# Retrieval-Grounded Relevance Guardrail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent valid knowledge-base questions from being rejected before search by grounding relevance classification in retrieved chunks, making the classifier deterministic, and exposing its reason through the API.

**Architecture:** Keep Azure AI Content Safety at the API boundary before the LangGraph workflow. Reorder the graph to retrieve first and then run the relevance guardrail against the retrieved evidence; only relevant requests proceed to intent detection and answer generation. Give `GuardrailAgent` a per-agent temperature override of `0.0` and add `guardrail_reason` to every `QueryResponse` path.

**Tech Stack:** Python 3.12, pytest 7, LangGraph 0.0.x, LangChain `AzureChatOpenAI`, FastAPI, Pydantic 2, Azure AI Search.

## Global Constraints

- Keep Azure AI Content Safety input checks before retrieval for API requests.
- Do not use a fixed threshold on Azure hybrid-search scores.
- Preserve `GUARDRAIL_ENABLED`, `GUARDRAIL_STRICTNESS`, existing rejection messages, and existing CLI response keys.
- Unit tests must make no Azure or Application Insights network calls.
- Do not modify or stage anything under `outputs/`.

---

### Task 1: Deterministic Guardrail LLM Configuration

**Files:**
- Create: `tests/test_relevance_guardrail.py`
- Modify: `src/agents/base_agent.py:24-60`
- Modify: `src/agents/rag_agent.py:61-77`

**Interfaces:**
- Consumes: `BaseAgent(agent_name: str, system_prompt: str | None)` and application `LLM_TEMPERATURE`.
- Produces: `BaseAgent(agent_name: str, system_prompt: str | None, temperature: float | None = None)`; `GuardrailAgent` uses `temperature=0.0`.

- [ ] **Step 1: Write the failing temperature-override test**

```python
from types import SimpleNamespace

from src.agents import base_agent
from src.agents.rag_agent import GuardrailAgent


def test_guardrail_uses_zero_temperature(monkeypatch):
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

    assert captured["temperature"] == 0.0
```

- [ ] **Step 2: Run the test and verify RED**

Run: `/Users/dukeisyourdaddy/Desktop/multiagent-rag/.venv/bin/python -m pytest tests/test_relevance_guardrail.py::test_guardrail_uses_zero_temperature -q`

Expected: FAIL because the current guardrail inherits `llm_temperature=1.0`.

- [ ] **Step 3: Add the minimal per-agent override**

```python
class BaseAgent:
    def __init__(self, agent_name: str, system_prompt: str = None, temperature: float = None):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.llm = self._initialize_llm()

    def _initialize_llm(self) -> AzureChatOpenAI:
        config = get_config()
        temperature = config.llm_temperature if self.temperature is None else self.temperature
        return AzureChatOpenAI(
            azure_endpoint=config.ai_foundry_gpt4_endpoint,
            azure_deployment=config.ai_foundry_gpt4_deployment,
            api_key=config.ai_foundry_gpt4_key,
            api_version=config.ai_foundry_gpt4_api_version,
            temperature=temperature,
        )
```

Pass `temperature=0.0` from `GuardrailAgent.__init__`; leave all other agents unchanged.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `/Users/dukeisyourdaddy/Desktop/multiagent-rag/.venv/bin/python -m pytest tests/test_relevance_guardrail.py::test_guardrail_uses_zero_temperature -q`

Expected: `1 passed`.

- [ ] **Step 5: Commit Task 1**

```bash
git add tests/test_relevance_guardrail.py src/agents/base_agent.py src/agents/rag_agent.py
git commit -m "Make relevance guardrail deterministic"
```

---

### Task 2: Retrieve Before Evidence-Based Relevance Classification

**Files:**
- Modify: `tests/test_relevance_guardrail.py`
- Modify: `src/agents/rag_agent.py:79-192`
- Modify: `src/agents/rag_agent.py:543-600`

**Interfaces:**
- Consumes: `RagState.retrieved_chunks`, `RagState.retrieved_metadata`, current/previous questions, and `GUARDRAIL_STRICTNESS`.
- Produces: `RagState.guardrail_passed` and `RagState.guardrail_reason` based on retrieved evidence; graph order `supervisor_retrieval -> guardrail -> intent_identifier -> answer_generator`.

- [ ] **Step 1: Add failing prompt-grounding and graph-order tests**

```python
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
        "current_question": "In 2024 how much did TAL Group paid and over how many customers?",
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
    monkeypatch.setattr(BaseAgent, "_initialize_llm", lambda self: llm)
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


def test_workflow_retrieves_before_guardrail_and_stops_on_rejection(monkeypatch):
    calls = []

    def fake_base_init(self, agent_name, system_prompt=None, temperature=None):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.temperature = temperature
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
    monkeypatch.setattr(AzureSearchRetriever, "__init__", lambda self, **kwargs: None)
    monkeypatch.setattr(SupervisorRetrievalAgent, "execute", retrieve)
    monkeypatch.setattr(GuardrailAgent, "execute", reject)
    monkeypatch.setattr(IntentIdentifierAgent, "execute", identify)
    monkeypatch.setattr(AnswerGeneratorAgent, "execute", answer)

    result = MultiAgentRAG().graph.invoke(make_state())

    assert calls == ["retrieval", "guardrail"]
    assert result["guardrail_passed"] is False
```

- [ ] **Step 2: Run both new tests and verify RED**

Run: `/Users/dukeisyourdaddy/Desktop/multiagent-rag/.venv/bin/python -m pytest tests/test_relevance_guardrail.py -q`

Expected: the grounding test fails because chunks are absent from the prompt, and the order test fails because the guardrail runs before retrieval.

- [ ] **Step 3: Ground the prompt in retrieved evidence**

In `GuardrailAgent.execute`, format each chunk with its source metadata, include it under `Retrieved Evidence`, and tell the classifier to decide whether that evidence materially supports the current question. Include explicit low/medium/high strictness criteria. If no chunks are available, reject with `No retrieved document evidence was available for relevance assessment.` without invoking the LLM.

- [ ] **Step 4: Reorder the LangGraph edges**

Set `supervisor_retrieval` as the entry point, add a direct edge from retrieval to guardrail, retain conditional routing after guardrail, and keep the accepted path through intent and answer generation.

- [ ] **Step 5: Run the relevance tests and verify GREEN**

Run: `/Users/dukeisyourdaddy/Desktop/multiagent-rag/.venv/bin/python -m pytest tests/test_relevance_guardrail.py -q`

Expected: all relevance tests pass and the rejected flow never calls intent or answer generation.

- [ ] **Step 6: Commit Task 2**

```bash
git add tests/test_relevance_guardrail.py src/agents/rag_agent.py
git commit -m "Ground relevance checks in retrieved evidence"
```

---

### Task 3: Expose Guardrail Reasons Through the API

**Files:**
- Create: `tests/test_query_api.py`
- Modify: `src/api/models.py:75-129`
- Modify: `src/api/routes.py:315-594`

**Interfaces:**
- Consumes: Azure Content Safety trigger details and `RagState.guardrail_reason`.
- Produces: required `QueryResponse.guardrail_reason: str` for input-safety rejection, relevance rejection, and successful query responses.

- [ ] **Step 1: Write failing API response tests**

```python
from types import SimpleNamespace

import pytest

from src.api import routes
from src.api.models import QueryRequest
from src.guardrails.content_safety import STATUS_DETECTED


class FakeConfig:
    def __init__(self, input_enabled=False):
        self.input_guardrail_enabled = input_enabled
        self.output_guardrail_enabled = False

    def get_optional_env(self, key, default):
        values = {
            "SESSION_TIMEOUT_MINUTES": "30",
            "GUARDRAIL_ENABLED": "true",
            "GUARDRAIL_STRICTNESS": "medium",
        }
        return values.get(key, default)


class FakeSessionManager:
    def get_session(self, session_id):
        return None

    def update_session(self, **kwargs):
        return None


def install_query_fakes(monkeypatch, graph_result, config=None, guardrail=None):
    config = config or FakeConfig()
    guardrail = guardrail or SimpleNamespace()

    class FakeRAG:
        def __init__(self, **kwargs):
            self.graph = SimpleNamespace(invoke=lambda state: graph_result)

    monkeypatch.setattr(routes, "get_config", lambda: config)
    monkeypatch.setattr(routes, "get_session_manager", lambda **kwargs: FakeSessionManager())
    monkeypatch.setattr(routes, "get_guardrail", lambda: guardrail)
    monkeypatch.setattr(routes, "MultiAgentRAG", FakeRAG)
    monkeypatch.setattr(routes, "track_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(routes, "track_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(routes, "log_request_start", lambda *args, **kwargs: None)
    monkeypatch.setattr(routes, "log_request_complete", lambda *args, **kwargs: None)


def request(question="test question"):
    return QueryRequest(question=question, session_id="test-session", top_k=5)


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

    response = await routes.query_documents.__wrapped__(request("What is the weather?"))

    assert response.guardrail_reason == reason


@pytest.mark.asyncio
async def test_success_returns_guardrail_reason(monkeypatch):
    reason = "The retrieved passage directly answers the TAL claims question."
    install_query_fakes(
        monkeypatch,
        {
            "guardrail_passed": True,
            "guardrail_reason": reason,
            "retrieved_chunks": ["TAL paid over $4.2 billion to over 50,128 customers."],
            "retrieved_metadata": [],
            "intent": "factual",
            "answer": "TAL paid over $4.2 billion to over 50,128 customers.",
        },
    )

    response = await routes.query_documents.__wrapped__(request())

    assert response.guardrail_reason == reason


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

    response = await routes.query_documents.__wrapped__(request())

    assert "Violence=4 (>= 4)" in response.guardrail_reason
```

- [ ] **Step 2: Run the API tests and verify RED**

Run: `/Users/dukeisyourdaddy/Desktop/multiagent-rag/.venv/bin/python -m pytest tests/test_query_api.py -q`

Expected: FAIL because `QueryResponse` does not define or populate `guardrail_reason`.

- [ ] **Step 3: Add and populate the response field**

Add this required model field and update its schema example:

```python
guardrail_reason: str = Field(
    ...,
    description="Why the safety or relevance guardrail accepted or rejected the request",
)
```

Populate it in all three `QueryResponse` constructors. For an input-safety block, format the reason from `input_gr.triggered`; for relevance and success paths, use `result["guardrail_reason"]`.

- [ ] **Step 4: Run the API tests and verify GREEN**

Run: `/Users/dukeisyourdaddy/Desktop/multiagent-rag/.venv/bin/python -m pytest tests/test_query_api.py -q`

Expected: all API response tests pass without network calls.

- [ ] **Step 5: Commit Task 3**

```bash
git add tests/test_query_api.py src/api/models.py src/api/routes.py
git commit -m "Expose guardrail reasons in query responses"
```

---

### Task 4: Documentation and End-to-End Verification

**Files:**
- Modify: `README.md:34-56`
- Modify: `README.md:386-465`
- Modify: `src/api/main.py:37-61`
- Modify: `docs/superpowers/specs/2026-07-19-retrieval-grounded-relevance-guardrail-design.md`

**Interfaces:**
- Consumes: final graph order and API response contract.
- Produces: documentation that matches the implementation and a verified live TAL query.

- [ ] **Step 1: Update architecture and response documentation**

Document `Content Safety -> retrieval -> relevance guardrail -> intent -> answer`, explain that relevance uses retrieved chunks, and add `guardrail_reason` to successful and rejected JSON examples. Clarify that unexpected API failures continue to use FastAPI's HTTP error response rather than `QueryResponse`.

- [ ] **Step 2: Run all offline checks**

Run:

```bash
/Users/dukeisyourdaddy/Desktop/multiagent-rag/.venv/bin/python -m pytest tests -q
/Users/dukeisyourdaddy/Desktop/multiagent-rag/.venv/bin/python -m compileall -q config src main.py tests
/Users/dukeisyourdaddy/Desktop/multiagent-rag/.venv/bin/python -m pip check
/Users/dukeisyourdaddy/Desktop/multiagent-rag/.venv/bin/python -m black --check src/agents/base_agent.py src/agents/rag_agent.py src/api/models.py src/api/routes.py tests
git diff --check
```

Expected: all tests and checks pass with no dependency conflicts or whitespace errors.

- [ ] **Step 3: Run the live TAL regression query**

Run:

```bash
/Users/dukeisyourdaddy/Desktop/multiagent-rag/.venv/bin/python main.py query "In 2024 how much did TAL Group paid and over how many customers?" --top-k 5
```

Expected: guardrail passes after retrieval, source is `doc.pdf`, and the answer contains `$4.2 billion` and `50,128 customers`.

- [ ] **Step 4: Verify the live API contract**

Start Uvicorn on an unused local port and POST the TAL query to `/api/v1/query`. Assert HTTP 200, `success=true`, `guardrail_passed=true`, and a non-empty `guardrail_reason`; then stop the server cleanly.

- [ ] **Step 5: Commit Task 4**

```bash
git add README.md src/api/main.py docs/superpowers/specs/2026-07-19-retrieval-grounded-relevance-guardrail-design.md
git commit -m "Document retrieval-grounded guardrails"
```

- [ ] **Step 6: Review and publish**

Inspect `git status`, `git diff main...HEAD`, and the commit list; exclude `outputs/`. Push `codex/relevance-guardrail` to `origin` and open a draft pull request targeting `main` with root cause, behavior change, compatibility impact, and verification results.
