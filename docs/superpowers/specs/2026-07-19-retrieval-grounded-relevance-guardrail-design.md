# Retrieval-Grounded Relevance Guardrail Design

## Problem

The current LangGraph workflow asks an LLM to decide whether a question is related to the knowledge base before any search occurs. The classifier sees the question and optional conversation history, but no corpus description or retrieved content. It can therefore reject valid document questions, as happened with the TAL Group claims question, even though Azure AI Search returns the exact supporting passage as its first result.

The same classifier inherits the application's general generation temperature (currently `1.0`), so repeated classifications can vary. The API response also omits the classifier's reason, which hides the cause of a rejection from clients.

## Considered Approaches

### A. Retrieve first, then classify against retrieved evidence (selected)

Run hybrid retrieval before the relevance classifier and include the retrieved chunks in the classifier prompt. Use stable, model-supported sampling for this classifier only. This directly grounds the decision in the same evidence the answer generator will use and preserves the ability to reject unrelated questions.

Trade-off: every safe question incurs an embedding and search request, including unrelated questions. This is acceptable because relevance cannot be established reliably without consulting the corpus.

### B. Remove the relevance blocker entirely

Always retrieve and let the answer generator return an "insufficient information" response when context is unrelated. This minimizes classifier complexity and false rejections, but unrelated requests still reach answer generation and consume more model tokens.

### C. Classify against a static corpus summary before retrieval

Give the classifier a maintained description of indexed document topics. This keeps irrelevant queries away from search, but the summary can become stale and still cannot prove whether a specific fact is present.

Approach A is selected because it provides the strongest relevance signal while keeping the existing rejection behavior.

## Architecture and Data Flow

The API-level Azure AI Content Safety check remains the first operation and continues to block moderation or prompt-shield violations before retrieval.

For safe requests, the LangGraph order changes from:

```text
relevance guardrail -> retrieval -> intent -> answer
```

to:

```text
retrieval -> relevance guardrail -> intent -> answer
                | irrelevant
                +-------------> end
```

The relevance agent receives:

- the current question;
- the previous question for follow-up awareness;
- the retrieved chunk contents and source metadata;
- the configured strictness level.

The relevance decision is evidence-based:

- `relevant`: at least one retrieved chunk materially supports answering the question, or the request is a valid follow-up supported by the current/previous context;
- `irrelevant`: the retrieved evidence does not materially address the question;
- `unsafe`: retained as a defensive classification, although API requests are already protected by Azure AI Content Safety.

If `GUARDRAIL_ENABLED=false`, retrieval still runs and the relevance result does not block the downstream agents, preserving the current configuration contract.

## Classifier Determinism

`BaseAgent` accepts optional per-agent temperature and seed overrides. The deployed `gpt-5-mini` model rejects `temperature=0.0` and only supports its default value of `1`; `GuardrailAgent` therefore uses `temperature=1.0` with `seed=0`. Azure documents seed as best-effort rather than guaranteed determinism, so retrieved evidence and the constrained classification prompt remain the primary reliability controls. Intent identification and answer generation continue using `LLM_TEMPERATURE` from application configuration.

`GUARDRAIL_STRICTNESS` remains backward compatible and will be included explicitly in the relevance prompt:

- `low`: allow plausible topical support;
- `medium`: require material supporting information;
- `high`: require direct evidence for the requested fact or task.

## API Contract

`QueryResponse` will add a `guardrail_reason: str` field.

- Input safety rejection: reason describes the triggered Content Safety checks.
- Relevance rejection: reason is the relevance classifier explanation.
- Successful answer: reason records why the retrieved evidence was accepted.
- CLI wrapper failure: reason identifies the processing failure without exposing credentials.

Existing fields and rejection messages remain unchanged, so clients using the current schema remain compatible apart from receiving one additional response property.

Unexpected API failures continue to use FastAPI's existing HTTP error response rather than `QueryResponse`.

## Error Handling

Malformed relevance-classifier output keeps the existing conservative parsing behavior, but its failure reason is returned and logged. LLM invocation failures remain fail-open so a transient classifier outage does not block valid knowledge-base questions.

Azure retrieval failures continue through the existing retrieval error path; this change does not alter retry or search-service behavior.

## Tests and Acceptance Criteria

Automated tests will use deterministic fakes and make no Azure calls.

The change is accepted when:

1. The compiled workflow invokes retrieval before relevance classification.
2. An irrelevant decision stops before intent identification and answer generation.
3. The relevance prompt contains the TAL supporting passage and the current question.
4. `GuardrailAgent` initializes its LLM with the GPT-5-supported temperature `1.0` and best-effort `seed=0`, independently of `LLM_TEMPERATURE`.
5. `QueryResponse` serializes `guardrail_reason` for successful and rejected responses.
6. The existing CLI query path still returns `guardrail_passed` and `guardrail_reason`.
7. The full local test suite passes.
8. A live query for the TAL question retrieves `doc.pdf` and returns the `$4.2 billion` and `50,128 customers` facts.

## Out of Scope

- Changing Azure Search ranking, chunking, or embedding models.
- Introducing a fixed threshold on raw Azure hybrid-search scores; reciprocal-rank-fusion scores are not calibrated probabilities.
- Redesigning session storage or conversation-memory depth.
- Modifying the generated presentation and diagram artifacts under `outputs/`.
