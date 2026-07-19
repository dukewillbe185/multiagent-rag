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
