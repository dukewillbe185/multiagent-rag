"""
Azure AI Content Safety guardrails (input + output).

Deterministic, low-latency safety checks that wrap the RAG pipeline:

- Input guardrail  : text moderation (Hate / SelfHarm / Sexual / Violence) +
                     Prompt Shields (jailbreak / prompt-injection detection) on
                     the user's question, BEFORE any LLM work is done.
- Output guardrail : text moderation on the generated answer, BEFORE it is
                     returned to the user.

Calls the Content Safety REST API directly (via ``requests``) so it does not
depend on the ``azure-ai-contentsafety`` SDK version.

Flow:  msg -> check_input -> (RAG agents / LLM) -> answer -> check_output -> user
        └─ if detected, the caller returns immediately without continuing.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

from config import get_config

logger = logging.getLogger(__name__)

# Content Safety moderation categories
MODERATION_CATEGORIES = ["Hate", "SelfHarm", "Sexual", "Violence"]

# Human-readable statuses surfaced in the API response
STATUS_CLEAN = "nothing detected"
STATUS_DETECTED = "detected"
STATUS_SKIPPED = "not evaluated"


@dataclass
class GuardrailResult:
    """Outcome of a single guardrail evaluation."""
    detected: bool
    status: str
    categories: Dict[str, int] = field(default_factory=dict)  # category -> severity (0/2/4/6)
    triggered: List[str] = field(default_factory=list)        # reasons that crossed threshold
    jailbreak: bool = False
    error: Optional[str] = None

    def to_details(self) -> Dict:
        return {
            "detected": self.detected,
            "status": self.status,
            "categories": self.categories,
            "triggered": self.triggered,
            "jailbreak_attack_detected": self.jailbreak,
            "error": self.error,
        }


class ContentSafetyGuardrail:
    """Thin client around Azure AI Content Safety text:analyze + text:shieldPrompt."""

    def __init__(self):
        config = get_config()
        self.endpoint = (config.content_safety_endpoint or "").rstrip("/")
        self.key = config.content_safety_key
        self.api_version = config.content_safety_api_version
        self.block_severity = config.content_safety_block_severity
        # Per-category severity thresholds; fall back to the global block severity.
        self.category_thresholds = getattr(config, "content_safety_category_thresholds", {})
        self.configured = bool(self.endpoint and self.key)
        if not self.configured:
            logger.warning(
                "Content Safety endpoint/key not configured; guardrails will pass through."
            )

    def _headers(self) -> Dict[str, str]:
        return {"Ocp-Apim-Subscription-Key": self.key, "Content-Type": "application/json"}

    def _moderate(self, text: str) -> GuardrailResult:
        """Run text moderation; flag if any category severity >= configured threshold."""
        if not self.configured:
            return GuardrailResult(detected=False, status=STATUS_CLEAN, error="not_configured")
        if not text or not text.strip():
            return GuardrailResult(detected=False, status=STATUS_CLEAN)

        url = f"{self.endpoint}/contentsafety/text:analyze?api-version={self.api_version}"
        body = {
            "text": text[:10000],  # API accepts up to 10K chars per call
            "categories": MODERATION_CATEGORIES,
            "outputType": "FourSeverityLevels",
        }
        try:
            resp = requests.post(url, headers=self._headers(), json=body, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            # Fail-open for availability, but surface the error in details.
            logger.error(f"Content Safety moderation error: {e}")
            return GuardrailResult(detected=False, status=STATUS_CLEAN, error=str(e))

        cats = {c["category"]: c.get("severity", 0) for c in data.get("categoriesAnalysis", [])}
        triggered = []
        for category, severity in cats.items():
            threshold = self.category_thresholds.get(category, self.block_severity)
            if severity >= threshold:
                triggered.append(f"{category}={severity} (>= {threshold})")
        detected = len(triggered) > 0
        return GuardrailResult(
            detected=detected,
            status=STATUS_DETECTED if detected else STATUS_CLEAN,
            categories=cats,
            triggered=triggered,
        )

    def _shield_prompt(self, text: str) -> Tuple[bool, Optional[str]]:
        """Prompt Shields: detect jailbreak / prompt-injection. Returns (attack_detected, error)."""
        if not self.configured:
            return False, "not_configured"
        url = f"{self.endpoint}/contentsafety/text:shieldPrompt?api-version={self.api_version}"
        body = {"userPrompt": text[:10000], "documents": []}
        try:
            resp = requests.post(url, headers=self._headers(), json=body, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Content Safety prompt shield error: {e}")
            return False, str(e)
        attack = bool(data.get("userPromptAnalysis", {}).get("attackDetected", False))
        return attack, None

    def check_input(self, text: str) -> GuardrailResult:
        """Input guardrail: moderation + jailbreak/prompt-injection detection."""
        result = self._moderate(text)
        attack, err = self._shield_prompt(text)
        if attack:
            result.jailbreak = True
            result.detected = True
            result.status = STATUS_DETECTED
            result.triggered.append("Jailbreak/PromptInjection")
        if err and result.error is None:
            result.error = err
        return result

    def check_output(self, text: str) -> GuardrailResult:
        """Output guardrail: moderation on the generated answer."""
        return self._moderate(text)


# Module-level singleton
_guardrail: Optional[ContentSafetyGuardrail] = None


def get_guardrail() -> ContentSafetyGuardrail:
    """Return the shared ContentSafetyGuardrail instance."""
    global _guardrail
    if _guardrail is None:
        _guardrail = ContentSafetyGuardrail()
    return _guardrail
