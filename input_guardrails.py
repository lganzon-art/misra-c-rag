"""Guardrails AI input validation before retrieval, generation, and tracing."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "data" / "model_cache"))
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

from guardrails import Guard
from guardrails.settings import settings
settings.rc.enable_metrics = False

from guardrails_ai.detect_jailbreak import DetectJailbreak
from guardrails_ai.guardrails_pii import GuardrailsPII
from guardrails_ai.restricttotopic import RestrictToTopic
from openai import OpenAI
from presidio_analyzer.nlp_engine import NlpEngineProvider


AUDIT_PATH = Path(__file__).parent / "data" / "guardrail_events.jsonl"
_NUMBERED_RULE = re.compile(r"\b(?:rules?|directives?)\s+\d+(?:\.\d+){1,3}\b", re.I)


@dataclass(frozen=True)
class ScreenedQuestion:
    text: str
    blocked_reason: str | None
    pii_types: tuple[str, ...]
    guardrail: str = "none"


def _audit(outcome: str, guardrail: str, pii_masked: bool) -> None:
    """Record the decision without storing question text or validator exceptions."""
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "event_id": uuid.uuid4().hex,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
        "guardrail": guardrail,
        "pii_masked": pii_masked,
    }
    with AUDIT_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")


@lru_cache(maxsize=1)
def _guards() -> tuple[Guard, Guard, Guard]:
    # Model-backed PII and jailbreak validators run locally. The topic
    # validator sees only text returned by the PII validator.
    pii, jailbreak, topic = Guard(), Guard(), Guard()
    # Guard construction reloads .guardrailsrc. Disable its metrics after all
    # guards are constructed and before any validator sees a user question.
    for guard in (pii, jailbreak, topic):
        guard.configure(allow_metrics_collection=False)
    settings.rc.enable_metrics = False

    small_spacy_config = Path(__file__).parent / "presidio_nlp.yaml"
    with patch.object(NlpEngineProvider, "_get_full_conf_path", return_value=small_spacy_config):
        pii.use(GuardrailsPII(
            entities=["CREDIT_CARD", "US_SSN", "PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON"],
            use_local=True, on_fail="fix",
        ))
    jailbreak.use(DetectJailbreak(use_local=True, on_fail="noop"))
    # RestrictToTopic constructs a classifier even with disable_classifier=True.
    # Reuse the already-loaded jailbreak model; only its LLM topic check runs.
    topic.use(RestrictToTopic(
        valid_topics=["MISRA C", "C coding standard", "C programming rules", "software compliance"],
        invalid_topics=["weather", "cooking", "sports", "investment advice"],
        disable_classifier=True, disable_llm=False, llm_callable="gpt-4o",
        model="zhx123/ftrobertallm",
        use_local=True, on_fail="noop",
    ))
    return pii, jailbreak, topic


def warm_up_guardrails() -> None:
    """Load validators and run local models before the first user question."""
    pii, jailbreak, _topic = _guards()
    sample = "What does MISRA C Rule 2.3 cover?"
    pii.validate(sample)
    jailbreak.validate(sample)


def screen_question(question: str) -> ScreenedQuestion:
    """Fail closed if a validator is unavailable; never log the raw question."""
    safe = "[QUESTION_REDACTED]"
    pii_masked = False
    stage = "GuardrailsPII"
    try:
        pii, jailbreak, topic = _guards()
        pii_result = pii.validate(question)
        if not isinstance(pii_result.validated_output, str):
            raise RuntimeError("PII validator returned no sanitized text")
        safe = pii_result.validated_output
        pii_masked = safe != question
        if not safe.strip() or len(safe) > 4000:
            _audit("blocked", "input_length", pii_masked)
            return ScreenedQuestion(safe, "Please enter a MISRA C question under 4,000 characters.",
                                    ("pii",) if pii_masked else (), "input_length")

        stage = "DetectJailbreak"
        if not jailbreak.validate(safe).validation_passed:
            _audit("blocked", stage, pii_masked)
            return ScreenedQuestion(safe, "The question was blocked by the prompt-injection guardrail.",
                                    ("pii",) if pii_masked else (), stage)

        stage = "OpenAI moderation"
        moderation = OpenAI().moderations.create(model="omni-moderation-latest", input=safe)
        if moderation.results[0].flagged:
            _audit("blocked", stage, pii_masked)
            return ScreenedQuestion(safe, "The question was blocked by the safety guardrail.",
                                    ("pii",) if pii_masked else (), stage)

        stage = "RestrictToTopic"
        # A short question such as "What is rule 2.3?" omits the MISRA name.
        # Give the topic validator the application's document domain while
        # preserving the original sanitized question for retrieval and traces.
        topic_text = f"In the MISRA C coding standard, {safe}" if _NUMBERED_RULE.search(safe) else safe
        if not topic.validate(topic_text).validation_passed:
            _audit("blocked", stage, pii_masked)
            return ScreenedQuestion(safe, "This assistant answers MISRA C coding-standard questions.",
                                    ("pii",) if pii_masked else (), stage)

        _audit("allowed", "all", pii_masked)
        return ScreenedQuestion(safe, None, ("pii",) if pii_masked else ())
    except Exception:
        _audit("error_blocked", stage, pii_masked)
        return ScreenedQuestion(safe, "A guardrail could not verify this question. Please try again later.",
                                ("pii",) if pii_masked else (), stage)
