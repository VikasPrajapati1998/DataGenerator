"""Prompt-injection resistance for inputs and LLM outputs.

Inputs: user instructions are length-capped, stripped of control characters
and of sentences that try to override the system prompt, then wrapped in
explicit data delimiters.

Outputs: generated text is rejected when it contains instructions, code,
markup, URLs, or patterns that resemble real personal identifiers.
"""

from __future__ import annotations

import re

from app.monitoring.logger import get_logger

logger = get_logger(__name__)

MAX_INSTRUCTION_CHARS = 2000
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INJECTION = re.compile(
    r"(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|system)"
    r"|you\s+are\s+now|act\s+as\s+(an?\s+)?(unrestricted|jailbroken)|system\s+prompt|developer\s+mode"
    r"|reveal\s+(your|the)\s+(instructions|prompt)|</?\s*(system|assistant|instructions)\s*>",
    re.IGNORECASE,
)
_OUTPUT_FORBIDDEN = (
    (re.compile(r"https?://|www\.", re.IGNORECASE), "contains a URL"),
    (re.compile(r"<\s*/?\s*(script|iframe|img|a|div|html)\b", re.IGNORECASE), "contains markup"),
    (re.compile(r"```|^\s*(import|from)\s+\w+|\bdef\s+\w+\(|\bSELECT\s+[\w*,\s]+\s+FROM\s+\w+"), "contains code"),
    (_INJECTION, "contains instruction-like text"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "resembles a US SSN"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "resembles an Aadhaar/card number"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "resembles an Indian PAN"),
    (re.compile(r"(lorem ipsum|as an ai\b|i cannot (help|assist|comply)|i'm sorry, but|\bplaceholder\b"
                r"|\btodo\b|^n/?a$)", re.IGNORECASE), "is placeholder or refusal text"),
)


def sanitize_instructions(text: str | None) -> str:
    """Return user instructions that are safe to embed in a prompt.

    Args:
        text: Raw user instructions.

    Returns:
        Cleaned instructions (possibly empty).
    """
    if not text:
        return ""
    cleaned = _CONTROL.sub(" ", text)[:MAX_INSTRUCTION_CHARS]
    sentences = re.split(r"(?<=[.!?\n])\s+", cleaned)
    kept = [s for s in sentences if not _INJECTION.search(s)]
    if len(kept) != len(sentences):
        logger.warning("Removed %d instruction sentence(s) that looked like prompt injection",
                       len(sentences) - len(kept))
    return " ".join(kept).strip()


def wrap_untrusted(label: str, text: str) -> str:
    """Wrap untrusted text in labelled delimiters the system prompt refers to.

    Args:
        label: Tag name, e.g. ``user_instructions``.
        text: Untrusted content (already sanitised).

    Returns:
        The delimited block.
    """
    safe = text.replace(f"</{label}>", "")
    return f"<{label}>\n{safe}\n</{label}>"


def check_generated_text(value: str, max_length: int) -> str | None:
    """Return a rejection reason for unsafe LLM text, or ``None`` if acceptable.

    Args:
        value: Generated text.
        max_length: Maximum allowed length.

    Returns:
        Reason string when the value must be rejected.
    """
    if not value or not value.strip():
        return "is empty"
    if len(value) > max_length:
        return f"exceeds {max_length} characters"
    if _CONTROL.search(value):
        return "contains control characters"
    for pattern, reason in _OUTPUT_FORBIDDEN:
        if pattern.search(value):
            return reason
    return None
