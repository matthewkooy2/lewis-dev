from __future__ import annotations

import re


SECRET_VALUE_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\b(?:sk|xox[pbar]|gh[pousr])[_-][A-Za-z0-9_=-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"\b(api[_ -]?key|password|passwd|pwd|secret|token)\b"
        r"(\s*(?:is|=|:)\s*)\S{8,}",
        re.IGNORECASE,
    ),
)


def redact_secret_values(text: str) -> str:
    redacted = text
    for pattern in SECRET_VALUE_PATTERNS:
        if pattern.groups >= 2:
            redacted = pattern.sub(
                lambda match: f"{match[1]}{match[2]}[redacted]", redacted
            )
        else:
            redacted = pattern.sub("[redacted]", redacted)
    return redacted
