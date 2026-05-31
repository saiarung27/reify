"""PII redaction extractor for the write path.

This extracts and redacts personally identifiable information (PII) from
memories before storage, protecting user privacy.

Deterministic regex-based redaction covers:
- Email addresses
- Phone numbers (US and international formats)
- SSN-shaped strings (XXX-XX-XXXX format)
- Credit-card-shaped strings (13-19 digit groups)

Future work may add LLM-based redaction for contextual PII detection.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reify.extract import ExtractionContext


class PIIRedactor:
    """PII redaction extractor.

    Deterministic regex-based redaction for common PII patterns.
    Gated by REIFY_PII_REDACT=on environment variable.
    """

    # Email pattern: standard email format
    _EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )

    # US phone: (XXX) XXX-XXXX, XXX-XXX-XXXX, XXX XXX XXXX
    _PHONE_US_PATTERN = re.compile(
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    )

    # International phone: +XX XXX XXX XXXX, +XX-XXX-XXX-XXXX
    _PHONE_INTL_PATTERN = re.compile(
        r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}'
    )

    # SSN pattern: XXX-XX-XXXX (not actual SSNs, just the format)
    _SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

    # Credit card pattern: 16 digits in groups of 4 with spaces/dashes
    _CC_PATTERN = re.compile(
        r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
    )

    def __init__(self) -> None:
        self._enabled = os.getenv('REIFY_PII_REDACT', '').lower() == 'on'

    async def extract(self, text: str, context: ExtractionContext) -> str:
        """Extract and redact PII from text.

        Returns redacted text if REIFY_PII_REDACT=on, otherwise returns text unchanged.
        """
        if not self._enabled:
            return text

        redacted = text

        # Redact emails
        redacted = self._EMAIL_PATTERN.sub('[EMAIL]', redacted)

        # Redact phone numbers
        redacted = self._PHONE_US_PATTERN.sub('[PHONE]', redacted)
        redacted = self._PHONE_INTL_PATTERN.sub('[PHONE]', redacted)

        # Redact SSN-shaped strings
        redacted = self._SSN_PATTERN.sub('[SSN]', redacted)

        # Redact credit-card-shaped strings
        redacted = self._CC_PATTERN.sub('[CC]', redacted)

        return redacted


__all__ = [
    "PIIRedactor",
]
