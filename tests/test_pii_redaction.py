"""Tests for PII redaction extractor."""

from __future__ import annotations

import os

from reify.extract.pii_redaction import PIIRedactor


async def test_pii_redactor_disabled_by_default() -> None:
    """PII redaction is off by default."""
    redactor = PIIRedactor()
    text = "My email is john@example.com and phone is 555-123-4567"
    result = await redactor.extract(text, context=None)
    assert result == text


async def test_pii_redactor_enabled_with_env() -> None:
    """PII redaction works when REIFY_PII_REDACT=on."""
    os.environ['REIFY_PII_REDACT'] = 'on'
    try:
        redactor = PIIRedactor()
        text = "My email is john@example.com and phone is 555-123-4567"
        result = await redactor.extract(text, context=None)
        assert '[EMAIL]' in result
        assert '[PHONE]' in result
        assert 'john@example.com' not in result
        assert '555-123-4567' not in result
    finally:
        os.environ.pop('REIFY_PII_REDACT', None)


async def test_email_redaction() -> None:
    """Email addresses are redacted."""
    os.environ['REIFY_PII_REDACT'] = 'on'
    try:
        redactor = PIIRedactor()
        text = "Contact me at john@example.com or jane.doe@company.co.uk"
        result = await redactor.extract(text, context=None)
        assert result == "Contact me at [EMAIL] or [EMAIL]"
    finally:
        os.environ.pop('REIFY_PII_REDACT', None)


async def test_phone_us_redaction() -> None:
    """US phone numbers in common formats are redacted."""
    os.environ['REIFY_PII_REDACT'] = 'on'
    try:
        redactor = PIIRedactor()
        text = "Call me at 555-123-4567 or 555 123 4567"
        result = await redactor.extract(text, context=None)
        assert result == "Call me at [PHONE] or [PHONE]"
    finally:
        os.environ.pop('REIFY_PII_REDACT', None)


async def test_phone_intl_redaction() -> None:
    """International phone numbers are redacted."""
    os.environ['REIFY_PII_REDACT'] = 'on'
    try:
        redactor = PIIRedactor()
        text = "International: +1 555 123 4567"
        result = await redactor.extract(text, context=None)
        assert '[PHONE]' in result
        assert '555 123 4567' not in result
    finally:
        os.environ.pop('REIFY_PII_REDACT', None)


async def test_ssn_redaction() -> None:
    """SSN-shaped strings are redacted."""
    os.environ['REIFY_PII_REDACT'] = 'on'
    try:
        redactor = PIIRedactor()
        text = "My SSN is 123-45-6789"
        result = await redactor.extract(text, context=None)
        assert result == "My SSN is [SSN]"
    finally:
        os.environ.pop('REIFY_PII_REDACT', None)


async def test_credit_card_redaction() -> None:
    """Credit-card-shaped strings are redacted."""
    os.environ['REIFY_PII_REDACT'] = 'on'
    try:
        redactor = PIIRedactor()
        text = "Card: 4111-1111-1111-1111 or 4111 1111 1111 1111"
        result = await redactor.extract(text, context=None)
        assert result == "Card: [CC] or [CC]"
    finally:
        os.environ.pop('REIFY_PII_REDACT', None)


async def test_mixed_pii_redaction() -> None:
    """Multiple PII types in one text are all redacted."""
    os.environ['REIFY_PII_REDACT'] = 'on'
    try:
        redactor = PIIRedactor()
        text = "Email: john@example.com, Phone: 555-123-4567, SSN: 123-45-6789, Card: 4111-1111-1111-1111"
        result = await redactor.extract(text, context=None)
        assert '[EMAIL]' in result
        assert '[PHONE]' in result
        assert '[SSN]' in result
        assert '[CC]' in result
        assert 'john@example.com' not in result
        assert '555-123-4567' not in result
        assert '123-45-6789' not in result
        assert '4111-1111-1111-1111' not in result
    finally:
        os.environ.pop('REIFY_PII_REDACT', None)


async def test_no_false_positives() -> None:
    """Non-PII text is not redacted."""
    os.environ['REIFY_PII_REDACT'] = 'on'
    try:
        redactor = PIIRedactor()
        text = "My phone number is not in this text and no email either"
        result = await redactor.extract(text, context=None)
        assert result == text
    finally:
        os.environ.pop('REIFY_PII_REDACT', None)
