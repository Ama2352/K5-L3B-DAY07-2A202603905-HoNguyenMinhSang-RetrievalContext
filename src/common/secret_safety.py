"""Remove known credentials and database connection strings from output records."""
import os
import re
from urllib.parse import unquote, urlsplit

DB_URI = re.compile(r'(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s\"\'<>]+')
BEARER = re.compile(r'(?i)(\bbearer\s+)[A-Za-z0-9._~+/-]+')
ASSIGNMENT = re.compile(r'''(?i)(\b(?:api[_-]?key|password|secret|access[_-]?token)\b\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s,;]+)''')
TOKEN = re.compile(r'\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|AIza[A-Za-z0-9_-]{25,}|AKIA[A-Z0-9]{16})\b')
_SENSITIVE_VALUES = set()


def redact_text(value):
    result = value
    for secret in sorted(_SENSITIVE_VALUES, key=len, reverse=True):
        result = result.replace(secret, '[REDACTED]')
    # Replace values, never log them; short values are ignored to avoid mangling ordinary text.
    for key, secret in os.environ.items():
        upper = key.upper()
        if (upper == 'DATABASE_URL' or upper.endswith(('_API_KEY', '_PASSWORD', '_SECRET', '_TOKEN'))) and len(secret) >= 8:
            result = result.replace(secret, '[REDACTED]')
    result = DB_URI.sub('[REDACTED_DATABASE_URL]', result)
    result = BEARER.sub(r'\1[REDACTED]', result)
    return TOKEN.sub('[REDACTED_TOKEN]', ASSIGNMENT.sub(r'\1[REDACTED]', result))


def redact(value):
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            safe_key = redact_text(key) if isinstance(key, str) else key
            # Retain all values if different secret-bearing keys collapse after redaction.
            if safe_key in result:
                safe_key = f'{safe_key}_[redacted_duplicate_{len(result)}]'
            result[safe_key] = redact(item)
        return result
    return value


def database_url(day6_root):
    """Read connection settings privately from the environment or Day 6's local .env."""
    value = os.environ.get('DATABASE_URL')
    if not value:
        from dotenv import dotenv_values
        value = dotenv_values(day6_root / 'starter' / '.env').get('DATABASE_URL')
    if not value:
        raise ValueError('Database connection is not configured in the environment or local Day 6 .env')
    _SENSITIVE_VALUES.add(value)
    try:
        password = urlsplit(value).password
        if password:
            for variant in (password, unquote(password)):
                if len(variant) >= 4:
                    _SENSITIVE_VALUES.add(variant)
    except ValueError:
        pass
    # Process-local registration allows error messages to be scrubbed; never print it.
    os.environ['DATABASE_URL'] = value
    return value
