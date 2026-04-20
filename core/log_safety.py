"""Безопасные summary-хелперы для логирования без утечки данных."""

from __future__ import annotations

import hashlib
import re
from typing import Any

_TABLE_REF_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b")
_QUOTED_TABLE_REF_RE = re.compile(
    r'"([a-zA-Z_][a-zA-Z0-9_]*)"\s*\.\s*"([a-zA-Z_][a-zA-Z0-9_]*)"'
)


def _extract_tables(sql: str) -> list[str]:
    tables: list[str] = []
    for regex in (_QUOTED_TABLE_REF_RE, _TABLE_REF_RE):
        for schema, table in regex.findall(sql):
            full = f"{schema}.{table}"
            if full not in tables:
                tables.append(full)
    return tables


def _sanitize_sql_preview(sql: str, max_len: int = 120) -> str:
    preview = str(sql or "")
    # redact string literals
    preview = re.sub(r"'([^']|'')*'", "'?'", preview)
    # redact bare numbers
    preview = re.sub(r"\b\d+\b", "?", preview)
    preview = re.sub(r"\s+", " ", preview).strip()
    if len(preview) > max_len:
        preview = preview[: max_len - 3] + "..."
    return preview


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def summarize_text(text: str | None, *, label: str = "text") -> str:
    """Краткая безопасная сводка произвольного текста без раскрытия содержимого."""
    raw = str(text or "")
    return f"{label}[len={len(raw)}, sha={_hash_text(raw)}]"


def summarize_sql(sql: str | None) -> str:
    """Краткая безопасная сводка SQL без текста запроса и литералов."""
    raw = str(sql or "")
    tables = _extract_tables(raw)
    tables_preview = ",".join(tables[:5]) if tables else "-"
    if len(tables) > 5:
        tables_preview += ",..."
    return (
        f"sql[len={len(raw)}, sha={_hash_text(raw)}, tables={tables_preview}, "
        f"preview=\"{_sanitize_sql_preview(raw)}\"]"
    )


def summarize_dict_keys(payload: dict[str, Any] | None, *, label: str = "dict") -> str:
    """Безопасно описать dict только по ключам верхнего уровня."""
    keys = sorted(str(k) for k in (payload or {}).keys())
    preview = ",".join(keys[:10]) if keys else "-"
    if len(keys) > 10:
        preview += ",..."
    return f"{label}[keys={preview}, size={len(keys)}]"
