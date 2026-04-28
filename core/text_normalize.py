"""Единая точка нормализации русско-английских текстов для scoring/matching.

Используется: filter_ranking, semantic_frame, where_resolver. Раньше каждый
из этих модулей держал собственную копию тех же функций.
"""

from __future__ import annotations

import re

_NON_WORD_RE = re.compile(r"[^0-9a-zа-я_ ]+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")

_RUSSIAN_SUFFIXES: tuple[str, ...] = (
    "ыми", "ими", "ого", "его", "ому", "ему", "ая", "яя", "ое", "ее",
    "ые", "ие", "ый", "ий", "ой", "ом", "ем", "ым", "им", "ах", "ях",
    "ов", "ев", "ей", "ам", "ям", "у", "ю", "а", "я", "ы", "и", "е", "о",
)

_MIN_TOKEN_LEN = 2
_MIN_STEM_LEN = 4


def normalize_text(text: str) -> str:
    """Привести текст к нижнему регистру, убрать пунктуацию, сжать пробелы."""
    text = str(text or "").lower().replace("ё", "е")
    text = _NON_WORD_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Разбить текст на токены длины ≥ 2."""
    return [tok for tok in normalize_text(text).split() if len(tok) >= _MIN_TOKEN_LEN]


def stem(token: str) -> str:
    """Отрезать русские окончания, оставив корень длиной ≥ 4."""
    token = normalize_text(token)
    for suffix in _RUSSIAN_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM_LEN:
            return token[: -len(suffix)]
    return token


def stem_set(text: str) -> set[str]:
    """Множество стемов из текста."""
    return {stem(tok) for tok in tokenize(text)}
