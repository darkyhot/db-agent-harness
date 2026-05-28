"""Маппинг dType из attr_list.csv → Postgres-тип и базовый Python-генератор значений.

Используется DDL- и data-генераторами. Прод-кода не касается.
"""
from __future__ import annotations

import datetime as _dt
import random
import re
import string
from dataclasses import dataclass
from typing import Any, Callable

# Окно дат для синтетики: данные «происходят» в 2025-01-01 … 2026-05-26.
DEFAULT_DATE_MIN = _dt.date(2025, 1, 1)
DEFAULT_DATE_MAX = _dt.date(2026, 5, 26)

_VARCHAR_RE = re.compile(r"^varchar\((\d+)\)$", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"^numeric(?:\(\s*(\d+)\s*,\s*(\d+)\s*\))?$", re.IGNORECASE)


@dataclass(frozen=True)
class SqlType:
    """Описание Postgres-типа колонки + дефолтный генератор."""

    pg_type: str
    gen: Callable[[random.Random], Any]


def _int_gen(low: int, high: int) -> Callable[[random.Random], int]:
    return lambda rng: rng.randint(low, high)


def _date_gen(
    low: _dt.date = DEFAULT_DATE_MIN, high: _dt.date = DEFAULT_DATE_MAX
) -> Callable[[random.Random], _dt.date]:
    delta_days = (high - low).days
    return lambda rng: low + _dt.timedelta(days=rng.randint(0, delta_days))


def _timestamp_gen(
    low: _dt.date = DEFAULT_DATE_MIN, high: _dt.date = DEFAULT_DATE_MAX
) -> Callable[[random.Random], _dt.datetime]:
    delta_seconds = int((high - low).total_seconds()) if isinstance(high, _dt.datetime) else (high - low).days * 86400
    base = _dt.datetime.combine(low, _dt.time(0, 0))
    return lambda rng: base + _dt.timedelta(seconds=rng.randint(0, delta_seconds))


def _numeric_gen(precision: int, scale: int) -> Callable[[random.Random], float]:
    """numeric(precision, scale): значение в безопасном диапазоне."""
    # Ограничиваем integer-часть до 9 цифр чтобы не упереться в precision слишком близко.
    int_digits = max(1, min(precision - scale, 9))
    upper = 10 ** int_digits - 1
    factor = 10 ** scale
    def _g(rng: random.Random) -> float:
        raw = rng.randint(0, upper * factor) / factor
        # Случайный знак для амортизации
        if rng.random() < 0.05:
            raw = -raw
        return raw
    return _g


def _varchar_gen(max_len: int) -> Callable[[random.Random], str]:
    """Произвольный varchar — короткий человекочитаемый токен."""
    alphabet = string.ascii_uppercase + string.digits
    lo = 1 if max_len < 3 else 3
    hi = min(max_len, 12)
    if hi < lo:
        hi = lo
    def _g(rng: random.Random) -> str:
        n = rng.randint(lo, hi)
        return "".join(rng.choices(alphabet, k=n))
    return _g


def _text_gen() -> Callable[[random.Random], str]:
    return _varchar_gen(40)


def _boolean_gen() -> Callable[[random.Random], bool]:
    return lambda rng: bool(rng.getrandbits(1))


def resolve(dtype: str) -> SqlType:
    """Маппинг dType → SqlType.

    Args:
        dtype: значение колонки `dType` из attr_list.csv.

    Returns:
        SqlType с pg_type и дефолтным генератором.
    """
    d = (dtype or "").strip()
    dl = d.lower()
    if dl == "boolean":
        return SqlType("BOOLEAN", _boolean_gen())
    if dl == "date":
        return SqlType("DATE", _date_gen())
    if dl == "timestamp":
        return SqlType("TIMESTAMP", _timestamp_gen())
    if dl in ("int2", "smallint"):
        return SqlType("SMALLINT", _int_gen(0, 32_000))
    if dl in ("int4", "integer", "int"):
        return SqlType("INTEGER", _int_gen(0, 2_000_000_000))
    if dl in ("int8", "bigint"):
        return SqlType("BIGINT", _int_gen(0, 9_000_000_000_000))
    if dl == "text":
        return SqlType("TEXT", _text_gen())
    m = _VARCHAR_RE.match(d)
    if m:
        n = int(m.group(1))
        return SqlType(f"VARCHAR({n})", _varchar_gen(n))
    m = _NUMERIC_RE.match(d)
    if m:
        precision = int(m.group(1)) if m.group(1) else 18
        scale = int(m.group(2)) if m.group(2) else 2
        return SqlType(f"NUMERIC({precision},{scale})", _numeric_gen(precision, scale))
    # Безопасный fallback — TEXT (никогда не должно сработать на текущем attr_list.csv).
    return SqlType("TEXT", _text_gen())
