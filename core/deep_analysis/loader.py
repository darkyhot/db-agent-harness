"""Memory-safe table loader for deep analysis.

Strategy:
1. Fetch DDL + 10k row sample to measure actual max string length per column.
2. Drop columns whose max string length exceeds WIDE_TEXT_THRESHOLD. Important
   business/key columns (inferred by role heuristics + name patterns) are
   preserved regardless, because losing them would defeat the analysis.
3. Estimate full-table RAM usage from row count × average bytes per row on the
   sample. If the estimate exceeds SAFE_MAX_DF_BYTES, drop more columns
   aggressively or fall back to random sampling with an explicit note in the
   profile (the pipeline documents the downsample in the final report).
4. Execute the actual full-load SELECT with the allowed projection.

When the caller supplies a `where` clause, every read operation (row count,
profile sample, final SELECT) applies the same predicate so the analysis
stays consistent with the user-requested slice (e.g. one year, one ИНН set).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import psutil
from sqlalchemy import text

from core.database import DatabaseManager, _validate_identifier
from core.deep_analysis.logging_setup import get_logger

WIDE_TEXT_THRESHOLD = 100          # max chars after which a text column is dropped
PROFILE_SAMPLE_ROWS = 10_000
SAFE_MAX_DF_BYTES = 60 * 1024 ** 3  # ~60 GB hard cap for one DataFrame
MIN_RAM_HEADROOM_BYTES = 8 * 1024 ** 3  # leave at least 8 GB free

# Tokens that must never appear in a user-supplied WHERE clause. We're inlining
# the predicate into raw SQL (parameter binding can't represent the structure
# of a free-form filter), so the safe-list approach is to reject anything that
# could mutate state, terminate the statement, or open a comment.
_WHERE_FORBIDDEN_TOKENS = (
    ";", "--", "/*", "*/",
)
_WHERE_FORBIDDEN_KEYWORDS = (
    "drop", "delete", "update", "insert", "alter", "create", "truncate",
    "grant", "revoke", "merge", "call", "execute", "exec", "copy",
    "vacuum", "attach", "detach", "into",
)
_WHERE_KEYWORD_RE = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(_WHERE_FORBIDDEN_KEYWORDS) + r")(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)
_WHERE_MAX_LEN = 1000


def _sanitize_where_clause(where: str | None) -> str | None:
    """Validate a free-form WHERE predicate and return a normalised form.

    None / empty string → returns None (no filter). Anything that would
    terminate the surrounding SELECT, open a SQL comment, or invoke a
    state-changing statement is rejected with a descriptive ValueError.

    The check is intentionally conservative — false positives are easier to
    explain to the user than a smuggled DELETE.
    """
    if where is None:
        return None
    cleaned = where.strip()
    if not cleaned:
        return None
    if cleaned.lower().startswith("where "):
        cleaned = cleaned[6:].lstrip()
    if not cleaned:
        return None
    if len(cleaned) > _WHERE_MAX_LEN:
        raise ValueError(
            f"WHERE-условие слишком длинное (>{_WHERE_MAX_LEN} символов)."
        )
    for tok in _WHERE_FORBIDDEN_TOKENS:
        if tok in cleaned:
            raise ValueError(
                f"WHERE-условие содержит запрещённую последовательность '{tok}'."
            )
    match = _WHERE_KEYWORD_RE.search(cleaned)
    if match:
        raise ValueError(
            f"WHERE-условие содержит запрещённое ключевое слово "
            f"'{match.group(1)}' — допускаются только предикаты SELECT."
        )
    if cleaned.count("'") % 2 != 0:
        raise ValueError("WHERE-условие имеет непарную кавычку — проверьте строки.")
    return cleaned


@dataclass
class LoadPlan:
    schema: str
    table: str
    total_rows: int
    kept_columns: list[str]
    dropped_wide_text: list[str]
    strategy: str                 # "full" | "sample"
    sample_rows: int | None       # populated when strategy == "sample"
    est_bytes_per_row: float
    est_full_bytes: float
    where_clause: str | None = None  # echoes the user's --where for reporting


class SafeLoader:
    """Loads a table into pandas while protecting host RAM."""

    # Patterns whose name hints at business-critical info — never drop even if wide.
    _PROTECTED_NAME_SUBSTRINGS = (
        "id", "inn", "kpp", "email", "phone", "date", "dt", "time", "status",
        "type", "code", "category", "segment", "flag", "amount", "sum", "value",
    )

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._log = get_logger()

    def plan_and_load(
        self,
        schema: str,
        table: str,
        *,
        progress_cb=None,
        where: str | None = None,
    ) -> tuple[pd.DataFrame, LoadPlan]:
        schema = _validate_identifier(schema, "schema")
        table = _validate_identifier(table, "table")
        where_clause = _sanitize_where_clause(where)
        if where_clause:
            self._log.info("Loading slice with WHERE: %s", where_clause)

        if progress_cb:
            if where_clause:
                progress_cb(f"Считаю количество строк (WHERE {where_clause})...")
            else:
                progress_cb("Считаю количество строк в таблице...")
        total_rows = self._db.get_row_count(schema, table, where=where_clause)

        if progress_cb:
            progress_cb("Профилирую первые 10 000 строк для подбора колонок...")
        sample = self._db.get_sample(
            schema, table, n=PROFILE_SAMPLE_ROWS, where=where_clause
        )

        kept, dropped, str_widths = self._select_columns(sample)
        est_row_bytes = self._estimate_row_bytes(sample[kept])
        est_full_bytes = est_row_bytes * max(total_rows, 1)
        available = psutil.virtual_memory().available
        allow_full = (
            est_full_bytes <= SAFE_MAX_DF_BYTES
            and est_full_bytes + MIN_RAM_HEADROOM_BYTES <= available
        )

        self._log.info(
            "Load estimate for %s.%s: rows=%d, kept=%d, dropped_wide=%d, "
            "bytes/row=%.0f, full=%.2fGB, available=%.2fGB, allow_full=%s",
            schema, table, total_rows, len(kept), len(dropped),
            est_row_bytes, est_full_bytes / 1024**3, available / 1024**3, allow_full,
        )

        if allow_full:
            if progress_cb:
                progress_cb(
                    f"Загружаю таблицу полностью "
                    f"({total_rows} строк, ~{est_full_bytes / 1024**3:.1f} ГБ)..."
                )
            df = self._load_projection(
                schema, table, kept, limit=None, where=where_clause
            )
            plan = LoadPlan(
                schema=schema, table=table, total_rows=total_rows,
                kept_columns=kept, dropped_wide_text=dropped,
                strategy="full", sample_rows=None,
                est_bytes_per_row=est_row_bytes, est_full_bytes=est_full_bytes,
                where_clause=where_clause,
            )
            return df, plan

        # Fallback: random downsample sized to fit the RAM budget.
        budget = min(SAFE_MAX_DF_BYTES, max(available - MIN_RAM_HEADROOM_BYTES, 1))
        sample_rows = max(1, int(budget / max(est_row_bytes, 1)))
        sample_rows = min(sample_rows, total_rows)
        self._log.warning(
            "Full load exceeds budget — falling back to random sample of %d rows",
            sample_rows,
        )
        if progress_cb:
            progress_cb(
                f"Таблица слишком большая, беру случайную выборку {sample_rows} строк..."
            )
        df = self._db.get_random_sample(
            schema, table, n=sample_rows, columns=kept, where=where_clause
        )
        plan = LoadPlan(
            schema=schema, table=table, total_rows=total_rows,
            kept_columns=kept, dropped_wide_text=dropped,
            strategy="sample", sample_rows=sample_rows,
            est_bytes_per_row=est_row_bytes, est_full_bytes=est_full_bytes,
            where_clause=where_clause,
        )
        return df, plan

    # ---------- internals ----------

    def _select_columns(
        self, sample: pd.DataFrame
    ) -> tuple[list[str], list[str], dict[str, int]]:
        """Return (kept_columns, dropped_wide_text, max_str_len_by_col)."""
        kept: list[str] = []
        dropped: list[str] = []
        widths: dict[str, int] = {}
        for col in sample.columns:
            series = sample[col]
            if series.dtype == object:
                try:
                    max_len = int(series.dropna().astype(str).str.len().max() or 0)
                except Exception:
                    max_len = 0
                widths[col] = max_len
                if max_len > WIDE_TEXT_THRESHOLD and not self._is_protected(col):
                    dropped.append(col)
                    continue
            kept.append(col)
        return kept, dropped, widths

    def _is_protected(self, column_name: str) -> bool:
        low = column_name.lower()
        return any(needle in low for needle in self._PROTECTED_NAME_SUBSTRINGS)

    def _estimate_row_bytes(self, sample_df: pd.DataFrame) -> float:
        if sample_df.empty:
            return 1.0
        # memory_usage(deep=True) approximates Python-object size including the
        # char buffer of strings — the right measure for our risk model.
        total = sample_df.memory_usage(deep=True, index=False).sum()
        return float(total) / len(sample_df)

    def _load_projection(
        self,
        schema: str,
        table: str,
        columns: list[str],
        *,
        limit: int | None,
        where: str | None = None,
    ) -> pd.DataFrame:
        safe_cols = [_validate_identifier(c, "column") for c in columns]
        projection = ", ".join(f'"{c}"' for c in safe_cols) if safe_cols else "*"
        sql = f'SELECT {projection} FROM "{schema}"."{table}"'
        if where:
            sql += f" WHERE {where}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        self._log.info("Executing load SQL: %s", sql[:300])
        with self._db.get_engine().connect() as conn:
            return pd.read_sql(text(sql), conn)
