"""Column profiler: infers the semantic role of every column.

Roles drive hypothesis instantiation. For example, seasonality runner needs at
least one date column + one numeric column; group_anomalies needs at least one
id-like column + one date + one numeric or flag.

Heuristics blend pandas dtype with column name patterns and value statistics.
Name patterns are Russian-aware because the project targets Сбер-style schemas.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from core.deep_analysis.loader import LoadPlan
from core.deep_analysis.logging_setup import get_logger
from core.deep_analysis.types import ColumnProfile, ColumnRole, TableProfile

_MONEY_HINTS = (
    "amount", "amt", "sum", "price", "cost", "revenue", "rub", "usd", "eur",
    "сум", "сумм", "стоим", "выручк", "платеж", "баланс", "bonus", "salary",
    "зарплат", "зп", "_zp", "payroll",
)
_PERCENT_HINTS = ("pct", "perc", "rate", "ratio", "share", "процент", "доля")
# Exact name tokens that almost always denote a canonical entity identifier,
# regardless of cardinality. Helpful when a table has only a handful of such
# values (e.g. small dictionary sample).
_ID_EXACT_NAMES = (
    "inn", "kpp", "ogrn", "okato", "oktmo", "snils", "epk_id", "saphr_id",
    "author_login",
)
# Substrings that suggest ID but still need a cardinality check to avoid
# misclassifying low-cardinality coded categories (e.g. `status_id` with 5
# values should be CATEGORY, not ID).
_ID_HINTS = (
    "_id", "uuid", "sha1", "saphr", "epk_", "tid", "_tid",
    "account", "acc_", "_acc_", "client_", "employee_", "person_", "user_",
    "_code", "_num", "номер", "login",
)
_DATE_HINTS = ("date", "_dt", "_dttm", "day", "time", "ts", "timestamp", "дата", "дт", "время")
_FLAG_HINTS = (
    "flag", "flg", "is_", "has_", "active", "enabled", "blocked", "closed", "cancel",
    "призн", "флаг", "индикатор",
)
_CATEGORY_HINTS = (
    "type", "status", "category", "segment", "kind", "group", "class", "channel",
    "priority", "industry", "vertical", "role_name", "_name", "_descr",
    "тип", "статус", "категор", "сегмент", "канал", "вид", "приоритет", "отрасл", "вертикал",
)
# Columns ending with these suffixes are numeric counts/quantities — keep
# them as NUMERIC even when a name hint would push them elsewhere.
_QTY_SUFFIXES = ("_qty", "_cnt", "_val")
# Minimum cardinality for a column to qualify as ID (entity key). Below this,
# even a `_id`-suffixed column is more useful as a CATEGORY.
_ID_MIN_CARDINALITY = 50
# Narrative/PII text columns we want as TEXT regardless of cardinality — they
# should never be treated as categories (cardinality thresholds would miss
# them on small samples).
_TEXT_EXACT_NAMES = ("fio", "_fio", "comment", "infopovod", "description")

_RU_NAME_RE = re.compile(r"[а-яё]", re.IGNORECASE)


def _has_hint(col: str, hints: tuple[str, ...]) -> bool:
    low = col.lower()
    return any(h in low for h in hints)


def _coerce_to_datetime(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if series.dtype == object:
        non_null = series.dropna()
        if non_null.empty:
            return None
        sample = non_null.head(200)
        # Require >90% parseability to accept as date column.
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
        if parsed.notna().mean() >= 0.9:
            return pd.to_datetime(series, errors="coerce", utc=False)
    return None


def _date_or_datetime(col: str, series: pd.Series) -> ColumnRole:
    """Pick DATE vs DATETIME for a datetime64 series.

    Rule: if the column has no sub-day component anywhere in the sample, it's
    a calendar date. Otherwise a timestamp. This sidesteps the corp-schema
    quirk where some `*_dt` columns are actually typed `timestamp` (e.g.
    task_created_dt) — dtype and name agree that it holds a time, so role
    should reflect the data, not the suffix.
    """
    if "_dttm" in col.lower():
        return ColumnRole.DATETIME
    sample = series.dropna()
    if sample.empty:
        return ColumnRole.DATE if _has_hint(col, _DATE_HINTS) else ColumnRole.DATETIME
    try:
        times = sample.dt.time
        # If every value is 00:00:00 → pure calendar dates.
        from datetime import time as dtime
        if all(t == dtime(0, 0) for t in times.head(200)):
            return ColumnRole.DATE
    except Exception:
        pass
    return ColumnRole.DATETIME


def _is_exact_name(col: str, names: tuple[str, ...]) -> bool:
    low = col.lower()
    return any(low == n or low.endswith(n) for n in names)


def _looks_like_id(col: str, n_unique: int, n_rows: int) -> bool:
    """Is `col` an entity-identifier column?

    ID role gets special treatment downstream (picked as entity_col in
    group_anomalies, excluded from dependency scans). We want it only when
    the column plausibly identifies distinct business entities — i.e. has
    enough distinct values to form a cohort.
    """
    if _is_exact_name(col, _ID_EXACT_NAMES):
        return True
    if not _has_hint(col, _ID_HINTS):
        return False
    if n_unique >= _ID_MIN_CARDINALITY:
        return True
    # Categorical dictionaries with `_id` suffix and low cardinality are not
    # entities — fall through so they can be classified as CATEGORY.
    return False


def _infer_role(
    col: str,
    series: pd.Series,
    n_unique: int,
    n_rows: int,
    max_str_len: int | None,
) -> tuple[ColumnRole, pd.Series]:
    """Return (role, possibly-coerced series).

    The coerced series is what the profile stores — e.g. string dates are
    converted to datetime64 so downstream runners don't re-parse.
    """

    if pd.api.types.is_bool_dtype(series):
        return ColumnRole.FLAG, series.astype(object)

    if pd.api.types.is_numeric_dtype(series):
        # Explicit quantity suffixes win over hint tables to keep counts
        # numeric (e.g. staff_qty must not be mistaken for a category).
        if any(col.lower().endswith(s) for s in _QTY_SUFFIXES):
            non_null = series.dropna()
            if not non_null.empty and non_null.nunique() <= 2 and set(non_null.unique()).issubset({0, 1, 0.0, 1.0}):
                return ColumnRole.FLAG, series
            return ColumnRole.NUMERIC, series
        if _has_hint(col, _PERCENT_HINTS):
            return ColumnRole.PERCENT, series
        if _has_hint(col, _MONEY_HINTS):
            return ColumnRole.MONEY, series
        # Binary numeric (0/1) → flag, regardless of name.
        non_null = series.dropna()
        if not non_null.empty and non_null.nunique() <= 2 and set(non_null.unique()).issubset({0, 1, 0.0, 1.0}):
            return ColumnRole.FLAG, series
        if _looks_like_id(col, n_unique, n_rows):
            return ColumnRole.ID, series
        # _id-suffix with low cardinality — treat as CATEGORY.
        if _has_hint(col, _ID_HINTS):
            return ColumnRole.CATEGORY, series
        return ColumnRole.NUMERIC, series

    if pd.api.types.is_datetime64_any_dtype(series):
        return (_date_or_datetime(col, series), series)

    coerced = _coerce_to_datetime(series)
    if coerced is not None:
        return (_date_or_datetime(col, coerced), coerced)

    # Narrative/PII text should never be a category even when sampled small.
    if _is_exact_name(col, _TEXT_EXACT_NAMES):
        if max_str_len is not None and max_str_len > 50:
            return ColumnRole.TEXT_LONG, series
        return ColumnRole.TEXT_SHORT, series

    if _looks_like_id(col, n_unique, n_rows):
        return ColumnRole.ID, series

    if _has_hint(col, _FLAG_HINTS):
        return ColumnRole.FLAG, series

    # Cardinality-based fallback for object columns. Cardinality is the
    # strongest signal: a name-hint like "_name" implies category ONLY when
    # the column actually has few distinct values. `company_name` with 5000
    # unique names is narrative text, not a dimension.
    non_null = series.dropna()
    if non_null.empty:
        return ColumnRole.UNKNOWN, series
    unique_ratio = n_unique / max(len(non_null), 1)
    high_cardinality = n_unique > 500 and unique_ratio > 0.5
    if not high_cardinality and (
        n_unique <= 50 or unique_ratio <= 0.01 or _has_hint(col, _CATEGORY_HINTS)
    ):
        return ColumnRole.CATEGORY, series
    if max_str_len is not None and max_str_len > 50:
        return ColumnRole.TEXT_LONG, series
    return ColumnRole.TEXT_SHORT, series


def _numeric_stats(series: pd.Series) -> dict[str, float] | None:
    if not pd.api.types.is_numeric_dtype(series):
        return None
    non_null = series.dropna()
    if non_null.empty:
        return None
    arr = non_null.to_numpy(dtype=float, copy=False)
    q05, q50, q95 = np.quantile(arr, [0.05, 0.5, 0.95])
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(q50),
        "q05": float(q05),
        "q95": float(q95),
        "skew": float(pd.Series(arr).skew()) if len(arr) > 2 else 0.0,
    }


def profile_dataframe(
    df: pd.DataFrame,
    plan: LoadPlan,
    *,
    table_description: str = "",
) -> tuple[pd.DataFrame, TableProfile]:
    """Profile every column and return (coerced_df, TableProfile).

    The coerced DataFrame may differ from the input (string→datetime), so
    downstream code should always use the returned one.
    """
    log = get_logger()
    columns: dict[str, ColumnProfile] = {}
    n_rows = len(df)
    coerced_df = df.copy(deep=False)

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        try:
            n_unique = int(non_null.nunique())
        except TypeError:
            n_unique = int(non_null.astype(str).nunique())
        max_str_len: int | None = None
        if series.dtype == object:
            try:
                max_str_len = int(non_null.astype(str).str.len().max() or 0)
            except Exception:
                max_str_len = None

        role, new_series = _infer_role(col, series, n_unique, n_rows, max_str_len)
        if new_series is not series:
            coerced_df[col] = new_series
            series = new_series

        top_values: list[tuple] = []
        if role in (ColumnRole.CATEGORY, ColumnRole.FLAG) and not non_null.empty:
            counts = series.value_counts(dropna=True).head(10)
            top_values = [(idx, int(cnt)) for idx, cnt in counts.items()]

        date_range: tuple[str, str] | None = None
        if role in (ColumnRole.DATE, ColumnRole.DATETIME):
            try:
                vals = pd.to_datetime(series, errors="coerce").dropna()
                if not vals.empty:
                    date_range = (vals.min().isoformat(), vals.max().isoformat())
            except Exception:
                date_range = None

        profile = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            role=role,
            n_rows=n_rows,
            n_null=int(series.isna().sum()),
            n_unique=n_unique,
            top_values=top_values,
            numeric_stats=_numeric_stats(series),
            date_range=date_range,
            max_str_len=max_str_len,
        )
        columns[col] = profile
        log.debug("Column %s: role=%s unique=%d null_pct=%.1f", col, role, n_unique, profile.null_pct)

    table_profile = TableProfile(
        schema=plan.schema,
        table=plan.table,
        n_rows=plan.total_rows,
        n_cols=len(columns),
        columns=columns,
        dropped_wide_text=list(plan.dropped_wide_text),
        load_strategy=plan.strategy,
        table_description=table_description,
        where_clause=plan.where_clause,
    )
    return coerced_df, table_profile


def profile_to_brief(profile: TableProfile) -> str:
    """Produce a compact textual digest of the profile for LLM prompts.

    Full per-column detail is too many tokens, so we emit roles + stats only.
    """
    lines = [
        f"Таблица: {profile.schema}.{profile.table}",
        f"Описание: {profile.table_description or '(не задано)'}",
        f"Строк всего: {profile.n_rows}, колонок: {profile.n_cols}",
        f"Стратегия загрузки: {profile.load_strategy}",
        f"Фильтр (WHERE): {profile.where_clause or '—'}",
        f"Откинутые широкие текстовые колонки: {', '.join(profile.dropped_wide_text) or '—'}",
        "",
        "Колонки (имя | роль | уник | null% | доп):",
    ]
    for name, c in profile.columns.items():
        extra = []
        if c.numeric_stats:
            s = c.numeric_stats
            extra.append(f"min={s['min']:.2f} med={s['median']:.2f} max={s['max']:.2f}")
        if c.date_range:
            extra.append(f"даты {c.date_range[0][:10]}..{c.date_range[1][:10]}")
        if c.top_values:
            tops = ", ".join(f"{v}={n}" for v, n in c.top_values[:5])
            extra.append(f"топ: {tops}")
        lines.append(
            f"  {name} | {c.role.value} | uniq={c.n_unique} | null={c.null_pct:.1f}% | "
            + "; ".join(extra)
        )
    return "\n".join(lines)
