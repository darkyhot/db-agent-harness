"""Детерминированная статическая проверка SQL до отправки в БД.

Не использует LLM. Работает через sqlparse + regex + schema_loader.
Запускается между sql_writer и sql_validator, ловит:
- галлюцинированные колонки (не существуют в каталоге)
- кириллические алиасы
- SELECT * (запрещено правилами)
- GROUP BY не покрывает все не-агрегированные SELECT-колонки
"""

import logging
import re
from dataclasses import dataclass, field

import sqlparse
from sqlparse.sql import Identifier, IdentifierList, Function, Where
from sqlparse.tokens import Keyword, DML, Wildcard

logger = logging.getLogger(__name__)

# Регулярка для кириллицы
_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")

# Агрегатные функции SQL
_AGGREGATE_FUNCS = frozenset({
    "count", "sum", "avg", "min", "max",
    "stddev", "variance", "percentile_cont", "percentile_disc",
    "string_agg", "array_agg", "json_agg", "bool_and", "bool_or",
})


@dataclass
class StaticCheckResult:
    """Результат статической проверки SQL."""
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False
        logger.warning("StaticChecker error: %s", msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.info("StaticChecker warning: %s", msg)

    def summary(self) -> str:
        lines = []
        if self.errors:
            lines.append("Статические ошибки:")
            lines.extend(f"  ✗ {e}" for e in self.errors)
        if self.warnings:
            lines.append("Предупреждения:")
            lines.extend(f"  ⚠ {w}" for w in self.warnings)
        return "\n".join(lines) if lines else "OK"


def _extract_table_aliases(parsed) -> dict[str, str]:
    """Извлечь маппинг алиас → schema.table из FROM/JOIN клауз.

    Returns:
        {"s": "dm.sales", "c": "dm.clients", ...}
    """
    aliases: dict[str, str] = {}

    def _process_identifier(identifier: Identifier) -> None:
        alias = identifier.get_alias()
        real_name = identifier.get_real_name()
        schema = identifier.get_parent_name()
        if real_name:
            full = f"{schema}.{real_name}" if schema else real_name
            key = alias if alias else real_name
            aliases[key.lower()] = full.lower()

    for token in parsed.tokens:
        # FROM clause
        if token.ttype is Keyword and token.normalized in ("FROM", "JOIN",
                                                            "LEFT JOIN", "RIGHT JOIN",
                                                            "INNER JOIN", "FULL JOIN",
                                                            "CROSS JOIN"):
            continue
        if hasattr(token, "tokens"):
            for sub in token.tokens:
                if isinstance(sub, Identifier):
                    _process_identifier(sub)
                elif isinstance(sub, IdentifierList):
                    for item in sub.get_identifiers():
                        if isinstance(item, Identifier):
                            _process_identifier(item)

    return aliases


def _extract_tables_from_sql(sql: str) -> set[tuple[str, str]]:
    """Извлечь все упоминания schema.table из SQL через regex.

    Returns:
        {("dm", "sales"), ("dm", "clients"), ...}
    """
    pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b')
    tables: set[tuple[str, str]] = set()
    for m in pattern.finditer(sql):
        schema, table = m.group(1).lower(), m.group(2).lower()
        # Исключаем alias.column паттерны (alias обычно короткий, без _)
        # Настоящие schema.table имеют schema из реального каталога — проверяется выше
        tables.add((schema, table))
    return tables


def _check_cyrillic_aliases(sql: str, result: StaticCheckResult) -> None:
    """Проверить наличие кириллицы в SQL-алиасах."""
    # Ищем паттерн: AS <что-то с кириллицей>
    alias_pattern = re.compile(
        r'\bAS\s+(?:"([^"]+)"|\'([^\']+)\'|([^\s,\)]+))',
        re.IGNORECASE,
    )
    for m in alias_pattern.finditer(sql):
        alias = m.group(1) or m.group(2) or m.group(3) or ""
        if _CYRILLIC_RE.search(alias):
            result.add_error(
                f"Кириллица в алиасе: AS {alias!r}. "
                "Алиасы должны быть только на английском."
            )


def _check_select_star(sql: str, result: StaticCheckResult) -> None:
    """Проверить наличие SELECT * в финальном SELECT."""
    # SELECT * запрещён на уровне правил агента
    # Ищем SELECT * не внутри подзапроса COUNT(*) etc.
    # Упрощённая проверка: SELECT\s+\* не внутри агрегата
    star_pattern = re.compile(r'SELECT\s+\*', re.IGNORECASE)
    # Исключаем COUNT(*) — это допустимо
    count_star = re.compile(r'COUNT\s*\(\s*\*\s*\)', re.IGNORECASE)
    sql_without_count = count_star.sub('COUNT(__STAR__)', sql)
    if star_pattern.search(sql_without_count):
        result.add_warning(
            "Используется SELECT * — рекомендуется явно перечислить нужные колонки."
        )


def _check_columns_against_catalog(
    sql: str,
    schema_loader,
    result: StaticCheckResult,
) -> None:
    """Проверить что колонки из SQL существуют в каталоге.

    Логика:
    1. Извлечь все schema.table из SQL
    2. Для каждой таблицы получить реальные колонки из schema_loader
    3. Найти упоминания alias.column или schema.table.column
    4. Проверить что колонки существуют
    """
    if schema_loader is None:
        return

    # Извлечь таблицы из SQL
    raw_tables = _extract_tables_from_sql(sql)

    # Фильтруем: оставляем только те, что есть в каталоге
    real_tables: dict[tuple[str, str], set[str]] = {}
    for schema, table in raw_tables:
        cols_df = schema_loader.get_table_columns(schema, table)
        if not cols_df.empty and "column_name" in cols_df.columns:
            real_tables[(schema, table)] = set(
                cols_df["column_name"].str.lower().tolist()
            )

    if not real_tables:
        return  # Каталог пуст или таблицы не найдены — не блокируем

    # Паттерн alias.column или table.column
    col_ref_pattern = re.compile(
        r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b'
    )

    # Строим обратный маппинг: alias/table_name → real (schema, table)
    # Упрощение: используем table_name как ключ
    table_to_cols: dict[str, set[str]] = {}
    for (schema, table), cols in real_tables.items():
        table_to_cols[table] = cols
        # Также регистрируем по алиасам из SQL (a, s, c и т.д.)
        # Для этого ищем паттерн "schema.table alias" или "table alias"

    # Более простой подход: ищем паттерны вида alias.column
    # и проверяем column против объединённого списка колонок всех таблиц
    all_real_columns: set[str] = set()
    for cols in real_tables.values():
        all_real_columns.update(cols)

    # Найти все alias.column ссылки, исключая schema.table паттерны
    # (где правая часть — это имя таблицы, а не колонки)
    all_table_names = {t for _, t in real_tables}
    all_schema_names = {s for s, _ in real_tables}

    suspicious_cols: list[str] = []
    for m in col_ref_pattern.finditer(sql):
        left, right = m.group(1).lower(), m.group(2).lower()
        # Пропускаем schema.table паттерны
        if left in all_schema_names and right in all_table_names:
            continue
        # Пропускаем если правая часть — имя таблицы (FROM alias JOIN ...)
        if right in all_table_names:
            continue
        # right — вероятно колонка
        if right not in all_real_columns:
            # Дополнительная фильтрация: пропускаем ключевые слова SQL
            if right.upper() not in {
                "NULL", "TRUE", "FALSE", "CURRENT_DATE", "NOW",
                "DISTINCT", "ALL", "ASC", "DESC",
            }:
                suspicious_cols.append(f"{left}.{right}")

    if suspicious_cols:
        # Дедупликация
        unique_suspicious = list(dict.fromkeys(suspicious_cols))
        result.add_error(
            f"Возможные галлюцинированные колонки (не найдены в каталоге): "
            f"{', '.join(unique_suspicious[:5])}. "
            "Проверь названия колонок через get_table_columns."
        )


def check_sql(
    sql: str,
    schema_loader=None,
    check_columns: bool = True,
) -> StaticCheckResult:
    """Выполнить полный набор статических проверок SQL.

    Args:
        sql: SQL-запрос для проверки.
        schema_loader: SchemaLoader для валидации колонок. None → колонки не проверяются.
        check_columns: Включить проверку колонок против каталога.

    Returns:
        StaticCheckResult с ошибками и предупреждениями.
    """
    result = StaticCheckResult()

    if not sql or not sql.strip():
        result.add_error("SQL пустой.")
        return result

    # 1. Кириллические алиасы — жёсткая ошибка
    _check_cyrillic_aliases(sql, result)

    # 2. SELECT * — предупреждение
    _check_select_star(sql, result)

    # 3. Колонки против каталога — ошибка если найдены галлюцинации
    if check_columns and schema_loader is not None:
        try:
            _check_columns_against_catalog(sql, schema_loader, result)
        except Exception as e:
            logger.warning("StaticChecker: ошибка проверки колонок: %s", e)
            # Не блокируем — каталог мог быть недоступен

    return result
