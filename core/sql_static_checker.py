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


def _extract_select_columns(sql: str) -> list[str]:
    """Извлечь колонки из SELECT-клаузы (без агрегатных функций).

    Возвращает список имён колонок, которые НЕ обёрнуты в агрегатную функцию.
    Только верхний уровень SELECT (не CTEs, не подзапросы).
    Упрощённый парсер через regex — работает для стандартных случаев.
    """
    # Убираем CTE-часть (WITH ... AS (...))
    sql_no_cte = re.sub(r"(?i)\bWITH\b.+?\)\s*(?=SELECT)", "", sql, flags=re.DOTALL)
    # Берём первый верхний SELECT
    select_match = re.search(r"(?i)\bSELECT\b(.+?)\bFROM\b", sql_no_cte, re.DOTALL)
    if not select_match:
        return []

    select_body = select_match.group(1)

    # Убираем вложенные скобки (подзапросы, агрегаты)
    depth = 0
    tokens: list[str] = []
    current = ""
    for ch in select_body:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            tokens.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        tokens.append(current.strip())

    non_agg_cols: list[str] = []
    _agg_pattern = re.compile(
        r"(?i)^(COUNT|SUM|AVG|MIN|MAX|STDDEV|VARIANCE|STRING_AGG|ARRAY_AGG"
        r"|PERCENTILE_CONT|PERCENTILE_DISC|BOOL_AND|BOOL_OR)\s*\(",
    )

    for token in tokens:
        token = token.strip()
        if not token:
            continue
        # Пропускаем агрегаты
        if _agg_pattern.match(token):
            continue
        # Убираем алиас: col AS alias → col
        alias_match = re.match(r"(?i)^(.+?)\s+AS\s+\S+$", token)
        if alias_match:
            token = alias_match.group(1).strip()
        # Убираем квалификаторы таблиц: t.col → col
        if "." in token:
            token = token.rsplit(".", 1)[-1]
        # Только простые идентификаторы
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", token):
            non_agg_cols.append(token.lower())

    return non_agg_cols


def _extract_group_by_columns(sql: str) -> list[str]:
    """Извлечь колонки из GROUP BY клаузы."""
    gb_match = re.search(r"(?i)\bGROUP\s+BY\b(.+?)(?:\bHAVING\b|\bORDER\b|\bLIMIT\b|$)", sql, re.DOTALL)
    if not gb_match:
        return []

    gb_body = gb_match.group(1).strip()
    cols: list[str] = []
    for col in gb_body.split(","):
        col = col.strip()
        # Убираем квалификатор таблицы
        if "." in col:
            col = col.rsplit(".", 1)[-1]
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", col):
            cols.append(col.lower())
    return cols


def _strip_ctes(sql: str) -> str:
    """Убрать CTE-блок (WITH ... AS (...)) и вернуть финальный SELECT.

    Упрощённый подход: убираем всё до последнего верхнего SELECT,
    который следует после закрытия всех CTE-скобок.
    """
    stripped = sql.strip()
    if not re.match(r"(?i)^\s*WITH\b", stripped):
        return stripped

    # Сканируем до нулевой глубины скобок, потом ищем SELECT
    depth = 0
    i = 0
    in_cte_body = False
    while i < len(stripped):
        ch = stripped[i]
        if ch == "(":
            depth += 1
            in_cte_body = True
        elif ch == ")":
            depth -= 1
            if depth == 0 and in_cte_body:
                # После закрытия CTE идёт запятая или финальный SELECT
                rest = stripped[i + 1:].lstrip()
                if rest.upper().startswith("SELECT"):
                    return rest
        i += 1
    return stripped


def _check_group_by_completeness(sql: str, result: StaticCheckResult) -> None:
    """Проверить что все не-агрегированные SELECT-колонки присутствуют в GROUP BY.

    Это самая частая ошибка LLM: добавить колонку в SELECT, но забыть в GROUP BY.
    Работает только когда в SQL есть GROUP BY на уровне финального SELECT.
    """
    # Работаем только с финальным SELECT (без CTE)
    outer_sql = _strip_ctes(sql)

    # Проверяем только если в финальном SELECT есть GROUP BY
    if not re.search(r"(?i)\bGROUP\s+BY\b", outer_sql):
        return

    select_cols = set(_extract_select_columns(outer_sql))
    group_by_cols = set(_extract_group_by_columns(outer_sql))

    if not select_cols:
        return  # Не смогли распарсить — не блокируем

    missing = select_cols - group_by_cols
    # Исключаем известные константы и псевдоколонки
    _KNOWN_CONSTANTS = {"true", "false", "null", "current_date", "current_timestamp"}
    missing -= _KNOWN_CONSTANTS

    if missing:
        result.add_warning(
            f"GROUP BY completeness: колонки {sorted(missing)} присутствуют в SELECT, "
            "но отсутствуют в GROUP BY. Возможна ошибка агрегации."
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

    # 4. GROUP BY completeness — предупреждение
    try:
        _check_group_by_completeness(sql, result)
    except Exception as e:
        logger.warning("StaticChecker: ошибка проверки GROUP BY: %s", e)

    return result
