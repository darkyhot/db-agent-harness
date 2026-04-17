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
    # Если soft-fix произвёл изменения — здесь лежит исправленный SQL.
    # Поле заполняется только при успешном auto-fix (без ошибок).
    fixed_sql: str | None = None

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


# --- Direction 3: soft-fix транслитерация кириллических алиасов ---

# Упрощённая схема транслитерации ГОСТ-подобная.
_TRANSLIT_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def _transliterate_identifier(alias: str) -> str:
    """Транслитерировать кириллицу в латиницу + очистить небезопасные символы."""
    out_chars: list[str] = []
    for ch in alias:
        lower = ch.lower()
        if lower in _TRANSLIT_MAP:
            repl = _TRANSLIT_MAP[lower]
            if ch.isupper() and repl:
                repl = repl[0].upper() + repl[1:]
            out_chars.append(repl)
        else:
            out_chars.append(ch)
    result = "".join(out_chars)
    result = re.sub(r"[^A-Za-z0-9_]+", "_", result).strip("_")
    if not result or not re.match(r"[A-Za-z_]", result[0]):
        result = "alias_" + result
    return result


def _soft_fix_cyrillic_aliases(sql: str) -> tuple[str, list[str]]:
    """Применить транслитерацию ко всем кириллическим AS-алиасам.

    Returns:
        (исправленный_sql, список_замен).
    """
    replacements: list[str] = []
    alias_pattern = re.compile(
        r'\bAS\s+(?:"([^"]+)"|\'([^\']+)\'|([^\s,\)]+))',
        re.IGNORECASE,
    )

    def _replace(match: re.Match) -> str:
        alias_raw = match.group(1) or match.group(2) or match.group(3) or ""
        if not _CYRILLIC_RE.search(alias_raw):
            return match.group(0)
        fixed = _transliterate_identifier(alias_raw)
        replacements.append(f"{alias_raw} → {fixed}")
        return f"AS {fixed}"

    new_sql = alias_pattern.sub(_replace, sql)
    return new_sql, replacements


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
        result.add_error(
            "SELECT * запрещён. Явно перечисли нужные колонки в SELECT."
        )


def _build_alias_to_table_map(sql: str, real_tables: dict[tuple[str, str], set[str]]) -> dict[str, tuple[str, str]]:
    """Построить маппинг алиас/имя_таблицы → (schema, table).

    Парсит FROM и JOIN клаузы для извлечения явных алиасов.
    Также регистрирует короткое имя таблицы как алиас (без schema-префикса).

    Returns:
        {"s": ("dm", "sales"), "sales": ("dm", "sales"), ...}
    """
    alias_map: dict[str, tuple[str, str]] = {}

    # Регистрируем table_name (без schema) как implicit alias
    for (schema, table) in real_tables:
        alias_map[table.lower()] = (schema, table)

    # Ищем паттерны: schema.table [AS] alias или просто schema.table alias
    # Поддерживаем: JOIN dm.sales s, FROM dm.sales AS s, FROM dm.sales s
    from_join_pat = re.compile(
        r'(?:FROM|JOIN)\s+'
        r'([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)'
        r'(?:\s+AS\s+|\s+)([a-zA-Z_][a-zA-Z0-9_]*)',
        re.IGNORECASE,
    )
    for m in from_join_pat.finditer(sql):
        schema_part = m.group(1).lower()
        table_part = m.group(2).lower()
        alias_part = m.group(3).lower()
        # Проверяем что это реальная таблица из каталога
        if (schema_part, table_part) in real_tables:
            # Пропускаем если alias — ключевое слово SQL
            if alias_part.upper() not in {
                'WHERE', 'ON', 'SET', 'JOIN', 'LEFT', 'RIGHT',
                'INNER', 'OUTER', 'CROSS', 'FULL', 'GROUP', 'ORDER',
                'HAVING', 'LIMIT', 'UNION', 'EXCEPT', 'INTERSECT',
            }:
                alias_map[alias_part] = (schema_part, table_part)

    return alias_map


def _check_columns_against_catalog(
    sql: str,
    schema_loader,
    result: StaticCheckResult,
) -> None:
    """Проверить что колонки из SQL существуют в каталоге.

    Логика:
    1. Извлечь все schema.table из SQL
    2. Построить маппинг alias → (schema, table) с учётом явных алиасов
    3. Для каждой ссылки alias.column найти реальную таблицу по алиасу
    4. Проверить column против колонок КОНКРЕТНОЙ таблицы (не всех сразу)
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

    # Строим alias → (schema, table) маппинг с учётом явных алиасов из SQL
    alias_to_table = _build_alias_to_table_map(sql, real_tables)

    # Объединённый список всех реальных колонок (для fallback)
    all_real_columns: set[str] = set()
    for cols in real_tables.values():
        all_real_columns.update(cols)

    all_schema_names = {s for s, _ in real_tables}
    all_table_names = {t for _, t in real_tables}

    # SQL-ключевые слова, которые могут выглядеть как колонки
    _SQL_KEYWORDS = {
        "NULL", "TRUE", "FALSE", "CURRENT_DATE", "CURRENT_TIMESTAMP",
        "NOW", "DISTINCT", "ALL", "ASC", "DESC", "OVER", "PARTITION",
        "ROWS", "RANGE", "PRECEDING", "FOLLOWING", "UNBOUNDED", "CURRENT",
    }

    col_ref_pattern = re.compile(
        r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b'
    )

    suspicious_cols: list[str] = []
    for m in col_ref_pattern.finditer(sql):
        left, right = m.group(1).lower(), m.group(2).lower()

        # Пропускаем schema.table паттерны
        if left in all_schema_names and right in all_table_names:
            continue
        # Пропускаем если правая часть — имя таблицы
        if right in all_table_names:
            continue
        # Пропускаем SQL-ключевые слова
        if right.upper() in _SQL_KEYWORDS:
            continue

        # Пытаемся найти таблицу по алиасу (точная проверка)
        tbl_key = alias_to_table.get(left)
        if tbl_key is not None:
            # Проверяем против колонок конкретной таблицы
            table_cols = real_tables.get(tbl_key, set())
            if right not in table_cols:
                suspicious_cols.append(f"{left}.{right}")
        else:
            # Алиас не распознан → fallback: проверяем против всех колонок
            if right not in all_real_columns:
                suspicious_cols.append(f"{left}.{right}")

    if suspicious_cols:
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


def _has_aggregation(sql: str) -> bool:
    """Проверить что SQL содержит агрегатные функции (кроме COUNT(*))."""
    # Убираем COUNT(*) чтобы не путать его с «настоящей» агрегацией
    sql_no_count_star = re.sub(r'COUNT\s*\(\s*\*\s*\)', 'COUNT_STAR', sql, flags=re.I)
    agg_pat = re.compile(
        r'\b(COUNT|SUM|AVG|MIN|MAX|STDDEV|VARIANCE|STRING_AGG|ARRAY_AGG'
        r'|PERCENTILE_CONT|PERCENTILE_DISC|BOOL_AND|BOOL_OR)\s*\(',
        re.I,
    )
    return bool(agg_pat.search(sql_no_count_star))


def _check_group_by_completeness(sql: str, result: StaticCheckResult) -> None:
    """Проверить что все не-агрегированные SELECT-колонки присутствуют в GROUP BY.

    Это самая частая ошибка LLM: добавить колонку в SELECT, но забыть в GROUP BY.
    Работает только когда в SQL есть GROUP BY на уровне финального SELECT.

    Severity:
    - ERROR если запрос содержит агрегатные функции (SUM, AVG, MIN, MAX, ...)
      — пропущенный GROUP BY гарантированно сломает запрос в PostgreSQL/Greenplum
    - WARNING если агрегации нет (например, только GROUP BY без агрегата)
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

    _KNOWN_CONSTANTS = {"true", "false", "null", "current_date", "current_timestamp"}

    missing = select_cols - group_by_cols
    # Исключаем известные константы и псевдоколонки
    missing -= _KNOWN_CONSTANTS

    if missing:
        msg = (
            f"GROUP BY completeness: колонки {sorted(missing)} присутствуют в SELECT, "
            "но отсутствуют в GROUP BY."
        )
        if _has_aggregation(outer_sql):
            # Агрегация есть → пропущенные колонки в GROUP BY = жёсткая ошибка SQL
            result.add_error(msg + " При наличии агрегатов это вызовет ошибку БД.")
        else:
            result.add_warning(msg + " Возможна ошибка агрегации.")

    # Симметричная проверка: GROUP BY ⊆ SELECT (включая алиасы).
    # PostgreSQL не упадёт на «лишних» колонках в GROUP BY, но это типичная ошибка
    # планировщика — означает, что filter-колонка утекла в GROUP BY и пропала из SELECT.
    select_aliases = {a.lower() for a in _extract_select_aliases(outer_sql)}
    extra = group_by_cols - select_cols - _KNOWN_CONSTANTS - select_aliases
    if extra:
        msg = (
            f"GROUP BY содержит колонки {sorted(extra)}, отсутствующие в SELECT — "
            "это часто признак того, что filter-колонка ошибочно утекла в GROUP BY."
        )
        result.add_warning(msg)


def _extract_select_aliases(sql: str) -> set[str]:
    """Извлечь алиасы, объявленные через AS в SELECT (например SUM(x) AS cnt → cnt)."""
    aliases: set[str] = set()
    for m in re.finditer(r'\bAS\s+([a-zA-Z_][a-zA-Z0-9_]*)\b', sql, re.IGNORECASE):
        aliases.add(m.group(1).lower())
    return aliases


def _check_order_by_aliases(sql: str, result: StaticCheckResult) -> None:
    """Проверить что ORDER BY ссылается только на колонки/алиасы из SELECT.

    Распространённая ошибка LLM: ORDER BY cnt, но cnt не объявлен в SELECT.
    Работает только для финального SELECT (без CTE).
    """
    outer_sql = _strip_ctes(sql)

    ob_match = re.search(
        r"(?i)\bORDER\s+BY\b(.+?)(?:\bLIMIT\b|;|$)", outer_sql, re.DOTALL
    )
    if not ob_match:
        return

    ob_cols: list[str] = []
    for part in ob_match.group(1).split(","):
        col = re.sub(r"\s+(ASC|DESC)\b", "", part.strip(), flags=re.IGNORECASE).strip()
        # Убираем квалификатор таблицы
        if "." in col:
            col = col.rsplit(".", 1)[-1]
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", col):
            ob_cols.append(col.lower())

    if not ob_cols:
        return

    defined = set(_extract_select_columns(outer_sql)) | _extract_select_aliases(outer_sql)
    missing = [c for c in ob_cols if c not in defined]
    if missing:
        result.add_error(
            f"ORDER BY ссылается на неопределённые имена: {missing}. "
            "Алиас или колонка должны быть объявлены в SELECT."
        )


def _check_distinct_in_final_select(sql: str, result: StaticCheckResult) -> None:
    """SELECT DISTINCT в финальной проекции запрещён правилами агента.

    CTE и подзапросы с DISTINCT разрешены (они часто нужны для dim-агрегации).
    Проверяем только самый внешний SELECT после CTE-блока.
    """
    outer = _strip_ctes(sql)
    # SELECT DISTINCT могут сопровождать whitespace + возможно комментарии.
    if re.match(r"(?i)^\s*SELECT\s+DISTINCT\b", outer):
        result.add_error(
            "SELECT DISTINCT в финальной проекции запрещён. "
            "Если нужно устранить дубли — используй GROUP BY или CTE с агрегатами."
        )


def _dtype_bucket(dtype: str) -> str:
    """Свести pg/gp dtype к укрупнённому bucket для сравнения с литералом."""
    d = (dtype or "").lower().strip()
    if not d:
        return ""
    if d.startswith("int") or d in {"bigint", "smallint", "integer", "serial", "bigserial"}:
        return "int"
    if "numeric" in d or "decimal" in d or "float" in d or "double" in d or "real" in d:
        return "num"
    if "char" in d or d == "text":
        return "text"
    if "date" in d or "time" in d:
        return "date"
    if "bool" in d:
        return "bool"
    if "json" in d:
        return "json"
    return d


def _literal_bucket(literal: str) -> str:
    """Определить тип литерала из WHERE-клаузы."""
    lit = literal.strip()
    if not lit:
        return ""
    if lit.lower() in {"true", "false"}:
        return "bool"
    if lit.lower() == "null":
        return ""
    # Строка в одинарных кавычках — может быть text или date
    if lit.startswith("'") and lit.endswith("'"):
        inner = lit[1:-1]
        # ISO-подобная дата YYYY-MM-DD или timestamp
        if re.match(r"^\d{4}-\d{2}-\d{2}(\s\d{2}:\d{2}:\d{2})?$", inner):
            return "date"
        return "text"
    # Числа
    if re.match(r"^-?\d+$", lit):
        return "int"
    if re.match(r"^-?\d+\.\d+$", lit):
        return "num"
    return ""


def _check_type_compatibility(
    sql: str, schema_loader, result: StaticCheckResult
) -> None:
    """Быстрая проверка типов в WHERE: dtype колонки vs тип литерала.

    Работает на простых паттернах `<col> <op> <literal>`, где col — qualified
    alias.col или schema.table.col. Не пытаемся парсить подзапросы.
    """
    if schema_loader is None:
        return

    raw_tables = _extract_tables_from_sql(sql)
    real_tables: dict[tuple[str, str], dict[str, str]] = {}
    for schema, table in raw_tables:
        cols_df = schema_loader.get_table_columns(schema, table)
        if cols_df.empty:
            continue
        real_tables[(schema, table)] = {
            str(r["column_name"]).lower(): str(r.get("dType", "") or "")
            for _, r in cols_df.iterrows()
        }

    if not real_tables:
        return

    alias_to_table = _build_alias_to_table_map(sql, real_tables)

    # <alias>.<col> <op> <literal>
    where_match = re.search(
        r"(?i)\bWHERE\b(.+?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|\bHAVING\b|;|$)",
        sql, re.DOTALL,
    )
    if not where_match:
        return
    where_clause = where_match.group(1)

    pat = re.compile(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*"
        r"(=|!=|<>|<=|>=|<|>)\s*"
        r"(\'[^\']*\'|-?\d+(?:\.\d+)?|TRUE|FALSE|NULL)",
        re.IGNORECASE,
    )
    issues: list[str] = []
    for m in pat.finditer(where_clause):
        alias = m.group(1).lower()
        col = m.group(2).lower()
        literal = m.group(4)
        tbl_key = alias_to_table.get(alias)
        if not tbl_key:
            continue
        dtype = real_tables.get(tbl_key, {}).get(col, "")
        col_b = _dtype_bucket(dtype)
        lit_b = _literal_bucket(literal)
        if not col_b or not lit_b:
            continue
        # Допустимые совместимости:
        compatible_pairs = {
            ("int", "int"), ("int", "num"),
            ("num", "int"), ("num", "num"),
            ("text", "text"),
            ("date", "date"), ("date", "text"),  # '2024-01-01' → date
            ("bool", "bool"),
        }
        if (col_b, lit_b) not in compatible_pairs:
            issues.append(
                f"{alias}.{col} ({dtype}) {m.group(3)} {literal} — несовместимые типы ({col_b} vs {lit_b})"
            )

    if issues:
        result.add_error(
            "Несовместимые типы в WHERE: "
            + "; ".join(issues[:3])
            + (". Приведи литерал к типу колонки." if len(issues) <= 3 else f" (+ещё {len(issues)-3}).")
        )


def _check_allowed_tables(
    sql: str,
    allowed_tables: list[str],
    result: StaticCheckResult,
) -> None:
    """Проверить, что все таблицы в FROM/JOIN присутствуют в белом списке allowed_tables.

    Галлюцинированная таблица — немедленная критическая ошибка.
    Схема нормализуется в lowercase для сравнения.

    Args:
        sql: SQL-запрос.
        allowed_tables: Список разрешённых "schema.table" (из table_resolver).
        result: StaticCheckResult для записи ошибок.
    """
    if not allowed_tables:
        return

    allowed_lower = {t.lower() for t in allowed_tables}

    # Regex: FROM/JOIN schema.table [alias]
    from_join_pat = re.compile(
        r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)',
        re.IGNORECASE,
    )
    found_tables: list[str] = []
    for m in from_join_pat.finditer(sql):
        schema_part = m.group(1).lower()
        table_part = m.group(2).lower()
        full = f"{schema_part}.{table_part}"
        if full not in allowed_lower:
            found_tables.append(f"{m.group(1)}.{m.group(2)}")

    if found_tables:
        unique = list(dict.fromkeys(found_tables))
        result.add_error(
            f"SQL содержит таблицы, не входящие в разрешённый список: "
            f"{', '.join(unique)}. "
            f"Разрешены только: {', '.join(sorted(allowed_tables))}. "
            "Исправь SQL — используй только указанные таблицы."
        )


def check_sql(
    sql: str,
    schema_loader=None,
    check_columns: bool = True,
    allowed_tables: list[str] | None = None,
    *,
    auto_fix_cyrillic: bool = True,
) -> StaticCheckResult:
    """Выполнить полный набор статических проверок SQL.

    Args:
        sql: SQL-запрос для проверки.
        schema_loader: SchemaLoader для валидации колонок. None → колонки не проверяются.
        check_columns: Включить проверку колонок против каталога.
        allowed_tables: Белый список "schema.table". None → проверка таблиц пропускается.
        auto_fix_cyrillic: Если True — кириллические AS-алиасы транслитерируются и
            сохраняются в `result.fixed_sql` (без ошибки). Если False — возвращается
            ошибка как прежде.

    Returns:
        StaticCheckResult с ошибками и предупреждениями.
    """
    result = StaticCheckResult()

    if not sql or not sql.strip():
        result.add_error("SQL пустой.")
        return result

    # Soft-fix: транслитерируем кириллицу до остальных проверок.
    if auto_fix_cyrillic:
        fixed_sql, replacements = _soft_fix_cyrillic_aliases(sql)
        if replacements:
            result.fixed_sql = fixed_sql
            result.add_warning(
                "Кириллические алиасы транслитерированы: "
                + ", ".join(replacements[:5])
                + (f" (+ещё {len(replacements) - 5})" if len(replacements) > 5 else "")
            )
            sql = fixed_sql  # все дальнейшие проверки работают с исправленной версией

    # 0. Белый список таблиц — критическая проверка (до всего остального)
    if allowed_tables:
        try:
            _check_allowed_tables(sql, allowed_tables, result)
        except Exception as e:
            logger.warning("StaticChecker: ошибка проверки allowed_tables: %s", e)

    # 1. Кириллические алиасы — жёсткая ошибка, если остались
    _check_cyrillic_aliases(sql, result)

    # 2. SELECT * — жёсткая ошибка (запрещено правилами агента)
    _check_select_star(sql, result)

    # 2a. SELECT DISTINCT в финальной проекции — жёсткая ошибка
    try:
        _check_distinct_in_final_select(sql, result)
    except Exception as e:
        logger.warning("StaticChecker: ошибка проверки DISTINCT: %s", e)

    # 3. Колонки против каталога — ошибка если найдены галлюцинации
    if check_columns and schema_loader is not None:
        try:
            _check_columns_against_catalog(sql, schema_loader, result)
        except Exception as e:
            logger.warning("StaticChecker: ошибка проверки колонок: %s", e)
            # Не блокируем — каталог мог быть недоступен

    # 3a. Совместимость типов в WHERE — ошибка при явном несоответствии.
    # Независима от check_columns: type-check опирается только на dtype,
    # не на полный список колонок.
    if schema_loader is not None:
        try:
            _check_type_compatibility(sql, schema_loader, result)
        except Exception as e:
            logger.warning("StaticChecker: ошибка type-check: %s", e)

    # 4. GROUP BY completeness — предупреждение/ошибка
    try:
        _check_group_by_completeness(sql, result)
    except Exception as e:
        logger.warning("StaticChecker: ошибка проверки GROUP BY: %s", e)

    # 5. ORDER BY алиасы — жёсткая ошибка
    try:
        _check_order_by_aliases(sql, result)
    except Exception as e:
        logger.warning("StaticChecker: ошибка проверки ORDER BY: %s", e)

    return result
