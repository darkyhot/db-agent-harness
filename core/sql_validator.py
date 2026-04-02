"""Валидация SQL-запросов: синтаксис, EXPLAIN, проверка JOIN-ов."""

import logging
import re
from enum import Enum
from typing import Any

import sqlparse

logger = logging.getLogger(__name__)


class SQLMode(Enum):
    """Режим SQL-запроса."""
    READ = "READ"
    WRITE = "WRITE"
    DDL = "DDL"


class ValidationResult:
    """Результат валидации SQL-запроса."""

    def __init__(self, is_valid: bool, mode: SQLMode) -> None:
        self.is_valid: bool = is_valid
        self.mode: SQLMode = mode
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.needs_confirmation: bool = False
        self.confirmation_message: str = ""
        self.explain_plan: str = ""
        self.join_checks: list[dict[str, Any]] = []
        self.rewrite_suggestions: list[str] = []
        self.multiplication_factor: float = 1.0

    def add_warning(self, msg: str) -> None:
        """Добавить предупреждение."""
        self.warnings.append(msg)
        logger.warning("SQL validation warning: %s", msg)

    def add_error(self, msg: str) -> None:
        """Добавить ошибку и пометить как невалидный."""
        self.errors.append(msg)
        self.is_valid = False
        logger.error("SQL validation error: %s", msg)

    def require_confirmation(self, msg: str) -> None:
        """Запросить подтверждение пользователя."""
        self.needs_confirmation = True
        self.confirmation_message = msg

    def summary(self) -> str:
        """Текстовое резюме валидации."""
        lines = [f"Режим: {self.mode.value}", f"Валидно: {'да' if self.is_valid else 'нет'}"]
        if self.errors:
            lines.append("Ошибки:")
            for e in self.errors:
                lines.append(f"  вњ— {e}")
        if self.warnings:
            lines.append("Предупреждения:")
            for w in self.warnings:
                lines.append(f"  вљ  {w}")
        if self.needs_confirmation:
            lines.append(f"Требуется подтверждение: {self.confirmation_message}")
        if self.join_checks:
            lines.append("Проверка JOIN-ов:")
            for jc in self.join_checks:
                cardinality = jc.get("cardinality", "unknown")
                join_expr = jc.get("join", f"{jc.get('table', '?')}.({jc.get('columns', '?')})")
                status = "вњ“ safe" if jc.get("is_safe") else "вњ— risky"
                lines.append(f"  {join_expr}: {cardinality} ({status})")
            if self.multiplication_factor > 1.0:
                lines.append(f"  Multiplication factor: {self.multiplication_factor:.1f}x")
        if self.rewrite_suggestions:
            lines.append("Рекомендации по переписыванию:")
            for s in self.rewrite_suggestions:
                lines.append(f"  {s}")
        return "\n".join(lines)


def detect_mode(sql: str) -> SQLMode:
    """Определить режим SQL-запроса.

    Поддерживает CTE: ``WITH cte AS (...) INSERT INTO ...`` определяется как WRITE,
    а не как READ.

    Args:
        sql: SQL-запрос.

    Returns:
        SQLMode (READ, WRITE или DDL).
    """
    normalized = sql.strip().upper()
    # Убираем комментарии в начале
    for line in normalized.split("\n"):
        line = line.strip()
        if line and not line.startswith("--"):
            normalized = line
            break

    # Если запрос начинается с WITH (CTE), ищем основной statement после CTE
    if normalized.startswith("WITH"):
        # Найти основной statement после всех CTE-определений.
        # Ищем последний закрывающий ')' CTE, за которым идёт ключевое слово.
        main_stmt = re.search(
            r'\)\s*(SELECT|INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE)\b',
            normalized,
        )
        if main_stmt:
            keyword = main_stmt.group(1)
            if keyword in ("CREATE", "ALTER", "DROP", "TRUNCATE"):
                return SQLMode.DDL
            if keyword in ("INSERT", "UPDATE", "DELETE", "MERGE"):
                return SQLMode.WRITE
            return SQLMode.READ

    if normalized.startswith(("CREATE", "ALTER", "DROP", "TRUNCATE")):
        return SQLMode.DDL
    if normalized.startswith(("INSERT", "UPDATE", "DELETE", "MERGE")):
        return SQLMode.WRITE
    return SQLMode.READ


def _build_alias_map(sql: str) -> dict[str, tuple[str, str]]:
    """Построить маппинг alias → (schema, table) из FROM и JOIN.

    Обрабатывает:
    - FROM schema.table alias
    - FROM schema.table AS alias
    - JOIN schema.table alias ON ...
    - FROM schema.table (без alias → table name как alias)
    """
    alias_map: dict[str, tuple[str, str]] = {}

    _SQL_KEYWORDS = frozenset((
        "ON", "WHERE", "SET", "LEFT", "RIGHT", "INNER", "OUTER",
        "FULL", "CROSS", "NATURAL", "JOIN", "GROUP", "ORDER",
        "HAVING", "LIMIT", "UNION", "EXCEPT", "INTERSECT",
    ))

    # Паттерн: schema.table с опциональным alias (AS keyword опционален)
    table_ref = re.compile(
        r'(?:FROM|JOIN)\s+'
        r'["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?'
        r'(?:\s+(?:AS\s+)?(\w+))?',
        re.IGNORECASE,
    )

    for m in table_ref.finditer(sql):
        schema, table = m.group(1), m.group(2)
        alias = m.group(3)
        if alias and alias.upper() not in _SQL_KEYWORDS:
            alias_map[alias.lower()] = (schema, table)
        alias_map[table.lower()] = (schema, table)

    # Обработка comma-separated таблиц в FROM (implicit JOIN):
    # FROM hr.emp e, hr.dept d, hr.loc l
    comma_ref = re.compile(
        r',\s*["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?'
        r'(?:\s+(?:AS\s+)?(\w+))?',
        re.IGNORECASE,
    )
    # Ищем только внутри FROM-клаузы (до WHERE/JOIN/GROUP и т.д.)
    from_clause_pat = re.compile(
        r'\bFROM\s+(.*?)(?=\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bJOIN\b|\bHAVING\b|\Z)',
        re.IGNORECASE | re.DOTALL,
    )
    from_m = from_clause_pat.search(sql)
    if from_m:
        from_text = from_m.group(1)
        for cm in comma_ref.finditer(from_text):
            schema, table = cm.group(1), cm.group(2)
            alias = cm.group(3)
            if alias and alias.upper() not in _SQL_KEYWORDS:
                alias_map[alias.lower()] = (schema, table)
            alias_map[table.lower()] = (schema, table)

    return alias_map


def _find_subquery_join_aliases(sql: str) -> set[str]:
    """Найти алиасы JOIN-ов, которые используют подзапросы.

    Подзапрос в JOIN (например, JOIN (SELECT DISTINCT ...) alias ON ...)
    считается уже безопасным — DISTINCT/GROUP BY внутри предотвращает дубли.

    Returns:
        Множество алиасов (в нижнем регистре), которые являются подзапросами.
    """
    safe_aliases: set[str] = set()
    # Паттерн: JOIN (SELECT ...) alias ON
    # Ищем JOIN за которым следует открывающая скобка (подзапрос)
    subq_pattern = re.compile(
        r'JOIN\s*\(\s*SELECT\b.*?\)\s+(?:AS\s+)?(\w+)\s+ON\b',
        re.IGNORECASE | re.DOTALL,
    )
    for m in subq_pattern.finditer(sql):
        safe_aliases.add(m.group(1).lower())
    return safe_aliases


def _extract_join_conditions(sql: str) -> list[dict[str, str]]:
    """Извлечь таблицы и колонки из JOIN условий.

    Поддерживает:
    - Explicit JOINs с алиасами (JOIN schema.table alias ON alias.col = ...)
    - LEFT/RIGHT/FULL/INNER JOIN
    - Multi-column ON (AND conditions)
    - CROSS JOIN (без ON — всегда explosion)
    - Implicit JOINs (FROM t1, t2 WHERE t1.col = t2.col)
    - Обе стороны JOIN (left и right)
    - Подзапросы в JOIN: JOIN (SELECT DISTINCT ...) alias — пропускаются как безопасные

    Returns:
        Список словарей с schema, table, column для каждой стороны каждого JOIN.
    """
    joins: list[dict[str, str]] = []
    subquery_aliases = _find_subquery_join_aliases(sql)

    # 0. Извлечь CTE-тела и обработать их рекурсивно
    cte_pattern = re.compile(
        r'\bWITH\b\s+(?:RECURSIVE\s+)?'
        r'(.*?)\bSELECT\b',
        re.IGNORECASE | re.DOTALL,
    )
    cte_body_pattern = re.compile(
        r'\w+\s+AS\s*\(\s*(.*?)\s*\)',
        re.IGNORECASE | re.DOTALL,
    )
    cte_match = cte_pattern.match(sql.strip())
    if cte_match:
        cte_defs = cte_match.group(1)
        for body_match in cte_body_pattern.finditer(cte_defs):
            cte_body = body_match.group(1)
            joins.extend(_extract_join_conditions(cte_body))

    # 1. Построить alias map
    alias_map = _build_alias_map(sql)

    def _resolve(ref: str) -> tuple[str, str] | None:
        """Resolve alias/table name to (schema, table)."""
        return alias_map.get(ref.lower())

    # 2. Explicit JOINs: extract ON conditions
    # Найти все JOIN ... ON ... блоки
    join_on_pattern = re.compile(
        r'JOIN\s+["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?'
        r'\s+(?:(?:AS\s+)?(\w+)\s+)?ON\s+(.*?)(?=\bJOIN\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bUNION\b|\bHAVING\b|;|\)|\Z)',
        re.IGNORECASE | re.DOTALL,
    )

    for m in join_on_pattern.finditer(sql):
        join_schema, join_table = m.group(1), m.group(2)
        on_clause = m.group(4)

        # Извлечь все equality conditions из ON clause (поддержка multi-column)
        eq_pattern = re.compile(
            r'["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?'
            r'\s*=\s*'
            r'["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
        )
        for eq in eq_pattern.finditer(on_clause):
            left_ref, left_col = eq.group(1), eq.group(2)
            right_ref, right_col = eq.group(3), eq.group(4)

            # Пропускаем стороны, которые ссылаются на подзапрос (уже безопасны)
            left_is_subquery = left_ref.lower() in subquery_aliases
            right_is_subquery = right_ref.lower() in subquery_aliases

            # Resolve обе стороны через alias map
            left_resolved = _resolve(left_ref)
            right_resolved = _resolve(right_ref)

            if left_resolved and not left_is_subquery:
                joins.append({
                    "schema": left_resolved[0],
                    "table": left_resolved[1],
                    "column": left_col,
                })
            if right_resolved and not right_is_subquery:
                joins.append({
                    "schema": right_resolved[0],
                    "table": right_resolved[1],
                    "column": right_col,
                })

    # 3. CROSS JOIN (нет ON → гарантированный explosion)
    cross_pattern = re.compile(
        r'CROSS\s+JOIN\s+["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
        re.IGNORECASE,
    )
    for m in cross_pattern.finditer(sql):
        joins.append({
            "schema": m.group(1),
            "table": m.group(2),
            "column": "__CROSS_JOIN__",
        })

    # 4. Implicit JOINs: FROM t1, t2 WHERE t1.col = t2.col
    # Детектируем comma-separated tables в FROM
    from_pattern = re.compile(
        r'\bFROM\s+(.*?)(?=\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bJOIN\b|\Z)',
        re.IGNORECASE | re.DOTALL,
    )
    from_match = from_pattern.search(sql)
    if from_match:
        from_clause = from_match.group(1)
        # Ищем comma-separated tables (минимум 2 через запятую)
        table_refs = [t.strip() for t in from_clause.split(",") if t.strip()]
        if len(table_refs) >= 2:
            # Есть implicit join — ищем equality conditions в WHERE
            where_pattern = re.compile(
                r'\bWHERE\b\s+(.*?)(?=\bGROUP\b|\bORDER\b|\bLIMIT\b|\bUNION\b|\Z)',
                re.IGNORECASE | re.DOTALL,
            )
            where_match = where_pattern.search(sql)
            if where_match:
                where_clause = where_match.group(1)
                eq_pattern = re.compile(
                    r'["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?'
                    r'\s*=\s*'
                    r'["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
                )
                for eq in eq_pattern.finditer(where_clause):
                    left_ref, left_col = eq.group(1), eq.group(2)
                    right_ref, right_col = eq.group(3), eq.group(4)

                    left_resolved = _resolve(left_ref)
                    right_resolved = _resolve(right_ref)

                    if left_resolved and left_ref.lower() not in subquery_aliases:
                        joins.append({
                            "schema": left_resolved[0],
                            "table": left_resolved[1],
                            "column": left_col,
                        })
                    if right_resolved and right_ref.lower() not in subquery_aliases:
                        joins.append({
                            "schema": right_resolved[0],
                            "table": right_resolved[1],
                            "column": right_col,
                        })

    # 5. Дедупликация (одна и та же schema.table.column может появиться из разных мест)
    seen = set()
    unique_joins = []
    for j in joins:
        key = (j["schema"].lower(), j["table"].lower(), j["column"].lower())
        if key not in seen:
            seen.add(key)
            unique_joins.append(j)

    return unique_joins


def _extract_join_pairs(sql: str) -> list[dict[str, Any]]:
    """Извлечь пары сторон JOIN с указанием присоединяемой стороны."""
    pairs: list[dict[str, Any]] = []
    subquery_aliases = _find_subquery_join_aliases(sql)
    alias_map = _build_alias_map(sql)

    def _resolve(ref: str) -> tuple[str, str] | None:
        return alias_map.get(ref.lower())

    join_on_pattern = re.compile(
        r'JOIN\s+["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?'
        r'\s+(?:(?:AS\s+)?(\w+)\s+)?ON\s+(.*?)(?=\bJOIN\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bUNION\b|\bHAVING\b|;|\)|\Z)',
        re.IGNORECASE | re.DOTALL,
    )
    eq_pattern = re.compile(
        r'["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?'
        r'\s*=\s*'
        r'["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
    )

    for match in join_on_pattern.finditer(sql):
        join_table = match.group(2)
        join_alias = (match.group(3) or join_table).lower()
        on_clause = match.group(4)

        for eq in eq_pattern.finditer(on_clause):
            left_ref, left_col = eq.group(1), eq.group(2)
            right_ref, right_col = eq.group(3), eq.group(4)

            if left_ref.lower() in subquery_aliases or right_ref.lower() in subquery_aliases:
                continue

            left_resolved = _resolve(left_ref)
            right_resolved = _resolve(right_ref)
            if not left_resolved or not right_resolved:
                continue

            joined_side = "left" if left_ref.lower() == join_alias else "right"
            if left_ref.lower() != join_alias and right_ref.lower() != join_alias:
                joined_side = "right"

            pairs.append({
                "type": "join",
                "joined_side": joined_side,
                "left": {
                    "schema": left_resolved[0],
                    "table": left_resolved[1],
                    "column": left_col,
                },
                "right": {
                    "schema": right_resolved[0],
                    "table": right_resolved[1],
                    "column": right_col,
                },
            })

    cross_pattern = re.compile(
        r'CROSS\s+JOIN\s+["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
        re.IGNORECASE,
    )
    for match in cross_pattern.finditer(sql):
        pairs.append({
            "type": "cross_join",
            "joined_side": "right",
            "left": None,
            "right": {
                "schema": match.group(1),
                "table": match.group(2),
                "column": "__CROSS_JOIN__",
            },
        })

    return pairs


def _has_where_or_limit(sql: str) -> bool:
    """Проверить наличие WHERE или LIMIT в основном запросе.

    Убирает подзапросы в скобках и строковые литералы перед проверкой,
    чтобы избежать ложных срабатываний.
    """
    # Убираем строковые литералы
    cleaned = re.sub(r"'[^']*'", "''", sql)
    # Убираем содержимое подзапросов в скобках (рекурсивно, до 5 уровней)
    for _ in range(5):
        prev = cleaned
        cleaned = re.sub(r'\([^()]*\)', '()', cleaned)
        if cleaned == prev:
            break
    upper = cleaned.upper()
    return "WHERE" in upper or "LIMIT" in upper


def _has_top_level_where(sql: str) -> bool:
    """Проверить наличие WHERE на верхнем уровне statement.

    Не учитывает WHERE в строковых литералах, комментариях и подзапросах.
    """
    statements = sqlparse.parse(sql)
    if not statements:
        return False

    for token in statements[0].tokens:
        if isinstance(token, sqlparse.sql.Where):
            return True
    return False


class SQLValidator:
    """Валидатор SQL-запросов для Greenplum."""

    def __init__(self, db_manager: Any, schema_loader: Any = None) -> None:
        """Инициализация валидатора.

        Args:
            db_manager: Экземпляр DatabaseManager для выполнения EXPLAIN и проверок.
            schema_loader: Опциональный SchemaLoader для CSV-first проверки ключей.
        """
        self._db = db_manager
        self._schema_loader = schema_loader

    def validate(self, sql: str) -> ValidationResult:
        """Валидировать SQL-запрос.

        Args:
            sql: SQL-запрос.

        Returns:
            ValidationResult с результатами проверки.
        """
        mode = detect_mode(sql)
        result = ValidationResult(is_valid=True, mode=mode)

        logger.info("Валидация SQL (режим %s): %s", mode.value, sql[:200])

        if mode == SQLMode.READ:
            self._validate_read(sql, result)
        elif mode == SQLMode.WRITE:
            self._validate_write(sql, result)
        elif mode == SQLMode.DDL:
            self._validate_ddl(sql, result)

        return result

    @staticmethod
    def _estimate_multiplication_factor(join_checks: list[dict[str, Any]]) -> float:
        """Оценка множителя размножения строк из-за неуникальных JOIN.

        Для каждого неуникального JOIN: factor = 100 / unique_perc.
        Общий factor — произведение по всем JOIN.
        """
        total = 1.0
        for jc in join_checks:
            if "factor" in jc:
                total *= jc["factor"]
                continue
            if not jc["is_unique"]:
                unique_perc = 100.0 - jc["duplicate_pct"]
                if unique_perc > 0:
                    total *= min(100.0 / unique_perc, 100.0)
                else:
                    total *= 100.0
        return total

    def _generate_rewrite_suggestion(
        self,
        joined: dict[str, str],
        existing: dict[str, str] | None = None,
    ) -> str:
        """Сгенерировать конкретный шаблон переписывания JOIN с учётом типов таблиц.

        Стратегии:
        1. fact + fact → CTE с GROUP BY для обеих сторон
        2. fact + dim/ref → уникальная выборка из справочника
        3. dim/ref + fact → CTE с GROUP BY для фактов
        4. dim/ref + dim/ref → уникальные выборки из обеих сторон
        """
        schema, table, column = joined["schema"], joined["table"], joined["column"]
        full_joined = f"{schema}.{table}"

        # Определяем типы таблиц через schema_loader
        joined_type = "unknown"
        existing_type = "unknown"
        if self._schema_loader is not None:
            from core.join_analysis import detect_table_type
            joined_cols_df = self._schema_loader.get_table_columns(schema, table)
            joined_type = detect_table_type(table, joined_cols_df)
            if existing:
                existing_cols_df = self._schema_loader.get_table_columns(
                    existing["schema"], existing["table"],
                )
                existing_type = detect_table_type(existing["table"], existing_cols_df)

        full_existing = f"{existing['schema']}.{existing['table']}" if existing else "?"
        ex_col = existing["column"] if existing else "?"

        is_fact_j = joined_type in ("fact", "unknown")
        is_dim_j = joined_type in ("dim", "ref")
        is_fact_e = existing_type in ("fact", "unknown")
        is_dim_e = existing_type in ("dim", "ref")

        header = (
            f"ROW EXPLOSION: JOIN ключ {full_joined}.{column} не уникален.\n"
            f"Тип таблиц: {full_existing} ({existing_type}) ↔ {full_joined} ({joined_type})\n"
        )

        if is_fact_e and is_fact_j:
            strategy = (
                f"СТРАТЕГИЯ: ФАКТ + ФАКТ — предварительная агрегация ОБЕИХ сторон в CTE.\n"
                f"  WITH cte_left AS (\n"
                f"    SELECT {ex_col}, SUM(<метрика>) AS val FROM {full_existing} GROUP BY {ex_col}\n"
                f"  ), cte_right AS (\n"
                f"    SELECT {column}, SUM(<метрика>) AS val FROM {full_joined} GROUP BY {column}\n"
                f"  )\n"
                f"  SELECT * FROM cte_left\n"
                f"  JOIN cte_right ON cte_right.{column} = cte_left.{ex_col}"
            )
        elif is_fact_e and is_dim_j:
            strategy = (
                f"СТРАТЕГИЯ: ФАКТ + СПРАВОЧНИК — уникальная выборка из справочника.\n"
                f"  JOIN (\n"
                f"    SELECT DISTINCT ON ({column}) {column}, <нужные_колонки>\n"
                f"    FROM {full_joined} ORDER BY {column}, <дата_актуальности> DESC\n"
                f"  ) d ON d.{column} = <факт_алиас>.{ex_col}"
            )
        elif is_dim_e and is_fact_j:
            strategy = (
                f"СТРАТЕГИЯ: СПРАВОЧНИК + ФАКТ — агрегация фактов в CTE/подзапросе.\n"
                f"  SELECT d.*, agg.val\n"
                f"  FROM {full_existing} d\n"
                f"  JOIN (\n"
                f"    SELECT {column}, SUM(<метрика>) AS val FROM {full_joined} GROUP BY {column}\n"
                f"  ) agg ON agg.{column} = d.{ex_col}"
            )
        elif is_dim_e and is_dim_j:
            strategy = (
                f"СТРАТЕГИЯ: СПРАВОЧНИК + СПРАВОЧНИК — уникальные выборки из обеих сторон.\n"
                f"  WITH d1 AS (\n"
                f"    SELECT DISTINCT ON ({ex_col}) {ex_col}, <колонки> FROM {full_existing}\n"
                f"    ORDER BY {ex_col}, <дата> DESC\n"
                f"  ), d2 AS (\n"
                f"    SELECT DISTINCT ON ({column}) {column}, <колонки> FROM {full_joined}\n"
                f"    ORDER BY {column}, <дата> DESC\n"
                f"  )\n"
                f"  SELECT * FROM d1 JOIN d2 ON d2.{column} = d1.{ex_col}"
            )
        else:
            strategy = (
                f"ОБЯЗАТЕЛЬНО: вызови get_sample для {full_joined} и изучи причину дублей.\n"
                f"  Вариант 1 — статусы/версии:\n"
                f"    JOIN (SELECT DISTINCT ON ({column}) * FROM {full_joined} "
                f"ORDER BY {column}, <дата> DESC) alias ON ...\n"
                f"  Вариант 2 — несколько фактов:\n"
                f"    JOIN (SELECT {column}, SUM(<метрика>) AS total FROM {full_joined} "
                f"GROUP BY {column}) alias ON ...\n"
                f"  Вариант 3 — технические дубли:\n"
                f"    JOIN (SELECT DISTINCT {column}, <колонки> FROM {full_joined}) alias ON ..."
            )

        footer = (
            "\nЗАПРЕЩЕНО: добавлять DISTINCT к внешнему SELECT — это маскирует проблему.\n"
            "ЗАПРЕЩЕНО: применять DISTINCT без понимания причины дублей."
        )
        return header + strategy + footer

    def _check_key_uniqueness(
        self, schema: str, table: str, columns: list[str],
    ) -> dict[str, Any]:
        """Проверить уникальность ключа: сначала CSV, потом DB fallback."""
        if self._schema_loader is not None:
            csv_result = self._schema_loader.check_key_uniqueness(schema, table, columns)
            if csv_result.get("status") in {"safe", "risky"}:
                return {
                    "is_unique": csv_result["is_unique"],
                    "duplicate_pct": csv_result["duplicate_pct"],
                    "status": csv_result["status"],
                }
        db_result = self._db.check_key_uniqueness(schema, table, columns)
        db_result["status"] = "safe" if db_result["is_unique"] else "risky"
        return db_result

    def _validate_read(self, sql: str, result: ValidationResult) -> None:
        """Валидация SELECT-запросов."""
        # 1. EXPLAIN — синтаксическая проверка
        try:
            plan = self._db.explain_query(sql)
            result.explain_plan = plan
        except Exception as e:
            result.add_error(f"Синтаксическая ошибка (EXPLAIN): {e}")
            return

        # 2. Проверка JOIN-ов на кардинальность
        join_pairs = _extract_join_pairs(sql)
        for pair in join_pairs:
            if pair["type"] == "cross_join":
                joined = pair["right"]
                result.join_checks.append({
                    "join": f"CROSS JOIN {joined['schema']}.{joined['table']}",
                    "table": f"{joined['schema']}.{joined['table']}",
                    "columns": "CROSS JOIN",
                    "cardinality": "cross-join",
                    "is_unique": False,
                    "is_safe": False,
                    "duplicate_pct": 100.0,
                    "factor": 100.0,
                })
                result.rewrite_suggestions.append(
                    f"ROW EXPLOSION: CROSS JOIN с {joined['schema']}.{joined['table']} "
                    "создаёт декартово произведение. Замени на обычный JOIN с условием."
                )
                continue

            left = pair["left"]
            right = pair["right"]
            joined_side = pair["joined_side"]
            joined = left if joined_side == "left" else right
            existing = right if joined_side == "left" else left

            try:
                existing_check = self._check_key_uniqueness(
                    existing["schema"], existing["table"], [existing["column"]]
                )
                joined_check = self._check_key_uniqueness(
                    joined["schema"], joined["table"], [joined["column"]]
                )
            except Exception as e:
                logger.warning("Не удалось проверить кардинальность JOIN: %s", e)
                continue

            existing_status = existing_check.get("status", "unknown")
            joined_status = joined_check.get("status", "unknown")
            existing_unique = existing_status == "safe"
            joined_unique = joined_status == "safe"
            joined_dup = joined_check.get("duplicate_pct")
            joined_factor = 2.0
            if joined_dup is not None:
                joined_unique_perc = max(0.01, 100.0 - float(joined_dup))
                joined_factor = min(100.0 / joined_unique_perc, 100.0)

            if existing_unique and joined_unique:
                cardinality = "one-to-one"
                is_safe = True
                factor = 1.0
            elif joined_unique:
                cardinality = "many-to-one"
                is_safe = True
                factor = 1.0
            elif existing_unique and joined_status == "risky":
                cardinality = "one-to-many"
                is_safe = False
                factor = joined_factor
            elif existing_status == "risky" and joined_status == "risky":
                cardinality = "many-to-many"
                is_safe = False
                existing_dup = existing_check.get("duplicate_pct", 50.0)
                existing_unique_perc = max(0.01, 100.0 - float(existing_dup))
                factor = min((100.0 / existing_unique_perc) * joined_factor, 100.0)
            else:
                cardinality = "unknown"
                is_safe = False
                factor = max(joined_factor, 2.0)

            join_label = (
                f"{left['schema']}.{left['table']}.{left['column']} = "
                f"{right['schema']}.{right['table']}.{right['column']}"
            )
            check_info = {
                "join": join_label,
                "table": f"{joined['schema']}.{joined['table']}",
                "columns": joined["column"],
                "cardinality": cardinality,
                "is_unique": joined_unique,
                "is_safe": is_safe,
                "duplicate_pct": joined_check.get("duplicate_pct"),
                "factor": factor,
            }
            result.join_checks.append(check_info)

            if not is_safe:
                result.rewrite_suggestions.append(
                    self._generate_rewrite_suggestion(joined, existing)
                )

        # 2b. Composite key re-check: если несколько колонок между одними таблицами,
        #     проверяем их как составной ключ (вместе могут быть уникальными)
        from core.join_analysis import group_composite_keys
        composite_groups = group_composite_keys(join_pairs)
        for group in composite_groups:
            if len(group.columns) < 2:
                continue
            # Проверяем составной ключ для joined-стороны
            joined_cols = [col_pair[1] for col_pair in group.columns]
            existing_cols = [col_pair[0] for col_pair in group.columns]
            try:
                composite_check = self._check_key_uniqueness(
                    group.right_schema, group.right_table, joined_cols,
                )
                if composite_check.get("status") == "safe":
                    # Составной ключ уникален → обновляем risky checks на safe
                    composite_label = ", ".join(
                        f"{group.left_schema}.{group.left_table}.{ec} = "
                        f"{group.right_schema}.{group.right_table}.{jc}"
                        for ec, jc in group.columns
                    )
                    for jc_info in result.join_checks:
                        if (jc_info.get("table") == f"{group.right_schema}.{group.right_table}"
                                and jc_info.get("columns") in joined_cols
                                and not jc_info.get("is_safe")):
                            jc_info["is_safe"] = True
                            jc_info["cardinality"] = "composite-key-safe"
                            jc_info["factor"] = 1.0
                    logger.info(
                        "Composite key %s safe: %s",
                        f"{group.right_schema}.{group.right_table}",
                        joined_cols,
                    )
            except Exception as e:
                logger.warning("Composite key check failed: %s", e)

        # 3. Оценка multiplication factor и решение pass/warn/block
        if result.join_checks:
            factor = self._estimate_multiplication_factor(result.join_checks)
            result.multiplication_factor = factor

            hard_risk = [
                jc for jc in result.join_checks if jc.get("cardinality") in {"many-to-many", "cross-join"}
            ]
            soft_risk = [
                jc for jc in result.join_checks if jc.get("cardinality") in {"one-to-many", "unknown"}
            ]

            if hard_risk:
                details = "; ".join(f"{jc['join']} [{jc['cardinality']}]" for jc in hard_risk)
                result.add_error(
                    f"ROW EXPLOSION (factor={factor:.1f}x): {details}. "
                    "Обнаружен JOIN, который размножает строки результата.\n"
                    + "\n".join(result.rewrite_suggestions)
                )
            elif soft_risk:
                details = "; ".join(f"{jc['join']} [{jc['cardinality']}]" for jc in soft_risk)
                result.add_error(
                    f"JOIN RISK (factor={factor:.1f}x): {details}. "
                    "Присоединяемая сторона не доказана как уникальная и может размножить строки."
                )

        # 4. Предупреждение если нет WHERE/LIMIT для больших таблиц
        if not _has_where_or_limit(sql):
            result.add_warning(
                "Запрос без WHERE/LIMIT. Для больших таблиц это может вернуть много данных."
            )

    def _validate_write(self, sql: str, result: ValidationResult) -> None:
        """Валидация INSERT/UPDATE/DELETE."""
        normalized = sql.strip().upper()

        # 1. EXPLAIN
        try:
            plan = self._db.explain_query(sql)
            result.explain_plan = plan
        except Exception as e:
            result.add_error(f"Синтаксическая ошибка (EXPLAIN): {e}")
            return

        # 2. UPDATE/DELETE без WHERE — требуем подтверждение
        is_update_or_delete = normalized.startswith(("UPDATE", "DELETE"))
        if is_update_or_delete and not _has_top_level_where(sql):
            result.require_confirmation(
                "UPDATE/DELETE без WHERE затронет ВСЕ строки таблицы. Вы уверены?"
            )

        # 3. Оценка затронутых строк
        if is_update_or_delete:
            self._estimate_write_impact(sql, result)

    def _estimate_write_impact(self, sql: str, result: ValidationResult) -> None:
        """Estimate write impact without executing arbitrary WHERE fragments.

        Security boundary: precise row counting by dynamic WHERE is disabled.
        """
        normalized = sql.strip()
        upper = normalized.upper()

        try:
            if upper.startswith("DELETE"):
                match = re.search(
                    r'DELETE\s+FROM\s+["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
                    normalized, re.IGNORECASE,
                )
                if match:
                    schema, table = match.group(1), match.group(2)
                    result.add_warning(
                        f"Оценка затронутых строк для {schema}.{table} отключена по соображениям безопасности."
                    )

            elif upper.startswith("UPDATE"):
                match = re.search(
                    r'UPDATE\s+["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
                    normalized, re.IGNORECASE,
                )
                if match:
                    schema, table = match.group(1), match.group(2)
                    result.add_warning(
                        f"Оценка затронутых строк для {schema}.{table} отключена по соображениям безопасности."
                    )
        except Exception as e:
            logger.warning("Не удалось оценить количество затронутых строк: %s", e)
    def _validate_ddl(self, sql: str, result: ValidationResult) -> None:
        """Валидация DDL-запросов."""
        normalized = sql.strip().upper()

        # 1. DROP / TRUNCATE — требуем подтверждение
        if normalized.startswith(("DROP", "TRUNCATE")):
            result.require_confirmation(
                "Вы собираетесь выполнить DROP/TRUNCATE. Это необратимая операция. "
                "Введите YES для подтверждения."
            )

        # 2. CREATE TABLE — проверка существования
        if normalized.startswith("CREATE TABLE"):
            match = re.search(
                r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
                sql, re.IGNORECASE,
            )
            if match:
                schema, table = match.group(1), match.group(2)
                try:
                    if self._db.table_exists(schema, table):
                        result.add_error(
                            f"Таблица {schema}.{table} уже существует. "
                            "Используйте IF NOT EXISTS или DROP перед созданием."
                        )
                except Exception as e:
                    logger.warning("Не удалось проверить существование таблицы: %s", e)

        # 3. ALTER — показать текущую структуру
        if normalized.startswith("ALTER"):
            match = re.search(
                r'ALTER\s+TABLE\s+["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
                sql, re.IGNORECASE,
            )
            if match:
                schema, table = match.group(1), match.group(2)
                try:
                    ddl = self._db.get_table_ddl(schema, table)
                    result.add_warning(f"Текущая структура таблицы:\n{ddl}")
                except Exception as e:
                    logger.warning("Не удалось получить DDL таблицы: %s", e)

