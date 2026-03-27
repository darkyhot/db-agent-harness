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
                lines.append(f"  ✗ {e}")
        if self.warnings:
            lines.append("Предупреждения:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        if self.needs_confirmation:
            lines.append(f"Требуется подтверждение: {self.confirmation_message}")
        if self.join_checks:
            lines.append("Проверка JOIN-ов:")
            for jc in self.join_checks:
                status = "✓ уникален" if jc["is_unique"] else f"✗ дубли: {jc['duplicate_pct']}%"
                lines.append(f"  {jc['table']}.({jc['columns']}): {status}")
            if self.multiplication_factor > 1.0:
                lines.append(f"  Multiplication factor: {self.multiplication_factor:.1f}x")
        if self.rewrite_suggestions:
            lines.append("Рекомендации по переписыванию:")
            for s in self.rewrite_suggestions:
                lines.append(f"  {s}")
        return "\n".join(lines)


def detect_mode(sql: str) -> SQLMode:
    """Определить режим SQL-запроса.

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

    if normalized.startswith(("CREATE", "ALTER", "DROP", "TRUNCATE")):
        return SQLMode.DDL
    if normalized.startswith(("INSERT", "UPDATE", "DELETE", "MERGE")):
        return SQLMode.WRITE
    return SQLMode.READ


def _extract_join_conditions(sql: str) -> list[dict[str, str]]:
    """Извлечь таблицы и колонки из JOIN условий.

    Args:
        sql: SQL-запрос.

    Returns:
        Список словарей с table, schema, column для каждого JOIN.
    """
    joins = []
    parsed = sqlparse.parse(sql)
    if not parsed:
        return joins

    sql_upper = sql.upper()
    # Находим паттерны JOIN ... ON ... = ...
    join_pattern = re.compile(
        r'JOIN\s+["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?'
        r'\s+(?:\w+\s+)?ON\s+.*?["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?'
        r'\s*=\s*["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
        re.IGNORECASE | re.DOTALL,
    )

    for match in join_pattern.finditer(sql):
        join_schema = match.group(1)
        join_table = match.group(2)
        # Определяем какая сторона относится к join-таблице
        left_table = match.group(3)
        left_col = match.group(4)
        right_table = match.group(5)
        right_col = match.group(6)

        if left_table.lower() == join_table.lower():
            joins.append({
                "schema": join_schema,
                "table": join_table,
                "column": left_col,
            })
        elif right_table.lower() == join_table.lower():
            joins.append({
                "schema": join_schema,
                "table": join_table,
                "column": right_col,
            })
        else:
            # Fallback — проверяем обе стороны
            joins.append({
                "schema": join_schema,
                "table": join_table,
                "column": left_col,
            })

    return joins


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
            if not jc["is_unique"]:
                unique_perc = 100.0 - jc["duplicate_pct"]
                if unique_perc > 0:
                    total *= min(100.0 / unique_perc, 100.0)
                else:
                    total *= 100.0
        return total

    @staticmethod
    def _generate_rewrite_suggestion(join: dict[str, str]) -> str:
        """Сгенерировать конкретный шаблон переписывания JOIN."""
        schema, table, column = join["schema"], join["table"], join["column"]
        return (
            f"ROW EXPLOSION: JOIN ключ {schema}.{table}.{column} не уникален.\n"
            f"ИСПРАВЛЕНИЕ: Оберни {schema}.{table} в подзапрос с DISTINCT:\n"
            f"  БЫЛО: JOIN {schema}.{table} alias ON ... = alias.{column}\n"
            f"  СТАЛО: JOIN (SELECT DISTINCT {column}, <нужные_колонки> "
            f"FROM {schema}.{table}) alias ON ... = alias.{column}\n"
            f"Или используй предварительную агрегацию (GROUP BY + SUM/COUNT), "
            f"если нужна агрегация.\n"
            f"ЗАПРЕЩЕНО: добавлять DISTINCT к внешнему SELECT — это маскирует проблему."
        )

    def _check_key_uniqueness(
        self, schema: str, table: str, columns: list[str],
    ) -> dict[str, Any]:
        """Проверить уникальность ключа: сначала CSV, потом DB fallback."""
        if self._schema_loader is not None:
            csv_result = self._schema_loader.check_key_uniqueness(schema, table, columns)
            if csv_result.get("is_unique") is not None:
                return {
                    "is_unique": csv_result["is_unique"],
                    "duplicate_pct": csv_result["duplicate_pct"],
                }
        return self._db.check_key_uniqueness(schema, table, columns)

    def _validate_read(self, sql: str, result: ValidationResult) -> None:
        """Валидация SELECT-запросов."""
        # 1. EXPLAIN — синтаксическая проверка
        try:
            plan = self._db.explain_query(sql)
            result.explain_plan = plan
        except Exception as e:
            result.add_error(f"Синтаксическая ошибка (EXPLAIN): {e}")
            return

        # 2. Проверка JOIN-ов на уникальность ключей
        joins = _extract_join_conditions(sql)
        for join in joins:
            try:
                check = self._check_key_uniqueness(
                    join["schema"], join["table"], [join["column"]]
                )
                check_info = {
                    "table": f"{join['schema']}.{join['table']}",
                    "columns": join["column"],
                    "is_unique": check["is_unique"],
                    "duplicate_pct": check["duplicate_pct"],
                }
                result.join_checks.append(check_info)
                if not check["is_unique"]:
                    suggestion = self._generate_rewrite_suggestion(join)
                    result.rewrite_suggestions.append(suggestion)
            except Exception as e:
                logger.warning("Не удалось проверить уникальность JOIN: %s", e)

        # 3. Оценка multiplication factor и решение pass/warn/block
        if result.join_checks:
            factor = self._estimate_multiplication_factor(result.join_checks)
            result.multiplication_factor = factor

            non_unique = [
                jc for jc in result.join_checks if not jc["is_unique"]
            ]
            if non_unique and factor > 5.0:
                # BLOCK — row explosion очень вероятен
                details = "; ".join(
                    f"{jc['table']}.{jc['columns']} (дубли: {jc['duplicate_pct']}%)"
                    for jc in non_unique
                )
                result.add_error(
                    f"ROW EXPLOSION (factor={factor:.1f}x): "
                    f"Неуникальные JOIN-ключи: {details}. "
                    "Перепиши SQL с подзапросом (DISTINCT) или агрегацией.\n"
                    + "\n".join(result.rewrite_suggestions)
                )
            elif non_unique and factor > 1.5:
                # WARN — умеренный риск
                details = "; ".join(
                    f"{jc['table']}.{jc['columns']} (дубли: {jc['duplicate_pct']}%)"
                    for jc in non_unique
                )
                result.add_warning(
                    f"JOIN risk (factor={factor:.1f}x): "
                    f"Неуникальные ключи: {details}. "
                    "Рассмотри подзапрос с DISTINCT или агрегацию."
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
        if is_update_or_delete and "WHERE" not in normalized:
            result.require_confirmation(
                "UPDATE/DELETE без WHERE затронет ВСЕ строки таблицы. Вы уверены?"
            )

        # 3. Оценка затронутых строк
        if is_update_or_delete:
            self._estimate_write_impact(sql, result)

    def _estimate_write_impact(self, sql: str, result: ValidationResult) -> None:
        """Оценить количество затронутых строк для UPDATE/DELETE."""
        # Извлекаем таблицу и WHERE
        normalized = sql.strip()
        upper = normalized.upper()

        try:
            if upper.startswith("DELETE"):
                # DELETE FROM schema.table WHERE ...
                match = re.search(
                    r'DELETE\s+FROM\s+["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
                    normalized, re.IGNORECASE,
                )
                if match:
                    schema, table = match.group(1), match.group(2)
                    where_match = re.search(r'\bWHERE\b\s+(.*)', normalized, re.IGNORECASE | re.DOTALL)
                    if where_match:
                        where_clause = where_match.group(1).rstrip(";")
                        count = self._db.estimate_affected_rows(where_clause, schema, table)
                        result.add_warning(f"Будет затронуто строк: {count}")

            elif upper.startswith("UPDATE"):
                match = re.search(
                    r'UPDATE\s+["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
                    normalized, re.IGNORECASE,
                )
                if match:
                    schema, table = match.group(1), match.group(2)
                    where_match = re.search(r'\bWHERE\b\s+(.*)', normalized, re.IGNORECASE | re.DOTALL)
                    if where_match:
                        where_clause = where_match.group(1).rstrip(";")
                        count = self._db.estimate_affected_rows(where_clause, schema, table)
                        result.add_warning(f"Будет затронуто строк: {count}")
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
