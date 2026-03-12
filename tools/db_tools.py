"""Инструменты для работы с базой данных Greenplum."""

import logging
from typing import Any

from langchain_core.tools import tool

from core.database import DatabaseManager

logger = logging.getLogger(__name__)

# Глобальный экземпляр — инициализируется при сборке агента
_db: DatabaseManager | None = None


def init_db_tools(db_manager: DatabaseManager) -> None:
    """Инициализировать модуль экземпляром DatabaseManager.

    Args:
        db_manager: Настроенный экземпляр DatabaseManager.
    """
    global _db
    _db = db_manager
    logger.info("db_tools инициализированы")


def _get_db() -> DatabaseManager:
    """Получить экземпляр DatabaseManager."""
    if _db is None:
        raise RuntimeError("db_tools не инициализированы. Вызовите init_db_tools() сначала.")
    return _db


# === READ ===

@tool
def execute_query(sql: str, limit: int = 1000) -> str:
    """Выполнить SELECT-запрос и вернуть результат в виде markdown-таблицы.

    Args:
        sql: SQL-запрос (SELECT).
        limit: Максимальное количество строк (по умолчанию 1000).

    Returns:
        Результат в формате markdown-таблицы или сообщение об ошибке.
    """
    try:
        db = _get_db()
        df = db.execute_query(sql, limit=limit)
        if df.empty:
            return "Запрос выполнен. Результат пуст."
        return df.to_markdown(index=False)
    except Exception as e:
        logger.error("execute_query error: %s", e)
        return f"Ошибка выполнения запроса: {e}"


@tool
def get_row_count(schema: str, table: str) -> str:
    """Получить количество строк в таблице.

    Args:
        schema: Имя схемы.
        table: Имя таблицы.

    Returns:
        Количество строк.
    """
    try:
        db = _get_db()
        count = db.get_row_count(schema, table)
        return f"Таблица {schema}.{table}: {count:,} строк"
    except Exception as e:
        logger.error("get_row_count error: %s", e)
        return f"Ошибка: {e}"


@tool
def check_key_uniqueness(schema: str, table: str, columns: str) -> str:
    """Проверить уникальность комбинации колонок (для валидации JOIN).

    Args:
        schema: Имя схемы.
        table: Имя таблицы.
        columns: Имена колонок через запятую (например: 'id,date').

    Returns:
        Результат проверки уникальности.
    """
    try:
        db = _get_db()
        cols = [c.strip() for c in columns.split(",")]
        result = db.check_key_uniqueness(schema, table, cols)
        if result["is_unique"]:
            return f"Ключ ({columns}) в {schema}.{table} уникален. Строк: {result['total_rows']:,}"
        return (
            f"Ключ ({columns}) в {schema}.{table} НЕ уникален.\n"
            f"Всего строк: {result['total_rows']:,}, "
            f"уникальных: {result['unique_keys']:,}, "
            f"дублей: {result['duplicate_pct']}%"
        )
    except Exception as e:
        logger.error("check_key_uniqueness error: %s", e)
        return f"Ошибка: {e}"


@tool
def get_sample(schema: str, table: str, n: int = 10) -> str:
    """Получить образец данных из таблицы.

    Args:
        schema: Имя схемы.
        table: Имя таблицы.
        n: Количество строк (по умолчанию 10).

    Returns:
        Markdown-таблица с образцом данных.
    """
    try:
        db = _get_db()
        df = db.get_sample(schema, table, n)
        if df.empty:
            return f"Таблица {schema}.{table} пуста."
        return df.to_markdown(index=False)
    except Exception as e:
        logger.error("get_sample error: %s", e)
        return f"Ошибка: {e}"


@tool
def explain_query(sql: str) -> str:
    """Показать план выполнения запроса (EXPLAIN) без реального выполнения.

    Args:
        sql: SQL-запрос для анализа.

    Returns:
        План выполнения запроса.
    """
    try:
        db = _get_db()
        plan = db.explain_query(sql)
        return f"План выполнения:\n{plan}"
    except Exception as e:
        logger.error("explain_query error: %s", e)
        return f"Ошибка: {e}"


@tool
def table_exists(schema: str, table: str) -> str:
    """Проверить существование таблицы.

    Args:
        schema: Имя схемы.
        table: Имя таблицы.

    Returns:
        Существует таблица или нет.
    """
    try:
        db = _get_db()
        exists = db.table_exists(schema, table)
        if exists:
            return f"Таблица {schema}.{table} существует."
        return f"Таблица {schema}.{table} НЕ найдена."
    except Exception as e:
        logger.error("table_exists error: %s", e)
        return f"Ошибка: {e}"


@tool
def get_table_ddl(schema: str, table: str) -> str:
    """Получить DDL (структуру) таблицы.

    Args:
        schema: Имя схемы.
        table: Имя таблицы.

    Returns:
        DDL таблицы.
    """
    try:
        db = _get_db()
        return db.get_table_ddl(schema, table)
    except Exception as e:
        logger.error("get_table_ddl error: %s", e)
        return f"Ошибка: {e}"


# === WRITE ===

@tool
def execute_write(sql: str) -> str:
    """Выполнить INSERT/UPDATE/DELETE и вернуть количество затронутых строк.

    Args:
        sql: SQL-запрос (INSERT/UPDATE/DELETE).

    Returns:
        Количество затронутых строк.
    """
    try:
        db = _get_db()
        affected = db.execute_write(sql)
        return f"Запрос выполнен. Затронуто строк: {affected}"
    except Exception as e:
        logger.error("execute_write error: %s", e)
        return f"Ошибка: {e}"


@tool
def estimate_affected_rows(where_clause: str, schema: str, table: str) -> str:
    """Оценить количество строк, которые будут затронуты WHERE-условием.

    Args:
        where_clause: Условие WHERE (без ключевого слова WHERE).
        schema: Имя схемы.
        table: Имя таблицы.

    Returns:
        Количество строк, подходящих под условие.
    """
    try:
        db = _get_db()
        count = db.estimate_affected_rows(where_clause, schema, table)
        return f"Оценка: {count:,} строк будет затронуто в {schema}.{table}"
    except Exception as e:
        logger.error("estimate_affected_rows error: %s", e)
        return f"Ошибка: {e}"


# === DDL ===

@tool
def execute_ddl(sql: str) -> str:
    """Выполнить DDL-запрос (CREATE/ALTER/DROP/TRUNCATE).

    Args:
        sql: DDL-запрос.

    Returns:
        Сообщение об успехе или ошибке.
    """
    try:
        db = _get_db()
        return db.execute_ddl(sql)
    except Exception as e:
        logger.error("execute_ddl error: %s", e)
        return f"Ошибка: {e}"


# Список всех инструментов для регистрации в агенте
DB_TOOLS = [
    execute_query,
    get_row_count,
    check_key_uniqueness,
    get_sample,
    explain_query,
    table_exists,
    get_table_ddl,
    execute_write,
    estimate_affected_rows,
    execute_ddl,
]
