"""Инструменты для работы с базой данных Greenplum.

Используют фабрику create_db_tools() для DI вместо глобальных переменных.
"""

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from core.database import DatabaseManager
from core.sql_validator import SQLValidator, detect_mode, SQLMode

WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"

logger = logging.getLogger(__name__)


def create_db_tools(
    db_manager: DatabaseManager, sql_validator: SQLValidator | None = None
) -> list:
    """Создать инструменты БД через замыкания (DI без глобальных переменных).

    Args:
        db_manager: Настроенный экземпляр DatabaseManager.
        sql_validator: Опциональный валидатор для DDL-проверок в tool.

    Returns:
        Список LangChain tools.
    """

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
            df = db_manager.execute_query(sql, limit=limit)
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
            count = db_manager.get_row_count(schema, table)
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
            cols = [c.strip() for c in columns.split(",")]
            result = db_manager.check_key_uniqueness(schema, table, cols)
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
            df = db_manager.get_sample(schema, table, n)
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
            plan = db_manager.explain_query(sql)
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
            exists = db_manager.table_exists(schema, table)
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
            return db_manager.get_table_ddl(schema, table)
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
            affected = db_manager.execute_write(sql)
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
            count = db_manager.estimate_affected_rows(where_clause, schema, table)
            return f"Оценка: {count:,} строк будет затронуто в {schema}.{table}"
        except Exception as e:
            logger.error("estimate_affected_rows error: %s", e)
            return f"Ошибка: {e}"

    # === DDL ===

    @tool
    def execute_ddl(sql: str) -> str:
        """Выполнить DDL-запрос (CREATE/ALTER/DROP/TRUNCATE).

        Перед выполнением проходит валидацию: DROP/TRUNCATE требуют подтверждения,
        CREATE TABLE проверяет существование таблицы.

        Args:
            sql: DDL-запрос.

        Returns:
            Сообщение об успехе или ошибке.
        """
        try:
            # Принудительная валидация DDL на уровне tool
            if sql_validator is not None:
                validation = sql_validator.validate(sql)
                if not validation.is_valid:
                    return f"DDL отклонён валидатором:\n{validation.summary()}"
                if validation.needs_confirmation:
                    return (
                        f"DDL требует подтверждения: {validation.confirmation_message}\n"
                        "Используйте команду подтверждения перед выполнением."
                    )
            return db_manager.execute_ddl(sql)
        except Exception as e:
            logger.error("execute_ddl error: %s", e)
            return f"Ошибка: {e}"

    @tool
    def export_query(sql: str, filename: str, output_format: str = "csv") -> str:
        """Выполнить SELECT-запрос и сохранить результат в файл в workspace/.

        Используй этот инструмент когда нужно сделать выгрузку данных в файл.
        Не нужно сначала вызывать execute_query, а потом save_dataframe —
        этот инструмент делает всё за один шаг.

        Args:
            sql: SQL-запрос (SELECT).
            filename: Имя файла (например: 'report.csv', 'sample/outflow.csv').
            output_format: Формат — 'csv' или 'excel'.

        Returns:
            Сообщение об успехе с количеством строк или ошибка.
        """
        try:
            df = db_manager.execute_query(sql)
            file_path = WORKSPACE_DIR / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if output_format == "excel":
                df.to_excel(file_path, index=False)
            else:
                df.to_csv(file_path, index=False, encoding="utf-8")

            logger.info("Экспорт: %s (%d строк)", file_path, len(df))
            return f"Сохранено в {filename} ({len(df)} строк)"
        except Exception as e:
            logger.error("export_query error: %s", e)
            return f"Ошибка экспорта: {e}"

    tools_list = [
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
        export_query,
    ]
    return tools_list
