"""Tools for Greenplum database access with DI factory."""

import json
import logging
from pathlib import Path

from langchain_core.tools import tool

from core.database import DatabaseManager
from core.schema_loader import SchemaLoader
from core.sql_validator import SQLValidator
from tools.path_safety import resolve_workspace_path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"
logger = logging.getLogger(__name__)


def _build_sql_tool_payload(
    *,
    message: str,
    preview_markdown: str = "",
    total_rows: int = 0,
    is_empty: bool = False,
    saved_file: str | None = None,
    mode: str = "preview",
) -> str:
    """Return a structured SQL tool result contract as JSON string."""
    payload = {
        "message": message,
        "preview_markdown": preview_markdown,
        "total_rows": int(total_rows),
        "is_empty": bool(is_empty),
        "saved_file": saved_file,
        "mode": mode,
    }
    return json.dumps(payload, ensure_ascii=False)


def create_db_tools(
    db_manager: DatabaseManager,
    sql_validator: SQLValidator | None = None,
    schema_loader: SchemaLoader | None = None,
) -> list:
    """Create DB tools via closures (dependency injection)."""

    PREVIEW_ROWS = 20

    @tool
    def execute_query(sql: str, limit: int = 1000) -> str:
        """Run SELECT query and return a structured preview/full-result metadata payload."""
        try:
            df = db_manager.preview_query(sql, limit=limit)
            if df.empty:
                return _build_sql_tool_payload(
                    message="Запрос выполнен. Результат пуст.",
                    preview_markdown="",
                    total_rows=0,
                    is_empty=True,
                    saved_file=None,
                    mode="preview",
                )

            total = len(df)
            # Всегда сохраняем CSV — чтобы last_query_result.csv всегда содержал
            # результат ПОСЛЕДНЕГО запроса, независимо от размера выборки.
            auto_file = resolve_workspace_path(WORKSPACE_DIR, "last_query_result.csv")
            df.to_csv(auto_file, index=False, encoding="utf-8")

            if total <= PREVIEW_ROWS:
                return _build_sql_tool_payload(
                    message=f"Показаны все {total} строк.",
                    preview_markdown=df.to_markdown(index=False),
                    total_rows=total,
                    is_empty=False,
                    saved_file="last_query_result.csv",
                    mode="preview",
                )

            preview = df.head(PREVIEW_ROWS).to_markdown(index=False)
            return _build_sql_tool_payload(
                message=(
                    f"Показано {PREVIEW_ROWS} из {total} строк. "
                    "Полный результат сохранен в last_query_result.csv"
                ),
                preview_markdown=preview,
                total_rows=total,
                is_empty=False,
                saved_file="last_query_result.csv",
                mode="preview",
            )
        except Exception as e:
            logger.error("execute_query error: %s", e)
            raise RuntimeError(f"Ошибка выполнения запроса: {e}") from e

    @tool
    def get_row_count(schema: str, table: str) -> str:
        """Return table row count."""
        try:
            count = db_manager.get_row_count(schema, table)
            return f"Таблица {schema}.{table}: {count:,} строк"
        except Exception as e:
            logger.error("get_row_count error: %s", e)
            raise RuntimeError(f"Ошибка: {e}") from e

    @tool
    def check_key_uniqueness(schema: str, table: str, columns: str) -> str:
        """Check key uniqueness in CSV metadata (fallback to DB)."""
        cols = [c.strip() for c in columns.split(",")]
        if schema_loader is not None:
            result = schema_loader.check_key_uniqueness(schema, table, cols)
            if result.get("error"):
                raise ValueError(result["error"])
            if result["is_unique"]:
                reason = (
                    "все колонки являются PK"
                    if result["all_pk"]
                    else "одна из колонок уникальна на 100%"
                )
                return (
                    f"Ключ ({columns}) в {schema}.{table} уникален ({reason}).\n"
                    + "\n".join(
                        f"  {c}: unique_perc={d.get('unique_perc', '?')}%, is_pk={d.get('is_primary_key', False)}"
                        for c, d in result["columns"].items()
                        if d.get("found")
                    )
                )
            if result.get("status") == "unknown":
                return (
                    f"Ключ ({columns}) в {schema}.{table} не получил надёжного подтверждения из CSV.\n"
                    + "\n".join(
                        f"  {c}: unique_perc={d.get('unique_perc', '?')}%, is_pk={d.get('is_primary_key', False)}"
                        for c, d in result["columns"].items()
                        if d.get("found")
                    )
                )
            return (
                f"Ключ ({columns}) в {schema}.{table} НЕ уникален.\n"
                f"Минимальный unique_perc среди колонок: {result['min_unique_perc']}% "
                f"(дублей ~{result['duplicate_pct']}%)\n"
                + "\n".join(
                    f"  {c}: unique_perc={d.get('unique_perc', '?')}%, is_pk={d.get('is_primary_key', False)}"
                    for c, d in result["columns"].items()
                    if d.get("found")
                )
            )

        try:
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
            raise RuntimeError(f"Ошибка: {e}") from e

    @tool
    def get_sample(schema: str, table: str, n: int = 10) -> str:
        """Return sample rows from table."""
        try:
            df = db_manager.get_sample(schema, table, n)
            if df.empty:
                return f"Таблица {schema}.{table} пуста."

            total = len(df)
            if total <= PREVIEW_ROWS:
                return df.to_markdown(index=False)

            preview = df.head(PREVIEW_ROWS).to_markdown(index=False)
            auto_file = resolve_workspace_path(WORKSPACE_DIR, f"sample_{schema}_{table}.csv")
            df.to_csv(auto_file, index=False, encoding="utf-8")
            return (
                f"{preview}\n\n"
                f"... показано {PREVIEW_ROWS} из {total} строк.\n"
                f"Полный результат сохранен в sample_{schema}_{table}.csv"
            )
        except Exception as e:
            logger.error("get_sample error: %s", e)
            raise RuntimeError(f"Ошибка: {e}") from e

    @tool
    def explain_query(sql: str) -> str:
        """Run EXPLAIN without executing query."""
        try:
            plan = db_manager.explain_query(sql)
            return f"План выполнения:\n{plan}"
        except Exception as e:
            logger.error("explain_query error: %s", e)
            raise RuntimeError(f"Ошибка: {e}") from e

    @tool
    def table_exists(schema: str, table: str) -> str:
        """Check table exists."""
        try:
            exists = db_manager.table_exists(schema, table)
            if exists:
                return f"Таблица {schema}.{table} существует."
            return f"Таблица {schema}.{table} НЕ найдена."
        except Exception as e:
            logger.error("table_exists error: %s", e)
            raise RuntimeError(f"Ошибка: {e}") from e

    @tool
    def get_table_ddl(schema: str, table: str) -> str:
        """Get table DDL."""
        if schema_loader is not None:
            return schema_loader.generate_ddl(schema, table)
        try:
            return db_manager.get_table_ddl(schema, table)
        except Exception as e:
            logger.error("get_table_ddl error: %s", e)
            raise RuntimeError(f"Ошибка: {e}") from e

    @tool
    def execute_write(sql: str) -> str:
        """Run INSERT/UPDATE/DELETE and return structured payload."""
        try:
            affected = db_manager.execute_write(sql)
            return _build_sql_tool_payload(
                message=f"Запрос выполнен. Затронуто строк: {affected}",
                preview_markdown="",
                total_rows=affected,
                is_empty=affected == 0,
                saved_file=None,
                mode="write",
            )
        except Exception as e:
            logger.error("execute_write error: %s", e)
            raise RuntimeError(f"Ошибка: {e}") from e

    @tool
    def estimate_affected_rows(where_clause: str, schema: str, table: str) -> str:
        """Disabled for security reasons."""
        _ = (where_clause, schema, table)
        raise RuntimeError(
            "Инструмент отключен по соображениям безопасности: произвольный where_clause не поддерживается."
        )

    @tool
    def execute_ddl(sql: str) -> str:
        """Run DDL query with optional validator checks."""
        try:
            if sql_validator is not None:
                validation = sql_validator.validate(sql)
                if not validation.is_valid:
                    raise ValueError(f"DDL отклонён валидатором:\n{validation.summary()}")
                if validation.needs_confirmation:
                    raise PermissionError(
                        f"DDL требует подтверждения: {validation.confirmation_message}\n"
                        "Используйте команду подтверждения перед выполнением."
                    )
            ddl_message = db_manager.execute_ddl(sql)
            return _build_sql_tool_payload(
                message=ddl_message,
                preview_markdown="",
                total_rows=0,
                is_empty=True,
                saved_file=None,
                mode="ddl",
            )
        except Exception as e:
            logger.error("execute_ddl error: %s", e)
            raise RuntimeError(f"Ошибка: {e}") from e

    @tool
    def export_query(sql: str, filename: str, output_format: str = "csv") -> str:
        """Run SELECT and save result to workspace file."""
        try:
            file_path, row_count = db_manager.export_query_to_file(
                sql=sql,
                filename=filename,
                output_format=output_format,
                workspace_dir=WORKSPACE_DIR,
            )
            logger.info("Экспорт: %s (%d строк)", file_path, row_count)
            rel_path = str(file_path.relative_to(WORKSPACE_DIR))
            return _build_sql_tool_payload(
                message=f"Сохранено в {rel_path} ({row_count} строк)",
                preview_markdown="",
                total_rows=row_count,
                is_empty=row_count == 0,
                saved_file=rel_path,
                mode="export",
            )
        except Exception as e:
            logger.error("export_query error: %s", e)
            raise RuntimeError(f"Ошибка экспорта: {e}") from e

    tools_list = [
        execute_query,
        get_row_count,
        check_key_uniqueness,
        get_sample,
        explain_query,
        table_exists,
        get_table_ddl,
        execute_write,
        execute_ddl,
        export_query,
    ]
    return tools_list
