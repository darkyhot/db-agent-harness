"""Подключение к Greenplum (PostgreSQL-совместимый) и выполнение запросов."""

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import sqlparse
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine

from tools.path_safety import resolve_workspace_path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

# Regex для валидации идентификаторов (схема, таблица, колонка)
_IDENTIFIER_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

# Таймаут на SQL-запросы (мс)
STATEMENT_TIMEOUT_MS = 300_000  # 5 минут


def _has_top_level_limit(sql: str) -> bool:
    """Проверить наличие LIMIT на верхнем уровне SQL statement."""
    statements = sqlparse.parse(sql)
    if not statements:
        return False

    statement = statements[0]
    for token in statement.tokens:
        if token.is_whitespace:
            continue
        if token.ttype in sqlparse.tokens.Keyword and token.normalized == "LIMIT":
            return True
    return False


def _validate_identifier(name: str, kind: str = "identifier") -> str:
    """Проверить что строка — допустимый SQL-идентификатор.

    Args:
        name: Имя для проверки.
        kind: Тип идентификатора (для сообщения об ошибке).

    Returns:
        Проверенное имя.

    Raises:
        ValueError: Если имя содержит недопустимые символы.
    """
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Недопустимый {kind}: '{name}'. "
            "Допустимы только латинские буквы, цифры и подчёркивание."
        )
    return name


class DatabaseManager:
    """Менеджер подключения к Greenplum через SQLAlchemy."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Инициализация из config.json.

        Args:
            config_path: Путь к файлу конфигурации. По умолчанию — config.json в корне проекта.
        """
        self._config_path = config_path or CONFIG_PATH
        self._engine: Engine | None = None
        self._config: dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Загрузка конфигурации из JSON-файла."""
        try:
            with open(self._config_path, encoding="utf-8") as f:
                self._config = json.load(f)
            logger.info("Конфигурация загружена из %s", self._config_path)
        except FileNotFoundError:
            logger.warning("Файл конфигурации не найден: %s", self._config_path)
            self._config = {}

    def save_config(
        self, user_id: str, host: str, port: int = 5432, database: str = "prom"
    ) -> None:
        """Сохранить конфигурацию подключения в config.json.

        Args:
            user_id: Имя пользователя БД.
            host: Хост сервера.
            port: Порт подключения.
            database: Имя базы данных.
        """
        self._config = {
            "user_id": user_id,
            "host": host,
            "port": port,
            "database": database,
            "debug_prompt": self._config.get("debug_prompt", False),
        }
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=4, ensure_ascii=False)
        self._engine = None
        logger.info("Конфигурация сохранена: %s@%s:%d/%s", user_id, host, port, database)

    @property
    def runtime_config(self) -> dict[str, Any]:
        """Текущая конфигурация runtime (копия)."""
        return dict(self._config)

    def set_debug_prompt(self, enabled: bool) -> None:
        """Обновить флаг debug_prompt в конфигурации."""
        self._config["debug_prompt"] = bool(enabled)

    @property
    def is_configured(self) -> bool:
        """Проверка наличия минимальной конфигурации."""
        return bool(self._config.get("user_id") and self._config.get("host"))

    @property
    def config_summary(self) -> str:
        """Строка с текущей конфигурацией для отображения."""
        if not self.is_configured:
            return "не настроено"
        c = self._config
        return f"{c['user_id']}@{c['host']}:{c.get('port', 5432)}/{c.get('database', 'prom')}"

    def get_engine(self) -> Engine:
        """Получить или создать SQLAlchemy engine.

        Returns:
            Экземпляр Engine.

        Raises:
            RuntimeError: Если конфигурация не задана.
        """
        if self._engine is not None:
            return self._engine

        if not self.is_configured:
            raise RuntimeError(
                "БД не настроена. Используйте команду 'config' для настройки подключения."
            )

        user = self._config["user_id"]
        host = self._config["host"]
        port = self._config.get("port", 5432)
        database = self._config.get("database", "prom")

        url = f"postgresql://{user}@{host}:{port}/{database}"
        self._engine = create_engine(
            url,
            pool_pre_ping=True,
            connect_args={"options": f"-c statement_timeout={STATEMENT_TIMEOUT_MS}"},
        )
        logger.info("Engine создан: %s@%s:%d/%s (timeout=%dms)", user, host, port, database, STATEMENT_TIMEOUT_MS)
        return self._engine

    def preview_query(self, sql: str, limit: int = 1000) -> pd.DataFrame:
        """Выполнить SELECT-запрос в режиме preview (с авто-LIMIT).

        Args:
            sql: SQL-запрос (SELECT).
            limit: Максимальное количество строк.

        Returns:
            DataFrame с результатами.
        """
        sql_stripped = sql.strip().rstrip(";")
        if not _has_top_level_limit(sql_stripped):
            sql_stripped = f"SELECT * FROM ({sql_stripped}) _sub LIMIT :_limit"
            logger.info("Выполнение SELECT (с авто-LIMIT): %s", sql_stripped[:200])
            with self.get_engine().connect() as conn:
                df = pd.read_sql(text(sql_stripped), conn, params={"_limit": limit})
        else:
            logger.info("Выполнение SELECT: %s", sql_stripped[:200])
            with self.get_engine().connect() as conn:
                df = pd.read_sql(text(sql_stripped), conn)

        logger.info("Получено строк: %d", len(df))
        return df

    def run_read_query(self, sql: str) -> pd.DataFrame:
        """Выполнить SELECT без авто-LIMIT (full read/export mode)."""
        sql_stripped = sql.strip().rstrip(";")
        logger.info("Выполнение SELECT без авто-LIMIT: %s", sql_stripped[:200])
        with self.get_engine().connect() as conn:
            df = pd.read_sql(text(sql_stripped), conn)
        logger.info("Получено строк (full): %d", len(df))
        return df

    def execute_query(self, sql: str, limit: int = 1000) -> pd.DataFrame:
        """Совместимость: execute_query = preview_query."""
        return self.preview_query(sql, limit=limit)

    def export_query(self, sql: str) -> pd.DataFrame:
        """Выполнить SELECT для полной выгрузки без авто-LIMIT."""
        return self.run_read_query(sql)

    def export_query_to_file(
        self,
        sql: str,
        filename: str,
        output_format: str,
        workspace_dir: Path,
    ) -> tuple[Path, int]:
        """Выгрузить результат SELECT в файл внутри workspace."""
        file_path = resolve_workspace_path(workspace_dir, filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.export_query(sql)
        if output_format == "excel":
            df.to_excel(file_path, index=False)
        else:
            df.to_csv(file_path, index=False, encoding="utf-8")
        return file_path, len(df)

    def execute_write(self, sql: str) -> int:
        """Выполнить INSERT/UPDATE/DELETE и вернуть количество затронутых строк.

        Args:
            sql: SQL-запрос (INSERT/UPDATE/DELETE).

        Returns:
            Количество затронутых строк.
        """
        engine = self.get_engine()
        logger.info("Выполнение WRITE: %s", sql[:200])
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            conn.commit()
            affected = result.rowcount
        logger.info("Затронуто строк: %d", affected)
        return affected

    def execute_ddl(self, sql: str) -> str:
        """Выполнить DDL-запрос (CREATE/ALTER/DROP/TRUNCATE).

        Args:
            sql: DDL-запрос.

        Returns:
            Сообщение об успешном выполнении.
        """
        engine = self.get_engine()
        logger.info("Выполнение DDL: %s", sql[:200])
        with engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()
        logger.info("DDL выполнен успешно")
        return "DDL выполнен успешно."

    def explain_query(self, sql: str) -> str:
        """Выполнить EXPLAIN для SQL-запроса (без реального выполнения).

        Args:
            sql: SQL-запрос для анализа.

        Returns:
            План выполнения запроса.
        """
        engine = self.get_engine()
        explain_sql = f"EXPLAIN {sql.strip().rstrip(';')}"
        logger.debug("EXPLAIN: %s", explain_sql[:200])
        with engine.connect() as conn:
            result = conn.execute(text(explain_sql))
            plan = "\n".join(row[0] for row in result)
        return plan

    def get_row_count(self, schema: str, table: str) -> int:
        """Получить количество строк в таблице.

        Args:
            schema: Имя схемы.
            table: Имя таблицы.

        Returns:
            Количество строк.
        """
        schema = _validate_identifier(schema, "schema")
        table = _validate_identifier(table, "table")
        sql = text(
            f'SELECT COUNT(*) as cnt FROM "{schema}"."{table}"'
        )
        engine = self.get_engine()
        with engine.connect() as conn:
            result = conn.execute(sql)
            count = result.scalar()
        logger.info("Строк в %s.%s: %d", schema, table, count)
        return count

    def check_key_uniqueness(
        self, schema: str, table: str, columns: list[str]
    ) -> dict[str, Any]:
        """Проверить уникальность комбинации колонок (для валидации JOIN).

        Args:
            schema: Имя схемы.
            table: Имя таблицы.
            columns: Список колонок для проверки.

        Returns:
            Словарь с total_rows, unique_keys, duplicate_pct.
        """
        schema = _validate_identifier(schema, "schema")
        table = _validate_identifier(table, "table")
        for col in columns:
            _validate_identifier(col, "column")

        cols = ", ".join(f'"{c}"' for c in columns)
        sql = text(f"""
            SELECT
                COUNT(*) as total_rows,
                COUNT(DISTINCT ({cols})) as unique_keys
            FROM "{schema}"."{table}"
        """)
        engine = self.get_engine()
        with engine.connect() as conn:
            row = conn.execute(sql).fetchone()

        total = row[0]
        unique = row[1]
        dup_pct = round((1 - unique / total) * 100, 2) if total > 0 else 0.0

        result = {
            "total_rows": total,
            "unique_keys": unique,
            "duplicate_pct": dup_pct,
            "is_unique": dup_pct == 0.0,
        }
        logger.info("Уникальность %s.%s(%s): %s", schema, table, cols, result)
        return result

    def get_sample(self, schema: str, table: str, n: int = 10) -> pd.DataFrame:
        """Получить выборку строк из таблицы.

        Args:
            schema: Имя схемы.
            table: Имя таблицы.
            n: Количество строк.

        Returns:
            DataFrame с образцом данных.
        """
        schema = _validate_identifier(schema, "schema")
        table = _validate_identifier(table, "table")
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"Недопустимое значение n: {n}")
        sql = text(f'SELECT * FROM "{schema}"."{table}" LIMIT :n')
        engine = self.get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"n": n})
        return df

    def table_exists(self, schema: str, table: str) -> bool:
        """Проверить существование таблицы.

        Args:
            schema: Имя схемы.
            table: Имя таблицы.

        Returns:
            True если таблица существует.
        """
        sql = """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = :schema AND table_name = :table
            )
        """
        engine = self.get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql), {"schema": schema, "table": table})
            return result.scalar()

    def get_table_ddl(self, schema: str, table: str) -> str:
        """Получить DDL (структуру) таблицы через information_schema.

        Args:
            schema: Имя схемы.
            table: Имя таблицы.

        Returns:
            Текстовое представление DDL.
        """
        sql = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = :schema AND table_name = :table
            ORDER BY ordinal_position
        """
        engine = self.get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params={"schema": schema, "table": table})

        if df.empty:
            return f"Таблица {schema}.{table} не найдена."

        lines = [f'CREATE TABLE "{schema}"."{table}" (']
        for _, row in df.iterrows():
            nullable = "" if row["is_nullable"] == "YES" else " NOT NULL"
            default = f" DEFAULT {row['column_default']}" if row["column_default"] else ""
            lines.append(f'    "{row["column_name"]}" {row["data_type"]}{nullable}{default},')
        lines[-1] = lines[-1].rstrip(",")
        lines.append(");")
        return "\n".join(lines)

    def count_affected_rows_readonly(
        self, where_clause: str, schema: str, table: str
    ) -> int:
        """Точно посчитать количество строк, затронутых WHERE-условием.

        Выполняет `COUNT(*)` в read-only транзакции: это точное значение,
        а не приблизительная оценка.

        Args:
            where_clause: Условие WHERE (без ключевого слова WHERE).
            schema: Имя схемы.
            table: Имя таблицы.

        Returns:
            Точное количество строк.
        """
        schema = _validate_identifier(schema, "schema")
        table = _validate_identifier(table, "table")
        # Выполняем COUNT в read-only транзакции для защиты от SQL-инъекций
        # через where_clause (даже если where_clause содержит деструктивный подзапрос,
        # read-only транзакция не позволит изменить данные)
        count_sql = f'SELECT COUNT(*) FROM "{schema}"."{table}" WHERE {where_clause}'
        engine = self.get_engine()
        try:
            with engine.connect() as conn:
                conn.execute(text("SET TRANSACTION READ ONLY"))
                result = conn.execute(text(count_sql))
                count = result.scalar()
                conn.rollback()  # завершаем read-only транзакцию без commit
                return count
        except Exception as e:
            logger.error("count_affected_rows_readonly ошибка: %s", e)
            raise

    def estimate_affected_rows(self, where_clause: str, schema: str, table: str) -> int:
        """Совместимость: устаревший алиас для count_affected_rows_readonly."""
        return self.count_affected_rows_readonly(where_clause, schema, table)
