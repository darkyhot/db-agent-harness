"""Подключение к Greenplum (PostgreSQL-совместимый) и выполнение запросов."""

import json
import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import sqlparse
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine

from core.exceptions import KERBEROS_USER_MESSAGE, KerberosAuthError, is_kerberos_auth_error
from core.log_safety import summarize_sql
from tools.path_safety import resolve_workspace_path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
DEFAULT_CONFIG: dict[str, Any] = {
    "user_id": "",
    "host": "",
    "port": 5432,
    "database": "prom",
    "llm_model": "GigaChat-2-Max",
    "debug_prompt": False,
    "show_plan": False,
    "llm_verifier_enabled": False,
}
CONNECTION_CONFIG_KEYS = ("user_id", "host", "port", "database")
RUNTIME_CONFIG_KEYS = ("llm_model", "debug_prompt", "show_plan", "llm_verifier_enabled")

# Regex для валидации идентификаторов (схема, таблица, колонка)
_IDENTIFIER_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

# Таймаут на SQL-запросы (мс).
# Синхронизирован с wall-clock таймаутом графа, чтобы долгие запросы
# не прерывались преждевременно на уровне оркестрации.
STATEMENT_TIMEOUT_MS = 600_000  # 600 секунд (10 минут)


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
        self._config: dict[str, Any] = dict(DEFAULT_CONFIG)
        self._loaded_keys: set[str] = set()
        self._config_file_exists = False
        self._load_config()

    def _load_config(self) -> None:
        """Загрузка конфигурации из JSON-файла."""
        try:
            with open(self._config_path, encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("config.json должен содержать JSON-объект")
            self._config = {**DEFAULT_CONFIG, **loaded}
            self._loaded_keys = set(loaded.keys())
            self._config_file_exists = True
            logger.info("Конфигурация загружена из %s", self._config_path)
        except FileNotFoundError:
            logger.warning("Файл конфигурации не найден: %s", self._config_path)
            self._config = dict(DEFAULT_CONFIG)
            self._loaded_keys = set()
            self._config_file_exists = False
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Некорректный файл конфигурации %s: %s", self._config_path, e)
            self._config = dict(DEFAULT_CONFIG)
            self._loaded_keys = set()
            self._config_file_exists = False

    def _write_config(self) -> None:
        """Записать текущую конфигурацию на диск."""
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=4, ensure_ascii=False)
        self._loaded_keys = set(self._config.keys())
        self._config_file_exists = True

    def save_config(
        self, user_id: str, host: str, port: int = 5432, database: str = "prom"
    ) -> None:
        """Совместимость: сохранить connection-конфигурацию в config.json.

        Args:
            user_id: Имя пользователя БД.
            host: Хост сервера.
            port: Порт подключения.
            database: Имя базы данных.
        """
        self.save_connection_config(user_id, host, port, database)

    def save_connection_config(
        self, user_id: str, host: str, port: int = 5432, database: str = "prom"
    ) -> None:
        """Сохранить только параметры подключения, не трогая runtime-флаги."""
        self._config.update({
            "user_id": user_id,
            "host": host,
            "port": port,
            "database": database,
        })
        self._write_config()
        self._engine = None
        logger.info("Конфигурация сохранена: %s@%s:%d/%s", user_id, host, port, database)

    def save_runtime_params(
        self,
        *,
        debug_prompt: bool,
        show_plan: bool,
        llm_model: str | None = None,
        llm_verifier_enabled: bool | None = None,
    ) -> None:
        """Сохранить runtime-параметры CLI/графа в config.json."""
        self._config.update({
            "debug_prompt": bool(debug_prompt),
            "show_plan": bool(show_plan),
        })
        if llm_model is not None:
            self._config["llm_model"] = str(llm_model).strip() or DEFAULT_CONFIG["llm_model"]
        if llm_verifier_enabled is not None:
            self._config["llm_verifier_enabled"] = bool(llm_verifier_enabled)
        self._write_config()
        logger.info(
            "Runtime-параметры сохранены: llm_model=%s, debug_prompt=%s, "
            "show_plan=%s, llm_verifier_enabled=%s",
            self._config.get("llm_model"),
            bool(debug_prompt),
            bool(show_plan),
            self._config.get("llm_verifier_enabled"),
        )

    @property
    def runtime_config(self) -> dict[str, Any]:
        """Текущая конфигурация runtime (копия)."""
        return dict(self._config)

    def set_debug_prompt(self, enabled: bool) -> None:
        """Обновить флаг debug_prompt в конфигурации."""
        self._config["debug_prompt"] = bool(enabled)

    @property
    def config_file_exists(self) -> bool:
        """Существует ли корректно прочитанный config.json."""
        return self._config_file_exists

    def missing_connection_fields(self) -> list[str]:
        """Список отсутствующих полей connection-секции."""
        missing: list[str] = []
        for key in CONNECTION_CONFIG_KEYS:
            value = self._config.get(key)
            if key not in self._loaded_keys:
                missing.append(key)
            elif isinstance(value, str) and not value.strip():
                missing.append(key)
        return missing

    def missing_runtime_fields(self) -> list[str]:
        """Список отсутствующих полей runtime-секции."""
        missing: list[str] = []
        for key in RUNTIME_CONFIG_KEYS:
            if key not in self._loaded_keys:
                missing.append(key)
        return missing

    @property
    def is_configured(self) -> bool:
        """Проверка наличия минимальной конфигурации."""
        return not self.missing_connection_fields()

    @property
    def has_runtime_params(self) -> bool:
        """Заполнены ли runtime-параметры в конфиге."""
        return not self.missing_runtime_fields()

    @property
    def has_complete_config(self) -> bool:
        """Полностью ли заполнен конфиг (connection + runtime)."""
        return self.is_configured and self.has_runtime_params

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
        engine = create_engine(
            url,
            pool_pre_ping=True,
            connect_args={"options": f"-c statement_timeout={STATEMENT_TIMEOUT_MS}"},
        )
        try:
            with engine.connect():
                pass
        except Exception as exc:
            engine.dispose()
            if is_kerberos_auth_error(exc):
                raise KerberosAuthError(KERBEROS_USER_MESSAGE) from exc
            raise

        self._engine = engine
        logger.info("Engine создан: %s@%s:%d/%s (timeout=%dms)", user, host, port, database, STATEMENT_TIMEOUT_MS)
        return self._engine

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        """Open a DB connection and normalize Kerberos/GSSAPI auth failures."""
        try:
            with self.get_engine().connect() as conn:
                yield conn
        except KerberosAuthError:
            raise
        except Exception as exc:
            if is_kerberos_auth_error(exc):
                raise KerberosAuthError(KERBEROS_USER_MESSAGE) from exc
            raise

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
            logger.info("Выполнение SELECT (с авто-LIMIT): %s", summarize_sql(sql_stripped))
            with self._connect() as conn:
                df = pd.read_sql(text(sql_stripped), conn, params={"_limit": limit})
        else:
            logger.info("Выполнение SELECT: %s", summarize_sql(sql_stripped))
            with self._connect() as conn:
                df = pd.read_sql(text(sql_stripped), conn)

        logger.info("Получено строк: %d", len(df))
        return df

    def run_read_query(self, sql: str) -> pd.DataFrame:
        """Выполнить SELECT без авто-LIMIT (full read/export mode)."""
        sql_stripped = sql.strip().rstrip(";")
        logger.info("Выполнение SELECT без авто-LIMIT: %s", summarize_sql(sql_stripped))
        with self._connect() as conn:
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
        logger.info("Выполнение WRITE: %s", summarize_sql(sql))
        with self._connect() as conn:
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
        logger.info("Выполнение DDL: %s", summarize_sql(sql))
        with self._connect() as conn:
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
        explain_sql = f"EXPLAIN {sql.strip().rstrip(';')}"
        logger.debug("EXPLAIN: %s", summarize_sql(explain_sql))
        with self._connect() as conn:
            result = conn.execute(text(explain_sql))
            plan = "\n".join(row[0] for row in result)
        return plan

    def get_row_count(
        self, schema: str, table: str, *, where: str | None = None
    ) -> int:
        """Получить количество строк в таблице.

        Args:
            schema: Имя схемы.
            table: Имя таблицы.
            where: опциональное SQL-условие (без слова WHERE), применяется к
                подсчёту. Должно быть предварительно пройдено через
                ``_sanitize_where_clause``; этот метод не делает повторной
                проверки и доверяет вызывающей стороне.

        Returns:
            Количество строк.
        """
        schema = _validate_identifier(schema, "schema")
        table = _validate_identifier(table, "table")
        sql_str = f'SELECT COUNT(*) as cnt FROM "{schema}"."{table}"'
        if where:
            sql_str += f" WHERE {where}"
        sql = text(sql_str)
        with self._connect() as conn:
            result = conn.execute(sql)
            count = result.scalar()
        logger.info("Строк в %s.%s%s: %d",
                    schema, table, f" WHERE {where}" if where else "", count)
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
        with self._connect() as conn:
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
        logger.info(
            "Уникальность %s.%s(%s): total_rows=%s unique_keys=%s duplicate_pct=%.2f is_unique=%s",
            schema, table, cols, total, unique, dup_pct, dup_pct == 0.0,
        )
        return result

    def get_sample(
        self,
        schema: str,
        table: str,
        n: int = 10,
        *,
        where: str | None = None,
    ) -> pd.DataFrame:
        """Получить выборку строк из таблицы.

        Args:
            schema: Имя схемы.
            table: Имя таблицы.
            n: Количество строк.
            where: опциональное SQL-условие (без WHERE).

        Returns:
            DataFrame с образцом данных.
        """
        schema = _validate_identifier(schema, "schema")
        table = _validate_identifier(table, "table")
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"Недопустимое значение n: {n}")
        sql_str = f'SELECT * FROM "{schema}"."{table}"'
        if where:
            sql_str += f" WHERE {where}"
        sql_str += " LIMIT :n"
        sql = text(sql_str)
        with self._connect() as conn:
            df = pd.read_sql(sql, conn, params={"n": n})
        return df

    def get_random_sample(
        self,
        schema: str,
        table: str,
        n: int = 100_000,
        columns: list[str] | None = None,
        *,
        where: str | None = None,
    ) -> pd.DataFrame:
        """Получить random sample строк из таблицы."""
        schema = _validate_identifier(schema, "schema")
        table = _validate_identifier(table, "table")
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"Недопустимое значение n: {n}")
        projection = "*"
        if columns:
            safe_columns = [_validate_identifier(col, "column") for col in columns]
            projection = ", ".join(f'"{col}"' for col in safe_columns)
        sql_str = f'SELECT {projection} FROM "{schema}"."{table}"'
        if where:
            sql_str += f" WHERE {where}"
        sql_str += " ORDER BY random() LIMIT :n"
        sql = text(sql_str)
        with self._connect() as conn:
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
        with self._connect() as conn:
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
        with self._connect() as conn:
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
        """Deprecated and disabled.

        Security boundary: this method does not accept arbitrary SQL fragments.
        """
        _ = (where_clause, schema, table)
        raise RuntimeError(
            "count_affected_rows_readonly отключен: произвольный where_clause не поддерживается."
        )

    def estimate_affected_rows(self, where_clause: str, schema: str, table: str) -> int:
        """Deprecated compatibility alias."""
        return self.count_affected_rows_readonly(where_clause, schema, table)
