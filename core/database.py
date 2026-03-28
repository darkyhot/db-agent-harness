"""РџРѕРґРєР»СЋС‡РµРЅРёРµ Рє Greenplum (PostgreSQL-СЃРѕРІРјРµСЃС‚РёРјС‹Р№) Рё РІС‹РїРѕР»РЅРµРЅРёРµ Р·Р°РїСЂРѕСЃРѕРІ."""

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

# Regex РґР»СЏ РІР°Р»РёРґР°С†РёРё РёРґРµРЅС‚РёС„РёРєР°С‚РѕСЂРѕРІ (СЃС…РµРјР°, С‚Р°Р±Р»РёС†Р°, РєРѕР»РѕРЅРєР°)
_IDENTIFIER_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

# РўР°Р№РјР°СѓС‚ РЅР° SQL-Р·Р°РїСЂРѕСЃС‹ (РјСЃ)
STATEMENT_TIMEOUT_MS = 300_000  # 5 РјРёРЅСѓС‚


def _has_top_level_limit(sql: str) -> bool:
    """РџСЂРѕРІРµСЂРёС‚СЊ РЅР°Р»РёС‡РёРµ LIMIT РЅР° РІРµСЂС…РЅРµРј СѓСЂРѕРІРЅРµ SQL statement."""
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
    """РџСЂРѕРІРµСЂРёС‚СЊ С‡С‚Рѕ СЃС‚СЂРѕРєР° вЂ” РґРѕРїСѓСЃС‚РёРјС‹Р№ SQL-РёРґРµРЅС‚РёС„РёРєР°С‚РѕСЂ.

    Args:
        name: РРјСЏ РґР»СЏ РїСЂРѕРІРµСЂРєРё.
        kind: РўРёРї РёРґРµРЅС‚РёС„РёРєР°С‚РѕСЂР° (РґР»СЏ СЃРѕРѕР±С‰РµРЅРёСЏ РѕР± РѕС€РёР±РєРµ).

    Returns:
        РџСЂРѕРІРµСЂРµРЅРЅРѕРµ РёРјСЏ.

    Raises:
        ValueError: Р•СЃР»Рё РёРјСЏ СЃРѕРґРµСЂР¶РёС‚ РЅРµРґРѕРїСѓСЃС‚РёРјС‹Рµ СЃРёРјРІРѕР»С‹.
    """
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(
            f"РќРµРґРѕРїСѓСЃС‚РёРјС‹Р№ {kind}: '{name}'. "
            "Р”РѕРїСѓСЃС‚РёРјС‹ С‚РѕР»СЊРєРѕ Р»Р°С‚РёРЅСЃРєРёРµ Р±СѓРєРІС‹, С†РёС„СЂС‹ Рё РїРѕРґС‡С‘СЂРєРёРІР°РЅРёРµ."
        )
    return name


class DatabaseManager:
    """РњРµРЅРµРґР¶РµСЂ РїРѕРґРєР»СЋС‡РµРЅРёСЏ Рє Greenplum С‡РµСЂРµР· SQLAlchemy."""

    def __init__(self, config_path: Path | None = None) -> None:
        """РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ РёР· config.json.

        Args:
            config_path: РџСѓС‚СЊ Рє С„Р°Р№Р»Сѓ РєРѕРЅС„РёРіСѓСЂР°С†РёРё. РџРѕ СѓРјРѕР»С‡Р°РЅРёСЋ вЂ” config.json РІ РєРѕСЂРЅРµ РїСЂРѕРµРєС‚Р°.
        """
        self._config_path = config_path or CONFIG_PATH
        self._engine: Engine | None = None
        self._config: dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Р—Р°РіСЂСѓР·РєР° РєРѕРЅС„РёРіСѓСЂР°С†РёРё РёР· JSON-С„Р°Р№Р»Р°."""
        try:
            with open(self._config_path, encoding="utf-8") as f:
                self._config = json.load(f)
            logger.info("РљРѕРЅС„РёРіСѓСЂР°С†РёСЏ Р·Р°РіСЂСѓР¶РµРЅР° РёР· %s", self._config_path)
        except FileNotFoundError:
            logger.warning("Р¤Р°Р№Р» РєРѕРЅС„РёРіСѓСЂР°С†РёРё РЅРµ РЅР°Р№РґРµРЅ: %s", self._config_path)
            self._config = {}

    def save_config(
        self, user_id: str, host: str, port: int = 5432, database: str = "prom"
    ) -> None:
        """РЎРѕС…СЂР°РЅРёС‚СЊ РєРѕРЅС„РёРіСѓСЂР°С†РёСЋ РїРѕРґРєР»СЋС‡РµРЅРёСЏ РІ config.json.

        Args:
            user_id: РРјСЏ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ Р‘Р”.
            host: РҐРѕСЃС‚ СЃРµСЂРІРµСЂР°.
            port: РџРѕСЂС‚ РїРѕРґРєР»СЋС‡РµРЅРёСЏ.
            database: РРјСЏ Р±Р°Р·С‹ РґР°РЅРЅС‹С….
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
        logger.info("РљРѕРЅС„РёРіСѓСЂР°С†РёСЏ СЃРѕС…СЂР°РЅРµРЅР°: %s@%s:%d/%s", user_id, host, port, database)

    @property
    def runtime_config(self) -> dict[str, Any]:
        """РўРµРєСѓС‰Р°СЏ РєРѕРЅС„РёРіСѓСЂР°С†РёСЏ runtime (РєРѕРїРёСЏ)."""
        return dict(self._config)

    def set_debug_prompt(self, enabled: bool) -> None:
        """РћР±РЅРѕРІРёС‚СЊ С„Р»Р°Рі debug_prompt РІ РєРѕРЅС„РёРіСѓСЂР°С†РёРё."""
        self._config["debug_prompt"] = bool(enabled)

    @property
    def is_configured(self) -> bool:
        """РџСЂРѕРІРµСЂРєР° РЅР°Р»РёС‡РёСЏ РјРёРЅРёРјР°Р»СЊРЅРѕР№ РєРѕРЅС„РёРіСѓСЂР°С†РёРё."""
        return bool(self._config.get("user_id") and self._config.get("host"))

    @property
    def config_summary(self) -> str:
        """РЎС‚СЂРѕРєР° СЃ С‚РµРєСѓС‰РµР№ РєРѕРЅС„РёРіСѓСЂР°С†РёРµР№ РґР»СЏ РѕС‚РѕР±СЂР°Р¶РµРЅРёСЏ."""
        if not self.is_configured:
            return "РЅРµ РЅР°СЃС‚СЂРѕРµРЅРѕ"
        c = self._config
        return f"{c['user_id']}@{c['host']}:{c.get('port', 5432)}/{c.get('database', 'prom')}"

    def get_engine(self) -> Engine:
        """РџРѕР»СѓС‡РёС‚СЊ РёР»Рё СЃРѕР·РґР°С‚СЊ SQLAlchemy engine.

        Returns:
            Р­РєР·РµРјРїР»СЏСЂ Engine.

        Raises:
            RuntimeError: Р•СЃР»Рё РєРѕРЅС„РёРіСѓСЂР°С†РёСЏ РЅРµ Р·Р°РґР°РЅР°.
        """
        if self._engine is not None:
            return self._engine

        if not self.is_configured:
            raise RuntimeError(
                "Р‘Р” РЅРµ РЅР°СЃС‚СЂРѕРµРЅР°. РСЃРїРѕР»СЊР·СѓР№С‚Рµ РєРѕРјР°РЅРґСѓ 'config' РґР»СЏ РЅР°СЃС‚СЂРѕР№РєРё РїРѕРґРєР»СЋС‡РµРЅРёСЏ."
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
        logger.info("Engine СЃРѕР·РґР°РЅ: %s@%s:%d/%s (timeout=%dms)", user, host, port, database, STATEMENT_TIMEOUT_MS)
        return self._engine

    def preview_query(self, sql: str, limit: int = 1000) -> pd.DataFrame:
        """Р’С‹РїРѕР»РЅРёС‚СЊ SELECT-Р·Р°РїСЂРѕСЃ РІ СЂРµР¶РёРјРµ preview (СЃ Р°РІС‚Рѕ-LIMIT).

        Args:
            sql: SQL-Р·Р°РїСЂРѕСЃ (SELECT).
            limit: РњР°РєСЃРёРјР°Р»СЊРЅРѕРµ РєРѕР»РёС‡РµСЃС‚РІРѕ СЃС‚СЂРѕРє.

        Returns:
            DataFrame СЃ СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё.
        """
        sql_stripped = sql.strip().rstrip(";")
        if not _has_top_level_limit(sql_stripped):
            sql_stripped = f"SELECT * FROM ({sql_stripped}) _sub LIMIT :_limit"
            logger.info("Р’С‹РїРѕР»РЅРµРЅРёРµ SELECT (СЃ Р°РІС‚Рѕ-LIMIT): %s", sql_stripped[:200])
            with self.get_engine().connect() as conn:
                df = pd.read_sql(text(sql_stripped), conn, params={"_limit": limit})
        else:
            logger.info("Р’С‹РїРѕР»РЅРµРЅРёРµ SELECT: %s", sql_stripped[:200])
            with self.get_engine().connect() as conn:
                df = pd.read_sql(text(sql_stripped), conn)

        logger.info("РџРѕР»СѓС‡РµРЅРѕ СЃС‚СЂРѕРє: %d", len(df))
        return df

    def run_read_query(self, sql: str) -> pd.DataFrame:
        """Р’С‹РїРѕР»РЅРёС‚СЊ SELECT Р±РµР· Р°РІС‚Рѕ-LIMIT (full read/export mode)."""
        sql_stripped = sql.strip().rstrip(";")
        logger.info("Р’С‹РїРѕР»РЅРµРЅРёРµ SELECT Р±РµР· Р°РІС‚Рѕ-LIMIT: %s", sql_stripped[:200])
        with self.get_engine().connect() as conn:
            df = pd.read_sql(text(sql_stripped), conn)
        logger.info("РџРѕР»СѓС‡РµРЅРѕ СЃС‚СЂРѕРє (full): %d", len(df))
        return df

    def execute_query(self, sql: str, limit: int = 1000) -> pd.DataFrame:
        """РЎРѕРІРјРµСЃС‚РёРјРѕСЃС‚СЊ: execute_query = preview_query."""
        return self.preview_query(sql, limit=limit)

    def export_query(self, sql: str) -> pd.DataFrame:
        """Р’С‹РїРѕР»РЅРёС‚СЊ SELECT РґР»СЏ РїРѕР»РЅРѕР№ РІС‹РіСЂСѓР·РєРё Р±РµР· Р°РІС‚Рѕ-LIMIT."""
        return self.run_read_query(sql)

    def export_query_to_file(
        self,
        sql: str,
        filename: str,
        output_format: str,
        workspace_dir: Path,
    ) -> tuple[Path, int]:
        """Р’С‹РіСЂСѓР·РёС‚СЊ СЂРµР·СѓР»СЊС‚Р°С‚ SELECT РІ С„Р°Р№Р» РІРЅСѓС‚СЂРё workspace."""
        file_path = resolve_workspace_path(workspace_dir, filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.export_query(sql)
        if output_format == "excel":
            df.to_excel(file_path, index=False)
        else:
            df.to_csv(file_path, index=False, encoding="utf-8")
        return file_path, len(df)

    def execute_write(self, sql: str) -> int:
        """Р’С‹РїРѕР»РЅРёС‚СЊ INSERT/UPDATE/DELETE Рё РІРµСЂРЅСѓС‚СЊ РєРѕР»РёС‡РµСЃС‚РІРѕ Р·Р°С‚СЂРѕРЅСѓС‚С‹С… СЃС‚СЂРѕРє.

        Args:
            sql: SQL-Р·Р°РїСЂРѕСЃ (INSERT/UPDATE/DELETE).

        Returns:
            РљРѕР»РёС‡РµСЃС‚РІРѕ Р·Р°С‚СЂРѕРЅСѓС‚С‹С… СЃС‚СЂРѕРє.
        """
        engine = self.get_engine()
        logger.info("Р’С‹РїРѕР»РЅРµРЅРёРµ WRITE: %s", sql[:200])
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            conn.commit()
            affected = result.rowcount
        logger.info("Р—Р°С‚СЂРѕРЅСѓС‚Рѕ СЃС‚СЂРѕРє: %d", affected)
        return affected

    def execute_ddl(self, sql: str) -> str:
        """Р’С‹РїРѕР»РЅРёС‚СЊ DDL-Р·Р°РїСЂРѕСЃ (CREATE/ALTER/DROP/TRUNCATE).

        Args:
            sql: DDL-Р·Р°РїСЂРѕСЃ.

        Returns:
            РЎРѕРѕР±С‰РµРЅРёРµ РѕР± СѓСЃРїРµС€РЅРѕРј РІС‹РїРѕР»РЅРµРЅРёРё.
        """
        engine = self.get_engine()
        logger.info("Р’С‹РїРѕР»РЅРµРЅРёРµ DDL: %s", sql[:200])
        with engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()
        logger.info("DDL РІС‹РїРѕР»РЅРµРЅ СѓСЃРїРµС€РЅРѕ")
        return "DDL РІС‹РїРѕР»РЅРµРЅ СѓСЃРїРµС€РЅРѕ."

    def explain_query(self, sql: str) -> str:
        """Р’С‹РїРѕР»РЅРёС‚СЊ EXPLAIN РґР»СЏ SQL-Р·Р°РїСЂРѕСЃР° (Р±РµР· СЂРµР°Р»СЊРЅРѕРіРѕ РІС‹РїРѕР»РЅРµРЅРёСЏ).

        Args:
            sql: SQL-Р·Р°РїСЂРѕСЃ РґР»СЏ Р°РЅР°Р»РёР·Р°.

        Returns:
            РџР»Р°РЅ РІС‹РїРѕР»РЅРµРЅРёСЏ Р·Р°РїСЂРѕСЃР°.
        """
        engine = self.get_engine()
        explain_sql = f"EXPLAIN {sql.strip().rstrip(';')}"
        logger.debug("EXPLAIN: %s", explain_sql[:200])
        with engine.connect() as conn:
            result = conn.execute(text(explain_sql))
            plan = "\n".join(row[0] for row in result)
        return plan

    def get_row_count(self, schema: str, table: str) -> int:
        """РџРѕР»СѓС‡РёС‚СЊ РєРѕР»РёС‡РµСЃС‚РІРѕ СЃС‚СЂРѕРє РІ С‚Р°Р±Р»РёС†Рµ.

        Args:
            schema: РРјСЏ СЃС…РµРјС‹.
            table: РРјСЏ С‚Р°Р±Р»РёС†С‹.

        Returns:
            РљРѕР»РёС‡РµСЃС‚РІРѕ СЃС‚СЂРѕРє.
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
        logger.info("РЎС‚СЂРѕРє РІ %s.%s: %d", schema, table, count)
        return count

    def check_key_uniqueness(
        self, schema: str, table: str, columns: list[str]
    ) -> dict[str, Any]:
        """РџСЂРѕРІРµСЂРёС‚СЊ СѓРЅРёРєР°Р»СЊРЅРѕСЃС‚СЊ РєРѕРјР±РёРЅР°С†РёРё РєРѕР»РѕРЅРѕРє (РґР»СЏ РІР°Р»РёРґР°С†РёРё JOIN).

        Args:
            schema: РРјСЏ СЃС…РµРјС‹.
            table: РРјСЏ С‚Р°Р±Р»РёС†С‹.
            columns: РЎРїРёСЃРѕРє РєРѕР»РѕРЅРѕРє РґР»СЏ РїСЂРѕРІРµСЂРєРё.

        Returns:
            РЎР»РѕРІР°СЂСЊ СЃ total_rows, unique_keys, duplicate_pct.
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
        logger.info("РЈРЅРёРєР°Р»СЊРЅРѕСЃС‚СЊ %s.%s(%s): %s", schema, table, cols, result)
        return result

    def get_sample(self, schema: str, table: str, n: int = 10) -> pd.DataFrame:
        """РџРѕР»СѓС‡РёС‚СЊ РІС‹Р±РѕСЂРєСѓ СЃС‚СЂРѕРє РёР· С‚Р°Р±Р»РёС†С‹.

        Args:
            schema: РРјСЏ СЃС…РµРјС‹.
            table: РРјСЏ С‚Р°Р±Р»РёС†С‹.
            n: РљРѕР»РёС‡РµСЃС‚РІРѕ СЃС‚СЂРѕРє.

        Returns:
            DataFrame СЃ РѕР±СЂР°Р·С†РѕРј РґР°РЅРЅС‹С….
        """
        schema = _validate_identifier(schema, "schema")
        table = _validate_identifier(table, "table")
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"РќРµРґРѕРїСѓСЃС‚РёРјРѕРµ Р·РЅР°С‡РµРЅРёРµ n: {n}")
        sql = text(f'SELECT * FROM "{schema}"."{table}" LIMIT :n')
        engine = self.get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"n": n})
        return df

    def table_exists(self, schema: str, table: str) -> bool:
        """РџСЂРѕРІРµСЂРёС‚СЊ СЃСѓС‰РµСЃС‚РІРѕРІР°РЅРёРµ С‚Р°Р±Р»РёС†С‹.

        Args:
            schema: РРјСЏ СЃС…РµРјС‹.
            table: РРјСЏ С‚Р°Р±Р»РёС†С‹.

        Returns:
            True РµСЃР»Рё С‚Р°Р±Р»РёС†Р° СЃСѓС‰РµСЃС‚РІСѓРµС‚.
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
        """РџРѕР»СѓС‡РёС‚СЊ DDL (СЃС‚СЂСѓРєС‚СѓСЂСѓ) С‚Р°Р±Р»РёС†С‹ С‡РµСЂРµР· information_schema.

        Args:
            schema: РРјСЏ СЃС…РµРјС‹.
            table: РРјСЏ С‚Р°Р±Р»РёС†С‹.

        Returns:
            РўРµРєСЃС‚РѕРІРѕРµ РїСЂРµРґСЃС‚Р°РІР»РµРЅРёРµ DDL.
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
            return f"РўР°Р±Р»РёС†Р° {schema}.{table} РЅРµ РЅР°Р№РґРµРЅР°."

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
