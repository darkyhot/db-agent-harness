"""Загрузка сгенерированных строк в тестовый Postgres.

Использует SQLAlchemy и существующий ``DatabaseManager`` — никаких прямых
psycopg2 импортов, чтобы конфиг и логика подключения оставались в одном месте.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

from sqlalchemy import text

from core.database import DatabaseManager
from tests.integration.synth.ddl_generator import (
    TableMeta,
    build_alter_fk_sql,
    build_create_table_sql,
    build_drop_schemas_sql,
    build_schemas_sql,
)

logger = logging.getLogger(__name__)

# bulk insert батчами — чтобы не упереться в лимит размера запроса.
INSERT_BATCH_SIZE = 500


def _quote_ident(name: str) -> str:
    return f'"{name.replace(chr(34), chr(34) * 2)}"'


def execute_script(db: DatabaseManager, statements: Iterable[str]) -> None:
    """Выполнить последовательность DDL/SQL в одном соединении."""
    engine = db.get_engine()
    with engine.connect() as conn:
        for stmt in statements:
            if not stmt or not stmt.strip():
                continue
            conn.execute(text(stmt))
        conn.commit()


def drop_and_create_schema(
    db: DatabaseManager, tables: dict[str, TableMeta]
) -> None:
    """Полный пересоздать схему: DROP CASCADE → CREATE SCHEMA → CREATE TABLE → FK."""
    drops = build_drop_schemas_sql(tables)
    creates_schema = build_schemas_sql(tables)
    creates_tables = [build_create_table_sql(tm) for tm in tables.values()]
    fk_stmts: list[str] = []
    for tm in tables.values():
        fk_stmts.extend(build_alter_fk_sql(tm))
    logger.info(
        "DDL: drops=%d, schemas=%d, tables=%d, fks=%d",
        len(drops), len(creates_schema), len(creates_tables), len(fk_stmts),
    )
    execute_script(db, drops)
    execute_script(db, creates_schema)
    execute_script(db, creates_tables)
    execute_script(db, fk_stmts)


def insert_rows(
    db: DatabaseManager,
    tm: TableMeta,
    rows: list[dict[str, Any]],
) -> int:
    """Bulk-INSERT через executemany. Возвращает число вставленных строк."""
    if not rows:
        return 0
    col_names = [c.name for c in tm.columns]
    cols_sql = ", ".join(_quote_ident(n) for n in col_names)
    placeholders = ", ".join(f":{n}" for n in col_names)
    insert_sql = (
        f'INSERT INTO {_quote_ident(tm.schema)}.{_quote_ident(tm.name)} '
        f'({cols_sql}) VALUES ({placeholders})'
    )
    stmt = text(insert_sql)
    engine = db.get_engine()
    inserted = 0
    with engine.connect() as conn:
        for i in range(0, len(rows), INSERT_BATCH_SIZE):
            chunk = rows[i:i + INSERT_BATCH_SIZE]
            conn.execute(stmt, chunk)
            inserted += len(chunk)
        conn.commit()
    return inserted


def load_all(
    db: DatabaseManager,
    order: list[TableMeta],
    data: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    """Загрузить все таблицы в порядке topo-sort. Возвращает {full_name: rowcount}."""
    counts: dict[str, int] = {}
    for tm in order:
        rows = data.get(tm.full, [])
        counts[tm.full] = insert_rows(db, tm, rows)
        logger.info("Загружено %d строк в %s", counts[tm.full], tm.full)
    return counts
