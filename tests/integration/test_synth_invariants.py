"""Инварианты синтетических данных.

Эти тесты не зависят от GigaChat: проверяют корректность DDL и наполнения
через `DatabaseManager` поверх тестового Postgres. Запуск:

    docker compose -f tests/integration/docker-compose.test.yml up -d
    pytest tests/integration/test_synth_invariants.py -m integration -v
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from tests.integration.synth.ddl_generator import load_metadata


def test_all_tables_created(seeded_db):
    """Все таблицы из метаданных существуют в БД."""
    tables = load_metadata()
    missing = [
        tm.full
        for tm in tables.values()
        if not seeded_db.table_exists(tm.schema, tm.name)
    ]
    assert not missing, f"Не созданы таблицы: {missing}"


def test_all_tables_have_rows(seeded_db):
    """В каждой таблице есть хотя бы одна строка."""
    tables = load_metadata()
    empty = []
    for tm in tables.values():
        cnt = seeded_db.get_row_count(tm.schema, tm.name)
        if cnt == 0:
            empty.append(tm.full)
    assert not empty, f"Пустые таблицы: {empty}"


def test_not_null_columns_respected(seeded_db):
    """Колонки is_not_null=True не содержат NULL."""
    tables = load_metadata()
    violations: list[str] = []
    engine = seeded_db.get_engine()
    with engine.connect() as conn:
        for tm in tables.values():
            for c in tm.columns:
                if not c.is_not_null:
                    continue
                sql = text(
                    f'SELECT COUNT(*) FROM "{tm.schema}"."{tm.name}" '
                    f'WHERE "{c.name}" IS NULL'
                )
                n = conn.execute(sql).scalar()
                if n:
                    violations.append(f"{tm.full}.{c.name}: {n} NULL")
    assert not violations, "\n".join(violations)


def test_explicit_fks_resolve(seeded_db):
    """Все явные FK значения присутствуют в целевой таблице."""
    tables = load_metadata()
    failures: list[str] = []
    engine = seeded_db.get_engine()
    with engine.connect() as conn:
        for tm in tables.values():
            for c in tm.columns:
                if not c.fk_target:
                    continue
                parts = c.fk_target.split(".")
                if len(parts) != 3:
                    continue
                ref_schema, ref_table, ref_col = parts
                sql = text(
                    f'SELECT COUNT(*) FROM "{tm.schema}"."{tm.name}" t '
                    f'LEFT JOIN "{ref_schema}"."{ref_table}" r '
                    f'ON t."{c.name}" = r."{ref_col}" '
                    f'WHERE t."{c.name}" IS NOT NULL AND r."{ref_col}" IS NULL'
                )
                unresolved = conn.execute(sql).scalar()
                if unresolved:
                    failures.append(
                        f"{tm.full}.{c.name} → {c.fk_target}: {unresolved} "
                        "значений не находят целевую строку"
                    )
    assert not failures, "\n".join(failures)


def test_implicit_join_intersection(seeded_db):
    """Неявный join fact_outflow ↔ dim_gosb по (tb_id, gosb_id) ↔ (tb_id, old_gosb_id)
    должен возвращать хотя бы несколько строк — иначе e2e-кейсы упадут."""
    tables = load_metadata()
    fact = tables[next(k for k in tables if k.endswith(".uzp_dwh_fact_outflow"))]
    dim = tables[next(k for k in tables if k.endswith(".uzp_dim_gosb"))]
    sql = text(
        f'SELECT COUNT(*) AS n '
        f'FROM "{fact.schema}"."{fact.name}" f '
        f'JOIN "{dim.schema}"."{dim.name}" d '
        f'  ON d."tb_id" = f."tb_id" AND d."old_gosb_id" = f."gosb_id"'
    )
    with seeded_db.get_engine().connect() as conn:
        n = conn.execute(sql).scalar()
    assert n > 0, "JOIN fact_outflow × dim_gosb пуст — синтетика не пересекает ключи"


def test_pk_uniqueness(seeded_db):
    """В таблицах с PK не должно быть дублей по PK."""
    tables = load_metadata()
    violations: list[str] = []
    with seeded_db.get_engine().connect() as conn:
        for tm in tables.values():
            pk = tm.pk_columns()
            if not pk:
                continue
            cols_sql = ", ".join(f'"{c.name}"' for c in pk)
            sql = text(
                f'SELECT COUNT(*) - COUNT(DISTINCT ({cols_sql})) '
                f'FROM "{tm.schema}"."{tm.name}"'
            )
            dups = conn.execute(sql).scalar()
            if dups:
                violations.append(f"{tm.full} PK({cols_sql}): {dups} дублей")
    assert not violations, "\n".join(violations)
