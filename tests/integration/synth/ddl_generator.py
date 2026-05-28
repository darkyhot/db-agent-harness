"""Генератор DDL по метаданным data_for_agent/{tables_list,attr_list}.csv.

Строит:
  - CREATE SCHEMA IF NOT EXISTS <schema>;
  - CREATE TABLE <schema>.<table> (col TYPE [NOT NULL], ...);
  - Композитный PRIMARY KEY если несколько колонок is_primary_key=True.
  - Явные FOREIGN KEY на основании foreign_key_target = "schema.table.column".

Прод-код не трогает.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from tests.integration.synth.type_mapping import SqlType, resolve

DATA_DIR_DEFAULT = Path(__file__).resolve().parents[3] / "data_for_agent"


@dataclass
class ColumnMeta:
    schema: str
    table: str
    name: str
    dtype: str
    is_not_null: bool
    is_primary_key: bool
    fk_target: str | None  # "schema.table.column" или None
    sample_values: str
    partition_key: bool
    sql_type: SqlType = field(init=False)

    def __post_init__(self) -> None:
        self.sql_type = resolve(self.dtype)


@dataclass
class TableMeta:
    schema: str
    name: str
    description: str
    grain: str
    columns: list[ColumnMeta] = field(default_factory=list)

    @property
    def full(self) -> str:
        return f"{self.schema}.{self.name}"

    def pk_columns(self) -> list[ColumnMeta]:
        return [c for c in self.columns if c.is_primary_key]


def _read_tables(path: Path) -> dict[str, TableMeta]:
    out: dict[str, TableMeta] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tm = TableMeta(
                schema=row["schema_name"].strip(),
                name=row["table_name"].strip(),
                description=row.get("description", "").strip(),
                grain=row.get("grain", "").strip(),
            )
            out[tm.full] = tm
    return out


def _to_bool(v: str | None) -> bool:
    return (v or "").strip().lower() == "true"


def _read_columns(path: Path, tables: dict[str, TableMeta]) -> None:
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            schema = row["schema_name"].strip()
            table = row["table_name"].strip()
            full = f"{schema}.{table}"
            tm = tables.get(full)
            if tm is None:
                continue
            fk_raw = (row.get("foreign_key_target") or "").strip()
            # FK валиден только если выглядит как "schema.table.column".
            fk_target = fk_raw if fk_raw.count(".") == 2 else None
            tm.columns.append(
                ColumnMeta(
                    schema=schema,
                    table=table,
                    name=row["column_name"].strip(),
                    dtype=row["dType"].strip(),
                    is_not_null=_to_bool(row.get("is_not_null")),
                    is_primary_key=_to_bool(row.get("is_primary_key")),
                    fk_target=fk_target,
                    sample_values=(row.get("sample_values") or "").strip(),
                    partition_key=_to_bool(row.get("partition_key")),
                )
            )


def load_metadata(data_dir: Path | None = None) -> dict[str, TableMeta]:
    """Прочитать tables_list.csv + attr_list.csv в словарь {full_name: TableMeta}."""
    d = data_dir or DATA_DIR_DEFAULT
    tables = _read_tables(d / "tables_list.csv")
    _read_columns(d / "attr_list.csv", tables)
    # Отфильтровать таблицы без колонок (защита от расхождений данных).
    return {k: v for k, v in tables.items() if v.columns}


def _quote_ident(name: str) -> str:
    """Безопасно процитировать идентификатор Postgres."""
    safe = name.replace('"', '""')
    return f'"{safe}"'


def build_create_table_sql(tm: TableMeta) -> str:
    """Сгенерировать CREATE TABLE без FK (FK добавляются вторым проходом)."""
    cols_sql: list[str] = []
    for c in tm.columns:
        nullable = " NOT NULL" if c.is_not_null else ""
        cols_sql.append(f'    {_quote_ident(c.name)} {c.sql_type.pg_type}{nullable}')
    pk_cols = tm.pk_columns()
    if pk_cols:
        pk_names = ", ".join(_quote_ident(c.name) for c in pk_cols)
        cols_sql.append(f'    PRIMARY KEY ({pk_names})')
    body = ",\n".join(cols_sql)
    return (
        f'CREATE TABLE IF NOT EXISTS {_quote_ident(tm.schema)}.{_quote_ident(tm.name)} (\n'
        f'{body}\n);'
    )


def build_alter_fk_sql(tm: TableMeta) -> list[str]:
    """Сгенерировать ALTER TABLE ... ADD CONSTRAINT FOREIGN KEY ... для каждой FK-колонки."""
    out: list[str] = []
    for c in tm.columns:
        if not c.fk_target:
            continue
        parts = c.fk_target.split(".")
        if len(parts) != 3:
            continue
        ref_schema, ref_table, ref_col = parts
        constraint_name = f"fk_{tm.name}_{c.name}"
        out.append(
            f'ALTER TABLE {_quote_ident(tm.schema)}.{_quote_ident(tm.name)} '
            f'ADD CONSTRAINT {_quote_ident(constraint_name)} '
            f'FOREIGN KEY ({_quote_ident(c.name)}) '
            f'REFERENCES {_quote_ident(ref_schema)}.{_quote_ident(ref_table)}({_quote_ident(ref_col)});'
        )
    return out


def build_schemas_sql(tables: dict[str, TableMeta]) -> list[str]:
    """CREATE SCHEMA для всех уникальных схем."""
    schemas = sorted({tm.schema for tm in tables.values()})
    return [f"CREATE SCHEMA IF NOT EXISTS {_quote_ident(s)};" for s in schemas]


def build_drop_schemas_sql(tables: dict[str, TableMeta]) -> list[str]:
    """DROP SCHEMA ... CASCADE — для чистого пересоздания."""
    schemas = sorted({tm.schema for tm in tables.values()})
    return [f"DROP SCHEMA IF EXISTS {_quote_ident(s)} CASCADE;" for s in schemas]
