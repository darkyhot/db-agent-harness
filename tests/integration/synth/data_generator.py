"""Генератор синтетических данных по метаданным.

Стратегия:
  1. Сортируем таблицы топологически: словари (grain=dictionary) первыми,
     затем таблицы без явных FK, затем зависимые.
  2. Для каждой колонки строим значение:
     - FK → sample из value_pool целевой колонки;
     - известное «связующее» имя (tb_id, gosb_id, old_gosb_id, epk_id, …)
       → sample из shared_pool, если уже есть; иначе генерим и добавляем в pool;
     - partition_key и колонки типа report_dt → равномерно по месяцам в окне дат;
     - иначе — дефолтный генератор по dType.
  3. Композитный PK обеспечивается через set-дедуп; при коллизии — перегенерация.

Прод-код не трогает.
"""
from __future__ import annotations

import datetime as _dt
import logging
import random
from typing import Any, Iterable

from tests.integration.synth.ddl_generator import ColumnMeta, TableMeta, load_metadata
from tests.integration.synth.type_mapping import DEFAULT_DATE_MIN, DEFAULT_DATE_MAX

logger = logging.getLogger(__name__)

# Имена колонок, значения которых одинаковы между таблицами (общий пул).
SHARED_KEY_COLUMNS = {
    "tb_id",
    "gosb_id",
    "old_gosb_id",
    "new_gosb_id",
    "epk_id",
    "saphr_id",
}

# Имена колонок, которые трактуем как даты «отчётного периода» и
# распределяем равномерно по месяцам в [2025-01-01 … 2026-05-26].
REPORT_DATE_COLUMNS = {"report_dt"}


def _topological_order(tables: dict[str, TableMeta]) -> list[TableMeta]:
    """Топологически отсортировать таблицы: словари → независимые → зависимые."""
    deps: dict[str, set[str]] = {full: set() for full in tables}
    for full, tm in tables.items():
        for c in tm.columns:
            if c.fk_target:
                ref = ".".join(c.fk_target.split(".")[:2])
                if ref in tables and ref != full:
                    deps[full].add(ref)

    def grain_rank(tm: TableMeta) -> int:
        if tm.grain == "dictionary":
            return 0
        if tm.grain == "organization":  # справочник организаций
            return 1
        return 2

    visited: set[str] = set()
    order: list[TableMeta] = []

    def visit(full: str) -> None:
        if full in visited:
            return
        visited.add(full)
        for dep in sorted(deps[full], key=lambda d: (grain_rank(tables[d]), d)):
            visit(dep)
        order.append(tables[full])

    for full in sorted(tables, key=lambda f: (grain_rank(tables[f]), f)):
        visit(full)
    return order


def _month_starts(low: _dt.date, high: _dt.date) -> list[_dt.date]:
    """Список первых чисел месяцев в окне [low, high]."""
    out: list[_dt.date] = []
    d = _dt.date(low.year, low.month, 1)
    while d <= high:
        out.append(d)
        # next month
        if d.month == 12:
            d = _dt.date(d.year + 1, 1, 1)
        else:
            d = _dt.date(d.year, d.month + 1, 1)
    return out


def _gen_partition_date(rng: random.Random, idx: int, total: int) -> _dt.date:
    """Распределить дату равномерно по месяцам в окне."""
    months = _month_starts(DEFAULT_DATE_MIN, DEFAULT_DATE_MAX)
    bucket = months[idx % len(months)]
    # внутри месяца — случайный день
    return bucket + _dt.timedelta(days=rng.randint(0, 27))


def _gen_value(
    col: ColumnMeta,
    rng: random.Random,
    *,
    shared_pool: dict[str, list[Any]],
    fk_pool: dict[tuple[str, str, str], list[Any]],
    row_index: int,
    rows_total: int,
) -> Any:
    """Сгенерировать значение для одной колонки одной строки."""
    # 1) Явный FK
    if col.fk_target:
        target = tuple(col.fk_target.split("."))  # (schema, table, column)
        pool = fk_pool.get(target)
        if pool:
            return rng.choice(pool)
        # FK таргет ещё не загружен — деградируем в shared/type-based.

    # 2) Партиционный/отчётный столбец-дата
    if col.partition_key or col.name.lower() in REPORT_DATE_COLUMNS:
        if col.sql_type.pg_type == "DATE":
            return _gen_partition_date(rng, row_index, rows_total)

    # 3) Общий пул по имени колонки (для неявных связей tb_id, gosb_id, …)
    short = col.name.lower()
    if short in SHARED_KEY_COLUMNS:
        pool = shared_pool.get(short)
        if pool:
            return rng.choice(pool)

    # 4) Дефолт по типу
    return col.sql_type.gen(rng)


def _seed_shared_keys(shared_pool: dict[str, list[Any]], rng: random.Random) -> None:
    """Засеять небольшой пул значений для общих ключей.

    Намеренно компактные диапазоны: реальный Сбер имеет ~15 ТБ и сотни ГОСБ;
    маленькие пулы дают высокую вероятность пересечений → JOIN-ы работают.
    """
    shared_pool.setdefault("tb_id", [rng.randint(1, 15) for _ in range(15)])
    shared_pool.setdefault(
        "old_gosb_id", list({rng.randint(100, 999) for _ in range(60)})
    )
    shared_pool.setdefault(
        "new_gosb_id", list({rng.randint(100, 999) for _ in range(60)})
    )
    shared_pool.setdefault("gosb_id", shared_pool["old_gosb_id"])
    shared_pool.setdefault(
        "epk_id", list({rng.randint(10_000_000, 99_999_999) for _ in range(300)})
    )
    shared_pool.setdefault(
        "saphr_id", list({rng.randint(1_000_000, 9_999_999) for _ in range(150)})
    )


def generate_rows(
    tm: TableMeta,
    rng: random.Random,
    *,
    n: int,
    shared_pool: dict[str, list[Any]],
    fk_pool: dict[tuple[str, str, str], list[Any]],
) -> list[dict[str, Any]]:
    """Сгенерировать n уникальных по PK строк для таблицы."""
    pk_cols = tm.pk_columns()
    pk_names = [c.name for c in pk_cols]
    seen_pk: set[tuple] = set()
    rows: list[dict[str, Any]] = []

    max_attempts = n * 20
    attempts = 0
    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        row: dict[str, Any] = {}
        for c in tm.columns:
            row[c.name] = _gen_value(
                c,
                rng,
                shared_pool=shared_pool,
                fk_pool=fk_pool,
                row_index=len(rows),
                rows_total=n,
            )

        if pk_names:
            key = tuple(row[k] for k in pk_names)
            if key in seen_pk:
                continue
            seen_pk.add(key)
        rows.append(row)

    if len(rows) < n:
        logger.warning(
            "Таблица %s: запрошено %d строк, сгенерировано %d (исчерпан PK-простор)",
            tm.full, n, len(rows),
        )
    return rows


def generate_all(
    rows_per_table: int = 200,
    *,
    seed: int = 42,
    data_dir=None,
) -> tuple[list[TableMeta], dict[str, list[dict[str, Any]]]]:
    """Сгенерировать данные для всех таблиц.

    Returns:
        (порядок_таблиц, {full_name: rows}).
    """
    tables = load_metadata(data_dir)
    order = _topological_order(tables)
    rng = random.Random(seed)

    shared_pool: dict[str, list[Any]] = {}
    fk_pool: dict[tuple[str, str, str], list[Any]] = {}
    _seed_shared_keys(shared_pool, rng)

    out: dict[str, list[dict[str, Any]]] = {}
    for tm in order:
        rows = generate_rows(
            tm, rng,
            n=rows_per_table,
            shared_pool=shared_pool,
            fk_pool=fk_pool,
        )
        out[tm.full] = rows
        # обновить пулы значениями этой таблицы
        for c in tm.columns:
            values = [r[c.name] for r in rows if r[c.name] is not None]
            if not values:
                continue
            # обновляем fk_pool по тройке (schema, table, column)
            fk_pool[(c.schema, c.table, c.name)] = values
            short = c.name.lower()
            if short in SHARED_KEY_COLUMNS:
                shared_pool.setdefault(short, []).extend(values[:50])
        logger.info("Сгенерировано %d строк для %s", len(rows), tm.full)

    return order, out
