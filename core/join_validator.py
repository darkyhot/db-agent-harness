"""SQL-запросы для детерминированной runtime-валидации безопасности JOIN.

Все запросы — SELECT-only (read-only), безопасны для production БД.

Предназначение:
  Перед выполнением финального SQL проверяем фактические данные:
  1. Уникальность JOIN-ключа в dim-таблице (не устаревшие CSV-метрики)
  2. Фанаут JOIN: умножает ли dim строки из fact?
  3. FK-покрытие: какой % строк fact имеет пару в dim?
  4. NULL в ключевой колонке (потеря строк при INNER JOIN)

Эти запросы «не положат» аналитическую БД: они агрегирующие, без сортировки
по большим полям, без подзапросов с декартовым произведением.
"""

from typing import Any


# ---------------------------------------------------------------------------
# Генераторы SQL-запросов
# ---------------------------------------------------------------------------

def build_uniqueness_check_sql(schema: str, table: str, column: str) -> str:
    """SQL: уникальность колонки в таблице.

    Результат: одна строка с {total_rows, unique_vals, dup_pct}.
    dup_pct = 0.0 → safe=True (JOIN не умножит строки).

    Пример вывода:
        total_rows=1000, unique_vals=1000, dup_pct=0.00  → уникален
        total_rows=1000, unique_vals=800,  dup_pct=20.00 → 20% дублей
    """
    full = f"{schema}.{table}"
    return (
        f"SELECT\n"
        f"    COUNT(*)                                             AS total_rows,\n"
        f"    COUNT(DISTINCT {column})                             AS unique_vals,\n"
        f"    ROUND(\n"
        f"        (COUNT(*) - COUNT(DISTINCT {column}))::numeric\n"
        f"        / NULLIF(COUNT(*), 0) * 100, 2\n"
        f"    )                                                    AS dup_pct\n"
        f"FROM {full}"
    )


def build_join_fanout_check_sql(
    fact_schema: str, fact_table: str, fact_col: str,
    dim_schema: str, dim_table: str, dim_col: str,
) -> str:
    """SQL: фанаут JOIN (умножение строк факт-таблицы после JOIN с dim).

    Ожидаемый результат для safe JOIN: fanout = 1.000.
    fanout > 1.001 → dim не уникален по ключу → нужен DISTINCT ON или подзапрос.

    Результат: одна строка с {fact_rows, joined_rows, fanout}.

    ВАЖНО: запрос НЕ материализует полный JOIN — только считает строки,
    что безопасно для MPP БД (Greenplum распараллеливает COUNT).
    """
    fact = f"{fact_schema}.{fact_table}"
    dim = f"{dim_schema}.{dim_table}"
    return (
        f"WITH fact_cnt AS (\n"
        f"    SELECT COUNT(*) AS cnt FROM {fact}\n"
        f"),\n"
        f"joined_cnt AS (\n"
        f"    SELECT COUNT(*) AS cnt\n"
        f"    FROM {fact} f\n"
        f"    JOIN {dim} d ON d.{dim_col} = f.{fact_col}\n"
        f")\n"
        f"SELECT\n"
        f"    fact_cnt.cnt                                         AS fact_rows,\n"
        f"    joined_cnt.cnt                                       AS joined_rows,\n"
        f"    ROUND(\n"
        f"        joined_cnt.cnt::numeric / NULLIF(fact_cnt.cnt, 0), 3\n"
        f"    )                                                    AS fanout\n"
        f"FROM fact_cnt, joined_cnt"
    )


def build_fk_coverage_check_sql(
    fact_schema: str, fact_table: str, fact_col: str,
    dim_schema: str, dim_table: str, dim_col: str,
) -> str:
    """SQL: FK-покрытие — какой % строк fact находит пару в dim.

    coverage_pct = 100 → все строки fact имеют пару (INNER JOIN не потеряет строки).
    coverage_pct < 100 → INNER JOIN отбросит (100 - coverage_pct)% строк fact.

    Результат: одна строка с {fact_rows, matched_rows, coverage_pct}.
    """
    fact = f"{fact_schema}.{fact_table}"
    dim = f"{dim_schema}.{dim_table}"
    return (
        f"SELECT\n"
        f"    COUNT(*)                                             AS fact_rows,\n"
        f"    COUNT(d.{dim_col})                                   AS matched_rows,\n"
        f"    ROUND(\n"
        f"        COUNT(d.{dim_col})::numeric\n"
        f"        / NULLIF(COUNT(*), 0) * 100, 2\n"
        f"    )                                                    AS coverage_pct\n"
        f"FROM {fact} f\n"
        f"LEFT JOIN {dim} d ON d.{dim_col} = f.{fact_col}"
    )


def build_null_check_sql(schema: str, table: str, column: str) -> str:
    """SQL: NULL-значения в колонке.

    null_pct > 0 → WARNING при использовании колонки в JOIN или GROUP BY.
    Нулевые значения → потеря строк при INNER JOIN, NULL-группа в GROUP BY.

    Результат: одна строка с {total_rows, not_null_rows, null_pct}.
    """
    full = f"{schema}.{table}"
    return (
        f"SELECT\n"
        f"    COUNT(*)                                             AS total_rows,\n"
        f"    COUNT({column})                                      AS not_null_rows,\n"
        f"    ROUND(\n"
        f"        (COUNT(*) - COUNT({column}))::numeric\n"
        f"        / NULLIF(COUNT(*), 0) * 100, 2\n"
        f"    )                                                    AS null_pct\n"
        f"FROM {full}"
    )


def build_group_by_cardinality_sql(
    schema: str, table: str, columns: list[str], limit: int = 5
) -> str:
    """SQL: мощность GROUP BY — сколько уникальных комбинаций по выбранным колонкам.

    Помогает оценить объём результирующей выборки ДО выполнения финального запроса.
    Если cardinality очень большая → возможно стоит добавить LIMIT или фильтр.

    Args:
        columns: список имён колонок для GROUP BY
        limit:   максимальное количество строк в оценке (SELECT LIMIT)
    """
    full = f"{schema}.{table}"
    cols_str = ", ".join(columns)
    return (
        f"SELECT COUNT(*) AS group_count\n"
        f"FROM (\n"
        f"    SELECT {cols_str}\n"
        f"    FROM {full}\n"
        f"    GROUP BY {cols_str}\n"
        f"    LIMIT {limit * 1000}\n"
        f") sub"
    )


# ---------------------------------------------------------------------------
# Интерпретация результатов
# ---------------------------------------------------------------------------

def interpret_uniqueness(row: dict[str, Any]) -> dict[str, Any]:
    """Интерпретировать результат build_uniqueness_check_sql.

    Args:
        row: dict-строка из cursor (total_rows, unique_vals, dup_pct)

    Returns:
        {is_unique, total_rows, unique_vals, dup_pct, warning}
    """
    total = int(row.get('total_rows') or 0)
    unique = int(row.get('unique_vals') or 0)
    dup_pct = float(row.get('dup_pct') or 0)
    is_unique = (total > 0 and total == unique)
    return {
        'is_unique': is_unique,
        'total_rows': total,
        'unique_vals': unique,
        'dup_pct': dup_pct,
        'warning': None if is_unique else (
            f"Колонка не уникальна: {total} строк, "
            f"{unique} уник. значений ({dup_pct:.1f}% дублей). "
            "JOIN по этому ключу умножит строки — нужен DISTINCT ON."
        ),
    }


def interpret_fanout(row: dict[str, Any]) -> dict[str, Any]:
    """Интерпретировать результат build_join_fanout_check_sql.

    Returns:
        {is_safe, fanout, fact_rows, joined_rows, warning}
    """
    fact_rows = int(row.get('fact_rows') or 0)
    joined_rows = int(row.get('joined_rows') or 0)
    fanout = float(row.get('fanout') or 1.0)
    is_safe = fanout <= 1.001  # допуск на float-погрешность

    warning = None
    if not is_safe:
        extra = joined_rows - fact_rows
        warning = (
            f"JOIN умножает строки: {fact_rows} → {joined_rows} "
            f"(фанаут {fanout:.3f}×, +{extra} лишних строк). "
            "Используй DISTINCT ON или агрегацию в подзапросе."
        )
    return {
        'is_safe': is_safe,
        'fanout': fanout,
        'fact_rows': fact_rows,
        'joined_rows': joined_rows,
        'warning': warning,
    }


def interpret_fk_coverage(row: dict[str, Any]) -> dict[str, Any]:
    """Интерпретировать результат build_fk_coverage_check_sql.

    Returns:
        {is_full, coverage_pct, fact_rows, matched_rows, warning}
    """
    fact_rows = int(row.get('fact_rows') or 0)
    matched = int(row.get('matched_rows') or 0)
    cov_pct = float(row.get('coverage_pct') or 0)
    is_full = cov_pct >= 99.9

    warning = None
    if not is_full:
        lost = fact_rows - matched
        warning = (
            f"FK-покрытие {cov_pct:.1f}%: INNER JOIN потеряет ~{lost} строк fact "
            f"({100 - cov_pct:.1f}% без пары в dim). "
            "Рассмотри LEFT JOIN если нужны все строки."
        )
    return {
        'is_full': is_full,
        'coverage_pct': cov_pct,
        'fact_rows': fact_rows,
        'matched_rows': matched,
        'warning': warning,
    }


def interpret_null_check(row: dict[str, Any]) -> dict[str, Any]:
    """Интерпретировать результат build_null_check_sql."""
    total = int(row.get('total_rows') or 0)
    not_null = int(row.get('not_null_rows') or 0)
    null_pct = float(row.get('null_pct') or 0)
    has_nulls = null_pct > 0

    warning = None
    if has_nulls:
        warning = (
            f"Колонка содержит NULL: {null_pct:.1f}% строк ({total - not_null} шт.). "
            "NULL-строки выпадут из INNER JOIN и создадут NULL-группу в GROUP BY."
        )
    return {
        'has_nulls': has_nulls,
        'null_pct': null_pct,
        'total_rows': total,
        'not_null_rows': not_null,
        'warning': warning,
    }


# ---------------------------------------------------------------------------
# Сводный план валидации для join_spec
# ---------------------------------------------------------------------------

def build_validation_plan(join_spec: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Сформировать список SQL-проверок для заданного join_spec.

    Возвращает список dict с ключами:
      check_type: "uniqueness" | "fanout" | "fk_coverage" | "null_check"
      sql:        готовый SQL для выполнения
      context:    описание проверки (для логирования)
      interpret:  имя функции интерпретации (строка)

    Caller выполняет каждый SQL и передаёт результат в соответствующую interpret_*().
    """
    plan: list[dict[str, Any]] = []

    for jk in join_spec:
        left = jk.get('left', '')
        right = jk.get('right', '')
        if not left or not right:
            continue

        left_parts = left.rsplit('.', 1)
        right_parts = right.rsplit('.', 1)
        if len(left_parts) != 2 or len(right_parts) != 2:
            continue

        left_tbl_full, left_col = left_parts
        right_tbl_full, right_col = right_parts

        ltp = left_tbl_full.split('.', 1)
        rtp = right_tbl_full.split('.', 1)
        if len(ltp) != 2 or len(rtp) != 2:
            continue

        l_schema, l_table = ltp
        r_schema, r_table = rtp

        # 1. Уникальность правой стороны (dim-ключ)
        plan.append({
            'check_type': 'uniqueness',
            'sql': build_uniqueness_check_sql(r_schema, r_table, right_col),
            'context': f'Уникальность {right_tbl_full}.{right_col}',
            'interpret': 'interpret_uniqueness',
            'meta': {'table': right_tbl_full, 'column': right_col},
        })

        # 2. NULL в ключах обеих сторон
        plan.append({
            'check_type': 'null_check',
            'sql': build_null_check_sql(l_schema, l_table, left_col),
            'context': f'NULL в {left_tbl_full}.{left_col}',
            'interpret': 'interpret_null_check',
            'meta': {'table': left_tbl_full, 'column': left_col},
        })
        plan.append({
            'check_type': 'null_check',
            'sql': build_null_check_sql(r_schema, r_table, right_col),
            'context': f'NULL в {right_tbl_full}.{right_col}',
            'interpret': 'interpret_null_check',
            'meta': {'table': right_tbl_full, 'column': right_col},
        })

        # 3. Фанаут и покрытие (только если join_spec помечен как unsafe)
        if not jk.get('safe', True):
            plan.append({
                'check_type': 'fanout',
                'sql': build_join_fanout_check_sql(
                    l_schema, l_table, left_col,
                    r_schema, r_table, right_col,
                ),
                'context': f'Фанаут JOIN {left_tbl_full} → {right_tbl_full}',
                'interpret': 'interpret_fanout',
                'meta': {
                    'fact_table': left_tbl_full, 'fact_col': left_col,
                    'dim_table': right_tbl_full, 'dim_col': right_col,
                },
            })
            plan.append({
                'check_type': 'fk_coverage',
                'sql': build_fk_coverage_check_sql(
                    l_schema, l_table, left_col,
                    r_schema, r_table, right_col,
                ),
                'context': f'FK-покрытие {left_tbl_full}.{left_col} → {right_tbl_full}',
                'interpret': 'interpret_fk_coverage',
                'meta': {
                    'fact_table': left_tbl_full, 'fact_col': left_col,
                    'dim_table': right_tbl_full, 'dim_col': right_col,
                },
            })

    return plan
