"""SQL Query Builder: конвертирует структурированную QuerySpec в SQL через SQLAlchemy Core.

Вместо того чтобы LLM писала сырой SQL, она теперь описывает запрос в виде
структурированного JSON (QuerySpec), а этот модуль компилирует его в корректный SQL.

Это решает проблему умножения строк при JOIN-ах: LLM явно указывает use_subquery=True,
когда правая таблица может иметь дубликаты по ключу соединения.

Формат QuerySpec (JSON):
{
  "ctes": [                        # опционально: WITH-блоки
    {"name": "dept_avg", "spec": { ...вложенная QuerySpec... }}
  ],
  "from": {                        # обязательно
    "schema": "hr",
    "table": "employees",
    "alias": "e"                   # опционально
  },
  "select": [                      # опционально (по умолчанию *)
    {"expr": "column", "table": "e", "column": "name", "alias": "emp_name"},
    {"expr": "column", "table": "e", "column": "salary"},
    {"expr": "func", "func": "COUNT", "args": ["*"], "alias": "cnt"},
    {"expr": "func", "func": "SUM", "args": ["e.amount"], "alias": "total"},
    {
      "expr": "window",
      "func": "ROW_NUMBER",
      "args": [],
      "partition_by": ["e.dept_id"],
      "order_by": [{"expr": "e.salary", "direction": "desc"}],
      "alias": "rn"
    }
  ],
  "joins": [                       # опционально
    {
      "type": "inner|left|right|full",
      "schema": "hr",
      "table": "departments",
      "alias": "d",
      "on": {"left": "e.dept_id", "right": "d.id"},
      "use_subquery": false        # true — обернуть в подзапрос (SELECT * FROM ...)
    }
  ],
  "where": [                       # опционально
    {"column": "e.salary", "op": ">|<|=|!=|>=|<=|IN|NOT IN|IS NULL|IS NOT NULL|LIKE|ILIKE", "value": 50000}
  ],
  "group_by": ["d.name", "e.dept_id"],   # опционально
  "having": [                            # опционально
    {"column": "cnt", "op": ">", "value": 5}
  ],
  "order_by": [                          # опционально
    {"expr": "e.salary", "direction": "desc|asc"}
  ],
  "limit": 100,                          # опционально
  "offset": 0                            # опционально
}
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SQLBuilderError(Exception):
    """Ошибка построения SQL из QuerySpec."""


class SQLQueryBuilder:
    """Конвертирует QuerySpec (dict) в SQL-строку.

    Использует строковую сборку с явным квотированием идентификаторов
    для поддержки PostgreSQL/Greenplum без зависимости от MetaData/reflect.
    """

    ALLOWED_OPS = {
        ">", "<", "=", "!=", "<>", ">=", "<=",
        "IN", "NOT IN", "IS NULL", "IS NOT NULL",
        "LIKE", "ILIKE", "NOT LIKE", "BETWEEN",
    }

    JOIN_TYPES = {"inner": "INNER JOIN", "left": "LEFT JOIN", "right": "RIGHT JOIN", "full": "FULL OUTER JOIN"}

    WINDOW_FUNCS = {"ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE", "LAG", "LEAD",
                    "FIRST_VALUE", "LAST_VALUE", "SUM", "AVG", "COUNT", "MIN", "MAX",
                    "PERCENT_RANK", "CUME_DIST"}

    AGG_FUNCS = {"COUNT", "SUM", "AVG", "MIN", "MAX", "STRING_AGG", "ARRAY_AGG",
                 "COUNT DISTINCT", "COALESCE", "NULLIF", "GREATEST", "LEAST"}

    def build(self, spec: dict[str, Any]) -> str:
        """Построить SQL-строку из QuerySpec.

        Args:
            spec: Словарь с описанием запроса (QuerySpec).

        Returns:
            Готовая SQL-строка.

        Raises:
            SQLBuilderError: Если спецификация некорректна.
        """
        if not isinstance(spec, dict):
            raise SQLBuilderError("QuerySpec должен быть словарём (dict)")

        parts: list[str] = []

        # WITH (CTEs)
        ctes_sql = self._build_ctes(spec.get("ctes", []))
        if ctes_sql:
            parts.append(ctes_sql)

        # SELECT
        select_sql = self._build_select(spec.get("select"))
        parts.append(select_sql)

        # FROM
        from_spec = spec.get("from")
        if not from_spec:
            raise SQLBuilderError("QuerySpec обязан содержать поле 'from'")
        from_sql = self._build_from(from_spec)
        parts.append(from_sql)

        # JOIN
        for join in spec.get("joins", []):
            parts.append(self._build_join(join))

        # WHERE
        where_list = spec.get("where", [])
        if where_list:
            parts.append(self._build_where(where_list))

        # GROUP BY
        group_by = spec.get("group_by", [])
        if group_by:
            cols = ", ".join(self._col_ref(c) for c in group_by)
            parts.append(f"GROUP BY {cols}")

        # HAVING
        having_list = spec.get("having", [])
        if having_list:
            parts.append(self._build_having(having_list))

        # ORDER BY
        order_by = spec.get("order_by", [])
        if order_by:
            parts.append(self._build_order_by(order_by))

        # LIMIT / OFFSET
        if spec.get("limit") is not None:
            parts.append(f"LIMIT {int(spec['limit'])}")
        if spec.get("offset") is not None:
            parts.append(f"OFFSET {int(spec['offset'])}")

        sql = "\n".join(parts)
        logger.debug("SQLQueryBuilder построил запрос:\n%s", sql)
        return sql

    # ─── CTEs ────────────────────────────────────────────────────────────────

    def _build_ctes(self, ctes: list[dict]) -> str:
        if not ctes:
            return ""
        cte_parts = []
        for cte in ctes:
            name = self._quote_id(cte["name"])
            inner_sql = self.build(cte["spec"])
            cte_parts.append(f"{name} AS (\n{inner_sql}\n)")
        return "WITH " + ",\n".join(cte_parts)

    # ─── SELECT ───────────────────────────────────────────────────────────────

    def _build_select(self, select: list[dict] | None) -> str:
        if not select:
            return "SELECT *"
        exprs = [self._build_select_expr(e) for e in select]
        return "SELECT " + ",\n       ".join(exprs)

    def _build_select_expr(self, expr: dict) -> str:
        kind = expr.get("expr", "column")
        alias = expr.get("alias")

        if kind == "column":
            result = self._col_ref_from_parts(expr.get("table"), expr["column"])
        elif kind == "func":
            result = self._build_func_expr(expr)
        elif kind == "window":
            result = self._build_window_expr(expr)
        elif kind == "literal":
            result = self._literal(expr["value"])
        elif kind == "raw":
            # Осторожно: только для экранированных значений
            result = expr["sql"]
        else:
            raise SQLBuilderError(f"Неизвестный тип выражения SELECT: {kind!r}")

        if alias:
            result += f" AS {self._quote_id(alias)}"
        return result

    def _build_func_expr(self, expr: dict) -> str:
        func = expr.get("func", "").upper()
        if func not in self.AGG_FUNCS and not func.isidentifier():
            raise SQLBuilderError(f"Недопустимое имя функции: {func!r}")
        args = expr.get("args", [])
        args_sql = ", ".join(self._col_ref(a) if a != "*" else "*" for a in args)
        if func == "COUNT DISTINCT" and args:
            return f"COUNT(DISTINCT {args_sql})"
        return f"{func}({args_sql})"

    def _build_window_expr(self, expr: dict) -> str:
        func = expr.get("func", "").upper()
        if func not in self.WINDOW_FUNCS:
            raise SQLBuilderError(f"Недопустимая оконная функция: {func!r}")

        args = expr.get("args", [])
        args_sql = ", ".join(self._col_ref(a) if a != "*" else "*" for a in args)
        func_call = f"{func}({args_sql})"

        window_parts = []
        partition_by = expr.get("partition_by", [])
        if partition_by:
            cols = ", ".join(self._col_ref(c) for c in partition_by)
            window_parts.append(f"PARTITION BY {cols}")

        order_by = expr.get("order_by", [])
        if order_by:
            ob_parts = []
            for ob in order_by:
                col = self._col_ref(ob.get("expr") or ob.get("column", ""))
                direction = ob.get("direction", "asc").upper()
                if direction not in ("ASC", "DESC"):
                    direction = "ASC"
                ob_parts.append(f"{col} {direction}")
            window_parts.append("ORDER BY " + ", ".join(ob_parts))

        frame = expr.get("frame")
        if frame:
            window_parts.append(frame)

        return f"{func_call} OVER ({' '.join(window_parts)})"

    # ─── FROM ────────────────────────────────────────────────────────────────

    def _build_from(self, from_spec: dict) -> str:
        table_sql = self._table_ref(from_spec["schema"], from_spec["table"])
        alias = from_spec.get("alias")
        if alias:
            table_sql += f" AS {self._quote_id(alias)}"
        return f"FROM {table_sql}"

    # ─── JOIN ────────────────────────────────────────────────────────────────

    def _build_join(self, join: dict) -> str:
        join_type = self.JOIN_TYPES.get(join.get("type", "inner").lower())
        if not join_type:
            raise SQLBuilderError(f"Неизвестный тип JOIN: {join.get('type')!r}")

        schema = join["schema"]
        table = join["table"]
        alias = join.get("alias")
        use_subquery = join.get("use_subquery", False)

        if use_subquery:
            # Оборачиваем таблицу в подзапрос, чтобы избежать умножения строк
            sub_alias = alias or f"sub_{table}"
            table_sql = f"(SELECT * FROM {self._table_ref(schema, table)}) AS {self._quote_id(sub_alias)}"
            effective_alias = sub_alias
        else:
            table_sql = self._table_ref(schema, table)
            if alias:
                table_sql += f" AS {self._quote_id(alias)}"
            effective_alias = alias or table

        # ON условие
        on = join.get("on")
        if on:
            left = self._col_ref(on["left"])
            right = self._col_ref(on["right"])
            on_sql = f"ON {left} = {right}"
        elif join.get("on_raw"):
            on_sql = f"ON {join['on_raw']}"
        else:
            raise SQLBuilderError("JOIN требует поле 'on' или 'on_raw'")

        return f"{join_type} {table_sql} {on_sql}"

    # ─── WHERE ────────────────────────────────────────────────────────────────

    def _build_where(self, conditions: list[dict]) -> str:
        parts = [self._build_condition(c) for c in conditions]
        return "WHERE " + "\n  AND ".join(parts)

    def _build_condition(self, cond: dict) -> str:
        col = self._col_ref(cond["column"])
        op = cond.get("op", "=").upper()

        if op not in self.ALLOWED_OPS:
            raise SQLBuilderError(f"Недопустимый оператор: {op!r}")

        if op in ("IS NULL", "IS NOT NULL"):
            return f"{col} {op}"

        value = cond.get("value")
        if op in ("IN", "NOT IN"):
            if not isinstance(value, (list, tuple)):
                raise SQLBuilderError(f"IN/NOT IN требует список значений, получено: {type(value)}")
            vals = ", ".join(self._literal(v) for v in value)
            return f"{col} {op} ({vals})"

        if op == "BETWEEN":
            vals = value if isinstance(value, (list, tuple)) and len(value) == 2 else None
            if not vals:
                raise SQLBuilderError("BETWEEN требует список из 2 значений")
            return f"{col} BETWEEN {self._literal(vals[0])} AND {self._literal(vals[1])}"

        return f"{col} {op} {self._literal(value)}"

    # ─── HAVING ───────────────────────────────────────────────────────────────

    def _build_having(self, conditions: list[dict]) -> str:
        parts = []
        for cond in conditions:
            # В HAVING может быть агрегатная функция или алиас
            col_raw = cond.get("column") or cond.get("expr", "")
            # Если это агрегатная функция — оставляем как есть (raw)
            col = col_raw if any(f in col_raw.upper() for f in ["COUNT", "SUM", "AVG", "MIN", "MAX"]) else self._col_ref(col_raw)
            op = cond.get("op", "=").upper()
            if op not in self.ALLOWED_OPS:
                raise SQLBuilderError(f"Недопустимый оператор в HAVING: {op!r}")
            value = cond.get("value")
            parts.append(f"{col} {op} {self._literal(value)}")
        return "HAVING " + "\n   AND ".join(parts)

    # ─── ORDER BY ─────────────────────────────────────────────────────────────

    def _build_order_by(self, order_by: list[dict]) -> str:
        parts = []
        for ob in order_by:
            col = self._col_ref(ob.get("expr") or ob.get("column", ""))
            direction = ob.get("direction", "asc").upper()
            if direction not in ("ASC", "DESC"):
                direction = "ASC"
            nulls = ob.get("nulls", "").upper()
            nulls_sql = f" NULLS {nulls}" if nulls in ("FIRST", "LAST") else ""
            parts.append(f"{col} {direction}{nulls_sql}")
        return "ORDER BY " + ", ".join(parts)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _quote_id(self, name: str) -> str:
        """Обернуть идентификатор в двойные кавычки (PostgreSQL стиль)."""
        # Если уже в кавычках — вернуть как есть
        if name.startswith('"') and name.endswith('"'):
            return name
        return f'"{name}"'

    def _table_ref(self, schema: str, table: str) -> str:
        """Собрать ссылку на таблицу: "schema"."table"."""
        return f"{self._quote_id(schema)}.{self._quote_id(table)}"

    def _col_ref(self, col: str) -> str:
        """Преобразовать ссылку на колонку в формат "alias"."col" или "col".

        Поддерживает:
          - "alias.column"  → "alias"."column"
          - "column"        → "column"
          - "*"             → *
          - агрегатные выражения (COUNT(*), SUM(e.amount)) — оставляем как есть
        """
        if not col:
            return ""
        col = col.strip()

        # Оставляем агрегатные функции и специальные выражения как есть
        upper = col.upper()
        for func in list(self.AGG_FUNCS) + list(self.WINDOW_FUNCS):
            if upper.startswith(func + "(") or upper.startswith(func + " "):
                return col

        if col == "*":
            return "*"

        # Уже квотировано
        if '"' in col:
            return col

        if "." in col:
            parts = col.split(".", 1)
            return f"{self._quote_id(parts[0])}.{self._quote_id(parts[1])}"
        return self._quote_id(col)

    def _col_ref_from_parts(self, table_alias: str | None, column: str) -> str:
        """Собрать ссылку из отдельных таблицы и колонки."""
        if table_alias:
            return f"{self._quote_id(table_alias)}.{self._quote_id(column)}"
        return self._quote_id(column)

    def _literal(self, value: Any) -> str:
        """Безопасно превратить Python-значение в SQL-литерал."""
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            # Экранируем одинарные кавычки
            escaped = value.replace("'", "''")
            return f"'{escaped}'"
        # Для остальных типов (например datetime) — строковое представление
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"
