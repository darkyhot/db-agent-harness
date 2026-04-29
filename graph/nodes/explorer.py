"""Узлы разведки таблиц и выбора колонок.

Содержит ExplorerNodes — миксин для GraphNodes с методами:
- table_explorer: разведка структуры и семплов выбранных таблиц
- column_selector: LLM-выбор релевантных колонок для SQL-запроса
"""

import json
import logging
import re
import time
from typing import Any

from core.column_binding import bind_columns
from core.join_analysis import detect_table_type, format_join_analysis
from core.join_selection import normalize_join_spec
from core.query_ir import QuerySpec
from graph.state import AgentState

logger = logging.getLogger(__name__)


def _apply_explicit_join_override(
    join_spec: list[dict[str, Any]],
    selected_columns: dict[str, Any],
    intent: dict[str, Any],
    user_hints: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Применить explicit_join из интента / user_hints как override join_spec.

    Когда пользователь явно указал ключ JOIN ("по инн", "по дате") — он имеет
    абсолютный приоритет. Ищем колонку по col_hint семантически в selected_columns,
    подставляем в join_spec вместо auto-detected пары.

    Источники (оба опциональны, обрабатываются в порядке приоритета):
    - user_hints.join_fields — свежий результат LLM/regex-экстрактора, primary.
    - intent.explicit_join — результат intent_classifier, secondary.

    Конфликт разрешается так: ключи из user_hints применяются первыми;
    ключи из intent.explicit_join — только для col_hint-ов, которые не
    были покрыты user_hints.

    Returns:
        Обновлённый join_spec (исходный если оба источника пусты / не нашли совпадений).
    """
    explicit_joins: list[dict[str, Any]] = []
    covered_hints: set[str] = set()

    for field in (user_hints or {}).get("join_fields") or []:
        col = str(field).lower().strip()
        if col and col not in covered_hints:
            explicit_joins.append({"column_hint": col, "table_hint": ""})
            covered_hints.add(col)

    for ej in intent.get("explicit_join") or []:
        if not isinstance(ej, dict):
            continue
        col = str(ej.get("column_hint") or "").lower().strip()
        if not col or col in covered_hints:
            continue
        explicit_joins.append(ej)
        covered_hints.add(col)

    if not explicit_joins:
        return join_spec

    result_spec = list(join_spec)

    for ej in explicit_joins:
        if not isinstance(ej, dict):
            continue
        col_hint = str(ej.get("column_hint") or "").lower().strip()
        tbl_hint = str(ej.get("table_hint") or "").lower().strip()
        if not col_hint:
            continue

        # Ищем колонку по col_hint во всех таблицах selected_columns
        _candidates: list[tuple[float, str, str]] = []  # (score, tbl, col)
        for tbl, roles in selected_columns.items():
            tbl_lower = tbl.lower()
            all_cols = (
                roles.get("select", [])
                + roles.get("filter", [])
                + roles.get("group_by", [])
                + roles.get("aggregate", [])
            )
            for c in dict.fromkeys(all_cols):  # уникальные, порядок сохранён
                c_lower = c.lower()
                if col_hint in c_lower or c_lower in col_hint:
                    score = len(col_hint) / max(len(c_lower), 1) * 100
                    if tbl_hint and (tbl_hint in tbl_lower or tbl_lower.endswith(tbl_hint)):
                        score += 50
                    _candidates.append((score, tbl, c))

        if len(_candidates) < 2:
            if _candidates:
                logger.warning(
                    "_apply_explicit_join_override: col_hint=%r найден только в одной таблице — "
                    "недостаточно для JOIN-пары",
                    col_hint,
                )
            continue

        _candidates.sort(reverse=True)
        # Берём лучшего с tbl_hint (правая сторона JOIN) и лучшего без (левая)
        right_tbl, right_col = _candidates[0][1], _candidates[0][2]
        left_tbl, left_col = None, None
        for _, tbl, col in _candidates[1:]:
            if tbl != right_tbl:
                left_tbl, left_col = tbl, col
                break

        if left_tbl is None:
            logger.warning(
                "_apply_explicit_join_override: col_hint=%r только в одной таблице %s — пропускаем",
                col_hint, right_tbl,
            )
            continue

        override = {
            "left": f"{left_tbl}.{left_col}",
            "right": f"{right_tbl}.{right_col}",
            "safe": False,
            "strategy": "explicit_user",
            "risk": "user_specified_key",
        }

        # Удаляем авто-detected пары между теми же таблицами
        _left_tbl_prefix = ".".join(left_tbl.split(".")[:2])
        _right_tbl_prefix = ".".join(right_tbl.split(".")[:2])
        result_spec = [
            j for j in result_spec
            if not (
                ".".join(j.get("left", "").split(".")[:2]) == _left_tbl_prefix
                and ".".join(j.get("right", "").split(".")[:2]) == _right_tbl_prefix
            )
        ]
        result_spec.insert(0, override)
        logger.info(
            "_apply_explicit_join_override: override join %s.%s = %s.%s (col_hint=%r)",
            left_tbl, left_col, right_tbl, right_col, col_hint,
        )

    return result_spec


class ExplorerNodes:
    """Миксин с узлами table_explorer и column_selector для GraphNodes."""

    # --------------------------------------------------------------------------
    # table_explorer
    # --------------------------------------------------------------------------

    def table_explorer(self, state: AgentState) -> dict[str, Any]:
        """Узел автоматической разведки таблиц: подгружает структуру и семплы.

        Извлекает таблицы из state["selected_tables"] (или fallback из плана),
        загружает описание колонок из CSV-справочника и семпл 10 строк из БД
        для каждой таблицы. Сохраняет результаты в структурированные поля.

        Args:
            state: Текущее состояние графа.

        Returns:
            Обновления состояния с обогащённым контекстом таблиц.
        """
        # 1. Получаем таблицы из state["selected_tables"]
        selected = state.get("selected_tables", [])
        found_tables: set[tuple[str, str]] = set()
        explicit_source_guard = bool(
            state.get("allowed_tables")
            or (state.get("user_hints") or {}).get("must_keep_tables")
        )
        excluded_tables = {
            str(item).strip().lower()
            for item in (state.get("excluded_tables") or (state.get("user_hints") or {}).get("excluded_tables") or [])
            if str(item).strip()
        }

        if selected:
            for item in selected:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    full_name = f"{item[0]}.{item[1]}".lower()
                    if full_name not in excluded_tables:
                        found_tables.add((item[0], item[1]))
        elif state.get("allowed_tables") or (state.get("user_hints") or {}).get("must_keep_tables"):
            explicit_sources = list(state.get("allowed_tables") or [])
            explicit_sources.extend(
                f"{item[0]}.{item[1]}"
                for item in (state.get("user_hints") or {}).get("must_keep_tables", [])
                if isinstance(item, (list, tuple)) and len(item) == 2
            )
            for source in explicit_sources:
                parts = str(source or "").strip().split(".", 1)
                if len(parts) != 2:
                    continue
                full_name = f"{parts[0]}.{parts[1]}".lower()
                if full_name not in excluded_tables:
                    found_tables.add((parts[0], parts[1]))

        # 2. Fallback: извлечение schema.table из плана и user_input
        if not found_tables and explicit_source_guard:
            must_keep_full = [
                f"{item[0]}.{item[1]}"
                for item in (state.get("user_hints") or {}).get("must_keep_tables", [])
                if isinstance(item, (list, tuple)) and len(item) == 2
            ]
            allowed_full = list(state.get("allowed_tables") or [])
            explicit_full = list(dict.fromkeys(must_keep_full + allowed_full))
            filtered_by_excluded = [
                t for t in explicit_full if t.lower() in excluded_tables
            ]
            err_kind = (
                "conflict_with_excluded" if filtered_by_excluded else "no_eligible_source"
            )
            logger.warning(
                "TableExplorer: explicit source guard active (kind=%s, must_keep=%s, "
                "allowed=%s, excluded=%s, filtered=%s, selected=%s)",
                err_kind,
                must_keep_full,
                allowed_full,
                sorted(excluded_tables),
                filtered_by_excluded,
                selected,
            )
            empty_ctx = (
                "=== РАЗВЕДКА ТАБЛИЦ ===\n\n"
                "Явные источники заданы, но после фильтрации доступных таблиц не осталось."
            )
            return {
                "tables_context": empty_ctx,
                "table_structures": {},
                "table_samples": {},
                "table_types": {},
                "join_analysis_data": {},
                "explorer_error": {
                    "kind": err_kind,
                    "must_keep_filtered": filtered_by_excluded,
                    "explicit_sources": explicit_full,
                    "excluded_tables": sorted(excluded_tables),
                },
            }

        if not found_tables:
            plan_text = "\n".join(state.get("plan", []))
            user_input = state.get("user_input", "")
            scan_text = f"{plan_text}\n{user_input}"

            df = self.schema.tables_df
            if df.empty:
                logger.info("TableExplorer: каталог таблиц пуст, пропускаем")
                return {
                    "tables_context": "",
                    "table_structures": {},
                    "table_samples": {},
                    "table_types": {},
                    "join_analysis_data": {},
                    "explorer_error": {},
                }

            pattern = re.compile(
                r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b'
            )
            for m in pattern.finditer(scan_text):
                schema_name, table_name = m.group(1).lower(), m.group(2).lower()
                mask = (
                    df["schema_name"].str.lower() == schema_name
                ) & (
                    df["table_name"].str.lower() == table_name
                )
                if not df[mask].empty:
                    row = df[mask].iloc[0]
                    full_name = f"{row['schema_name']}.{row['table_name']}".lower()
                    if full_name not in excluded_tables:
                        found_tables.add((row["schema_name"], row["table_name"]))

            # Fallback — поиск таблиц по ключевым словам
            if not found_tables:
                keywords = set()
                for word in re.findall(r'[a-zA-Zа-яА-ЯёЁ_]{3,}', scan_text.lower()):
                    if word not in self._STOP_WORDS:
                        keywords.add(word)
                if keywords:
                    search_q = " ".join(list(keywords)[:5])
                    try:
                        search_df = self.schema.search_by_description(search_q)
                        if not search_df.empty:
                            for _, row in search_df.drop_duplicates(
                                subset=["schema_name", "table_name"],
                            ).head(3).iterrows():
                                s = row.get("schema_name", "")
                                t = row.get("table_name", "")
                                if s and t and f"{s}.{t}".lower() not in excluded_tables:
                                    found_tables.add((s, t))
                            if found_tables:
                                logger.info(
                                    "TableExplorer fallback: найдено %d таблиц через "
                                    "search_by_description: %s",
                                    len(found_tables),
                                    ", ".join(f"{s}.{t}" for s, t in found_tables),
                                )
                    except Exception as e:
                        logger.warning("TableExplorer fallback search failed: %s", e)

        if not found_tables:
            logger.info("TableExplorer: таблицы не найдены в плане, пропускаем")
            empty_ctx = (
                "=== РАЗВЕДКА ТАБЛИЦ ===\n\n"
                "Таблицы не были определены на этапе планирования.\n"
                "Используй get_table_columns или search_tables для получения "
                "структуры нужных таблиц перед написанием SQL."
            )
            return {
                "tables_context": empty_ctx,
                "table_structures": {},
                "table_samples": {},
                "table_types": {},
                "join_analysis_data": {},
                "explorer_error": {},
            }

        logger.info(
            "TableExplorer: найдено %d таблиц для разведки: %s",
            len(found_tables),
            ", ".join(f"{s}.{t}" for s, t in found_tables),
        )

        # --- Сбор данных по каждой таблице ---
        table_structures: dict[str, str] = {}
        table_samples: dict[str, str] = {}
        table_types: dict[str, str] = {}
        sections: list[str] = []

        for schema_name, table_name in sorted(found_tables):
            full_name = f"{schema_name}.{table_name}"

            # 1. Описание колонок из CSV-справочника
            table_info = self.schema.get_table_info(schema_name, table_name)
            table_structures[full_name] = table_info

            # 2. Семпл 10 строк из БД (с TTL-кэшем)
            cache_key = (schema_name, table_name)
            cached = self._sample_cache.get(cache_key)
            if cached and (time.monotonic() - cached[0]) < self.SAMPLE_CACHE_TTL:
                sample_text = cached[1]
                logger.debug(
                    "TableExplorer: семпл %s.%s из кэша", schema_name, table_name
                )
            else:
                try:
                    sample_df = self.db.get_sample(schema_name, table_name, 10)
                    if sample_df.empty:
                        sample_text = "(таблица пуста)"
                    else:
                        sample_text = sample_df.to_markdown(index=False)
                    self._sample_cache[cache_key] = (time.monotonic(), sample_text)
                except Exception as e:
                    logger.warning(
                        "TableExplorer: ошибка семпла %s.%s: %s",
                        schema_name, table_name, e,
                    )
                    sample_text = f"(ошибка загрузки семпла: {e})"

            table_samples[full_name] = sample_text

            # 3. Тип таблицы
            cols_df = self.schema.get_table_columns(schema_name, table_name)
            table_types[full_name] = detect_table_type(table_name, cols_df)

            # Секция для обратной совместимости (tables_context)
            sections.append(
                f"### {full_name}\n\n"
                f"**Структура (из справочника):**\n{table_info}\n\n"
                f"**Образец данных (10 строк):**\n{sample_text}"
            )

        # --- JOIN safety analysis ---
        join_analysis_data: dict[str, Any] = {}
        join_analysis_parts: list[str] = []
        table_list = sorted(found_tables)

        # Кэш DataFrames колонок и PK-count
        cols_cache: dict[tuple[str, str], Any] = {}
        pk_count_cache: dict[tuple[str, str], int] = {}

        def _get_cols(s: str, t: str) -> Any:
            key = (s, t)
            if key not in cols_cache:
                cols_cache[key] = self.schema.get_table_columns(s, t)
            return cols_cache[key]

        def _get_pk_count(s: str, t: str) -> int:
            key = (s, t)
            if key not in pk_count_cache:
                cols = _get_cols(s, t)
                if not cols.empty and "is_primary_key" in cols.columns:
                    pk_count_cache[key] = int(
                        cols["is_primary_key"].astype(bool).sum()
                    )
                else:
                    pk_count_cache[key] = 1
            return pk_count_cache[key]

        if len(table_list) >= 2:
            for i, (s1, t1) in enumerate(table_list):
                cols1 = _get_cols(s1, t1)
                for s2, t2 in table_list[i + 1:]:
                    cols2 = _get_cols(s2, t2)
                    if cols1.empty or cols2.empty:
                        continue

                    block = format_join_analysis(
                        s1, t1, cols1, s2, t2, cols2,
                        _get_pk_count(s1, t1), _get_pk_count(s2, t2),
                    )
                    pair_key = f"{s1}.{t1}|{s2}.{t2}"
                    if block:
                        join_analysis_parts.append(block)
                        join_analysis_data[pair_key] = {
                            "text": block,
                            "table1": f"{s1}.{t1}",
                            "table2": f"{s2}.{t2}",
                            "table1_type": table_types.get(f"{s1}.{t1}", "unknown"),
                            "table2_type": table_types.get(f"{s2}.{t2}", "unknown"),
                        }

        # --- Обратная совместимость: монолитная строка tables_context ---
        join_analysis_text = ""
        if join_analysis_parts:
            join_analysis_text = (
                "\n\n=== JOIN-АНАЛИЗ (ранжированные кандидаты) ===\n"
                + "\n".join(join_analysis_parts)
            )

        tables_context = (
            "=== РАЗВЕДКА ТАБЛИЦ (автоматически подгруженные данные) ===\n\n"
            "Изучи структуру и образцы данных ПЕРЕД написанием SQL.\n"
            "Обрати внимание на:\n"
            "- Гранулярность: что является одной строкой? Есть ли дубликаты по "
            "ключевым полям?\n"
            "- NULL'ы и пустые значения в колонках\n"
            "- Формат дат, числовых значений, кодов\n"
            "- Какие колонки можно использовать для фильтрации и группировки\n"
            "- Связь колонок с запросом пользователя: какие колонки соответствуют "
            "терминам из вопроса?\n\n"
            + "\n\n".join(sections)
            + join_analysis_text
        )

        self.memory.add_message(
            "tool",
            f"[table_explorer] Подгружена структура и семплы для: "
            f"{', '.join(f'{s}.{t}' for s, t in sorted(found_tables))}",
        )

        return {
            "tables_context": tables_context,
            "table_structures": table_structures,
            "table_samples": table_samples,
            "table_types": table_types,
            "join_analysis_data": join_analysis_data,
            "explorer_error": {},
            "messages": state["messages"] + [
                {
                    "role": "assistant",
                    "content": (
                        f"Подгружена структура и семплы таблиц: "
                        f"{', '.join(f'{s}.{t}' for s, t in sorted(found_tables))}"
                    ),
                }
            ],
        }

    # --------------------------------------------------------------------------
    # column_selector
    # --------------------------------------------------------------------------

    def column_selector(self, state: AgentState) -> dict[str, Any]:
        """LLM-узел выбора релевантных колонок из структур таблиц.

        На основе запроса пользователя, интента и структуры таблиц выбирает
        нужные колонки, определяет их роли и ключи для JOIN.

        Args:
            state: Текущее состояние графа.

        Returns:
            Обновления состояния с выбранными колонками и JOIN-спецификацией.
        """
        user_input = state.get("user_input", "")
        intent = state.get("intent", {})
        table_structures = state.get("table_structures", {})
        table_samples = state.get("table_samples", {})
        join_analysis_data = state.get("join_analysis_data", {})
        excluded_tables = {
            str(item).strip().lower()
            for item in (state.get("excluded_tables") or (state.get("user_hints") or {}).get("excluded_tables") or [])
            if str(item).strip()
        }
        if excluded_tables:
            table_structures = {k: v for k, v in table_structures.items() if k.lower() not in excluded_tables}
            table_samples = {k: v for k, v in table_samples.items() if k.lower() not in excluded_tables}
            join_analysis_data = {
                k: v for k, v in (join_analysis_data or {}).items()
                if not any(part.strip().lower() in excluded_tables for part in str(k).split("|"))
            }

        logger.info(
            "ColumnSelector: выбор колонок для %d таблиц", len(table_structures)
        )

        if not table_structures:
            logger.warning("ColumnSelector: нет структур таблиц, пропускаем")
            return {
                "selected_columns": {},
                "join_spec": [],
                "messages": state["messages"],
                "graph_iterations": state.get("graph_iterations", 0) + 1,
            }

        raw_query_spec = state.get("query_spec") or {}
        if not raw_query_spec and intent:
            raw_query_spec = {
                "task": "answer_data",
                "strategy": "aggregate",
                "entities": [],
                "metrics": [
                    {
                        "operation": str(intent.get("aggregation_hint") or "count").lower(),
                        "target": (list(intent.get("entities") or []) or [None])[0],
                        "distinct_policy": "auto",
                        "confidence": 0.5,
                    }
                ],
                "dimensions": [],
                "filters": [],
                "source_constraints": [],
                "join_constraints": [],
                "clarification_needed": False,
                "confidence": 0.5,
            }
        spec, spec_errors = QuerySpec.from_dict(raw_query_spec)
        bound_result = None if spec is None else bind_columns(
            query_spec=spec,
            table_structures=table_structures,
            table_types=state.get("table_types", {}) or {},
            schema_loader=self.schema,
            llm_invoker=self,
        )
        if bound_result and bound_result.get("selected_columns"):
            logger.info(
                "ColumnSelector: QuerySpec binding confidence=%.2f (%s)",
                float(bound_result.get("confidence", 0.0) or 0.0),
                bound_result.get("reason", ""),
            )
            self.memory.add_message(
                "assistant",
                f"[column_binding] таблиц: {len(bound_result['selected_columns'])}, "
                f"join-ключей: {len(bound_result.get('join_spec') or [])}",
            )
            return {
                "selected_columns": bound_result["selected_columns"],
                "join_spec": bound_result.get("join_spec") or [],
                "column_selector_hint": "",
                "messages": state["messages"] + [
                    {
                        "role": "assistant",
                        "content": (
                            "[binding] Колонки связаны по QuerySpec: "
                            f"{', '.join(bound_result['selected_columns'].keys())}"
                        ),
                    }
                ],
                "graph_iterations": state.get("graph_iterations", 0) + 1,
            }
        if spec_errors:
            logger.warning("ColumnSelector: QuerySpec invalid for binding: %s", "; ".join(spec_errors))
        logger.info("ColumnSelector: запускаем LLM column selector")

        # --- Системный промпт ---
        system_prompt = (
            "Ты — селектор колонок для SQL-запроса в Greenplum (PostgreSQL-совместимая "
            "MPP СУБД).\n\n"
            "Задача: на основе запроса пользователя и структуры таблиц выбери "
            "ТОЛЬКО нужные колонки и определи их роли.\n\n"
            "Верни ТОЛЬКО JSON:\n"
            "{\n"
            '  "columns": {\n'
            '    "<schema.table>": {\n'
            '      "select": ["<колонка для SELECT>", ...],\n'
            '      "filter": ["<колонка для WHERE/HAVING>", ...],\n'
            '      "aggregate": ["<колонка для агрегации>", ...],\n'
            '      "group_by": ["<колонка для GROUP BY>", ...]\n'
            "    },\n"
            "    ...\n"
            "  },\n"
            '  "join_keys": [\n'
            '    {"left": "<schema.table.column>", "right": "<schema.table.column>", '
            '"safe": <true|false>, "strategy": "<direct|through_dim|subquery>"},\n'
            "    ...\n"
            "  ],\n"
            '  "null_warnings": [\n'
            '    "<schema.table.column — описание проблемы с NULL>",\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Правила:\n"
            "- Используй ТОЛЬКО колонки, которые РЕАЛЬНО существуют в структуре таблиц\n"
            "- Для фильтров по датам указывай ожидаемый формат "
            "(например: 'YYYY-MM-DD', timestamp)\n"
            "- Если колонка используется в нескольких ролях — укажи её в каждой\n"
            "- Предупреждай о колонках с возможными NULL через null_warnings\n"
            "- join_keys заполняй только если таблиц больше одной\n"
            "- safe=true если JOIN по первичному/уникальному ключу без риска дубликатов\n"
            "- КРИТИЧНО: если пользователь просит 'название', 'наименование' или 'имя' "
            "какой-либо сущности — ОБЯЗАТЕЛЬНО включи колонку с суффиксом "
            "_name/_short_name/_full_name из справочной (dim) таблицы в selected_columns. "
            "Числовой идентификатор (_id) из фактовой таблицы НЕ является названием.\n"
            "- КРИТИЧНО: если доступны таблицы нескольких типов (факт + справочник) — "
            "включи ОБЕ таблицы в columns и заполни join_keys между ними.\n"
            "- КРИТИЧНО: если пользователь говорит «по X», «сгруппируй по X», «в разбивке по X» "
            "— колонка X должна быть в 'select' И 'group_by', а НЕ в 'filter'. "
            "Роль 'filter' используется ТОЛЬКО для WHERE-условий с конкретным значением "
            "(например: 'регион = Москва', 'дата > 2024-01-01').\n"
            "- КРИТИЧНО: если пользователь спрашивает «сколько всего есть X» или «количество X» "
            "БЕЗ явной группировки («по Y», «в разбивке по Y») — это COUNT DISTINCT, не GROUP BY. "
            "Помести PK-колонку(и) в 'aggregate' и оставь 'group_by' пустым. "
            "Пример: «Сколько клиентов?» → aggregate: [customer_id], group_by: [] "
            "→ SELECT COUNT(DISTINCT customer_id) AS customer_count FROM ...\n"
        )

        # --- Пользовательский промпт ---
        # Интент (компактно)
        intent_repr = json.dumps(intent, ensure_ascii=False, indent=None)

        # Структуры таблиц (полные)
        structures_block = ""
        for tbl, info in sorted(table_structures.items()):
            structures_block += f"\n### {tbl}\n{info}\n"

        # Семплы (обрезаем до 3 строк для компактности)
        samples_block = ""
        for tbl, sample in sorted(table_samples.items()):
            lines = sample.strip().split("\n")
            # Markdown table: header + separator + data rows; берём первые 3 строки данных
            if len(lines) > 5:
                trimmed = "\n".join(lines[:5]) + "\n... (ещё строки)"
            else:
                trimmed = sample
            samples_block += f"\n### {tbl}\n{trimmed}\n"

        # JOIN-анализ
        join_block = ""
        if join_analysis_data:
            join_parts = []
            for pair_key, data in join_analysis_data.items():
                text = data.get("text", "") if isinstance(data, dict) else str(data)
                if text:
                    join_parts.append(text)
            if join_parts:
                join_block = (
                    "\n\n=== JOIN-анализ ===\n" + "\n".join(join_parts)
                )

        user_prompt = (
            f"Запрос пользователя: {user_input}\n\n"
            f"Интент и сущности: {intent_repr}\n\n"
            f"=== Структуры таблиц (все колонки) ==={structures_block}\n\n"
            f"=== Образцы данных (первые строки) ==={samples_block}"
            f"{join_block}"
        )

        # Если sql_planner обнаружил пропущенную таблицу — добавляем корректирующую подсказку
        hint = state.get("column_selector_hint", "")
        if hint:
            user_prompt += f"\n\n=== КОРРЕКТИРУЮЩАЯ ИНСТРУКЦИЯ ===\n{hint}"
            logger.info("ColumnSelector: применяю корректирующую подсказку")

        # Известные ошибки из памяти — передаём чтобы агент не повторял их
        correction_examples: list[str] = state.get("correction_examples", [])
        if not correction_examples:
            try:
                correction_examples = self.memory.get_memory_list("correction_examples")
            except Exception:
                correction_examples = []
        if correction_examples:
            recent = correction_examples[-3:]
            corrections_text = "\n".join(f"- {ex}" for ex in recent)
            user_prompt += (
                f"\n\n=== ИЗВЕСТНЫЕ ОШИБКИ — НЕ ПОВТОРЯТЬ ===\n"
                f"{corrections_text}\n"
                f"Если таблицы из этих примеров присутствуют в текущем запросе — "
                f"учти исправление при выборе join_keys."
            )
            logger.info(
                "ColumnSelector: добавлено %d correction_examples в промпт",
                len(recent),
            )

        system_prompt, user_prompt = self._trim_to_budget(system_prompt, user_prompt)

        if self.debug_prompt:
            print(
                f"\n{'='*80}\n[DEBUG PROMPT — column_selector]\n{'='*80}\n"
                f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n{'='*80}\n"
            )

        response = self.llm.invoke_with_system(
            system_prompt, user_prompt, temperature=0.2
        )

        # --- Парсинг JSON ---
        cleaned = self._clean_llm_json(response)
        parsed: dict[str, Any] = {}
        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            try:
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "ColumnSelector: не удалось распарсить JSON, используем fallback"
                )
                parsed = {"columns": {}, "join_keys": [], "null_warnings": []}

        # --- Извлечение и валидация колонок ---
        raw_columns = parsed.get("columns", {})
        selected_columns: dict[str, Any] = {}

        # Белый список таблиц: фильтруем LLM-ответ, отбрасываем чужие таблицы.
        allowed_tables_set: set[str] = set()
        _allowed_raw = state.get("allowed_tables") or []
        if _allowed_raw:
            allowed_tables_set = {t.lower() for t in _allowed_raw}
            allowed_tables_set -= excluded_tables
            _raw_keys = list(raw_columns.keys())
            for _tk in _raw_keys:
                if _tk.lower() not in allowed_tables_set:
                    logger.warning(
                        "ColumnSelector: таблица %r не в allowed_tables %s — отбрасываем",
                        _tk, _allowed_raw,
                    )
                    raw_columns.pop(_tk, None)

        for table_key, roles in raw_columns.items():
            if not isinstance(roles, dict):
                continue

            # Получаем реальные колонки для валидации
            parts = table_key.split(".", 1)
            if len(parts) == 2:
                real_cols_df = self.schema.get_table_columns(parts[0], parts[1])
                if not real_cols_df.empty and "column_name" in real_cols_df.columns:
                    real_col_names = set(
                        real_cols_df["column_name"].str.lower().tolist()
                    )
                else:
                    real_col_names = None
            else:
                real_col_names = None

            validated_roles: dict[str, list[str]] = {}
            for role in ("select", "filter", "aggregate", "group_by"):
                cols = roles.get(role, [])
                if not isinstance(cols, list):
                    cols = [cols] if cols else []
                validated: list[str] = []
                for col in cols:
                    col_str = str(col).strip()
                    if not col_str:
                        continue
                    if real_col_names is not None:
                        if col_str.lower() not in real_col_names:
                            logger.warning(
                                "ColumnSelector: колонка %s.%s не существует "
                                "в каталоге — пропускаем",
                                table_key, col_str,
                            )
                            continue
                    validated.append(col_str)
                if validated:
                    validated_roles[role] = validated

            if validated_roles:
                selected_columns[table_key] = validated_roles

        # --- JOIN-спецификация ---
        raw_join_keys = parsed.get("join_keys", [])
        join_spec: list[dict[str, Any]] = []
        for jk in raw_join_keys:
            if isinstance(jk, dict) and "left" in jk and "right" in jk:
                entry: dict[str, Any] = {
                    "left": str(jk["left"]),
                    "right": str(jk["right"]),
                    "safe": bool(jk.get("safe", False)),
                    "strategy": str(jk.get("strategy", "direct")),
                }

                join_spec.append(entry)

        # --- Блок B: explicit_join override (LLM путь) ---
        join_spec = _apply_explicit_join_override(
            join_spec, selected_columns, intent,
            user_hints=state.get("user_hints"),
        )

        before_normalize = len(join_spec)
        join_spec = normalize_join_spec(
            join_spec,
            self.schema,
            state.get("table_types", {}) or {},
        )
        if len(join_spec) > before_normalize:
            logger.info(
                "ColumnSelector: дополнено %d составных join-пар после LLM/override",
                len(join_spec) - before_normalize,
            )

        logger.info(
            "ColumnSelector: выбрано колонок для %d таблиц, %d join-ключей",
            len(selected_columns),
            len(join_spec),
        )

        self.memory.add_message(
            "assistant",
            f"[column_selector] Выбраны колонки для: "
            f"{', '.join(selected_columns.keys()) or 'нет'}, "
            f"join-ключей: {len(join_spec)}",
        )

        return {
            "selected_columns": selected_columns,
            "join_spec": join_spec,
            # Сбрасываем hint после использования, чтобы не зациклиться
            "column_selector_hint": "",
            "messages": state["messages"] + [
                {
                    "role": "assistant",
                    "content": (
                        f"Выбраны колонки для таблиц: "
                        f"{', '.join(selected_columns.keys())}\n"
                        f"JOIN-ключей: {len(join_spec)}"
                    ),
                }
            ],
            "graph_iterations": state.get("graph_iterations", 0) + 1,
        }
