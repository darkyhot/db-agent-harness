"""Узел tool_dispatcher: выполнение не-SQL инструментов кодом (без LLM)."""

import logging
from typing import Any

from graph.state import AgentState

logger = logging.getLogger(__name__)


class DispatcherNodes:
    """Mixin с узлом tool_dispatcher."""

    def tool_dispatcher(self, state: AgentState) -> dict[str, Any]:
        """Узел диспетчера инструментов: выполняет не-SQL вызовы кодом.

        Обрабатывает запросы от intent_classifier (needs_search) и
        error_diagnoser (needs_sample). Выполняет инструмент и возвращает
        результат в state для повторного входа в вызывающий узел.
        """
        iterations = state.get("graph_iterations", 0) + 1
        intent = state.get("intent", {})
        diagnosis = state.get("error_diagnosis", {})

        # Случай 1: intent_classifier запросил поиск таблиц
        if intent.get("needs_search"):
            search_query = " ".join(intent.get("entities", []))
            if not search_query:
                search_query = state["user_input"]

            logger.info("ToolDispatcher: поиск таблиц по запросу: %s", search_query[:100])

            # Пробуем search_by_description, затем search_tables
            result_str = ""
            for tool_name in ("search_by_description", "search_tables"):
                if tool_name in self.tool_map:
                    tool_result = self._call_tool(tool_name, {"query": search_query})
                    if tool_result.success and tool_result.data:
                        result_str = str(tool_result)
                        break
                    result_str = str(tool_result)

            # Обновляем intent — поиск выполнен
            updated_intent = dict(intent)
            updated_intent["needs_search"] = False
            updated_intent["search_results"] = result_str[:3000]

            self.memory.add_message("tool", f"[tool_dispatcher:search] {result_str[:500]}")

            return {
                "intent": updated_intent,
                "graph_iterations": iterations,
                "tool_calls": state.get("tool_calls", []) + [
                    {"tool": "search", "args": {"query": search_query}, "result": result_str}
                ],
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"Поиск таблиц: {result_str[:1000]}"}
                ],
            }

        # Случай 2: error_diagnoser запросил sample данных
        if diagnosis.get("needs_sample"):
            # Извлекаем таблицу из диагноза или из последнего SQL
            sample_table = diagnosis.get("sample_table", "")
            if not sample_table:
                # Пробуем извлечь из SQL в последнем tool_call
                import re
                recent_calls = state.get("tool_calls", [])
                if recent_calls:
                    last_sql = recent_calls[-1].get("args", {}).get("sql", "")
                    tables_in_sql = re.findall(
                        r'\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b', last_sql,
                    )
                    if tables_in_sql:
                        sample_table = f"{tables_in_sql[0][0]}.{tables_in_sql[0][1]}"

            result_str = ""
            if sample_table and "." in sample_table:
                schema_name, table_name = sample_table.split(".", 1)
                logger.info("ToolDispatcher: загрузка sample для %s", sample_table)
                tool_result = self._call_tool("get_sample", {
                    "schema": schema_name, "table": table_name, "n": 10,
                })
                result_str = str(tool_result)

                # Добавляем sample в table_samples
                table_samples = dict(state.get("table_samples", {}))
                table_samples[sample_table] = result_str
            else:
                result_str = "Не удалось определить таблицу для загрузки sample"
                table_samples = state.get("table_samples", {})

            # Обновляем диагноз — sample загружен
            updated_diagnosis = dict(diagnosis)
            updated_diagnosis["needs_sample"] = False
            updated_diagnosis["sample_data"] = result_str[:3000]

            self.memory.add_message("tool", f"[tool_dispatcher:sample] {result_str[:500]}")

            return {
                "error_diagnosis": updated_diagnosis,
                "table_samples": table_samples,
                "graph_iterations": iterations,
                "tool_calls": state.get("tool_calls", []) + [
                    {"tool": "get_sample", "args": {"table": sample_table}, "result": result_str}
                ],
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"Sample загружен: {result_str[:1000]}"}
                ],
            }

        # Случай по умолчанию: нечего диспатчить
        logger.warning("ToolDispatcher: вызван без активного запроса на инструмент")
        return {"graph_iterations": iterations}
