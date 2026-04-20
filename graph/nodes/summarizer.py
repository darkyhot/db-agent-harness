"""Узел summarizer: формирование финального ответа пользователю."""

import json
import logging
from typing import Any

from graph.state import AgentState

logger = logging.getLogger(__name__)


class SummarizerNodes:
    """Mixin с узлом summarizer."""

    def _extract_preview_markdown(self, tool_calls: list[dict[str, Any]]) -> str:
        """Достать markdown-preview из последнего execute_query."""
        for tc in reversed(tool_calls or []):
            if tc.get("tool") != "execute_query":
                continue
            payload = self._parse_sql_tool_payload(tc.get("result", ""))
            if not payload:
                continue
            preview = str(payload.get("preview_markdown", "") or "").strip()
            if preview:
                return preview
        return ""

    def _get_summarizer_system_prompt(self) -> str:
        """Системный промпт для формирования финального ответа."""
        return (
            "Ты — аналитический агент для Greenplum. Формируешь финальный ответ пользователю.\n\n"
            "Правила ответа:\n"
            "- Отвечай на русском языке\n"
            "- SQL-алиасы только на английском\n"
            "- Табличные данные оформляй в markdown-таблицу\n"
            "- SQL-код оборачивай в ```sql блок\n"
            "- Не пересказывай шаги плана — только результат\n"
            "- Не повторяй вопрос пользователя\n"
            "- Если были предупреждения — упомяни кратко в конце\n"
            "- Если был выполнен SQL-запрос — покажи его в блоке ```sql и кратко объясни логику\n"
            "- Интерпретируй результат в бизнес-терминах, если это возможно\n"
            "- Если данные обрезаны — укажи это и покажи общее количество строк\n"
            "- Если результат большой — покажи топ-10 строк и общую статистику\n"
            "- КРИТИЧНО: в блоке ```sql РАЗРЕШЁН ТОЛЬКО тот запрос, который дословно "
            "находится в разделе «Результаты инструментов». Запрещено писать ЛЮБОЙ другой "
            "SQL — даже если считаешь, что результат не отвечает на вопрос. "
            "Если данных недостаточно — напиши ТОЛЬКО текст: "
            "«Данных недостаточно, требуется дополнительный запрос» — без SQL-блока.\n"
        )

    def summarizer(self, state: AgentState) -> dict[str, Any]:
        """Узел формирования финального ответа пользователю."""
        # Сохраняем примеры исправлений в долгосрочную память
        new_examples = state.get("correction_examples", [])
        if new_examples:
            existing = self.memory.get_memory_list("correction_examples")
            combined = (existing + new_examples)[-20:]
            self.memory.set_memory("correction_examples", json.dumps(combined, ensure_ascii=False))

        # Если уже есть финальный ответ (например, от corrector при исчерпании попыток)
        if state.get("final_answer"):
            self.memory.add_message("assistant", state["final_answer"])
            return {}

        # Используем только последние N tool_calls — предотвращаем раздувание промпта
        capped_tool_calls = self._cap_tool_calls(state.get("tool_calls", []))
        tool_results_parts = []
        for tc in capped_tool_calls:
            sql = tc.get("args", {}).get("sql", "")
            sql_line = f"\n  SQL: {sql}" if sql else ""
            tool_results_parts.append(
                f"- {tc['tool']}{sql_line}\n  Результат: {tc['result'][:5000]}"
            )
        tool_results = "\n".join(tool_results_parts)

        plan_text = "\n".join(state.get("plan", []))

        system_prompt = self._get_summarizer_system_prompt()

        # Используем структурированные данные вместо монолитного tables_context
        tables_summary = ""
        table_structures = state.get("table_structures", {})
        if table_structures:
            tables_summary = "Контекст таблиц (колонки):\n" + "\n\n".join(
                f"### {name}\n{info}" for name, info in table_structures.items()
            )
        elif state.get("tables_context", ""):
            # Fallback: старый формат (обратная совместимость)
            tables_ctx = state["tables_context"]
            ctx_lines = tables_ctx.split("\n")
            summary_lines = []
            skip_sample = False
            for line in ctx_lines:
                if "Образец данных" in line or "Sample" in line:
                    skip_sample = True
                    continue
                if skip_sample and (line.strip() == "" or line.startswith("[")):
                    skip_sample = False
                if not skip_sample:
                    summary_lines.append(line)
            tables_summary = "\n".join(summary_lines).strip()

        user_parts = [f"Запрос пользователя: {state['user_input']}"]
        user_parts.append(f"План:\n{plan_text}")
        if tables_summary:
            user_parts.append(tables_summary)
        user_parts.append(f"Результаты инструментов:\n{tool_results}")
        user_prompt = "\n\n".join(user_parts)

        system_prompt, user_prompt = self._trim_to_budget(system_prompt, user_prompt)

        if self.debug_prompt:
            print(f"\n{'='*80}\n[DEBUG PROMPT — summarizer]\n{'='*80}\n"
                  f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n{'='*80}\n")

        answer = self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.2)
        preview_markdown = self._extract_preview_markdown(capped_tool_calls)
        if preview_markdown and preview_markdown not in answer:
            answer = (
                f"{answer.rstrip()}\n\n"
                f"Предварительный результат:\n{preview_markdown}"
            ).strip()
        self.memory.add_message("assistant", answer)

        logger.info("Summarizer: ответ сформирован")
        return {
            "final_answer": answer,
            "messages": state["messages"] + [
                {"role": "assistant", "content": answer}
            ],
        }
