"""LLM-центричный оркестратор (plan-and-execute).

В центре агента — orchestrator-узел: он динамически выбирает следующий шаг
из реестра способностей, перепланирует на ошибках и сам решает, когда задача
завершена. Чистый сырой SQL обходит LLM полностью (детерминированный guard),
а тяжёлый аналитический pipeline вызывается как один составной шаг
``run_analytics`` (скомпилированный build_analytics_subgraph).

Каждый step-узел возвращает управление обратно в ``orchestrator`` — это
единственный хаб маршрутизации (см. _route_after_orchestrator в graph.graph).
"""

from __future__ import annotations

import logging
import re
from pathlib import PurePosixPath
from typing import Any

import sqlparse

from core.log_safety import _extract_tables, summarize_sql, summarize_text
from core.sql_validator import SQLMode
from graph.state import AgentState

logger = logging.getLogger(__name__)


# Шаги, которые LLM-планировщику разрешено выбирать. "summarize"/"finish" —
# терминальные директивы (роутер уводит их в summarizer), "ask_clarification"
# выставляет needs_clarification и отдаёт управление CLI.
_VALID_ORCH_STEPS = frozenset({
    "extract_sources",
    "pull_metadata",
    "explain_plan",
    "explain_sql",
    "execute_sql",
    "create_directory",
    "file_operation",
    "run_analytics",
    "summarize",
    "finish",
    "ask_clarification",
})

_FILE_OP_TOOLS = frozenset({
    "create_file",
    "read_file",
    "edit_file",
    "delete_file",
    "list_files",
})

# Ведущее ключевое слово SQL после необязательных строчных комментариев.
_SQL_LEAD_RE = re.compile(
    r"^\s*(?:--[^\n]*\n\s*)*"
    r"(select|with|insert|update|delete|create|alter|drop|truncate|merge)\b",
    re.IGNORECASE,
)


def _extract_leading_sql(text: str) -> str | None:
    """Вернуть SQL, если ввод — это самостоятельный SQL-стейтмент.

    Возвращает None для естественного языка (в т.ч. "объясни запрос: SELECT …",
    т.к. первый токен — не SQL-ключевое слово). LLM не участвует — поэтому
    чистый случай "select * from t" не может «упасть».
    """
    s = (text or "").strip()
    if not s or not _SQL_LEAD_RE.match(s):
        return None
    try:
        statements = [st for st in sqlparse.parse(s) if str(st).strip()]
    except Exception:  # noqa: BLE001
        return None
    if not statements:
        return None
    if all(st.get_type() == "UNKNOWN" for st in statements):
        return None
    if len(statements) == 1:
        return s.rstrip(";").strip()
    return s


_CREATE_DIR_RE = re.compile(
    r"^\s*(?:создай|создать|сделай|сделать)\s+(?:мне\s+)?"
    r"(?:папку|директори[юя]|каталог)\s*(?P<path>.*)$",
    re.IGNORECASE,
)
_MKDIR_RE = re.compile(r"^\s*mkdir\s+(?:-p\s+)?(?P<path>.+)$", re.IGNORECASE)


def _extract_create_directory_path(text: str) -> str | None:
    """Return requested workspace-relative directory path, if input asks for it."""
    raw = (text or "").strip()
    match = _CREATE_DIR_RE.match(raw) or _MKDIR_RE.match(raw)
    if not match:
        return None
    path = str(match.group("path") or "").strip()
    path = path.strip("`'\"«»“”")
    path = re.sub(r"\s+(?:пожалуйста|please)\s*$", "", path, flags=re.IGNORECASE).strip()
    path = path.rstrip(" .;")
    return path


def _strip_workspace_prefix(path: str) -> str:
    """Убрать ведущий 'workspace/' из пути, чтобы пользователь мог писать оба варианта."""
    p = path.strip().replace("\\", "/")
    if p.lower().startswith("workspace/"):
        p = p[len("workspace/"):]
    return p.strip("/")


def _directory_path_error(path: str) -> str:
    """Validate path before passing it to the workspace-scoped filesystem tool."""
    if not path:
        return "Не указано имя папки. Укажите относительный путь внутри workspace/."

    normalized = path.replace("\\", "/").strip()
    posix = PurePosixPath(normalized)
    if posix.is_absolute() or re.match(r"^[A-Za-z]:", normalized):
        return "Можно создавать папки только по относительному пути внутри workspace/."
    if any(part in ("", ".", "..") for part in posix.parts):
        return "Путь папки должен оставаться внутри workspace/ и не может содержать '.' или '..'."
    return ""


class OrchestratorNodes:
    """Mixin: orchestrator-узел и step-узлы plan-and-execute графа.

    Доступно из BaseNodeMixin: self.llm, self.db, self.schema, self.validator,
    self.memory, self._call_tool, self._llm_json_with_retry, self._trim_to_budget,
    self._render_tool_result, self._get_tables_detail_context, self.summarizer.
    self._analytics_subgraph выставляется в build_orchestrated_graph.
    """

    # ------------------------------------------------------------------
    # Планировщик
    # ------------------------------------------------------------------

    def orchestrator(self, state: AgentState) -> dict[str, Any]:
        iterations = state.get("graph_iterations", 0) + 1
        step_count = int(state.get("orch_step_count", 0) or 0)
        user_input = state.get("user_input", "") or ""

        paused = bool(
            state.get("plan_preview_pending")
            or state.get("needs_clarification")
            or state.get("needs_confirmation")
            or state.get("needs_disambiguation")
        )

        # 0) Уже есть финальный ответ и нет паузы → завершаемся. Должно стоять
        #    ПЕРЕД resume-проверкой: иначе после run_analytics, который оставил
        #    в state plan_preview_approved+sql_blueprint, resume-условие
        #    зациклило бы повторный вызов аналитики.
        if state.get("final_answer") and not paused:
            return {"orch_next_step": "finish", "graph_iterations": iterations}

        # 1) Resume после паузы на пользователя (plan-preview / clarify /
        #    confirm). Зеркалит fast-path query_interpreter — на рекурсивном
        #    перезапуске из CLI идём прямо в аналитический шаг, не перепланируя.
        if (
            state.get("orch_resume_step") == "run_analytics"
            or (state.get("plan_edit_text") and state.get("sql_blueprint"))
            or (state.get("plan_preview_approved") and state.get("sql_blueprint"))
        ):
            logger.info("Orchestrator: resume → run_analytics")
            return {
                "orch_next_step": "run_analytics",
                "orch_resume_step": "run_analytics",
                "graph_iterations": iterations,
            }

        # 2) Терминальная директива, выставленная step-узлом — пропускаем без
        #    повторного планирования (иначе лишний LLM-вызов / зацикливание).
        pending = str(state.get("orch_next_step") or "")
        if pending in ("summarize", "finish") and state.get("orch_history"):
            return {"orch_next_step": pending, "graph_iterations": iterations}

        # 3) Очередь предзапланированных шагов (детерм. guard / полный LLM-план).
        queued = list(state.get("orch_plan") or [])
        if queued:
            nxt = queued.pop(0)
            logger.info("Orchestrator: queued → %s", nxt)
            return {
                "orch_next_step": nxt,
                "orch_plan": queued,
                "orch_step_count": step_count + 1,
                "graph_iterations": iterations,
            }

        # 4) Первый вход: детерминированный guard сырого SQL (без LLM).
        if step_count == 0 and not state.get("orch_history"):
            raw = _extract_leading_sql(user_input)
            if raw is not None:
                logger.info(
                    "Orchestrator: deterministic raw-SQL guard → execute_sql %s",
                    summarize_sql(raw),
                )
                return {
                    "orch_sql": raw,
                    "orch_next_step": "execute_sql",
                    "orch_plan": ["summarize"],
                    "orch_step_count": step_count + 1,
                    "orch_history": [{
                        "step": "guard:raw_sql",
                        "reason": "input is a standalone SQL statement",
                        "ok": True,
                    }],
                    "graph_iterations": iterations,
                }

            fs_path = _extract_create_directory_path(user_input)
            if fs_path is not None:
                logger.info("Orchestrator: deterministic create-directory guard → %s", fs_path or "<empty>")
                return {
                    "orch_fs_path": fs_path,
                    "orch_next_step": "create_directory",
                    "orch_step_count": step_count + 1,
                    "orch_history": [{
                        "step": "guard:create_directory",
                        "reason": "input asks to create workspace directory",
                        "ok": True,
                    }],
                    "graph_iterations": iterations,
                }

        # 4b) Раньше здесь стоял детерминированный NL fast-path, который любой
        #     не-SQL ввод жёстко гнал в run_analytics, минуя решение
        #     оркестратора. Это лишало агента способности самому решить:
        #     запустить аналитику, вызвать tool, уточнить или просто ответить
        #     (напр. запрос "создай папку" вне его возможностей). Fast-path
        #     убран — не-SQL ввод проходит через LLM-планирование (шаг 5).
        #     Платим ~5с на вызов оркестратора ради корректной агентной
        #     маршрутизации (снята P5-оптимизация латентности).

        # 5) LLM-планирование.
        logger.info(
            "Orchestrator: planning %s (step=%d)",
            summarize_text(user_input, label="user_input"),
            step_count,
        )
        system_prompt = self._orchestrator_system_prompt()
        user_prompt = self._orchestrator_user_prompt(state)
        system_prompt, user_prompt = self._trim_to_budget(system_prompt, user_prompt)
        if self.debug_prompt:
            print(
                f"\n{'='*80}\n[DEBUG PROMPT — orchestrator]\n{'='*80}\n"
                f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n{'='*80}\n"
            )

        decision = self._llm_json_with_retry(
            system_prompt,
            user_prompt,
            temperature=0.0,
            failure_tag="orchestrator",
            expect="object",
        )

        history = list(state.get("orch_history", []))
        if not isinstance(decision, dict) or not decision.get("step"):
            logger.warning("Orchestrator: нет валидного решения LLM — fallback run_analytics")
            return {
                "orch_next_step": "run_analytics",
                "orch_step_count": step_count + 1,
                "graph_iterations": iterations,
                "orch_history": history + [{
                    "step": "run_analytics",
                    "reason": "planner fallback (no valid decision)",
                    "ok": True,
                }],
            }

        step = str(decision.get("step") or "").strip()
        reason = str(decision.get("reason") or "")[:300]
        sql = str(decision.get("sql") or "").strip()
        fs_path = str(decision.get("path") or decision.get("folder") or "").strip()
        fs_tool = str(decision.get("fs_tool") or "").strip()
        fs_content = str(decision.get("content") or "").strip()
        question = str(decision.get("question") or "").strip()
        answer = str(decision.get("answer") or "").strip()

        if step not in _VALID_ORCH_STEPS:
            logger.warning("Orchestrator: неизвестный шаг '%s' → run_analytics", step)
            step = "run_analytics"

        update: dict[str, Any] = {
            "orch_next_step": step,
            "orch_step_count": step_count + 1,
            "graph_iterations": iterations,
            "orch_history": history + [{"step": step, "reason": reason, "ok": True}],
        }
        if sql:
            update["orch_sql"] = sql
        if fs_path:
            update["orch_fs_path"] = fs_path
        if fs_tool:
            update["orch_fs_tool"] = fs_tool
        if fs_content:
            update["orch_fs_content"] = fs_content

        # Необязательный полный план: первый шаг сейчас, остаток — в очередь.
        plan = decision.get("plan")
        if isinstance(plan, list) and plan:
            seq = [str(x).strip() for x in plan if str(x).strip() in _VALID_ORCH_STEPS]
            if seq:
                update["orch_next_step"] = seq[0]
                update["orch_plan"] = seq[1:]
                step = seq[0]

        if step == "ask_clarification":
            update["needs_clarification"] = True
            update["clarification_message"] = question or "Уточните, пожалуйста, запрос."
        elif step == "finish":
            if answer:
                update["final_answer"] = answer

        return update

    # ------------------------------------------------------------------
    # Step-узлы (каждый возвращает управление в orchestrator)
    # ------------------------------------------------------------------

    def orch_extract_sources(self, state: AgentState) -> dict[str, Any]:
        sql = state.get("orch_sql") or state.get("user_input", "") or ""
        tables = _extract_tables(sql)
        real = self._filter_catalog_tables(tables)
        sources = real or tables
        logger.info("Orchestrator/extract_sources: %s", sources)
        return {
            "orch_sources": sources,
            "orch_next_step": "",
            "orch_history": list(state.get("orch_history", [])) + [{
                "step": "extract_sources",
                "reason": f"{len(sources)} tables",
                "ok": True,
            }],
        }

    def orch_pull_metadata(self, state: AgentState) -> dict[str, Any]:
        sql = state.get("orch_sql") or state.get("user_input", "") or ""
        try:
            metadata = self._get_tables_detail_context(sql)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Orchestrator/pull_metadata failed: %s", exc)
            metadata = ""
        return {
            "orch_metadata": metadata,
            "orch_next_step": "",
            "orch_history": list(state.get("orch_history", [])) + [{
                "step": "pull_metadata",
                "reason": "ok" if metadata else "no catalog match",
                "ok": True,
            }],
        }

    def orch_explain_plan(self, state: AgentState) -> dict[str, Any]:
        sql = (state.get("orch_sql") or "").strip()
        if not sql:
            plan_text = "(EXPLAIN недоступен: нет SQL)"
            ok = False
        else:
            try:
                plan_text = str(self.db.explain_query(sql))
                ok = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Orchestrator/explain_plan failed: %s", exc)
                plan_text = f"(EXPLAIN недоступен: {exc})"
                ok = False
        return {
            "orch_explain_plan": plan_text,
            "orch_next_step": "",
            "orch_history": list(state.get("orch_history", [])) + [{
                "step": "explain_plan",
                "reason": "ok" if ok else "unavailable",
                "ok": ok,
            }],
        }

    def orch_explain_sql(self, state: AgentState) -> dict[str, Any]:
        sql = (state.get("orch_sql") or state.get("user_input", "") or "").strip()
        if not sql:
            msg = "Не нашёл SQL для объяснения. Пришлите запрос текстом."
            self.memory.add_message("assistant", msg)
            return {
                "final_answer": msg,
                "orch_next_step": "finish",
                "orch_history": list(state.get("orch_history", [])) + [{
                    "step": "explain_sql", "reason": "no sql", "ok": False,
                }],
            }

        # Досбор недостающих артефактов (оркестратор мог не вызвать шаги явно).
        metadata = state.get("orch_metadata") or ""
        if not metadata:
            try:
                metadata = self._get_tables_detail_context(sql)
            except Exception:  # noqa: BLE001
                metadata = ""
        explain_plan = state.get("orch_explain_plan") or ""
        if not explain_plan:
            try:
                explain_plan = str(self.db.explain_query(sql))
            except Exception as exc:  # noqa: BLE001
                explain_plan = f"(EXPLAIN недоступен: {exc})"

        system_prompt = (
            "Ты — аналитический SQL-агент. Объясни пользователю SQL-запрос на "
            "русском языке: что он делает, какие таблицы и колонки использует, "
            "какие фильтры/агрегации/джойны применяются, на какой вопрос отвечает.\n"
            "- Опирайся на описания таблиц/колонок из справочника, если они даны.\n"
            "- Если есть EXPLAIN-план — кратко прокомментируй стоимость и риски "
            "(seq scan, дорогие join, потенциальное размножение строк).\n"
            "- НЕ выполняй запрос и НЕ переписывай его. Только объяснение.\n"
            "- Структурируй ответ: суть → таблицы/колонки → фильтры/агрегации → "
            "(опц.) замечания по плану."
        )
        parts = [f"SQL-запрос:\n```sql\n{sql}\n```"]
        if metadata:
            parts.append(f"Справочник задействованных таблиц:\n{metadata}")
        if explain_plan:
            parts.append(f"EXPLAIN-план:\n{explain_plan}")
        user_prompt = "\n\n".join(parts)
        system_prompt, user_prompt = self._trim_to_budget(system_prompt, user_prompt)

        answer = str(self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.2))
        self.memory.add_message("assistant", answer)
        logger.info("Orchestrator/explain_sql: объяснение сформировано")
        return {
            "final_answer": answer,
            "orch_metadata": metadata,
            "orch_explain_plan": explain_plan,
            "orch_next_step": "finish",
            "messages": state["messages"] + [{"role": "assistant", "content": answer}],
            "orch_history": list(state.get("orch_history", [])) + [{
                "step": "explain_sql", "reason": "explained", "ok": True,
            }],
        }

    def orch_execute_sql(self, state: AgentState) -> dict[str, Any]:
        sql = (state.get("orch_sql") or state.get("user_input", "") or "").strip()
        sql = sql.rstrip(";").strip()
        history = list(state.get("orch_history", []))
        if not sql:
            msg = "Не нашёл SQL для выполнения."
            self.memory.add_message("assistant", msg)
            return {
                "final_answer": msg,
                "orch_next_step": "finish",
                "orch_history": history + [{"step": "execute_sql", "reason": "no sql", "ok": False}],
            }

        result = self.validator.validate(sql)
        if not result.is_valid:
            msg = "SQL не прошёл валидацию:\n" + "\n".join(result.errors or ["неизвестная ошибка"])
            self.memory.add_message("assistant", msg)
            return {
                "final_answer": msg,
                "orch_next_step": "finish",
                "orch_history": history + [{"step": "execute_sql", "reason": "invalid sql", "ok": False}],
            }

        mode = result.mode

        if mode == SQLMode.READ:
            tool_result = self._call_tool("execute_query", {"sql": sql})
            raw = str(tool_result)
            tool_calls = list(state.get("tool_calls", [])) + [
                {"tool": "execute_query", "args": {"sql": sql}, "result": raw}
            ]
            if not tool_result.success:
                rendered = self._render_tool_result(raw)
                err = f"Ошибка выполнения SQL:\n{rendered}"
                self.memory.add_message("assistant", err)
                return {
                    "final_answer": err,
                    "tool_calls": tool_calls,
                    "orch_next_step": "finish",
                    "orch_history": history + [{"step": "execute_sql", "reason": "exec error", "ok": False}],
                }
            self.memory.add_message("tool", f"[execute_query] {self._render_tool_result(raw)[:500]}")
            return {
                "tool_calls": tool_calls,
                "orch_next_step": "summarize",
                "orch_history": history + [{"step": "execute_sql", "reason": "READ ok", "ok": True}],
                "messages": state["messages"] + [
                    {"role": "assistant", "content": "Сырой SQL выполнен (READ)."}
                ],
            }

        # WRITE / DDL → подтверждение через существующий CLI-обработчик.
        tool_name = "execute_write" if mode == SQLMode.WRITE else "execute_ddl"
        conf_msg = result.confirmation_message or (
            f"Запрос изменяет данные (режим {mode.value}). Выполнить?\n\n```sql\n{sql}\n```"
        )
        logger.info("Orchestrator/execute_sql: %s требует подтверждения", mode.value)
        return {
            "needs_confirmation": True,
            "confirmation_message": conf_msg,
            "sql_to_validate": sql,
            "pending_sql_tool_call": {"tool": tool_name, "args": {"sql": sql}},
            "orch_next_step": "",
            "orch_history": history + [{
                "step": "execute_sql",
                "reason": f"{mode.value} needs confirmation",
                "ok": True,
            }],
        }

    def orch_create_directory(self, state: AgentState) -> dict[str, Any]:
        path = _strip_workspace_prefix(str(state.get("orch_fs_path") or ""))
        history = list(state.get("orch_history", []))
        error = _directory_path_error(path)
        if error:
            self.memory.add_message("assistant", error)
            return {
                "final_answer": error,
                "orch_next_step": "finish",
                "orch_history": history + [{
                    "step": "create_directory",
                    "reason": "invalid workspace path",
                    "ok": False,
                }],
            }

        if "create_directory" not in self.tool_map:
            msg = "Инструмент создания папок недоступен в текущей сборке агента."
            self.memory.add_message("assistant", msg)
            return {
                "final_answer": msg,
                "orch_next_step": "finish",
                "orch_history": history + [{
                    "step": "create_directory",
                    "reason": "tool missing",
                    "ok": False,
                }],
            }

        tool_result = self._call_tool("create_directory", {"path": path})
        result_text = str(tool_result)
        tool_calls = list(state.get("tool_calls", [])) + [
            {"tool": "create_directory", "args": {"path": path}, "result": result_text}
        ]
        if not tool_result.success:
            msg = f"Не удалось создать папку: {result_text}"
            self.memory.add_message("assistant", msg)
            return {
                "final_answer": msg,
                "tool_calls": tool_calls,
                "orch_next_step": "finish",
                "orch_history": history + [{
                    "step": "create_directory",
                    "reason": "tool error",
                    "ok": False,
                }],
            }

        answer = f"Папка создана в workspace/: {path}"
        self.memory.add_message("tool", f"[create_directory] {result_text}")
        self.memory.add_message("assistant", answer)
        return {
            "final_answer": answer,
            "tool_calls": tool_calls,
            "orch_next_step": "finish",
            "messages": state["messages"] + [{"role": "assistant", "content": answer}],
            "orch_history": history + [{
                "step": "create_directory",
                "reason": "ok",
                "ok": True,
            }],
        }

    def orch_file_operation(self, state: AgentState) -> dict[str, Any]:
        fs_tool = str(state.get("orch_fs_tool") or "").strip()
        path = _strip_workspace_prefix(str(state.get("orch_fs_path") or ""))
        content = str(state.get("orch_fs_content") or "")
        history = list(state.get("orch_history", []))

        if fs_tool not in _FILE_OP_TOOLS:
            msg = (
                f"Неизвестная файловая операция: '{fs_tool}'. "
                f"Допустимые: {', '.join(sorted(_FILE_OP_TOOLS))}."
            )
            self.memory.add_message("assistant", msg)
            return {
                "final_answer": msg,
                "orch_next_step": "finish",
                "orch_history": history + [{"step": "file_operation", "reason": "unknown tool", "ok": False}],
            }

        if fs_tool == "list_files":
            args: dict[str, Any] = {"subdir": path}
        elif fs_tool in ("create_file", "edit_file"):
            if not path:
                msg = "Не указан путь файла."
                self.memory.add_message("assistant", msg)
                return {
                    "final_answer": msg,
                    "orch_next_step": "finish",
                    "orch_history": history + [{"step": "file_operation", "reason": "no path", "ok": False}],
                }
            args = {"path": path, "content": content}
        else:
            if not path:
                msg = "Не указан путь файла."
                self.memory.add_message("assistant", msg)
                return {
                    "final_answer": msg,
                    "orch_next_step": "finish",
                    "orch_history": history + [{"step": "file_operation", "reason": "no path", "ok": False}],
                }
            args = {"path": path}

        tool_result = self._call_tool(fs_tool, args)
        result_text = str(tool_result)
        tool_calls = list(state.get("tool_calls", [])) + [
            {"tool": fs_tool, "args": args, "result": result_text}
        ]
        if not tool_result.success:
            msg = f"Ошибка операции {fs_tool}: {result_text}"
            self.memory.add_message("assistant", msg)
            return {
                "final_answer": msg,
                "tool_calls": tool_calls,
                "orch_next_step": "finish",
                "orch_history": history + [{"step": "file_operation", "reason": "tool error", "ok": False}],
            }

        answer = self._render_tool_result(result_text)
        self.memory.add_message("tool", f"[{fs_tool}] {answer[:500]}")
        self.memory.add_message("assistant", answer)
        return {
            "final_answer": answer,
            "tool_calls": tool_calls,
            "orch_next_step": "finish",
            "messages": state["messages"] + [{"role": "assistant", "content": answer}],
            "orch_history": history + [{"step": "file_operation", "reason": f"{fs_tool} ok", "ok": True}],
        }

    def orch_run_analytics(self, state: AgentState) -> dict[str, Any]:
        sub = getattr(self, "_analytics_subgraph", None)
        history = list(state.get("orch_history", []))
        if sub is None:
            logger.error("Orchestrator/run_analytics: подграф не привязан")
            return {
                "final_answer": "Внутренняя ошибка: аналитический pipeline не инициализирован.",
                "orch_next_step": "finish",
                "orch_history": history + [{"step": "run_analytics", "reason": "no subgraph", "ok": False}],
            }

        final_state: dict[str, Any] = dict(state)
        try:
            for event in sub.stream(dict(state)):
                payload = list(event.values())[0]
                if isinstance(payload, dict):
                    final_state.update(payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Orchestrator/run_analytics failed")
            return {
                "final_answer": f"Аналитический pipeline завершился ошибкой: {exc}",
                "orch_next_step": "finish",
                "orch_history": history + [{"step": "run_analytics", "reason": str(exc)[:200], "ok": False}],
            }

        paused = bool(
            final_state.get("plan_preview_pending")
            or final_state.get("needs_clarification")
            or final_state.get("needs_confirmation")
            or final_state.get("needs_disambiguation")
        )
        out = dict(final_state)
        out["orch_history"] = history + [{
            "step": "run_analytics",
            "reason": "paused" if paused else "done",
            "ok": True,
        }]
        if paused:
            out["orch_resume_step"] = "run_analytics"
            out["orch_next_step"] = "run_analytics"
        else:
            out["orch_resume_step"] = ""
            out["orch_next_step"] = "finish"
        return out

    # ------------------------------------------------------------------
    # Промпты планировщика
    # ------------------------------------------------------------------

    def _orchestrator_system_prompt(self) -> str:
        return (
            "Ты — оркестратор аналитического SQL-агента (plan-and-execute).\n"
            "Каждый ход выбирай ОДИН следующий шаг из реестра (или верни полный "
            "план списком 'plan'). Решай по контексту и журналу уже сделанного.\n\n"
            "Реестр шагов:\n"
            "- execute_sql: пользователь прислал ГОТОВЫЙ SQL и хочет его выполнить. "
            "Положи запрос в поле 'sql'.\n"
            "- create_directory: пользователь просит создать папку/директорию. "
            "Разрешено только внутри workspace/; путь положи в поле 'path'.\n"
            "- file_operation: пользователь просит создать/прочитать/изменить/удалить "
            "файл или показать список файлов в workspace/. Укажи:\n"
            "  'fs_tool': одно из create_file|read_file|edit_file|delete_file|list_files,\n"
            "  'path': относительный путь внутри workspace/ (без префикса 'workspace/'),\n"
            "  'content': содержимое файла (только для create_file и edit_file).\n"
            "- explain_sql: пользователь просит ОБЪЯСНИТЬ присланный SQL (не "
            "выполнять). Положи запрос в 'sql'. Можно предварительно вызвать "
            "extract_sources/pull_metadata/explain_plan.\n"
            "- extract_sources: извлечь таблицы из SQL (поле 'sql').\n"
            "- pull_metadata: подтянуть описания таблиц/колонок из справочника.\n"
            "- explain_plan: получить EXPLAIN-план SQL из БД (без выполнения).\n"
            "- run_analytics: пользователь задал вопрос ПО ДАННЫМ каталога на "
            "естественном языке (посчитай/покажи/сравни/сколько…) — запустить "
            "полноценный аналитический pipeline.\n"
            "- ask_clarification: запрос про данные, но непонятен/неоднозначен "
            "(неясно что считать, какая сущность/период) — задай вопрос в поле "
            "'question'.\n"
            "- summarize: данные уже получены (execute_sql) — сформировать ответ.\n"
            "- finish: задача завершена ИЛИ запрос вне твоих возможностей. "
            "Готовый текст ответа положи в 'answer'.\n\n"
            "Границы возможностей:\n"
            "- Ты умеешь отвечать на вопросы по данным каталога (SQL-аналитика, "
            "объяснение/выполнение SQL), создавать папки и работать с файлами "
            "(создать, прочитать, изменить, удалить, показать список) внутри workspace/.\n"
            "- Действия вне этого (отправить письмо, операции ОС вне workspace/, "
            "изменить систему) и болтовня — "
            "ВЫПОЛНИТЬ НЕЛЬЗЯ: выбирай 'finish' "
            "и в 'answer' честно скажи, что это вне твоих возможностей и что ты "
            "отвечаешь на вопросы по данным каталога. НЕ запускай run_analytics "
            "для таких запросов.\n\n"
            "Правила:\n"
            "- Верни ТОЛЬКО JSON-объект, без markdown.\n"
            "- Формат: {\"step\": \"...\", \"sql\": \"...\", \"path\": \"...\", "
            "\"fs_tool\": \"...\", \"content\": \"...\", "
            "\"question\": \"...\", \"answer\": \"...\", \"plan\": [\"...\"], \"reason\": \"...\"}.\n"
            "- 'sql'/'path'/'fs_tool'/'content'/'question'/'answer'/'plan' заполняй только когда нужны.\n"
            "- Если запрос ПО ДАННЫМ, но сомневаешься в деталях — run_analytics "
            "(безопасный дефолт для аналитики). Сомнение про данные ≠ запрос вне "
            "возможностей: последнее всегда 'finish'.\n"
            "- Не зацикливайся: после execute_sql(READ) иди в summarize/finish; "
            "после explain_sql — finish.\n"
        )

    def _orchestrator_user_prompt(self, state: AgentState) -> str:
        parts: list[str] = [f"Запрос пользователя:\n{state.get('user_input', '') or ''}"]

        # Контекст предыдущих ходов из сессионной памяти — позволяет разрешать
        # анафоры ("этот файл", "ту папку") и строить на предыдущих действиях.
        try:
            session_msgs = self.memory.get_session_messages() or []
            if session_msgs:
                # Берём последние 10 сообщений; каждое обрезаем до 250 символов
                recent = session_msgs[-10:]
                lines = [
                    f"[{m.get('role', '')}] {str(m.get('content', '') or '')[:250]}"
                    for m in recent
                ]
                parts.append("Контекст сессии (последние действия):\n" + "\n".join(lines))
        except Exception:
            pass

        prev_sql = state.get("prev_sql", "") or ""
        if prev_sql:
            parts.append(f"Предыдущий SQL (multi-turn):\n{prev_sql[:800]}")

        history = state.get("orch_history", []) or []
        if history:
            lines = [
                f"  {i+1}. {h.get('step')} — {h.get('reason', '')} "
                f"({'ok' if h.get('ok', True) else 'fail'})"
                for i, h in enumerate(history[-8:])
            ]
            parts.append("Журнал сделанных шагов:\n" + "\n".join(lines))

        artifacts = []
        if state.get("orch_sources"):
            artifacts.append(f"sources={state.get('orch_sources')}")
        if state.get("orch_metadata"):
            artifacts.append("metadata=собраны")
        if state.get("orch_explain_plan"):
            artifacts.append("explain_plan=получен")
        tcs = state.get("tool_calls") or []
        if any(tc.get("tool") == "execute_query" for tc in tcs):
            artifacts.append("execute_query=выполнен (готово к summarize)")
        if artifacts:
            parts.append("Доступные артефакты: " + ", ".join(artifacts))

        parts.append("Верни JSON со следующим шагом:")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Вспомогательное
    # ------------------------------------------------------------------

    def _filter_catalog_tables(self, candidates: list[str]) -> list[str]:
        """Оставить только реальные таблицы каталога (schema.table)."""
        try:
            df = self.schema.tables_df
            known = {
                f"{str(r['schema_name']).lower()}.{str(r['table_name']).lower()}"
                for _, r in df.iterrows()
            }
        except Exception:  # noqa: BLE001
            return []
        return [c for c in candidates if c.lower() in known]
