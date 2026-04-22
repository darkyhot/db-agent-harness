"""CLI интерфейс: приветствие, команды, input loop."""

import json
import logging
import re
import sys
import time
from pathlib import Path

from IPython.display import clear_output, display, Markdown

from core.database import DatabaseManager
from core.llm import RateLimitedLLM
from core.memory import MemoryManager
from core.query_cache import QueryCache
from core.enrichment_pipeline import EnrichmentPipeline
from core.schema_loader import SchemaLoader
from core.sql_validator import SQLValidator, detect_mode, SQLMode
from graph.graph import build_graph, create_initial_state
from tools.db_tools import create_db_tools
from tools.fs_tools import FS_TOOLS
from tools.path_safety import resolve_workspace_path
from tools.schema_tools import create_schema_tools

logger = logging.getLogger(__name__)

# Настройка логирования
WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = WORKSPACE_DIR / "agent.log"

# Маппинг узлов графа на пользовательские статусы
_NODE_STATUS = {
    # Новая архитектура (11 узлов)
    "intent_classifier": "Анализирую запрос...",
    "table_resolver": "Определяю нужные таблицы...",
    "table_explorer": "Подгружаю структуру и образцы данных таблиц...",
    "column_selector": "Выбираю колонки для запроса...",
    "sql_planner": "Планирую стратегию SQL...",
    "plan_preview": "Готовлю план запроса...",
    "explicit_mode_dispatcher": "Определяю режим запроса...",
    "sql_writer": "Пишу SQL-запрос...",
    "sql_validator": "Проверяю и выполняю SQL...",
    "error_diagnoser": "Анализирую ошибку...",
    "sql_fixer": "Исправляю SQL...",
    "tool_dispatcher": "Выполняю поиск...",
    "summarizer": "Формирую ответ...",
}

_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Ширина строки для полной перезаписи (Jupyter не поддерживает ANSI \033[K)
_LINE_WIDTH = 80


def _status_print(msg: str, done: bool = False) -> None:
    """Вывести статус с перезаписью текущей строки."""
    padded = msg.ljust(_LINE_WIDTH)[:_LINE_WIDTH]
    if done:
        sys.stdout.write(f"\r{padded}\n")
    else:
        sys.stdout.write(f"\r{padded}")
    sys.stdout.flush()


def _match_clarification_to_choice(
    clarification: str,
    filter_candidates: dict,
    already_resolved: dict[str, str] | None = None,
) -> dict[str, str]:
    """Сопоставить ответ пользователя с конкретным кандидатом из clarification.

    Ищем первый request_id (не закрытый ранее), у которого в топ-2 кандидатах
    имя колонки или описание совпадает (подстрочно, регистронезависимо) с
    ответом пользователя. Возвращаем {request_id: column_name} или пустой
    словарь, если однозначного соответствия не нашлось.
    """
    normalized = (clarification or "").strip().lower().replace("ё", "е")
    if not normalized:
        return {}
    already_resolved = already_resolved or {}
    for request_id, candidates in (filter_candidates or {}).items():
        if str(request_id) in already_resolved:
            continue
        if not candidates:
            continue
        for cand in candidates[:2]:
            column = str(cand.get("column") or "").strip().lower().replace("ё", "е")
            description = str(cand.get("description") or "").strip().lower().replace("ё", "е")
            if column and (column in normalized or normalized in column):
                return {str(request_id): str(cand.get("column") or "")}
            if description and description in normalized:
                return {str(request_id): str(cand.get("column") or "")}
    return {}


def _is_yes_reply(text: str) -> bool:
    normalized = (text or "").strip().lower().replace("ё", "е")
    return normalized in {"да", "ага", "угу", "yes", "y", "ok", "ок", "верно", "подтверждаю"}


def _is_no_reply(text: str) -> bool:
    normalized = (text or "").strip().lower().replace("ё", "е")
    return normalized in {"нет", "no", "n", "неа", "не", "неверно"}


def _interpret_filter_clarification(
    clarification: str,
    where_resolution: dict,
    *,
    already_resolved: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Понять ответ пользователя на typed clarification по фильтрам."""
    already_resolved = already_resolved or {}
    spec = (where_resolution or {}).get("clarification_spec", {}) or {}
    request_id = str(spec.get("request_id") or "").strip()
    options = list(spec.get("options") or [])

    if request_id and request_id not in already_resolved:
        if spec.get("type") == "confirm" and options:
            candidate_column = str(options[0].get("column") or "").strip()
            if candidate_column:
                if _is_yes_reply(clarification):
                    return ({request_id: candidate_column}, {})
                if _is_no_reply(clarification):
                    return ({}, {request_id: [candidate_column]})

        if spec.get("type") == "choice":
            choice_candidates = {
                request_id: [
                    {
                        "column": opt.get("column"),
                        "description": opt.get("label"),
                    }
                    for opt in options
                ]
            }
            matched = _match_clarification_to_choice(
                clarification,
                choice_candidates,
                already_resolved=already_resolved,
            )
            if matched:
                return (matched, {})

    matched = _match_clarification_to_choice(
        clarification,
        (where_resolution or {}).get("filter_candidates", {}) or {},
        already_resolved=already_resolved,
    )
    return (matched, {})


def setup_logging() -> None:
    """Настройка логирования: только файл, консоль отключена."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Убираем все существующие хендлеры, чтобы не дублировать при повторном вызове
    root.handlers.clear()

    # Только файл — всё подробно (DEBUG). Консоль не нужна — есть статусная строка.
    file_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    root.addHandler(file_handler)


HELP_TEXT = """
Доступные команды:
  /help                — показать этот список
  /config              — показать доступные разделы настройки
  /config connection   — настроить подключение к БД
  /config params       — настроить runtime-параметры агента
  /memory              — просмотр и управление долгосрочной памятью
  /metrics             — метрики качества генерации SQL (последние 30 дней)
  /reset               — сбросить контекст текущей сессии
  /clear               — очистить вывод ячейки
  /exit                — завершить работу (сохранить резюме сессии)

Любой ввод, не начинающийся с `/`, обрабатывается как запрос к агенту.
""".strip()

CONFIG_HELP_TEXT = """
Доступные варианты `/config`:
  /config connection   — изменить подключение к БД: user_id, host, port, database
  /config params       — изменить параметры агента: debug_prompt, show_plan

Когда что использовать:
  /config connection   — если меняется база, хост, порт или пользователь
  /config params       — если меняется режим работы агента без переподключения к БД
""".strip()


class CLIInterface:
    """CLI интерфейс агента для работы из Jupyter Notebook."""

    def __init__(self) -> None:
        """Инициализация всех компонентов агента."""
        setup_logging()

        self.db = DatabaseManager()
        self._ensure_startup_config()
        self.llm = RateLimitedLLM()
        self.memory = MemoryManager()
        self.schema = SchemaLoader()
        EnrichmentPipeline(self.schema, llm=self.llm, db_manager=self.db).run()
        self.validator = SQLValidator(self.db, schema_loader=self.schema)

        # Создание tools через DI (замыкания)
        db_tools = create_db_tools(self.db, self.validator, self.schema)
        schema_tools = create_schema_tools(self.schema)
        all_tools = FS_TOOLS + db_tools + schema_tools

        # Кэш повторных запросов
        self.query_cache = QueryCache(self.memory)
        # Multi-turn контекст: последний успешный SQL и краткое резюме
        self._prev_sql: str = ""
        self._prev_result_summary: str = ""

        # Флаги из конфига
        self.debug_prompt = self.db.runtime_config.get("debug_prompt", False)
        self.show_plan = self.db.runtime_config.get("show_plan", False)

        # Сборка графа
        self.graph = build_graph(
            self.llm, self.db, self.schema, self.memory, self.validator, all_tools,
            debug_prompt=self.debug_prompt,
            show_plan=self.show_plan,
        )

        # Периодическая очистка старых сессий (старше 90 дней)
        cleaned = self.memory.cleanup_old_sessions(keep_days=90)
        if cleaned:
            logger.info("Очищено %d старых сессий при старте", cleaned)

        # Восстановление долгосрочной памяти из предыдущих незавершённых сессий
        self._recover_unsaved_memory()

        # Стартуем сессию
        user_id = self.db.runtime_config.get("user_id", "")
        self.memory.start_session(user_id)

        logger.info("CLIInterface инициализирован")

    def _recover_unsaved_memory(self) -> None:
        """Восстановить долгосрочную память из предыдущих сессий без резюме.

        Если агент был закрыт без корректного exit/reset (например, перезапуск ядра),
        резюме и долгосрочная память не были сохранены. Эта функция находит такие сессии
        и извлекает из них факты.
        """
        unsaved = self.memory.get_unsummarized_sessions()
        if not unsaved:
            return
        logger.info("Найдено %d незавершённых сессий, восстанавливаю память", len(unsaved))
        for session_id in unsaved:
            old_session = self.memory._session_id
            try:
                messages = self.memory.get_session_messages(session_id)
                if not messages:
                    continue
                # Временно переключаемся на старую сессию для сохранения резюме
                self.memory._session_id = session_id
                self._save_session_summary()
                self._extract_long_term_memory_from_messages(messages)
                self.memory._session_id = old_session
            except Exception as e:
                logger.warning("Ошибка восстановления сессии %s: %s", session_id, e)
                self.memory._session_id = old_session
        logger.info("Восстановление памяти завершено")

    def _build_memory_extraction_prompt(self, dialog: str) -> tuple[str, str]:
        """Единый промпт для извлечения долгосрочной памяти.

        Returns:
            Кортеж (system_prompt, user_prompt) для invoke_with_system.
        """
        existing_facts = self.memory.get_memory("user_facts") or "[]"
        existing_patterns = self.memory.get_memory("behavior_patterns") or "[]"
        existing_instructions = self.memory.get_memory("user_instructions") or "[]"

        system_prompt = (
            "Ты — модуль извлечения памяти аналитического агента.\n"
            "Анализируешь диалог и извлекаешь информацию о пользователе.\n\n"
            "Верни СТРОГО JSON без пояснений:\n"
            "{\n"
            '  "user_facts": ["факт1", "факт2"],\n'
            '  "behavior_patterns": ["паттерн1"],\n'
            '  "user_instructions": ["инструкция1"]\n'
            "}\n\n"
            "Категории:\n"
            "- user_facts: имя, стек, предпочтения, роль, контекст работы\n"
            "- behavior_patterns: стиль взаимодействия (краткость, формат ответов)\n"
            "- user_instructions: явные указания агенту (всегда LEFT JOIN, сохранять в /reports/)\n\n"
            "Правила:\n"
            "- Каждый элемент — одно короткое предложение\n"
            "- Объедини с существующими, убери дубли\n"
            "- Если нет новой информации — верни существующий список\n"
            "- Не более 20 элементов в категории"
        )

        user_prompt = (
            f"Существующие данные:\n"
            f"  Факты: {existing_facts}\n"
            f"  Паттерны: {existing_patterns}\n"
            f"  Инструкции: {existing_instructions}\n\n"
            f"Диалог:\n{dialog}"
        )

        return system_prompt, user_prompt

    def _save_memory_from_llm_response(self, response: str) -> None:
        """Парсинг и сохранение памяти из ответа LLM."""
        try:
            # Убираем markdown-обёртки
            cleaned = re.sub(r'```(?:json)?\s*\n?', '', response)
            cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE)
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if not match:
                logger.warning("Не удалось найти JSON в ответе извлечения памяти")
                return
            data = json.loads(match.group())
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Ошибка парсинга JSON извлечения памяти: %s", e)
            return

        for key in ("user_facts", "behavior_patterns", "user_instructions"):
            value = data.get(key)
            if isinstance(value, list):
                self.memory.set_memory(key, json.dumps(value, ensure_ascii=False))

    def _extract_long_term_memory_from_messages(self, messages: list[dict]) -> None:
        """Извлечь долгосрочную память из произвольного списка сообщений."""
        if not messages:
            return

        dialog = "\n".join(
            f"{m['role']}: {m['content'][:200]}" for m in messages[-20:]
        )

        system_prompt, user_prompt = self._build_memory_extraction_prompt(dialog)

        try:
            response = self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.1)
        except Exception as e:
            logger.warning("Ошибка LLM при извлечении долгосрочной памяти: %s", e)
            return

        self._save_memory_from_llm_response(response)

    def _print_banner(self) -> None:
        """Показать приветственное сообщение."""
        config_str = self.db.config_summary
        tables_count = self.schema.tables_count
        attrs_count = self.schema.attrs_count
        sessions = self.memory.session_count

        banner = f"""
╔══════════════════════════════════════════╗
║       Database Agent Harness v1.0        ║
║  Аналитический агент для работы с БД     ║
╚══════════════════════════════════════════╝

Текущая конфигурация: {config_str}
Загружено таблиц: {tables_count} | Загружено атрибутов: {attrs_count}
Память: загружено {sessions} предыдущих сессий
Отладка промптов: {"ВКЛ" if self.debug_prompt else "ВЫКЛ"} (debug_prompt в config.json)
Показ плана: {"ВКЛ" if self.show_plan else "ВЫКЛ"} (show_plan в config.json)

Введите запрос или slash-команду. /help — список команд.
"""
        print(banner)

    @staticmethod
    def _parse_command(user_input: str) -> tuple[str | None, list[str]]:
        """Разобрать slash-команду или legacy-алиас."""
        tokens = user_input.strip().split()
        if not tokens:
            return (None, [])

        command = tokens[0].lower()
        if command.startswith("/"):
            command = command[1:]
        elif command not in {"help", "config", "memory", "metrics", "reset", "clear", "exit"}:
            return (None, [])

        return (command, [token.lower() for token in tokens[1:]])

    def _refresh_runtime_flags(self) -> None:
        """Подтянуть runtime-флаги из config.json в CLI."""
        runtime = self.db.runtime_config
        self.debug_prompt = runtime.get("debug_prompt", False)
        self.show_plan = runtime.get("show_plan", False)

    def _rebuild_graph(self) -> None:
        """Пересобрать graph и tool bindings после смены конфига."""
        self._refresh_runtime_flags()
        db_tools = create_db_tools(self.db, self.validator, self.schema)
        schema_tools = create_schema_tools(self.schema)
        all_tools = FS_TOOLS + db_tools + schema_tools
        self.graph = build_graph(
            self.llm, self.db, self.schema, self.memory, self.validator, all_tools,
            debug_prompt=self.debug_prompt,
            show_plan=self.show_plan,
        )

    def _prompt_text(
        self,
        label: str,
        *,
        current: str | None = None,
        default: str | None = None,
        required: bool = False,
    ) -> str:
        """Запросить строковое значение с поддержкой текущего значения/дефолта."""
        while True:
            prompt_default = current if current not in (None, "") else default
            if prompt_default not in (None, ""):
                raw = input(f"{label} [{prompt_default}]: ").strip()
            else:
                raw = input(f"{label}: ").strip()

            if raw:
                return raw
            if current not in (None, ""):
                return current
            if default not in (None, ""):
                return default
            if not required:
                return ""
            print("Это поле обязательно. Нужна непустая строка.")

    def _prompt_int(
        self,
        label: str,
        *,
        current: int | None = None,
        default: int | None = None,
        required: bool = False,
    ) -> int:
        """Запросить целое число."""
        while True:
            prompt_default = current if current is not None else default
            if prompt_default is not None:
                raw = input(f"{label} [{prompt_default}]: ").strip()
            else:
                raw = input(f"{label}: ").strip()

            if not raw:
                if current is not None:
                    return current
                if default is not None:
                    return default
                if not required:
                    return 0
                print("Это поле обязательно. Нужен целый номер.")
                continue

            try:
                return int(raw)
            except ValueError:
                print("Нужно ввести целое число, например 5432.")

    def _prompt_bool(self, label: str, *, current: bool = False) -> bool:
        """Запросить булево значение в формате true/false."""
        while True:
            raw = input(f"{label} (true/false) [{current}]: ").strip().lower()
            if not raw:
                return current
            if raw in ("true", "1", "yes", "y", "да"):
                return True
            if raw in ("false", "0", "no", "n", "нет"):
                return False
            print("Введите `true` или `false`.")

    def _print_config_help(self) -> None:
        """Показать подсказку по разделам /config."""
        print(f"\n{CONFIG_HELP_TEXT}\n")

    def _configure_connection(self) -> None:
        """Интерактивно настроить только подключение к БД."""
        runtime = self.db.runtime_config
        print("\n--- Настройка подключения к БД ---")
        user_id = self._prompt_text(
            "user_id",
            current=runtime.get("user_id") or None,
            required=True,
        )
        host = self._prompt_text(
            "host",
            current=runtime.get("host") or None,
            required=True,
        )
        port = self._prompt_int(
            "port",
            current=runtime.get("port"),
            default=5432,
            required=True,
        )
        database = self._prompt_text(
            "database",
            current=runtime.get("database") or None,
            default="prom",
            required=True,
        )
        self.db.save_connection_config(user_id, host, port, database)
        print(f"\n✓ Подключение сохранено: {self.db.config_summary}")

    def _configure_params(self) -> None:
        """Интерактивно настроить только runtime-параметры агента."""
        runtime = self.db.runtime_config
        print("\n--- Настройка параметров агента ---")
        print("Подсказка: `debug_prompt` включает печать внутренних промптов, "
              "`show_plan` показывает план запроса перед выполнением.")
        debug_prompt = self._prompt_bool(
            "debug_prompt",
            current=runtime.get("debug_prompt", False),
        )
        show_plan = self._prompt_bool(
            "show_plan",
            current=runtime.get("show_plan", False),
        )
        self.db.save_runtime_params(debug_prompt=debug_prompt, show_plan=show_plan)
        print("\n✓ Параметры сохранены:")
        print(f"  debug_prompt: {debug_prompt}")
        print(f"  show_plan: {show_plan}")

    def _run_full_config_setup(self) -> None:
        """Полная первичная настройка config: connection + params."""
        print("\nЗапускаю полную настройку конфигурации.")
        self._configure_connection()
        self._configure_params()

    def _ensure_startup_config(self) -> None:
        """При старте проверить config.json и, если нужно, запустить onboarding."""
        missing_connection = self.db.missing_connection_fields()
        missing_runtime = self.db.missing_runtime_fields()
        if not missing_connection and not missing_runtime:
            return

        print("\nКонфигурация не найдена или заполнена не полностью.")
        if missing_connection:
            print(f"Не хватает параметров подключения: {', '.join(missing_connection)}")
        if missing_runtime:
            print(f"Не хватает параметров агента: {', '.join(missing_runtime)}")
        self._run_full_config_setup()

    def _handle_config(self, args: list[str] | None = None) -> None:
        """Обработать `/config` и его подкоманды."""
        args = args or []
        if not args:
            self._print_config_help()
            return

        section = args[0]
        if section == "connection":
            self._configure_connection()
        elif section == "params":
            self._configure_params()
        else:
            print(f"\nНеизвестная подкоманда `/config {' '.join(args)}`.")
            self._print_config_help()
            return

        if hasattr(self, "llm") and hasattr(self, "validator") and hasattr(self, "schema"):
            self._rebuild_graph()

    def _handle_reset(self) -> None:
        """Сброс контекста текущей сессии."""
        self._save_session_summary()
        self._extract_long_term_memory()
        user_id = self.db.runtime_config.get("user_id", "")
        self.memory.start_session(user_id)
        print("✓ Контекст сброшен. Новая сессия начата.")

    def _handle_memory(self) -> None:
        """Просмотр и управление долгосрочной памятью."""
        all_memory = self.memory.get_all_memory()
        layer_keys = {"user_facts", "behavior_patterns", "user_instructions", "correction_examples"}

        if not all_memory:
            print("Долгосрочная память пуста.")
            return

        print("\n=== Долгосрочная память ===\n")
        for key in sorted(all_memory.keys()):
            value = all_memory[key]
            # Пытаемся распарсить JSON-списки для красивого вывода
            try:
                items = json.loads(value)
                if isinstance(items, list):
                    print(f"[{key}]")
                    for i, item in enumerate(items, 1):
                        print(f"  {i}. {item}")
                    print()
                    continue
            except (json.JSONDecodeError, TypeError):
                pass
            print(f"[{key}]: {value}\n")

        # Предлагаем удаление
        print("Для удаления записи введите: delete <ключ>")
        print("Для выхода нажмите Enter.")
        try:
            choice = input(">>> ").strip()
        except EOFError:
            return

        if choice.startswith("delete "):
            key_to_delete = choice[7:].strip()
            if key_to_delete in all_memory:
                self.memory.delete_memory(key_to_delete)
                print(f"✓ Ключ '{key_to_delete}' удалён из долгосрочной памяти.")
            else:
                print(f"✗ Ключ '{key_to_delete}' не найден.")

    def _handle_metrics(self) -> None:
        """Показать метрики качества генерации SQL."""
        metrics = self.memory.get_sql_quality_metrics(days=30)
        total = metrics.get("total_queries", 0)

        if total == 0:
            print("\nНет данных за последние 30 дней.\nВыполните несколько запросов для накопления статистики.")
            return

        success_rate = metrics.get("success_rate", 0)
        first_try = metrics.get("first_try_success_rate", 0)
        avg_retry = metrics.get("avg_retries", 0)
        max_retry = metrics.get("max_retries", 0)
        avg_ms = metrics.get("avg_duration_ms", 0)
        errors = metrics.get("error_distribution", {})
        statuses = metrics.get("status_distribution", {})

        print(f"""
╔══════════════════════════════════════════╗
║       Метрики качества SQL (30 дней)     ║
╚══════════════════════════════════════════╝

  Всего запросов:       {total}
  Успешность:           {success_rate:.1f}%
  С первой попытки:     {first_try:.1f}%
  Среднее retry:        {avg_retry:.2f}  (макс: {max_retry})
  Среднее время:        {avg_ms:.0f} мс

  Распределение статусов:""")
        for status, cnt in sorted(statuses.items(), key=lambda x: -x[1]):
            bar = "█" * min(int(cnt / max(statuses.values()) * 20), 20)
            print(f"    {status:<15} {cnt:>4}  {bar}")

        if errors:
            print("\n  Топ ошибок:")
            for err_type, cnt in sorted(errors.items(), key=lambda x: -x[1])[:5]:
                print(f"    {err_type:<25} {cnt}")
        print()

    def _ask_feedback(self, user_input: str, sql: str) -> None:
        """Запросить у пользователя оценку ответа (y/n/skip).

        При отрицательной оценке предлагает ввести правильный SQL.
        Результат записывается в memory/feedback.jsonl.
        """
        try:
            verdict_raw = input("\nКак ответ? [y=хороший, n=плохой, skip=пропустить]: ").strip().lower()
        except EOFError:
            return

        if verdict_raw in ("skip", "s", ""):
            return

        if verdict_raw in ("y", "yes", "д", "да", "1", "+"):
            self.memory.log_user_feedback(user_input, sql, verdict="up")
            return

        if verdict_raw in ("n", "no", "н", "нет", "0", "-"):
            try:
                corrected = input(
                    "Введите правильный SQL (или оставьте пустым): "
                ).strip()
            except EOFError:
                corrected = ""
            self.memory.log_user_feedback(
                user_input, sql, verdict="down",
                corrected_sql=corrected or None,
            )
            print("✓ Отзыв записан. Спасибо — это улучшает качество ответов.")
            return

    def _handle_exit(self) -> None:
        """Завершение работы с сохранением резюме."""
        _status_print("Сохранение резюме сессии...")
        self._save_session_summary()
        self._extract_long_term_memory()
        self.memory.close()
        _status_print("До свидания!", done=True)

    def _save_session_summary(self) -> None:
        """Сгенерировать и сохранить резюме текущей сессии."""
        messages = self.memory.get_session_messages()
        if not messages:
            return

        dialog = "\n".join(
            f"{m['role']}: {m['content'][:200]}" for m in messages[-20:]
        )
        summary = self.llm.invoke_with_system(
            "Ты — модуль резюмирования. Верни 2-3 предложения: что спрашивал пользователь и какие результаты.",
            f"Диалог:\n{dialog}",
            temperature=0.3,
        )
        self.memory.save_session_summary(summary)

    def _extract_long_term_memory(self) -> None:
        """Извлечь слои долгосрочной памяти из текущей сессии."""
        messages = self.memory.get_session_messages()
        if not messages:
            return

        dialog = "\n".join(
            f"{m['role']}: {m['content'][:200]}" for m in messages[-20:]
        )

        system_prompt, user_prompt = self._build_memory_extraction_prompt(dialog)

        try:
            response = self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.1)
        except Exception as e:
            logger.warning("Ошибка LLM при извлечении долгосрочной памяти: %s", e)
            return

        self._save_memory_from_llm_response(response)
        logger.info("Долгосрочная память (слои) обновлена")

    # Константы для plan-preview подтверждения
    _PLAN_APPROVE_TOKENS = frozenset({
        "ок", "ok", "да", "yes", "подтверждаю", "ага", "ладно", "давай",
        "подтвердить", "выполнить", "запустить",
    })
    _PLAN_MAX_ITERATIONS = 3

    def _process_query(
        self,
        user_input: str,
        user_filter_choices: dict[str, str] | None = None,
        plan_preview_approved: bool = False,
        _plan_iteration: int = 0,
        rejected_filter_choices: dict[str, list[str]] | None = None,
    ) -> None:
        """Обработать запрос пользователя через граф агента.

        Args:
            user_input: Запрос пользователя.
            user_filter_choices: Явные выборы колонок от пользователя, собранные
                при предыдущих уточнениях. Передаются в state нетронутыми —
                где_resolver использует их, чтобы не задавать тот же вопрос снова.
        """
        # --- Проверка кэша (только для read-запросов без явных write-ключевых слов) ---
        _write_keywords = ("insert", "update", "delete", "drop", "create", "truncate", "alter")
        _is_write = any(kw in user_input.lower() for kw in _write_keywords)
        # Не отдаём из кэша, если пользователь уточнил фильтр — ответ может поменяться.
        if not _is_write and not user_filter_choices:
            cached = self.query_cache.get(user_input)
            if cached:
                from datetime import datetime, timezone
                try:
                    ts = datetime.fromisoformat(cached["created_at"])
                    age_min = int((datetime.now(timezone.utc) - ts).total_seconds() / 60)
                    age_str = f"{age_min} мин. назад" if age_min < 60 else f"{age_min // 60} ч. назад"
                except Exception:
                    age_str = "ранее"
                print(f"\n💾 Найден кэшированный ответ ({age_str}). Используется без запроса к БД.\n")
                print(cached["final_answer"])
                return

        state = create_initial_state(
            user_input,
            prev_sql=self._prev_sql,
            prev_result_summary=self._prev_result_summary,
            user_filter_choices=dict(user_filter_choices or {}),
            plan_preview_approved=plan_preview_approved,
            rejected_filter_choices={k: list(v) for k, v in dict(rejected_filter_choices or {}).items()},
        )
        result = {}
        spinner_idx = 0
        start_time = time.time()

        try:
            for event in self.graph.stream(state):
                node_name = list(event.keys())[0]
                node_payload = event.get(node_name)
                if isinstance(node_payload, dict):
                    result.update(node_payload)

                elapsed = time.time() - start_time
                status_text = _NODE_STATUS.get(node_name, node_name)

                # Дополняем статус sql_writer номером шага
                if node_name in ("sql_writer", "executor"):
                    step = result.get("current_step", 0)
                    total = len(result.get("plan", []))
                    if total:
                        status_text = f"Выполняю шаг {step}/{total}..."

                spinner = _SPINNER[spinner_idx % len(_SPINNER)]
                spinner_idx += 1
                _status_print(f"{spinner} {status_text} ({elapsed:.0f}с)")

            # Очищаем статусную строку
            _status_print("")

            # Plan-preview: показываем план и ждём подтверждения пользователя
            if result.get("plan_preview_pending"):
                plan_msg = result.get("confirmation_message", "")
                if plan_msg:
                    print(f"\n{plan_msg}\n")
                else:
                    print("\n[Plan preview: план недоступен]\n")

                if _plan_iteration >= self._PLAN_MAX_ITERATIONS:
                    print(
                        f"⚠️  Достигнут лимит правок плана ({self._PLAN_MAX_ITERATIONS}). "
                        "Продолжаем с текущим планом."
                    )
                    self._process_query(
                        user_input,
                        user_filter_choices=user_filter_choices,
                        plan_preview_approved=True,
                        _plan_iteration=_plan_iteration,
                        rejected_filter_choices=rejected_filter_choices,
                    )
                    return

                try:
                    plan_input = input("Ваш ответ (ок/правка): ").strip()
                except EOFError:
                    plan_input = "ok"

                normalized = plan_input.lower().strip()
                if normalized in self._PLAN_APPROVE_TOKENS or not normalized:
                    # Подтверждено — перезапускаем с флагом approved
                    self.memory.add_message(
                        "user", f"[plan_preview] подтверждено: {plan_input}"
                    )
                    self._process_query(
                        user_input,
                        user_filter_choices=user_filter_choices,
                        plan_preview_approved=True,
                        _plan_iteration=_plan_iteration,
                        rejected_filter_choices=rejected_filter_choices,
                    )
                else:
                    # Пользователь хочет изменить план — мёржим новые хинты в запрос
                    self.memory.add_message(
                        "user", f"[plan_preview] правка #{_plan_iteration + 1}: {plan_input}"
                    )
                    # Добавляем уточнение к запросу, чтобы hint_extractor перечитал хинты
                    augmented = f"{user_input}\nУточнение пользователя: {plan_input}"
                    self._process_query(
                        augmented,
                        user_filter_choices=user_filter_choices,
                        plan_preview_approved=False,
                        _plan_iteration=_plan_iteration + 1,
                        rejected_filter_choices=rejected_filter_choices,
                    )
                return

            # Проверяем нужна ли disambiguation (несколько таблиц)
            if result.get("needs_clarification"):
                msg = result.get(
                    "clarification_message",
                    "Не хватает деталей для выполнения запроса.",
                )
                print(f"\n{msg}")
                clarification = input("\nУточнение: ").strip()
                if clarification:
                    self.memory.add_message("user", f"Уточнение пользователя: {clarification}")
                    where_resolution = result.get("where_resolution", {}) or {}
                    filter_candidates = where_resolution.get("filter_candidates", {}) or {}
                    prev_choices = dict(
                        (user_filter_choices or {})
                        | (where_resolution.get("user_filter_choices", {}) or {})
                    )
                    prev_rejections = dict(
                        (rejected_filter_choices or {})
                        | (where_resolution.get("rejected_filter_choices", {}) or {})
                    )
                    new_choices, new_rejections = _interpret_filter_clarification(
                        clarification,
                        where_resolution,
                        already_resolved=prev_choices,
                    )
                    if new_choices:
                        merged_choices = {**prev_choices, **new_choices}
                        self._process_query(
                            user_input,
                            user_filter_choices=merged_choices,
                            rejected_filter_choices=prev_rejections or None,
                        )
                    elif new_rejections:
                        merged_rejections = dict(prev_rejections)
                        for request_id, columns in new_rejections.items():
                            merged_rejections[request_id] = list(
                                dict.fromkeys((merged_rejections.get(request_id, []) or []) + list(columns))
                            )
                        self._process_query(
                            user_input,
                            user_filter_choices=prev_choices or None,
                            rejected_filter_choices=merged_rejections,
                        )
                    else:
                        # Фолбэк: не смогли сопоставить ответ с кандидатами — как раньше,
                        # дописываем в сам текст запроса, чтобы explicit-choice в
                        # where_resolver мог поймать имя колонки напрямую.
                        augmented_input = f"{user_input}\nУточнение пользователя: {clarification}"
                        self._process_query(
                            augmented_input,
                            user_filter_choices=prev_choices or None,
                            rejected_filter_choices=prev_rejections or None,
                        )
                else:
                    print("Уточнение не получено. Операция отменена.")
                return

            # Проверяем нужна ли disambiguation (несколько таблиц)
            if result.get("needs_disambiguation"):
                msg = result.get("confirmation_message", "Найдено несколько таблиц.")
                print(f"\n{msg}")
                options = result.get("disambiguation_options", [])
                choice = input(f"\nВведите номер витрины (1-{len(options)}): ").strip()
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(options):
                        chosen = options[idx]
                        chosen_table = f"{chosen['schema']}.{chosen['table']}"
                        print(f"\n✓ Выбрана витрина: {chosen_table}")
                        self.memory.add_message("user", f"Выбрана витрина: {chosen_table}")
                        augmented_input = f"{user_input} (использовать таблицу {chosen_table})"
                        self._process_query(augmented_input)
                    else:
                        print("Некорректный номер. Операция отменена.")
                except ValueError:
                    print("Некорректный ввод. Операция отменена.")
                return

            # Проверяем нужно ли подтверждение
            if result.get("needs_confirmation"):
                msg = result.get("confirmation_message", "Требуется подтверждение.")
                print(f"\n⚠️  {msg}")
                confirm = input("Введите YES для подтверждения: ").strip()
                if confirm.upper() == "YES":
                    pending_call = result.get("pending_sql_tool_call") or {}
                    tool_name = pending_call.get("tool")
                    tool_args = pending_call.get("args", {})
                    sql = result.get("sql_to_validate", "")
                    if sql:
                        mode = detect_mode(sql)
                        if tool_name == "execute_write" or mode == SQLMode.WRITE:
                            res = self.db.execute_write(sql)
                            print(f"\n✓ Выполнено. Затронуто строк: {res}")
                        elif tool_name == "execute_ddl" or mode == SQLMode.DDL:
                            res = self.db.execute_ddl(sql)
                            print(f"\n✓ {res}")
                        elif tool_name == "export_query":
                            filename = tool_args.get("filename", "export.csv")
                            output_format = tool_args.get("output_format", "csv")
                            file_path, row_count = self.db.export_query_to_file(
                                sql=sql,
                                filename=filename,
                                output_format=output_format,
                                workspace_dir=WORKSPACE_DIR,
                            )
                            safe_rel_name = resolve_workspace_path(WORKSPACE_DIR, file_path).relative_to(WORKSPACE_DIR)
                            print(f"\n✓ Сохранено в {safe_rel_name} ({row_count} строк)")
                        else:
                            df = self.db.execute_query(sql)
                            print(f"\n{df.to_markdown(index=False)}")
                    else:
                        print("SQL не найден в состоянии. Операция не выполнена.")
                    self.memory.add_message("assistant", "Запрос подтверждён и выполнен.")
                else:
                    print("Отменено.")
                    self.memory.add_message("assistant", "Запрос отменён пользователем.")
                return

            # Выводим финальный ответ
            answer = result.get("final_answer", "Нет ответа.")
            print(f"\n{answer}")

            # Сохраняем успешные read-запросы в кэш + multi-turn контекст
            executed_sql: str | None = None
            if not _is_write and answer and answer != "Нет ответа.":
                for tc in reversed(result.get("tool_calls", [])):
                    if tc.get("tool") == "execute_query":
                        executed_sql = tc.get("args", {}).get("sql")
                        break
                self.query_cache.put(user_input, answer, sql=executed_sql)
                # Обновляем multi-turn контекст для следующего запроса
                if executed_sql:
                    self._prev_sql = executed_sql
                    summary = re.sub(r'```[^`]*```', '', answer, flags=re.DOTALL).strip()
                    self._prev_result_summary = summary[:200]

            # Feedback loop: запрашиваем оценку ответа
            if answer and answer != "Нет ответа.":
                self._ask_feedback(user_input, executed_sql or "")

        except Exception as e:
            _status_print("")
            logger.error("Ошибка обработки запроса: %s", e, exc_info=True)
            print(f"\n❌ Ошибка: {e}")

    def run(self) -> None:
        """Основной цикл обработки ввода пользователя."""
        self._print_banner()

        while True:
            try:
                user_input = input("\n🟢 > ").strip()
            except (EOFError, KeyboardInterrupt):
                self._handle_exit()
                break

            if not user_input:
                continue

            command, args = self._parse_command(user_input)

            if command == "exit":
                self._handle_exit()
                break
            elif command == "help":
                print(HELP_TEXT)
            elif command == "config":
                self._handle_config(args)
            elif command == "memory":
                self._handle_memory()
            elif command == "metrics":
                self._handle_metrics()
            elif command == "reset":
                self._handle_reset()
            elif command == "clear":
                clear_output(wait=True)
                self._print_banner()
            elif user_input.startswith("/"):
                print(f"Неизвестная команда: {user_input}")
                print("Используйте /help для списка команд.")
            else:
                self._process_query(user_input)
