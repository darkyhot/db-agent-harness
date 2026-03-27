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
from core.schema_loader import SchemaLoader
from core.sql_validator import SQLValidator, detect_mode, SQLMode
from graph.graph import build_graph, create_initial_state
from tools.db_tools import create_db_tools
from tools.fs_tools import FS_TOOLS
from tools.schema_tools import create_schema_tools

logger = logging.getLogger(__name__)

# Настройка логирования
WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = WORKSPACE_DIR / "agent.log"

# Маппинг узлов графа на пользовательские статусы
_NODE_STATUS = {
    "planner": "Анализирую запрос и составляю план...",
    "table_explorer": "Подгружаю структуру и образцы данных таблиц...",
    "executor": "Выполняю шаг плана...",
    "sql_validator": "Проверяю SQL-запрос...",
    "corrector": "Исправляю ошибку...",
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
  help   — показать этот список
  config — настроить подключение к БД (user, host, port)
  reset  — сбросить контекст текущей сессии
  clear  — очистить вывод ячейки
  exit   — завершить работу (сохранить резюме сессии)

Любой другой ввод обрабатывается как запрос к агенту.
""".strip()


class CLIInterface:
    """CLI интерфейс агента для работы из Jupyter Notebook."""

    def __init__(self) -> None:
        """Инициализация всех компонентов агента."""
        setup_logging()

        self.db = DatabaseManager()
        self.llm = RateLimitedLLM()
        self.memory = MemoryManager()
        self.schema = SchemaLoader()
        self.validator = SQLValidator(self.db, schema_loader=self.schema)

        # Создание tools через DI (замыкания)
        db_tools = create_db_tools(self.db, self.validator, self.schema)
        schema_tools = create_schema_tools(self.schema)
        all_tools = FS_TOOLS + db_tools + schema_tools

        # Флаг отладки промптов из конфига
        self.debug_prompt = self.db._config.get("debug_prompt", False)

        # Сборка графа
        self.graph = build_graph(
            self.llm, self.db, self.schema, self.memory, self.validator, all_tools,
            debug_prompt=self.debug_prompt,
        )

        # Периодическая очистка старых сессий (старше 90 дней)
        cleaned = self.memory.cleanup_old_sessions(keep_days=90)
        if cleaned:
            logger.info("Очищено %d старых сессий при старте", cleaned)

        # Восстановление долгосрочной памяти из предыдущих незавершённых сессий
        self._recover_unsaved_memory()

        # Стартуем сессию
        user_id = self.db._config.get("user_id", "")
        self.memory.start_session(user_id)

        logger.info("CLIInterface инициализирован")

    def _recover_unsaved_memory(self) -> None:
        """Восстановить долгосрочную память из предыдущих сессий без резюме.

        Если агент был закрыт без корректного exit/reset (например, перезапуск ядра),
        резюме и долгосрочная память не были сохранены. Эта функция находит такие сессии
        и извлекает из них факты.
        """
        try:
            unsaved = self.memory.get_unsummarized_sessions()
            if not unsaved:
                return
            logger.info("Найдено %d незавершённых сессий, восстанавливаю память", len(unsaved))
            for session_id in unsaved:
                messages = self.memory.get_session_messages(session_id)
                if not messages:
                    continue
                # Временно переключаемся на старую сессию для сохранения резюме
                old_session = self.memory._session_id
                self.memory._session_id = session_id
                self._save_session_summary()
                self._extract_long_term_memory_from_messages(messages)
                self.memory._session_id = old_session
            logger.info("Восстановление памяти завершено")
        except Exception as e:
            logger.warning("Ошибка восстановления памяти из незавершённых сессий: %s", e)

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

Введите запрос или команду. help — список команд.
"""
        print(banner)

    def _handle_config(self) -> None:
        """Интерактивная настройка подключения к БД."""
        print("\n--- Настройка подключения ---")
        user_id = input("user_id: ").strip()
        if not user_id:
            print("Отменено.")
            return
        host = input("host: ").strip()
        if not host:
            print("Отменено.")
            return
        port_str = input("port [5432]: ").strip()
        port = int(port_str) if port_str else 5432
        database = input("database [prom]: ").strip() or "prom"

        current_debug = self.db._config.get("debug_prompt", False)
        debug_str = input(f"debug_prompt (true/false) [{current_debug}]: ").strip().lower()
        if debug_str in ("true", "1", "yes", "да"):
            debug_prompt = True
        elif debug_str in ("false", "0", "no", "нет"):
            debug_prompt = False
        else:
            debug_prompt = current_debug

        self.db._config["debug_prompt"] = debug_prompt
        self.db.save_config(user_id, host, port, database)
        self.debug_prompt = debug_prompt
        print(f"\n✓ Конфигурация сохранена: {self.db.config_summary}")
        print(f"  debug_prompt: {debug_prompt}")

    def _handle_reset(self) -> None:
        """Сброс контекста текущей сессии."""
        self._save_session_summary()
        self._extract_long_term_memory()
        user_id = self.db._config.get("user_id", "")
        self.memory.start_session(user_id)
        print("✓ Контекст сброшен. Новая сессия начата.")

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

    def _process_query(self, user_input: str) -> None:
        """Обработать запрос пользователя через граф агента.

        Args:
            user_input: Запрос пользователя.
        """
        state = create_initial_state(user_input)
        result = {}
        spinner_idx = 0
        start_time = time.time()

        try:
            for event in self.graph.stream(state):
                node_name = list(event.keys())[0]
                result.update(event[node_name])

                elapsed = time.time() - start_time
                status_text = _NODE_STATUS.get(node_name, node_name)

                # Дополняем статус executor номером шага
                if node_name == "executor":
                    step = result.get("current_step", 0)
                    total = len(result.get("plan", []))
                    if total:
                        status_text = f"Выполняю шаг {step}/{total}..."

                spinner = _SPINNER[spinner_idx % len(_SPINNER)]
                spinner_idx += 1
                _status_print(f"{spinner} {status_text} ({elapsed:.0f}с)")

            # Очищаем статусную строку
            _status_print("")

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
                    sql = result.get("sql_to_validate", "")
                    if sql:
                        mode = detect_mode(sql)
                        if mode == SQLMode.WRITE:
                            res = self.db.execute_write(sql)
                            print(f"\n✓ Выполнено. Затронуто строк: {res}")
                        elif mode == SQLMode.DDL:
                            res = self.db.execute_ddl(sql)
                            print(f"\n✓ {res}")
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

            command = user_input.lower()

            if command == "exit":
                self._handle_exit()
                break
            elif command == "help":
                print(HELP_TEXT)
            elif command == "config":
                self._handle_config()
            elif command == "reset":
                self._handle_reset()
            elif command == "clear":
                clear_output(wait=True)
                self._print_banner()
            else:
                self._process_query(user_input)
