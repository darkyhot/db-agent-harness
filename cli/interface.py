"""CLI интерфейс: приветствие, команды, input loop."""

import logging
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
        self.validator = SQLValidator(self.db)

        # Создание tools через DI (замыкания)
        db_tools = create_db_tools(self.db, self.validator, self.schema)
        schema_tools = create_schema_tools(self.schema)
        all_tools = FS_TOOLS + db_tools + schema_tools

        # Сборка графа
        self.graph = build_graph(
            self.llm, self.db, self.schema, self.memory, self.validator, all_tools
        )

        # Стартуем сессию
        user_id = self.db._config.get("user_id", "")
        self.memory.start_session(user_id)

        logger.info("CLIInterface инициализирован")

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

        self.db.save_config(user_id, host, port, database)
        print(f"\n✓ Конфигурация сохранена: {self.db.config_summary}")

    def _handle_reset(self) -> None:
        """Сброс контекста текущей сессии."""
        self._save_session_summary()
        user_id = self.db._config.get("user_id", "")
        self.memory.start_session(user_id)
        print("✓ Контекст сброшен. Новая сессия начата.")

    def _handle_exit(self) -> None:
        """Завершение работы с сохранением резюме."""
        _status_print("Сохранение резюме сессии...")
        self._save_session_summary()
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
        prompt = (
            "Сделай краткое резюме (2-3 предложения) этой сессии работы с базой данных.\n"
            "Что спрашивал пользователь, какие результаты получены.\n\n"
            f"Диалог:\n{dialog}"
        )
        summary = self.llm.invoke(prompt)
        self.memory.save_session_summary(summary)

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
