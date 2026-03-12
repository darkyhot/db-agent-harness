"""CLI интерфейс: приветствие, команды, input loop."""

import logging
import sys
from pathlib import Path

from IPython.display import clear_output, display, Markdown

from core.database import DatabaseManager
from core.llm import RateLimitedLLM
from core.memory import MemoryManager
from core.schema_loader import SchemaLoader
from core.sql_validator import SQLValidator
from graph.graph import build_graph, create_initial_state
from tools.db_tools import init_db_tools
from tools.schema_tools import init_schema_tools

logger = logging.getLogger(__name__)

# Настройка логирования
WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = WORKSPACE_DIR / "agent.log"


def setup_logging() -> None:
    """Настройка логирования: INFO в консоль, DEBUG в файл."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Консоль — INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    # Файл — DEBUG
    file_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    root.addHandler(console_handler)
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

        # Инициализация tools
        init_db_tools(self.db)
        init_schema_tools(self.schema)

        # Сборка графа
        self.graph = build_graph(
            self.llm, self.db, self.schema, self.memory, self.validator
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

⚠️  Режим доступа: FULL (READ + WRITE + DDL). Будьте внимательны с деструктивными операциями.

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
        # Сохраняем резюме текущей сессии
        self._save_session_summary()
        # Начинаем новую
        user_id = self.db._config.get("user_id", "")
        self.memory.start_session(user_id)
        print("✓ Контекст сброшен. Новая сессия начата.")

    def _handle_exit(self) -> None:
        """Завершение работы с сохранением резюме."""
        print("\nСохранение резюме сессии...")
        self._save_session_summary()
        print("До свидания! 👋")

    def _save_session_summary(self) -> None:
        """Сгенерировать и сохранить резюме текущей сессии."""
        messages = self.memory.get_session_messages()
        if not messages:
            return

        # Формируем контекст для резюме
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
        logger.info("Резюме сессии сохранено")

    def _process_query(self, user_input: str) -> None:
        """Обработать запрос пользователя через граф агента.

        Args:
            user_input: Запрос пользователя.
        """
        state = create_initial_state(user_input)

        try:
            result = self.graph.invoke(state)

            # Проверяем нужно ли подтверждение
            if result.get("needs_confirmation"):
                msg = result.get("confirmation_message", "Требуется подтверждение.")
                print(f"\n⚠️  {msg}")
                confirm = input("Введите YES для подтверждения: ").strip()
                if confirm.upper() == "YES":
                    # Повторяем с подтверждением — выполняем SQL напрямую
                    sql = result.get("sql_to_validate", "")
                    if sql:
                        from core.sql_validator import detect_mode, SQLMode
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
                    self.memory.add_message("assistant", "Запрос подтверждён и выполнен.")
                else:
                    print("Отменено.")
                    self.memory.add_message("assistant", "Запрос отменён пользователем.")
                return

            # Выводим финальный ответ
            answer = result.get("final_answer", "Нет ответа.")
            print(f"\n{answer}")

        except Exception as e:
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
