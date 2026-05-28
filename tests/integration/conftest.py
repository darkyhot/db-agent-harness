"""Фикстуры интеграционных тестов.

Принцип: прод-код в core/database.py и core/llm.py остаётся как на проме.
Подменяем только через существующие точки расширения:
  - DatabaseManager(config_path=...) — встроенный параметр;
  - GIGACHAT_API_URL / JPY_API_TOKEN — env-переменные (pytest-dotenv грузит .env.test).

Все интеграционные фикстуры требуют маркер @pytest.mark.integration на тестах,
а тесты сами имеют автоматический pytestmark = pytest.mark.integration через
этот conftest (см. ``pytest_collection_modifyitems``).
"""
from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_CONFIG = Path(__file__).resolve().parent / "test_config.json"


# ---------------------------------------------------------------------------
# Авто-маркер integration на все тесты в этой директории
# ---------------------------------------------------------------------------
def pytest_collection_modifyitems(config, items):
    for item in items:
        if str(item.fspath).startswith(str(Path(__file__).resolve().parent)):
            item.add_marker(pytest.mark.integration)


# ---------------------------------------------------------------------------
# Загрузка .env.test
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _load_env_test():
    """Подгрузить tests/integration/.env.test если присутствует."""
    env_path = Path(__file__).resolve().parent / ".env.test"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        pytest.skip("python-dotenv не установлен (см. requirements-test.txt)")
    load_dotenv(env_path, override=False)


# ---------------------------------------------------------------------------
# Тестовый конфиг (без правок продового config.json)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def test_config_path(tmp_path_factory) -> Path:
    """Путь к тестовому config.json.

    Приоритет: tests/integration/test_config.json → временный из env.
    """
    if DEFAULT_TEST_CONFIG.exists():
        return DEFAULT_TEST_CONFIG
    cfg = {
        "user_id": os.getenv("TEST_PG_USER", "test"),
        "host": os.getenv("TEST_PG_HOST", "localhost"),
        "port": int(os.getenv("TEST_PG_PORT", "55432")),
        "database": os.getenv("TEST_PG_DB", "agent_test"),
        "llm_model": os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max"),
        "debug_prompt": False,
        "show_plan": False,
        "llm_verifier_enabled": False,
    }
    path = tmp_path_factory.mktemp("cfg") / "config.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Проверка доступности Postgres
# ---------------------------------------------------------------------------
def _pg_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def pg_ready(test_config_path) -> dict:
    """Гарантировать что тестовый Postgres поднят. Иначе skip."""
    cfg = json.loads(test_config_path.read_text())
    if not _pg_reachable(cfg["host"], cfg["port"], timeout=2.0):
        pytest.skip(
            f"Тестовый Postgres недоступен на {cfg['host']}:{cfg['port']}. "
            f"Запустите: docker compose -f tests/integration/docker-compose.test.yml up -d"
        )
    return cfg


# ---------------------------------------------------------------------------
# DatabaseManager поверх тестового конфига
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def integration_db(test_config_path, pg_ready):
    """Готовый DatabaseManager, указывающий на тестовый Postgres."""
    from core.database import DatabaseManager
    db = DatabaseManager(config_path=test_config_path)
    if not db.is_configured:
        pytest.skip(f"Тестовый config неполон: {test_config_path}")
    # Прогрев — упасть рано, если конфиг битый.
    db.get_engine()
    return db


# ---------------------------------------------------------------------------
# Заполненная синтетикой БД (один раз на сессию)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def seeded_db(integration_db):
    """Гарантировать что в тестовой БД есть схема и данные.

    Перегенерирует данные только если запрошено через env RESEED=1, иначе
    проверяет существование таблиц и при отсутствии — сидит.
    """
    from tests.integration.synth.ddl_generator import load_metadata
    from tests.integration.synth.data_generator import generate_all
    from tests.integration.synth.loader import drop_and_create_schema, load_all

    tables = load_metadata()
    reseed = os.getenv("RESEED", "").lower() in ("1", "true", "yes")
    need_seed = reseed
    if not need_seed:
        first = next(iter(tables.values()))
        try:
            need_seed = not integration_db.table_exists(first.schema, first.name)
        except Exception:
            need_seed = True

    if need_seed:
        drop_and_create_schema(integration_db, tables)
        rows_per_table = int(os.getenv("SYNTH_ROWS", "200"))
        order, data = generate_all(rows_per_table=rows_per_table)
        load_all(integration_db, order, data)

    return integration_db


# ---------------------------------------------------------------------------
# Реальный LLM: DeepSeek (дёшево) или GigaChat (прод). Выбор через env.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def real_llm():
    """LLM-бэкенд для интеграционных тестов.

    Бэкенд выбирается env-переменной `TEST_LLM_BACKEND`:
      - `deepseek` (default) — DeepSeek API (cheap iteration);
      - `gigachat`           — реальный RateLimitedLLM (final prod-parity run).

    Пропускается, если для выбранного бэкенда нет credentials.
    """
    backend = os.getenv("TEST_LLM_BACKEND", "deepseek").strip().lower()
    if backend in ("gigachat", "giga"):
        if not os.getenv("JPY_API_TOKEN"):
            pytest.skip("TEST_LLM_BACKEND=gigachat, но JPY_API_TOKEN не задан.")
        if not os.getenv("GIGACHAT_API_URL"):
            pytest.skip("TEST_LLM_BACKEND=gigachat, но GIGACHAT_API_URL не задан.")
    elif backend in ("deepseek", "ds"):
        if not os.getenv("DEEPSEEK_API_KEY"):
            pytest.skip("TEST_LLM_BACKEND=deepseek, но DEEPSEEK_API_KEY не задан.")
    else:
        pytest.skip(f"Неизвестный TEST_LLM_BACKEND={backend!r}.")

    from tests.integration.llm_backends import build_llm
    return build_llm()


# ---------------------------------------------------------------------------
# Готовый агент-контекст и оркестрованный граф для e2e
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def agent_context(test_config_path, seeded_db, real_llm, monkeypatch_session):
    """AgentContext с тестовой БД и выбранным LLM-бэкендом.

    Подмена через monkeypatch модульных ссылок (прод-код не меняется):
      - core.database.DatabaseManager → создаётся с тестовым config_path;
      - core.agent_factory.RateLimitedLLM → возвращает выбранный бэкенд
        (DeepSeek / GigaChat) из фикстуры real_llm.

    build_agent_context импортирует оба класса внутри функции, поэтому
    monkeypatch модульных имён работает безопасно.
    """
    from core import database as _db_module
    real_db_cls = _db_module.DatabaseManager

    def _patched_db(*args, **kwargs):
        kwargs.setdefault("config_path", test_config_path)
        return real_db_cls(*args, **kwargs)

    monkeypatch_session.setattr(_db_module, "DatabaseManager", _patched_db)

    # Подмена LLM-фабрики на выбранный бэкенд (real_llm уже собран фикстурой).
    import core.llm as _llm_module
    monkeypatch_session.setattr(_llm_module, "RateLimitedLLM", lambda: real_llm)

    from core.agent_factory import build_agent_context
    # run_enrichment=False — обогащение метаданных ходит в реальную БД и
    # LLM, мы хотим контролируемый запуск только в самих тестах.
    ctx = build_agent_context(run_enrichment=False)
    return ctx


@pytest.fixture(scope="session")
def analytics_graph(agent_context):
    """Аналитический подграф (детерминированный pipeline без оркестратора).

    Использовать в e2e-тестах: оркестратор + LLM-планировщик добавляют
    случайность, тогда как аналитический подграф предсказуем.
    """
    from graph.graph import build_analytics_subgraph
    ctx = agent_context
    return build_analytics_subgraph(
        ctx.llm, ctx.db, ctx.schema, ctx.memory, ctx.validator, ctx.tools,
        debug_prompt=False,
        show_plan=False,
        llm_verifier_enabled=False,
    )


# ---------------------------------------------------------------------------
# Сессионный monkeypatch (pytest даёт только функциональный по умолчанию)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()
