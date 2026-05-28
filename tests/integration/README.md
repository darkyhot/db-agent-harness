# Интеграционные тесты

E2E-тесты NL→SQL пайплайна на **реальном Postgres** с синтетикой и
**реальном GigaChat API**. Прод-код (`core/database.py`, `core/llm.py`,
`core/agent_factory.py`) не меняется — подмена идёт только через встроенные
точки расширения: `DatabaseManager(config_path=...)` и env-переменные GigaChat.

## Что внутри

```
tests/integration/
├── conftest.py                  # фикстуры pg_ready, integration_db, seeded_db,
│                                # real_llm, agent_context, analytics_graph
├── docker-compose.test.yml      # postgres:15 на порту 55432
├── test_config.example.json     # шаблон тестового config.json
├── .env.test.example            # шаблон env (GigaChat token + PG creds)
├── synth/                       # генератор синтетики по data_for_agent/*.csv
│   ├── type_mapping.py          # dType (int2, varchar(N)…) → Postgres-тип
│   ├── ddl_generator.py         # CREATE SCHEMA / TABLE / FK
│   ├── data_generator.py        # топосорт + value pool для неявных join-ов
│   ├── loader.py                # bulk INSERT через SQLAlchemy
│   └── seed.py                  # CLI: пересоздать схему и залить данные
├── cases/                       # YAML-кейсы расширенного golden-формата
│   └── e2e_*.yaml
├── test_synth_invariants.py     # не требует GigaChat — проверяет синтетику
└── test_e2e_real_pipeline.py    # требует GigaChat — полный e2e
```

## Первый запуск

### 1. Установить тестовые зависимости

```bash
pip install -r requirements-test.txt
```

### 2. Поднять тестовый Postgres

```bash
docker compose -f tests/integration/docker-compose.test.yml up -d
```

Контейнер `db-agent-test-pg` слушает `localhost:55432`, пользователь `test`,
БД `agent_test`, режим `trust` (без пароля).

Проверить здоровье:
```bash
docker compose -f tests/integration/docker-compose.test.yml ps
```

### 3. Заполнить `.env.test` и `test_config.json`

```bash
cp tests/integration/.env.test.example tests/integration/.env.test
cp tests/integration/test_config.example.json tests/integration/test_config.json
# Отредактировать .env.test — вставить DEEPSEEK_API_KEY и/или JPY_API_TOKEN.
```

> `.env.test` и `test_config.json` **не коммитятся** (см. `.gitignore`).

#### Выбор LLM-бэкенда (`TEST_LLM_BACKEND`)

| Значение | Когда |
| --- | --- |
| `deepseek` (default) | Дешёвая итерация во время разработки тестов |
| `gigachat`           | Финальный прогон на той же модели, что и прод |

Бэкенд переключается одной env-переменной — прод-код в [core/llm.py](core/llm.py)
не меняется. Подмена идёт через monkeypatch модульной ссылки
`core.llm.RateLimitedLLM` на сессионном уровне (см. `tests/integration/conftest.py::agent_context`).

DeepSeek быстрее (нет rate-limit 5с) и дешевле в разы; используется по умолчанию.
GigaChat включается командой:

```bash
TEST_LLM_BACKEND=gigachat pytest tests/integration/ -m integration -v
```

### 4. Сгенерировать и залить синтетику

```bash
python -m tests.integration.synth.seed --drop
```

Вывод вроде:
```
[seed] using config: tests/integration/test_config.json
[seed] target: test@localhost:55432/agent_test
[seed] metadata loaded: 7 tables
[seed] DROP + CREATE schema/tables/FKs...
[seed] generating rows (200/table, seed=42)...
[seed] inserting...
[seed] done: 7 tables, 1400 rows total
```

Опции:
- `--rows 500` — больше строк (или `SYNTH_ROWS=500`)
- `--seed 7` — другой seed RNG (стабильный по умолчанию)
- `--drop` — пересоздать схему с нуля

### 5. Прогнать тесты

```bash
# Только синтетика (без GigaChat) — быстро
pytest tests/integration/test_synth_invariants.py -m integration -v

# Полный e2e (с GigaChat, ~5с/запрос из-за rate-limit)
pytest tests/integration/test_e2e_real_pipeline.py -m integration -v

# Всё интеграционное
pytest -m integration -v
```

## Что НЕ запускается по умолчанию

По умолчанию `pytest` идёт с `-m "not integration"` (см. `pytest.ini`).
Регулярный CI прогон unit/golden тестов остаётся быстрым и не требует
ни Postgres, ни токена GigaChat.

## Как это работает

### Постгрес-конфиг

`DatabaseManager(config_path=tests/integration/test_config.json)` —
встроенный параметр прод-класса. Фикстура `agent_context` патчит модульную
ссылку `core.database.DatabaseManager` так, чтобы каждое инстансирование
внутри `agent_factory.build_agent_context()` подбирало именно тестовый
конфиг. Сам класс не модифицирован.

### LLM-бэкенды

Фикстура `real_llm` собирает LLM по `TEST_LLM_BACKEND`:
  - `deepseek` → [tests/integration/llm_backends.py](llm_backends.py) `DeepseekLLM`
    — OpenAI-совместимый клиент к `api.deepseek.com`, читает
    `DEEPSEEK_API_KEY`, `DEEPSEEK_API_URL`, `DEEPSEEK_MODEL`.
  - `gigachat` → реальный `core.llm.RateLimitedLLM`, читает `GIGACHAT_API_URL`,
    `JPY_API_TOKEN`. Прод-класс не модифицирован.

Оба класса имеют **одинаковый интерфейс** (`.invoke()`, `.invoke_with_system()`),
поэтому узлы графа их не различают. В `agent_context` мы патчим
`core.llm.RateLimitedLLM` → `lambda: real_llm`, и `build_agent_context`
получает выбранный бэкенд.

`.env.test` подгружается через `python-dotenv` в session-scoped фикстуре
`_load_env_test` (autouse). Если для выбранного бэкенда нет токена —
`pytest.skip`. `test_synth_invariants.py` LLM не требует и идёт без токена.

### Синтетика

Метаданные читаются из `data_for_agent/{tables_list,attr_list}.csv`
(уже существующих и используемых прод-кодом). Генератор:

1. Маппит `dType` → Postgres-тип (`type_mapping.py`).
2. Строит DDL с композитными PK и явными FK (`ddl_generator.py`).
3. Сортирует таблицы топологически: словари → независимые → зависимые
   (`data_generator.py`).
4. Поддерживает пул значений по имени «связующих» колонок (`tb_id`, `gosb_id`,
   `old_gosb_id`, `epk_id`, …) — благодаря этому фактовые таблицы получают
   значения, пересекающиеся со словарями, и JOIN-ы в тестах что-то находят.
5. Колонки `partition_key=True` и `report_dt` равномерно раскиданы по
   месяцам в `[2025-01-01, 2026-05-26]`.

### Контракт e2e-кейса

`cases/*.yaml` — расширение существующего golden-формата:

```yaml
query: "Сколько записей в витрине оттока?"
intent_must_be: analytics
sql_must_contain: ["COUNT", "UZP_DWH_FACT_OUTFLOW"]
sql_must_not_contain: ["JOIN"]
should_reach_node: sql_validator
execute_succeeds: true                    # реально выполнить полученный SQL
expected_rowcount: {min: 1, max: 1}
max_retries: 1
```

## Перегенерация / очистка

```bash
# Перегенерировать данные в существующей БД (DROP + INSERT)
python -m tests.integration.synth.seed --drop

# Принудительная пересборка прямо в pytest
RESEED=1 pytest tests/integration/ -m integration -v

# Полностью снести БД
docker compose -f tests/integration/docker-compose.test.yml down -v
```
