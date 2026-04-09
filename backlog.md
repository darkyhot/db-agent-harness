# Backlog агента-разработчика: DB Agent Harness

Результат жёсткого ревью архитектуры. Пункты отсортированы по impact на надёжность.
Текущий уровень надёжности: ~65–70%. Цель: 99%.

---

## РЕАЛИЗОВАНО В ЭТОЙ ВЕТКЕ

- [x] `core/sql_static_checker.py` — детерминированная проверка SQL до БД
- [x] TF-IDF поиск в `core/schema_loader.py`
- [x] Ограничение роста `messages` и `tool_calls` в `graph/nodes/common.py`
- [x] `STATEMENT_TIMEOUT_MS` снижен до 90s в `core/database.py`
- [x] Команда `metrics` в `cli/interface.py`
- [x] Few-shot retrieval из audit log в `core/few_shot_retriever.py`
- [x] Query cache в `core/query_cache.py`

---

## ПРИОРИТЕТ 1 — КРИТИЧНО ДЛЯ НАДЁЖНОСТИ

### P1-1. sql_static_checker: детерминированная валидация SQL до отправки в БД
**Файлы:** `core/sql_static_checker.py` (новый), `graph/nodes/sql_pipeline.py`, `graph/graph.py`

**Проблема:** `sql_writer` генерирует SQL через LLM после `column_selector`. Но LLM может:
- Добавить колонку, которой нет в `selected_columns` (галлюцинация)
- Написать кириллический алиас (`AS выручка`) несмотря на запрет
- Сделать неполный `GROUP BY`

Первый барьер — EXPLAIN в `sql_validator_node` — не ловит несуществующие колонки до выполнения.
Второй барьер — `column_selector` валидирует колонки, но `sql_writer` их не видит.

**Решение:** Новая нода `sql_static_checker` между `sql_writer` и `sql_validator`.
Без LLM, только Python + sqlparse + schema_loader:
1. Парсинг SQL → извлечение `schema.table` и `alias.column`
2. Для каждой таблицы: проверка найденных колонок против `schema_loader.get_table_columns()`
3. Проверка: нет ли кириллицы в алиасах (`AS [а-яА-Я]`)
4. Проверка: нет ли `SELECT *` (запрещено по правилам)
5. Если нарушения → сразу `error_diagnoser` с типом `hallucinated_column`

**Ожидаемый impact: +10–15% надёжности**

---

### P1-2. TF-IDF поиск в SchemaLoader
**Файл:** `core/schema_loader.py`

**Проблема:** `search_tables()` и `search_by_description()` — `str.contains()` с synonym expansion.
При каталоге 500+ таблиц с техническими именами и описаниями на английском поиск по
русским запросам работает только если synonym_map покрывает термин.

**Решение:** TF-IDF индекс по объединённому тексту `table_name + description + column_names`.
- `scikit-learn.TfidfVectorizer` — строится при инициализации SchemaLoader (один раз)
- `cosine_similarity` с query-вектором → список таблиц с весами
- Merge: `max(tfidf_score, keyword_score)` → единый ранжированный список
- Зависимость: `scikit-learn>=1.4` добавить в `requirements.txt`

**Ожидаемый impact: +8–12% на запросах без явного имени таблицы**

---

### P1-3. Few-shot из audit log в sql_writer
**Файл:** `core/few_shot_retriever.py` (новый), `graph/nodes/sql_pipeline.py`

**Проблема:** `sql_audit` хранит историю успешных запросов, но они нигде не используются.
sql_writer получает только статичные few-shot примеры из `_STRATEGY_EXAMPLES`.

**Решение:**
- `FewShotRetriever.get_similar(user_input, strategy, n=2)` — по TF-IDF или keyword overlap с `user_input` из аудита
- Выбирает `n` записей с `status='success'` и `retry_count=0` для той же стратегии
- В `sql_writer`: если найдены похожие — добавить в промпт блок `=== УСПЕШНЫЕ ПРИМЕРЫ ИЗ ИСТОРИИ ===`
- Кэшировать список успешных запросов в памяти (обновлять при старте сессии)

**Ожидаемый impact: +3–8% после накопления 30+ успешных запросов**

---

## ПРИОРИТЕТ 2 — АРХИТЕКТУРНЫЕ ПРОБЛЕМЫ

### P2-1. Неограниченный рост `messages` и `tool_calls` в state
**Файл:** `graph/nodes/common.py` — `BaseNodeMixin`

**Проблема:** Каждая нода делает `state["messages"] + [...]`. При 3 retry-циклах:
`11 нод × 2 сообщения × 3 retry = ~66 сообщений`. Всё это попадает в `summarizer`.
`tool_calls` растёт аналогично. Это прямо раздувает промпт summarizer'а и жрёт токены.

**Решение:**
- Добавить в `BaseNodeMixin` утилиту `_cap_messages(messages, max=20)` — обрезать начало
- Добавить `_cap_tool_calls(tool_calls, max=10)` — оставлять последние N
- Вызывать в каждой ноде перед созданием обновления состояния

---

### P2-2. `STATEMENT_TIMEOUT_MS = 300_000` (5 мин) = весь wall-clock бюджет графа
**Файл:** `core/database.py`

**Проблема:** Один тяжёлый SQL-запрос может потратить весь бюджет в 300 секунд.
При этом граф просто зависнет на выполнении, а на исправление ошибок времени не останется.

**Решение:** Снизить `STATEMENT_TIMEOUT_MS` до `90_000` (90 секунд).
При `timeout` ошибка идёт в `error_diagnoser` с типом `other` → replanning.
Добавить в `GIGACHAT_COMMON_ERRORS`: предупреждение о `statement_timeout`.

---

### P2-3. Фильтрация каталога в `_get_schema_context` при >50 таблицах
**Файл:** `graph/nodes/common.py` — `_get_schema_context`

**Проблема:** При каталоге >50 таблиц (`_MAX_TABLES_FOR_FULL_CATALOG = 50`) применяется
keyword-фильтрация. Если нужная таблица не совпадает по ключевым словам (разные языки,
аббревиатуры), `table_resolver` её не увидит и не выберет — физически отсутствует в промпте.

**Решение:**
- Использовать новый TF-IDF поиск SchemaLoader'а вместо raw keyword filter
- Показывать топ-30 по релевантности + обязательно добавлять fallback-строку:
  `"Используй search_tables / search_by_description для поиска за пределами топ-30"`
- Поднять `_MAX_TABLES_FOR_FULL_CATALOG` до 100 (если каталог ≤100 — показывать всё)

---

### P2-4. `_semantic_sql_check` работает ПОСЛЕ выполнения SQL
**Файл:** `graph/nodes/sql_pipeline.py` — `sql_validator_node`

**Проблема:** `_semantic_sql_check` (LLM-вызов) запускается после `execute_query`.
Для неправильного SQL это: выполнить → получить плохой результат → проверить семантику.
Пустая проверка: уже потрачено время выполнения и один LLM-вызов.

**Решение:** Перенести `_semantic_sql_check` в `sql_static_checker` (P1-1).
Там расширить детерминированные проверки:
- `WHERE date_col > X` без верхней границы → предупреждение (уже есть в semantic check)
- Нет `GROUP BY` при наличии агрегатов и не-агрегированных колонок в SELECT

Текущий `_semantic_sql_check` — сохранить как post-execution хук только для сложных семантических
несоответствий (не для базовых ошибок SQL структуры).

---

### P2-5. `export_query` не требует подтверждения
**Файл:** `tools/db_tools.py`, `cli/interface.py`

**Проблема:** LLM может вызвать `export_query` без подтверждения пользователя.
Это создаёт файлы на сервере без ведома пользователя и может экспортировать большие датасеты.

**Решение:** Добавить `execute_query` и `export_query` в список инструментов,
требующих подтверждения при `total_rows > 10_000` (уже есть механизм `needs_confirmation`).
В `sql_validator_node`: если `export_query` и `total_rows > 10_000` → `needs_confirmation = True`.

---

### P2-6. `_parse_tool_call` retry = лишний LLM-вызов при плохом форматировании
**Файл:** `graph/nodes/common.py` — `_parse_tool_call`

**Проблема:** Когда GigaChat возвращает ответ без JSON (пояснения + SQL без обёртки),
`_parse_tool_call` делает retry LLM-вызов ("отформатируй в JSON"). Это:
- +5 секунд задержки (rate limit)
- Расход лимита попыток

**Решение:** Перед retry добавить эвристический extractor:
- Найти SQL-блоки в тексте (`SELECT ... FROM ...` через regex)
- Если найден SQL → обернуть в `{"tool": "execute_query", "args": {"sql": "..."}}`
- Только если и это не сработало → делать LLM retry

---

## ПРИОРИТЕТ 3 — НОВЫЕ ФИЧИ

### P3-1. Команда `metrics` в CLI
**Файл:** `cli/interface.py`

**Проблема:** `memory.get_sql_quality_metrics()` полностью реализован в `core/memory.py`,
но нигде не вызывается из UI.

**Решение:** Добавить `elif command == "metrics"` в `CLIInterface.run()`:
```
=== Метрики качества (последние 30 дней) ===
Всего запросов: 47
Успешность: 89.4%  |  С первой попытки: 72.3%
Среднее retry: 0.4  |  Максимум: 3
Топ ошибок: column_not_found (8), row_explosion (3), syntax_error (2)
```

**Ожидаемый impact: Видимость прогресса → осознанные улучшения**

---

### P3-2. Query cache для идентичных запросов
**Файл:** `core/query_cache.py` (новый), `cli/interface.py`

**Проблема:** Повторные запросы прогоняют весь граф (11 нод × 5с rate limit = 55+ сек минимум).

**Решение:**
- SQLite-таблица `query_cache (query_hash, user_input, sql, answer, created_at)`
- TTL: 1 час (данные в Greenplum обновляются)
- Ключ: SHA256 от normalized(user_input)
- В `CLIInterface._process_query`: cache lookup перед запуском графа
- При hit: показать кэшированный ответ + сообщить возраст кэша, предложить обновить

**Ожидаемый impact: ×10 скорость на повторных запросах**

---

### P3-3. Multi-turn контекст: ссылки на предыдущий результат
**Файлы:** `graph/state.py`, `cli/interface.py`, `graph/nodes/intent.py`

**Проблема:** "А теперь сгруппируй это по регионам" — агент не знает что "это".

**Решение:**
- В `AgentState` добавить поле `prev_sql: str` и `prev_result_summary: str`
- В `CLIInterface._process_query`: после успешного выполнения сохранять `last_sql` и краткое описание результата
- При следующем запросе: если `intent.get("followup")` → включить `prev_sql` в промпт `sql_writer`
- В `intent_classifier`: добавить тип `followup` — когда запрос ссылается на предыдущий ("это", "ещё раз", "теперь", "добавь к этому")

---

### P3-4. Golden test suite
**Файл:** `tests/test_golden.py` (новый), `tests/golden/*.yaml`

**Проблема:** Нет regression suite. Любой рефакторинг промпта — риск неизвестного влияния на качество.

**Формат:**
```yaml
# tests/golden/basic_count.yaml
query: "Сколько клиентов?"
expected_tables_include: ["clients"]
sql_must_contain: ["COUNT"]
sql_must_not_contain: ["кириллица"]
max_retries: 0
```

**Тесты:** 20 базовых (COUNT/SUM/GROUP BY) + 10 JOIN + 5 date filters + 5 edge cases.
Запуск с `MockLLM` → детерминированные ответы → проверка финального SQL.

---

### P3-5. Confidence scoring в `table_resolver`
**Файл:** `graph/nodes/intent.py`

**Проблема:** Все найденные таблицы имеют одинаковый "вес". Нет сигнала низкой уверенности.

**Решение:** Добавить в ответ `table_resolver` поле `confidence` (0–100):
- 100 = явное `schema.table` в запросе
- 70–99 = прямое совпадение по имени/TF-IDF
- 40–69 = через synonym expansion
- <40 = слабое косвенное соответствие

Если `confidence < 50` И только одна таблица → добавить предупреждение в финальный ответ:
`"Таблица выбрана с низкой уверенностью. Если результат неверный — уточните запрос."`

---

### P3-6. Greenplum distribution key awareness
**Файлы:** `core/schema_loader.py`, `tools/catalog_builder.py`

**Проблема:** Greenplum — MPP-система. Запросы без фильтра по distribution key делают
full table scan по всем сегментам. Агент ничего не знает о distribution keys.

**Решение:**
- Добавить колонку `distributed_by` в `attr_list.csv` (генерировать из `pg_catalog.gp_distribution_policy`)
- В `table_info` показывать distribution key: `DISTRIBUTED BY (client_id)`
- В `column_selector` и `sql_writer`: если таблица имеет distribution key и он не используется в WHERE/JOIN → добавить предупреждение в промпт

---

## ТЕСТИРОВАНИЕ

### Немедленно добавить тесты (нет в репозитории):

**`tests/test_sql_static_checker.py`:**
- SQL с несуществующей колонкой → ошибка `hallucinated_column`
- SQL с кириллическим алиасом → ошибка `cyrillic_alias`
- SQL с `SELECT *` → предупреждение
- Чистый SQL → нет ошибок

**`tests/test_schema_loader_tfidf.py`:**
- Запрос "выручка" → в топ-3 таблицы с revenue/sales в названии/описании
- Запрос "отток клиентов" → в топ-3 таблицы с churn/attrition
- Запрос на несуществующее → пустой список

**`tests/test_query_cache.py`:**
- Cache miss → None
- Cache set + get → возвращает правильное значение
- TTL expiry → None после истечения

**Дополнить `tests/test_graph_e2e.py`:**
- Добавить тесты с MockLLM для реального прохождения нод (happy path)
- Тест цикла коррекции: sql_validator error → error_diagnoser → sql_fixer → sql_validator success

---

## ИЗВЕСТНЫЕ ТЕХНИЧЕСКИЕ ДОЛГИ (не срочно)

| ID | Описание | Файл |
|----|----------|------|
| TD-1 | `correction_examples` в LT-памяти в verbose формате — тратит токены | `core/memory.py` |
| TD-2 | `_check_disambiguation_needed` не вызывается в pipeline (мёртвый код?) | `graph/nodes/common.py` |
| TD-3 | `disambiguation_options` в state никогда не заполняется после рефакторинга | `graph/state.py` |
| TD-4 | `tables_context` помечено как DEPRECATED но ещё используется в summarizer fallback | `graph/state.py` |
| TD-5 | `sql_planner` fallback использует `list(selected_columns.keys())[0]` — порядок dict non-deterministic | `graph/nodes/sql_pipeline.py` |
| TD-6 | `STATEMENT_TIMEOUT_MS` в db_manager не пробрасывается как `SET statement_timeout` в сессию | `core/database.py` |
| TD-7 | `needs_disambiguation` пути в графе ведут в `END` → recursive call в CLIInterface | `cli/interface.py` |
| TD-8 | `test_graph_e2e.py` не тестирует реальные ноды, только state creation и парсинг | `tests/test_graph_e2e.py` |
