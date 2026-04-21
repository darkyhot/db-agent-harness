# Roadmap: баланс LLM и детерминизма в db-agent-harness

Документ для агента-разработчика. Цель — внедрить **приоритетный слой явного пользовательского intent** поверх существующей детерминированной архитектуры без раздувания промптов и без потери текущих гарантий безопасности JOIN.

---

## 1. Контекст

Сейчас проект построен на «детерминированном» подходе: хэуристики выбирают таблицы, колонки, JOIN-стратегию; LLM (GigaChat) подключается только для финального SQL и узких задач. Проблема — жёсткие пороги отклоняют валидные запросы, когда пользователь добавляет новые измерения ("по task_code"), явно указывает таблицы ("возьми X, соедини с Y") или получает пустой результат (0 строк трактуется как ошибка на уровне UX).

Решение — не выбирать между LLM и детерминизмом, а **добавить третий слой: explicit user intent**, который имеет приоритет над обоими. Механика: расширить уже существующий `user_hint_extractor`, встроить override'ы в детерминированные узлы, добавить opt-in plan-preview.

Все изменения подчиняются `CLAUDE.md`: без хардкода схем, 5-сек. задержка между LLM-вызовами, новая логика идёт в новую ноду, а не в раздутый промпт.

---

## 2. Принципы

1. **Приоритет решений:** `explicit user intent > deterministic heuristic > LLM fallback`. Если пользователь явно указал — это работает, даже если эвристика сомневается.
2. **Не раздувать существующие промпты.** Если новая логика требует >30% к промпту — создавать отдельную ноду с отдельным вызовом LLM (или детерминированным рендером без LLM).
3. **Не ломать 47 существующих тестов.** Каждая задача — запустить `pytest tests/` полностью и убедиться, что регрессий нет. Новые golden-кейсы под каждую правку.
4. **Сервер без выхода в интернет.** Ничего нельзя скачивать при запуске: ни HuggingFace моделей, ни внешних образов, ни pre-commit-репозиториев из GitHub. Все зависимости уже должны быть в окружении или устанавливаться локально из оффлайн-зеркала pip. GitHub Actions и аналогичные облачные CI — не применимы.
5. **Rate-limit'ы разных API:** текстовый GigaChat — **5 сек** между вызовами (`RateLimitedLLM` в [core/llm.py](core/llm.py)). GigaChat Embeddings — **6 сек** между батч-вызовами (отдельный клиент). Это разные rate-limit'ы, не путать.

---

## 3. Фазы

### Фаза 1 — Tactical (неделя 1)

Четыре задачи, закрывающие конкретные боли пользователя: «посчитай по task_code», «возьми таблицу X», «помесячно», «0 строк».

### Фаза 2 — Architectural (недели 2–3)

Plan-preview и explicit mode для power-users.

### Фаза 3 — Strategic (месяц 2+)

Семантический matching, feedback loop, observability.

---

## 4. Задачи

### Задача 1.1 — Расширить `user_hint_extractor` до 8 типов хинтов

**Цель:** парсер должен вытаскивать не только таблицы/JOIN-поля, но и group-by, агрегаты, гранулярность времени, негативные фильтры.

**Файлы:**
- [core/user_hint_extractor.py](core/user_hint_extractor.py) — добавить регексы и функции валидации
- [graph/state.py](graph/state.py) — расширить type-комментарий поля `user_hints`
- [graph/graph.py](graph/graph.py) `create_initial_state` — добавить новые ключи в дефолтный `user_hints`
- [tests/test_user_hint_extractor.py](tests/test_user_hint_extractor.py) — по одному кейсу на каждый новый тип

**Новые поля в `user_hints`:**

| Ключ | Тип | Пример запроса | Пример значения |
|---|---|---|---|
| `group_by_hints` | `list[str]` | "по task_code", "сгруппируй по региону" | `["task_code"]` |
| `aggregate_hints` | `list[tuple[str, str]]` | "посчитай задачи", "сумма по выручке" | `[("count", "task"), ("sum", "revenue")]` |
| `time_granularity` | `str \| None` | "помесячно", "по кварталам" | `"month"` / `"quarter"` / `"year"` / `"day"` |
| `negative_filters` | `list[str]` | "не учитывай X", "исключи Y" | `["канцелярия"]` |

**Правила реализации:**
- Регексы для русского и английского **раздельно** (паттерны с `\bon\b|по`, `\bby\b|группирова\w*` и т.п.)
- Любой identifier (`task_code`, `region`, ...) валидируется через `schema_loader.search_columns` на наличие хотя бы в одной таблице. Если не найден — хинт отбрасывается и пишется `logger.debug`.
- **Нельзя** хардкодить имена колонок. Всё приходит из каталога.
- `time_granularity` нормализуется: "помесячно"/"по месяцам"/"monthly" → "month". Список нормализаций — в константе модуля.

**Acceptance:**
- Запрос «посчитай задачи по фактическому оттоку в феврале 26. По task_code» → `user_hints.group_by_hints == ["task_code"]`, `aggregate_hints == [("count", ...)]`.
- Запрос «возьми dm.fact_churn, посчитай помесячно» → `time_granularity == "month"`, `must_keep_tables == [("dm","fact_churn")]`.
- Запрос «покажи всех клиентов, не учитывай канцелярию» → `negative_filters == ["канцелярия"]`.
- Все существующие тесты `tests/test_user_hint_extractor.py` проходят без изменений.

**Зависимости:** нет. Делать первой.

---

### Задача 1.2 — Hint-override в детерминированных узлах

**Цель:** если пользователь явно назвал что-то в запросе, детерминированные пороги не должны блокировать пайплайн.

**Файлы и точные изменения:**

#### [graph/nodes/explorer.py:438](graph/nodes/explorer.py#L438)

Сейчас:
```python
if _det_conf >= _DET_CONFIDENCE_THRESHOLD and not state.get("column_selector_hint", ""):
    # skip LLM
```

Стало:
```python
_hints = state.get("user_hints", {}) or {}
_has_column_hint = bool(_hints.get("group_by_hints") or _hints.get("aggregate_hints"))
if (
    _det_conf >= _DET_CONFIDENCE_THRESHOLD
    and not state.get("column_selector_hint", "")
    and not _has_column_hint
):
    # skip LLM
```

Если hint есть — проходим через LLM-путь (или передаём `force_user_hints=True` в `_det_select_columns`, если такую опцию добавим).

#### [core/where_resolver.py:139-148](core/where_resolver.py#L139-L148)

Перед `break` добавить:
```python
request_id_str = str(request_id)
if request_id_str in user_filter_choices:
    # Пользователь уже выбрал — не прерываем
    pass
elif _user_explicitly_named(user_input, best, second):
    # Пользователь назвал колонку по имени — берём best, не прерываем
    pass
else:
    # старая логика с break
```

`_user_explicitly_named` — новая приватная функция: проверяет, встречается ли `best.column` или `second.column` (нормализованное имя) в `user_input`. Если встречается только одна — берём её, `clarification_message` не выставляется.

#### [core/sql_planner_deterministic.py](core/sql_planner_deterministic.py)

В функции сборки `group_by` (найти по `"group_by"` / `"aggregation"`):
- Если `state["user_hints"]["time_granularity"]` задан и в выбранных колонках есть хотя бы одна date/timestamp-колонка (по `dtype`) — **обязательно** обернуть её в `DATE_TRUNC('{granularity}', col)` и добавить в `group_by`.
- Если дата-колонки нет — записать в `state["planning_confidence"]["components"]["filter_confidence"]["evidence"]` причину и **не** бросать ошибку.

**Acceptance:**
- «возьми dm.fact_churn, сгруппируй помесячно по task_code» → итоговый SQL содержит `GROUP BY DATE_TRUNC('month', <date_col>), task_code`.
- «посчитай задачи по task_code» → не срабатывает clarification в where_resolver; SQL выполняется; task_code в `GROUP BY`.
- Существующие golden-тесты (`tests/test_golden.py`, `tests/test_regression_matrix.py`) проходят.

**Тесты добавить:**
- `tests/test_column_selector_deterministic.py` — один кейс с непустыми group_by_hints.
- `tests/test_where_resolver.py` — кейс с user_filter_choices и кейс с явно названной колонкой.
- `tests/test_sql_planner_deterministic.py` — кейс с `time_granularity="month"`.

**Зависимости:** нужна задача 1.1 (расширенный `user_hints`).

---

### Задача 1.3 — Weighted confidence с hint-boost

**Цель:** один слабый компонент не должен блокировать весь пайплайн, если пользователь явно указал направление.

**Файл:** [core/confidence.py:135](core/confidence.py#L135) — функция `build_planning_confidence`.

**Сейчас:**
```python
score = min(
    float(components["table_confidence"]["score"]),
    float(components["filter_confidence"]["score"]),
    float(components["join_confidence"]["score"]),
)
```

**Стало** (псевдокод — при реализации добавить сигнатурный параметр `user_hints` и прокинуть его из вызывающего кода в [graph/nodes/common.py](graph/nodes/common.py) / [graph/nodes/explorer.py](graph/nodes/explorer.py)):
```python
table_score = components["table_confidence"]["score"]
filter_score = components["filter_confidence"]["score"]
join_score = components["join_confidence"]["score"]

hints = user_hints or {}
if hints.get("must_keep_tables"):
    table_score = max(table_score, 0.9)
if hints.get("group_by_hints") or hints.get("aggregate_hints"):
    filter_score = max(filter_score, 0.8)
if hints.get("join_fields"):
    join_score = max(join_score, 0.9)

score = 0.4 * table_score + 0.3 * filter_score + 0.3 * join_score
```

Коэффициенты подобрать так, чтобы:
- `min`-режим существующих тестов продолжал работать (`scores=[0.9, 0.9, 0.9] → high`, `[0.3, 0.9, 0.9] → low`).
- Новые тесты с hint-boost давали `high`.

**Acceptance:**
- [tests/test_confidence.py](tests/test_confidence.py) — все существующие тесты проходят (может потребовать ручной подгонки ожиданий, но не ослабления semantics).
- Новый тест: запрос с `user_hints.must_keep_tables=[("dm","fact_churn")]` и `table_confidence.score=0.5` → `planning_confidence.level == "high"`.
- Защита сохраняется: запрос БЕЗ hints и `table_confidence.score=0.3` → `level == "low"`, `action == "stop"`.

**Зависимости:** требует задачу 1.1 (поля в `user_hints`). Порядок: делать после 1.1, можно параллельно с 1.2.

---

### Задача 1.4 — Soft empty_result в summarizer

**Цель:** 0 строк — валидный ответ «данных за период нет», а не «ошибка/retry».

**Проверено по коду:** [graph/nodes/sql_pipeline.py:951-971](graph/nodes/sql_pipeline.py#L951-L971) уже обрабатывает `is_empty` как warning, не как ошибку. Проблема **в summarizer** — он не превращает warning в внятный текст для пользователя.

**Файл:** [graph/nodes/summarizer.py](graph/nodes/summarizer.py).

**Изменение в `_get_summarizer_system_prompt`:**

Добавить в конец системного промпта блок (≤ 30% от текущей длины, иначе — выделить в отдельную ноду `empty_result_advisor`):
```
- Если в результатах инструментов явно указан `is_empty: true` или 0 строк:
  * Не говори просто «данных нет».
  * Перечисли какие фильтры применялись (период, условия WHERE).
  * Предложи: расширить диапазон дат, проверить написание значений, или посмотреть соседние периоды.
  * Не извиняйся за SQL — он корректный.
```

**Альтернатива (если промпт вырастет > 30%):** создать `graph/nodes/empty_result_advisor.py`, подключить в [graph/graph.py](graph/graph.py) перед summarizer, если `tool_calls[-1].result.is_empty == true`. Этот узел даёт отдельный короткий LLM-вызов, собирающий объяснение (соблюдать 5-сек. задержку — `time.sleep(5)` между вызовами в цепочке).

**Acceptance:**
- SQL вернул 0 строк → ответ содержит конкретные фильтры («за февраль 2026, reason_code='actual_churn'») и предложение расширить период. Не содержит «ошибка/сбой/retry».
- Тест: [tests/test_summarizer_preview.py](tests/test_summarizer_preview.py) — новый кейс с `is_empty: true`, проверяющий что в `final_answer` есть строка с диапазоном дат и рекомендация.

**Зависимости:** нет. Можно делать параллельно с 1.1–1.3.

---

### Задача 2.1 — Plan-preview режим (opt-in)

**Цель:** перед выполнением SQL показать человекочитаемый план, дать пользователю возможность править свободным текстом. Выключено по умолчанию — чтобы не ломать Jupyter UX.

**Новый файл:** `graph/nodes/plan_preview.py` — mixin по образцу [graph/nodes/hint_extractor.py](graph/nodes/hint_extractor.py).

**Точка врезки в графе:** [graph/graph.py:330](graph/graph.py#L330) — заменить `sql_planner → sql_writer` на `sql_planner → plan_preview → sql_writer`. `plan_preview` проверяет конфиг и либо проходит транзитом, либо выставляет `needs_confirmation=True` с текстом в `confirmation_message`.

**Логика узла** (детерминированный рендер, без LLM):
1. Читает `state["sql_blueprint"]`, `state["selected_columns"]`, `state["join_spec"]`, `state["where_resolution"]`, `state["user_hints"]`.
2. Собирает Markdown-описание:
   ```
   План:
   - Главная таблица: dm.fact_churn
   - Агрегация: COUNT(DISTINCT task_code)
   - Фильтры: reason_code='actual_churn', date ∈ [2026-02-01, 2026-02-28]
   - Группировка: DATE_TRUNC('month', date), task_code
   ```
3. Если `config.show_plan == True` (или `state["explicit_mode"] == True` — см. 2.2) — `needs_confirmation=True`, `confirmation_message=<Markdown>`.
4. Иначе — транзит.

**Обработка ответа пользователя** (в [cli/interface.py](cli/interface.py), расширить существующий clarification-loop):
- «ок» / «yes» / «подтверждаю» / «ага» → сброс `needs_confirmation`, граф продолжает в `sql_writer`.
- Любой другой текст → прогоняется через расширенный `user_hint_extractor` (задача 1.1), полученные хинты **мёржатся** в `state["user_hints"]` (не заменяют), граф перезапускается с узла `hint_extractor` (или ближайшего подходящего).
- Максимум 3 итерации правок — потом `break` с предупреждением.

**Конфиг:** [config.json](config.json) — добавить ключ `"show_plan": false`. Считывание — в [cli/interface.py](cli/interface.py) при запуске.

**Acceptance:**
- `show_plan=false` (default) — golden-тесты `tests/test_golden.py`, `tests/test_graph_e2e.py` проходят без изменений.
- CLI-сценарий с `show_plan=true`: запрос → появляется «План: ...» → пользователь вводит «поменяй granularity на неделю» → второй показ плана с `DATE_TRUNC('week', ...)` → «ок» → выполнение.
- Тесты: новый `tests/test_plan_preview.py` — минимум 3 кейса (транзит, confirmation, правка).

**Зависимости:** требует задачи 1.1 (для правок пользователя) и 1.2 (чтобы override'ы применились при перезапуске).

---

### Задача 2.2 — Explicit SQL mode для power-users

**Цель:** когда пользователь явно задал ≥2 параметра (таблицу + JOIN-поле, или таблицу + гранулярность, etc.) — считаем запрос power-user-ским, включаем plan-preview принудительно, строже применяем хинты.

**Новый файл:** `graph/nodes/explicit_mode_dispatcher.py` (mixin).

**Точка врезки:** после [graph/nodes/hint_extractor.py](graph/nodes/hint_extractor.py), перед `table_resolver`. Добавить в [graph/graph.py](graph/graph.py) как промежуточный узел:
```python
graph.add_edge("hint_extractor", "explicit_mode_dispatcher")
graph.add_edge("explicit_mode_dispatcher", "table_resolver")
```

**Логика узла:**
```python
hints = state.get("user_hints", {})
non_empty_count = sum(
    1 for key in ("must_keep_tables", "join_fields", "group_by_hints", "time_granularity")
    if hints.get(key)
)
explicit = non_empty_count >= 2
return {"explicit_mode": explicit}
```

**Новое поле в [graph/state.py](graph/state.py):** `explicit_mode: bool`. Добавить в [graph/graph.py](graph/graph.py) `create_initial_state` со значением `False`.

**Поведение других узлов:**
- [graph/nodes/plan_preview.py](graph/nodes/plan_preview.py) (задача 2.1): если `explicit_mode=True` → `needs_confirmation=True` игнорируя `config.show_plan`.
- [core/column_selector_deterministic.py](core/column_selector_deterministic.py), [core/sql_planner_deterministic.py](core/sql_planner_deterministic.py): при `explicit_mode=True` передавать `strict_user_hints=True` в существующие функции. `strict_user_hints` — означает «не пытайся заменить хинт своим выбором, даже если уверен». Интерпретация зависит от конкретной функции — прописать в docstring.
- [core/confidence.py](core/confidence.py): при `explicit_mode=True` и наличии соответствующих hints — не занижать confidence, даже если heuristic не смог подтвердить (hint-boost применяется жёстче: `max(score, 0.95)` вместо `0.9`).

**Acceptance:**
- Запрос «возьми dm.fact_churn, соедини с dm.dim_reason по reason_code, посчитай помесячно» → `explicit_mode=True` в state, план показан, SQL содержит все три жёстких указания.
- Запрос без явных указаний («покажи статистику по клиентам») → `explicit_mode=False`, поведение не меняется.
- Тесты: новый `tests/test_explicit_mode.py`.

**Зависимости:** требует задачи 1.1, 1.2, 2.1.

---

### Задача 3.1 — Семантический matching через GigaChat Embeddings

**Цель:** «фактический отток» должен находить `reason_code='actual_churn'` даже без подстрочного совпадения. Использовать **локально развёрнутый GigaChat Embeddings** (сервер без выхода в интернет — внешние модели вроде sentence-transformers недоступны).

**Ограничение API:** между вызовами GigaChat Embeddings — **6 секунд** (не 5, как у текстового GigaChat). Батчинг обязателен: `MAX_BATCH_SIZE_CHARS=1_000_000`, `MAX_BATCH_SIZE_PARTS=90`. Один батч-вызов покрывает до 90 текстов за раз, так что 500 колонок → ~6 батчей × 6 сек = ~36 сек на старте. Приемлемо.

**Новый файл:** `core/gigachat_embeddings.py` — обёртка с rate-limit по шаблону пользователя:
```python
from langchain_gigachat.embeddings import GigaChatEmbeddings
from time import perf_counter, sleep
from typing import List
import os

GIGA_EMBED_DELAY = 6
GIGA_EMBED_LAST_INVOKE = 0.0
MAX_BATCH_SIZE_CHARS = 1_000_000
MAX_BATCH_SIZE_PARTS = 90


class GigaChatEmbeddingsDelayed(GigaChatEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        global GIGA_EMBED_LAST_INVOKE
        result: List[List[float]] = []
        local_texts: List[str] = []
        size = 0
        embed_kwargs = {}
        if self.model is not None:
            embed_kwargs["model"] = self.model

        def _flush():
            nonlocal local_texts, size
            global GIGA_EMBED_LAST_INVOKE
            if not local_texts:
                return
            # rate-limit 6s
            elapsed = perf_counter() - GIGA_EMBED_LAST_INVOKE
            if elapsed < GIGA_EMBED_DELAY:
                sleep(GIGA_EMBED_DELAY - elapsed)
            GIGA_EMBED_LAST_INVOKE = perf_counter()
            response = self._client.embeddings(texts=local_texts, **embed_kwargs)
            for item in response.data:
                result.append(item.embedding)
            local_texts = []
            size = 0

        for text in texts:
            local_texts.append(text)
            size += len(text)
            if size > MAX_BATCH_SIZE_CHARS or len(local_texts) >= MAX_BATCH_SIZE_PARTS:
                _flush()
        _flush()
        return result


def build_embedder() -> GigaChatEmbeddingsDelayed:
    return GigaChatEmbeddingsDelayed(
        base_url=os.getenv("GIGACHAT_API_URL"),
        access_token=os.getenv("JPY_API_TOKEN"),
        verify_ssl_certs=False,  # если локальный сервер с self-signed — иначе убрать
    )
```

**Новый файл:** `core/semantic_index.py` — индекс поверх эмбеддера:
1. При инициализации проверяет `data_for_agent/embeddings.npz`:
   - Если файл есть и mtime исходного `attr_list.csv` / `tables_list.csv` меньше — загружает кэш, **эмбеддер не вызывается**.
   - Иначе — прогоняет все тексты (`"{schema}.{table}.{column} {description}"`) через `GigaChatEmbeddingsDelayed.embed_documents` (одним вызовом — внутри батчится), сохраняет `.npz` (numpy `savez_compressed`: `keys`, `vectors`).
2. Метод `similarity(query: str, candidate_keys: list[str]) -> dict[str, float]`:
   - Эмбеддит `query` (один вызов, тоже через delay).
   - Считает cosine для каждого ключа из кэша.
3. Метод `semantic_search(query: str, top_k=10) -> list[tuple[str, float]]` — топ-N.
4. Кэш query-эмбеддингов в памяти (LRU, 256 записей) — чтобы повторные запросы не платили 6 сек.

**Интеграция:**
- [core/schema_loader.py](core/schema_loader.py) — новый публичный метод `semantic_search_tables(query, top_k=10)` рядом с `search_tables` (не замена, а дополнительный сигнал).
- [core/filter_ranking.py:144](core/filter_ranking.py#L144) `_column_reference_score` — **бонусный score**, не замена:
  ```python
  base = <существующая логика: 120 или 35 или 0>
  sem_score = semantic_index.similarity(user_input, [column_key]).get(column_key, 0.0)
  bonus = 20.0 if sem_score >= 0.8 else (10.0 if sem_score >= 0.6 else 0.0)
  return base + bonus
  ```
- **Важно:** semantic_index вызывается в detective-узлах (filter_ranking, where_resolver), которые работают ДО sql_writer. GigaChat Embeddings — это отдельный API, 6-сек. rate-limit применяется внутри обёртки и не влияет на 5-сек. rate-limit текстового `RateLimitedLLM` ([core/llm.py](core/llm.py)). Это два разных клиента.

**Правила:**
- Embedding-файл НЕ коммитится в git. Добавить `data_for_agent/embeddings.npz` в `.gitignore`.
- Пакет `langchain-gigachat` уже есть в проекте (используется в [core/llm.py](core/llm.py)) — дополнительных зависимостей не нужно. Проверить: `grep gigachat requirements.txt`.
- Переменные окружения `GIGACHAT_API_URL`, `JPY_API_TOKEN` — те же, что использует текстовый клиент. Прокинуть в semantic_index через конструктор, не через os.getenv напрямую (тестируемость).
- Если эмбеддер недоступен (нет переменных, сервер не отвечает) — `semantic_index.semantic_search` возвращает пустой список. Существующий fallback substring-поиск должен работать без регрессов.

**Acceptance:**
- Запрос «фактический отток» → `schema_loader.semantic_search_tables(...)` в топ-10 возвращает таблицу с колонкой `reason_code='actual_churn'`.
- Первый старт: `embeddings.npz` отсутствует → индекс строится (видно в логах: `semantic_index: построен индекс, N=500 колонок, batches=6, time=36s`). Второй старт: кэш загружается мгновенно.
- [tests/test_schema_loader.py](tests/test_schema_loader.py), [tests/test_filter_ranking.py](tests/test_filter_ranking.py) — существующие тесты проходят.
- Новые тесты: мокают `GigaChatEmbeddingsDelayed.embed_documents` (возвращают заранее заданные векторы) — проверяют что:
  - batch flush срабатывает при превышении `MAX_BATCH_SIZE_PARTS`,
  - rate-limit sleep вызывается (мокнуть `sleep` и `perf_counter`),
  - bonus-score в `_column_reference_score` применяется корректно.

**Зависимости:** не связана с 1.x/2.x, можно делать параллельно после 1.x.

---

### Задача 3.2 — Feedback loop и negative examples

**Цель:** пользователь может поставить 👍/👎 на ответ; плохие ответы становятся «анти-примерами» в few-shot retrieval.

**Файлы:**

#### [core/memory.py](core/memory.py)
Добавить метод:
```python
def log_user_feedback(
    self,
    query: str,
    sql: str,
    verdict: Literal["up", "down"],
    corrected_sql: str | None = None,
    comment: str | None = None,
) -> None
```
Пишет JSON-строку в `memory/feedback.jsonl` (append-only).

#### [cli/interface.py](cli/interface.py)
После каждого успешного ответа (там, где сейчас печатается `final_answer`):
```
Как ответ? [y=хороший, n=плохой, skip=пропустить]:
```
При `n` — спросить: «хотите предложить правильный SQL? (введите или оставьте пусто):». Передать в `memory.log_user_feedback(...)`.

#### [core/few_shot_retriever.py](core/few_shot_retriever.py)
Расширить сигнатуру `retrieve_examples(query, top_k=3, include_negatives: bool = False)`:
- `include_negatives=True` → дополнительно читает `memory/feedback.jsonl`, выбирает записи с `verdict="down"` и непустым `corrected_sql`, рендерит их в отдельную секцию few-shot:
  ```
  НЕ ДЕЛАЙ ТАК (пользователь исправил):
  Запрос: <query>
  Плохой SQL: <sql>
  Правильный SQL: <corrected_sql>
  ```
- Включать `include_negatives=True` в вызовах из sql_writer / sql_fixer, где это осмысленно.

**Acceptance:**
- Сценарий: запрос → ответ → 👎 → ввод правильного SQL → повтор того же запроса в новой сессии → retrieval показывает negative example → финальный SQL ≠ старому плохому.
- Тесты: [tests/test_few_shot_retriever.py](tests/test_few_shot_retriever.py) — новый кейс с временным `feedback.jsonl` в tmpdir.

**Зависимости:** нет. Самостоятельная задача.

---

### Задача 3.3 — Observability (локальная)

**Цель:** видеть, какой узел сколько работает; иметь локальный скрипт прогона линтеров и тестов. **GitHub Actions не применимо** — сервер без интернета. Pre-commit-хуки ставятся локально без скачивания (либо ставить зависимости ruff/mypy вручную и запускать shell-скриптом).

#### Node timings
**Файл:** [graph/nodes/common.py](graph/nodes/common.py) — декоратор `@track_duration`:
```python
def track_duration(fn):
    @functools.wraps(fn)
    def wrapper(self, state):
        t0 = time.monotonic()
        result = fn(self, state)
        dt = int((time.monotonic() - t0) * 1000)
        trace = state.get("evidence_trace", {}) or {}
        timings = dict(trace.get("node_timings", {}) or {})
        timings[fn.__name__] = timings.get(fn.__name__, 0) + dt
        trace["node_timings"] = timings
        if isinstance(result, dict):
            result.setdefault("evidence_trace", trace)
        return result
    return wrapper
```
Повесить на все методы `*_node` в узлах (intent_classifier, hint_extractor, column_selector, sql_planner, sql_writer, summarizer). Не трогать утилиты и приватные методы.

#### Локальный скрипт проверок
Создать `scripts/check.sh` (или `Makefile`):
```bash
#!/usr/bin/env bash
set -e
ruff check core/ graph/ tools/ tests/ cli/
mypy core/ graph/ --ignore-missing-imports
pytest tests/ -v --ignore=tests/integration
```
Запускается вручную перед каждым коммитом. Зависимости (`ruff`, `mypy`, `pytest`) должны быть предустановлены в окружении сервера — если их нет, добавить инструкцию установки в README/requirements-dev.txt и запросить у пользователя разрешение ставить.

#### Локальный pre-commit (без скачивания из сети)
Вариант 1 — простой git hook: создать `.git/hooks/pre-commit`:
```bash
#!/usr/bin/env bash
exec ./scripts/check.sh
```
Вариант 2 — если хочется конфига, создать `.pre-commit-config.yaml` с `local` repo:
```yaml
repos:
  - repo: local
    hooks:
      - id: local-check
        name: ruff + mypy + pytest
        entry: ./scripts/check.sh
        language: system
        pass_filenames: false
```
`local` repos не требуют доступа в интернет — pre-commit использует системный ruff.

**Acceptance:**
- При `debug=True` в CLI в `state["evidence_trace"]["node_timings"]` появляются ключи для каждого узла со значением в миллисекундах.
- `./scripts/check.sh` запускается локально, все существующие тесты проходят.
- (Опционально) `.git/hooks/pre-commit` настроен и срабатывает при коммите.

**Зависимости:** нет. Самая независимая задача — можно взять «на разогрев».

---

## 5. Общие правила разработки

1. **Не коммитить полу-готовые задачи.** Задача считается выполненной только когда:
   - Все acceptance-criteria выполнены.
   - `pytest tests/` проходит полностью (кроме `tests/integration` — они требуют live GigaChat).
   - Новые тесты добавлены и проходят.

2. **После каждой задачи — полный прогон тестов.** Если упал golden-тест в [tests/test_golden.py](tests/test_golden.py) или [tests/test_regression_matrix.py](tests/test_regression_matrix.py) — разобраться: это реальный регресс (чинить) или ожидаемое изменение (обновить golden после ревью).

3. **Новые промпты — только если нельзя детерминированно.** Прежде чем расширять промпт, спросить: «можно ли собрать это регексом / из state?». Если да — делать детерминированно.

4. **Проверять длину промптов.** При каждом изменении системного промпта измерить длину до/после. Если прирост > 30% — разбить на отдельную ноду (правило `CLAUDE.md`).

5. **Не хардкодить таблицы/схемы/бизнес-термины.** Все identifiers валидируются через `schema_loader.search_tables` / `search_columns`. Если не нашлось — хинт отбрасывается, а не попадает в state.

6. **5-сек. задержка между LLM-вызовами.** При добавлении любого нового LLM-вызова проверить, что задержка соблюдается через `RateLimitedLLM` ([core/llm.py](core/llm.py)) или явный `time.sleep(5)`.

7. **Каждая задача — отдельный коммит (или PR).** Название: `feat: task 1.1 — extend user_hint_extractor` или `fix: task 1.3 — weighted planning confidence`.

8. **Если обнаружил, что задача больше описанной** — остановиться, обновить этот roadmap (добавить подзадачи), спросить пользователя. Не разрастайся молча.

9. **Jupyter + CLI** — обе точки входа официально поддерживаются (правило `CLAUDE.md`). Новая функциональность должна работать в обеих. Тесты — минимум по одному E2E-кейсу на каждую.

---

## 6. Порядок выполнения и зависимости

```
1.1 (hint_extractor расширение)
  ├─> 1.2 (override в узлах)           ──┐
  └─> 1.3 (weighted confidence)         ─┼──> 2.1 (plan-preview) ──> 2.2 (explicit mode)
                                         │
1.4 (empty_result в summarizer) ────────┘  (независимо)

3.1 (embeddings)      ─ независимо, параллельно с 1.x–2.x
3.2 (feedback loop)   ─ независимо
3.3 (observability)   ─ независимо, можно первой для разогрева
```

**Рекомендуемый порядок:** 3.3 → 1.1 → 1.4 (параллельно) → 1.2 → 1.3 → 2.1 → 2.2 → 3.1 → 3.2.

---

## 7. Верификация roadmap'а

Перед тем как начать, агент-разработчик должен:
1. Открыть этот файл и прочитать от начала до конца.
2. Проверить, что файлы и номера строк, упомянутые в задачах, существуют:
   - `grep -n "_column_reference_score" core/filter_ranking.py` → строка 144
   - `grep -n "if _det_conf >= _DET_CONFIDENCE_THRESHOLD" graph/nodes/explorer.py` → строка 438
   - `grep -n "score = min(" core/confidence.py` → строка 135
3. Запустить `pytest tests/` — зафиксировать baseline (должно быть 0 failures перед началом работ).
4. Приступить к задаче 1.1 (или 3.3 для разогрева).

---

## Приложение: критические файлы по задачам

| Задача | Файлы |
|---|---|
| 1.1 | [core/user_hint_extractor.py](core/user_hint_extractor.py), [graph/state.py](graph/state.py), [graph/graph.py](graph/graph.py), [tests/test_user_hint_extractor.py](tests/test_user_hint_extractor.py) |
| 1.2 | [graph/nodes/explorer.py:438](graph/nodes/explorer.py#L438), [core/where_resolver.py:139](core/where_resolver.py#L139), [core/sql_planner_deterministic.py](core/sql_planner_deterministic.py) |
| 1.3 | [core/confidence.py:135](core/confidence.py#L135), [tests/test_confidence.py](tests/test_confidence.py) |
| 1.4 | [graph/nodes/summarizer.py](graph/nodes/summarizer.py), [graph/nodes/sql_pipeline.py:967](graph/nodes/sql_pipeline.py#L967) |
| 2.1 | `graph/nodes/plan_preview.py` (новый), [graph/graph.py:330](graph/graph.py#L330), [cli/interface.py](cli/interface.py), [config.json](config.json) |
| 2.2 | `graph/nodes/explicit_mode_dispatcher.py` (новый), [graph/state.py](graph/state.py), [core/confidence.py](core/confidence.py) |
| 3.1 | `core/gigachat_embeddings.py` (новый), `core/semantic_index.py` (новый), [core/filter_ranking.py:144](core/filter_ranking.py#L144), [core/schema_loader.py](core/schema_loader.py), `.gitignore` |
| 3.2 | [core/memory.py](core/memory.py), [core/few_shot_retriever.py](core/few_shot_retriever.py), [cli/interface.py](cli/interface.py) |
| 3.3 | [graph/nodes/common.py](graph/nodes/common.py), `scripts/check.sh` (новый), `.git/hooks/pre-commit` или `.pre-commit-config.yaml` (local) |
