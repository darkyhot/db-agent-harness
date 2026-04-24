# Промпт для будущего Claude: ревью логов /deep_table_analysis

Этот файл — инструкция самому себе в следующей сессии. Пользователь пришлёт
артефакты прогона `/deep_table_analysis` (доступ к БД у него закрытый
контур — я её не вижу никогда). Вся обратная связь — через эти файлы.

---

## 1. Сначала прочитай контекст

1. `~/.claude/projects/-home-darky-db-agent-harness/memory/MEMORY.md` — общий индекс.
2. `~/.claude/projects/-home-darky-db-agent-harness/memory/project_deep_analysis.md` — что уже сделано и что отложено.
3. Исходники:
   - `core/deep_analysis/` — весь пайплайн.
   - `core/deep_analysis/runners/` — каждый runner изолирован.
   - `cli/interface.py:_handle_deep_table_analysis` — точка входа CLI.
4. Тесты:
   - `tests/test_deep_analysis.py` — integration на синтетике.
   - `tests/test_deep_analysis_role_detection.py` — ~55 реальных колонок схемы.

Перед любой правкой прогоняй `.venv/bin/pytest tests/test_deep_analysis.py tests/test_deep_analysis_role_detection.py -v` — это baseline.

---

## 2. Что пользователь обычно присылает

Артефакты лежат в `workspace/deep_analysis/<schema.table>/<ts>/`:

| Файл | Что показывает |
| --- | --- |
| `report.md` | Основной читаемый артефакт. Начинай с него. |
| `findings.jsonl` | Машинный список находок (по 1 на строку). |
| `diagnostics.jsonl` | Запись выполнения каждой гипотезы (status / runner / seconds / n_findings / error_summary). |
| `deep_analysis.log` | Полный отладочный лог — DEBUG-уровень, profile brief, per-hypothesis timing. |
| `entities_*.csv` | CSV-выгрузки нарушителей/сводок — по одному на находку. Comma-separated. |

Если пользователь прислал только часть — попроси недостающее. В 90% случаев достаточно `report.md` + `deep_analysis.log`.

---

## 3. Порядок чтения

### 3.1. `report.md` → блок «Диагностика выполнения»

Смотри на счётчик статусов:
- `✅ ok` — гипотеза отработала. Количество `n_findings` = 0 ещё не значит «плохо»: бывает, что аномалий реально нет.
- `⏭ skip` — runner не найден. Не должно случаться; если есть — опечатка в каталоге или LLM вернул невалидный `runner`.
- `❌ error` — runner упал. Читать `error_summary`. Если это `KeyError`/`AttributeError` — скорее всего рассинхрон с каталогом; если `ValueError` внутри statsmodels/ruptures/scipy — вопрос к данным (см. §5).
- `⏰ budget` — исчерпан лимит времени. Проверь, сколько гипотез успело отработать до этого и какие пропустили.

### 3.2. `report.md` → блок «Профайл колонок»

Тут дайджест профайла. Проверяй:
- **Роли колонок** — всё ли распознано правильно? Если `amt` помечен `numeric` вместо `money`, или `inn` стал `numeric` вместо `id` — это сигнал править `core/deep_analysis/profiler.py` (hint-кортежи).
- **Кардинальности и null%** — нет ли колонок с 100% null (тогда runners на них не дадут ничего). Нет ли ID-шек со cardinality=n_rows (праймери-ключи не годятся для `entity_col`).
- **Стратегия загрузки** — если `sample` вместо `full`, значит SafeLoader упал в fallback. Посмотри `deep_analysis.log` за причиной.
- **Откинутые широкие текстовые колонки** — проверь, нет ли среди них бизнес-критичных. Если есть — надо либо расширить `_PROTECTED_NAME_SUBSTRINGS` в `core/deep_analysis/loader.py`, либо обсудить с пользователем порог `WIDE_TEXT_THRESHOLD=100`.

### 3.3. Находки в `report.md`

Пробегаю по severity:
- `🔥 critical` / `⚠️ strong` — первое, что надо обсудить. Спроси у пользователя: «это реально паттерн или false positive?». На основе ответа тюнь пороги (см. §4).
- `ℹ️ notable` — middle ground. Если их много и большинство ложные — поднять пороги.
- `•  info` — обычно шум. Если сильно мусорит отчёт — поднять порог severity.

### 3.4. `deep_analysis.log`

Смотри:
- Строки `Load estimate` — подтверждают, что loader посчитал память корректно.
- `Profile brief` — полный дайджест (урезанный в report.md).
- Per-hypothesis `took Xs, produced N findings` — понять, кто долгий.
- Traceback-и ошибок — для error-статуса.

---

## 4. Типичные тюнинги и где их делать

### 4.1. Пороги severity

| Где | Файл | Константы |
| --- | --- | --- |
| Group anomalies (MAD-z) | `core/deep_analysis/runners/group_anomalies.py` | `_THRESHOLD = 2.5`, `_MIN_ENTITY_SAMPLE = 10` |
| Outliers (MAD) | `core/deep_analysis/runners/outliers.py` | `_MAD_THRESHOLD = 4.0` |
| Outliers (IsolationForest) | `core/deep_analysis/runners/outliers.py` | `_IFOREST_CONTAMINATION = 0.01` |
| Seasonality (шумы отсеиваются) | `core/deep_analysis/runners/seasonality.py` | `p_value > 0.01 and abs(top_dev) < 0.15` |
| Dependencies | `core/deep_analysis/runners/dependencies.py` | `_CRAMER_V_STRONG = 0.25`, `_SPEARMAN_STRONG = 0.4`, `_ETA_SQ_STRONG = 0.1` |
| Regime shifts | `core/deep_analysis/runners/regime_shifts.py` | `_MIN_REL_SHIFT = 0.2`, PELT penalty = 3σ² |

Правило большого пальца: если пользователь говорит «слишком много false positives» — поднимай пороги на 1 шаг (например, `_CRAMER_V_STRONG: 0.25 → 0.35`). Если «пропускает очевидные вещи» — опускай.

### 4.2. Роли колонок

Файл: `core/deep_analysis/profiler.py`. Hint-кортежи:
- `_MONEY_HINTS`, `_PERCENT_HINTS`, `_ID_HINTS`, `_ID_EXACT_NAMES`, `_DATE_HINTS`, `_FLAG_HINTS`, `_CATEGORY_HINTS`, `_QTY_SUFFIXES`, `_TEXT_EXACT_NAMES`.

Важное правило: `_ID_MIN_CARDINALITY = 50` — колонка становится `ID` только при ≥50 уникальных. Иначе `status_id` с 5 значениями правильно классифицируется как `CATEGORY`.

**Алгоритм правки:**
1. Взять несовпадающую колонку из report.md.
2. Добавить её в `tests/test_deep_analysis_role_detection.py:CASES` с ожидаемой ролью.
3. Прогнать → увидеть падение.
4. Править hint-кортеж так, чтобы тест зелёный, И старые кейсы остались зелёными.

### 4.3. Каталог гипотез

Файл: `core/deep_analysis/hypothesis_catalog.py`. Приоритеты (чем больше — тем раньше запускается):
- `end_of_quarter_shift`: 0.95
- `group_anomalies rate`: 0.9
- `group_anomalies volume`: 0.85
- `group_anomalies mean`: 0.8
- `seasonality category`: 0.75
- `dependencies pairwise`: 0.7
- `seasonality rows`: 0.7
- `regime count`: 0.65
- `regime numeric`: 0.6
- `seasonality numeric`: 0.6
- `outliers mv`: 0.55
- `outliers univariate`: 0.45

Если пользователь скажет «для наших таблиц X важнее Y» — подкрути приоритеты.

### 4.4. Бюджет

Файл: `core/deep_analysis/orchestrator.py`:
- `FAST_BUDGET_SEC = 40 * 60`
- `DEEP_BUDGET_SEC = 3 * 60 * 60`

Если пользователь говорит «fast-режим не укладывается» — снижай `_MAX_ROWS_FOR_TESTS` в `dependencies.py` или `_IFOREST_MAX_ROWS` в `outliers.py`. Не поднимай бюджет просто так — симптом, что нужна оптимизация.

---

## 5. Типичные падения и что делать

### 5.1. `statsmodels` / `scipy` падает на малых/вырожденных выборках
Обычно когда в бакете < `_MIN_SAMPLES_PER_BUCKET` (по умолчанию 10) или std=0.
**Фикс:** добавить ранний `return []` перед вызовом теста. Паттерн уже есть в seasonality — повторить в runner-е где упало.

### 5.2. `ruptures` падает на постоянном ряду
Уже есть guard `if std == 0: return []`. Если всё равно падает — дебажить через лог penalty и длину ряда.

### 5.3. MemoryError при загрузке
Loader должен был уйти в sample. Если упал — проверить `psutil.virtual_memory().available` в логе, возможно нужно снизить `SAFE_MAX_DF_BYTES` в `core/deep_analysis/loader.py` (сейчас 60 GB).

### 5.4. LLM вернул невалидный JSON
В `core/deep_analysis/hypothesis_llm.py` уже есть defensive parse. Если часто ломается — возможно локальный LLM сильно отличается от GigaChat, и надо добавить более толерантный `_parse_llm_json`. Или temperature понизить до 0.05.

### 5.5. Плохие гипотезы от LLM
LLM предлагает колонки, которые есть в профайле, но не имеют смысла в комбинации. Фильтр в `_validate_hypothesis_dict` уже проверяет существование колонок, но не семантику. Если мусорно — либо ужесточать system-prompt, либо добавлять post-валидацию по ролям (например, `seasonality.date_col` должен быть DATE/DATETIME).

### 5.6. Plan-preview пользовательской гипотезы — бесконечные правки
Лимит = 3 (`for _ in range(3)` в CLI). Если пользователь жалуется, что не успевает объяснить — поднять до 5. Но обычно 3 достаточно.

---

## 6. Что обсуждать с пользователем (не делать молча)

Эти пункты заблокированы отсутствием живых данных, обсуждаем проактивно, если в отчёте есть сигналы:

- **Chunked loader для таблиц >60 ГБ.** Если видишь `strategy=sample` и большой объём — спрашиваем: «твоя таблица, скорее всего, партиционирована? По какой колонке?». В зависимости от ответа — стратегия чанкования (по дате / по id / через `pg_class.reltuples`).
- **Association rules runner.** Если у пользователя категориальные колонки с богатыми комбинациями и он говорит «хочу видеть правила вида if X=A then Y=B» — доставать `mlxtend` или ручную реализацию. Но только после первого запроса.
- **Golden-тесты.** Спросить пользователя, готов ли он сам написать ожидания по конкретным таблицам (findings-expectations), которые закрепим в CI.

---

## 7. Что НЕ трогать без обсуждения

- Hint-кортежи в `profiler.py` — любая правка должна сопровождаться обновлением `tests/test_deep_analysis_role_detection.py:CASES`. Иначе дрифт не детектится.
- `_ID_MIN_CARDINALITY` — критичная граница. Если меняешь — проверь минимум `gosb_id` (ID) vs `tb_id` (CATEGORY) в тестах.
- Формат CSV (comma-separated). Пользователь явно попросил такой.
- Ограничение 30 минут на ранний этап — формально отменено (fast=40, deep=3ч), но не увеличивай dryless.

---

## 8. Цикл работы за один присланный отчёт

1. Прочитать `report.md` целиком.
2. Выписать найденные проблемы: (a) неправильные роли колонок, (b) error/skip гипотез, (c) false positive / false negative по находкам.
3. Для каждой — предложить пользователю фикс (не чинить молча, если не очевидно). Указать конкретный файл и константу.
4. Если пользователь согласился — сделать правку + обновить соответствующий тест + прогнать сьюты:
   - `.venv/bin/pytest tests/test_deep_analysis.py tests/test_deep_analysis_role_detection.py -v` (≈5 мин, основные)
   - `.venv/bin/pytest --ignore=tests/test_deep_analysis.py --ignore=tests/test_deep_analysis_role_detection.py -q` (≈15с, регрессионный)
5. Обновить `project_deep_analysis.md`, если изменилось что-то архитектурно важное.

---

## 9. Про данные

Пользователь — банк, закрытый контур, LLM локально, Python 3.12, .venv. БД — Greenplum/PostgreSQL. Таблицы могут быть до сотен ГБ. В metadata (`data_for_agent/attr_list.csv` и соседние файлы) лежат схемы реальных таблиц с описаниями — это единственный легальный источник «правды» о структуре данных. Используй его для проверки гипотез о паттернах именования.

Когда пользователь загрузит больше таблиц в metadata — это повод пройтись по `attr_list.csv` и расширить hint-ы под новые паттерны + дополнить `CASES` в role-detection тесте.
