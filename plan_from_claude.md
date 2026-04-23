# Фикс трёх багов SQL-агента: COUNT(DISTINCT), plan_patcher, is_task

## Context

Запрос пользователя: «Сколько задач по фактическому оттоку поставили в феврале 26» к snapshot-таблице `uzp_dwh_fact_outflow`. Агент:

1. Сгенерировал бессмысленный `COUNT(DISTINCT report_dt)` (считает количество уникальных дат, а не задач).
2. Не применил две последовательные правки пользователя (`plan_patcher: applied 0 operations`).
3. Подсосал `is_task = true` (это работает после коммита `0268fa2`), но механизм хрупкий: ломается, если subject определяется как `outflow`, а не `task`.

Цель — универсальное решение без хардкода имён таблиц/колонок, с сохранением правил из CLAUDE.md (≥5 сек между LLM-вызовами, минимум новых LLM-нод, тесты обязательны).

Решения, согласованные с пользователем:
- COUNT: для `strategy=simple_select` (одна таблица, нет JOIN) **по умолчанию `COUNT(*)`** — PK гарантирует уникальность строк, риска дублей нет. `requires_single_entity_count`-логика сохраняется только для JOIN-стратегий, где реален риск размножения. Override через `user_hints.aggregation_preferences` поддерживает оба направления.
- Patcher: расширить LLM-промпт чтобы он генерировал `operations`, плюс добавить русские regex.
- is_task: убрать жёсткую зависимость от точности `subject`.

---

## Часть 1: COUNT(DISTINCT report_dt) → корректная колонка / COUNT(*)

### 1.0. Главный фикс — для simple_select всегда COUNT(*)

Файл: [core/sql_planner_deterministic.py](core/sql_planner_deterministic.py)

Корень проблемы пользовательского кейса лежит здесь. Для одиночной таблицы с PK `requires_single_entity_count` бессмыслен: дублей нет, COUNT(*) безопасен. DISTINCT-защита нужна только когда есть JOIN.

Шаги:

1. В [build_blueprint](core/sql_planner_deterministic.py#L716) после вычисления `strategy` (строка 723) и до safety-net (строка 745) добавить:
   ```
   skip_single_entity_safety = strategy == "simple_select"
   ```
2. На [строке 745](core/sql_planner_deterministic.py#L745) расширить условие safety-net:
   ```
   if (
       not skip_single_entity_safety
       and str((intent or {}).get("aggregation_hint") or "").lower().strip() == "count"
       and (semantic_frame or {}).get("requires_single_entity_count")
       and schema_loader is not None
       and main_table
   ):
       ... существующий блок ...
   ```
3. В [_compute_aggregation](core/sql_planner_deterministic.py#L383) добавить параметр `strategy: str = ""`. На [строках 405-415](core/sql_planner_deterministic.py#L405-L415) при определении `distinct` пропускать ветку `_count_column_should_be_distinct` если `strategy == "simple_select"`. В fallback-ветке [строка 438](core/sql_planner_deterministic.py#L438) — `COUNT(*)` без изменений (уже корректно).
4. В [_count_column_should_be_distinct](core/sql_planner_deterministic.py#L307) добавить `strategy: str = ""`; на [строке 316](core/sql_planner_deterministic.py#L316) ранний return `False` если `strategy == "simple_select"`. Это страхует на случай когда `_compute_aggregation` вызывается из других мест.
5. В call-site `_compute_aggregation` ([строка 716](core/sql_planner_deterministic.py#L716) и [строка 760](core/sql_planner_deterministic.py#L760)) пробросить `strategy=strategy`.

После этого фикса для `uzp_dwh_fact_outflow` запрос даёт ожидаемый SQL:
```sql
SELECT COUNT(*) AS count_all
FROM s_grnplm_1d_salesntwrk_pcap_sn_uzp.uzp_dwh_fact_outflow
WHERE report_dt >= '2026-02-01'::date
  AND report_dt <  '2026-03-01'::date
  AND is_task = true
```

Изменения 1a/1b/1c ниже остаются актуальными для **JOIN-стратегий** (fact_dim_join, fact_fact_join, dim_fact_join), где safety-net продолжает работать и важен корректный выбор колонки.

### 1a. Сделать `_choose_count_identifier_column` subject-aware и date-averse

Файл: [core/sql_planner_deterministic.py](core/sql_planner_deterministic.py)

**Корневая причина**: функция [_choose_count_identifier_column](core/sql_planner_deterministic.py#L343-L380) не получает `semantic_frame`, не знает про subject и не штрафует date-колонки. У `uzp_dwh_fact_outflow` **составной PK** `(report_dt, gosb_id, inn)` — все три колонки имеют `is_primary_key=True` и одинаково получают `+100`. При прочих равных `report_dt` выигрывает, потому что часто идёт первой в `cols_df` по физическому порядку ИЛИ помечен `semantic_class=identifier` в метаданных (даёт `+60`). То же касается логики [_count_column_should_be_distinct](core/sql_planner_deterministic.py#L307-L340), которая при `requires_single_entity_count=True` безусловно возвращает True и соглашается с выбором любой PK-колонки.

Шаги:

1. Расширить сигнатуру функции на [строке 343](core/sql_planner_deterministic.py#L343): добавить `semantic_frame: dict | None = None, user_input: str = ""`.
2. Внутри функции до цикла построить:
   - `subject_stems` через локальный импорт `core.filter_ranking._subject_alias_stems(subject, schema_loader)` — переиспользуем готовую логику.
   - `query_stems = core.filter_ranking._stem_set(user_input)` — fallback когда subject не в lexicon.
3. В цикле scoring ([строки 360-374](core/sql_planner_deterministic.py#L360-L374)) добавить:
   - **Subject-bonus**: если стеммы `col_name` пересекаются с `subject_stems ∪ query_stems` → `+50`.
   - **Description-bonus**: дополнительно `_text_score(subject, col_name, description)` — до `+14`.
   - **Date-penalty**: если `lower_name.endswith(_DATE_AXIS_SUFFIXES)` или semantic_class в `{date, datetime, timestamp}` → `-80`. Date-колонки никогда не должны выигрывать как entity identifier.
   - **Snapshot-grain caveat**: для grain=snapshot + date-suffix + unique≥95 ещё `-40`.
4. Опустить порог принятия с 50 до 40 на [строке 378](core/sql_planner_deterministic.py#L378), чтобы пройти могла и subject-only колонка.
5. Прокинуть параметры из call-site на [строке 755](core/sql_planner_deterministic.py#L755): `semantic_frame=semantic_frame, user_input=user_input` (оба уже есть в `build_blueprint`).

**Замечание про составной PK**: date-penalty `-80` перевешивает PK-бонус `+100` лишь на `20` баллов, чего мало чтобы потерять колонку целиком. Но `inn`/`gosb_id` с PK-бонусом `+100` + subject-bonus `+50` (через query_stems пересечение) уверенно обгонят `report_dt` с net `+20`. То же правило в [_count_column_should_be_distinct](core/sql_planner_deterministic.py#L307-L340): при составном PK нельзя возвращать True «потому что колонка PK» для date-колонки — добавить там же date-suffix guard (если `lower_name.endswith(_DATE_AXIS_SUFFIXES)` и PK составной — считать как не-identifier).

### 1b. Поддержать override `force_count_star` и `distinct=False`

Файл: тот же.

В блоке override [строки 419-435](core/sql_planner_deterministic.py#L419-L435) и зеркальном fallback [440-454](core/sql_planner_deterministic.py#L440-L454):

- Если `override.get("distinct") is False` (явно False, не falsy) → `result.pop("distinct", None)`.
- Если `override.get("force_count_star") is True` → `result["column"] = "*"; result["alias"] = "count_all"; result.pop("distinct", None)`.

Safety-net на [строках 745-770](core/sql_planner_deterministic.py#L745-L770) обернуть проверкой:
```
if (user_hints or {}).get("aggregation_preferences", {}).get("force_count_star"):
    pass  # дать _compute_aggregation вернуть COUNT(*)
else:
    ... существующий блок ...
```

### 1c. Извлекать override из текста пользователя

Файл: [core/user_hint_extractor.py](core/user_hint_extractor.py)

Добавить regex-паттерны после строки 153:
- `_COUNT_STAR_PATTERNS`: `r"(?:просто|именно)\s+(?:количество\s+)?строк"`, `r"(?:не|без)\s+(?:по\s+)?уникальн\w*"`, `r"\bcount\s*\(\s*\*\s*\)"`, `r"количество\s+записей"`, `r"\bвсе\s+строки\b"`.
- `_NO_DISTINCT_PATTERNS`: `r"\bбез\s+distinct\b"`, `r"не\s+уникальн\w*"`.

В `extract_user_hints` после блока `_COUNT_DISTINCT_PATTERNS` (около строки 562) добавить ветку: если совпал `_COUNT_STAR_PATTERNS` → `aggregation_preferences = {"function":"count","force_count_star":True,"distinct":False}`. Иначе если `_NO_DISTINCT_PATTERNS` → проставить только `distinct=False`.

---

## Часть 2: plan_patcher applied 0 operations

### 2a. Русские regex в `_deterministic_route`

Файл: [graph/nodes/plan_edit.py](graph/nodes/plan_edit.py)

Новые class-level regex после [строки 139](graph/nodes/plan_edit.py#L139):
- `_PATCH_NO_DISTINCT = re.compile(r"(?:без\s+distinct|не\s+(?:считай\s+)?(?:по\s+)?уникальн\w*|не\s+distinct)", re.IGNORECASE)`
- `_PATCH_COUNT_STAR = re.compile(r"(?:просто|именно)?\s*(?:посчита[йи]\w*\s+)?(?:количество\s+)?строк(?:и|у|ой)?\b|count\s*\(\s*\*\s*\)|все\s+строки", re.IGNORECASE)`
- `_PATCH_COUNT_BY_FIELD = re.compile(r"посчита[йи]\w*\s+по\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)`

В `_deterministic_route` после [строки 312](graph/nodes/plan_edit.py#L312) (до date_shift) эмитить операции:
- При `_PATCH_COUNT_STAR` match: `aggregation.column="*"`, `aggregation.distinct=False`, `aggregation.alias="count_all"`.
- При `_PATCH_NO_DISTINCT` match: только `aggregation.distinct=False`.
- При `_PATCH_COUNT_BY_FIELD` match (если колонка в `_collect_known_columns`): `aggregation.function=COUNT`, `aggregation.column=<col>`, `aggregation.distinct=True`.

Конфликт на [строке 237](graph/nodes/plan_edit.py#L237): паттерн `не\s+считать` сейчас гонит правку в `rewrite`. Добавить negative lookahead: `не\s+(?:надо\s+)?считать(?!\s+по\s+уникальн|\s+уникальн|\s+distinct)`. Гарантирует, что «не надо считать по уникальной дате» уйдёт в patch, а не в rewrite.

### 2b. Расширить `_apply_patch_operations`

В [_apply_patch_operations](graph/nodes/plan_edit.py#L396):
- На ветке `aggregation.distinct` (строка 418): когда `value` False — делать `agg.pop("distinct", None)` вместо `agg["distinct"] = False`. Чище и совпадает с тем, что делает `_compute_aggregation`.
- Новая ветка: `path == "aggregation.alias"` → `agg["alias"] = str(value)`.
- На ветке `aggregation.column` (строка 414): если `value == "*"` — также `agg.pop("distinct", None)` (COUNT(DISTINCT *) недопустим).
- После применения изменения column/alias: пройтись по `bp["order_by"]`, заменить старый alias (`count_report_dt`) на новый (`count_all`), иначе `plan_edit_validator` отклонит план.

### 2c. LLM-промпт: вторая фокусированная стадия (без новой ноды)

Если расширять [_fallback_route_with_model](graph/nodes/plan_edit.py#L175-L219) up-front полным описанием схемы операций — промпт растёт более чем на 30%, что нарушает правило CLAUDE.md по контексту GigaChat. Решение: сохранить классификацию kind как 1-й вызов, добавить **2-й короткий LLM-вызов** только когда `kind=="patch"` и `payload.operations` пуст. Между вызовами — Python-логика, что естественно держит ≥5 сек паузу.

В [_fallback_route_with_model](graph/nodes/plan_edit.py#L175) после строки 212:
```
if parsed.get("edit_kind") == "patch" and not (parsed.get("payload") or {}).get("operations"):
    parsed["payload"] = parsed.get("payload") or {}
    parsed["payload"]["operations"] = self._llm_generate_patch_operations(blueprint, edit_text)
```

Новый метод `_llm_generate_patch_operations(blueprint, edit_text)` рядом. System-prompt (~150 токенов) перечисляет допустимые `path` и формат:
```
Ты строишь список операций для правки SQL-плана.
Каждая операция — JSON: {"op":"replace|remove|add|remove_filter","path":"...","value":...}.
Допустимые path:
- "aggregation.function" (SUM/COUNT/AVG/MIN/MAX)
- "aggregation.column" (имя или "*")
- "aggregation.distinct" (true/false)
- "aggregation.alias"
- "order_by.direction" (ASC/DESC)
- "limit" (целое; для remove значение не нужно)
- "group_by" (имя для add, [] для replace)
- "where.date.from" / "where.date.to" (YYYY-MM-DD)
- "remove_filter" с полем "column"
Верни ТОЛЬКО JSON: {"operations":[...]}.
```

User-prompt (~80 токенов) показывает текущий план через `_render_compact_plan(blueprint)` и edit_text + 2 коротких подсказки про COUNT(*) и DISTINCT.

Постобработка: парсить через `_parse_json_response`, отбросить операции с неизвестным `path` или ссылающиеся на колонки вне `_collect_known_columns(state.get("selected_columns"))` — защита от галлюцинаций.

---

## Часть 3: is_task universalization

Файл: [core/filter_ranking.py](core/filter_ranking.py)

### 3a. Снизить порог fuzzy-match для коротких subject'ов

В [_column_looks_like_subject_flag](core/filter_ranking.py#L86-L104) после strip префикса/суффикса:
1. Exact match в `subject_stems` → True.
2. Иначе lowercase-strip английского `s`/`es` (для `is_tasks`, `task_flgs`) и retry.
3. Иначе для `len>=4` использовать ratio `0.8` (текущий порог требует `len>=5`, ratio `0.85`).

### 3b. Расширить subject-fallback

В [_collect_implicit_subject_flag_requests](core/filter_ranking.py#L272) добавить параметр `query_stems: set[str] = ()`. Build `relevant_stems = subject_stems | query_stems` и использовать его в `_column_looks_like_subject_flag`.

В [rank_filter_candidates](core/filter_ranking.py#L335) (call-site) построить `query_stems = _stem_set(user_input)` и передать.

В [_subject_alias_stems](core/filter_ranking.py#L69): когда lexicon не содержит subject, добавить fallback через built-in alias map (`task↔задач`, `client↔клиент`, `employee↔сотрудник`, `organization↔организац`). Чтобы не дублировать с [semantic_frame.py:328-335](core/semantic_frame.py#L328-L335), вынести этот словарь в новый модуль `core/semantic_registry.py` (тонкая утилита `builtin_subject_aliases() -> dict[str,list[str]]`) и импортировать в обоих местах.

Это устраняет hard-coded зависимость от точности `subject`: даже если `business_event=outflow` забивает `subject`, флаг `is_task` подсосётся через `query_stems` (содержит «задач»).

---

## Тесты

`tests/test_sql_planner_deterministic.py` (новые):
- `test_simple_select_count_skips_single_entity_safety` — одна snapshot-таблица с составным PK `(report_dt, gosb_id, inn)`, intent=count, semantic_frame.requires_single_entity_count=True. Ожидаем агрегацию `COUNT(*)`, safety-net не сработал.
- `test_simple_select_no_distinct_even_when_aggregate_role_set` — одна таблица с PK, agg-роль уже содержит колонку. Ожидаем `distinct` отсутствует в результате.
- `test_join_strategy_still_uses_distinct` — fact_dim_join + requires_single_entity_count=True → safety-net срабатывает и DISTINCT добавляется.
- `test_choose_count_identifier_avoids_date_in_composite_pk` — фикстура с составным PK `(report_dt, gosb_id, inn)`, fact_dim_join, ожидаем выбор `gosb_id` или `inn`, НЕ `report_dt`.
- `test_choose_count_identifier_uses_subject_stems` — `task_code` без PK выигрывает у `inn` при subject="task" в JOIN-стратегии.
- `test_force_count_star_disables_safety_net` — user_hints с force_count_star отключает entity-substitution в JOIN-стратегии.
- `test_aggregation_preferences_distinct_false_drops_flag` — `distinct=False` убирает ключ.

`tests/test_user_hint_extractor.py` (новые):
- `test_extract_force_count_star_for_prosto_stroki` — «посчитай просто количество строк».
- `test_extract_no_distinct_phrase` — «не надо считать по уникальной дате».

`tests/test_plan_edit.py` (новые):
- `test_router_patch_count_star_russian` — обе пользовательские формулировки → правильные ops.
- `test_router_patch_no_distinct_only` — только distinct=False.
- `test_patcher_count_star_strips_distinct_and_updates_order_by` — order_by alias обновляется.
- `test_router_count_by_field` — «посчитай по task_code».
- `test_fallback_llm_emits_operations` — monkeypatch `LLM.invoke_with_system`, проверить что 2-й вызов делается и ops пробрасываются.

`tests/test_filter_ranking.py` (новые):
- `test_implicit_flag_found_when_subject_misclassified_to_outflow` — subject="outflow", entities=["задача"], user_input="Сколько задач по оттоку" → is_task всё равно подсасывается.
- `test_column_looks_like_subject_flag_handles_english_plural` — `is_tasks` → True.
- `test_subject_alias_stems_falls_back_to_builtin_map` — без lexicon всё равно даёт стеммы.

Запуск: `.venv/bin/pytest tests/test_filter_ranking.py tests/test_sql_planner_deterministic.py tests/test_plan_edit.py tests/test_user_hint_extractor.py tests/test_where_resolver.py` — все green. Также прогнать `tests/test_golden.py` (имеется baseline 7 pre-existing failures — новых не должно появиться).

---

## Verification (end-to-end через agent.ipynb)

1. **Bug 1 (simple_select)**: запрос «Сколько задач по фактическому оттоку поставили в феврале 26».
   - Ожидаемый SQL:
     ```sql
     SELECT COUNT(*) AS count_all
     FROM s_grnplm_1d_salesntwrk_pcap_sn_uzp.uzp_dwh_fact_outflow
     WHERE report_dt >= '2026-02-01'::date
       AND report_dt <  '2026-03-01'::date
       AND is_task = true
     ```
   - В логе `DeterministicPlanner: single-entity safety net` НЕ должен сработать (skip_single_entity_safety=True для simple_select).
   - Фильтр `is_task = true` сохраняется (Bug 3 уже работает).

1a. **Bug 1 (JOIN-сценарий, регрессионно)**: запрос с JOIN, например «Сколько уникальных клиентов с задачами по оттоку в феврале по отделению X» (форсирует fact_dim_join).
   - Ожидаемая агрегация: `COUNT(DISTINCT inn)` или `COUNT(DISTINCT gosb_id)` — НЕ `report_dt`.
   - В логе `DeterministicPlanner: single-entity safety net → ...uzp_dwh_fact_outflow.<inn|gosb_id>` (date-колонка отбита штрафом).

2. **Bug 2 (1-я правка)**: «Посчитай просто количество строк с признаком задачи true».
   - Лог: `plan_edit_router: kind=patch confidence=0.95` (deterministic path, без LLM).
   - Лог: `plan_patcher: applied 3 operations`.
   - Новая агрегация: `COUNT(*) AS count_all`, фильтр `is_task=true` остаётся.

3. **Bug 2 (2-я правка)**: «Не надо считать по уникальной дате, просто посчитай количество строк».
   - Аналогично: 3 операции, COUNT(*).

4. **Bug 3**: запрос «Сколько задач по оттоку» (искусственно создаём ambiguity).
   - В логе `filter_ranking` видна заявка `implicit_subject_flag:...is_task` даже если `semantic_frame.subject == "outflow"`.

5. **No-regression**: pytest полный, golden-tests без новых падений сверх baseline.

---

## Sequencing

1. **Bug 3** (filter_ranking + semantic_registry) — независимо, наименьший риск, нужен для корректных фикстур bug-1 тестов.
2. **Bug 1** (planner + hint_extractor) — зависит от стабильного flag-detection.
3. **Bug 2** (plan_edit) — после bug 1, чтобы новые ops применялись к корректному стартовому плану.

Никаких новых LangGraph-нод. Один новый LLM-вызов (2-я стадия в plan_edit fallback) — срабатывает только когда первая стадия вернула пустые operations, разделён Python-работой ≥5 сек от первого вызова.

## Critical files

- [core/sql_planner_deterministic.py](core/sql_planner_deterministic.py)
- [core/user_hint_extractor.py](core/user_hint_extractor.py)
- [core/filter_ranking.py](core/filter_ranking.py)
- [core/semantic_registry.py](core/semantic_registry.py) — новый модуль для built-in aliases
- [graph/nodes/plan_edit.py](graph/nodes/plan_edit.py)
- [tests/test_sql_planner_deterministic.py](tests/test_sql_planner_deterministic.py)
- [tests/test_user_hint_extractor.py](tests/test_user_hint_extractor.py)
- [tests/test_plan_edit.py](tests/test_plan_edit.py)
- [tests/test_filter_ranking.py](tests/test_filter_ranking.py)
