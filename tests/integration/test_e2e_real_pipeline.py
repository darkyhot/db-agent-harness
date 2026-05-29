"""E2E-тесты пайплайна на реальной БД + реальном GigaChat.

Запускают аналитический подграф (build_analytics_subgraph) поверх тестового
Postgres и реального GigaChat. Каждый YAML-кейс из tests/integration/cases/
проверяет:
  - intent классифицирован как ожидаемый;
  - финальный SQL содержит/не содержит требуемые подстроки;
  - граф дошёл до sql_validator (= реально выполнил SQL);
  - возвращённый rowcount укладывается в expected_rowcount.

Запуск:
    docker compose -f tests/integration/docker-compose.test.yml up -d
    pytest tests/integration/test_e2e_real_pipeline.py -m integration -v
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

CASES_DIR = Path(__file__).resolve().parent / "cases"


def _load_cases() -> list[tuple[str, dict]]:
    if not yaml or not CASES_DIR.exists():
        return []
    out: list[tuple[str, dict]] = []
    for p in sorted(CASES_DIR.glob("*.yaml")):
        with p.open(encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if isinstance(doc, dict) and "query" in doc:
            out.append((p.stem, doc))
    return out


def _has_cyrillic_alias(sql: str) -> bool:
    pattern = re.compile(
        r'\bAS\s+(?:"[^"]*[а-яёА-ЯЁ][^"]*"|[а-яёА-ЯЁ]\w*)',
        re.IGNORECASE,
    )
    return bool(pattern.search(sql))


def _check_substrings(sql: str, required: list[str], forbidden: list[str]) -> list[str]:
    sql_upper = sql.upper()
    errors: list[str] = []
    for needle in required or []:
        if needle.upper() not in sql_upper:
            errors.append(f"SQL не содержит '{needle}'")
    for needle in forbidden or []:
        if needle.upper() in sql_upper:
            errors.append(f"SQL содержит запрещённое '{needle}'")
    return errors


@pytest.mark.skipif(yaml is None, reason="pyyaml не установлен")
@pytest.mark.parametrize(
    "case_name, case",
    _load_cases(),
    ids=[name for name, _ in _load_cases()] or ["<no-cases>"],
)
def test_e2e_real_pipeline(
    case_name: str,
    case: dict,
    analytics_graph,
    seeded_db,
):
    """Один e2e-кейс: реальный граф + реальный GigaChat + реальный Postgres."""
    from graph.graph import create_initial_state

    plan_context = case.get("plan_context") or {}
    # YAML отдаёт selected_tables списком списков, конвертируем в кортежи.
    if isinstance(plan_context.get("selected_tables"), list):
        plan_context["selected_tables"] = [
            tuple(t) if isinstance(t, list) else t
            for t in plan_context["selected_tables"]
        ]

    # Кейс может закрепить QuerySpec (блок `query_spec:`), чтобы прогнать
    # детерминированный pipeline (grounder→planner→SQL-gen) без недетерминизма
    # LLM-интерпретатора. query_interpreter подхватывает pinned_query_spec и
    # пропускает вызов LLM.
    pinned = case.get("query_spec")
    if isinstance(pinned, dict):
        plan_context = dict(plan_context)
        plan_context["pinned_query_spec"] = pinned

    state = create_initial_state(
        case["query"],
        plan_preview_approved=True,  # skip plan-preview pause
        plan_context=plan_context or None,
    )
    visited: list[str] = []
    result: dict[str, Any] = dict(state)
    for event in analytics_graph.stream(state):
        node_name = list(event.keys())[0]
        visited.append(node_name)
        payload = event.get(node_name)
        if isinstance(payload, dict):
            result.update(payload)

    # ---- ожидаем clarification/disambiguation: SQL может отсутствовать ----
    if case.get("expect_disambiguation"):
        # Реальный пайплайн может предложить выбор двумя путями:
        # 1) structured disambiguation_options (needs_disambiguation=True);
        # 2) свободный clarification_message (needs_clarification=True).
        # Принимаем оба, главное — обе таблицы должны фигурировать.
        required = case.get("disambiguation_tables_must_include", []) or []
        opts = result.get("disambiguation_options", []) or []
        opt_names = {f"{o.get('schema')}.{o.get('table')}" for o in opts}
        structured_ok = (
            bool(result.get("needs_disambiguation"))
            and all(t in opt_names for t in required)
        )
        msg = (result.get("clarification_message")
               or result.get("confirmation_message") or "")
        msg_lower = msg.lower()
        clarification_ok = (
            bool(result.get("needs_clarification"))
            and all(t.lower() in msg_lower for t in required)
        )
        assert structured_ok or clarification_ok, (
            f"[{case_name}] ожидался выбор между {required}; "
            f"needs_disambiguation={result.get('needs_disambiguation')!r}, "
            f"options={sorted(opt_names)}, "
            f"needs_clarification={result.get('needs_clarification')!r}, "
            f"clarification_message={msg!r}, visited={visited}"
        )
        return  # SQL не ожидается на этом turn-е

    if case.get("expect_clarification"):
        assert result.get("needs_clarification"), (
            f"[{case_name}] ожидался needs_clarification=True; "
            f"got={result.get('needs_clarification')!r}, visited={visited}"
        )
        required_substrings = case.get("clarification_must_contain", []) or []
        msg = (result.get("clarification_message")
               or result.get("confirmation_message") or "")
        msg_lower = msg.lower()
        missing = [s for s in required_substrings if s.lower() not in msg_lower]
        assert not missing, (
            f"[{case_name}] clarification_message не содержит {missing}; "
            f"actual={msg!r}"
        )
        return

    # ---- intent ----
    expected_intent = case.get("intent_must_be")
    if expected_intent:
        intent = result.get("intent", {}) or {}
        actual = intent.get("intent")
        assert actual == expected_intent, (
            f"[{case_name}] intent={actual!r}, ожидалось {expected_intent!r}; "
            f"visited={visited}"
        )

    # ---- SQL substrings ----
    # sql_validator после успеха обнуляет sql_to_validate, поэтому ищем
    # последний execute_query в tool_calls — там SQL, который реально
    # пошёл в БД (логика та же, что в cli/interface.py).
    sql = (
        result.get("sql_preview")
        or result.get("sql_to_validate")
        or ""
    )
    if not sql:
        for tc in reversed(result.get("tool_calls", []) or []):
            if tc.get("tool") in ("execute_query", "execute_write"):
                sql = (tc.get("args") or {}).get("sql") or ""
                if sql:
                    break
    if not sql:
        # final_answer был сформирован без SQL (schema-вопрос и т.п.)
        pass
    else:
        errors = _check_substrings(
            sql,
            case.get("sql_must_contain", []) or [],
            case.get("sql_must_not_contain", []) or [],
        )
        if case.get("sql_alias_no_cyrillic") and _has_cyrillic_alias(sql):
            errors.append("SQL содержит кириллический алиас")
        assert not errors, (
            f"[{case_name}] SQL не прошёл проверки:\n  "
            + "\n  ".join(errors)
            + f"\nSQL:\n{sql}"
        )

    # ---- маршрут ----
    must_visit = case.get("should_reach_node")
    if must_visit:
        assert must_visit in visited, (
            f"[{case_name}] узел {must_visit!r} не посещён; visited={visited}"
        )
    must_not_visit = case.get("should_not_reach_node")
    if must_not_visit:
        assert must_not_visit not in visited, (
            f"[{case_name}] узел {must_not_visit!r} был посещён; visited={visited}"
        )

    # ---- retries ----
    max_retries = case.get("max_retries")
    if max_retries is not None:
        retries = result.get("total_retry_count", result.get("retry_count", 0))
        assert retries <= max_retries, (
            f"[{case_name}] retry_count={retries} > max {max_retries}"
        )

    # ---- реальное выполнение SQL и rowcount ----
    if case.get("execute_succeeds"):
        assert sql, f"[{case_name}] execute_succeeds=true, но SQL отсутствует"
        # Используем тот же DatabaseManager для подсчёта строк — он применит авто-LIMIT.
        df = seeded_db.preview_query(sql, limit=10_000)
        rc = case.get("expected_rowcount") or {}
        lo = rc.get("min")
        hi = rc.get("max")
        if lo is not None:
            assert len(df) >= lo, (
                f"[{case_name}] rows={len(df)} < expected min {lo}; SQL:\n{sql}"
            )
        if hi is not None:
            assert len(df) <= hi, (
                f"[{case_name}] rows={len(df)} > expected max {hi}; SQL:\n{sql}"
            )

    # ---- last_error не остался висеть ----
    err = result.get("last_error")
    if err and not case.get("expect_error"):
        pytest.fail(f"[{case_name}] last_error={err!r}; visited={visited}")
