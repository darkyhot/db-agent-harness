"""Translate a free-form user hypothesis into a formal check plan.

The user writes something like "Сотрудники под конец квартала берут отпуск и
при этом работают чтобы занизить план". This module asks the LLM to map that
onto one of the existing runners with concrete column references, and produces
a human-readable plan for plan-preview approval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from core.deep_analysis.hypothesis_llm import _RUNNER_SCHEMAS, _format_runner_schemas, _parse_llm_json, _validate_hypothesis_dict
from core.deep_analysis.logging_setup import get_logger
from core.deep_analysis.profiler import profile_to_brief
from core.deep_analysis.types import HypothesisSpec, TableProfile
from core.llm import RateLimitedLLM


@dataclass
class UserHypothesisPlan:
    hypothesis: HypothesisSpec
    plan_text: str              # markdown-formatted, shown to user for approval


_SYSTEM_PROMPT = """Ты — аналитик. Пользователь сформулировал гипотезу о данных. Твоя задача — превратить её в ФОРМАЛЬНЫЙ план проверки, который исполнит один из доступных runners.

Доступные runners:
%s

Верни СТРОГО JSON без пояснений:
{
  "title": "короткое название",
  "rationale": "переформулировка гипотезы в 1-2 предложения",
  "runner": "seasonality|group_anomalies|outliers",
  "params": { ... },
  "priority": 1.0,
  "plan_explanation": "человекочитаемое объяснение, что именно будет посчитано: какие колонки, какие разрезы, какие метрики, какой критерий аномалии. 3-6 строк."
}

Правила:
- Используй только колонки из профиля таблицы.
- Приоритет всегда 1.0 (пользовательская гипотеза идёт первой).
- plan_explanation должен быть конкретным: "группировка по employee_id и quarter(date), метрика — доля type='отпуск', порог z-score > 2".
"""


def build_user_hypothesis_plan(
    llm: RateLimitedLLM,
    user_text: str,
    profile: TableProfile,
    table_semantics: str = "",
) -> UserHypothesisPlan | None:
    """Ask the LLM to translate user_text into a HypothesisSpec + a plan doc.

    Returns None on parse/validation failure so the CLI can ask the user to
    rephrase rather than run a bad plan.
    """
    log = get_logger()
    system = _SYSTEM_PROMPT % _format_runner_schemas()
    user = (
        f"{profile_to_brief(profile)}\n\n"
        f"Семантика таблицы: {table_semantics or '(нет)'}\n\n"
        f"Гипотеза пользователя:\n{user_text}"
    )
    try:
        response = llm.invoke_with_system(system, user, temperature=0.1)
    except Exception as exc:
        log.warning("User-hypothesis LLM call failed: %s", exc)
        return None

    try:
        data = _parse_llm_json(response)
    except Exception as exc:
        log.warning("User-hypothesis parse failed: %s. Raw head: %s", exc, response[:300])
        return None

    spec = _validate_hypothesis_dict(data, profile, idx=0)
    if spec is None:
        log.warning("User hypothesis validation failed: %s", response[:300])
        return None
    spec.hypothesis_id = "user_0_" + spec.runner
    spec.source = "user"
    spec.priority = 1.0

    plan_explanation = str(data.get("plan_explanation") or "").strip()
    plan_text = _format_plan_preview(spec, plan_explanation, user_text)
    return UserHypothesisPlan(hypothesis=spec, plan_text=plan_text)


def _format_plan_preview(spec: HypothesisSpec, explanation: str, user_text: str) -> str:
    params_lines = "\n".join(f"  - {k}: {v}" for k, v in spec.params.items())
    return (
        "# План проверки вашей гипотезы\n\n"
        f"**Гипотеза пользователя:** {user_text}\n\n"
        f"**Название:** {spec.title}\n\n"
        f"**Переформулировка:** {spec.rationale}\n\n"
        f"**Runner:** `{spec.runner}`\n\n"
        f"**Параметры:**\n{params_lines}\n\n"
        f"**Что будет посчитано:**\n{explanation or '(LLM не вернула объяснения)'}\n\n"
        "Введите `ок` чтобы запустить, или опишите правки свободным текстом."
    )


def apply_user_plan_edit(
    llm: RateLimitedLLM,
    original: UserHypothesisPlan,
    user_feedback: str,
    profile: TableProfile,
) -> UserHypothesisPlan | None:
    """Re-derive the plan after the user edits it.

    We simply re-run build_user_hypothesis_plan with the combined context so
    the LLM can honor both the original intent and the feedback.
    """
    combined = (
        f"Исходная гипотеза: {original.hypothesis.rationale}\n"
        f"Предыдущий план:\n{original.plan_text}\n"
        f"Правка пользователя: {user_feedback}"
    )
    return build_user_hypothesis_plan(llm, combined, profile)
