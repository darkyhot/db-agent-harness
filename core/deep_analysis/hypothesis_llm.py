"""LLM layer: produce domain-specific hypotheses and rank the combined pool.

The LLM is not allowed to invent runners — it must map each suggestion to an
existing runner key and the parameter shape it accepts. Output is JSON,
validated and stripped of anything referring to columns that don't exist.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.deep_analysis.logging_setup import get_logger
from core.deep_analysis.profiler import profile_to_brief
from core.deep_analysis.types import HypothesisSpec, TableProfile
from core.llm import RateLimitedLLM

_RUNNER_SCHEMAS = {
    "seasonality": {
        "required": ["date_col"],
        "optional": ["value_col", "agg", "group_col"],
        "description": "Периодичность метрики по времени (день недели / месяц / квартал / бизнес-календарь РФ).",
    },
    "group_anomalies": {
        "required": ["entity_col", "metric"],
        "optional": ["date_col", "value_col", "category_col", "period"],
        "description": (
            "Массовые отклонения на уровне сущности (сотрудник/клиент): возвращает CSV нарушителей. "
            "metric = row_count | mean | rate | end_of_quarter_shift."
        ),
    },
    "outliers": {
        "required": ["value_cols"],
        "optional": ["method"],
        "description": "Выбросы в числовых колонках (robust-z/MAD или IsolationForest).",
    },
    "dependencies": {
        "required": ["columns"],
        "optional": ["max_pairs"],
        "description": (
            "Попарные зависимости: Spearman для числовых, Cramér's V для категорий, η² для смешанных. "
            "Используй, когда нужно найти скрытые связи между произвольным набором колонок."
        ),
    },
    "regime_shifts": {
        "required": ["date_col"],
        "optional": ["value_col", "agg", "freq"],
        "description": (
            "Точки смены режима временного ряда (ruptures PELT). "
            "Ищет даты, в которых поведение метрики скачкообразно изменилось. "
            "freq = day | week | month."
        ),
    },
}


_SYSTEM_PROMPT = """Ты — аналитический ассистент для поиска закономерностей в банковских данных.
На вход ты получаешь профиль таблицы и список уже сформированных гипотез от каталога.
Твоя задача:
1. Предложить ДОПОЛНИТЕЛЬНЫЕ бизнес-значимые гипотезы, которые каталог упустил из-за отсутствия знаний о домене (банк, сотрудники, клиенты, платежи, отпуска, kpi, фрод).
2. Отранжировать ВСЕ гипотезы (каталог + твои) по приоритету с т.з. бизнес-ценности.

Доступные runners и их параметры:
%s

Строгие правила:
- Каждая гипотеза ссылается только на реально существующие колонки из профиля.
- Используй ТОЛЬКО runners и параметры из списка выше. Не придумывай новых.
- priority — число от 0 до 1, 1 = максимум.
- Не дублируй существующие гипотезы каталога (сравнивай по смыслу, а не по id).

Верни СТРОГО JSON без пояснений:
{
  "new_hypotheses": [
    {
      "title": "короткое название",
      "rationale": "зачем это проверять бизнесу (1-2 предложения)",
      "runner": "seasonality|group_anomalies|outliers",
      "params": { ... },
      "priority": 0.0..1.0
    }
  ],
  "priority_overrides": {
    "<hypothesis_id>": 0.0..1.0
  }
}
"""


def _format_runner_schemas() -> str:
    lines = []
    for runner, spec in _RUNNER_SCHEMAS.items():
        lines.append(
            f"- {runner}: {spec['description']}\n"
            f"    required: {spec['required']}\n"
            f"    optional: {spec['optional']}"
        )
    return "\n".join(lines)


def _catalog_brief(catalog: list[HypothesisSpec]) -> str:
    lines = []
    for h in catalog:
        lines.append(f"- id={h.hypothesis_id} | runner={h.runner} | p={h.priority:.2f} | {h.title}")
    return "\n".join(lines) if lines else "(каталог пуст)"


def _parse_llm_json(response: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", response)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("LLM response has no JSON object")
    return json.loads(match.group())


def _validate_hypothesis_dict(
    raw: dict[str, Any],
    profile: TableProfile,
    idx: int,
) -> HypothesisSpec | None:
    runner = raw.get("runner")
    if runner not in _RUNNER_SCHEMAS:
        return None
    schema = _RUNNER_SCHEMAS[runner]
    params = raw.get("params") or {}
    if not isinstance(params, dict):
        return None
    for key in schema["required"]:
        if key not in params:
            return None

    def _cols_exist(value) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value in profile.columns
        if isinstance(value, list):
            return all(isinstance(v, str) and v in profile.columns for v in value)
        return True

    # Any column-shaped param must reference a real column.
    for col_like in ("date_col", "value_col", "entity_col", "group_col", "category_col"):
        if col_like in params and not _cols_exist(params[col_like]):
            return None
    if "value_cols" in params and not _cols_exist(params["value_cols"]):
        return None

    title = str(raw.get("title") or f"LLM гипотеза {idx + 1}").strip()
    rationale = str(raw.get("rationale") or "").strip()
    try:
        priority = float(raw.get("priority", 0.5))
    except (TypeError, ValueError):
        priority = 0.5
    priority = max(0.0, min(1.0, priority))

    return HypothesisSpec(
        hypothesis_id=f"llm_{idx}_{runner}",
        runner=runner,
        title=title,
        rationale=rationale,
        params=params,
        priority=priority,
        source="llm",
        est_cost_seconds=20.0,
    )


def enrich_hypotheses(
    llm: RateLimitedLLM,
    profile: TableProfile,
    catalog: list[HypothesisSpec],
    table_semantics: str = "",
) -> list[HypothesisSpec]:
    """Ask the LLM for domain-specific additions and priority overrides.

    Failure is non-fatal: on any error the original catalog is returned intact.
    """
    log = get_logger()
    system = _SYSTEM_PROMPT % _format_runner_schemas()
    user = (
        f"{profile_to_brief(profile)}\n\n"
        f"Семантика таблицы (из корпоративного каталога):\n{table_semantics or '(нет)'}\n\n"
        f"Уже сформированные гипотезы:\n{_catalog_brief(catalog)}"
    )
    try:
        response = llm.invoke_with_system(system, user, temperature=0.2)
    except Exception as exc:
        log.warning("LLM enrichment failed: %s", exc)
        return catalog

    try:
        data = _parse_llm_json(response)
    except Exception as exc:
        log.warning("LLM response parse failed: %s. Raw head: %s", exc, response[:300])
        return catalog

    merged = list(catalog)
    new_raw = data.get("new_hypotheses") or []
    if isinstance(new_raw, list):
        for i, raw in enumerate(new_raw):
            if not isinstance(raw, dict):
                continue
            spec = _validate_hypothesis_dict(raw, profile, i)
            if spec is not None:
                merged.append(spec)

    overrides = data.get("priority_overrides") or {}
    if isinstance(overrides, dict):
        for h in merged:
            if h.hypothesis_id in overrides:
                try:
                    h.priority = max(0.0, min(1.0, float(overrides[h.hypothesis_id])))
                except (TypeError, ValueError):
                    pass

    log.info(
        "LLM enrichment: +%d new hypotheses, %d priority overrides",
        len(merged) - len(catalog),
        len(overrides) if isinstance(overrides, dict) else 0,
    )
    return merged
