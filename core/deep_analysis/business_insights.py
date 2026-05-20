"""Business-insight stage: distill findings into 0-5 actionable takeaways.

Runs after all metric runners produce findings. The mechanical TL;DR in
``report.py`` already ranks findings by severity × impact score, but it does
not explain what those findings mean for the business. This module asks the
LLM to pick at most ``max_insights`` items that are worth a stakeholder's
attention and to phrase each as: куда смотреть → на что это влияет → что
сделать.

Failure is non-fatal: on any error the function returns ``[]`` and the
report falls back to its existing TL;DR + per-theme structure.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.deep_analysis.hypothesis_llm import _parse_llm_json
from core.deep_analysis.logging_setup import get_logger
from core.deep_analysis.report import _impact_score
from core.deep_analysis.types import BusinessInsight, Finding, TableAnalysisContext, TableProfile
from core.llm import RateLimitedLLM

# How many top findings we hand to the LLM. Larger = better recall but more
# tokens per call; RateLimitedLLM has no prompt caching, so we keep it tight.
_MAX_FINDINGS_TO_LLM = 20

# Severities that are eligible to feed the LLM. ``info`` is excluded — those
# are noise that shouldn't drive business actions.
_ELIGIBLE_SEVERITIES = {"critical", "strong", "notable"}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


_SYSTEM_PROMPT = """Ты — старший бизнес-аналитик банка. Тебе дают бизнес-контекст таблицы, (опц.) гипотезу пользователя и список находок (severity, заголовок, summary, метрики, runner, примеры entity-карточек). Твой читатель — руководитель/владелец процесса, НЕ инженер. Технический разбор таблицы его не интересует — только что это значит для бизнеса и что делать.

Задача — выделить НЕ БОЛЕЕ {max_insights} бизнес-инсайтов (лучше 2 сильных, чем 5 натянутых).

Каждый инсайт отвечает на ТРИ вопроса ПРОСТЫМ деловым языком:
1. **where_to_look** — конкретный объект: сегмент/период/подразделение/файл нарушителей (с числом из находки: "8 сотрудников", "отдел X", "последние 10 дней квартала").
2. **business_impact** — деньги/риск/клиенты/операционка. НЕ пересказывай summary — назови бизнес-следствие.
3. **recommended_action** — конкретный шаг и кому отдать (HR-аудит, фрод-команда, владелец витрины).

ПЕРЕВОДИ МЕТРИКИ, А НЕ ПОВТОРЯЙ ИХ. Глоссарий (эти слова в ответе НЕ употреблять):
- robust_z / max_abs_z / z — насколько сильно объект выбивается из своей группы ("резко выше остальных").
- cramer_v / eta_sq / spearman_rho — насколько жёстко одно предопределяет другое ("значение почти всегда определяет ...").
- p_value / chi2 / kw_stat / f_stat — статистическая надёжность; пользователю НЕ озвучивать вообще.
- rel_deviation_pct / max_rel_deviation_pct / rel_shift_pct — отклонение от нормы в процентах ("на 18% выше обычного").
- n_violators / n_outliers — сколько объектов аномальны ("у 8 сотрудников").
- changepoint / regime shift — момент, когда поведение скачком изменилось.

ЗАПРЕЩЕНО (это пустые шаблоны — за них штраф):
- "нужно провести аналитику / разобраться / понять что привело к этому"
- "плохое распределение данных", "данные требуют внимания", "возможны аномалии"
- любой инсайт без конкретного объекта ИЛИ без числа из находки ИЛИ без адресата действия.

НЕ ВЫДУМЫВАЙ числа, даты, проценты и имена. Используй ТОЛЬКО значения, присутствующие в переданных находках. Если в находке нет числа — описывай качественно, без цифр.
Если семантика таблицы = "(нет)" — выведи домен из имён таблицы/колонок в находках и опирайся на него.
Игнорируй info-severity и слабые статистические связи без бизнес-смысла. Дубли по одному явлению — объедини, перечислив все id в related_finding_ids.

Пример ПЛОХО: "Обнаружено плохое распределение по отделам, нужно провести аналитику и понять причины."
Пример ХОРОШО: "8 сотрудников отдела продаж оформляют отпуск строго в последние 3 дня квартала — типичный признак подгонки KPI; передать список (entities_*.csv) в HR-аудит для сверки с фактической явкой."

Верни СТРОГО JSON без пояснений:
{{
  "insights": [
    {{
      "title": "короткий заголовок одной строкой, без жаргона",
      "priority": "top|high|medium",
      "where_to_look": "...",
      "business_impact": "1-3 предложения, бизнес-следствие",
      "recommended_action": "конкретное действие + кому отдать",
      "related_finding_ids": ["hypothesis_id1", "hypothesis_id2"],
      "confidence": "high|medium|low"
    }}
  ]
}}

Если ничего не достойно бизнес-внимания — верни {{"insights": []}}.
"""


def _slugify(text: str, fallback: str) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return s[:48] or fallback


def _select_top_findings(findings: list[Finding]) -> list[Finding]:
    eligible = [f for f in findings if f.severity in _ELIGIBLE_SEVERITIES]
    if not eligible:
        return []
    severity_order = {"critical": 0, "strong": 1, "notable": 2}
    eligible.sort(
        key=lambda f: (severity_order.get(f.severity, 99), -_impact_score(f))
    )
    return eligible[:_MAX_FINDINGS_TO_LLM]


def _serialize_findings(findings: list[Finding]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in findings:
        # Keep metrics compact — drop heavy nested objects, format floats.
        compact_metrics: dict[str, Any] = {}
        for k, v in (f.metrics or {}).items():
            if isinstance(v, float):
                compact_metrics[k] = round(v, 4)
            elif isinstance(v, (int, str, bool)) or v is None:
                compact_metrics[k] = v
        out.append({
            "hypothesis_id": f.hypothesis_id,
            "runner": f.runner,
            "severity": f.severity,
            "title": f.title,
            "summary": f.summary,
            "metrics": compact_metrics,
            "entity_csv": f.entity_csv,
            "examples": (f.details or {}).get("examples", [])[:5],
        })
    return out


def _validate_insight(
    raw: dict[str, Any],
    idx: int,
    valid_finding_ids: set[str],
) -> BusinessInsight | None:
    title = str(raw.get("title") or "").strip()
    if not title:
        return None
    priority = str(raw.get("priority") or "medium").lower().strip()
    if priority not in {"top", "high", "medium"}:
        priority = "medium"
    confidence = str(raw.get("confidence") or "medium").lower().strip()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    where_to_look = str(raw.get("where_to_look") or "").strip()
    business_impact = str(raw.get("business_impact") or "").strip()
    recommended_action = str(raw.get("recommended_action") or "").strip()
    if not (where_to_look and business_impact and recommended_action):
        return None
    related_raw = raw.get("related_finding_ids") or []
    if not isinstance(related_raw, list):
        related_raw = []
    related = [
        str(x) for x in related_raw
        if isinstance(x, str) and x in valid_finding_ids
    ]
    insight_id = _slugify(title, fallback=f"insight-{idx + 1}")
    return BusinessInsight(
        insight_id=insight_id,
        title=title,
        priority=priority,
        where_to_look=where_to_look,
        business_impact=business_impact,
        recommended_action=recommended_action,
        related_finding_ids=related,
        confidence=confidence,
    )


def _dedupe_ids(insights: list[BusinessInsight]) -> list[BusinessInsight]:
    seen: set[str] = set()
    for ins in insights:
        base = ins.insight_id
        candidate = base
        i = 2
        while candidate in seen:
            candidate = f"{base}-{i}"
            i += 1
        ins.insight_id = candidate
        seen.add(candidate)
    return insights


def extract_business_insights(
    llm: RateLimitedLLM,
    findings: list[Finding],
    profile: TableProfile,
    user_hypothesis_text: str | None = None,
    table_semantics: str | None = None,
    analysis_context: TableAnalysisContext | None = None,
    max_insights: int = 5,
) -> list[BusinessInsight]:
    """Distill findings into <= max_insights actionable business takeaways.

    Returns ``[]`` if there are no eligible findings, the LLM call fails, or
    the response can't be parsed — caller should treat empty as "no business
    insights this run" and continue building the report.
    """
    log = get_logger()
    top_findings = _select_top_findings(findings)
    if not top_findings:
        log.info("Business insights: no findings with severity >= notable, skipping LLM")
        return []

    valid_ids = {f.hypothesis_id for f in top_findings}
    user = (
        f"Таблица: {profile.schema}.{profile.table}\n"
        f"Семантика: {table_semantics or '(нет)'}\n"
        f"Бизнес-контекст:\n"
        f"{analysis_context.short_brief() if analysis_context else '(нет)'}\n"
        f"Гипотеза пользователя (фокус релевантности): {user_hypothesis_text or '(не задана)'}\n\n"
        f"Находки (top-{len(top_findings)} по severity и impact):\n"
        f"{json.dumps(_serialize_findings(top_findings), ensure_ascii=False, indent=2)}"
    )

    try:
        response = llm.invoke_with_system(
            _SYSTEM_PROMPT.format(max_insights=max_insights),
            user,
            temperature=0.2,
        )
    except Exception as exc:
        log.warning("Business insights LLM call failed: %s", exc)
        return []

    try:
        data = _parse_llm_json(response)
    except Exception as exc:
        log.warning(
            "Business insights JSON parse failed: %s. Raw head: %s",
            exc, response[:300],
        )
        return []

    raw_list = data.get("insights") or []
    if not isinstance(raw_list, list):
        log.warning("Business insights: 'insights' is not a list, got %r", type(raw_list))
        return []

    insights: list[BusinessInsight] = []
    for i, raw in enumerate(raw_list[:max_insights]):
        if not isinstance(raw, dict):
            continue
        ins = _validate_insight(raw, i, valid_ids)
        if ins is not None:
            insights.append(ins)

    insights = _dedupe_ids(insights)
    log.info("Business insights: produced %d / %d", len(insights), max_insights)
    return insights
