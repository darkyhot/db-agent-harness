"""Deterministic hypothesis catalog.

Each template inspects the TableProfile and instantiates zero or more concrete
HypothesisSpec objects. Templates are intentionally cheap to enumerate — the
LLM layer takes this output plus the profile and adds domain-specific ideas.

Priorities: lower-priority catalog items are pruned when budget runs short.
Group-level hypotheses get higher priority than single-column ones, because
they surface actionable entity lists — the main business deliverable.
"""

from __future__ import annotations

from typing import Callable

from core.deep_analysis.types import ColumnRole, HypothesisSpec, TableProfile

TemplateFn = Callable[[TableProfile], list[HypothesisSpec]]


def _hid(prefix: str, *parts: str) -> str:
    return "_".join([prefix, *[p.replace(".", "_") for p in parts]])


def _tpl_seasonality_numeric(p: TableProfile) -> list[HypothesisSpec]:
    """For each (date, numeric) pair, check seasonality on day/week/month/quarter."""
    out: list[HypothesisSpec] = []
    dates = p.date_columns()
    nums = p.numeric_columns()
    if not dates or not nums:
        return out
    # Take just the best date column (widest range) to keep combinatorics sane.
    best_date = max(dates, key=lambda c: (p.columns[c].n_unique, 0))
    for num in nums:
        out.append(HypothesisSpec(
            hypothesis_id=_hid("seasonality_num", best_date, num),
            runner="seasonality",
            title=f"Сезонность {num} по {best_date}",
            rationale=(
                f"Проверяем, есть ли периодичность (день недели, месяц, квартал) "
                f"в значении {num} по оси {best_date}."
            ),
            params={
                "date_col": best_date,
                "value_col": num,
                "agg": "sum",
            },
            priority=0.6,
            est_cost_seconds=15.0,
        ))
    return out


def _tpl_seasonality_counts(p: TableProfile) -> list[HypothesisSpec]:
    """Per-date counts + per-category counts over time (vacation-fraud style)."""
    out: list[HypothesisSpec] = []
    dates = p.date_columns()
    if not dates:
        return out
    best_date = max(dates, key=lambda c: (p.columns[c].n_unique, 0))
    # Event volume seasonality.
    out.append(HypothesisSpec(
        hypothesis_id=_hid("seasonality_rows", best_date),
        runner="seasonality",
        title=f"Сезонность количества событий по {best_date}",
        rationale="Неравномерное распределение числа записей по дням/неделям/кварталам может указывать на бизнес-паттерны.",
        params={"date_col": best_date, "value_col": None, "agg": "count"},
        priority=0.7,
        est_cost_seconds=10.0,
    ))
    # Category-wise seasonality — e.g. type=отпуск растёт в конце квартала.
    # Only iterate over equivalence-class representatives — running the same
    # seasonality on post_id, post_name, pos_name (all 1:1) triples the noise.
    cat_reps = p.representatives(p.category_columns(max_cardinality=50))
    for cat in cat_reps[:3]:
        out.append(HypothesisSpec(
            hypothesis_id=_hid("seasonality_cat", best_date, cat),
            runner="seasonality",
            title=f"Сезонность распределения {cat} по {best_date}",
            rationale=f"Проверяем, зависит ли доля категорий {cat} от периода (день недели / квартал).",
            params={"date_col": best_date, "value_col": None, "agg": "count", "group_col": cat},
            priority=0.75,
            est_cost_seconds=20.0,
        ))
    return out


def _tpl_group_anomalies(p: TableProfile) -> list[HypothesisSpec]:
    """Entity-level mass-deviation hypotheses — produce violators CSVs.

    We iterate up to 3 entity-candidate columns at different granularities so
    that small dictionaries (e.g. `tb_id` with 15 territorial banks) and
    larger entity sets (e.g. `employee_id` with thousands) both get checked.
    """
    out: list[HypothesisSpec] = []
    # Reduce to equivalence-class reps so e.g. saphr_id and a duplicate
    # pos_id↔post_id pair don't each spawn the full set of group hypotheses.
    candidates = p.representatives(
        p.entity_candidates(min_card=5, max_card=50_000)
    )
    dates = p.date_columns()
    if not candidates:
        return out

    best_date = max(dates, key=lambda c: p.columns[c].n_unique) if dates else None
    num_cols = p.representatives(p.numeric_columns())
    flag_cols = p.representatives(p.flag_columns())
    cat_cols = p.representatives(p.category_columns())

    # Pick up to 3 distinct granularity levels. Sort descending and take both
    # ends to capture finest + coarsest cohorts.
    if len(candidates) > 3:
        granular = candidates[:1] + candidates[len(candidates) // 2 : len(candidates) // 2 + 1] + candidates[-1:]
        # Dedup while preserving order.
        seen: set[str] = set()
        granular = [c for c in granular if not (c in seen or seen.add(c))]
    else:
        granular = candidates

    # Coarser cohorts (low cardinality) get a slightly lower priority — they
    # are statistically noisier, but still business-relevant.
    def _prio(base: float, entity: str) -> float:
        nu = p.columns[entity].n_unique
        if nu < 30:
            return base - 0.05
        return base

    for entity_id in granular:
        if best_date:
            out.append(HypothesisSpec(
                hypothesis_id=_hid("grp_volume", entity_id, best_date),
                runner="group_anomalies",
                title=f"Аномальный объём записей по {entity_id}",
                rationale=f"Ищем сущности ({entity_id}) с аномально высоким/низким объёмом событий относительно когорты.",
                params={
                    "entity_col": entity_id,
                    "date_col": best_date,
                    "metric": "row_count",
                    "period": "quarter",
                },
                priority=_prio(0.85, entity_id),
                est_cost_seconds=30.0,
            ))

        for num in num_cols[:3]:
            out.append(HypothesisSpec(
                hypothesis_id=_hid("grp_num", entity_id, num),
                runner="group_anomalies",
                title=f"Аномальные значения {num} в разрезе {entity_id}",
                rationale=f"Сущности ({entity_id}) с выбросами по средней/сумме {num}.",
                params={
                    "entity_col": entity_id,
                    "date_col": best_date,
                    "metric": "mean",
                    "value_col": num,
                    "period": "quarter" if best_date else None,
                },
                priority=_prio(0.8, entity_id),
                est_cost_seconds=25.0,
            ))

        for flag in flag_cols[:2]:
            out.append(HypothesisSpec(
                hypothesis_id=_hid("grp_flag", entity_id, flag),
                runner="group_anomalies",
                title=f"Сущности с аномальной долей {flag} ({entity_id})",
                rationale=f"{entity_id} с экстремально высокой/низкой долей {flag} относительно когорты.",
                params={
                    "entity_col": entity_id,
                    "date_col": best_date,
                    "metric": "rate",
                    "value_col": flag,
                    "period": "quarter" if best_date else None,
                },
                priority=_prio(0.9, entity_id),
                est_cost_seconds=25.0,
            ))

        for cat in cat_cols[:2]:
            if best_date and cat != entity_id:
                out.append(HypothesisSpec(
                    hypothesis_id=_hid("grp_cat_shift", entity_id, cat, best_date),
                    runner="group_anomalies",
                    title=f"Сдвиг распределения {cat} к концу квартала по {entity_id}",
                    rationale=(
                        f"Проверяем, не концентрируют ли отдельные {entity_id} значения "
                        f"{cat} в последних неделях квартала (типичный fraud-паттерн)."
                    ),
                    params={
                        "entity_col": entity_id,
                        "date_col": best_date,
                        "metric": "end_of_quarter_shift",
                        "category_col": cat,
                        "period": "quarter",
                    },
                    priority=_prio(0.95, entity_id),
                    est_cost_seconds=30.0,
                ))
    return out


def _tpl_outliers_numeric(p: TableProfile) -> list[HypothesisSpec]:
    out: list[HypothesisSpec] = []
    nums = p.numeric_columns()
    if not nums:
        return out
    out.append(HypothesisSpec(
        hypothesis_id=_hid("outliers_mv", *nums),
        runner="outliers",
        title="Многомерные выбросы по числовым колонкам",
        rationale="IsolationForest + robust-z по всем числовым атрибутам для поиска строк-аномалий.",
        params={"value_cols": nums, "method": "isolation_forest"},
        priority=0.55,
        est_cost_seconds=30.0,
    ))
    for num in nums[:5]:
        out.append(HypothesisSpec(
            hypothesis_id=_hid("outliers_uni", num),
            runner="outliers",
            title=f"Выбросы в колонке {num} (MAD)",
            rationale=f"Робастный z-score по {num} — ищем экстремальные значения.",
            params={"value_cols": [num], "method": "mad"},
            priority=0.45,
            est_cost_seconds=5.0,
        ))
    return out


def _tpl_dependencies(p: TableProfile) -> list[HypothesisSpec]:
    """Scan pairwise column dependencies. One hypothesis per analysis covers
    all analysable columns — the runner itself iterates pairs and caps work."""
    candidates = (
        p.numeric_columns()
        + p.category_columns(max_cardinality=50)
        + p.flag_columns()
    )
    # Drop obvious id columns — they have unique values and give no signal.
    candidates = [c for c in candidates if p.columns[c].role != ColumnRole.ID]
    # Drop members of equivalence classes — only keep the representative so
    # we don't enumerate trivially-1:1 pairs (post_id ↔ post_name).
    candidates = p.representatives(candidates)
    if len(candidates) < 2:
        return []
    return [HypothesisSpec(
        hypothesis_id=_hid("dependencies", "pairwise"),
        runner="dependencies",
        title="Скрытые зависимости между колонками",
        rationale=(
            "Проверяем пары колонок (Spearman для числовых, Cramér's V "
            "для категориальных, η² для смешанных) на значимые связи."
        ),
        params={"columns": candidates, "max_pairs": 80},
        priority=0.7,
        est_cost_seconds=60.0,
    )]


def _tpl_regime_shifts(p: TableProfile) -> list[HypothesisSpec]:
    out: list[HypothesisSpec] = []
    dates = p.date_columns()
    if not dates:
        return out
    best_date = max(dates, key=lambda c: p.columns[c].n_unique)
    # Overall volume regime changes.
    out.append(HypothesisSpec(
        hypothesis_id=_hid("regime_count", best_date),
        runner="regime_shifts",
        title=f"Смена режима объёма событий по {best_date}",
        rationale="Точки, в которых число событий существенно изменилось относительно истории.",
        params={"date_col": best_date, "value_col": None, "agg": "count", "freq": "day"},
        priority=0.65,
        est_cost_seconds=10.0,
    ))
    # One regime_shifts per numeric metric (top 3 to avoid combinatorics).
    for num in p.numeric_columns()[:3]:
        out.append(HypothesisSpec(
            hypothesis_id=_hid("regime_num", best_date, num),
            runner="regime_shifts",
            title=f"Смена режима {num} по {best_date}",
            rationale=f"Проверяем, были ли скачки среднего {num} во времени.",
            params={
                "date_col": best_date, "value_col": num,
                "agg": "mean", "freq": "day",
            },
            priority=0.6,
            est_cost_seconds=15.0,
        ))
    return out


TEMPLATES: tuple[TemplateFn, ...] = (
    _tpl_seasonality_counts,
    _tpl_seasonality_numeric,
    _tpl_group_anomalies,
    _tpl_outliers_numeric,
    _tpl_dependencies,
    _tpl_regime_shifts,
)


def generate_catalog_hypotheses(profile: TableProfile) -> list[HypothesisSpec]:
    out: list[HypothesisSpec] = []
    for tpl in TEMPLATES:
        try:
            items = tpl(profile)
        except Exception:
            items = []
        for item in items:
            item.source = "catalog"
        out.extend(items)
    # Dedup by id, keep highest priority instance.
    seen: dict[str, HypothesisSpec] = {}
    for h in out:
        existing = seen.get(h.hypothesis_id)
        if existing is None or h.priority > existing.priority:
            seen[h.hypothesis_id] = h
    return list(seen.values())
