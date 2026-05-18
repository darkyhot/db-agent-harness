"""Тесты для core/entity_resolver.py.

Проверяем универсальность матчинга «entity_term → column» без захардкоженных
доменных алиасов: только эмбеддинги (или text-overlap-fallback) + LLM-tiebreak.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.entity_resolver import (
    EntityResolution,
    resolve_entity_to_columns,
    reset_resolver_cache,
)
from core.schema_loader import SchemaLoader


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_resolver_cache()
    yield
    reset_resolver_cache()


def _loader_with_gosb(tmp_path) -> SchemaLoader:
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["uzp_dim_gosb"],
        "description": ["Справочник ТБ и ГОСБ"],
        "grain": ["dictionary"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 4,
        "table_name": ["uzp_dim_gosb"] * 4,
        "column_name": ["tb_id", "tb_short_name", "old_gosb_id", "new_gosb_id"],
        "dType": ["int4", "text", "int4", "int4"],
        "description": [
            "Номер ТБ",
            "Краткое наименование ТБ",
            "Старый номер ГОСБ",
            "Новый номер ГОСБ",
        ],
        "is_primary_key": [True, False, True, False],
        "unique_perc": [6.0, 6.0, 98.0, 58.0],
        "not_null_perc": [100.0, 100.0, 100.0, 100.0],
        "is_not_null": [False, False, False, False],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def _loader_synthetic_inventory(tmp_path) -> SchemaLoader:
    """Синтетический НЕбанковский каталог: SKU-каталог товаров.

    Проверяет, что resolver работает без какого-либо словаря «sku → sku_code».
    """
    tables_df = pd.DataFrame({
        "schema_name": ["wh"],
        "table_name": ["inventory_items"],
        "description": ["Справочник складских позиций"],
        "grain": ["product"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["wh"] * 3,
        "table_name": ["inventory_items"] * 3,
        "column_name": ["sku_code", "item_title", "stock_qty"],
        "dType": ["varchar(40)", "text", "int4"],
        "description": [
            "Код товарной позиции (SKU)",
            "Название товара",
            "Текущий остаток",
        ],
        "is_primary_key": [True, False, False],
        "unique_perc": [100.0, 95.0, 80.0],
        "not_null_perc": [100.0, 100.0, 100.0],
        "is_not_null": [False, False, False],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


class _StubLLM:
    """Имитатор llm_invoker: возвращает заданный verdict."""

    def __init__(self, verdict):
        self._verdict = verdict
        self.calls: list[dict] = []

    def _llm_json_with_retry(self, system_prompt, user_prompt, *, temperature, failure_tag, expect):
        import json

        self.calls.append({
            "system": system_prompt,
            "user": user_prompt,
            "tag": failure_tag,
        })
        return self._verdict


# ---------------------------------------------------------------------------


def test_resolves_tb_via_text_overlap_when_no_embeddings(tmp_path):
    """«ТБ» в пуле dim_gosb → выбирает tb_id по описанию «Номер ТБ» без алиасов."""
    loader = _loader_with_gosb(tmp_path)

    res = resolve_entity_to_columns(
        entity_term="ТБ",
        user_input="Сколько всего есть ТБ",
        candidate_table_keys=["dm.uzp_dim_gosb"],
        schema_loader=loader,
        llm_invoker=None,
        role_hint="id",
    )

    assert res.matched
    assert res.column == "tb_id"
    assert res.decision_path in {"text_only", "embedding_only"}


def test_resolves_sku_in_unrelated_domain_without_dictionaries(tmp_path):
    """Синтетический НЕдоменный каталог: «sku» → sku_code только по описанию."""
    loader = _loader_synthetic_inventory(tmp_path)

    res = resolve_entity_to_columns(
        entity_term="sku",
        user_input="Сколько SKU в каталоге",
        candidate_table_keys=["wh.inventory_items"],
        schema_loader=loader,
        llm_invoker=None,
        role_hint="id",
    )

    assert res.matched
    assert res.column == "sku_code"


def test_label_role_excludes_id_columns(tmp_path):
    """role=label не должен возвращать tb_id даже если описание совпадает."""
    loader = _loader_with_gosb(tmp_path)

    res = resolve_entity_to_columns(
        entity_term="ТБ",
        user_input="Названия ТБ",
        candidate_table_keys=["dm.uzp_dim_gosb"],
        schema_loader=loader,
        llm_invoker=None,
        role_hint="label",
    )

    assert res.matched
    assert res.column == "tb_short_name"


def test_llm_tiebreak_when_old_new_ambiguous(tmp_path):
    """«ГОСБ» при наличии old_gosb_id и new_gosb_id — оба близки → LLM выбирает new_gosb_id."""
    loader = _loader_with_gosb(tmp_path)
    stub = _StubLLM({
        "chosen_ref": "dm.uzp_dim_gosb.new_gosb_id",
        "confidence": 0.9,
        "reason": "канонический, без old_-префикса",
    })

    res = resolve_entity_to_columns(
        entity_term="ГОСБ",
        user_input="Сколько ГОСБ",
        candidate_table_keys=["dm.uzp_dim_gosb"],
        schema_loader=loader,
        llm_invoker=stub,
        role_hint="id",
        # Принудительно низкий floor чтобы вызвать LLM (без эмбеддингов score'ы
        # одинаковые ~0.7 на тексте, gap=0).
        high_confidence_floor=0.99,
    )

    assert res.matched
    assert res.column == "new_gosb_id"
    assert res.decision_path == "llm_tiebreak"
    assert len(stub.calls) == 1


def test_llm_failure_falls_back_to_top1(tmp_path):
    """LLM возвращает None → resolver не падает, выбирает top1 по score."""
    loader = _loader_with_gosb(tmp_path)

    class _FailingLLM:
        def _llm_json_with_retry(self, *args, **kwargs):
            return None

    res = resolve_entity_to_columns(
        entity_term="ГОСБ",
        user_input="Сколько ГОСБ",
        candidate_table_keys=["dm.uzp_dim_gosb"],
        schema_loader=loader,
        llm_invoker=_FailingLLM(),
        role_hint="id",
        high_confidence_floor=0.99,
    )

    assert res.matched
    assert res.column in {"old_gosb_id", "new_gosb_id"}
    assert "fallback" in res.reason.lower() or res.decision_path != "llm_tiebreak"


def test_llm_hallucinated_ref_falls_back_to_top1(tmp_path):
    loader = _loader_with_gosb(tmp_path)
    stub = _StubLLM({
        "chosen_ref": "dm.uzp_dim_gosb.nonexistent_col",
        "confidence": 0.9,
        "reason": "выдумал",
    })

    res = resolve_entity_to_columns(
        entity_term="ГОСБ",
        user_input="Сколько ГОСБ",
        candidate_table_keys=["dm.uzp_dim_gosb"],
        schema_loader=loader,
        llm_invoker=stub,
        role_hint="id",
        high_confidence_floor=0.99,
    )

    assert res.matched
    assert res.column in {"old_gosb_id", "new_gosb_id"}


def test_no_match_returns_empty_resolution(tmp_path):
    loader = _loader_with_gosb(tmp_path)

    res = resolve_entity_to_columns(
        entity_term="ZZZZ_не_существует",
        user_input="...",
        candidate_table_keys=["dm.uzp_dim_gosb"],
        schema_loader=loader,
        llm_invoker=None,
        role_hint="id",
    )

    assert not res.matched
    assert res.column_ref is None
    assert res.decision_path == "no_match"


def test_cache_avoids_repeat_llm_calls(tmp_path):
    loader = _loader_with_gosb(tmp_path)
    stub = _StubLLM({
        "chosen_ref": "dm.uzp_dim_gosb.new_gosb_id",
        "confidence": 0.9,
        "reason": "ok",
    })

    kwargs = dict(
        entity_term="ГОСБ",
        user_input="Сколько ГОСБ",
        candidate_table_keys=["dm.uzp_dim_gosb"],
        schema_loader=loader,
        llm_invoker=stub,
        role_hint="id",
        high_confidence_floor=0.99,
    )
    r1 = resolve_entity_to_columns(**kwargs)
    r2 = resolve_entity_to_columns(**kwargs)

    assert r1.column == r2.column == "new_gosb_id"
    assert len(stub.calls) == 1  # второй раз — из кэша
