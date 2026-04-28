"""Тесты детерминированного экстрактора подсказок пользователя.

Все таблицы синтетические — никаких реальных имён схем/таблиц заказчика.
Покрытие:
- must_keep_tables (явные «возьми из», «в таблице», schema.table)
- join_fields («по инн», «через customer_id»)
- dim_sources («сегмент возьми в TABLE по полю»)
- having_hints («от 3 человек», «более 5 клиентов», «не менее 10»)
- match_unit_column для HAVING COUNT(DISTINCT)
"""

import pandas as pd
import pytest

from core.schema_loader import SchemaLoader
from core.user_hint_extractor import (
    extract_user_hints,
    match_unit_column,
)


@pytest.fixture
def synthetic_loader(tmp_path):
    """Синтетический каталог: одна fact, одна dim, одна ref."""
    tables_df = pd.DataFrame({
        "schema_name": ["schema_a", "schema_a", "schema_a"],
        "table_name": ["fact_x", "dim_y", "ref_segments_table"],
        "description": [
            "Фактовая таблица событий",
            "Справочник клиентов с сегментом",
            "Длинное имя справочника сегментов",
        ],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)

    attrs_df = pd.DataFrame({
        "schema_name": ["schema_a"] * 10,
        "table_name": [
            "fact_x", "fact_x", "fact_x", "fact_x", "fact_x",
            "dim_y", "dim_y", "dim_y",
            "ref_segments_table", "ref_segments_table",
        ],
        "column_name": [
            "event_id", "inn", "amount", "tb_id", "gosb_id",
            "client_id", "inn", "segment_name",
            "segment_id", "segment_name",
        ],
        "dType": [
            "bigint", "varchar", "numeric", "varchar", "varchar",
            "bigint", "varchar", "varchar",
            "bigint", "varchar",
        ],
        "description": [
            "PK события", "ИНН клиента", "Сумма", "ТБ", "ГОСБ",
            "PK клиента", "ИНН клиента", "Сегмент",
            "PK сегмента", "Название сегмента",
        ],
        "is_primary_key": [
            True, False, False, False, False,
            True, False, False,
            True, False,
        ],
        "unique_perc": [100.0, 50.0, 1.0, 30.0, 20.0, 100.0, 80.0, 5.0, 100.0, 100.0],
        "not_null_perc": [100.0, 80.0, 99.0, 100.0, 100.0, 100.0, 95.0, 50.0, 100.0, 100.0],
    })
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)

    return SchemaLoader(data_dir=tmp_path)


class TestMustKeepTables:
    def test_explicit_schema_table(self, synthetic_loader):
        """schema.table в тексте → must_keep_tables."""
        hints = extract_user_hints(
            "посчитай метрику по schema_a.dim_y", synthetic_loader,
        )
        assert ("schema_a", "dim_y") in hints["must_keep_tables"]

    def test_vozmi_v_table_phrase(self, synthetic_loader):
        """«сегмент возьми в dim_y» → dim_y попадает в must_keep."""
        hints = extract_user_hints(
            "сумма по сегменту, сегмент возьми в dim_y", synthetic_loader,
        )
        names = {t for _, t in hints["must_keep_tables"]}
        assert "dim_y" in names

    def test_v_tablice_phrase(self, synthetic_loader):
        hints = extract_user_hints(
            "выбери всё в таблице fact_x", synthetic_loader,
        )
        names = {t for _, t in hints["must_keep_tables"]}
        assert "fact_x" in names

    def test_unknown_table_ignored(self, synthetic_loader):
        """Несуществующая таблица — игнорируется."""
        hints = extract_user_hints(
            "возьми в nonexistent_table", synthetic_loader,
        )
        assert hints["must_keep_tables"] == []


class TestJoinFields:
    def test_po_inn(self, synthetic_loader):
        """«по инн» → ['inn'] (есть в обеих таблицах)."""
        hints = extract_user_hints(
            "соедини fact_x и dim_y по инн", synthetic_loader,
        )
        assert "inn" in hints["join_fields"]

    def test_po_explicit_column(self, synthetic_loader):
        hints = extract_user_hints(
            "соедини через client_id", synthetic_loader,
        )
        # client_id есть только в dim_y — но в attrs_df он один
        # В стоп-словах нет, поэтому должен резолвиться.
        assert "client_id" in hints["join_fields"]

    def test_stopword_ignored(self, synthetic_loader):
        """«по дате» — «дате» обычно резолвится через _KEY_SYNONYMS, но
        конкретно слово «и» в стоп-листе должно игнорироваться."""
        hints = extract_user_hints(
            "сгруппируй по и потом", synthetic_loader,
        )
        # «и» — стоп-слово
        assert "и" not in hints["join_fields"]


class TestDimSources:
    def test_segment_take_from_table(self, synthetic_loader):
        """«сегмент возьми в dim_y по инн» → dim_sources['segment'] = {table, join_col}."""
        hints = extract_user_hints(
            "сумма по сегменту, сегмент возьми в dim_y по инн", synthetic_loader,
        )
        # ключ нормализован через SYNONYM_MAP
        slot_keys = list(hints["dim_sources"].keys())
        assert any("segment" in k or "сегмент" in k for k in slot_keys), slot_keys
        # есть запись с table=schema_a.dim_y
        bound_tables = {b.get("table") for b in hints["dim_sources"].values()}
        assert "schema_a.dim_y" in bound_tables
        # join_col=inn зафиксирован
        bound_cols = {b.get("join_col") for b in hints["dim_sources"].values()}
        assert "inn" in bound_cols

    def test_dim_source_adds_table_to_join_fields(self, synthetic_loader):
        """Если dim_source указал поле, оно автоматически в join_fields."""
        hints = extract_user_hints(
            "сегмент возьми в dim_y по инн", synthetic_loader,
        )
        assert "inn" in hints["join_fields"]


class TestHavingHints:
    def test_ot_3_chelovek(self, synthetic_loader):
        hints = extract_user_hints("отток от 3 человек", synthetic_loader)
        assert len(hints["having_hints"]) >= 1
        h = hints["having_hints"][0]
        assert h["op"] == ">=" and h["value"] == 3
        assert "челов" in h["unit_hint"]

    def test_bolee_5_klientov(self, synthetic_loader):
        hints = extract_user_hints("посчитай более 5 клиентов", synthetic_loader)
        assert any(h["value"] == 5 for h in hints["having_hints"])

    def test_ne_menee_10(self, synthetic_loader):
        hints = extract_user_hints("не менее 10 заказов", synthetic_loader)
        assert any(h["value"] == 10 for h in hints["having_hints"])

    def test_no_having_in_plain_query(self, synthetic_loader):
        hints = extract_user_hints("посчитай сумму", synthetic_loader)
        assert hints["having_hints"] == []


class TestMatchUnitColumn:
    def test_unit_human_resolves_to_emp_like(self, synthetic_loader):
        """unit_hint=«человек» в таблице с client_id → подбирается какая-то PK."""
        # В нашем синтетическом dim_y нет специфической колонки для «человек»,
        # но client_id похож на PK сущности.
        col = match_unit_column("клиент", "schema_a.dim_y", synthetic_loader)
        assert col == "client_id"

    def test_no_match_returns_none(self, synthetic_loader):
        col = match_unit_column("банан", "schema_a.dim_y", synthetic_loader)
        assert col is None

    def test_invalid_table_returns_none(self, synthetic_loader):
        col = match_unit_column("клиент", "schema_a.does_not_exist", synthetic_loader)
        assert col is None


class TestExtractUserHintsContract:
    def test_empty_input_returns_empty_structure(self, synthetic_loader):
        hints = extract_user_hints("", synthetic_loader)
        assert hints["must_keep_tables"] == []
        assert hints["join_fields"] == []
        assert hints["dim_sources"] == {}
        assert hints["having_hints"] == []
        assert hints["group_by_hints"] == []
        assert hints["aggregate_hints"] == []
        assert hints["aggregation_preferences"] == {}
        assert hints["time_granularity"] is None
        assert hints["negative_filters"] == []

    def test_none_loader_safe(self):
        """Не падает при отсутствии loader."""
        hints = extract_user_hints("посчитай отток по сегменту", None)
        # Возвращает пустую структуру вместо исключения
        assert hints["must_keep_tables"] == []


class TestGroupByHints:
    def test_sgruppiruй_po_column(self, synthetic_loader):
        """«сгруппируй по segment_name» → group_by_hints содержит segment_name."""
        hints = extract_user_hints(
            "сгруппируй по segment_name", synthetic_loader,
        )
        assert "segment_name" in hints["group_by_hints"]

    def test_po_column_no_join_context(self, synthetic_loader):
        """«по segment_name» без JOIN-контекста → group_by_hints."""
        hints = extract_user_hints(
            "посчитай задачи. По segment_name", synthetic_loader,
        )
        assert "segment_name" in hints["group_by_hints"]

    def test_po_column_with_join_context_not_group(self, synthetic_loader):
        """«соедини по inn» — JOIN-контекст, значит в join_fields, не в group_by."""
        hints = extract_user_hints(
            "соедини fact_x и dim_y по inn", synthetic_loader,
        )
        assert "inn" in hints["join_fields"]
        assert "inn" not in hints["group_by_hints"]

    def test_unknown_column_ignored(self, synthetic_loader):
        """Колонка не из каталога → игнорируется."""
        hints = extract_user_hints(
            "сгруппируй по nonexistent_column", synthetic_loader,
        )
        assert "nonexistent_column" not in hints["group_by_hints"]

    def test_group_by_english(self, synthetic_loader):
        """«group by segment_name» → group_by_hints."""
        hints = extract_user_hints(
            "group by segment_name", synthetic_loader,
        )
        assert "segment_name" in hints["group_by_hints"]


class TestAggregateHints:
    def test_poschitay_noun(self, synthetic_loader):
        """«посчитай amount» → aggregate_hints содержит ("count", "amount")."""
        hints = extract_user_hints(
            "посчитай amount за февраль", synthetic_loader,
        )
        agg_funcs = [a[0] for a in hints["aggregate_hints"]]
        assert "count" in agg_funcs

    def test_summa_po_noun(self, synthetic_loader):
        """«сумма по amount» → ("sum", "amount")."""
        hints = extract_user_hints(
            "сумма по amount", synthetic_loader,
        )
        assert any(a[0] == "sum" for a in hints["aggregate_hints"])

    def test_count_english(self, synthetic_loader):
        """«count of amount» → ("count", "amount")."""
        hints = extract_user_hints(
            "count of amount", synthetic_loader,
        )
        assert any(a[0] == "count" for a in hints["aggregate_hints"])

    def test_acceptance_count_task_code(self, synthetic_loader):
        """Acceptance: посчитай задачи по task_code → aggregate_hints count + group_by."""
        # event_id есть в каталоге, segment_name есть в каталоге
        hints = extract_user_hints(
            "посчитай event_id. По segment_name", synthetic_loader,
        )
        agg_funcs = [a[0] for a in hints["aggregate_hints"]]
        assert "count" in agg_funcs
        assert "segment_name" in hints["group_by_hints"]

    def test_count_distinct_preference(self, synthetic_loader):
        """count(distinct event_id) → aggregation_preferences.distinct=True."""
        hints = extract_user_hints(
            "возьми count(distinct event_id)", synthetic_loader,
        )
        assert hints["aggregation_preferences"] == {
            "function": "count",
            "column": "event_id",
            "distinct": True,
        }

    def test_extract_force_count_star_for_prosto_stroki(self, synthetic_loader):
        hints = extract_user_hints(
            "посчитай просто количество строк", synthetic_loader,
        )
        assert hints["aggregation_preferences"] == {
            "function": "count",
            "column": "*",
            "distinct": False,
            "force_count_star": True,
        }

    def test_extract_no_distinct_phrase(self, synthetic_loader):
        hints = extract_user_hints(
            "не надо считать по уникальной дате", synthetic_loader,
        )
        assert hints["aggregation_preferences"] == {
            "function": "count",
            "distinct": False,
        }

    def test_extract_multiple_distinct_count_targets(self, synthetic_loader):
        hints = extract_user_hints(
            "сколько всего есть уникальных тб и госб", synthetic_loader,
        )
        assert hints["aggregation_preferences_list"] == [
            {"function": "count", "column": "tb_id", "distinct": True},
            {"function": "count", "column": "gosb_id", "distinct": True},
        ]


class TestTimeGranularity:
    def test_pomesyachno(self, synthetic_loader):
        """«помесячно» → time_granularity == "month"."""
        hints = extract_user_hints("посчитай помесячно", synthetic_loader)
        assert hints["time_granularity"] == "month"

    def test_po_kvartalам(self, synthetic_loader):
        """«по кварталам» → time_granularity == "quarter"."""
        hints = extract_user_hints("отток по кварталам", synthetic_loader)
        assert hints["time_granularity"] == "quarter"

    def test_monthly_english(self, synthetic_loader):
        """«monthly» → time_granularity == "month"."""
        hints = extract_user_hints("show monthly stats", synthetic_loader)
        assert hints["time_granularity"] == "month"

    def test_yearly(self, synthetic_loader):
        """«ежегодно» → time_granularity == "year"."""
        hints = extract_user_hints("покажи ежегодно", synthetic_loader)
        assert hints["time_granularity"] == "year"

    def test_daily(self, synthetic_loader):
        """«ежедневно» → time_granularity == "day"."""
        hints = extract_user_hints("ежедневно отчёт", synthetic_loader)
        assert hints["time_granularity"] == "day"

    def test_no_granularity(self, synthetic_loader):
        """Без временной гранулярности → None."""
        hints = extract_user_hints("покажи сумму по клиентам", synthetic_loader)
        assert hints["time_granularity"] is None

    def test_acceptance_vozmi_fact_pomesyachno(self, synthetic_loader):
        """Acceptance: «возьми schema_a.fact_x, посчитай помесячно» → time_granularity=month + must_keep."""
        hints = extract_user_hints(
            "возьми schema_a.fact_x, посчитай помесячно", synthetic_loader,
        )
        assert hints["time_granularity"] == "month"
        assert ("schema_a", "fact_x") in hints["must_keep_tables"]


class TestNegativeFilters:
    def test_ne_uchityvay(self, synthetic_loader):
        """«не учитывай канцелярию» → negative_filters содержит "канцелярию"."""
        hints = extract_user_hints(
            "покажи всех клиентов, не учитывай канцелярию", synthetic_loader,
        )
        assert len(hints["negative_filters"]) >= 1
        assert any("канцеляр" in v for v in hints["negative_filters"])

    def test_isklyuchi(self, synthetic_loader):
        """«исключи тест» → negative_filters."""
        hints = extract_user_hints("исключи тест из выборки", synthetic_loader)
        assert len(hints["negative_filters"]) >= 1

    def test_krome(self, synthetic_loader):
        """«кроме X» → negative_filters."""
        hints = extract_user_hints("покажи всё кроме архива", synthetic_loader)
        assert len(hints["negative_filters"]) >= 1
        assert any("архив" in v for v in hints["negative_filters"])

    def test_exclude_english(self, synthetic_loader):
        """«excluding test» → negative_filters."""
        hints = extract_user_hints("show all excluding test data", synthetic_loader)
        assert len(hints["negative_filters"]) >= 1

    def test_no_negative_filter(self, synthetic_loader):
        """Без исключений → пустой список."""
        hints = extract_user_hints("посчитай отток по клиентам", synthetic_loader)
        assert hints["negative_filters"] == []
