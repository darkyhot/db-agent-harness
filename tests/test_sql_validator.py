"""Тесты валидации SQL: определение режима, извлечение JOIN, проверка WHERE/LIMIT."""

import pytest

from core.sql_validator import (
    SQLMode,
    SQLValidator,
    ValidationResult,
    detect_mode,
    _build_alias_map,
    _extract_join_conditions,
    _find_subquery_join_aliases,
    _has_top_level_where,
    _has_where_or_limit,
)


class TestDetectMode:
    def test_select(self):
        assert detect_mode("SELECT * FROM t") == SQLMode.READ

    def test_select_with_comment(self):
        assert detect_mode("-- comment\nSELECT * FROM t") == SQLMode.READ

    def test_insert(self):
        assert detect_mode("INSERT INTO t VALUES (1)") == SQLMode.WRITE

    def test_update(self):
        assert detect_mode("UPDATE t SET a = 1") == SQLMode.WRITE

    def test_delete(self):
        assert detect_mode("DELETE FROM t WHERE id = 1") == SQLMode.WRITE

    def test_create(self):
        assert detect_mode("CREATE TABLE s.t (id INT)") == SQLMode.DDL

    def test_drop(self):
        assert detect_mode("DROP TABLE s.t") == SQLMode.DDL

    def test_truncate(self):
        assert detect_mode("TRUNCATE TABLE s.t") == SQLMode.DDL

    def test_alter(self):
        assert detect_mode("ALTER TABLE s.t ADD COLUMN x INT") == SQLMode.DDL

    def test_cte_select(self):
        assert detect_mode("WITH cte AS (SELECT 1) SELECT * FROM cte") == SQLMode.READ

    def test_cte_insert(self):
        assert detect_mode(
            "WITH cte AS (SELECT id FROM s.t) INSERT INTO s.target SELECT * FROM cte"
        ) == SQLMode.WRITE

    def test_cte_delete(self):
        assert detect_mode(
            "WITH ids AS (SELECT id FROM s.t WHERE x > 10) DELETE FROM s.t WHERE id IN (SELECT id FROM ids)"
        ) == SQLMode.WRITE

    def test_cte_update(self):
        assert detect_mode(
            "WITH src AS (SELECT id, val FROM s.src) UPDATE s.target SET val = src.val FROM src WHERE s.target.id = src.id"
        ) == SQLMode.WRITE


class TestHasWhereOrLimit:
    def test_with_where(self):
        assert _has_where_or_limit("SELECT * FROM t WHERE id = 1") is True

    def test_with_limit(self):
        assert _has_where_or_limit("SELECT * FROM t LIMIT 10") is True

    def test_without_both(self):
        assert _has_where_or_limit("SELECT * FROM t") is False

    def test_where_in_subquery_only(self):
        # WHERE в подзапросе не считается
        sql = "SELECT * FROM (SELECT * FROM t WHERE id = 1) sub"
        assert _has_where_or_limit(sql) is False

    def test_where_in_string_literal(self):
        sql = "SELECT 'WHERE is a keyword' FROM t"
        assert _has_where_or_limit(sql) is False


class TestHasTopLevelWhere:
    def test_update_with_where(self):
        assert _has_top_level_where("UPDATE hr.emp SET name = 'x' WHERE id = 1") is True

    def test_update_with_where_in_string_only(self):
        sql = "UPDATE hr.emp SET note = 'WHERE id = 1'"
        assert _has_top_level_where(sql) is False

    def test_cte_update_with_top_level_where(self):
        sql = (
            "WITH src AS (SELECT id FROM hr.emp WHERE active = 1) "
            "UPDATE hr.emp e SET flag = 1 FROM src WHERE e.id = src.id"
        )
        assert _has_top_level_where(sql) is True


# ---------------------------------------------------------------------------
# _build_alias_map
# ---------------------------------------------------------------------------
class TestBuildAliasMap:
    def test_simple_alias(self):
        sql = "SELECT * FROM hr.emp e JOIN hr.dept d ON d.id = e.dept_id"
        m = _build_alias_map(sql)
        assert m["e"] == ("hr", "emp")
        assert m["d"] == ("hr", "dept")

    def test_as_keyword(self):
        sql = "SELECT * FROM hr.emp AS e"
        m = _build_alias_map(sql)
        assert m["e"] == ("hr", "emp")
        assert m["emp"] == ("hr", "emp")

    def test_no_alias_uses_table_name(self):
        sql = "SELECT * FROM hr.emp"
        m = _build_alias_map(sql)
        assert m["emp"] == ("hr", "emp")

    def test_keyword_not_treated_as_alias(self):
        """SQL keywords after table ref should not become aliases."""
        sql = "SELECT * FROM hr.emp WHERE id = 1"
        m = _build_alias_map(sql)
        assert "where" not in m
        assert "emp" in m

    def test_multiple_joins(self):
        sql = """
            SELECT * FROM hr.emp e
            JOIN hr.dept d ON d.id = e.dept_id
            JOIN hr.loc l ON l.id = d.loc_id
        """
        m = _build_alias_map(sql)
        assert m["e"] == ("hr", "emp")
        assert m["d"] == ("hr", "dept")
        assert m["l"] == ("hr", "loc")

    def test_left_right_join(self):
        sql = "SELECT * FROM hr.emp e LEFT JOIN hr.dept d ON d.id = e.dept_id"
        m = _build_alias_map(sql)
        assert m["d"] == ("hr", "dept")

    def test_quoted_identifiers(self):
        sql = 'SELECT * FROM "hr"."emp" e'
        m = _build_alias_map(sql)
        assert m["e"] == ("hr", "emp")


# ---------------------------------------------------------------------------
# _extract_join_conditions — comprehensive
# ---------------------------------------------------------------------------
class TestExtractJoinConditions:
    def test_simple_join(self):
        sql = """
            SELECT * FROM hr.emp e
            JOIN hr.dept d ON d.id = e.dept_id
        """
        joins = _extract_join_conditions(sql)
        assert len(joins) >= 1
        assert joins[0]["schema"] == "hr"
        assert joins[0]["table"] == "dept"

    def test_no_joins(self):
        sql = "SELECT * FROM hr.emp"
        joins = _extract_join_conditions(sql)
        assert joins == []

    def test_join_with_alias_resolves_both_sides(self):
        """Both sides of ON should be resolved via alias map."""
        sql = """
            SELECT * FROM hr.emp e
            JOIN hr.dept d ON d.id = e.dept_id
        """
        joins = _extract_join_conditions(sql)
        schemas_tables = [(j["schema"], j["table"], j["column"]) for j in joins]
        assert ("hr", "dept", "id") in schemas_tables
        assert ("hr", "emp", "dept_id") in schemas_tables

    def test_left_join(self):
        sql = """
            SELECT * FROM hr.emp e
            LEFT JOIN hr.dept d ON d.id = e.dept_id
        """
        joins = _extract_join_conditions(sql)
        tables = {j["table"] for j in joins}
        assert "dept" in tables
        assert "emp" in tables

    def test_right_join(self):
        sql = """
            SELECT * FROM hr.emp e
            RIGHT JOIN hr.dept d ON d.id = e.dept_id
        """
        joins = _extract_join_conditions(sql)
        tables = {j["table"] for j in joins}
        assert "dept" in tables

    def test_multi_column_on(self):
        """Multi-column ON should extract all columns."""
        sql = """
            SELECT * FROM hr.emp e
            JOIN hr.dept d ON d.id = e.dept_id AND d.code = e.dept_code
        """
        joins = _extract_join_conditions(sql)
        columns = {(j["table"], j["column"]) for j in joins}
        assert ("dept", "id") in columns
        assert ("dept", "code") in columns
        assert ("emp", "dept_id") in columns
        assert ("emp", "dept_code") in columns

    def test_cross_join(self):
        """CROSS JOIN should produce __CROSS_JOIN__ sentinel."""
        sql = "SELECT * FROM hr.emp e CROSS JOIN hr.dept"
        joins = _extract_join_conditions(sql)
        cross = [j for j in joins if j["column"] == "__CROSS_JOIN__"]
        assert len(cross) == 1
        assert cross[0]["schema"] == "hr"
        assert cross[0]["table"] == "dept"

    def test_implicit_join(self):
        """FROM t1, t2 WHERE t1.a = t2.a should be detected."""
        sql = """
            SELECT * FROM hr.emp e, hr.dept d
            WHERE e.dept_id = d.id
        """
        joins = _extract_join_conditions(sql)
        tables = {j["table"] for j in joins}
        assert "emp" in tables
        assert "dept" in tables

    def test_self_join(self):
        """Self-join: same table with different aliases."""
        sql = """
            SELECT * FROM hr.emp e1
            JOIN hr.emp e2 ON e2.manager_id = e1.id
        """
        joins = _extract_join_conditions(sql)
        # Both sides should resolve to hr.emp
        assert all(j["schema"] == "hr" and j["table"] == "emp" for j in joins)
        columns = {j["column"] for j in joins}
        assert "manager_id" in columns
        assert "id" in columns

    def test_deduplication(self):
        """Same schema.table.column should appear only once."""
        sql = """
            SELECT * FROM hr.emp e
            JOIN hr.dept d ON d.id = e.dept_id AND d.id = e.dept_id
        """
        joins = _extract_join_conditions(sql)
        keys = [(j["schema"], j["table"], j["column"]) for j in joins]
        assert len(keys) == len(set(keys))

    def test_multiple_joins_chain(self):
        """Chain of JOINs: emp → dept → loc."""
        sql = """
            SELECT * FROM hr.emp e
            JOIN hr.dept d ON d.id = e.dept_id
            JOIN hr.loc l ON l.id = d.loc_id
        """
        joins = _extract_join_conditions(sql)
        tables = {j["table"] for j in joins}
        assert "emp" in tables
        assert "dept" in tables
        assert "loc" in tables

    def test_cte_with_join(self):
        """CTE body containing a JOIN should be extracted."""
        sql = """
            WITH active_emp AS (
                SELECT e.id, e.dept_id
                FROM hr.emp e
                JOIN hr.dept d ON d.id = e.dept_id
                WHERE e.active = 1
            )
            SELECT * FROM active_emp
        """
        joins = _extract_join_conditions(sql)
        # The JOIN inside CTE should be found
        tables = {j["table"] for j in joins}
        assert "dept" in tables

    def test_sql_without_schema_prefix_returns_empty(self):
        """Tables without schema.table pattern are not extracted."""
        sql = "SELECT * FROM emp e JOIN dept d ON d.id = e.dept_id"
        joins = _extract_join_conditions(sql)
        # No schema prefix → alias map empty → no joins resolved
        assert joins == []

    def test_subquery_join_skipped(self):
        """JOIN with subquery (SELECT DISTINCT) should be treated as safe — entire ON skipped."""
        sql = (
            "SELECT a.id, sub.name "
            "FROM hr.emp a "
            "JOIN (SELECT DISTINCT dept_id, name FROM hr.dept) sub "
            "ON a.dept_id = sub.dept_id"
        )
        joins = _extract_join_conditions(sql)
        # Subquery JOIN is not matched by the explicit JOIN regex,
        # so neither side of the ON clause is checked — this is safe.
        assert joins == []

    def test_mixed_subquery_and_direct_join(self):
        """Mix of subquery JOIN (safe) and direct JOIN (checked)."""
        sql = (
            "SELECT a.id, sub.name, r.region_name "
            "FROM hr.emp a "
            "JOIN (SELECT DISTINCT dept_id, name FROM hr.dept) sub "
            "ON a.dept_id = sub.dept_id "
            "JOIN hr.regions r ON a.region_id = r.region_id"
        )
        joins = _extract_join_conditions(sql)
        # Direct JOIN hr.regions should be checked
        resolved_tables = {j["table"] for j in joins}
        assert "regions" in resolved_tables
        # Subquery JOIN should NOT produce dept entries
        assert "dept" not in resolved_tables

    def test_subquery_join_alias_detected(self):
        """_find_subquery_join_aliases correctly extracts aliases."""
        sql = (
            "SELECT * FROM hr.emp a "
            "JOIN (SELECT DISTINCT dept_id, name FROM hr.dept) sub ON a.dept_id = sub.dept_id "
            "LEFT JOIN (SELECT region_id, SUM(val) AS total FROM hr.regions GROUP BY region_id) reg "
            "ON a.region_id = reg.region_id"
        )
        aliases = _find_subquery_join_aliases(sql)
        assert "sub" in aliases
        assert "reg" in aliases
        assert "a" not in aliases


# ---------------------------------------------------------------------------
# Multiplication factor
# ---------------------------------------------------------------------------
class TestMultiplicationFactor:
    """Тесты оценки multiplication factor для row explosion."""

    def test_all_unique_joins(self):
        checks = [
            {"is_unique": True, "duplicate_pct": 0.0, "table": "t1", "columns": "id"},
            {"is_unique": True, "duplicate_pct": 0.0, "table": "t2", "columns": "id"},
        ]
        factor = SQLValidator._estimate_multiplication_factor(checks)
        assert factor == 1.0

    def test_single_non_unique_join(self):
        checks = [
            {"is_unique": False, "duplicate_pct": 80.0, "table": "t1", "columns": "key"},
        ]
        factor = SQLValidator._estimate_multiplication_factor(checks)
        # unique_perc = 20%, factor = 100/20 = 5.0
        assert factor == 5.0

    def test_multiple_non_unique_joins(self):
        checks = [
            {"is_unique": False, "duplicate_pct": 50.0, "table": "t1", "columns": "k1"},
            {"is_unique": False, "duplicate_pct": 80.0, "table": "t2", "columns": "k2"},
        ]
        factor = SQLValidator._estimate_multiplication_factor(checks)
        # 100/50 * 100/20 = 2 * 5 = 10
        assert factor == 10.0

    def test_mixed_unique_and_non_unique(self):
        checks = [
            {"is_unique": True, "duplicate_pct": 0.0, "table": "t1", "columns": "pk"},
            {"is_unique": False, "duplicate_pct": 75.0, "table": "t2", "columns": "fk"},
        ]
        factor = SQLValidator._estimate_multiplication_factor(checks)
        # unique * 100/25 = 1 * 4 = 4.0
        assert factor == 4.0

    def test_fully_non_unique(self):
        checks = [
            {"is_unique": False, "duplicate_pct": 100.0, "table": "t1", "columns": "k"},
        ]
        factor = SQLValidator._estimate_multiplication_factor(checks)
        assert factor == 100.0

    def test_empty_checks(self):
        assert SQLValidator._estimate_multiplication_factor([]) == 1.0


# ---------------------------------------------------------------------------
# Rewrite suggestion
# ---------------------------------------------------------------------------
class TestRewriteSuggestion:
    """Тесты генерации шаблонов переписывания."""

    def test_suggestion_format(self):
        join = {"schema": "dm", "table": "sales", "column": "manager_id"}
        suggestion = SQLValidator._generate_rewrite_suggestion(join)
        assert "ROW EXPLOSION" in suggestion
        assert "dm.sales" in suggestion
        assert "manager_id" in suggestion
        assert "DISTINCT" in suggestion
        assert "ЗАПРЕЩЕНО" in suggestion

    def test_suggestion_contains_template(self):
        join = {"schema": "hr", "table": "emp", "column": "dept_id"}
        suggestion = SQLValidator._generate_rewrite_suggestion(join)
        assert "SELECT DISTINCT" in suggestion
        assert "JOIN" in suggestion


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------
class TestValidationResultExplosion:
    """Тесты ValidationResult с данными row explosion."""

    def test_summary_with_multiplication_factor(self):
        result = ValidationResult(is_valid=True, mode=SQLMode.READ)
        result.join_checks = [
            {"table": "dm.t", "columns": "k", "is_unique": False, "duplicate_pct": 80.0}
        ]
        result.multiplication_factor = 5.0
        summary = result.summary()
        assert "5.0x" in summary

    def test_summary_with_rewrite_suggestions(self):
        result = ValidationResult(is_valid=False, mode=SQLMode.READ)
        result.rewrite_suggestions = ["Use DISTINCT subquery"]
        summary = result.summary()
        assert "Use DISTINCT subquery" in summary


# ---------------------------------------------------------------------------
# End-to-end _validate_read() with mock DB — threshold tests
# ---------------------------------------------------------------------------
class _MockDB:
    """Minimal mock for DatabaseManager used by SQLValidator."""

    def explain_query(self, sql: str) -> str:
        return "Seq Scan"

    def check_key_uniqueness(self, schema, table, columns):
        result = self._by_table.get(table, {"is_unique": False, "duplicate_pct": self._dup_pct})
        return dict(result)

    def __init__(self, dup_pct: float = 50.0, by_table=None):
        self._dup_pct = dup_pct
        self._by_table = by_table or {}


class TestValidateReadThresholds:
    """Тесты порогов BLOCK/SOFT BLOCK в _validate_read()."""

    def test_unique_join_passes(self):
        """All unique keys → is_valid=True, no errors."""
        db = _MockDB(dup_pct=0.0)
        db.check_key_uniqueness = lambda s, t, c: {"is_unique": True, "duplicate_pct": 0.0}
        v = SQLValidator(db)
        result = v.validate("SELECT * FROM hr.emp e JOIN hr.dept d ON d.id = e.dept_id WHERE 1=1")
        assert result.is_valid is True
        assert result.errors == []

    def test_no_join_passes(self):
        """SQL without JOIN → is_valid=True."""
        v = SQLValidator(_MockDB())
        result = v.validate("SELECT * FROM hr.emp WHERE id = 1")
        assert result.is_valid is True

    def test_hard_block_factor_above_2(self):
        """many-to-many join должен блокироваться как ROW EXPLOSION."""
        db = _MockDB(by_table={
            "emp": {"is_unique": False, "duplicate_pct": 80.0},
            "dept": {"is_unique": False, "duplicate_pct": 80.0},
        })
        v = SQLValidator(db)
        result = v.validate("SELECT * FROM hr.emp e JOIN hr.dept d ON d.id = e.dept_id WHERE 1=1")
        assert result.is_valid is False
        assert any("ROW EXPLOSION" in e for e in result.errors)

    def test_many_to_one_passes(self):
        """many-to-one join не должен блокироваться."""
        db = _MockDB(by_table={
            "emp": {"is_unique": False, "duplicate_pct": 10.0},
            "dept": {"is_unique": True, "duplicate_pct": 0.0},
        })
        v = SQLValidator(db)
        result = v.validate("SELECT * FROM hr.emp e JOIN hr.dept d ON d.id = e.dept_id WHERE 1=1")
        assert result.is_valid is True
        assert result.errors == []

    def test_one_to_many_is_join_risk(self):
        """one-to-many join должен идти в JOIN RISK."""
        db = _MockDB(by_table={
            "emp": {"is_unique": True, "duplicate_pct": 0.0},
            "dept": {"is_unique": False, "duplicate_pct": 10.0},
        })
        v = SQLValidator(db)
        result = v.validate("SELECT * FROM hr.emp e JOIN hr.dept d ON d.id = e.dept_id WHERE 1=1")
        assert result.is_valid is False
        assert any("JOIN RISK" in e for e in result.errors)

    def test_cross_join_always_blocks(self):
        """CROSS JOIN → always blocked regardless of DB uniqueness."""
        db = _MockDB(dup_pct=0.0)
        db.check_key_uniqueness = lambda s, t, c: {"is_unique": True, "duplicate_pct": 0.0}
        v = SQLValidator(db)
        result = v.validate("SELECT * FROM hr.emp e CROSS JOIN hr.dept WHERE 1=1")
        assert result.is_valid is False
