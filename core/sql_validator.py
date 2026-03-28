"""Р’Р°Р»РёРґР°С†РёСЏ SQL-Р·Р°РїСЂРѕСЃРѕРІ: СЃРёРЅС‚Р°РєСЃРёСЃ, EXPLAIN, РїСЂРѕРІРµСЂРєР° JOIN-РѕРІ."""

import logging
import re
from enum import Enum
from typing import Any

import sqlparse

logger = logging.getLogger(__name__)


class SQLMode(Enum):
    """Р РµР¶РёРј SQL-Р·Р°РїСЂРѕСЃР°."""
    READ = "READ"
    WRITE = "WRITE"
    DDL = "DDL"


class ValidationResult:
    """Р РµР·СѓР»СЊС‚Р°С‚ РІР°Р»РёРґР°С†РёРё SQL-Р·Р°РїСЂРѕСЃР°."""

    def __init__(self, is_valid: bool, mode: SQLMode) -> None:
        self.is_valid: bool = is_valid
        self.mode: SQLMode = mode
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.needs_confirmation: bool = False
        self.confirmation_message: str = ""
        self.explain_plan: str = ""
        self.join_checks: list[dict[str, Any]] = []
        self.rewrite_suggestions: list[str] = []
        self.multiplication_factor: float = 1.0

    def add_warning(self, msg: str) -> None:
        """Р”РѕР±Р°РІРёС‚СЊ РїСЂРµРґСѓРїСЂРµР¶РґРµРЅРёРµ."""
        self.warnings.append(msg)
        logger.warning("SQL validation warning: %s", msg)

    def add_error(self, msg: str) -> None:
        """Р”РѕР±Р°РІРёС‚СЊ РѕС€РёР±РєСѓ Рё РїРѕРјРµС‚РёС‚СЊ РєР°Рє РЅРµРІР°Р»РёРґРЅС‹Р№."""
        self.errors.append(msg)
        self.is_valid = False
        logger.error("SQL validation error: %s", msg)

    def require_confirmation(self, msg: str) -> None:
        """Р—Р°РїСЂРѕСЃРёС‚СЊ РїРѕРґС‚РІРµСЂР¶РґРµРЅРёРµ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ."""
        self.needs_confirmation = True
        self.confirmation_message = msg

    def summary(self) -> str:
        """РўРµРєСЃС‚РѕРІРѕРµ СЂРµР·СЋРјРµ РІР°Р»РёРґР°С†РёРё."""
        lines = [f"Р РµР¶РёРј: {self.mode.value}", f"Р’Р°Р»РёРґРЅРѕ: {'РґР°' if self.is_valid else 'РЅРµС‚'}"]
        if self.errors:
            lines.append("РћС€РёР±РєРё:")
            for e in self.errors:
                lines.append(f"  вњ— {e}")
        if self.warnings:
            lines.append("РџСЂРµРґСѓРїСЂРµР¶РґРµРЅРёСЏ:")
            for w in self.warnings:
                lines.append(f"  вљ  {w}")
        if self.needs_confirmation:
            lines.append(f"РўСЂРµР±СѓРµС‚СЃСЏ РїРѕРґС‚РІРµСЂР¶РґРµРЅРёРµ: {self.confirmation_message}")
        if self.join_checks:
            lines.append("РџСЂРѕРІРµСЂРєР° JOIN-РѕРІ:")
            for jc in self.join_checks:
                cardinality = jc.get("cardinality", "unknown")
                join_expr = jc.get("join", f"{jc.get('table', '?')}.({jc.get('columns', '?')})")
                status = "вњ“ safe" if jc.get("is_safe") else "вњ— risky"
                lines.append(f"  {join_expr}: {cardinality} ({status})")
            if self.multiplication_factor > 1.0:
                lines.append(f"  Multiplication factor: {self.multiplication_factor:.1f}x")
        if self.rewrite_suggestions:
            lines.append("Р РµРєРѕРјРµРЅРґР°С†РёРё РїРѕ РїРµСЂРµРїРёСЃС‹РІР°РЅРёСЋ:")
            for s in self.rewrite_suggestions:
                lines.append(f"  {s}")
        return "\n".join(lines)


def detect_mode(sql: str) -> SQLMode:
    """РћРїСЂРµРґРµР»РёС‚СЊ СЂРµР¶РёРј SQL-Р·Р°РїСЂРѕСЃР°.

    РџРѕРґРґРµСЂР¶РёРІР°РµС‚ CTE: ``WITH cte AS (...) INSERT INTO ...`` РѕРїСЂРµРґРµР»СЏРµС‚СЃСЏ РєР°Рє WRITE,
    Р° РЅРµ РєР°Рє READ.

    Args:
        sql: SQL-Р·Р°РїСЂРѕСЃ.

    Returns:
        SQLMode (READ, WRITE РёР»Рё DDL).
    """
    normalized = sql.strip().upper()
    # РЈР±РёСЂР°РµРј РєРѕРјРјРµРЅС‚Р°СЂРёРё РІ РЅР°С‡Р°Р»Рµ
    for line in normalized.split("\n"):
        line = line.strip()
        if line and not line.startswith("--"):
            normalized = line
            break

    # Р•СЃР»Рё Р·Р°РїСЂРѕСЃ РЅР°С‡РёРЅР°РµС‚СЃСЏ СЃ WITH (CTE), РёС‰РµРј РѕСЃРЅРѕРІРЅРѕР№ statement РїРѕСЃР»Рµ CTE
    if normalized.startswith("WITH"):
        # РќР°Р№С‚Рё РѕСЃРЅРѕРІРЅРѕР№ statement РїРѕСЃР»Рµ РІСЃРµС… CTE-РѕРїСЂРµРґРµР»РµРЅРёР№.
        # РС‰РµРј РїРѕСЃР»РµРґРЅРёР№ Р·Р°РєСЂС‹РІР°СЋС‰РёР№ ')' CTE, Р·Р° РєРѕС‚РѕСЂС‹Рј РёРґС‘С‚ РєР»СЋС‡РµРІРѕРµ СЃР»РѕРІРѕ.
        main_stmt = re.search(
            r'\)\s*(SELECT|INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE)\b',
            normalized,
        )
        if main_stmt:
            keyword = main_stmt.group(1)
            if keyword in ("CREATE", "ALTER", "DROP", "TRUNCATE"):
                return SQLMode.DDL
            if keyword in ("INSERT", "UPDATE", "DELETE", "MERGE"):
                return SQLMode.WRITE
            return SQLMode.READ

    if normalized.startswith(("CREATE", "ALTER", "DROP", "TRUNCATE")):
        return SQLMode.DDL
    if normalized.startswith(("INSERT", "UPDATE", "DELETE", "MERGE")):
        return SQLMode.WRITE
    return SQLMode.READ


def _build_alias_map(sql: str) -> dict[str, tuple[str, str]]:
    """РџРѕСЃС‚СЂРѕРёС‚СЊ РјР°РїРїРёРЅРі alias в†’ (schema, table) РёР· FROM Рё JOIN.

    РћР±СЂР°Р±Р°С‚С‹РІР°РµС‚:
    - FROM schema.table alias
    - FROM schema.table AS alias
    - JOIN schema.table alias ON ...
    - FROM schema.table (Р±РµР· alias в†’ table name РєР°Рє alias)
    """
    alias_map: dict[str, tuple[str, str]] = {}

    _SQL_KEYWORDS = frozenset((
        "ON", "WHERE", "SET", "LEFT", "RIGHT", "INNER", "OUTER",
        "FULL", "CROSS", "NATURAL", "JOIN", "GROUP", "ORDER",
        "HAVING", "LIMIT", "UNION", "EXCEPT", "INTERSECT",
    ))

    # РџР°С‚С‚РµСЂРЅ: schema.table СЃ РѕРїС†РёРѕРЅР°Р»СЊРЅС‹Рј alias (AS keyword РѕРїС†РёРѕРЅР°Р»РµРЅ)
    table_ref = re.compile(
        r'(?:FROM|JOIN)\s+'
        r'["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?'
        r'(?:\s+(?:AS\s+)?(\w+))?',
        re.IGNORECASE,
    )

    for m in table_ref.finditer(sql):
        schema, table = m.group(1), m.group(2)
        alias = m.group(3)
        if alias and alias.upper() not in _SQL_KEYWORDS:
            alias_map[alias.lower()] = (schema, table)
        alias_map[table.lower()] = (schema, table)

    # РћР±СЂР°Р±РѕС‚РєР° comma-separated С‚Р°Р±Р»РёС† РІ FROM (implicit JOIN):
    # FROM hr.emp e, hr.dept d, hr.loc l
    comma_ref = re.compile(
        r',\s*["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?'
        r'(?:\s+(?:AS\s+)?(\w+))?',
        re.IGNORECASE,
    )
    # РС‰РµРј С‚РѕР»СЊРєРѕ РІРЅСѓС‚СЂРё FROM-РєР»Р°СѓР·С‹ (РґРѕ WHERE/JOIN/GROUP Рё С‚.Рґ.)
    from_clause_pat = re.compile(
        r'\bFROM\s+(.*?)(?=\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bJOIN\b|\bHAVING\b|\Z)',
        re.IGNORECASE | re.DOTALL,
    )
    from_m = from_clause_pat.search(sql)
    if from_m:
        from_text = from_m.group(1)
        for cm in comma_ref.finditer(from_text):
            schema, table = cm.group(1), cm.group(2)
            alias = cm.group(3)
            if alias and alias.upper() not in _SQL_KEYWORDS:
                alias_map[alias.lower()] = (schema, table)
            alias_map[table.lower()] = (schema, table)

    return alias_map


def _find_subquery_join_aliases(sql: str) -> set[str]:
    """РќР°Р№С‚Рё Р°Р»РёР°СЃС‹ JOIN-РѕРІ, РєРѕС‚РѕСЂС‹Рµ РёСЃРїРѕР»СЊР·СѓСЋС‚ РїРѕРґР·Р°РїСЂРѕСЃС‹.

    РџРѕРґР·Р°РїСЂРѕСЃ РІ JOIN (РЅР°РїСЂРёРјРµСЂ, JOIN (SELECT DISTINCT ...) alias ON ...)
    СЃС‡РёС‚Р°РµС‚СЃСЏ СѓР¶Рµ Р±РµР·РѕРїР°СЃРЅС‹Рј вЂ” DISTINCT/GROUP BY РІРЅСѓС‚СЂРё РїСЂРµРґРѕС‚РІСЂР°С‰Р°РµС‚ РґСѓР±Р»Рё.

    Returns:
        РњРЅРѕР¶РµСЃС‚РІРѕ Р°Р»РёР°СЃРѕРІ (РІ РЅРёР¶РЅРµРј СЂРµРіРёСЃС‚СЂРµ), РєРѕС‚РѕСЂС‹Рµ СЏРІР»СЏСЋС‚СЃСЏ РїРѕРґР·Р°РїСЂРѕСЃР°РјРё.
    """
    safe_aliases: set[str] = set()
    # РџР°С‚С‚РµСЂРЅ: JOIN (SELECT ...) alias ON
    # РС‰РµРј JOIN Р·Р° РєРѕС‚РѕСЂС‹Рј СЃР»РµРґСѓРµС‚ РѕС‚РєСЂС‹РІР°СЋС‰Р°СЏ СЃРєРѕР±РєР° (РїРѕРґР·Р°РїСЂРѕСЃ)
    subq_pattern = re.compile(
        r'JOIN\s*\(\s*SELECT\b.*?\)\s+(?:AS\s+)?(\w+)\s+ON\b',
        re.IGNORECASE | re.DOTALL,
    )
    for m in subq_pattern.finditer(sql):
        safe_aliases.add(m.group(1).lower())
    return safe_aliases


def _extract_join_conditions(sql: str) -> list[dict[str, str]]:
    """РР·РІР»РµС‡СЊ С‚Р°Р±Р»РёС†С‹ Рё РєРѕР»РѕРЅРєРё РёР· JOIN СѓСЃР»РѕРІРёР№.

    РџРѕРґРґРµСЂР¶РёРІР°РµС‚:
    - Explicit JOINs СЃ Р°Р»РёР°СЃР°РјРё (JOIN schema.table alias ON alias.col = ...)
    - LEFT/RIGHT/FULL/INNER JOIN
    - Multi-column ON (AND conditions)
    - CROSS JOIN (Р±РµР· ON вЂ” РІСЃРµРіРґР° explosion)
    - Implicit JOINs (FROM t1, t2 WHERE t1.col = t2.col)
    - РћР±Рµ СЃС‚РѕСЂРѕРЅС‹ JOIN (left Рё right)
    - РџРѕРґР·Р°РїСЂРѕСЃС‹ РІ JOIN: JOIN (SELECT DISTINCT ...) alias вЂ” РїСЂРѕРїСѓСЃРєР°СЋС‚СЃСЏ РєР°Рє Р±РµР·РѕРїР°СЃРЅС‹Рµ

    Returns:
        РЎРїРёСЃРѕРє СЃР»РѕРІР°СЂРµР№ СЃ schema, table, column РґР»СЏ РєР°Р¶РґРѕР№ СЃС‚РѕСЂРѕРЅС‹ РєР°Р¶РґРѕРіРѕ JOIN.
    """
    joins: list[dict[str, str]] = []
    subquery_aliases = _find_subquery_join_aliases(sql)

    # 0. РР·РІР»РµС‡СЊ CTE-С‚РµР»Р° Рё РѕР±СЂР°Р±РѕС‚Р°С‚СЊ РёС… СЂРµРєСѓСЂСЃРёРІРЅРѕ
    cte_pattern = re.compile(
        r'\bWITH\b\s+(?:RECURSIVE\s+)?'
        r'(.*?)\bSELECT\b',
        re.IGNORECASE | re.DOTALL,
    )
    cte_body_pattern = re.compile(
        r'\w+\s+AS\s*\(\s*(.*?)\s*\)',
        re.IGNORECASE | re.DOTALL,
    )
    cte_match = cte_pattern.match(sql.strip())
    if cte_match:
        cte_defs = cte_match.group(1)
        for body_match in cte_body_pattern.finditer(cte_defs):
            cte_body = body_match.group(1)
            joins.extend(_extract_join_conditions(cte_body))

    # 1. РџРѕСЃС‚СЂРѕРёС‚СЊ alias map
    alias_map = _build_alias_map(sql)

    def _resolve(ref: str) -> tuple[str, str] | None:
        """Resolve alias/table name to (schema, table)."""
        return alias_map.get(ref.lower())

    # 2. Explicit JOINs: extract ON conditions
    # РќР°Р№С‚Рё РІСЃРµ JOIN ... ON ... Р±Р»РѕРєРё
    join_on_pattern = re.compile(
        r'JOIN\s+["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?'
        r'\s+(?:(?:AS\s+)?(\w+)\s+)?ON\s+(.*?)(?=\bJOIN\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bUNION\b|\bHAVING\b|;|\)|\Z)',
        re.IGNORECASE | re.DOTALL,
    )

    for m in join_on_pattern.finditer(sql):
        join_schema, join_table = m.group(1), m.group(2)
        on_clause = m.group(4)

        # РР·РІР»РµС‡СЊ РІСЃРµ equality conditions РёР· ON clause (РїРѕРґРґРµСЂР¶РєР° multi-column)
        eq_pattern = re.compile(
            r'["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?'
            r'\s*=\s*'
            r'["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
        )
        for eq in eq_pattern.finditer(on_clause):
            left_ref, left_col = eq.group(1), eq.group(2)
            right_ref, right_col = eq.group(3), eq.group(4)

            # РџСЂРѕРїСѓСЃРєР°РµРј СЃС‚РѕСЂРѕРЅС‹, РєРѕС‚РѕСЂС‹Рµ СЃСЃС‹Р»Р°СЋС‚СЃСЏ РЅР° РїРѕРґР·Р°РїСЂРѕСЃ (СѓР¶Рµ Р±РµР·РѕРїР°СЃРЅС‹)
            left_is_subquery = left_ref.lower() in subquery_aliases
            right_is_subquery = right_ref.lower() in subquery_aliases

            # Resolve РѕР±Рµ СЃС‚РѕСЂРѕРЅС‹ С‡РµСЂРµР· alias map
            left_resolved = _resolve(left_ref)
            right_resolved = _resolve(right_ref)

            if left_resolved and not left_is_subquery:
                joins.append({
                    "schema": left_resolved[0],
                    "table": left_resolved[1],
                    "column": left_col,
                })
            if right_resolved and not right_is_subquery:
                joins.append({
                    "schema": right_resolved[0],
                    "table": right_resolved[1],
                    "column": right_col,
                })

    # 3. CROSS JOIN (РЅРµС‚ ON в†’ РіР°СЂР°РЅС‚РёСЂРѕРІР°РЅРЅС‹Р№ explosion)
    cross_pattern = re.compile(
        r'CROSS\s+JOIN\s+["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
        re.IGNORECASE,
    )
    for m in cross_pattern.finditer(sql):
        joins.append({
            "schema": m.group(1),
            "table": m.group(2),
            "column": "__CROSS_JOIN__",
        })

    # 4. Implicit JOINs: FROM t1, t2 WHERE t1.col = t2.col
    # Р”РµС‚РµРєС‚РёСЂСѓРµРј comma-separated tables РІ FROM
    from_pattern = re.compile(
        r'\bFROM\s+(.*?)(?=\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bJOIN\b|\Z)',
        re.IGNORECASE | re.DOTALL,
    )
    from_match = from_pattern.search(sql)
    if from_match:
        from_clause = from_match.group(1)
        # РС‰РµРј comma-separated tables (РјРёРЅРёРјСѓРј 2 С‡РµСЂРµР· Р·Р°РїСЏС‚СѓСЋ)
        table_refs = [t.strip() for t in from_clause.split(",") if t.strip()]
        if len(table_refs) >= 2:
            # Р•СЃС‚СЊ implicit join вЂ” РёС‰РµРј equality conditions РІ WHERE
            where_pattern = re.compile(
                r'\bWHERE\b\s+(.*?)(?=\bGROUP\b|\bORDER\b|\bLIMIT\b|\bUNION\b|\Z)',
                re.IGNORECASE | re.DOTALL,
            )
            where_match = where_pattern.search(sql)
            if where_match:
                where_clause = where_match.group(1)
                eq_pattern = re.compile(
                    r'["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?'
                    r'\s*=\s*'
                    r'["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
                )
                for eq in eq_pattern.finditer(where_clause):
                    left_ref, left_col = eq.group(1), eq.group(2)
                    right_ref, right_col = eq.group(3), eq.group(4)

                    left_resolved = _resolve(left_ref)
                    right_resolved = _resolve(right_ref)

                    if left_resolved and left_ref.lower() not in subquery_aliases:
                        joins.append({
                            "schema": left_resolved[0],
                            "table": left_resolved[1],
                            "column": left_col,
                        })
                    if right_resolved and right_ref.lower() not in subquery_aliases:
                        joins.append({
                            "schema": right_resolved[0],
                            "table": right_resolved[1],
                            "column": right_col,
                        })

    # 5. Р”РµРґСѓРїР»РёРєР°С†РёСЏ (РѕРґРЅР° Рё С‚Р° Р¶Рµ schema.table.column РјРѕР¶РµС‚ РїРѕСЏРІРёС‚СЊСЃСЏ РёР· СЂР°Р·РЅС‹С… РјРµСЃС‚)
    seen = set()
    unique_joins = []
    for j in joins:
        key = (j["schema"].lower(), j["table"].lower(), j["column"].lower())
        if key not in seen:
            seen.add(key)
            unique_joins.append(j)

    return unique_joins


def _extract_join_pairs(sql: str) -> list[dict[str, Any]]:
    """РР·РІР»РµС‡СЊ РїР°СЂС‹ СЃС‚РѕСЂРѕРЅ JOIN СЃ СѓРєР°Р·Р°РЅРёРµРј РїСЂРёСЃРѕРµРґРёРЅСЏРµРјРѕР№ СЃС‚РѕСЂРѕРЅС‹."""
    pairs: list[dict[str, Any]] = []
    subquery_aliases = _find_subquery_join_aliases(sql)
    alias_map = _build_alias_map(sql)

    def _resolve(ref: str) -> tuple[str, str] | None:
        return alias_map.get(ref.lower())

    join_on_pattern = re.compile(
        r'JOIN\s+["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?'
        r'\s+(?:(?:AS\s+)?(\w+)\s+)?ON\s+(.*?)(?=\bJOIN\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bUNION\b|\bHAVING\b|;|\)|\Z)',
        re.IGNORECASE | re.DOTALL,
    )
    eq_pattern = re.compile(
        r'["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?'
        r'\s*=\s*'
        r'["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
    )

    for match in join_on_pattern.finditer(sql):
        join_table = match.group(2)
        join_alias = (match.group(3) or join_table).lower()
        on_clause = match.group(4)

        for eq in eq_pattern.finditer(on_clause):
            left_ref, left_col = eq.group(1), eq.group(2)
            right_ref, right_col = eq.group(3), eq.group(4)

            if left_ref.lower() in subquery_aliases or right_ref.lower() in subquery_aliases:
                continue

            left_resolved = _resolve(left_ref)
            right_resolved = _resolve(right_ref)
            if not left_resolved or not right_resolved:
                continue

            joined_side = "left" if left_ref.lower() == join_alias else "right"
            if left_ref.lower() != join_alias and right_ref.lower() != join_alias:
                joined_side = "right"

            pairs.append({
                "type": "join",
                "joined_side": joined_side,
                "left": {
                    "schema": left_resolved[0],
                    "table": left_resolved[1],
                    "column": left_col,
                },
                "right": {
                    "schema": right_resolved[0],
                    "table": right_resolved[1],
                    "column": right_col,
                },
            })

    cross_pattern = re.compile(
        r'CROSS\s+JOIN\s+["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
        re.IGNORECASE,
    )
    for match in cross_pattern.finditer(sql):
        pairs.append({
            "type": "cross_join",
            "joined_side": "right",
            "left": None,
            "right": {
                "schema": match.group(1),
                "table": match.group(2),
                "column": "__CROSS_JOIN__",
            },
        })

    return pairs


def _has_where_or_limit(sql: str) -> bool:
    """РџСЂРѕРІРµСЂРёС‚СЊ РЅР°Р»РёС‡РёРµ WHERE РёР»Рё LIMIT РІ РѕСЃРЅРѕРІРЅРѕРј Р·Р°РїСЂРѕСЃРµ.

    РЈР±РёСЂР°РµС‚ РїРѕРґР·Р°РїСЂРѕСЃС‹ РІ СЃРєРѕР±РєР°С… Рё СЃС‚СЂРѕРєРѕРІС‹Рµ Р»РёС‚РµСЂР°Р»С‹ РїРµСЂРµРґ РїСЂРѕРІРµСЂРєРѕР№,
    С‡С‚РѕР±С‹ РёР·Р±РµР¶Р°С‚СЊ Р»РѕР¶РЅС‹С… СЃСЂР°Р±Р°С‚С‹РІР°РЅРёР№.
    """
    # РЈР±РёСЂР°РµРј СЃС‚СЂРѕРєРѕРІС‹Рµ Р»РёС‚РµСЂР°Р»С‹
    cleaned = re.sub(r"'[^']*'", "''", sql)
    # РЈР±РёСЂР°РµРј СЃРѕРґРµСЂР¶РёРјРѕРµ РїРѕРґР·Р°РїСЂРѕСЃРѕРІ РІ СЃРєРѕР±РєР°С… (СЂРµРєСѓСЂСЃРёРІРЅРѕ, РґРѕ 5 СѓСЂРѕРІРЅРµР№)
    for _ in range(5):
        prev = cleaned
        cleaned = re.sub(r'\([^()]*\)', '()', cleaned)
        if cleaned == prev:
            break
    upper = cleaned.upper()
    return "WHERE" in upper or "LIMIT" in upper


def _has_top_level_where(sql: str) -> bool:
    """РџСЂРѕРІРµСЂРёС‚СЊ РЅР°Р»РёС‡РёРµ WHERE РЅР° РІРµСЂС…РЅРµРј СѓСЂРѕРІРЅРµ statement.

    РќРµ СѓС‡РёС‚С‹РІР°РµС‚ WHERE РІ СЃС‚СЂРѕРєРѕРІС‹С… Р»РёС‚РµСЂР°Р»Р°С…, РєРѕРјРјРµРЅС‚Р°СЂРёСЏС… Рё РїРѕРґР·Р°РїСЂРѕСЃР°С….
    """
    statements = sqlparse.parse(sql)
    if not statements:
        return False

    for token in statements[0].tokens:
        if isinstance(token, sqlparse.sql.Where):
            return True
    return False


class SQLValidator:
    """Р’Р°Р»РёРґР°С‚РѕСЂ SQL-Р·Р°РїСЂРѕСЃРѕРІ РґР»СЏ Greenplum."""

    def __init__(self, db_manager: Any, schema_loader: Any = None) -> None:
        """РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ РІР°Р»РёРґР°С‚РѕСЂР°.

        Args:
            db_manager: Р­РєР·РµРјРїР»СЏСЂ DatabaseManager РґР»СЏ РІС‹РїРѕР»РЅРµРЅРёСЏ EXPLAIN Рё РїСЂРѕРІРµСЂРѕРє.
            schema_loader: РћРїС†РёРѕРЅР°Р»СЊРЅС‹Р№ SchemaLoader РґР»СЏ CSV-first РїСЂРѕРІРµСЂРєРё РєР»СЋС‡РµР№.
        """
        self._db = db_manager
        self._schema_loader = schema_loader

    def validate(self, sql: str) -> ValidationResult:
        """Р’Р°Р»РёРґРёСЂРѕРІР°С‚СЊ SQL-Р·Р°РїСЂРѕСЃ.

        Args:
            sql: SQL-Р·Р°РїСЂРѕСЃ.

        Returns:
            ValidationResult СЃ СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё РїСЂРѕРІРµСЂРєРё.
        """
        mode = detect_mode(sql)
        result = ValidationResult(is_valid=True, mode=mode)

        logger.info("Р’Р°Р»РёРґР°С†РёСЏ SQL (СЂРµР¶РёРј %s): %s", mode.value, sql[:200])

        if mode == SQLMode.READ:
            self._validate_read(sql, result)
        elif mode == SQLMode.WRITE:
            self._validate_write(sql, result)
        elif mode == SQLMode.DDL:
            self._validate_ddl(sql, result)

        return result

    @staticmethod
    def _estimate_multiplication_factor(join_checks: list[dict[str, Any]]) -> float:
        """РћС†РµРЅРєР° РјРЅРѕР¶РёС‚РµР»СЏ СЂР°Р·РјРЅРѕР¶РµРЅРёСЏ СЃС‚СЂРѕРє РёР·-Р·Р° РЅРµСѓРЅРёРєР°Р»СЊРЅС‹С… JOIN.

        Р”Р»СЏ РєР°Р¶РґРѕРіРѕ РЅРµСѓРЅРёРєР°Р»СЊРЅРѕРіРѕ JOIN: factor = 100 / unique_perc.
        РћР±С‰РёР№ factor вЂ” РїСЂРѕРёР·РІРµРґРµРЅРёРµ РїРѕ РІСЃРµРј JOIN.
        """
        total = 1.0
        for jc in join_checks:
            if "factor" in jc:
                total *= jc["factor"]
                continue
            if not jc["is_unique"]:
                unique_perc = 100.0 - jc["duplicate_pct"]
                if unique_perc > 0:
                    total *= min(100.0 / unique_perc, 100.0)
                else:
                    total *= 100.0
        return total

    @staticmethod
    def _generate_rewrite_suggestion(join: dict[str, str]) -> str:
        """РЎРіРµРЅРµСЂРёСЂРѕРІР°С‚СЊ РєРѕРЅРєСЂРµС‚РЅС‹Р№ С€Р°Р±Р»РѕРЅ РїРµСЂРµРїРёСЃС‹РІР°РЅРёСЏ JOIN СЃ РґРёР°РіРЅРѕСЃС‚РёРєРѕР№ РїСЂРёС‡РёРЅС‹ РґСѓР±Р»РµР№."""
        schema, table, column = join["schema"], join["table"], join["column"]
        return (
            f"ROW EXPLOSION: JOIN РєР»СЋС‡ {schema}.{table}.{column} РЅРµ СѓРЅРёРєР°Р»РµРЅ.\n"
            f"РћР‘РЇР—РђРўР•Р›Р¬РќРћ: РІС‹Р·РѕРІРё get_sample РґР»СЏ {schema}.{table} Рё РёР·СѓС‡Рё РїСЂРёС‡РёРЅСѓ РґСѓР±Р»РµР№.\n"
            f"Р—Р°С‚РµРј РІС‹Р±РµСЂРё СЃС‚СЂР°С‚РµРіРёСЋ РёСЃРїСЂР°РІР»РµРЅРёСЏ:\n"
            f"  Р’Р°СЂРёР°РЅС‚ 1 вЂ” СЃС‚Р°С‚СѓСЃС‹/РІРµСЂСЃРёРё (active/liquidated, Р°РєС‚СѓР°Р»СЊРЅР°СЏ/Р°СЂС…РёРІРЅР°СЏ Р·Р°РїРёСЃСЊ):\n"
            f"    JOIN {schema}.{table} alias ON ... = alias.{column} WHERE alias.status = 'active'\n"
            f"    РёР»Рё: JOIN (SELECT DISTINCT ON ({column}) * FROM {schema}.{table} "
            f"ORDER BY {column}, <РґР°С‚Р°> DESC) alias ON ...\n"
            f"  Р’Р°СЂРёР°РЅС‚ 2 вЂ” РЅРµСЃРєРѕР»СЊРєРѕ С„Р°РєС‚РѕРІ РЅР° РѕР±СЉРµРєС‚ (С‚СЂР°РЅР·Р°РєС†РёРё, РїР»Р°С‚РµР¶Рё, СЃРѕР±С‹С‚РёСЏ):\n"
            f"    JOIN (SELECT {column}, SUM(<СЃСѓРјРјР°>) AS total FROM {schema}.{table} "
            f"GROUP BY {column}) alias ON ...\n"
            f"  Р’Р°СЂРёР°РЅС‚ 3 вЂ” С‚РµС…РЅРёС‡РµСЃРєРёРµ РґСѓР±Р»Рё (РїРѕР»РЅРѕСЃС‚СЊСЋ РёРґРµРЅС‚РёС‡РЅС‹Рµ СЃС‚СЂРѕРєРё, Р±Р°Рі РґР°РЅРЅС‹С…):\n"
            f"    JOIN (SELECT DISTINCT {column}, <РЅСѓР¶РЅС‹Рµ_РєРѕР»РѕРЅРєРё> FROM {schema}.{table}) "
            f"alias ON ... = alias.{column}\n"
            f"Р—РђРџР Р•Р©Р•РќРћ: РґРѕР±Р°РІР»СЏС‚СЊ DISTINCT Рє РІРЅРµС€РЅРµРјСѓ SELECT вЂ” СЌС‚Рѕ РјР°СЃРєРёСЂСѓРµС‚ РїСЂРѕР±Р»РµРјСѓ.\n"
            f"Р—РђРџР Р•Р©Р•РќРћ: РїСЂРёРјРµРЅСЏС‚СЊ DISTINCT Р±РµР· РїРѕРЅРёРјР°РЅРёСЏ РїСЂРёС‡РёРЅС‹ РґСѓР±Р»РµР№."
        )

    def _check_key_uniqueness(
        self, schema: str, table: str, columns: list[str],
    ) -> dict[str, Any]:
        """РџСЂРѕРІРµСЂРёС‚СЊ СѓРЅРёРєР°Р»СЊРЅРѕСЃС‚СЊ РєР»СЋС‡Р°: СЃРЅР°С‡Р°Р»Р° CSV, РїРѕС‚РѕРј DB fallback."""
        if self._schema_loader is not None:
            csv_result = self._schema_loader.check_key_uniqueness(schema, table, columns)
            if csv_result.get("status") in {"safe", "risky"}:
                return {
                    "is_unique": csv_result["is_unique"],
                    "duplicate_pct": csv_result["duplicate_pct"],
                    "status": csv_result["status"],
                }
        db_result = self._db.check_key_uniqueness(schema, table, columns)
        db_result["status"] = "safe" if db_result["is_unique"] else "risky"
        return db_result

    def _validate_read(self, sql: str, result: ValidationResult) -> None:
        """Р’Р°Р»РёРґР°С†РёСЏ SELECT-Р·Р°РїСЂРѕСЃРѕРІ."""
        # 1. EXPLAIN вЂ” СЃРёРЅС‚Р°РєСЃРёС‡РµСЃРєР°СЏ РїСЂРѕРІРµСЂРєР°
        try:
            plan = self._db.explain_query(sql)
            result.explain_plan = plan
        except Exception as e:
            result.add_error(f"РЎРёРЅС‚Р°РєСЃРёС‡РµСЃРєР°СЏ РѕС€РёР±РєР° (EXPLAIN): {e}")
            return

        # 2. РџСЂРѕРІРµСЂРєР° JOIN-РѕРІ РЅР° РєР°СЂРґРёРЅР°Р»СЊРЅРѕСЃС‚СЊ
        join_pairs = _extract_join_pairs(sql)
        for pair in join_pairs:
            if pair["type"] == "cross_join":
                joined = pair["right"]
                result.join_checks.append({
                    "join": f"CROSS JOIN {joined['schema']}.{joined['table']}",
                    "table": f"{joined['schema']}.{joined['table']}",
                    "columns": "CROSS JOIN",
                    "cardinality": "cross-join",
                    "is_unique": False,
                    "is_safe": False,
                    "duplicate_pct": 100.0,
                    "factor": 100.0,
                })
                result.rewrite_suggestions.append(
                    f"ROW EXPLOSION: CROSS JOIN СЃ {joined['schema']}.{joined['table']} "
                    "СЃРѕР·РґР°С‘С‚ РґРµРєР°СЂС‚РѕРІРѕ РїСЂРѕРёР·РІРµРґРµРЅРёРµ. Р—Р°РјРµРЅРё РЅР° РѕР±С‹С‡РЅС‹Р№ JOIN СЃ СѓСЃР»РѕРІРёРµРј."
                )
                continue

            left = pair["left"]
            right = pair["right"]
            joined_side = pair["joined_side"]
            joined = left if joined_side == "left" else right
            existing = right if joined_side == "left" else left

            try:
                existing_check = self._check_key_uniqueness(
                    existing["schema"], existing["table"], [existing["column"]]
                )
                joined_check = self._check_key_uniqueness(
                    joined["schema"], joined["table"], [joined["column"]]
                )
            except Exception as e:
                logger.warning("РќРµ СѓРґР°Р»РѕСЃСЊ РїСЂРѕРІРµСЂРёС‚СЊ РєР°СЂРґРёРЅР°Р»СЊРЅРѕСЃС‚СЊ JOIN: %s", e)
                continue

            existing_status = existing_check.get("status", "unknown")
            joined_status = joined_check.get("status", "unknown")
            existing_unique = existing_status == "safe"
            joined_unique = joined_status == "safe"
            joined_dup = joined_check.get("duplicate_pct")
            joined_factor = 2.0
            if joined_dup is not None:
                joined_unique_perc = max(0.01, 100.0 - float(joined_dup))
                joined_factor = min(100.0 / joined_unique_perc, 100.0)

            if existing_unique and joined_unique:
                cardinality = "one-to-one"
                is_safe = True
                factor = 1.0
            elif joined_unique:
                cardinality = "many-to-one"
                is_safe = True
                factor = 1.0
            elif existing_unique and joined_status == "risky":
                cardinality = "one-to-many"
                is_safe = False
                factor = joined_factor
            elif existing_status == "risky" and joined_status == "risky":
                cardinality = "many-to-many"
                is_safe = False
                existing_dup = existing_check.get("duplicate_pct", 50.0)
                existing_unique_perc = max(0.01, 100.0 - float(existing_dup))
                factor = min((100.0 / existing_unique_perc) * joined_factor, 100.0)
            else:
                cardinality = "unknown"
                is_safe = False
                factor = max(joined_factor, 2.0)

            join_label = (
                f"{left['schema']}.{left['table']}.{left['column']} = "
                f"{right['schema']}.{right['table']}.{right['column']}"
            )
            check_info = {
                "join": join_label,
                "table": f"{joined['schema']}.{joined['table']}",
                "columns": joined["column"],
                "cardinality": cardinality,
                "is_unique": joined_unique,
                "is_safe": is_safe,
                "duplicate_pct": joined_check.get("duplicate_pct"),
                "factor": factor,
            }
            result.join_checks.append(check_info)

            if not is_safe:
                result.rewrite_suggestions.append(self._generate_rewrite_suggestion(joined))

        # 3. РћС†РµРЅРєР° multiplication factor Рё СЂРµС€РµРЅРёРµ pass/warn/block
        if result.join_checks:
            factor = self._estimate_multiplication_factor(result.join_checks)
            result.multiplication_factor = factor

            hard_risk = [
                jc for jc in result.join_checks if jc.get("cardinality") in {"many-to-many", "cross-join"}
            ]
            soft_risk = [
                jc for jc in result.join_checks if jc.get("cardinality") in {"one-to-many", "unknown"}
            ]

            if hard_risk:
                details = "; ".join(f"{jc['join']} [{jc['cardinality']}]" for jc in hard_risk)
                result.add_error(
                    f"ROW EXPLOSION (factor={factor:.1f}x): {details}. "
                    "РћР±РЅР°СЂСѓР¶РµРЅ JOIN, РєРѕС‚РѕСЂС‹Р№ СЂР°Р·РјРЅРѕР¶Р°РµС‚ СЃС‚СЂРѕРєРё СЂРµР·СѓР»СЊС‚Р°С‚Р°.\n"
                    + "\n".join(result.rewrite_suggestions)
                )
            elif soft_risk:
                details = "; ".join(f"{jc['join']} [{jc['cardinality']}]" for jc in soft_risk)
                result.add_error(
                    f"JOIN RISK (factor={factor:.1f}x): {details}. "
                    "РџСЂРёСЃРѕРµРґРёРЅСЏРµРјР°СЏ СЃС‚РѕСЂРѕРЅР° РЅРµ РґРѕРєР°Р·Р°РЅР° РєР°Рє СѓРЅРёРєР°Р»СЊРЅР°СЏ Рё РјРѕР¶РµС‚ СЂР°Р·РјРЅРѕР¶РёС‚СЊ СЃС‚СЂРѕРєРё."
                )

        # 4. РџСЂРµРґСѓРїСЂРµР¶РґРµРЅРёРµ РµСЃР»Рё РЅРµС‚ WHERE/LIMIT РґР»СЏ Р±РѕР»СЊС€РёС… С‚Р°Р±Р»РёС†
        if not _has_where_or_limit(sql):
            result.add_warning(
                "Р—Р°РїСЂРѕСЃ Р±РµР· WHERE/LIMIT. Р”Р»СЏ Р±РѕР»СЊС€РёС… С‚Р°Р±Р»РёС† СЌС‚Рѕ РјРѕР¶РµС‚ РІРµСЂРЅСѓС‚СЊ РјРЅРѕРіРѕ РґР°РЅРЅС‹С…."
            )

    def _validate_write(self, sql: str, result: ValidationResult) -> None:
        """Р’Р°Р»РёРґР°С†РёСЏ INSERT/UPDATE/DELETE."""
        normalized = sql.strip().upper()

        # 1. EXPLAIN
        try:
            plan = self._db.explain_query(sql)
            result.explain_plan = plan
        except Exception as e:
            result.add_error(f"РЎРёРЅС‚Р°РєСЃРёС‡РµСЃРєР°СЏ РѕС€РёР±РєР° (EXPLAIN): {e}")
            return

        # 2. UPDATE/DELETE Р±РµР· WHERE вЂ” С‚СЂРµР±СѓРµРј РїРѕРґС‚РІРµСЂР¶РґРµРЅРёРµ
        is_update_or_delete = normalized.startswith(("UPDATE", "DELETE"))
        if is_update_or_delete and not _has_top_level_where(sql):
            result.require_confirmation(
                "UPDATE/DELETE Р±РµР· WHERE Р·Р°С‚СЂРѕРЅРµС‚ Р’РЎР• СЃС‚СЂРѕРєРё С‚Р°Р±Р»РёС†С‹. Р’С‹ СѓРІРµСЂРµРЅС‹?"
            )

        # 3. РћС†РµРЅРєР° Р·Р°С‚СЂРѕРЅСѓС‚С‹С… СЃС‚СЂРѕРє
        if is_update_or_delete:
            self._estimate_write_impact(sql, result)

    def _estimate_write_impact(self, sql: str, result: ValidationResult) -> None:
        """Estimate write impact without executing arbitrary WHERE fragments.

        Security boundary: precise row counting by dynamic WHERE is disabled.
        """
        normalized = sql.strip()
        upper = normalized.upper()

        try:
            if upper.startswith("DELETE"):
                match = re.search(
                    r'DELETE\s+FROM\s+["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
                    normalized, re.IGNORECASE,
                )
                if match:
                    schema, table = match.group(1), match.group(2)
                    result.add_warning(
                        f"Оценка затронутых строк для {schema}.{table} отключена по соображениям безопасности."
                    )

            elif upper.startswith("UPDATE"):
                match = re.search(
                    r'UPDATE\s+["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
                    normalized, re.IGNORECASE,
                )
                if match:
                    schema, table = match.group(1), match.group(2)
                    result.add_warning(
                        f"Оценка затронутых строк для {schema}.{table} отключена по соображениям безопасности."
                    )
        except Exception as e:
            logger.warning("Не удалось оценить количество затронутых строк: %s", e)
    def _validate_ddl(self, sql: str, result: ValidationResult) -> None:
        """Р’Р°Р»РёРґР°С†РёСЏ DDL-Р·Р°РїСЂРѕСЃРѕРІ."""
        normalized = sql.strip().upper()

        # 1. DROP / TRUNCATE вЂ” С‚СЂРµР±СѓРµРј РїРѕРґС‚РІРµСЂР¶РґРµРЅРёРµ
        if normalized.startswith(("DROP", "TRUNCATE")):
            result.require_confirmation(
                "Р’С‹ СЃРѕР±РёСЂР°РµС‚РµСЃСЊ РІС‹РїРѕР»РЅРёС‚СЊ DROP/TRUNCATE. Р­С‚Рѕ РЅРµРѕР±СЂР°С‚РёРјР°СЏ РѕРїРµСЂР°С†РёСЏ. "
                "Р’РІРµРґРёС‚Рµ YES РґР»СЏ РїРѕРґС‚РІРµСЂР¶РґРµРЅРёСЏ."
            )

        # 2. CREATE TABLE вЂ” РїСЂРѕРІРµСЂРєР° СЃСѓС‰РµСЃС‚РІРѕРІР°РЅРёСЏ
        if normalized.startswith("CREATE TABLE"):
            match = re.search(
                r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
                sql, re.IGNORECASE,
            )
            if match:
                schema, table = match.group(1), match.group(2)
                try:
                    if self._db.table_exists(schema, table):
                        result.add_error(
                            f"РўР°Р±Р»РёС†Р° {schema}.{table} СѓР¶Рµ СЃСѓС‰РµСЃС‚РІСѓРµС‚. "
                            "РСЃРїРѕР»СЊР·СѓР№С‚Рµ IF NOT EXISTS РёР»Рё DROP РїРµСЂРµРґ СЃРѕР·РґР°РЅРёРµРј."
                        )
                except Exception as e:
                    logger.warning("РќРµ СѓРґР°Р»РѕСЃСЊ РїСЂРѕРІРµСЂРёС‚СЊ СЃСѓС‰РµСЃС‚РІРѕРІР°РЅРёРµ С‚Р°Р±Р»РёС†С‹: %s", e)

        # 3. ALTER вЂ” РїРѕРєР°Р·Р°С‚СЊ С‚РµРєСѓС‰СѓСЋ СЃС‚СЂСѓРєС‚СѓСЂСѓ
        if normalized.startswith("ALTER"):
            match = re.search(
                r'ALTER\s+TABLE\s+["\']?(\w+)["\']?\s*\.\s*["\']?(\w+)["\']?',
                sql, re.IGNORECASE,
            )
            if match:
                schema, table = match.group(1), match.group(2)
                try:
                    ddl = self._db.get_table_ddl(schema, table)
                    result.add_warning(f"РўРµРєСѓС‰Р°СЏ СЃС‚СЂСѓРєС‚СѓСЂР° С‚Р°Р±Р»РёС†С‹:\n{ddl}")
                except Exception as e:
                    logger.warning("РќРµ СѓРґР°Р»РѕСЃСЊ РїРѕР»СѓС‡РёС‚СЊ DDL С‚Р°Р±Р»РёС†С‹: %s", e)

