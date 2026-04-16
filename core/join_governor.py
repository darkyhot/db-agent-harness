"""Управление JOIN-планом: pruning, риск fanout и решение single vs multi-table."""

from __future__ import annotations

from typing import Any

from core.domain_rules import table_can_satisfy_frame


def decide_join_plan(
    *,
    selected_tables: list[tuple[str, str]],
    main_table: tuple[str, str] | None,
    locked_tables: list[tuple[str, str]] | None,
    join_requested: bool,
    semantic_frame: dict[str, Any] | None,
    requested_grain: str | None,
    dimension_slots: list[str] | None,
    slot_scores: dict[str, dict[str, float]] | None,
    schema_loader,
) -> dict[str, Any]:
    """Решить, нужен ли multi-table plan, и если нет — безопасно его сузить."""
    locked = list(locked_tables or [])
    original = list(dict.fromkeys(selected_tables))
    if len(original) <= 1 or not main_table:
        only = list(original or ([main_table] if main_table else []))
        return {
            "allow_join": len(only) > 1,
            "selected_tables": only,
            "removed_tables": [],
            "risk_level": "low",
            "reason": "single_table_plan",
            "expected_grain_after_join": requested_grain or "",
            "evidence": ["join_not_required"],
        }

    if join_requested or locked:
        return {
            "allow_join": True,
            "selected_tables": original,
            "removed_tables": [],
            "risk_level": "medium" if len(original) > 1 else "low",
            "reason": "join_requested_or_locked",
            "expected_grain_after_join": requested_grain or "",
            "evidence": [
                "join_requested" if join_requested else "locked_tables_present",
            ],
        }

    main_key = f"{main_table[0]}.{main_table[1]}"
    main_semantics = schema_loader.get_table_semantics(main_table[0], main_table[1])
    main_grain = str(main_semantics.get("grain") or "")
    supports_frame = table_can_satisfy_frame(schema_loader, main_table[0], main_table[1], semantic_frame)
    evidence = [
        f"main_table={main_key}",
        f"main_grain={main_grain or 'unknown'}",
        f"frame_supported={supports_frame}",
    ]

    required_slots = [slot for slot in (dimension_slots or []) if slot != "date"]
    main_slot_scores = (slot_scores or {}).get(main_key, {})
    slots_need_external = False
    for slot in required_slots:
        main_score = float(main_slot_scores.get(slot, 0.0) or 0.0)
        best_other = max(
            float((slot_scores or {}).get(f"{s}.{t}", {}).get(slot, 0.0) or 0.0)
            for s, t in original
            if (s, t) != main_table
        ) if len(original) > 1 else 0.0
        evidence.append(f"slot:{slot}:main={main_score:.1f},other={best_other:.1f}")
        if best_other >= max(120.0, main_score + 90.0):
            slots_need_external = True

    grain_penalty = 0.0
    if requested_grain and main_grain and requested_grain != main_grain:
        grain_penalty = 1.0
        evidence.append("main_grain_mismatch")

    if supports_frame and not slots_need_external and grain_penalty == 0.0:
        return {
            "allow_join": False,
            "selected_tables": [main_table],
            "removed_tables": [t for t in original if t != main_table],
            "risk_level": "low",
            "reason": "main_table_covers_request",
            "expected_grain_after_join": requested_grain or main_grain or "",
            "evidence": evidence,
        }

    if slots_need_external:
        keep_tables = [main_table]
        for slot in required_slots:
            best_table = None
            best_score = float(main_slot_scores.get(slot, 0.0) or 0.0)
            for candidate in original:
                candidate_key = f"{candidate[0]}.{candidate[1]}"
                candidate_score = float((slot_scores or {}).get(candidate_key, {}).get(slot, 0.0) or 0.0)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_table = candidate
            if best_table and best_table not in keep_tables:
                keep_tables.append(best_table)
        keep_tables = list(dict.fromkeys(keep_tables))
        return {
            "allow_join": len(keep_tables) > 1,
            "selected_tables": keep_tables,
            "removed_tables": [t for t in original if t not in keep_tables],
            "risk_level": "medium",
            "reason": "external_tables_required",
            "expected_grain_after_join": requested_grain or main_grain or "",
            "evidence": evidence,
        }

    return {
        "allow_join": True,
        "selected_tables": original,
        "removed_tables": [],
        "risk_level": "medium" if slots_need_external else "high",
        "reason": "external_tables_required" if slots_need_external else "grain_or_frame_risk",
        "expected_grain_after_join": requested_grain or main_grain or "",
        "evidence": evidence,
    }
