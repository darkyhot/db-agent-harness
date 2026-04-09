"""Детерминированное форматирование и нормализация SQL.

Ноль LLM. Чистая детерминированная трансформация строки через sqlparse.
Применяется ПОСЛЕ любой генерации SQL — как шаблонной (SqlBuilder),
так и LLM (sql_writer) — для гарантии единообразного вывода.

Гарантии:
- Одинаковый SQL → одинаковый форматированный вывод (детерминированность)
- UPPER CASE для ключевых слов SQL
- Отступы 4 пробела
- Кириллические алиасы заменяются транслитом ДО передачи в static_checker
- Безопасно для любого валидного PostgreSQL/Greenplum SQL
"""

import re

import sqlparse


# ---------------------------------------------------------------------------
# Транслитерация кириллических алиасов
# ---------------------------------------------------------------------------

_TRANSLIT_TABLE = str.maketrans({
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'E',
    'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
    'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
    'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Shch',
    'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
})

# Паттерн: AS <кириллический алиас> (кавычки, одинарные или без)
_CYRILLIC_ALIAS_RE = re.compile(
    r'\bAS\s+(?:"([^"]*[а-яА-ЯёЁ][^"]*)"'
    r"|'([^']*[а-яА-ЯёЁ][^']*)'"
    r'|([а-яА-ЯёЁ][^\s,\)\n]+))',
    re.IGNORECASE,
)


def _translit(text: str) -> str:
    """Транслитерировать кириллицу в ASCII (нижний регистр)."""
    return text.lower().translate(_TRANSLIT_TABLE)


def fix_cyrillic_aliases(sql: str) -> str:
    """Заменить кириллические AS-алиасы на транслитерированные английские.

    Применяется ДО sqlparse.format() и static_checker.
    Примеры:
        AS "Сумма оттока"  → AS summa_ottoka
        AS итого           → AS itogo
        AS 'Регион'        → AS region
    """
    def _replace(m: re.Match) -> str:
        raw = m.group(1) or m.group(2) or m.group(3) or ''
        # Транслит + заменяем пробелы на _
        transliterated = _translit(raw).replace(' ', '_')
        # Убираем повторные _ и ведущие/хвостовые _
        transliterated = re.sub(r'_+', '_', transliterated).strip('_')
        if not transliterated:
            transliterated = 'alias'
        return f'AS {transliterated}'

    return _CYRILLIC_ALIAS_RE.sub(_replace, sql)


# ---------------------------------------------------------------------------
# Основное форматирование
# ---------------------------------------------------------------------------

def format_sql(sql: str) -> str:
    """Форматировать SQL детерминированно.

    Применяет последовательно:
    1. Нормализацию кириллических алиасов (→ транслит)
    2. sqlparse.format() с keyword_case='upper', reindent=True, indent_width=4
    3. Нормализацию переносов строк (убираем тройные пустые строки)
    4. Удаление trailing whitespace в каждой строке

    Args:
        sql: Любой SQL-запрос (может содержать кириллицу, смешанный регистр,
             лишние пробелы и т.д.)

    Returns:
        Отформатированный SQL-запрос. При пустом вводе возвращает исходную строку.
    """
    if not sql or not sql.strip():
        return sql

    # Шаг 1: нормализация кириллических алиасов
    sql = fix_cyrillic_aliases(sql)

    # Шаг 2: sqlparse форматирование
    # identifier_case='lower' переводит имена таблиц/колонок в нижний регистр,
    # что важно для Greenplum (case-insensitive, но соглашение — lowercase).
    formatted = sqlparse.format(
        sql.strip(),
        reindent=True,
        keyword_case='upper',
        identifier_case='lower',
        strip_comments=False,
        indent_width=4,
    )

    # Шаг 3: нормализация переносов
    formatted = re.sub(r'\n{3,}', '\n\n', formatted)

    # Шаг 4: trailing whitespace
    lines = [line.rstrip() for line in formatted.split('\n')]
    return '\n'.join(lines).strip()


def format_sql_safe(sql: str) -> str:
    """Форматировать SQL с перехватом исключений.

    Если sqlparse не может обработать SQL (например, неверный синтаксис),
    возвращает исходную строку без изменений + нормализацию кириллицы.

    Используется в production-коде где важна надёжность.
    """
    if not sql or not sql.strip():
        return sql
    try:
        return format_sql(sql)
    except Exception:
        # Минимальная нормализация: хотя бы кириллические алиасы
        return fix_cyrillic_aliases(sql).strip()
