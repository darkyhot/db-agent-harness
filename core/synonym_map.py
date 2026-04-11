"""Маппинг русских бизнес-терминов на английские имена таблиц/колонок."""

import re
from typing import Iterable


# Русский термин → набор английских keywords для поиска
SYNONYM_MAP: dict[str, list[str]] = {
    # Клиенты
    "клиент": ["client", "customer", "cust"],
    "клиенты": ["client", "customer", "cust"],
    "заказчик": ["client", "customer"],
    # Продажи
    "продажа": ["sale", "sales", "order", "revenue"],
    "продажи": ["sale", "sales", "order", "revenue"],
    "выручка": ["revenue", "sales", "income"],
    "доход": ["revenue", "income", "profit"],
    # Оттоки
    "отток": ["churn", "attrition", "outflow"],
    "отток клиентов": ["churn", "attrition"],
    # Платежи
    "платеж": ["payment", "pay", "transaction"],
    "платежи": ["payment", "pay", "transaction"],
    "транзакция": ["transaction", "txn", "payment"],
    "транзакции": ["transaction", "txn", "payment"],
    # Товары
    "товар": ["product", "item", "good", "sku"],
    "товары": ["product", "item", "good", "sku"],
    "продукт": ["product", "item"],
    "продукты": ["product", "item"],
    "категория": ["category", "cat", "group"],
    # Сотрудники
    "сотрудник": ["employee", "emp", "staff", "worker"],
    "сотрудники": ["employee", "emp", "staff", "worker"],
    "менеджер": ["manager", "mgr"],
    "менеджеры": ["manager", "mgr"],
    # Финансы
    "счет": ["account", "acc", "invoice", "bill"],
    "баланс": ["balance", "bal"],
    "бюджет": ["budget"],
    "расход": ["expense", "cost", "expenditure"],
    "расходы": ["expense", "cost", "expenditure"],
    # Регионы
    "регион": ["region", "reg", "area", "territory"],
    "город": ["city", "town"],
    "адрес": ["address", "addr", "location"],
    "филиал": ["branch", "office", "division"],
    # Даты и время
    "дата": ["date", "dt", "day"],
    "период": ["period", "interval", "range"],
    "месяц": ["month", "mon"],
    "год": ["year", "yr"],
    "квартал": ["quarter", "qtr"],
    # Статусы
    "статус": ["status", "state", "flag"],
    "активный": ["active", "enabled"],
    # Склады и логистика
    "склад": ["warehouse", "wh", "stock", "store"],
    "остаток": ["stock", "balance", "remainder", "inventory"],
    "остатки": ["stock", "balance", "remainder", "inventory"],
    "поставка": ["supply", "delivery", "shipment"],
    "поставщик": ["supplier", "vendor"],
    # Маркетинг
    "кампания": ["campaign", "camp"],
    "акция": ["promotion", "promo", "discount"],
    "скидка": ["discount", "promo"],
    # Отчёты
    "отчет": ["report", "rep", "summary"],
    "витрина": ["mart", "dm", "datamart", "showcase"],
    "витрины": ["mart", "dm", "datamart", "showcase"],
    "таблица": ["table", "mart", "dm", "datamart", "showcase", "fact", "dim", "ref"],
    "таблицы": ["table", "mart", "dm", "datamart", "showcase", "fact", "dim", "ref"],
    "справочник": ["dict", "ref", "reference", "lookup", "directory"],
}


def expand_with_synonyms(query: str) -> list[str]:
    """Расширить поисковый запрос синонимами.

    Args:
        query: Поисковый запрос на русском или английском.

    Returns:
        Список дополнительных ключевых слов для поиска.
    """
    query_lower = query.lower()
    extra_keywords: list[str] = []

    for ru_term, en_keywords in SYNONYM_MAP.items():
        if ru_term in query_lower:
            extra_keywords.extend(en_keywords)

    # Также добавляем оригинальные слова запроса
    words = re.findall(r'[a-zA-Zа-яА-ЯёЁ_]+', query_lower)
    extra_keywords.extend(words)

    return list(set(extra_keywords))


def match_tables_by_synonyms(
    query: str,
    table_names: Iterable[str],
    table_descriptions: dict[str, str] | None = None,
) -> list[str]:
    """Найти таблицы, соответствующие запросу с учётом синонимов.

    Args:
        query: Поисковый запрос.
        table_names: Имена таблиц (schema.table).
        table_descriptions: Описания таблиц (опционально).

    Returns:
        Список подходящих имён таблиц, отсортированный по релевантности.
    """
    keywords = expand_with_synonyms(query)
    if not keywords:
        return []

    scored: list[tuple[str, int]] = []
    for full_name in table_names:
        score = 0
        name_lower = full_name.lower()
        for kw in keywords:
            if kw in name_lower:
                score += 2  # Match in table name is strong signal
        if table_descriptions:
            desc = table_descriptions.get(full_name, "").lower()
            for kw in keywords:
                if kw in desc:
                    score += 1  # Match in description is weaker
        if score > 0:
            scored.append((full_name, score))

    scored.sort(key=lambda x: -x[1])
    return [name for name, _ in scored]
