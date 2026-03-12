"""Инструменты для поиска по схеме БД (таблицы, атрибуты, связи).

Используют фабрику create_schema_tools() для DI вместо глобальных переменных.
"""

import logging

from langchain_core.tools import tool

from core.schema_loader import SchemaLoader

logger = logging.getLogger(__name__)


def create_schema_tools(schema_loader: SchemaLoader) -> list:
    """Создать инструменты схемы через замыкания (DI без глобальных переменных).

    Args:
        schema_loader: Настроенный экземпляр SchemaLoader.

    Returns:
        Список LangChain tools.
    """

    @tool
    def search_tables(query: str) -> str:
        """Поиск таблиц по имени, схеме или описанию.

        Args:
            query: Строка поиска (например: 'зарплата', 'employee', 'hr').

        Returns:
            Список найденных таблиц с описаниями.
        """
        try:
            df = schema_loader.search_tables(query)
            if df.empty:
                return f"Таблицы по запросу '{query}' не найдены."
            lines = [f"Найдено таблиц: {len(df)}", ""]
            for _, row in df.iterrows():
                lines.append(f"  {row['schema_name']}.{row['table_name']} — {row['description']}")
            return "\n".join(lines)
        except Exception as e:
            logger.error("search_tables error: %s", e)
            return f"Ошибка: {e}"

    @tool
    def get_table_columns(schema: str, table: str) -> str:
        """Получить список атрибутов (колонок) конкретной таблицы.

        Args:
            schema: Имя схемы.
            table: Имя таблицы.

        Returns:
            Список колонок с типами и описаниями.
        """
        try:
            df = schema_loader.get_table_columns(schema, table)
            if df.empty:
                return f"Атрибуты для {schema}.{table} не найдены."
            lines = [f"Атрибуты таблицы {schema}.{table} ({len(df)} колонок):", ""]
            for _, row in df.iterrows():
                pk = " [PK]" if row.get("is_primary_key") else ""
                nn = " NOT NULL" if row.get("is_not_null") else ""
                desc = f" — {row['description']}" if str(row.get("description", "")) not in ("", "nan") else ""
                null_pct = f", null: {row['not_null_perc']}%" if "not_null_perc" in row else ""
                uniq_pct = f", unique: {row['unique_perc']}%" if "unique_perc" in row else ""
                lines.append(f"  {row['column_name']} {row['dType']}{pk}{nn}{desc}{null_pct}{uniq_pct}")
            return "\n".join(lines)
        except Exception as e:
            logger.error("get_table_columns error: %s", e)
            return f"Ошибка: {e}"

    @tool
    def find_tables_with_column(column_name: str) -> str:
        """Найти таблицы, содержащие колонку с указанным именем.

        Args:
            column_name: Имя колонки (или часть имени) для поиска.

        Returns:
            Список таблиц с найденной колонкой.
        """
        try:
            df = schema_loader.find_tables_with_column(column_name)
            if df.empty:
                return f"Колонка '{column_name}' не найдена ни в одной таблице."
            lines = [f"Колонка '{column_name}' найдена в {len(df)} местах:", ""]
            for _, row in df.iterrows():
                lines.append(f"  {row['schema_name']}.{row['table_name']}.{row['column_name']}")
            return "\n".join(lines)
        except Exception as e:
            logger.error("find_tables_with_column error: %s", e)
            return f"Ошибка: {e}"

    @tool
    def get_primary_keys(schema: str, table: str) -> str:
        """Получить первичные ключи таблицы.

        Args:
            schema: Имя схемы.
            table: Имя таблицы.

        Returns:
            Список колонок первичного ключа.
        """
        try:
            keys = schema_loader.get_primary_keys(schema, table)
            if not keys:
                return f"Первичные ключи для {schema}.{table} не найдены."
            return f"Первичные ключи {schema}.{table}: {', '.join(keys)}"
        except Exception as e:
            logger.error("get_primary_keys error: %s", e)
            return f"Ошибка: {e}"

    @tool
    def search_by_description(search_text: str) -> str:
        """Поиск таблиц и колонок по текстовому описанию.

        Args:
            search_text: Текст для поиска в описаниях (например: 'средняя зарплата', 'дата рождения').

        Returns:
            Найденные таблицы и колонки с описаниями.
        """
        try:
            df = schema_loader.search_by_description(search_text)
            if df.empty:
                return f"По описанию '{search_text}' ничего не найдено."

            tables = df[df["source"] == "table"]
            columns = df[df["source"] == "column"]

            lines = []
            if not tables.empty:
                lines.append(f"Таблицы ({len(tables)}):")
                for _, row in tables.iterrows():
                    lines.append(f"  {row['schema_name']}.{row['table_name']} — {row['description']}")

            if not columns.empty:
                lines.append(f"\nКолонки ({len(columns)}):")
                for _, row in columns.iterrows():
                    lines.append(
                        f"  {row['schema_name']}.{row['table_name']}.{row['column_name']} — {row['description']}"
                    )

            return "\n".join(lines)
        except Exception as e:
            logger.error("search_by_description error: %s", e)
            return f"Ошибка: {e}"

    return [
        search_tables,
        get_table_columns,
        find_tables_with_column,
        get_primary_keys,
        search_by_description,
    ]
