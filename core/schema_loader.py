"""Загрузка и индексирование CSV-файлов со схемой БД."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data_for_agent"


class SchemaLoader:
    """Загрузка tables_list.csv и attr_list.csv с кешированием в памяти."""

    def __init__(self, data_dir: Path | None = None) -> None:
        """Инициализация загрузчика.

        Args:
            data_dir: Директория с CSV-файлами. По умолчанию — data_for_agent/.
        """
        self._data_dir = data_dir or DATA_DIR
        self._tables_df: pd.DataFrame | None = None
        self._attrs_df: pd.DataFrame | None = None

    def _load_tables(self) -> pd.DataFrame:
        """Загрузить и закешировать tables_list.csv."""
        if self._tables_df is not None:
            return self._tables_df

        path = self._data_dir / "tables_list.csv"
        if not path.exists():
            logger.warning("Файл не найден: %s", path)
            self._tables_df = pd.DataFrame(
                columns=["schema_name", "table_name", "description"]
            )
            return self._tables_df

        self._tables_df = pd.read_csv(path, encoding="utf-8")
        logger.info("Загружено таблиц: %d", len(self._tables_df))
        return self._tables_df

    def _load_attrs(self) -> pd.DataFrame:
        """Загрузить и закешировать attr_list.csv."""
        if self._attrs_df is not None:
            return self._attrs_df

        path = self._data_dir / "attr_list.csv"
        if not path.exists():
            logger.warning("Файл не найден: %s", path)
            self._attrs_df = pd.DataFrame(
                columns=[
                    "schema_name", "table_name", "column_name", "dType",
                    "is_not_null", "description", "is_primary_key",
                    "not_null_perc", "unique_perc",
                ]
            )
            return self._attrs_df

        self._attrs_df = pd.read_csv(path, encoding="utf-8")
        logger.info("Загружено атрибутов: %d", len(self._attrs_df))
        return self._attrs_df

    @property
    def tables_df(self) -> pd.DataFrame:
        """Кешированный DataFrame таблиц."""
        return self._load_tables()

    @property
    def attrs_df(self) -> pd.DataFrame:
        """Кешированный DataFrame атрибутов."""
        return self._load_attrs()

    @property
    def tables_count(self) -> int:
        """Количество загруженных таблиц."""
        return len(self.tables_df)

    @property
    def attrs_count(self) -> int:
        """Количество загруженных атрибутов."""
        return len(self.attrs_df)

    def search_tables(self, query: str) -> pd.DataFrame:
        """Поиск таблиц по имени или описанию (case-insensitive).

        Args:
            query: Строка поиска.

        Returns:
            DataFrame с найденными таблицами.
        """
        df = self.tables_df
        q = query.lower()
        mask = (
            df["table_name"].str.lower().str.contains(q, na=False)
            | df["schema_name"].str.lower().str.contains(q, na=False)
            | df["description"].str.lower().str.contains(q, na=False)
        )
        result = df[mask].copy()
        logger.info("search_tables('%s'): найдено %d", query, len(result))
        return result

    def get_table_columns(self, schema: str, table: str) -> pd.DataFrame:
        """Получить атрибуты конкретной таблицы.

        Args:
            schema: Имя схемы.
            table: Имя таблицы.

        Returns:
            DataFrame с атрибутами таблицы.
        """
        df = self.attrs_df
        mask = (df["schema_name"] == schema) & (df["table_name"] == table)
        return df[mask].copy()

    def find_tables_with_column(self, column_name: str) -> pd.DataFrame:
        """Найти таблицы, содержащие колонку с указанным именем.

        Args:
            column_name: Имя колонки (case-insensitive).

        Returns:
            DataFrame с уникальными schema_name, table_name.
        """
        df = self.attrs_df
        mask = df["column_name"].str.lower().str.contains(column_name.lower(), na=False)
        result = df[mask][["schema_name", "table_name", "column_name"]].drop_duplicates()
        logger.info("find_tables_with_column('%s'): найдено %d", column_name, len(result))
        return result

    def get_primary_keys(self, schema: str, table: str) -> list[str]:
        """Получить список первичных ключей таблицы.

        Args:
            schema: Имя схемы.
            table: Имя таблицы.

        Returns:
            Список имён колонок, являющихся первичными ключами.
        """
        df = self.attrs_df
        mask = (
            (df["schema_name"] == schema)
            & (df["table_name"] == table)
            & (df["is_primary_key"] == True)  # noqa: E712
        )
        keys = df[mask]["column_name"].tolist()
        logger.info("PK для %s.%s: %s", schema, table, keys)
        return keys

    def search_by_description(self, text: str) -> pd.DataFrame:
        """Поиск таблиц и колонок по текстовому описанию.

        Args:
            text: Текст для поиска в описаниях.

        Returns:
            DataFrame с результатами (schema, table, column, description).
        """
        q = text.lower()

        # Поиск в описаниях таблиц
        tdf = self.tables_df
        t_mask = tdf["description"].str.lower().str.contains(q, na=False)
        tables_found = tdf[t_mask][["schema_name", "table_name", "description"]].copy()
        tables_found["column_name"] = ""
        tables_found["source"] = "table"

        # Поиск в описаниях атрибутов
        adf = self.attrs_df
        a_mask = adf["description"].str.lower().str.contains(q, na=False)
        attrs_found = adf[a_mask][
            ["schema_name", "table_name", "column_name", "description"]
        ].copy()
        attrs_found["source"] = "column"

        result = pd.concat([tables_found, attrs_found], ignore_index=True)
        logger.info("search_by_description('%s'): найдено %d", text, len(result))
        return result

    def get_table_info(self, schema: str, table: str) -> str:
        """Получить текстовое описание таблицы с колонками для контекста LLM.

        Args:
            schema: Имя схемы.
            table: Имя таблицы.

        Returns:
            Форматированная строка с описанием таблицы.
        """
        # Описание таблицы
        tdf = self.tables_df
        t_mask = (tdf["schema_name"] == schema) & (tdf["table_name"] == table)
        t_rows = tdf[t_mask]
        desc = t_rows["description"].iloc[0] if not t_rows.empty else "нет описания"

        # Колонки
        cols = self.get_table_columns(schema, table)
        lines = [f"Таблица: {schema}.{table}", f"Описание: {desc}", "Колонки:"]
        for _, row in cols.iterrows():
            pk = " [PK]" if row.get("is_primary_key") else ""
            nn = " NOT NULL" if row.get("is_not_null") else ""
            col_desc = f" -- {row['description']}" if pd.notna(row.get("description")) else ""
            lines.append(f"  {row['column_name']} {row['dType']}{pk}{nn}{col_desc}")

        return "\n".join(lines)
