"""Загрузка и индексирование CSV-файлов со схемой БД."""

import logging
from pathlib import Path

import pandas as pd

from core.synonym_map import expand_with_synonyms

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data_for_agent"

# TF-IDF опционально (scikit-learn может быть не установлен)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    _TFIDF_AVAILABLE = True
except ImportError:
    _TFIDF_AVAILABLE = False
    logger.info("scikit-learn не установлен — TF-IDF поиск недоступен, используется keyword поиск")


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
        # TF-IDF индекс (строится лениво при первом поиске)
        self._tfidf_vectorizer: "TfidfVectorizer | None" = None
        self._tfidf_matrix = None
        self._tfidf_index: list[tuple[str, str]] = []  # [(schema, table), ...]

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

    def _build_tfidf_index(self) -> None:
        """Построить TF-IDF индекс по таблицам (вызывается лениво)."""
        if not _TFIDF_AVAILABLE:
            return
        df = self.tables_df
        if df.empty:
            return

        # Объединяем schema, table_name и description в один текст для индексации
        docs = []
        index = []
        for _, row in df.iterrows():
            schema = str(row.get("schema_name", ""))
            table = str(row.get("table_name", ""))
            desc = str(row.get("description", ""))
            # Разбиваем snake_case на слова для лучшего матчинга
            table_words = table.replace("_", " ")
            schema_words = schema.replace("_", " ")
            text = f"{schema_words} {table_words} {table_words} {desc}"
            docs.append(text.lower())
            index.append((schema, table))

        if not docs:
            return

        try:
            vectorizer = TfidfVectorizer(
                analyzer="word",
                token_pattern=r"[a-zA-Zа-яА-ЯёЁ][a-zA-Zа-яА-ЯёЁ0-9]*",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            )
            matrix = vectorizer.fit_transform(docs)
            self._tfidf_vectorizer = vectorizer
            self._tfidf_matrix = matrix
            self._tfidf_index = index
            logger.info("TF-IDF индекс построен: %d таблиц", len(index))
        except Exception as e:
            logger.warning("Ошибка построения TF-IDF индекса: %s", e)

    def _tfidf_search(self, query: str, top_n: int = 20) -> list[tuple[str, str, float]]:
        """Поиск по TF-IDF индексу.

        Returns:
            Список (schema, table, score) отсортированный по убыванию score.
        """
        if not _TFIDF_AVAILABLE:
            return []

        # Ленивое построение индекса
        if self._tfidf_vectorizer is None:
            self._build_tfidf_index()

        if self._tfidf_vectorizer is None or self._tfidf_matrix is None:
            return []

        # Расширяем запрос синонимами для лучшего покрытия
        synonyms = expand_with_synonyms(query)
        expanded = f"{query} {' '.join(synonyms)}".lower()
        # Разбиваем snake_case
        expanded = expanded.replace("_", " ")

        try:
            query_vec = self._tfidf_vectorizer.transform([expanded])
            scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
            top_indices = np.argsort(scores)[::-1][:top_n]
            results = []
            for idx in top_indices:
                score = float(scores[idx])
                if score > 0.01:  # Отсекаем нерелевантные
                    schema, table = self._tfidf_index[idx]
                    results.append((schema, table, score))
            return results
        except Exception as e:
            logger.warning("Ошибка TF-IDF поиска: %s", e)
            return []

    def search_tables(self, query: str, top_n: int = 30) -> pd.DataFrame:
        """Поиск таблиц по имени или описанию с TF-IDF + keyword fallback.

        Args:
            query: Строка поиска (русский или английский).
            top_n: Максимальное количество результатов.

        Returns:
            DataFrame с найденными таблицами, отсортированный по релевантности.
        """
        df = self.tables_df
        if df.empty:
            return df

        q = query.lower()
        synonyms = expand_with_synonyms(query)

        # 1. Keyword поиск
        mask = (
            df["table_name"].str.lower().str.contains(q, na=False)
            | df["schema_name"].str.lower().str.contains(q, na=False)
            | df["description"].str.lower().str.contains(q, na=False)
        )
        for syn in synonyms:
            mask = mask | (
                df["table_name"].str.lower().str.contains(syn, na=False)
                | df["schema_name"].str.lower().str.contains(syn, na=False)
                | df["description"].str.lower().str.contains(syn, na=False)
            )
        keyword_result = df[mask].copy()
        keyword_result["_score"] = 2.0  # Keyword match — высокий приоритет

        # 2. TF-IDF поиск
        tfidf_hits = self._tfidf_search(query, top_n=top_n)
        if tfidf_hits:
            tfidf_rows = []
            for schema, table, score in tfidf_hits:
                row_mask = (
                    (df["schema_name"] == schema) & (df["table_name"] == table)
                )
                match = df[row_mask]
                if not match.empty:
                    r = match.iloc[0].to_dict()
                    r["_score"] = score
                    tfidf_rows.append(r)
            tfidf_df = pd.DataFrame(tfidf_rows) if tfidf_rows else pd.DataFrame()
        else:
            tfidf_df = pd.DataFrame()

        # 3. Объединяем, дедуплицируем, сортируем по score
        if not tfidf_df.empty and "_score" in tfidf_df.columns:
            combined = pd.concat([keyword_result, tfidf_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["schema_name", "table_name"], keep="first")
            combined = combined.sort_values("_score", ascending=False).head(top_n)
            result = combined.drop(columns=["_score"], errors="ignore")
        else:
            result = keyword_result.head(top_n)

        logger.info(
            "search_tables('%s'): найдено %d (keyword: %d, tfidf: %d)",
            query, len(result), len(keyword_result), len(tfidf_hits),
        )
        return result.reset_index(drop=True)

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
        """Поиск таблиц и колонок по текстовому описанию с учётом синонимов.

        Args:
            text: Текст для поиска в описаниях (русский или английский).

        Returns:
            DataFrame с результатами (schema, table, column, description).
        """
        q = text.lower()
        synonyms = expand_with_synonyms(text)

        # Поиск в описаниях таблиц
        tdf = self.tables_df
        t_mask = tdf["description"].str.lower().str.contains(q, na=False)
        for syn in synonyms:
            t_mask = t_mask | tdf["description"].str.lower().str.contains(syn, na=False)
        tables_found = tdf[t_mask][["schema_name", "table_name", "description"]].copy()
        tables_found["column_name"] = ""
        tables_found["source"] = "table"

        # Поиск в описаниях атрибутов
        adf = self.attrs_df
        a_mask = adf["description"].str.lower().str.contains(q, na=False)
        for syn in synonyms:
            a_mask = a_mask | adf["description"].str.lower().str.contains(syn, na=False)
        attrs_found = adf[a_mask][
            ["schema_name", "table_name", "column_name", "description"]
        ].copy()
        attrs_found["source"] = "column"

        result = pd.concat([tables_found, attrs_found], ignore_index=True)
        logger.info("search_by_description('%s'): найдено %d (синонимов: %d)", text, len(result), len(synonyms))
        return result

    def check_key_uniqueness(
        self, schema: str, table: str, columns: list[str]
    ) -> dict:
        """Оценить уникальность комбинации колонок по данным CSV-справочника.

        Логика:
        - Если все колонки являются первичными ключами → уникален.
        - Если хотя бы одна колонка имеет unique_perc == 100.0 → уникален
          (одна полностью уникальная колонка гарантирует уникальность комбинации).
        - Иначе — берём минимальный unique_perc среди колонок как оценку.

        Args:
            schema: Имя схемы.
            table: Имя таблицы.
            columns: Список колонок для проверки.

        Returns:
            Словарь с is_unique, all_pk, min_unique_perc и деталями по колонкам.
        """
        cols_df = self.get_table_columns(schema, table)
        if cols_df.empty:
            return {
                "is_unique": None,
                "all_pk": False,
                "min_unique_perc": None,
                "error": f"Таблица {schema}.{table} не найдена в справочнике.",
            }

        details = {}
        all_pk = True
        min_unique_perc = 100.0
        any_fully_unique = False

        for col in columns:
            row = cols_df[cols_df["column_name"] == col]
            if row.empty:
                details[col] = {"found": False}
                all_pk = False
                continue
            r = row.iloc[0]
            is_pk = bool(r.get("is_primary_key", False))
            u_perc = float(r.get("unique_perc", 0.0)) if pd.notna(r.get("unique_perc")) else 0.0
            details[col] = {
                "found": True,
                "is_primary_key": is_pk,
                "unique_perc": u_perc,
            }
            if not is_pk:
                all_pk = False
            if u_perc == 100.0:
                any_fully_unique = True
            min_unique_perc = min(min_unique_perc, u_perc)

        # Количество PK-колонок в таблице — для определения составного PK
        pk_count = (
            int(cols_df["is_primary_key"].astype(bool).sum())
            if "is_primary_key" in cols_df.columns
            else 0
        )

        # Для single column: одна полностью уникальная колонка достаточна.
        # НО: если колонка — часть составного PK (pk_count > 1) и имеет
        # низкий unique_perc, она НЕ уникальна сама по себе.
        # Для composite key: any_fully_unique недостаточно — другая колонка
        # может создавать дубли внутри уникальных значений первой.
        if len(columns) == 1:
            col_detail = list(details.values())[0]
            is_composite_pk_member = (
                col_detail.get("found")
                and col_detail.get("is_primary_key")
                and pk_count > 1
            )
            if is_composite_pk_member:
                # Член составного PK НИКОГДА не уникален сам по себе.
                # Безопасен только если unique_perc == 100%.
                is_unique = any_fully_unique
            else:
                is_unique = all_pk or any_fully_unique
        else:
            is_unique = all_pk or min_unique_perc >= 95.0
        duplicate_pct = round(100.0 - min_unique_perc, 2)
        status = "safe" if is_unique else "risky"

        return {
            "is_unique": is_unique,
            "all_pk": all_pk,
            "min_unique_perc": min_unique_perc,
            "duplicate_pct": duplicate_pct,
            "columns": details,
            "status": status,
        }

    def generate_ddl(self, schema: str, table: str) -> str:
        """Сгенерировать DDL таблицы из CSV-справочника атрибутов.

        Args:
            schema: Имя схемы.
            table: Имя таблицы.

        Returns:
            Текстовое представление DDL (CREATE TABLE).
        """
        cols = self.get_table_columns(schema, table)
        if cols.empty:
            return f"Таблица {schema}.{table} не найдена в справочнике атрибутов."

        lines = [f'CREATE TABLE {schema}.{table} (']
        for _, row in cols.iterrows():
            not_null = " NOT NULL" if row.get("is_not_null") else ""
            pk = " -- PK" if row.get("is_primary_key") else ""
            desc = row.get("description", "")
            comment = f" -- {desc}" if pd.notna(desc) and str(desc).strip() else ""
            # PK-пометка идёт первой в комментарии
            if pk and comment:
                comment = f" -- PK | {str(desc).strip()}"
            elif pk:
                comment = pk
            lines.append(f'    {row["column_name"]} {row["dType"]}{not_null}{comment},')

        lines[-1] = lines[-1].rstrip(",")
        lines.append(");")
        return "\n".join(lines)

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
