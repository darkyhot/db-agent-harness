"""Загрузка и индексирование CSV-файлов со схемой БД."""

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from core.column_semantics import build_column_semantics
from core.semantic_registry import (
    build_rule_registry,
    build_semantic_lexicon,
    find_best_subject,
)
from core.synonym_map import expand_with_synonyms
from core.table_semantics import build_table_semantics
from core.value_profiler import (
    build_db_profile,
    build_metadata_profile,
    discover_profile_candidates,
    fetch_table_profile_sample,
    merge_profiles,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data_for_agent"
_TABLES_CSV = "tables_list.csv"
_VALUE_PROFILES_JSON = "column_value_profiles.json"
_COLUMN_SEMANTICS_JSON = "column_semantics.json"
_TABLE_SEMANTICS_JSON = "table_semantics.json"
_SEMANTIC_LEXICON_JSON = "semantic_lexicon.json"
_RULE_REGISTRY_JSON = "rule_registry.json"

_TABLE_COLUMNS = ["schema_name", "table_name", "description", "grain"]
_ATTR_COLUMNS = [
    "schema_name", "table_name", "column_name", "dType",
    "is_not_null", "description", "is_primary_key",
    "not_null_perc", "unique_perc",
    # Расширения (опциональные, могут отсутствовать в исторических CSV).
    # foreign_key_target: "schema.table.column" — при непустом значении используется
    # как 0-й этап в core/join_analysis.rank_join_candidates.
    # sample_values: "v1|v2|v3" — типичные значения для value-resolver.
    # partition_key: bool — помечает колонку-партиционирование (для warnings в dry-run).
    # synonyms: "syn1,syn2,syn3" — синонимы имени (подсказки для семантического фрейма).
    "foreign_key_target", "sample_values", "partition_key", "synonyms",
]

_GRAIN_ENUM = (
    "task", "client", "employee", "organization", "event", "transaction",
    "snapshot", "dictionary", "document", "payment", "account", "product", "other",
)
_GRAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "task": ("task", "ticket", "issue", "задач", "воронка", "funnel"),
    "client": ("client", "customer", "cust", "клиент", "inn", "инн"),
    "employee": ("employee", "emp", "staff", "worker", "сотрудник"),
    "organization": ("org", "organization", "branch", "gosb", "tb", "госб", "тб"),
    "event": ("event", "history", "log", "отток", "событ", "истори"),
    "transaction": ("transaction", "txn", "order", "sale", "заказ", "продаж"),
    "snapshot": ("snapshot", "report", "report_dt", "period", "срез", "отчет"),
    "dictionary": ("dim", "dict", "lookup", "reference", "справочник"),
    "document": ("document", "doc", "ведомост", "файл"),
    "payment": ("payment", "payroll", "salary", "зп", "платеж"),
    "account": ("account", "acc", "balance", "счет"),
    "product": ("product", "sku", "товар", "продукт"),
}

# TF-IDF опционально (scikit-learn может быть не установлен)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    _TFIDF_AVAILABLE = True
except ImportError:
    _TFIDF_AVAILABLE = False
    logger.info("scikit-learn не установлен — TF-IDF поиск недоступен, используется keyword поиск")


def _literal_contains(series: pd.Series, needle: str) -> pd.Series:
    """Literal substring search for user input without regex interpretation."""
    return series.fillna("").astype(str).str.lower().str.contains(needle.lower(), regex=False)


def _should_run_semantic_search(query: str) -> bool:
    """Disable semantic fallback for regex-like or punctuation-heavy user input."""
    if not query.strip():
        return False
    if re.search(r"[\[\]\(\)\{\}\+\*\?\|\^\$\\]", query):
        return False
    tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9_]+", query)
    return any(len(token) >= 2 for token in tokens)


def _should_expand_synonyms(query: str) -> bool:
    """Synonym expansion is helpful for natural language, but harmful for literal fragments."""
    return _should_run_semantic_search(query)


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
        self._value_profiles: dict[str, dict[str, Any]] | None = None
        self._column_semantics: dict[str, dict[str, Any]] | None = None
        self._table_semantics: dict[str, dict[str, Any]] | None = None
        self._semantic_lexicon: dict[str, Any] | None = None
        self._rule_registry: dict[str, Any] | None = None

    @staticmethod
    def _load_json_artifact(path: Path) -> dict[str, Any] | None:
        """Загрузить JSON-артефакт как dict; вернуть None если файла нет/он битый."""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            logger.warning("Не удалось прочитать JSON-артефакт %s, пересоздаю", path)
            return None
        if not isinstance(data, dict):
            logger.warning("JSON-артефакт %s имеет неожиданный формат, пересоздаю", path)
            return None
        return data

    @staticmethod
    def _sanitize_rule_registry(registry: dict[str, Any] | None) -> dict[str, Any]:
        """Нормализовать загруженный rule_registry, даже если он был собран старой логикой."""
        if not isinstance(registry, dict):
            return {"rules": []}
        rules = registry.get("rules") or []
        sanitized_rules: list[dict[str, Any]] = []
        for raw in rules:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            column_key = str(item.get("column_key") or "")
            column_name = column_key.rsplit(".", 1)[-1]
            phrases = [
                str(phrase).strip()
                for phrase in (item.get("match_phrases") or [])
                if str(phrase).strip()
            ]
            item["match_phrases"] = [
                phrase
                for phrase in phrases
                if not (
                    (column_name.endswith("_type") or column_name.endswith("_category"))
                    and len(re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9_]+", phrase)) < 2
                )
            ]
            sanitized_rules.append(item)
        return {"rules": sanitized_rules}

    def _load_tables(self) -> pd.DataFrame:
        """Загрузить и закешировать tables_list.csv."""
        if self._tables_df is not None:
            return self._tables_df

        path = self._data_dir / _TABLES_CSV
        if not path.exists():
            logger.warning("Файл не найден: %s", path)
            self._tables_df = pd.DataFrame(columns=_TABLE_COLUMNS)
            return self._tables_df

        self._tables_df = pd.read_csv(path, encoding="utf-8")
        for col in _TABLE_COLUMNS:
            if col not in self._tables_df.columns:
                self._tables_df[col] = ""
        self._tables_df["grain"] = self._tables_df["grain"].fillna("").astype(str)
        logger.info("Загружено таблиц: %d", len(self._tables_df))
        return self._tables_df

    def _load_attrs(self) -> pd.DataFrame:
        """Загрузить и закешировать attr_list.csv."""
        if self._attrs_df is not None:
            return self._attrs_df

        path = self._data_dir / "attr_list.csv"
        if not path.exists():
            logger.warning("Файл не найден: %s", path)
            self._attrs_df = pd.DataFrame(columns=_ATTR_COLUMNS)
            return self._attrs_df

        self._attrs_df = pd.read_csv(path, encoding="utf-8")
        for col in _ATTR_COLUMNS:
            if col not in self._attrs_df.columns:
                # Для bool-расширений дефолт False; для остальных — пустая строка.
                default = False if col == "partition_key" else ""
                self._attrs_df[col] = default
        # Нормализация опциональных полей — пустые значения NaN → ""/False.
        for col in ("foreign_key_target", "sample_values", "synonyms"):
            if col in self._attrs_df.columns:
                self._attrs_df[col] = self._attrs_df[col].fillna("").astype(str)
        if "partition_key" in self._attrs_df.columns:
            self._attrs_df["partition_key"] = (
                self._attrs_df["partition_key"].fillna(False).astype(bool)
            )
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
            grain = str(row.get("grain", ""))
            # Разбиваем snake_case на слова для лучшего матчинга
            table_words = table.replace("_", " ")
            schema_words = schema.replace("_", " ")
            text = f"{schema_words} {table_words} {table_words} {desc} {grain}"
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
        synonyms = expand_with_synonyms(query) if _should_expand_synonyms(query) else []
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
        synonyms = expand_with_synonyms(query) if _should_expand_synonyms(query) else []

        # 1. Keyword поиск
        mask = (
            _literal_contains(df["table_name"], q)
            | _literal_contains(df["schema_name"], q)
            | _literal_contains(df["description"], q)
        )
        for syn in synonyms:
            mask = mask | (
                _literal_contains(df["table_name"], syn)
                | _literal_contains(df["schema_name"], syn)
                | _literal_contains(df["description"], syn)
            )
        keyword_result = df[mask].copy()
        keyword_result["_score"] = 2.0  # Keyword match — высокий приоритет

        # 2. TF-IDF поиск
        tfidf_hits = self._tfidf_search(query, top_n=top_n) if _should_run_semantic_search(query) else []
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

    def _persist_tables_df(self, df: pd.DataFrame) -> None:
        """Сохранить tables_list.csv и обновить кеш/индексы."""
        for col in _TABLE_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        path = self._data_dir / _TABLES_CSV
        df.to_csv(path, index=False, encoding="utf-8")
        self._tables_df = df.copy()
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._tfidf_index = []

    @staticmethod
    def _profile_key(schema: str, table: str, column: str) -> str:
        return f"{schema}.{table}.{column}".lower()

    def _value_profiles_path(self) -> Path:
        return self._data_dir / _VALUE_PROFILES_JSON

    def _column_semantics_path(self) -> Path:
        return self._data_dir / _COLUMN_SEMANTICS_JSON

    def _table_semantics_path(self) -> Path:
        return self._data_dir / _TABLE_SEMANTICS_JSON

    def _semantic_lexicon_path(self) -> Path:
        return self._data_dir / _SEMANTIC_LEXICON_JSON

    def _rule_registry_path(self) -> Path:
        return self._data_dir / _RULE_REGISTRY_JSON

    @staticmethod
    def _table_key(schema: str, table: str) -> str:
        return f"{schema}.{table}".lower()

    def ensure_column_semantics(self) -> None:
        """Построить и закешировать semantics колонок."""
        if self._column_semantics is not None:
            return

        path = self._column_semantics_path()
        persisted = self._load_json_artifact(path)
        if persisted is not None:
            self._column_semantics = persisted
            return

        generated = build_column_semantics(self.attrs_df)
        self._column_semantics = dict(generated)
        path.write_text(json.dumps(self._column_semantics, ensure_ascii=False, indent=2), encoding="utf-8")

    def ensure_table_semantics(self) -> None:
        """Построить и закешировать semantics таблиц."""
        if self._table_semantics is not None:
            return

        path = self._table_semantics_path()
        persisted = self._load_json_artifact(path)
        if persisted is not None:
            self._table_semantics = persisted
            return

        self.ensure_column_semantics()
        generated = build_table_semantics(self.tables_df, self.attrs_df, self._column_semantics or {})
        self._table_semantics = dict(generated)
        path.write_text(json.dumps(self._table_semantics, ensure_ascii=False, indent=2), encoding="utf-8")

    def ensure_value_profiles(self, db_manager: Any | None = None) -> None:
        """Построить и закешировать value profiles для фильтровых колонок."""
        if self._value_profiles is not None and db_manager is None:
            return

        path = self._value_profiles_path()
        persisted = self._load_json_artifact(path)
        if persisted is not None:
            self._value_profiles = persisted
            return

        self.ensure_column_semantics()
        profiles: dict[str, dict[str, Any]] = {}
        candidates = discover_profile_candidates(self.attrs_df, self._column_semantics or {})
        by_table: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in candidates:
            schema = str(row.get("schema_name", "") or "")
            table = str(row.get("table_name", "") or "")
            if schema and table:
                by_table.setdefault((schema, table), []).append(row)

        for (schema, table), table_rows in by_table.items():
            metadata_profiles: dict[str, dict[str, Any]] = {}
            sample_columns: list[str] = []
            for row in table_rows:
                column = str(row.get("column_name", "") or "")
                if not column:
                    continue
                key = self._profile_key(schema, table, column)
                semantics = (self._column_semantics or {}).get(key, {})
                metadata_profile = build_metadata_profile(row, semantics)
                metadata_profiles[column] = metadata_profile
                sample_columns.append(column)
            table_sample = (
                fetch_table_profile_sample(
                    db_manager,
                    schema=schema,
                    table=table,
                    columns=sample_columns,
                )
                if db_manager is not None and sample_columns
                else None
            )
            for row in table_rows:
                column = str(row.get("column_name", "") or "")
                key = self._profile_key(schema, table, column)
                semantics = (self._column_semantics or {}).get(key, {})
                metadata_profile = metadata_profiles.get(column) or build_metadata_profile(row, semantics)
                db_profile = (
                    build_db_profile(
                        table_sample,
                        column=column,
                        metadata_profile=metadata_profile,
                    )
                    if table_sample is not None
                    else {}
                )
                profiles[key] = merge_profiles(metadata_profile, db_profile)

        self._value_profiles = dict(profiles)
        path.write_text(json.dumps(self._value_profiles, ensure_ascii=False, indent=2), encoding="utf-8")

    def ensure_semantic_registry(self) -> None:
        """Построить и закешировать semantic lexicon и rule registry."""
        if self._semantic_lexicon is not None and self._rule_registry is not None:
            return

        lex_path = self._semantic_lexicon_path()
        rule_path = self._rule_registry_path()
        persisted_lex = self._load_json_artifact(lex_path)
        persisted_rules = self._load_json_artifact(rule_path)
        if persisted_lex is not None:
            self._semantic_lexicon = persisted_lex
        if persisted_rules is not None:
            self._rule_registry = self._sanitize_rule_registry(persisted_rules)
        if persisted_lex is not None and persisted_rules is not None:
            return

        self.ensure_column_semantics()
        self.ensure_table_semantics()
        self.ensure_value_profiles()

        if persisted_lex is None:
            self._semantic_lexicon = build_semantic_lexicon(
                self.tables_df,
                self.attrs_df,
                table_semantics=self._table_semantics or {},
                column_semantics=self._column_semantics or {},
                value_profiles=self._value_profiles or {},
            )
            lex_path.write_text(
                json.dumps(self._semantic_lexicon, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if persisted_rules is None:
            self._rule_registry = build_rule_registry(
                self.attrs_df,
                column_semantics=self._column_semantics or {},
                value_profiles=self._value_profiles or {},
            )
            self._rule_registry = self._sanitize_rule_registry(self._rule_registry)
            rule_path.write_text(
                json.dumps(self._rule_registry, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def get_value_profile(self, schema: str, table: str, column: str) -> dict[str, Any]:
        """Получить value profile для колонки.

        К основным полям (top_values, known_terms, value_mode) добавляем
        sample_values из attr_list.csv (Direction 5.3): where_resolver
        использует их как детерминированный источник значений для enum_like
        колонок, даже если value_profiles.json ещё не наполнен.
        """
        self.ensure_value_profiles()
        profile = dict(
            (self._value_profiles or {}).get(self._profile_key(schema, table, column), {})
        )
        try:
            samples = self.get_column_sample_values(schema, table, column)
        except Exception:
            samples = []
        if samples:
            merged = list(profile.get("known_terms", []) or [])
            for s in samples:
                if s not in merged:
                    merged.append(s)
            profile["known_terms"] = merged
            profile.setdefault("sample_values", list(samples))
        return profile

    def get_column_semantics(self, schema: str, table: str, column: str) -> dict[str, Any]:
        """Получить semantics колонки."""
        self.ensure_column_semantics()
        key = self._profile_key(schema, table, column)
        return dict((self._column_semantics or {}).get(key, {}))

    def get_table_semantics(self, schema: str, table: str) -> dict[str, Any]:
        """Получить semantics таблицы."""
        self.ensure_table_semantics()
        key = self._table_key(schema, table)
        return dict((self._table_semantics or {}).get(key, {}))

    def get_semantic_lexicon(self) -> dict[str, Any]:
        self.ensure_semantic_registry()
        return dict(self._semantic_lexicon or {})

    def get_rule_registry(self) -> dict[str, Any]:
        self.ensure_semantic_registry()
        return dict(self._rule_registry or {})

    def get_table_grain(self, schema: str, table: str) -> str:
        """Вернуть grain таблицы или пустую строку, если он не задан."""
        tbl_sem = self.get_table_semantics(schema, table)
        if tbl_sem.get("grain"):
            return str(tbl_sem.get("grain") or "").strip().lower()
        df = self.tables_df
        mask = (df["schema_name"] == schema) & (df["table_name"] == table)
        rows = df[mask]
        if rows.empty:
            return ""
        return str(rows.iloc[0].get("grain", "") or "").strip().lower()

    def infer_query_grain(
        self,
        query: str,
        entities: list[str] | None = None,
    ) -> str | None:
        """Оценить grain запроса по тексту пользователя и entities."""
        haystack = " ".join(
            p for p in [query or ""] + [str(e) for e in (entities or []) if e]
            if p
        ).lower()
        if not haystack.strip():
            return None

        try:
            subject = find_best_subject(haystack, self.get_semantic_lexicon())
            if subject:
                return subject
        except Exception:  # noqa: BLE001
            pass

        scored: list[tuple[int, str]] = []
        for grain, keywords in _GRAIN_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if kw and kw in haystack:
                    score += 1
            if score > 0:
                scored.append((score, grain))

        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][1]

    def _fallback_infer_table_grain(self, schema: str, table: str) -> str:
        """Эвристический fallback для grain, если LLM не дал ответ."""
        table_info = self.get_table_info(schema, table).lower()
        best_grain = "other"
        best_score = 0
        for grain, keywords in _GRAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in table_info)
            if score > best_score:
                best_score = score
                best_grain = grain
        return best_grain

    def _deterministic_grain(self, schema: str, table: str, description: str) -> tuple[str, int]:
        """Детерминированный grain по имени/описанию + минимальная мера уверенности.

        Returns:
            (grain, score) — где score = число совпавших keywords (>=2 даёт уверенный sign).
        """
        haystack = f"{table} {description}".lower()
        # Split snake_case for keyword matching
        haystack = haystack.replace("_", " ")
        scored: list[tuple[int, str]] = []
        for grain, keywords in _GRAIN_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw and kw in haystack)
            if count > 0:
                scored.append((count, grain))
        if not scored:
            return ("", 0)
        scored.sort(reverse=True)
        return (scored[0][1], scored[0][0])

    def fill_deterministic_grains(self) -> int:
        """Заполнить grain для таблиц, где keyword-match даёт уверенный результат.

        Критерий "уверенный": >= 2 совпавших keywords ИЛИ единственный кандидат.
        Данные пишутся в tables_list.csv и кеш инвалидируется.

        Returns:
            Количество таблиц, для которых grain был заполнен.
        """
        df = self.tables_df.copy()
        if "grain" not in df.columns:
            df["grain"] = ""
        df["grain"] = df["grain"].fillna("").astype(str)

        filled = 0
        for idx, row in df.iterrows():
            current = str(row.get("grain", "") or "").strip().lower()
            if current and current in _GRAIN_ENUM:
                continue
            schema = str(row.get("schema_name", "") or "")
            table = str(row.get("table_name", "") or "")
            description = str(row.get("description", "") or "")
            grain, score = self._deterministic_grain(schema, table, description)
            if grain and grain in _GRAIN_ENUM and score >= 2:
                df.at[idx, "grain"] = grain
                filled += 1

        if filled:
            self._persist_tables_df(df)
            # Инвалидируем зависимые кеши — их надо пересобрать с новым grain
            self._table_semantics = None
            logger.info("Grain metadata: детерминированно заполнено %d таблиц", filled)
        return filled

    def _build_grain_prompt_payload(self, batch: pd.DataFrame) -> list[dict[str, Any]]:
        """Собрать компактный контекст по таблицам для генерации grain."""
        payload: list[dict[str, Any]] = []
        for _, row in batch.iterrows():
            schema = str(row.get("schema_name", "") or "")
            table = str(row.get("table_name", "") or "")
            cols_df = self.get_table_columns(schema, table)
            informative_cols: list[str] = []
            if not cols_df.empty:
                ranked = cols_df.copy()
                ranked["__pk"] = ranked.get("is_primary_key", False).astype(bool)
                ranked["__nn"] = ranked.get("not_null_perc", 0).fillna(0)
                ranked = ranked.sort_values(["__pk", "__nn"], ascending=[False, False])
                for _, c_row in ranked.head(12).iterrows():
                    c_name = str(c_row.get("column_name", "") or "")
                    c_desc = str(c_row.get("description", "") or "")
                    c_type = str(c_row.get("dType", "") or "")
                    informative_cols.append(f"{c_name} ({c_type}) — {c_desc}".strip())

            payload.append({
                "schema": schema,
                "table": table,
                "description": str(row.get("description", "") or ""),
                "columns": informative_cols,
            })
        return payload

    def _parse_grain_response(self, response: str) -> dict[tuple[str, str], str]:
        """Распарсить ответ LLM с grain-метаданными."""
        cleaned = re.sub(r'```(?:json)?\s*\n?', '', response)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE)
        match = re.search(r'\{.*\}|\[.*\]', cleaned, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            return {}

        if isinstance(parsed, dict):
            items = parsed.get("tables", [])
        elif isinstance(parsed, list):
            items = parsed
        else:
            items = []

        result: dict[tuple[str, str], str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            schema = str(item.get("schema", "") or "").strip()
            table = str(item.get("table", "") or "").strip()
            grain = str(item.get("grain", "") or "").strip().lower()
            if grain not in _GRAIN_ENUM:
                continue
            if schema and table:
                result[(schema, table)] = grain
        return result

    def ensure_table_grains(
        self,
        llm: Any | None = None,
        batch_size: int = 25,
    ) -> int:
        """Гарантировать, что у всех таблиц заполнен grain.

        Если столбец отсутствует или есть пустые значения, а `llm` передан,
        значения догенерируются батчами и сохраняются обратно в CSV.
        """
        df = self.tables_df.copy()
        if "grain" not in df.columns:
            df["grain"] = ""
        df["grain"] = df["grain"].fillna("").astype(str)

        missing_mask = df["grain"].str.strip() == ""
        missing_count = int(missing_mask.sum())
        if missing_count == 0:
            return 0

        if llm is None:
            logger.info(
                "Grain metadata: найдено %d таблиц без grain, но LLM не передан",
                missing_count,
            )
            return missing_count

        logger.info("Grain metadata: требуется генерация для %d таблиц", missing_count)
        system_prompt = (
            "Ты определяешь grain таблиц для аналитического агента.\n"
            "Grain — это сущность одной строки таблицы.\n"
            f"Используй ТОЛЬКО одно значение из списка: {', '.join(_GRAIN_ENUM)}.\n"
            "Верни ТОЛЬКО JSON формата:\n"
            '{\n  "tables": [\n'
            '    {"schema": "...", "table": "...", "grain": "task"}\n'
            "  ]\n}\n"
            "Правила:\n"
            "- task: одна строка на задачу/тикет/элемент воронки\n"
            "- client: одна строка на клиента\n"
            "- employee: одна строка на сотрудника\n"
            "- organization: одна строка на орг-единицу/ГОСБ/ТБ\n"
            "- event: одна строка на событие/факт наступления\n"
            "- transaction: одна строка на транзакцию/заказ/продажу\n"
            "- snapshot: одна строка на периодический срез состояния\n"
            "- dictionary: справочник/lookup/reference\n"
            "- document/payment/account/product/other — по смыслу\n"
        )

        missing_df = df[missing_mask].copy()
        for start in range(0, len(missing_df), batch_size):
            batch = missing_df.iloc[start:start + batch_size]
            payload = self._build_grain_prompt_payload(batch)
            user_prompt = (
                "Определи grain для таблиц:\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
            )
            response = llm.invoke_with_system(system_prompt, user_prompt, temperature=0.0)
            generated = self._parse_grain_response(str(response))

            for _, row in batch.iterrows():
                schema = str(row.get("schema_name", "") or "")
                table = str(row.get("table_name", "") or "")
                grain = generated.get((schema, table)) or self._fallback_infer_table_grain(schema, table)
                df.loc[
                    (df["schema_name"] == schema) & (df["table_name"] == table),
                    "grain",
                ] = grain

        self._persist_tables_df(df)
        logger.info("Grain metadata: генерация завершена")
        return 0

    def get_foreign_key_target(
        self, schema: str, table: str, column: str
    ) -> tuple[str, str, str] | None:
        """Получить (ref_schema, ref_table, ref_column) если задан foreign_key_target.

        Формат значения в CSV: "schema.table.column". При отсутствии/ошибке — None.
        """
        df = self.attrs_df
        if "foreign_key_target" not in df.columns:
            return None
        mask = (
            (df["schema_name"] == schema)
            & (df["table_name"] == table)
            & (df["column_name"] == column)
        )
        rows = df[mask]
        if rows.empty:
            return None
        raw = str(rows.iloc[0].get("foreign_key_target", "") or "").strip()
        if not raw:
            return None
        parts = raw.split(".")
        if len(parts) != 3:
            return None
        ref_schema, ref_table, ref_column = (p.strip() for p in parts)
        if not (ref_schema and ref_table and ref_column):
            return None
        return (ref_schema, ref_table, ref_column)

    def get_column_sample_values(
        self, schema: str, table: str, column: str
    ) -> list[str]:
        """Получить sample_values колонки (разделитель — '|')."""
        df = self.attrs_df
        if "sample_values" not in df.columns:
            return []
        mask = (
            (df["schema_name"] == schema)
            & (df["table_name"] == table)
            & (df["column_name"] == column)
        )
        rows = df[mask]
        if rows.empty:
            return []
        raw = str(rows.iloc[0].get("sample_values", "") or "").strip()
        if not raw:
            return []
        return [part.strip() for part in raw.split("|") if part.strip()]

    def get_column_synonyms(self, schema: str, table: str, column: str) -> list[str]:
        """Получить synonyms колонки (разделитель — ',')."""
        df = self.attrs_df
        if "synonyms" not in df.columns:
            return []
        mask = (
            (df["schema_name"] == schema)
            & (df["table_name"] == table)
            & (df["column_name"] == column)
        )
        rows = df[mask]
        if rows.empty:
            return []
        raw = str(rows.iloc[0].get("synonyms", "") or "").strip()
        if not raw:
            return []
        return [part.strip().lower() for part in raw.split(",") if part.strip()]

    def is_partition_key(self, schema: str, table: str, column: str) -> bool:
        """Проверить, помечена ли колонка как partition_key."""
        df = self.attrs_df
        if "partition_key" not in df.columns:
            return False
        mask = (
            (df["schema_name"] == schema)
            & (df["table_name"] == table)
            & (df["column_name"] == column)
        )
        rows = df[mask]
        if rows.empty:
            return False
        return bool(rows.iloc[0].get("partition_key", False))

    def get_column_dtype(self, schema: str, table: str, column: str) -> str:
        """Получить dtype колонки (пустая строка если не найдена)."""
        df = self.attrs_df
        mask = (
            (df["schema_name"] == schema)
            & (df["table_name"] == table)
            & (df["column_name"] == column)
        )
        rows = df[mask]
        if rows.empty:
            return ""
        return str(rows.iloc[0].get("dType", "") or "").strip().lower()

    def infer_foreign_keys(self, dry_run: bool = False) -> dict[str, str]:
        """Детерминированная инференция foreign_key_target.

        Правила (консервативно):
        - Кандидат-колонка T.c совпадает по имени с PK другой таблицы R.pk
          (после нормализации через join_analysis._normalize_key_name).
        - У R.pk unique_perc == 100 (единственное значение — хорошая цель FK).
        - Типы совпадают (нормализация целых: int2/int4/int8 → int).
        - T != R (само-ссылок не делаем).

        Если не dry_run — заполняет attr_list.csv и перезагружает кеш.

        Returns:
            Словарь {T.c_key → "schema.table.column"} для заполненных/предложенных FK.
        """
        from core.join_analysis import _normalize_key_name  # lazy import

        df = self.attrs_df.copy()
        if "foreign_key_target" not in df.columns:
            df["foreign_key_target"] = ""

        def _dtype_bucket(dtype: str) -> str:
            d = (dtype or "").lower()
            if d.startswith("int") or d in {"bigint", "smallint", "integer"}:
                return "int"
            if "char" in d or d == "text":
                return "text"
            if "num" in d or "float" in d or "double" in d or "decimal" in d:
                return "num"
            if "date" in d or "time" in d:
                return "date"
            return d

        # Каталог PK с unique_perc=100 — потенциальные цели FK.
        pk_targets: list[tuple[str, str, str, str]] = []  # (schema, table, column, norm_name)
        for _, row in df.iterrows():
            if not bool(row.get("is_primary_key", False)):
                continue
            try:
                up = float(row.get("unique_perc", 0) or 0)
            except (TypeError, ValueError):
                up = 0.0
            if up < 100.0:
                continue
            pk_targets.append((
                str(row.get("schema_name", "")),
                str(row.get("table_name", "")),
                str(row.get("column_name", "")),
                _normalize_key_name(str(row.get("column_name", ""))),
            ))

        filled: dict[str, str] = {}
        for idx, row in df.iterrows():
            existing = str(row.get("foreign_key_target", "") or "").strip()
            if existing:
                continue
            if bool(row.get("is_primary_key", False)):
                continue  # PK сам — не FK
            col = str(row.get("column_name", ""))
            schema = str(row.get("schema_name", ""))
            table = str(row.get("table_name", ""))
            norm_col = _normalize_key_name(col)
            col_bucket = _dtype_bucket(str(row.get("dType", "")))
            matches = [
                t for t in pk_targets
                if t[3] == norm_col
                and (t[0] != schema or t[1] != table)  # не на себя
            ]
            if len(matches) != 1:
                continue  # требуем однозначность
            ref_schema, ref_table, ref_column, _ = matches[0]
            # Типы должны совпадать
            ref_row = df[
                (df["schema_name"] == ref_schema)
                & (df["table_name"] == ref_table)
                & (df["column_name"] == ref_column)
            ]
            if ref_row.empty:
                continue
            if _dtype_bucket(str(ref_row.iloc[0].get("dType", ""))) != col_bucket:
                continue
            target = f"{ref_schema}.{ref_table}.{ref_column}"
            filled[f"{schema}.{table}.{col}"] = target
            if not dry_run:
                df.at[idx, "foreign_key_target"] = target

        if filled and not dry_run:
            path = self._data_dir / "attr_list.csv"
            df.to_csv(path, index=False, encoding="utf-8")
            self._attrs_df = None  # force reload
            logger.info("FK inference: заполнено %d foreign_key_target", len(filled))
        return filled

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
        mask = _literal_contains(df["column_name"], column_name)
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
        synonyms = expand_with_synonyms(text) if _should_expand_synonyms(text) else []

        # Поиск в описаниях и именах таблиц
        tdf = self.tables_df
        t_mask = _literal_contains(tdf["description"], q) | _literal_contains(tdf["table_name"], q)
        for syn in synonyms:
            t_mask = (
                t_mask
                | _literal_contains(tdf["description"], syn)
                | _literal_contains(tdf["table_name"], syn)
            )
        tables_found = tdf[t_mask][["schema_name", "table_name", "description"]].copy()
        tables_found["column_name"] = ""
        tables_found["source"] = "table"

        # Поиск в описаниях и именах атрибутов
        adf = self.attrs_df
        a_mask = _literal_contains(adf["description"], q) | _literal_contains(adf["column_name"], q)
        for syn in synonyms:
            a_mask = (
                a_mask
                | _literal_contains(adf["description"], syn)
                | _literal_contains(adf["column_name"], syn)
            )
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

        # needs_db_probe: CSV-статистика не гарантирует отсутствие fanout.
        # Обязательна DB-верификация для:
        # - составных ключей без полной PK-гарантии (all_pk=False)
        # - ключей с колонками, отсутствующими в справочнике
        needs_db_probe = (
            len(columns) > 1 and not all_pk
        ) or any(not d.get("found") for d in details.values())

        return {
            "is_unique": is_unique,
            "all_pk": all_pk,
            "min_unique_perc": min_unique_perc,
            "duplicate_pct": duplicate_pct,
            "columns": details,
            "status": status,
            "needs_db_probe": needs_db_probe,
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
