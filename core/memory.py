"""Персистентная память агента на JSON-файлах."""

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory"


def _write_json_atomic(path: Path, data: Any) -> None:
    """Атомарно записать данные в JSON-файл (write-to-tmp + rename)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path, default: Any) -> Any:
    """Загрузить JSON-файл или вернуть default если файл не существует."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Ошибка чтения %s: %s — используем default", path, e)
        return default


class MemoryManager:
    """Управление персистентной памятью агента через JSON-файлы."""

    def __init__(
        self,
        memory_dir: Path | None = None,
        *,
        db_path: Path | None = None,
    ) -> None:
        """Инициализация менеджера памяти.

        Args:
            memory_dir: Директория для хранения JSON-файлов.
                        По умолчанию — memory/ в корне проекта.
            db_path: Устаревший параметр (SQLite-путь). Если передан —
                     используется его родительская директория.
        """
        if db_path is not None:
            # Обратная совместимость: принимаем путь к .db и берём директорию
            resolved_dir = db_path.parent
        else:
            resolved_dir = memory_dir or MEMORY_DIR

        self._memory_dir = resolved_dir
        self._memory_dir.mkdir(parents=True, exist_ok=True)

        self._sessions_path = self._memory_dir / "sessions.json"
        self._ltm_path = self._memory_dir / "long_term_memory.json"

        self._session_id: str | None = None
        logger.info("Память инициализирована: %s", self._memory_dir)

    # ------------------------------------------------------------------
    # Внутренние помощники
    # ------------------------------------------------------------------

    def _load_sessions(self) -> dict:
        return _load_json(self._sessions_path, {})

    def _save_sessions(self, data: dict) -> None:
        _write_json_atomic(self._sessions_path, data)

    def _load_ltm(self) -> dict:
        return _load_json(self._ltm_path, {})

    def _save_ltm(self, data: dict) -> None:
        _write_json_atomic(self._ltm_path, data)

    # ------------------------------------------------------------------
    # Управление сессиями
    # ------------------------------------------------------------------

    def start_session(self, user_id: str = "") -> str:
        """Начать новую сессию.

        Args:
            user_id: Идентификатор пользователя.

        Returns:
            ID новой сессии.
        """
        self._session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        sessions = self._load_sessions()
        sessions[self._session_id] = {
            "timestamp": now,
            "summary": "",
            "user_id": user_id,
            "messages": [],
            "sql_audit": [],
        }
        self._save_sessions(sessions)
        logger.info("Начата сессия: %s", self._session_id)
        return self._session_id

    @property
    def session_id(self) -> str | None:
        """ID текущей сессии."""
        return self._session_id

    def add_message(self, role: str, content: str) -> None:
        """Записать сообщение в текущую сессию.

        Args:
            role: Роль отправителя ('user', 'assistant', 'tool').
            content: Текст сообщения.
        """
        if not self._session_id:
            logger.warning("Нет активной сессии, сообщение не записано")
            return

        now = datetime.now(timezone.utc).isoformat()
        sessions = self._load_sessions()
        if self._session_id not in sessions:
            logger.warning("Сессия %s не найдена", self._session_id)
            return
        sessions[self._session_id]["messages"].append(
            {"role": role, "content": content, "timestamp": now}
        )
        self._save_sessions(sessions)

    def save_session_summary(self, summary: str) -> None:
        """Сохранить резюме текущей сессии.

        Args:
            summary: Текст резюме, сгенерированный LLM.
        """
        if not self._session_id:
            logger.warning("Нет активной сессии для сохранения резюме")
            return

        sessions = self._load_sessions()
        if self._session_id in sessions:
            sessions[self._session_id]["summary"] = summary
            self._save_sessions(sessions)
        logger.info("Резюме сессии сохранено: %s", self._session_id)

    def get_recent_sessions(self, n: int = 5) -> list[dict[str, str]]:
        """Получить резюме последних N сессий.

        Args:
            n: Количество сессий.

        Returns:
            Список словарей с id, timestamp, summary.
        """
        sessions = self._load_sessions()
        with_summary = [
            {"id": sid, "timestamp": data["timestamp"], "summary": data["summary"]}
            for sid, data in sessions.items()
            if data.get("summary")
        ]
        with_summary.sort(key=lambda x: x["timestamp"], reverse=True)
        return with_summary[:n]

    def get_sessions_context(self, n: int = 5) -> str:
        """Сформировать контекст из последних сессий для системного промпта.

        Args:
            n: Количество сессий.

        Returns:
            Форматированная строка с резюме.
        """
        sessions = self.get_recent_sessions(n)
        if not sessions:
            return "Предыдущих сессий нет."

        lines = ["Резюме предыдущих сессий:"]
        for s in reversed(sessions):
            lines.append(f"  [{s['timestamp'][:10]}] {s['summary']}")
        return "\n".join(lines)

    def get_session_messages(self, session_id: str | None = None) -> list[dict[str, str]]:
        """Получить сообщения сессии.

        Args:
            session_id: ID сессии. По умолчанию — текущая.

        Returns:
            Список сообщений с role и content.
        """
        sid = session_id or self._session_id
        if not sid:
            return []

        sessions = self._load_sessions()
        session = sessions.get(sid)
        if not session:
            return []
        return list(session.get("messages", []))

    def get_unsummarized_sessions(self) -> list[str]:
        """Получить ID сессий без резюме, у которых есть сообщения.

        Используется для восстановления памяти после аварийного завершения.

        Returns:
            Список ID незавершённых сессий (исключая текущую).
        """
        sessions = self._load_sessions()
        result = []
        for sid, data in sessions.items():
            if (
                sid != (self._session_id or "")
                and not data.get("summary")
                and data.get("messages")
            ):
                result.append(sid)
        # Сортируем по времени, берём последние 5
        result.sort(
            key=lambda s: sessions[s].get("timestamp", ""),
            reverse=True,
        )
        return result[:5]

    # ------------------------------------------------------------------
    # SQL-аудит
    # ------------------------------------------------------------------

    def log_sql_execution(
        self,
        user_input: str,
        sql: str,
        row_count: int,
        status: str,
        duration_ms: int,
        retry_count: int = 0,
        error_type: str = "",
    ) -> None:
        """Записать выполненный SQL в аудит-лог текущей сессии.

        Args:
            user_input: Исходный запрос пользователя.
            sql: Выполненный SQL-запрос.
            row_count: Количество строк в результате.
            status: Статус выполнения ('success', 'empty', 'error', 'row_explosion').
            duration_ms: Время выполнения в миллисекундах.
            retry_count: Количество попыток коррекции до успешного выполнения.
            error_type: Тип ошибки (например, 'syntax', 'join_explosion').
        """
        now = datetime.now(timezone.utc).isoformat()
        entry = {
            "session_id": self._session_id or "",
            "timestamp": now,
            "user_input": user_input,
            "sql": sql,
            "row_count": row_count,
            "status": status,
            "duration_ms": duration_ms,
            "retry_count": retry_count,
            "error_type": error_type,
        }
        try:
            sessions = self._load_sessions()
            sid = self._session_id or ""
            if sid in sessions:
                sessions[sid].setdefault("sql_audit", []).append(entry)
                self._save_sessions(sessions)
        except Exception as e:
            logger.warning("Ошибка записи SQL-аудита: %s", e)

    def iter_sql_audit(
        self,
        *,
        status: str | None = "success",
        max_retry_count: int | None = 0,
        min_sql_length: int = 20,
        min_row_count: int | None = 1,
        max_row_count: int | None = None,
        days: int | None = 90,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Перебрать записи sql_audit всех сессий с фильтрами.

        Публичное API поверх JSON-хранилища: заменяет ранее ошибочно
        использовавшийся `_connect()` (его в JSON-реализации нет).

        Args:
            status: Требуемый статус ('success', 'empty', 'error'). None — не фильтровать.
            max_retry_count: Максимум retry_count (0 = только first-try). None — не фильтровать.
            min_sql_length: Минимальная длина sql (по умолчанию 20).
            min_row_count: Минимальный row_count. None — не фильтровать.
            max_row_count: Максимальный row_count. None — не фильтровать.
            days: Срок давности (от сегодня). None — не фильтровать по времени.
            limit: Максимум записей в ответе. None — все.

        Returns:
            Список записей audit, отсортированных по timestamp DESC.
        """
        cutoff = ""
        if days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        sessions = self._load_sessions()
        entries: list[dict[str, Any]] = []
        for sid, data in sessions.items():
            for entry in data.get("sql_audit", []):
                if status is not None and entry.get("status") != status:
                    continue
                if max_retry_count is not None and int(entry.get("retry_count", 0)) > max_retry_count:
                    continue
                sql = str(entry.get("sql") or "")
                if len(sql) < min_sql_length:
                    continue
                rc = int(entry.get("row_count", 0))
                if min_row_count is not None and rc < min_row_count:
                    continue
                if max_row_count is not None and rc > max_row_count:
                    continue
                if cutoff and entry.get("timestamp", "") < cutoff:
                    continue
                entries.append(dict(entry))

        entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        if limit is not None:
            entries = entries[: max(0, int(limit))]
        return entries

    def get_sql_quality_metrics(self, days: int = 30) -> dict[str, Any]:
        """Получить метрики качества генерации SQL за указанный период.

        Args:
            days: Период в днях для анализа.

        Returns:
            Словарь с метриками качества.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        sessions = self._load_sessions()

        all_entries = []
        for data in sessions.values():
            for entry in data.get("sql_audit", []):
                if entry.get("timestamp", "") >= cutoff:
                    all_entries.append(entry)

        total = len(all_entries)
        if total == 0:
            return {"total_queries": 0, "period_days": days}

        status_dist: dict[str, int] = {}
        error_dist: dict[str, int] = {}
        total_retries = 0
        max_retries = 0
        total_duration = 0
        max_duration = 0
        first_try_success = 0

        for e in all_entries:
            st = e.get("status", "")
            status_dist[st] = status_dist.get(st, 0) + 1

            et = e.get("error_type", "")
            if et:
                error_dist[et] = error_dist.get(et, 0) + 1

            rc = int(e.get("retry_count", 0))
            total_retries += rc
            if rc > max_retries:
                max_retries = rc

            dm = int(e.get("duration_ms", 0))
            total_duration += dm
            if dm > max_duration:
                max_duration = dm

            if st == "success" and rc == 0:
                first_try_success += 1

        success_count = status_dist.get("success", 0)
        return {
            "total_queries": total,
            "period_days": days,
            "success_rate": round(success_count / total * 100, 1),
            "first_try_success_rate": round(first_try_success / total * 100, 1),
            "status_distribution": status_dist,
            "avg_retries": round(total_retries / total, 2),
            "max_retries": max_retries,
            "avg_duration_ms": round(total_duration / total, 0),
            "max_duration_ms": max_duration,
            "error_distribution": error_dist,
        }

    # ------------------------------------------------------------------
    # Row-count sanity (Direction 3.3)
    # ------------------------------------------------------------------

    _ROW_COUNT_STATS_KEY = "row_count_stats"
    _ROW_COUNT_MAX_SAMPLES = 200

    @staticmethod
    def _row_count_bucket(subject: str | None, metric: str | None) -> str:
        """Сформировать ключ бакета по (subject, metric) для row_count_stats."""
        s = (subject or "").strip().lower() or "_"
        m = (metric or "").strip().lower() or "_"
        return f"{s}|{m}"

    @staticmethod
    def _percentile(values: list[int], q: float) -> float:
        """Линейная интерполяция перцентиля без numpy."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        if len(sorted_vals) == 1:
            return float(sorted_vals[0])
        k = (len(sorted_vals) - 1) * q
        lo = int(k)
        hi = min(lo + 1, len(sorted_vals) - 1)
        frac = k - lo
        return float(sorted_vals[lo]) * (1 - frac) + float(sorted_vals[hi]) * frac

    def record_row_count_sample(
        self, subject: str | None, metric: str | None, row_count: int,
    ) -> None:
        """Записать наблюдение row_count в распределение по (subject, metric).

        Хранится в long_term_memory.json под `row_count_stats`. Скользящее окно
        из последних `_ROW_COUNT_MAX_SAMPLES` значений.
        """
        if row_count is None or row_count < 0:
            return
        bucket = self._row_count_bucket(subject, metric)
        now = datetime.now(timezone.utc).isoformat()
        try:
            ltm = self._load_ltm()
            stats_raw = ltm.get(self._ROW_COUNT_STATS_KEY)
            stats: dict[str, Any]
            if isinstance(stats_raw, dict) and "value" in stats_raw and isinstance(stats_raw["value"], dict):
                stats = dict(stats_raw["value"])
            elif isinstance(stats_raw, dict):
                stats = dict(stats_raw)
            else:
                stats = {}

            entry = stats.get(bucket) or {}
            samples: list[int] = list(entry.get("samples", []))
            samples.append(int(row_count))
            if len(samples) > self._ROW_COUNT_MAX_SAMPLES:
                samples = samples[-self._ROW_COUNT_MAX_SAMPLES:]
            entry["samples"] = samples
            entry["p95"] = self._percentile(samples, 0.95)
            entry["p50"] = self._percentile(samples, 0.50)
            entry["n"] = len(samples)
            entry["updated_at"] = now
            stats[bucket] = entry

            ltm[self._ROW_COUNT_STATS_KEY] = {"value": stats, "updated_at": now}
            self._save_ltm(ltm)
        except Exception as e:
            logger.warning("row_count_stats: не удалось сохранить наблюдение: %s", e)

    def check_row_count_suspicion(
        self,
        subject: str | None,
        metric: str | None,
        row_count: int,
        *,
        explosion_factor: float = 10.0,
        min_samples: int = 5,
    ) -> dict[str, Any]:
        """Определить, является ли текущий row_count подозрительно большим.

        Args:
            subject: semantic_frame.subject.
            metric: semantic_frame.metric_intent.
            row_count: количество строк из последнего запроса.
            explosion_factor: во сколько раз больше p95 трактовать как взрыв.
            min_samples: сколько наблюдений нужно, чтобы доверять p95.

        Returns:
            dict: {"is_suspect": bool, "p95": float, "n": int, "ratio": float}.
        """
        bucket = self._row_count_bucket(subject, metric)
        try:
            ltm = self._load_ltm()
            stats_raw = ltm.get(self._ROW_COUNT_STATS_KEY)
            if isinstance(stats_raw, dict) and "value" in stats_raw and isinstance(stats_raw["value"], dict):
                stats = stats_raw["value"]
            elif isinstance(stats_raw, dict):
                stats = stats_raw
            else:
                stats = {}
        except Exception as e:
            logger.warning("row_count_stats: ошибка чтения: %s", e)
            return {"is_suspect": False, "p95": 0.0, "n": 0, "ratio": 0.0}

        entry = stats.get(bucket) or {}
        n = int(entry.get("n", 0))
        p95 = float(entry.get("p95", 0.0))
        if n < min_samples or p95 <= 0:
            return {"is_suspect": False, "p95": p95, "n": n, "ratio": 0.0}
        threshold = p95 * explosion_factor
        ratio = float(row_count) / max(p95, 1.0)
        return {
            "is_suspect": row_count > threshold,
            "p95": p95,
            "n": n,
            "ratio": ratio,
        }

    # ------------------------------------------------------------------
    # Долгосрочная память
    # ------------------------------------------------------------------

    def set_memory(self, key: str, value: str) -> None:
        """Записать или обновить значение в долгосрочной памяти.

        Args:
            key: Ключ.
            value: Значение.
        """
        now = datetime.now(timezone.utc).isoformat()
        ltm = self._load_ltm()
        ltm[key] = {"value": value, "updated_at": now}
        self._save_ltm(ltm)
        logger.info("Долгосрочная память обновлена: key=%s", key)

    def get_memory(self, key: str) -> str | None:
        """Получить значение из долгосрочной памяти.

        Args:
            key: Ключ.

        Returns:
            Значение или None если не найдено.
        """
        ltm = self._load_ltm()
        entry = ltm.get(key)
        return entry["value"] if entry else None

    def get_all_memory(self) -> dict[str, str]:
        """Получить всю долгосрочную память.

        Returns:
            Словарь key -> value.
        """
        ltm = self._load_ltm()
        return {k: v["value"] for k, v in ltm.items()}

    def get_memory_list(self, key: str) -> list[str]:
        """Получить значение из долгосрочной памяти как JSON-список.

        Args:
            key: Ключ.

        Returns:
            Список строк или пустой список если не найдено / ошибка парсинга.
        """
        raw = self.get_memory(key)
        if not raw:
            return []
        try:
            val = json.loads(raw)
            return val if isinstance(val, list) else []
        except (json.JSONDecodeError, ValueError):
            return []

    def delete_memory(self, key: str) -> None:
        """Удалить запись из долгосрочной памяти.

        Args:
            key: Ключ для удаления.
        """
        ltm = self._load_ltm()
        if key in ltm:
            del ltm[key]
            self._save_ltm(ltm)
        logger.info("Долгосрочная память удалена: key=%s", key)

    # ------------------------------------------------------------------
    # Очистка
    # ------------------------------------------------------------------

    def cleanup_old_sessions(self, keep_days: int = 90) -> int:
        """Удалить сессии старше keep_days дней.

        Args:
            keep_days: Сколько дней хранить (по умолчанию 90).

        Returns:
            Количество удалённых сессий.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).isoformat()
        sessions = self._load_sessions()

        to_delete = [
            sid for sid, data in sessions.items()
            if data.get("timestamp", "") < cutoff
        ]

        if to_delete:
            for sid in to_delete:
                del sessions[sid]
            self._save_sessions(sessions)
            logger.info("Очищено %d старых сессий (старше %d дней)", len(to_delete), keep_days)

        return len(to_delete)

    # ------------------------------------------------------------------
    # Свойства и совместимость
    # ------------------------------------------------------------------

    @property
    def session_count(self) -> int:
        """Количество сессий с резюме."""
        sessions = self._load_sessions()
        return sum(1 for data in sessions.values() if data.get("summary"))

    def close(self) -> None:
        """Совместимость с прежним API."""
