"""Постоянный локальный журнал для совместной отладки работающего приложения.

Обычный текстовый лог хорошо отвечает на вопрос «какое сообщение было
последним», но почти бесполезен при рассинхронизации интерфейса и математической
модели. ``LiveDebugSession`` поэтому хранит два взаимодополняющих представления:

* ``events.jsonl`` — неизменяемую последовательность отдельных событий;
* ``state.json`` — атомарно заменяемый снимок актуального состояния.

JSON Lines означает, что каждая строка файла является самостоятельным JSON.
Так журнал можно читать командой ``tail -F`` ещё во время работы приложения, а
оборванная последняя запись не повреждает все предыдущие события.

Модуль намеренно не зависит от Qt. GUI передаёт сюда только обычные словари и
сам сохраняет изображения. Благодаря этому формат журнала проверяется быстрыми
unit-тестами без запуска оконной системы.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import threading
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


class LiveDebugSession:
    """Создать каталог живой диагностики и безопасно дописывать события.

    Все пути остаются локальными. Каталог ``outputs`` исключён из Git, поэтому
    кадры, координаты и имена исходных файлов не попадут в коммит случайно.

    Запись выполняется синхронно и построчно. Объём одного события мал, а
    ``flush`` после строки важнее микроскопического выигрыша скорости: при
    аварии процесса уже записанные наблюдения должны остаться читаемыми.
    Тяжёлые PNG сохраняет GUI только с ограниченной частотой.
    """

    SCHEMA_VERSION = 1

    def __init__(self, root_directory: str | Path) -> None:
        self.root_directory = Path(root_directory).expanduser().resolve()
        self.root_directory.mkdir(parents=True, exist_ok=True)

        started_at = datetime.now(UTC)
        timestamp = started_at.strftime("%Y%m%dT%H%M%S.%fZ")
        self.session_id = f"{timestamp}-pid{os.getpid()}-{secrets.token_hex(2)}"
        self.session_directory = self.root_directory / "sessions" / self.session_id
        self.session_directory.mkdir(parents=True, exist_ok=False)
        self.events_path = self.session_directory / "events.jsonl"
        self.state_path = self.session_directory / "state.json"
        self.incidents_directory = self.session_directory / "incidents"
        self.incidents_directory.mkdir()

        self._lock = threading.RLock()
        self._event_sequence = 0
        self._incident_sequence = 0
        self._closed = False
        self._events_file = self.events_path.open("a", encoding="utf-8", buffering=1)

        self._write_latest_pointer(started_at)
        self.record_event(
            "session_started",
            {
                "schema_version": self.SCHEMA_VERSION,
                "process_id": os.getpid(),
                "session_directory": str(self.session_directory),
            },
        )

    @property
    def latest_event_sequence(self) -> int:
        """Номер последнего полностью записанного события."""

        return self._event_sequence

    @property
    def closed(self) -> bool:
        """Был ли журнал уже корректно закрыт."""

        return self._closed

    def record_event(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Добавить одно структурированное событие и немедленно его сбросить.

        ``event_type`` — стабильное машинное имя вроде ``tracking_failed``.
        Человеческое объяснение ошибки хранится отдельным полем в ``payload``.
        Такое разделение позволяет фильтровать события без разбора русского
        текста.
        """

        if not event_type or not event_type.strip():
            raise ValueError("Тип диагностического события не может быть пустым")

        with self._lock:
            if self._closed:
                raise RuntimeError("Диагностическая сессия уже закрыта")
            self._event_sequence += 1
            event = {
                "schema_version": self.SCHEMA_VERSION,
                "session_id": self.session_id,
                "sequence": self._event_sequence,
                "timestamp_utc": _utc_now_text(),
                "event_type": event_type,
                "payload": _json_value(payload or {}),
            }
            self._events_file.write(
                json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            self._events_file.flush()
            return event

    def write_state(self, state: Mapping[str, Any]) -> Path:
        """Атомарно заменить снимок текущего состояния приложения.

        Временный файл создаётся рядом с целевым и затем заменяется через
        ``os.replace``. Читатель увидит либо старую, либо новую полную версию,
        но не половину JSON во время записи.
        """

        with self._lock:
            envelope = {
                "schema_version": self.SCHEMA_VERSION,
                "session_id": self.session_id,
                "updated_at_utc": _utc_now_text(),
                "latest_event_sequence": self._event_sequence,
                "state": _json_value(state),
            }
            _atomic_write_json(self.state_path, envelope)
            return self.state_path

    def create_incident(
        self,
        reason: str,
        state: Mapping[str, Any],
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> Path:
        """Зафиксировать отдельный каталог проблемного момента.

        Метод сохраняет только JSON. Кадр, карту и изображение окна добавляет
        GUI после возврата пути, потому что этот модуль ничего не знает о Qt и
        порядке цветовых каналов OpenCV.
        """

        with self._lock:
            if self._closed:
                raise RuntimeError("Диагностическая сессия уже закрыта")
            self._incident_sequence += 1
            safe_reason = _safe_filename_component(reason)
            incident_directory = self.incidents_directory / (
                f"{self._incident_sequence:06d}-{safe_reason}"
            )
            incident_directory.mkdir()
            _atomic_write_json(
                incident_directory / "incident.json",
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "session_id": self.session_id,
                    "incident_number": self._incident_sequence,
                    "created_at_utc": _utc_now_text(),
                    "reason": reason,
                    "latest_event_sequence": self._event_sequence,
                    "payload": _json_value(payload or {}),
                    "state": _json_value(state),
                },
            )
            return incident_directory

    def record_exception(
        self,
        exception_type: type[BaseException],
        exception: BaseException,
        traceback_object: object,
    ) -> None:
        """Сохранить необработанное исключение до штатного вывода Python."""

        formatted = "".join(
            traceback.format_exception(
                exception_type,
                exception,
                traceback_object,  # type: ignore[arg-type]
            )
        )
        self.record_event(
            "unhandled_exception",
            {
                "exception_type": exception_type.__name__,
                "message": str(exception),
                "traceback": formatted,
            },
        )

    def close(self) -> None:
        """Записать конец сессии и закрыть файл; повторный вызов безопасен."""

        with self._lock:
            if self._closed:
                return
            self.record_event("session_closed")
            self._closed = True
            self._events_file.close()

    def _write_latest_pointer(self, started_at: datetime) -> None:
        """Обновить маленький стабильный файл поиска текущей сессии."""

        _atomic_write_json(
            self.root_directory / "latest_session.json",
            {
                "schema_version": self.SCHEMA_VERSION,
                "session_id": self.session_id,
                "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
                "session_directory": str(self.session_directory),
                "events_path": str(self.events_path),
                "state_path": str(self.state_path),
            },
        )


def _utc_now_text() -> str:
    """Вернуть время UTC в однозначном ISO-формате."""

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_value(value: Any) -> Any:
    """Преобразовать NumPy и нечисловые значения в строгий JSON.

    ``NaN`` и бесконечности не входят в стандарт JSON. Они превращаются в
    ``null``: отдельные маски валидности в событиях сохраняют смысл такого
    отсутствующего измерения, а любой обычный JSON-парсер читает файл без
    специальных расширений.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    return str(value)


def _safe_filename_component(value: str) -> str:
    """Получить короткую переносимую часть имени каталога инцидента."""

    normalised = "".join(
        character.lower() if character.isalnum() else "-" for character in value
    )
    compact = "-".join(part for part in normalised.split("-") if part)
    return (compact or "incident")[:64]


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Записать JSON через временный соседний файл и атомарную замену."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(
                _json_value(payload),
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
