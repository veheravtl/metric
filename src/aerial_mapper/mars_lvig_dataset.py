"""Безопасное извлечение RGB и RTK-телеметрии из MARS-LVIG ROS bag.

Набор хранит много сенсоров, но текущему Teach/Repeat-эксперименту нужны только
сжатые кадры нижней камеры и контрольные RTK-координаты. Координаты экспортируются
в отдельный CSV и не должны подаваться на вход алгоритму визуальной привязки.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from rosbags.highlevel import AnyReader

CAMERA_MESSAGE_TYPE = "sensor_msgs/msg/CompressedImage"
RTK_MESSAGE_TYPE = "sensor_msgs/msg/NavSatFix"


@dataclass(frozen=True)
class MarsLvigExportSummary:
    """Проверяемая сводка одного извлечённого полёта."""

    bag_path: str
    output_directory: str
    camera_topic: str
    rtk_topic: str
    bag_start_timestamp_nanoseconds: int
    export_start_timestamp_nanoseconds: int
    route_start_offset_seconds: float
    image_count: int
    rtk_count: int
    rejected_rtk_count: int
    image_header_timestamp_count: int
    image_bag_timestamp_fallback_count: int
    rtk_header_timestamp_count: int
    rtk_bag_timestamp_fallback_count: int
    image_extension: str


def _message_timestamp_nanoseconds(
    message: Any,
    *,
    bag_timestamp_nanoseconds: int,
) -> tuple[int, str]:
    """Берёт время измерения из заголовка, при нулевом времени — из bag.

    Время записи bag может немного отличаться от физического момента экспозиции,
    поэтому нормальный путь использует header.stamp. Нулевой stamp встречается
    в некорректно настроенных узлах; тогда явный fallback лучше выдуманного времени.
    """

    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    seconds = getattr(stamp, "sec", None)
    nanoseconds = getattr(stamp, "nanosec", None)
    if (
        isinstance(seconds, int)
        and isinstance(nanoseconds, int)
        and seconds >= 0
        and 0 <= nanoseconds < 1_000_000_000
        and (seconds != 0 or nanoseconds != 0)
    ):
        return seconds * 1_000_000_000 + nanoseconds, "header"
    return int(bag_timestamp_nanoseconds), "bag"


def _compressed_image_extension(payload: bytes) -> str:
    """Определяет только два ожидаемых формата по сигнатуре, а не по имени."""

    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    raise ValueError("Кадр не имеет сигнатуры JPEG или PNG")


def _write_bytes_without_silent_overwrite(path: Path, payload: bytes) -> None:
    """Атомарно создаёт кадр; существующий допускается лишь при полном совпадении."""

    if path.exists():
        if path.stat().st_size == len(payload) and path.read_bytes() == payload:
            return
        raise RuntimeError(f"Существующий кадр отличается от bag: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".extract",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            temporary_file.write(payload)
            temporary_file.flush()
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(path)


def _write_text_atomically(path: Path, text: str) -> None:
    """Публикует метаданные одним переименованием после полного формирования."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".extract",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            temporary_file.write(text)
            temporary_file.flush()
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(path)


def _csv_text(header: list[str], rows: list[list[object]]) -> str:
    """Формирует одинаковый перенос строк независимо от операционной системы."""

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue()


def _require_strictly_increasing(
    previous_timestamp: int | None,
    timestamp: int,
    *,
    stream_name: str,
) -> None:
    """Не разрешает интерполяцию телеметрии по перемешанному времени."""

    if previous_timestamp is not None and timestamp <= previous_timestamp:
        raise RuntimeError(
            f"Временные метки {stream_name} не возрастают строго: "
            f"{timestamp} после {previous_timestamp}"
        )


def export_mars_lvig_bag(
    bag_path: Path,
    output_directory: Path,
    *,
    camera_topic: str,
    rtk_topic: str,
    route_start_offset_seconds: float,
) -> MarsLvigExportSummary:
    """Извлекает исходные JPEG/PNG и RTK в формат текущего оценочного конвейера.

    route_start_offset_seconds отбрасывает предстартовое зависание, которое
    авторы FAST-LIVO2 задают отдельно для каждого HKairport-полёта. Оно считается
    от первой временной метки bag, а не от первого кадра камеры.

    Функция не декодирует и не перекодирует изображение: сохранённые байты
    совпадают с sensor_msgs/CompressedImage. Это исключает дополнительную
    потерю качества JPEG до эксперимента.
    """

    if not bag_path.is_file():
        raise FileNotFoundError(f"ROS bag не найден: {bag_path}")
    if route_start_offset_seconds < 0.0 or not math.isfinite(
        route_start_offset_seconds
    ):
        raise ValueError(
            "Смещение начала маршрута должно быть конечным и неотрицательным"
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    images_directory = output_directory / "images"
    image_rows: list[list[object]] = []
    rtk_rows: list[list[object]] = []
    rejected_rtk_count = 0
    timestamp_sources = {
        "image_header": 0,
        "image_bag": 0,
        "rtk_header": 0,
        "rtk_bag": 0,
    }
    image_extension: str | None = None
    previous_image_timestamp: int | None = None
    previous_rtk_timestamp: int | None = None

    with AnyReader([bag_path]) as reader:
        camera_connections = [
            connection
            for connection in reader.connections
            if connection.topic == camera_topic
        ]
        rtk_connections = [
            connection
            for connection in reader.connections
            if connection.topic == rtk_topic
        ]
        if not camera_connections:
            raise RuntimeError(f"В bag нет темы камеры {camera_topic}")
        if not rtk_connections:
            raise RuntimeError(f"В bag нет темы RTK {rtk_topic}")
        wrong_camera_types = {
            connection.msgtype
            for connection in camera_connections
            if connection.msgtype != CAMERA_MESSAGE_TYPE
        }
        wrong_rtk_types = {
            connection.msgtype
            for connection in rtk_connections
            if connection.msgtype != RTK_MESSAGE_TYPE
        }
        if wrong_camera_types:
            raise RuntimeError(f"Неожиданный тип камеры: {sorted(wrong_camera_types)}")
        if wrong_rtk_types:
            raise RuntimeError(f"Неожиданный тип RTK: {sorted(wrong_rtk_types)}")

        bag_start_timestamp = int(reader.start_time)
        export_start_timestamp = bag_start_timestamp + round(
            route_start_offset_seconds * 1_000_000_000
        )
        selected_connections = [*camera_connections, *rtk_connections]
        for connection, bag_timestamp, rawdata in reader.messages(
            connections=selected_connections,
            start=export_start_timestamp,
        ):
            message = reader.deserialize(rawdata, connection.msgtype)
            timestamp, timestamp_source = _message_timestamp_nanoseconds(
                message,
                bag_timestamp_nanoseconds=bag_timestamp,
            )

            if connection.topic == camera_topic:
                _require_strictly_increasing(
                    previous_image_timestamp,
                    timestamp,
                    stream_name="камеры",
                )
                payload = bytes(message.data)
                extension = _compressed_image_extension(payload)
                if image_extension is None:
                    image_extension = extension
                elif extension != image_extension:
                    raise RuntimeError(
                        "Формат сжатых кадров изменился внутри одного полёта"
                    )
                image_id = len(image_rows)
                image_path = images_directory / f"{image_id:06d}{extension}"
                _write_bytes_without_silent_overwrite(image_path, payload)
                image_rows.append([timestamp, image_id])
                previous_image_timestamp = timestamp
                timestamp_sources[f"image_{timestamp_source}"] += 1
                continue

            _require_strictly_increasing(
                previous_rtk_timestamp,
                timestamp,
                stream_name="RTK",
            )
            previous_rtk_timestamp = timestamp
            latitude = float(message.latitude)
            longitude = float(message.longitude)
            altitude = float(message.altitude)
            status = int(message.status.status)
            valid_coordinates = (
                math.isfinite(latitude)
                and math.isfinite(longitude)
                and math.isfinite(altitude)
                and -90.0 <= latitude <= 90.0
                and -180.0 <= longitude <= 180.0
            )
            if status < 0 or not valid_coordinates:
                rejected_rtk_count += 1
                continue
            rtk_rows.append([timestamp, latitude, longitude, altitude])
            timestamp_sources[f"rtk_{timestamp_source}"] += 1

    if not image_rows:
        raise RuntimeError("После заданного смещения не найдено ни одного кадра")
    if len(rtk_rows) < 2:
        raise RuntimeError(
            "Для интерполяции контрольной траектории нужны хотя бы две "
            "валидные RTK-точки"
        )
    assert image_extension is not None

    summary = MarsLvigExportSummary(
        bag_path=str(bag_path),
        output_directory=str(output_directory),
        camera_topic=camera_topic,
        rtk_topic=rtk_topic,
        bag_start_timestamp_nanoseconds=bag_start_timestamp,
        export_start_timestamp_nanoseconds=export_start_timestamp,
        route_start_offset_seconds=route_start_offset_seconds,
        image_count=len(image_rows),
        rtk_count=len(rtk_rows),
        rejected_rtk_count=rejected_rtk_count,
        image_header_timestamp_count=timestamp_sources["image_header"],
        image_bag_timestamp_fallback_count=timestamp_sources["image_bag"],
        rtk_header_timestamp_count=timestamp_sources["rtk_header"],
        rtk_bag_timestamp_fallback_count=timestamp_sources["rtk_bag"],
        image_extension=image_extension,
    )
    _write_text_atomically(
        output_directory / "image_ids.csv",
        _csv_text(["Timestamp", "Image Id"], image_rows),
    )
    _write_text_atomically(
        output_directory / "gps.csv",
        _csv_text(
            ["Timestamp", "latitude", "longitude", "altitude"],
            rtk_rows,
        ),
    )
    _write_text_atomically(
        output_directory / "export_summary.json",
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
    )
    return summary
