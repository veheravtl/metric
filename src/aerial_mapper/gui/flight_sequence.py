"""Чтение извлечённого пролёта MARS-LVIG как последовательности кадров."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FrameTelemetry:
    """Метаданные одного кадра, синхронизированные по времени.

    Координаты равны ``None``, если время кадра лежит вне диапазона GPS. В этом
    случае мы намеренно не экстраполируем положение: выдуманная координата
    выглядела бы правдоподобно, но не подтверждалась бы измерением.
    """

    image_id: int
    timestamp_nanoseconds: int
    elapsed_seconds: float
    latitude_degrees: float | None
    longitude_degrees: float | None
    altitude_ellipsoid_meters: float | None

    @property
    def has_position(self) -> bool:
        """Есть ли у кадра интерполированная позиция внутри диапазона GPS."""

        return self.latitude_degrees is not None


@dataclass(frozen=True)
class MarsLvigFlight:
    """Проверенный индекс JPEG-кадров и сопутствующей телеметрии пролёта.

    Камера записана примерно с частотой 10 кадров/с, а GPS — примерно 5
    измерений/с. Положение каждого кадра получается линейной интерполяцией между
    соседними GPS-измерениями. Для короткого аэродромного маршрута это пригодно
    как навигационная подсказка, но не является геометрической привязкой кадра.
    """

    directory: Path
    image_paths: tuple[Path, ...]
    image_ids: NDArray[np.int64]
    timestamps_nanoseconds: NDArray[np.int64]
    latitudes_degrees: NDArray[np.float64]
    longitudes_degrees: NDArray[np.float64]
    altitudes_ellipsoid_meters: NDArray[np.float64]
    nominal_fps: float

    @classmethod
    def load(cls, directory: str | Path) -> MarsLvigFlight:
        """Прочитать каталог ``images + image_ids.csv + gps.csv``.

        Все кадры проверяются при открытии. Благодаря этому повреждённый экспорт
        обнаружится сразу, а не посередине воспроизведения.
        """

        flight_directory = Path(directory).expanduser().resolve()
        if not flight_directory.is_dir():
            raise ValueError(f"Каталог пролёта не найден: {flight_directory}")

        images_directory = flight_directory / "images"
        if not images_directory.is_dir():
            raise ValueError(f"Не найден каталог кадров: {images_directory}")

        image_timestamps, image_ids = _read_image_index(
            flight_directory / "image_ids.csv"
        )
        gps_timestamps, latitudes, longitudes, altitudes = _read_gps(
            flight_directory / "gps.csv"
        )

        image_paths = tuple(
            _resolve_image_path(images_directory, int(image_id))
            for image_id in image_ids
        )
        frame_seconds, gps_seconds = _relative_seconds(image_timestamps, gps_timestamps)
        valid_position_mask = (image_timestamps >= gps_timestamps[0]) & (
            image_timestamps <= gps_timestamps[-1]
        )

        frame_latitudes = np.full(image_ids.shape, np.nan, dtype=np.float64)
        frame_longitudes = np.full(image_ids.shape, np.nan, dtype=np.float64)
        frame_altitudes = np.full(image_ids.shape, np.nan, dtype=np.float64)
        frame_latitudes[valid_position_mask] = np.interp(
            frame_seconds[valid_position_mask], gps_seconds, latitudes
        )
        frame_longitudes[valid_position_mask] = np.interp(
            frame_seconds[valid_position_mask], gps_seconds, longitudes
        )
        frame_altitudes[valid_position_mask] = np.interp(
            frame_seconds[valid_position_mask], gps_seconds, altitudes
        )

        median_interval_nanoseconds = float(
            np.median(np.diff(image_timestamps).astype(np.float64))
        )
        nominal_fps = 1e9 / median_interval_nanoseconds

        return cls(
            directory=flight_directory,
            image_paths=image_paths,
            image_ids=image_ids,
            timestamps_nanoseconds=image_timestamps,
            latitudes_degrees=frame_latitudes,
            longitudes_degrees=frame_longitudes,
            altitudes_ellipsoid_meters=frame_altitudes,
            nominal_fps=nominal_fps,
        )

    @property
    def frame_count(self) -> int:
        """Количество кадров в проверенном индексе."""

        return len(self.image_paths)

    def read_frame_bgr(self, frame_index: int) -> NDArray[np.uint8]:
        """Декодировать указанный JPEG в формате каналов OpenCV BGR."""

        if not 0 <= frame_index < self.frame_count:
            raise IndexError(f"Кадр вне диапазона: {frame_index}")
        frame = cv2.imread(str(self.image_paths[frame_index]), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise OSError(f"Не удалось прочитать кадр: {self.image_paths[frame_index]}")
        return frame

    def telemetry(self, frame_index: int) -> FrameTelemetry:
        """Вернуть время и интерполированную GPS-позицию указанного кадра."""

        if not 0 <= frame_index < self.frame_count:
            raise IndexError(f"Кадр вне диапазона: {frame_index}")

        latitude = float(self.latitudes_degrees[frame_index])
        longitude = float(self.longitudes_degrees[frame_index])
        altitude = float(self.altitudes_ellipsoid_meters[frame_index])
        has_position = np.isfinite(latitude)
        elapsed_seconds = (
            float(
                self.timestamps_nanoseconds[frame_index]
                - self.timestamps_nanoseconds[0]
            )
            / 1e9
        )
        return FrameTelemetry(
            image_id=int(self.image_ids[frame_index]),
            timestamp_nanoseconds=int(self.timestamps_nanoseconds[frame_index]),
            elapsed_seconds=elapsed_seconds,
            latitude_degrees=latitude if has_position else None,
            longitude_degrees=longitude if has_position else None,
            altitude_ellipsoid_meters=altitude if has_position else None,
        )

    def interval_after_milliseconds(self, frame_index: int) -> int:
        """Вернуть записанный интервал до следующего кадра в миллисекундах."""

        if frame_index < 0 or frame_index >= self.frame_count - 1:
            return max(1, round(1000.0 / self.nominal_fps))
        interval_nanoseconds = int(
            self.timestamps_nanoseconds[frame_index + 1]
            - self.timestamps_nanoseconds[frame_index]
        )
        return max(1, round(interval_nanoseconds / 1e6))


def _read_image_index(path: Path) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    rows = _read_csv_rows(path, expected_columns=("Timestamp", "Image Id"))
    try:
        timestamps = np.asarray([int(row["Timestamp"]) for row in rows], dtype=np.int64)
        image_ids = np.asarray([int(row["Image Id"]) for row in rows], dtype=np.int64)
    except ValueError as error:
        raise ValueError(f"{path} содержит нецелое время или Image Id") from error

    _validate_increasing(timestamps, path, "временные метки кадров")
    if np.any(image_ids < 0) or np.unique(image_ids).size != image_ids.size:
        raise ValueError(f"{path}: Image Id должны быть уникальными и неотрицательными")
    return timestamps, image_ids


def _read_gps(
    path: Path,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    rows = _read_csv_rows(
        path, expected_columns=("Timestamp", "latitude", "longitude", "altitude")
    )
    try:
        timestamps = np.asarray([int(row["Timestamp"]) for row in rows], dtype=np.int64)
        values = np.asarray(
            [
                (
                    float(row["latitude"]),
                    float(row["longitude"]),
                    float(row["altitude"]),
                )
                for row in rows
            ],
            dtype=np.float64,
        )
    except ValueError as error:
        raise ValueError(f"{path} содержит некорректное число") from error

    _validate_increasing(timestamps, path, "временные метки GPS")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} содержит бесконечные или отсутствующие координаты")
    return timestamps, values[:, 0], values[:, 1], values[:, 2]


def _read_csv_rows(
    path: Path, *, expected_columns: tuple[str, ...]
) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Не найден файл метаданных: {path}")
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise ValueError(
                f"{path}: ожидались столбцы {expected_columns}, "
                f"получены {reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Файл метаданных пуст: {path}")
    return rows


def _validate_increasing(values: NDArray[np.int64], path: Path, label: str) -> None:
    if values.size < 2 or np.any(np.diff(values) <= 0):
        raise ValueError(f"{path}: {label} должны строго возрастать")


def _resolve_image_path(images_directory: Path, image_id: int) -> Path:
    for extension in (".jpg", ".jpeg", ".png"):
        candidate = images_directory / f"{image_id:06d}{extension}"
        if candidate.is_file():
            return candidate
    raise ValueError(f"Не найден кадр Image Id {image_id} в {images_directory}")


def _relative_seconds(
    image_timestamps: NDArray[np.int64], gps_timestamps: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    origin = min(int(image_timestamps[0]), int(gps_timestamps[0]))
    return (
        (image_timestamps - origin).astype(np.float64) / 1e9,
        (gps_timestamps - origin).astype(np.float64) / 1e9,
    )
