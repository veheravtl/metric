"""Чтение телеметрии CLOUD для независимой оценки Teach/Repeat-пар.

Рабочий алгоритм привязки не должен видеть координаты из этого модуля.
Телеметрия используется только экспериментальным кодом: выбрать ожидаемо
близкую и заведомо далёкую пару, а после обработки проверить результат.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pyproj import Transformer


@dataclass(frozen=True)
class PositionedImages:
    """Кадры, чьи координаты интерполированы на момент экспозиции.

    ``xy_meters`` имеет форму ``(N, 2)`` и хранит восточную и северную
    координаты в метрах в указанной проекции. Высота CLOUD является высотой
    над эллипсоидом WGS 84, а не высотой над землёй, поэтому для выбора пары
    в этом опыте она намеренно не используется.
    """

    image_ids: NDArray[np.int64]
    timestamps_nanoseconds: NDArray[np.float64]
    xy_meters: NDArray[np.float64]
    altitude_ellipsoid_meters: NDArray[np.float64]
    projected_crs: str


@dataclass(frozen=True)
class TeachRepeatPair:
    """Одна Repeat-проверка и два Teach-контроля с известной дистанцией."""

    repeat_image_id: int
    positive_teach_image_id: int
    positive_distance_meters: float
    negative_teach_image_id: int
    negative_distance_meters: float


def _read_numeric_csv(path: Path, *, expected_columns: int) -> NDArray[np.float64]:
    """Читает числовой CSV CLOUD и строго проверяет его прямоугольность."""

    values = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64)
    values = np.atleast_2d(values)
    if values.shape[1] != expected_columns:
        raise ValueError(
            f"{path} должен содержать {expected_columns} столбцов, "
            f"получено {values.shape[1]}"
        )
    if values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{path} пуст или содержит нечисловые значения")
    return values


def load_positioned_images(
    flight_directory: Path,
    *,
    projected_crs: str = "EPSG:32617",
) -> PositionedImages:
    """Сопоставляет кадры CLOUD с интерполированными GPS-координатами.

    Временные метки имеют порядок ``1e18`` наносекунд. Перед интерполяцией
    вычитается общий момент времени: это сохраняет малые интервалы между
    измерениями при вычислениях с ``float64``.

    Кадры вне временного диапазона GPS отбрасываются. Экстраполяция создала бы
    правдоподобную, но ничем не подтверждённую контрольную координату.
    """

    image_rows = _read_numeric_csv(
        flight_directory / "image_ids.csv", expected_columns=2
    )
    gps_rows = _read_numeric_csv(flight_directory / "gps.csv", expected_columns=4)

    image_timestamps = image_rows[:, 0]
    image_ids_float = image_rows[:, 1]
    gps_timestamps = gps_rows[:, 0]
    if np.any(np.diff(image_timestamps) <= 0) or np.any(np.diff(gps_timestamps) <= 0):
        raise ValueError("Временные метки изображений и GPS должны строго возрастать")
    if not np.all(image_ids_float == np.floor(image_ids_float)):
        raise ValueError("Image Id должен быть целым числом")

    valid_time_mask = (image_timestamps >= gps_timestamps[0]) & (
        image_timestamps <= gps_timestamps[-1]
    )
    valid_timestamps = image_timestamps[valid_time_mask]
    valid_image_ids = image_ids_float[valid_time_mask].astype(np.int64)
    if valid_timestamps.size == 0:
        raise ValueError("Ни один кадр не попадает во временной диапазон GPS")

    time_origin = float(gps_timestamps[0])
    gps_seconds = (gps_timestamps - time_origin) / 1e9
    image_seconds = (valid_timestamps - time_origin) / 1e9
    latitudes = np.interp(image_seconds, gps_seconds, gps_rows[:, 1])
    longitudes = np.interp(image_seconds, gps_seconds, gps_rows[:, 2])
    altitudes = np.interp(image_seconds, gps_seconds, gps_rows[:, 3])

    # always_xy=True закрепляет порядок longitude, latitude. Без него порядок
    # осей зависит от описания системы координат и легко получить тихую ошибку.
    transformer = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    eastings, northings = transformer.transform(longitudes, latitudes)
    xy_meters = np.column_stack((eastings, northings)).astype(np.float64)

    return PositionedImages(
        image_ids=valid_image_ids,
        timestamps_nanoseconds=valid_timestamps.astype(np.float64),
        xy_meters=xy_meters,
        altitude_ellipsoid_meters=altitudes.astype(np.float64),
        projected_crs=projected_crs,
    )


def select_teach_repeat_pairs(
    teach: PositionedImages,
    repeat: PositionedImages,
    *,
    query_count: int,
) -> list[TeachRepeatPair]:
    """Выбирает разнесённые Repeat-кадры и ближайший/дальний Teach-контроль.

    Ближайший кадр служит приближённой положительной разметкой, а самый дальний
    — лёгким отрицательным контролем. Обычный GPS не доказывает одинаковое поле
    зрения, поэтому такие пары подходят для разведочного smoke test, но не для
    финальной оценки точности.
    """

    if query_count <= 0:
        raise ValueError("Число запросов должно быть положительным")
    if teach.xy_meters.shape[0] == 0 or repeat.xy_meters.shape[0] == 0:
        raise ValueError("Teach и Repeat должны содержать хотя бы по одному кадру")
    if teach.projected_crs != repeat.projected_crs:
        raise ValueError("Teach и Repeat должны быть в одной системе координат")

    actual_count = min(query_count, repeat.image_ids.size)
    repeat_indices = np.linspace(
        0, repeat.image_ids.size - 1, actual_count, dtype=np.int64
    )
    pairs: list[TeachRepeatPair] = []
    for repeat_index in repeat_indices:
        distances = np.linalg.norm(
            teach.xy_meters - repeat.xy_meters[repeat_index], axis=1
        )
        positive_index = int(np.argmin(distances))
        negative_index = int(np.argmax(distances))
        pairs.append(
            TeachRepeatPair(
                repeat_image_id=int(repeat.image_ids[repeat_index]),
                positive_teach_image_id=int(teach.image_ids[positive_index]),
                positive_distance_meters=float(distances[positive_index]),
                negative_teach_image_id=int(teach.image_ids[negative_index]),
                negative_distance_meters=float(distances[negative_index]),
            )
        )
    return pairs


def select_spaced_image_indices(
    positioned_images: PositionedImages,
    *,
    minimum_distance_meters: float,
) -> NDArray[np.int64]:
    """Выбирает ключевые кадры по перемещению от последнего выбранного кадра.

    Первый и последний кадры сохраняются всегда. Промежуточный кадр добавляется,
    когда плоское расстояние от последнего выбранного кадра достигает заданного
    порога. Это удаляет плотные видеодубли, но сохраняет порядок маршрута.

    Возвращаются индексы строк ``PositionedImages``, а не ``Image Id``. Такое
    соглашение позволяет тем же массивом выбирать идентификаторы, координаты и
    временные метки без повторного поиска.
    """

    if minimum_distance_meters <= 0.0:
        raise ValueError("Минимальное расстояние должно быть положительным")
    image_count = positioned_images.image_ids.size
    if image_count == 0:
        raise ValueError("Нельзя выбрать ключевые кадры из пустого полёта")

    selected_indices = [0]
    last_selected_index = 0
    for candidate_index in range(1, image_count):
        distance = np.linalg.norm(
            positioned_images.xy_meters[candidate_index]
            - positioned_images.xy_meters[last_selected_index]
        )
        if distance >= minimum_distance_meters:
            selected_indices.append(candidate_index)
            last_selected_index = candidate_index

    final_index = image_count - 1
    if selected_indices[-1] != final_index:
        selected_indices.append(final_index)
    return np.asarray(selected_indices, dtype=np.int64)
