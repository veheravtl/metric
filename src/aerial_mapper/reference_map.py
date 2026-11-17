# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Калибровка обзорной карты и ограниченный выбор удалённого репера.

Обзорный снимок не является географической информационной системой и не
содержит встроенной привязки. Две показанные пользователем точки задают для
него локальное преобразование подобия: единый масштаб и поворот между
пикселями изображения и осями «восток — север» в метрах.

На круге радиусом 6 км плоское преобразование используется только для
интерфейса. Итоговые координаты, расстояние и азимут всегда вычисляются на
эллипсоиде WGS 84 библиотекой PyProj.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin

import numpy as np
from pyproj import Geod

from aerial_mapper.reference_event_measurement import GeographicPoint

_WGS84_GEOD = Geod(ellps="WGS84")

REFERENCE_MAP_WIDTH_PIXELS = 888
REFERENCE_MAP_HEIGHT_PIXELS = 795
MAXIMUM_REFERENCE_DISTANCE_METERS = 6_000.0

# Центры двух маркеров считаны с предоставленных пользователем снимков
# 888×795 px. Первый маркер — центр рабочего района HKairport, второй —
# независимая точка масштаба и ориентации. Ошибка чтения центра порядка одного
# пикселя соответствует примерно 17,6 м и остаётся ограничением этой схемы.
AIRPORT_CENTER = GeographicPoint(22.415641, 114.043616)
AIRPORT_CENTER_PIXEL = (443.5, 380.5)
SCALE_CONTROL_POINT = GeographicPoint(22.451025, 113.996237)
SCALE_CONTROL_PIXEL = (167.5, 157.5)


class ReferenceMapFailure(ValueError):
    """Обзорная карта или выбранный репер не прошли проверку."""


@dataclass(frozen=True)
class ReferenceMapCalibration:
    """Две географические опоры и соответствующие им пиксели снимка."""

    center: GeographicPoint
    center_pixel: tuple[float, float]
    scale_control: GeographicPoint
    scale_control_pixel: tuple[float, float]
    image_width_pixels: int
    image_height_pixels: int
    maximum_distance_meters: float = MAXIMUM_REFERENCE_DISTANCE_METERS


@dataclass(frozen=True)
class ReferencePlacement:
    """Проверенное положение репера внутри разрешённого круга."""

    point: GeographicPoint
    map_pixel: tuple[float, float]
    distance_from_center_meters: float
    azimuth_from_center_degrees: float


DEFAULT_REFERENCE_MAP_CALIBRATION = ReferenceMapCalibration(
    center=AIRPORT_CENTER,
    center_pixel=AIRPORT_CENTER_PIXEL,
    scale_control=SCALE_CONTROL_POINT,
    scale_control_pixel=SCALE_CONTROL_PIXEL,
    image_width_pixels=REFERENCE_MAP_WIDTH_PIXELS,
    image_height_pixels=REFERENCE_MAP_HEIGHT_PIXELS,
)


def reference_map_meters_per_pixel(
    calibration: ReferenceMapCalibration = DEFAULT_REFERENCE_MAP_CALIBRATION,
) -> float:
    """Вернуть единый наземный масштаб обзорного снимка."""

    east_per_pixel, north_per_pixel = _pixel_to_meter_coefficients(calibration)
    return hypot(east_per_pixel, north_per_pixel)


def place_reference_from_pixel(
    pixel: tuple[float, float],
    *,
    calibration: ReferenceMapCalibration = DEFAULT_REFERENCE_MAP_CALIBRATION,
) -> ReferencePlacement:
    """Преобразовать клик в репер и запретить точки дальше 6 км.

    Пиксель проверяется в исходной геометрии изображения, а не в координатах
    виджета. Поэтому масштабирование окна не изменяет географический результат.
    """

    _validate_calibration(calibration)
    x, y = _finite_pixel(pixel)
    if not (
        0.0 <= x <= calibration.image_width_pixels - 1.0
        and 0.0 <= y <= calibration.image_height_pixels - 1.0
    ):
        raise ReferenceMapFailure("Репер лежит за границами обзорной карты")

    east_meters, north_meters = _pixel_offset_to_local_meters(
        x - calibration.center_pixel[0],
        y - calibration.center_pixel[1],
        calibration,
    )
    distance_meters = hypot(east_meters, north_meters)
    if distance_meters > calibration.maximum_distance_meters + 1e-6:
        raise ReferenceMapFailure(
            "Репер должен находиться не дальше 6 км от центра аэропорта"
        )
    azimuth_degrees = (
        degrees(atan2(east_meters, north_meters)) % 360.0
        if distance_meters > 0.0
        else 0.0
    )
    longitude, latitude, _back_azimuth = _WGS84_GEOD.fwd(
        calibration.center.longitude_degrees,
        calibration.center.latitude_degrees,
        azimuth_degrees,
        distance_meters,
    )
    point = (
        calibration.center
        if distance_meters == 0.0
        else GeographicPoint(float(latitude), float(longitude))
    )
    return ReferencePlacement(
        point=point,
        map_pixel=(x, y),
        distance_from_center_meters=distance_meters,
        azimuth_from_center_degrees=azimuth_degrees,
    )


def reference_map_pixel_to_geographic(
    pixel: tuple[float, float],
    *,
    calibration: ReferenceMapCalibration = DEFAULT_REFERENCE_MAP_CALIBRATION,
) -> GeographicPoint:
    """Перевести любой пиксель снимка в WGS 84 без ограничения репера.

    Калибровочная точка масштаба лежит примерно в 6,256 км от центра, поэтому
    базовое преобразование обязано работать за пределами разрешённого круга.
    Ограничение 6 км применяется только функцией установки репера.
    """

    _validate_calibration(calibration)
    x, y = _finite_pixel(pixel)
    if not (
        0.0 <= x <= calibration.image_width_pixels - 1.0
        and 0.0 <= y <= calibration.image_height_pixels - 1.0
    ):
        raise ReferenceMapFailure("Точка лежит за границами обзорной карты")
    east_meters, north_meters = _pixel_offset_to_local_meters(
        x - calibration.center_pixel[0],
        y - calibration.center_pixel[1],
        calibration,
    )
    distance_meters = hypot(east_meters, north_meters)
    if distance_meters == 0.0:
        return calibration.center
    azimuth_degrees = degrees(atan2(east_meters, north_meters))
    longitude, latitude, _back_azimuth = _WGS84_GEOD.fwd(
        calibration.center.longitude_degrees,
        calibration.center.latitude_degrees,
        azimuth_degrees,
        distance_meters,
    )
    return GeographicPoint(float(latitude), float(longitude))


def place_reference_from_geographic(
    point: GeographicPoint,
    *,
    calibration: ReferenceMapCalibration = DEFAULT_REFERENCE_MAP_CALIBRATION,
) -> ReferencePlacement:
    """Восстановить сохранённый репер и проверить тот же предел 6 км."""

    _validate_calibration(calibration)
    latitude = float(point.latitude_degrees)
    longitude = float(point.longitude_degrees)
    if not np.isfinite(latitude) or not np.isfinite(longitude):
        raise ReferenceMapFailure("Координаты репера должны быть конечными")
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise ReferenceMapFailure("Координаты репера находятся вне WGS 84")

    azimuth, _back_azimuth, distance = _WGS84_GEOD.inv(
        calibration.center.longitude_degrees,
        calibration.center.latitude_degrees,
        longitude,
        latitude,
    )
    if distance > calibration.maximum_distance_meters + 1e-6:
        raise ReferenceMapFailure(
            "Сохранённый репер находится дальше 6 км от центра аэропорта"
        )
    azimuth_radians = radians(float(azimuth))
    east_meters = float(distance) * sin(azimuth_radians)
    north_meters = float(distance) * cos(azimuth_radians)
    dx, dy = _local_meters_to_pixel_offset(
        east_meters,
        north_meters,
        calibration,
    )
    pixel = (
        calibration.center_pixel[0] + dx,
        calibration.center_pixel[1] + dy,
    )
    # Повторный путь через пиксель гарантирует, что сохранённая координата
    # относится именно к видимой и допустимой части текущей версии снимка.
    placement = place_reference_from_pixel(pixel, calibration=calibration)
    return ReferencePlacement(
        point=point,
        map_pixel=placement.map_pixel,
        distance_from_center_meters=float(distance),
        azimuth_from_center_degrees=float(azimuth % 360.0),
    )


def _pixel_to_meter_coefficients(
    calibration: ReferenceMapCalibration,
) -> tuple[float, float]:
    """Вернуть метрические оси одного пикселя как комплексный коэффициент.

    Пара ``(east_per_pixel, north_per_pixel)`` означает смещение в метрах при
    движении на один пиксель вправо. Ось одного пикселя вниз получается тем же
    поворотом на 90°, что сохраняет одинаковый масштаб по X и Y.
    """

    azimuth, _back_azimuth, distance = _WGS84_GEOD.inv(
        calibration.center.longitude_degrees,
        calibration.center.latitude_degrees,
        calibration.scale_control.longitude_degrees,
        calibration.scale_control.latitude_degrees,
    )
    azimuth_radians = radians(float(azimuth))
    control_east = float(distance) * sin(azimuth_radians)
    control_north = float(distance) * cos(azimuth_radians)
    dx = calibration.scale_control_pixel[0] - calibration.center_pixel[0]
    dy = calibration.scale_control_pixel[1] - calibration.center_pixel[1]
    pixel_norm_squared = dx * dx + dy * dy
    if pixel_norm_squared <= 0.0:
        raise ReferenceMapFailure("Контрольные пиксели карты совпадают")

    # Ищем коэффициент комплексного деления (east + i*north)/(dx + i*dy).
    east_per_pixel = (control_east * dx + control_north * dy) / pixel_norm_squared
    north_per_pixel = (control_north * dx - control_east * dy) / pixel_norm_squared
    return east_per_pixel, north_per_pixel


def _pixel_offset_to_local_meters(
    dx: float,
    dy: float,
    calibration: ReferenceMapCalibration,
) -> tuple[float, float]:
    east_per_pixel, north_per_pixel = _pixel_to_meter_coefficients(calibration)
    east = east_per_pixel * dx - north_per_pixel * dy
    north = north_per_pixel * dx + east_per_pixel * dy
    return east, north


def _local_meters_to_pixel_offset(
    east: float,
    north: float,
    calibration: ReferenceMapCalibration,
) -> tuple[float, float]:
    east_per_pixel, north_per_pixel = _pixel_to_meter_coefficients(calibration)
    denominator = east_per_pixel**2 + north_per_pixel**2
    if denominator <= 0.0:
        raise ReferenceMapFailure("Масштаб обзорной карты равен нулю")
    dx = (east_per_pixel * east + north_per_pixel * north) / denominator
    dy = (-north_per_pixel * east + east_per_pixel * north) / denominator
    return dx, dy


def _finite_pixel(pixel: tuple[float, float]) -> tuple[float, float]:
    values = np.asarray(pixel, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ReferenceMapFailure("Пиксель репера должен быть конечной парой (x, y)")
    return float(values[0]), float(values[1])


def _validate_calibration(calibration: ReferenceMapCalibration) -> None:
    if calibration.image_width_pixels <= 0 or calibration.image_height_pixels <= 0:
        raise ReferenceMapFailure("Размер обзорной карты должен быть положительным")
    if (
        not np.isfinite(calibration.maximum_distance_meters)
        or calibration.maximum_distance_meters <= 0.0
    ):
        raise ReferenceMapFailure("Допустимый радиус должен быть положительным")
    for name, pixel in (
        ("центр", calibration.center_pixel),
        ("масштабная точка", calibration.scale_control_pixel),
    ):
        x, y = _finite_pixel(pixel)
        if not (
            0.0 <= x <= calibration.image_width_pixels - 1.0
            and 0.0 <= y <= calibration.image_height_pixels - 1.0
        ):
            raise ReferenceMapFailure(f"Контрольный пиксель «{name}» вне карты")
