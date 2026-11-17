# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Измерение ROI и события относительно удалённого геодезического репера.

Карта HKairport является север-ориентированным растровым снимком без встроенной
геопривязки. Единственная известная точка карты и наземный масштаб позволяют
перевести разность пикселей в локальные смещения на восток и север. Дальнейшие
расчёты выполняются библиотекой PyProj на эллипсоиде WGS 84: градусы широты и
долготы нельзя складывать с метрами или использовать как декартовы координаты.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin

import numpy as np
from numpy.typing import NDArray
from pyproj import Geod

FloatPoints = NDArray[np.float64]
_WGS84_GEOD = Geod(ellps="WGS84")


class ReferenceEventMeasurementFailure(ValueError):
    """Ожидаемый отказ при неполной или противоречивой геопривязке."""


@dataclass(frozen=True)
class GeographicPoint:
    """Точка WGS 84 в градусах: сначала широта, затем долгота."""

    latitude_degrees: float
    longitude_degrees: float


@dataclass(frozen=True)
class NorthUpMapCalibration:
    """Одна координатная опора север-ориентированной растровой карты.

    ``anchor_map_point_px`` относится к исходным пикселям файла: начало слева
    сверху, X направлен вправо, Y — вниз. Размер изображения входит в
    калибровку, чтобы случайно обрезанная или масштабированная копия не выдала
    правдоподобные, но неверные координаты.
    """

    anchor: GeographicPoint
    anchor_map_point_px: tuple[float, float]
    map_width_pixels: int
    map_height_pixels: int


@dataclass(frozen=True)
class ReferenceRayMeasurement:
    """Координата конца луча, азимут и дальность от общего репера."""

    point: GeographicPoint
    geographic_azimuth_degrees: float
    distance_meters: float


@dataclass(frozen=True)
class TargetEventReferenceMeasurement:
    """Целевой и фактический лучи от репера.

    Целевой луч ``репер → ROI`` определяет относительный угол 0°. Отклонение
    события положительно по часовой стрелке, то есть вправо при взгляде от
    репера к ROI. ``distance_difference_meters`` положительно для события,
    произошедшего дальше цели, и отрицательно для события ближе цели.
    """

    target: ReferenceRayMeasurement
    event: ReferenceRayMeasurement
    event_azimuth_deviation_degrees: float
    distance_difference_meters: float


def map_pixel_to_geographic(
    point_px: tuple[float, float] | NDArray[np.floating],
    *,
    calibration: NorthUpMapCalibration,
    map_width_pixels: int,
    map_height_pixels: int,
    map_scale_meters_per_pixel: float,
) -> GeographicPoint:
    """Перевести пиксель север-ориентированной карты в координату WGS 84.

    Разность X означает смещение на восток, а противоположная разность Y — на
    север. Масштаб считается наземным и одинаковым по обеим осям. Для короткого
    смещения внутри карты PyProj решает прямую геодезическую задачу: по
    координате опоры, направлению и расстоянию находит конечную точку.
    """

    _validate_calibration(
        calibration,
        map_width_pixels=map_width_pixels,
        map_height_pixels=map_height_pixels,
    )
    scale = float(map_scale_meters_per_pixel)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ReferenceEventMeasurementFailure(
            "Масштаб карты должен быть конечным и положительным"
        )

    point = np.asarray(point_px, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ReferenceEventMeasurementFailure(
            "Точка карты должна быть конечной парой (x, y)"
        )
    if not (
        0.0 <= point[0] <= map_width_pixels - 1.0
        and 0.0 <= point[1] <= map_height_pixels - 1.0
    ):
        raise ReferenceEventMeasurementFailure("Точка лежит за границами карты")

    anchor_x, anchor_y = calibration.anchor_map_point_px
    east_meters = float(point[0] - anchor_x) * scale
    north_meters = -float(point[1] - anchor_y) * scale
    distance_meters = hypot(east_meters, north_meters)
    if distance_meters == 0.0:
        return calibration.anchor

    # Географический азимут отсчитывается от севера по часовой стрелке.
    azimuth_degrees = degrees(atan2(east_meters, north_meters))
    longitude, latitude, _back_azimuth = _WGS84_GEOD.fwd(
        calibration.anchor.longitude_degrees,
        calibration.anchor.latitude_degrees,
        azimuth_degrees,
        distance_meters,
    )
    return GeographicPoint(
        latitude_degrees=float(latitude),
        longitude_degrees=float(longitude),
    )


def map_direction_to_geographic_point(
    origin_map_point_px: tuple[float, float] | NDArray[np.floating],
    *,
    target: GeographicPoint,
    calibration: NorthUpMapCalibration,
    map_width_pixels: int,
    map_height_pixels: int,
    map_scale_meters_per_pixel: float,
) -> tuple[float, float]:
    """Вернуть единичное направление на географическую точку в осях карты.

    Результат ``(dx, dy)`` задан в пиксельных осях север-ориентированной карты:
    X растёт на восток, Y — на юг. Географическая цель может находиться далеко
    за границей растрового снимка. Мы не экстраполируем туда гомографию, а
    решаем обратную геодезическую задачу только для направления из локальной
    точки карты.
    """

    origin = map_pixel_to_geographic(
        origin_map_point_px,
        calibration=calibration,
        map_width_pixels=map_width_pixels,
        map_height_pixels=map_height_pixels,
        map_scale_meters_per_pixel=map_scale_meters_per_pixel,
    )
    _validate_geographic_point(target, name="Географическая цель направления")
    azimuth, _back_azimuth, distance = _WGS84_GEOD.inv(
        origin.longitude_degrees,
        origin.latitude_degrees,
        target.longitude_degrees,
        target.latitude_degrees,
    )
    if not np.isfinite(azimuth) or not np.isfinite(distance) or distance <= 0.0:
        raise ReferenceEventMeasurementFailure(
            "Направление не определено для совпадающих исходной и целевой точек"
        )

    azimuth_radians = radians(float(azimuth))
    return (sin(azimuth_radians), -cos(azimuth_radians))


def measure_target_and_event_from_reference(
    map_points_px: NDArray[np.floating] | list[tuple[float, float]],
    *,
    calibration: NorthUpMapCalibration,
    reference: GeographicPoint,
    map_width_pixels: int,
    map_height_pixels: int,
    map_scale_meters_per_pixel: float,
) -> TargetEventReferenceMeasurement:
    """Измерить ROI и событие относительно удалённого репера.

    Вход имеет форму ``(2, 2)``: первая строка — ROI, вторая — событие. Для
    каждой точки сначала восстанавливается широта и долгота, затем обратная
    геодезическая задача возвращает истинный азимут и расстояние от репера.
    Пользовательский угол события является разностью этих двух азимутов.
    """

    points = np.asarray(map_points_px, dtype=np.float64)
    if points.shape != (2, 2) or not np.all(np.isfinite(points)):
        raise ReferenceEventMeasurementFailure(
            "Для ROI и события нужны ровно две конечные точки карты"
        )
    _validate_geographic_point(reference, name="Геодезический репер")

    geographic_points = tuple(
        map_pixel_to_geographic(
            point,
            calibration=calibration,
            map_width_pixels=map_width_pixels,
            map_height_pixels=map_height_pixels,
            map_scale_meters_per_pixel=map_scale_meters_per_pixel,
        )
        for point in points
    )
    target = _measure_reference_ray(reference, geographic_points[0])
    event = _measure_reference_ray(reference, geographic_points[1])
    deviation = _normalize_signed_degrees(
        event.geographic_azimuth_degrees - target.geographic_azimuth_degrees
    )
    return TargetEventReferenceMeasurement(
        target=target,
        event=event,
        event_azimuth_deviation_degrees=deviation,
        distance_difference_meters=event.distance_meters - target.distance_meters,
    )


def _measure_reference_ray(
    reference: GeographicPoint,
    endpoint: GeographicPoint,
) -> ReferenceRayMeasurement:
    """Решить обратную геодезическую задачу для одного луча."""

    azimuth, _back_azimuth, distance = _WGS84_GEOD.inv(
        reference.longitude_degrees,
        reference.latitude_degrees,
        endpoint.longitude_degrees,
        endpoint.latitude_degrees,
    )
    if not np.isfinite(azimuth) or not np.isfinite(distance) or distance <= 0.0:
        raise ReferenceEventMeasurementFailure(
            "Направление от репера не определено для совпадающих точек"
        )
    return ReferenceRayMeasurement(
        point=endpoint,
        geographic_azimuth_degrees=float(azimuth % 360.0),
        distance_meters=float(distance),
    )


def _validate_calibration(
    calibration: NorthUpMapCalibration,
    *,
    map_width_pixels: int,
    map_height_pixels: int,
) -> None:
    """Проверить опору и запретить применение к другой версии карты."""

    if (
        map_width_pixels != calibration.map_width_pixels
        or map_height_pixels != calibration.map_height_pixels
    ):
        raise ReferenceEventMeasurementFailure(
            "Размер карты не совпадает с координатной калибровкой: "
            f"ожидался {calibration.map_width_pixels}×"
            f"{calibration.map_height_pixels} px, получен "
            f"{map_width_pixels}×{map_height_pixels} px"
        )
    _validate_geographic_point(calibration.anchor, name="Координатная опора карты")
    anchor = np.asarray(calibration.anchor_map_point_px, dtype=np.float64)
    if anchor.shape != (2,) or not np.all(np.isfinite(anchor)):
        raise ReferenceEventMeasurementFailure(
            "Пиксель координатной опоры должен быть конечной парой (x, y)"
        )
    if not (
        0.0 <= anchor[0] <= map_width_pixels - 1.0
        and 0.0 <= anchor[1] <= map_height_pixels - 1.0
    ):
        raise ReferenceEventMeasurementFailure(
            "Координатная опора лежит за границами карты"
        )


def _validate_geographic_point(point: GeographicPoint, *, name: str) -> None:
    """Проверить порядок и допустимые диапазоны широты и долготы."""

    latitude = float(point.latitude_degrees)
    longitude = float(point.longitude_degrees)
    if not np.isfinite(latitude) or not np.isfinite(longitude):
        raise ReferenceEventMeasurementFailure(f"{name} содержит NaN или inf")
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise ReferenceEventMeasurementFailure(
            f"{name} содержит широту или долготу вне допустимого диапазона"
        )


def _normalize_signed_degrees(angle_degrees: float) -> float:
    """Привести разность углов к однозначному диапазону [-180, 180)."""

    normalized = (float(angle_degrees) + 180.0) % 360.0 - 180.0
    return 0.0 if abs(normalized) < 1e-12 else normalized
