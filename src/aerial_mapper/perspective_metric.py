"""Метрическая калибровка перспективного размеченного Teach-кадра.

В отличие от ортофото, у перспективного изображения нет постоянного масштаба
«метров на пиксель». Если измеряемая поверхность плоская, четыре или более
размеченных соответствия «пиксель Teach ↔ координата на земле в метрах»
задают проективное преобразование этой плоскости. Затем рабочая цепочка имеет
вид: Repeat-пиксель → Teach-пиксель → координата земли в метрах.

Модуль не знает позу Repeat-камеры. Истинные параметры синтетического рендера
используются только снаружи, в оценщике эксперимента.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from aerial_mapper.measurement import (
    MappedFramePoints,
    MeasurementFailure,
    SegmentMeasurement,
    map_frame_points_to_reference,
)
from aerial_mapper.synthetic import Homography

Float64Points = NDArray[np.float64]


@dataclass(frozen=True)
class PerspectiveReferenceCalibration:
    """Проективная метрическая разметка плоскости земли Teach-кадра.

    Матрица переводит центры пикселей перспективного эталона в локальные
    декартовы координаты земли. Эти координаты должны быть выражены в метрах.
    Репроекционная ошибка нужна для аудита качества ручной разметки, но сама по
    себе не доказывает правильность масштаба: систематически неверные мировые
    координаты могут хорошо согласовываться между собой.
    """

    homography_reference_to_world_xy: Homography
    control_point_count: int
    control_reprojection_rmse_m: float
    control_reprojection_max_m: float


def _points(
    values: NDArray[np.floating] | list[list[float]],
    *,
    name: str,
    minimum_count: int,
) -> Float64Points:
    """Преобразует набор двумерных конечных точек в проверенный float64-массив."""

    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise MeasurementFailure(f"{name} должны иметь форму (N, 2)")
    if points.shape[0] < minimum_count:
        raise MeasurementFailure(
            f"{name}: требуется не менее {minimum_count} точек, "
            f"получено {points.shape[0]}"
        )
    if not np.all(np.isfinite(points)):
        raise MeasurementFailure(f"{name} содержат NaN или бесконечность")
    return points


def calibrate_perspective_reference(
    reference_points_px: NDArray[np.floating] | list[list[float]],
    world_points_xy_m: NDArray[np.floating] | list[list[float]],
) -> PerspectiveReferenceCalibration:
    """Оценивает Teach-пиксель → метр земли по известным контрольным точкам.

    Используется обычная линейная оценка без RANSAC: Teach-разметка считается
    доверенной и должна содержать именно соответствующие точки. Тихо
    отбрасывать ошибочные отметки здесь опасно; высокую ошибку калибровки нужно
    увидеть и исправить в самой разметке.
    """

    reference = _points(
        reference_points_px,
        name="Пиксельные точки калибровки",
        minimum_count=4,
    )
    world = _points(
        world_points_xy_m,
        name="Метрические точки калибровки",
        minimum_count=4,
    )
    if reference.shape != world.shape:
        raise MeasurementFailure(
            "Пиксельные и метрические точки калибровки должны быть попарными"
        )

    homography, _ = cv2.findHomography(reference, world, method=0)
    if homography is None or not np.all(np.isfinite(homography)):
        raise MeasurementFailure("Не удалось оценить метрическую калибровку Teach")
    if np.isclose(homography[2, 2], 0.0):
        raise MeasurementFailure("Получена вырожденная калибровочная гомография")
    homography = homography.astype(np.float64)
    homography /= homography[2, 2]

    predicted = cv2.perspectiveTransform(
        reference.reshape(1, -1, 2),
        homography,
    ).reshape(-1, 2)
    if not np.all(np.isfinite(predicted)):
        raise MeasurementFailure(
            "Калибровка переносит контрольные точки в бесконечность"
        )
    errors = np.linalg.norm(predicted - world, axis=1)
    return PerspectiveReferenceCalibration(
        homography_reference_to_world_xy=homography,
        control_point_count=int(reference.shape[0]),
        control_reprojection_rmse_m=float(np.sqrt(np.mean(errors**2))),
        control_reprojection_max_m=float(np.max(errors)),
    )


def map_frame_points_via_perspective_reference(
    frame_points_px: NDArray[np.floating] | list[list[float]],
    homography_frame_to_reference: Homography,
    *,
    calibration: PerspectiveReferenceCalibration,
    reference_width_pixels: int,
    reference_height_pixels: int,
) -> MappedFramePoints:
    """Переносит Repeat-точки через перспективный Teach в метры земли."""

    frame = _points(frame_points_px, name="Точки Repeat-кадра", minimum_count=1)
    if reference_width_pixels <= 1 or reference_height_pixels <= 1:
        raise MeasurementFailure(
            "Размеры Teach-кадра должны быть больше одного пикселя"
        )

    reference = map_frame_points_to_reference(frame, homography_frame_to_reference)
    outside = (
        (reference[:, 0] < 0.0)
        | (reference[:, 0] > reference_width_pixels - 1.0)
        | (reference[:, 1] < 0.0)
        | (reference[:, 1] > reference_height_pixels - 1.0)
    )
    if np.any(outside):
        raise MeasurementFailure(
            "Хотя бы одна измеряемая точка перенесена за границы Teach-кадра"
        )

    world = map_frame_points_to_reference(
        reference,
        calibration.homography_reference_to_world_xy,
    )
    return MappedFramePoints(
        frame_points_px=frame,
        reference_points_px=reference,
        world_points_m=world,
    )


def measure_ground_segment_via_perspective_reference(
    frame_points_px: NDArray[np.floating] | list[list[float]],
    homography_frame_to_reference: Homography,
    *,
    calibration: PerspectiveReferenceCalibration,
    reference_width_pixels: int,
    reference_height_pixels: int,
) -> SegmentMeasurement:
    """Измеряет подтверждённый отрезок земли на Repeat-кадре."""

    points = _points(frame_points_px, name="Концы отрезка", minimum_count=2)
    if points.shape[0] != 2:
        raise MeasurementFailure("Для отрезка требуется ровно две точки")
    mapped = map_frame_points_via_perspective_reference(
        points,
        homography_frame_to_reference,
        calibration=calibration,
        reference_width_pixels=reference_width_pixels,
        reference_height_pixels=reference_height_pixels,
    )
    length = float(np.linalg.norm(mapped.world_points_m[1] - mapped.world_points_m[0]))
    return SegmentMeasurement(mapped_points=mapped, length_meters=length)
