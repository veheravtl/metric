"""Перенос точек кадра в метрическую систему GeoTIFF и измерение геометрии.

Этот модуль относится к рабочему пути будущей системы. Он получает только
оценённую гомографию, геопривязку эталона и точки пользователя. Истинная
матрица синтетического генератора здесь принципиально отсутствует.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pyproj import CRS
from rasterio import Affine
from shapely import Polygon, is_valid_reason

from aerial_mapper.synthetic import Homography

Float64Points = NDArray[np.float64]


class MeasurementFailure(ValueError):
    """Ожидаемый отказ, когда координаты нельзя безопасно измерить."""


@dataclass(frozen=True)
class MappedFramePoints:
    """Одни точки, выраженные в трёх согласованных системах координат."""

    frame_points_px: Float64Points
    reference_points_px: Float64Points
    world_points_m: Float64Points


@dataclass(frozen=True)
class SegmentMeasurement:
    """Два перенесённых конца и длина отрезка в метрах CRS."""

    mapped_points: MappedFramePoints
    length_meters: float


@dataclass(frozen=True)
class PolygonMeasurement:
    """Перенесённые вершины и площадь валидного полигона в квадратных метрах."""

    mapped_points: MappedFramePoints
    area_square_meters: float


def _validate_points(
    points_px: NDArray[np.floating]
    | list[list[float]]
    | tuple[tuple[float, float], ...],
    *,
    name: str,
    minimum_count: int,
) -> Float64Points:
    """Нормализует входные точки и запрещает неоднозначные формы массивов."""

    points = np.asarray(points_px, dtype=np.float64)
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


def _validate_metric_projected_crs(reference_crs: object) -> CRS:
    """Убеждается, что евклидова геометрия действительно вернёт метры."""

    try:
        crs = CRS.from_user_input(reference_crs)
    except Exception as error:  # noqa: BLE001 — преобразуем ошибку внешней библиотеки
        raise MeasurementFailure("Не удалось распознать CRS эталона") from error

    if not crs.is_projected:
        raise MeasurementFailure(
            "Для прямого измерения нужна проецированная CRS, а не градусы"
        )

    axis_info = crs.axis_info
    if len(axis_info) < 2 or any(
        not np.isclose(axis.unit_conversion_factor, 1.0) for axis in axis_info[:2]
    ):
        units = [axis.unit_name for axis in axis_info[:2]]
        raise MeasurementFailure(
            "Оси CRS должны быть непосредственно выражены в метрах; "
            f"получены единицы: {units}"
        )
    return crs


def map_frame_points_to_reference(
    frame_points_px: NDArray[np.floating]
    | list[list[float]]
    | tuple[tuple[float, float], ...],
    homography_frame_to_reference: Homography,
) -> Float64Points:
    """Переносит точки кадра в субпиксельные координаты эталона.

    Гомография определена только с точностью до общего множителя. Поэтому перед
    вычислением матрица нормируется по своему максимальному элементу. Это также
    позволяет осмысленно проверять знаменатель перспективного деления около
    нуля независимо от масштаба записи матрицы.
    """

    points = _validate_points(
        frame_points_px,
        name="Точки кадра",
        minimum_count=1,
    )
    homography = np.asarray(homography_frame_to_reference, dtype=np.float64)
    if homography.shape != (3, 3):
        raise MeasurementFailure("Гомография должна иметь форму (3, 3)")
    if not np.all(np.isfinite(homography)):
        raise MeasurementFailure("Гомография содержит NaN или бесконечность")

    normalization_scale = float(np.max(np.abs(homography)))
    if normalization_scale == 0.0:
        raise MeasurementFailure("Нулевая матрица не является гомографией")
    normalized_homography = homography / normalization_scale

    homogeneous_points = np.column_stack(
        (points, np.ones(points.shape[0], dtype=np.float64))
    )
    projected_homogeneous = homogeneous_points @ normalized_homography.T
    denominators = projected_homogeneous[:, 2]
    numerical_tolerance = np.finfo(np.float64).eps * 100.0
    if np.any(np.abs(denominators) <= numerical_tolerance):
        raise MeasurementFailure(
            "Гомография переносит хотя бы одну точку в бесконечность"
        )

    reference_points = projected_homogeneous[:, :2] / denominators[:, None]
    if not np.all(np.isfinite(reference_points)):
        raise MeasurementFailure("После перспективного деления получены NaN или inf")
    return reference_points


def map_reference_points_to_world(
    reference_points_px: NDArray[np.floating]
    | list[list[float]]
    | tuple[tuple[float, float], ...],
    *,
    reference_transform: Affine,
    reference_crs: object,
) -> Float64Points:
    """Переводит координаты центров пикселей эталона в метры его CRS.

    В OpenCV целочисленная координата соответствует центру отсчёта изображения.
    Affine GeoTIFF переводит координаты углов пикселей. Поэтому перед affine-
    преобразованием к обеим координатам добавляется 0,5 пикселя. Для разности
    координат постоянный сдвиг сократился бы, но для абсолютного положения он
    необходим.
    """

    _validate_metric_projected_crs(reference_crs)
    points = _validate_points(
        reference_points_px,
        name="Точки эталона",
        minimum_count=1,
    )
    transform_coefficients = np.asarray(
        [
            reference_transform.a,
            reference_transform.b,
            reference_transform.c,
            reference_transform.d,
            reference_transform.e,
            reference_transform.f,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(transform_coefficients)):
        raise MeasurementFailure("Affine-преобразование GeoTIFF содержит NaN или inf")

    pixel_corner_coordinates = points + 0.5
    columns = pixel_corner_coordinates[:, 0]
    rows = pixel_corner_coordinates[:, 1]
    eastings = (
        reference_transform.a * columns
        + reference_transform.b * rows
        + reference_transform.c
    )
    northings = (
        reference_transform.d * columns
        + reference_transform.e * rows
        + reference_transform.f
    )
    world_points = np.column_stack((eastings, northings)).astype(np.float64)
    if not np.all(np.isfinite(world_points)):
        raise MeasurementFailure("Преобразование GeoTIFF вернуло NaN или inf")
    return world_points


def map_frame_points_to_world(
    frame_points_px: NDArray[np.floating]
    | list[list[float]]
    | tuple[tuple[float, float], ...],
    homography_frame_to_reference: Homography,
    *,
    reference_transform: Affine,
    reference_crs: object,
    reference_width_pixels: int,
    reference_height_pixels: int,
) -> MappedFramePoints:
    """Выполняет полную цепочку кадр → эталон → метрические координаты."""

    if reference_width_pixels <= 1 or reference_height_pixels <= 1:
        raise MeasurementFailure("Размеры эталона должны быть больше одного пикселя")

    frame_points = _validate_points(
        frame_points_px,
        name="Точки кадра",
        minimum_count=1,
    )
    reference_points = map_frame_points_to_reference(
        frame_points,
        homography_frame_to_reference,
    )
    outside_reference = (
        (reference_points[:, 0] < 0.0)
        | (reference_points[:, 0] > reference_width_pixels - 1.0)
        | (reference_points[:, 1] < 0.0)
        | (reference_points[:, 1] > reference_height_pixels - 1.0)
    )
    if np.any(outside_reference):
        raise MeasurementFailure(
            "Хотя бы одна измеряемая точка перенесена за границы эталона"
        )

    world_points = map_reference_points_to_world(
        reference_points,
        reference_transform=reference_transform,
        reference_crs=reference_crs,
    )
    return MappedFramePoints(
        frame_points_px=frame_points,
        reference_points_px=reference_points,
        world_points_m=world_points,
    )


def measure_segment(
    frame_points_px: NDArray[np.floating]
    | list[list[float]]
    | tuple[tuple[float, float], ...],
    homography_frame_to_reference: Homography,
    *,
    reference_transform: Affine,
    reference_crs: object,
    reference_width_pixels: int,
    reference_height_pixels: int,
) -> SegmentMeasurement:
    """Измеряет один отрезок, заданный ровно двумя точками кадра."""

    points = _validate_points(
        frame_points_px,
        name="Концы отрезка",
        minimum_count=2,
    )
    if points.shape[0] != 2:
        raise MeasurementFailure("Для отрезка требуется ровно две точки")

    mapped = map_frame_points_to_world(
        points,
        homography_frame_to_reference,
        reference_transform=reference_transform,
        reference_crs=reference_crs,
        reference_width_pixels=reference_width_pixels,
        reference_height_pixels=reference_height_pixels,
    )
    length = float(np.linalg.norm(mapped.world_points_m[1] - mapped.world_points_m[0]))
    return SegmentMeasurement(mapped_points=mapped, length_meters=length)


def measure_polygon(
    frame_points_px: NDArray[np.floating]
    | list[list[float]]
    | tuple[tuple[float, float], ...],
    homography_frame_to_reference: Homography,
    *,
    reference_transform: Affine,
    reference_crs: object,
    reference_width_pixels: int,
    reference_height_pixels: int,
) -> PolygonMeasurement:
    """Измеряет площадь непересекающегося полигона из точек кадра.

    Некорректный контур не «исправляется» автоматически: операция ``make_valid``
    могла бы превратить пользовательский ввод в несколько фигур и тем самым
    скрыть ошибку разметки. Вместо этого возвращается мотивированный отказ.
    """

    points = _validate_points(
        frame_points_px,
        name="Вершины полигона",
        minimum_count=3,
    )
    mapped = map_frame_points_to_world(
        points,
        homography_frame_to_reference,
        reference_transform=reference_transform,
        reference_crs=reference_crs,
        reference_width_pixels=reference_width_pixels,
        reference_height_pixels=reference_height_pixels,
    )
    polygon = Polygon(mapped.world_points_m)
    if not polygon.is_valid:
        raise MeasurementFailure("Полигон невалиден: " + str(is_valid_reason(polygon)))
    if polygon.is_empty or polygon.area <= 0.0:
        raise MeasurementFailure("Полигон должен ограничивать ненулевую площадь")

    return PolygonMeasurement(
        mapped_points=mapped,
        area_square_meters=float(polygon.area),
    )
