"""Оценка одной гомографии на земле и неплоских поверхностях.

Модуль хранит независимые контрольные точки вместе с их типом поверхности.
Истинные соответствия позволяют построить oracle-гомографию земли — лучший
ответ плоской модели без ошибки SIFT. Затем одна и та же измерительная цепочка
оценивается на земле и крышах. Так параллакс не смешивается с ошибкой matcher-а.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from rasterio import Affine

from aerial_mapper.measurement import measure_segment
from aerial_mapper.synthetic import Homography

SurfaceKind = Literal["ground", "roof"]


@dataclass(frozen=True)
class SurfaceSegmentControl:
    """Контрольный отрезок с пикселями двух видов и мировой истиной."""

    name: str
    surface: SurfaceKind
    frame_points_px: NDArray[np.float64]
    reference_points_px: NDArray[np.float64]
    true_world_points_m: NDArray[np.float64]


@dataclass(frozen=True)
class SurfaceSegmentEvaluation:
    """Ошибка одного отрезка после применения заданной гомографии."""

    name: str
    surface: SurfaceKind
    true_length_m: float
    estimated_length_m: float
    absolute_length_error_m: float
    relative_length_error_percent: float
    endpoint_position_errors_m: NDArray[np.float64]
    maximum_endpoint_position_error_m: float


def estimate_oracle_homography(
    controls: tuple[SurfaceSegmentControl, ...],
    *,
    surface: SurfaceKind | None = None,
) -> Homography:
    """Строит точную наилучшую H по выбранным скрытым соответствиям.

    Если surface равен ground, матрица показывает предел плоской модели без
    ошибки поиска признаков. Если фильтр не задан, least-squares решение
    компромиссно смешивает землю и крышу, между которыми при параллаксе одной
    точной гомографии уже не существует.
    """

    selected = [
        control for control in controls if surface is None or control.surface == surface
    ]
    if not selected:
        raise ValueError("Нет контрольных отрезков для oracle-гомографии")
    frame_points = np.concatenate(
        [control.frame_points_px for control in selected],
        axis=0,
    ).astype(np.float64)
    reference_points = np.concatenate(
        [control.reference_points_px for control in selected],
        axis=0,
    ).astype(np.float64)
    if frame_points.shape[0] < 4:
        raise ValueError("Для oracle-гомографии нужны минимум четыре точки")

    homography, _ = cv2.findHomography(
        frame_points,
        reference_points,
        method=0,
    )
    if homography is None or not np.all(np.isfinite(homography)):
        raise ValueError("Контрольные соответствия не задали гомографию")
    if np.isclose(homography[2, 2], 0.0):
        raise ValueError("Получена вырожденная oracle-гомография")
    homography = homography.astype(np.float64)
    homography /= homography[2, 2]
    return homography


def evaluate_surface_segments(
    controls: tuple[SurfaceSegmentControl, ...],
    *,
    homography_frame_to_reference: Homography,
    reference_transform: Affine,
    reference_crs: object,
    reference_width_pixels: int,
    reference_height_pixels: int,
) -> tuple[SurfaceSegmentEvaluation, ...]:
    """Сравнивает измерения заданной H с прямой мировой истиной."""

    evaluations: list[SurfaceSegmentEvaluation] = []
    for control in controls:
        measurement = measure_segment(
            control.frame_points_px,
            homography_frame_to_reference,
            reference_transform=reference_transform,
            reference_crs=reference_crs,
            reference_width_pixels=reference_width_pixels,
            reference_height_pixels=reference_height_pixels,
        )
        estimated_world = measurement.mapped_points.world_points_m
        true_world = np.asarray(control.true_world_points_m, dtype=np.float64)
        if true_world.shape != (2, 2):
            raise ValueError(
                f"Мировые точки {control.name!r} должны иметь форму (2, 2)"
            )
        endpoint_errors = np.linalg.norm(estimated_world - true_world, axis=1)
        true_length = float(np.linalg.norm(true_world[1] - true_world[0]))
        if true_length <= 0.0:
            raise ValueError(
                f"Истинная длина {control.name!r} должна быть положительной"
            )
        estimated_length = measurement.length_meters
        absolute_error = abs(estimated_length - true_length)
        evaluations.append(
            SurfaceSegmentEvaluation(
                name=control.name,
                surface=control.surface,
                true_length_m=true_length,
                estimated_length_m=estimated_length,
                absolute_length_error_m=absolute_error,
                relative_length_error_percent=absolute_error / true_length * 100.0,
                endpoint_position_errors_m=endpoint_errors,
                maximum_endpoint_position_error_m=float(np.max(endpoint_errors)),
            )
        )
    return tuple(evaluations)


def surface_evaluation_to_dict(
    evaluation: SurfaceSegmentEvaluation,
) -> dict[str, object]:
    """Преобразует результат в JSON-совместимую запись без потери точности."""

    return {
        "name": evaluation.name,
        "surface": evaluation.surface,
        "true_length_m": evaluation.true_length_m,
        "estimated_length_m": evaluation.estimated_length_m,
        "absolute_length_error_m": evaluation.absolute_length_error_m,
        "relative_length_error_percent": evaluation.relative_length_error_percent,
        "endpoint_position_errors_m": evaluation.endpoint_position_errors_m.tolist(),
        "maximum_endpoint_position_error_m": (
            evaluation.maximum_endpoint_position_error_m
        ),
    }
