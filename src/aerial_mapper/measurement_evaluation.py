"""Независимая оценка метрических измерений по скрытой истинной гомографии."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from rasterio import Affine

from aerial_mapper.measurement import measure_polygon, measure_segment
from aerial_mapper.synthetic import Homography

ControlKind = Literal["segment", "polygon"]


@dataclass(frozen=True)
class MetricControlDefinition:
    """Фигура в кадре, не передаваемая алгоритму поиска соответствий."""

    name: str
    kind: ControlKind
    frame_points_px: NDArray[np.float64]


@dataclass(frozen=True)
class MetricControlEvaluation:
    """Истинное и измеренное значения одной контрольной фигуры."""

    name: str
    kind: ControlKind
    frame_points_px: NDArray[np.float64]
    true_reference_points_px: NDArray[np.float64]
    estimated_reference_points_px: NDArray[np.float64]
    true_world_points_m: NDArray[np.float64]
    estimated_world_points_m: NDArray[np.float64]
    true_value: float
    estimated_value: float
    unit: str
    absolute_error: float
    relative_error_percent: float
    mean_vertex_position_error_meters: float
    max_vertex_position_error_meters: float


def _fractional_points_to_frame_pixels(
    fractional_points: tuple[tuple[float, float], ...],
    *,
    frame_width_pixels: int,
    frame_height_pixels: int,
) -> NDArray[np.float64]:
    """Переводит стабильные доли кадра в его субпиксельные координаты."""

    fractions = np.asarray(fractional_points, dtype=np.float64)
    return fractions * np.asarray(
        [frame_width_pixels - 1.0, frame_height_pixels - 1.0],
        dtype=np.float64,
    )


def build_poc_control_definitions(
    *,
    frame_width_pixels: int,
    frame_height_pixels: int,
) -> tuple[MetricControlDefinition, ...]:
    """Создаёт набор отрезков и полигонов разных размеров и положений."""

    if frame_width_pixels <= 1 or frame_height_pixels <= 1:
        raise ValueError("Размеры кадра должны быть больше одного пикселя")

    raw_definitions: tuple[
        tuple[str, ControlKind, tuple[tuple[float, float], ...]], ...
    ] = (
        ("short_horizontal", "segment", ((0.45, 0.50), (0.55, 0.50))),
        ("long_vertical", "segment", ((0.20, 0.15), (0.20, 0.85))),
        ("long_diagonal", "segment", ((0.12, 0.18), (0.88, 0.82))),
        ("near_top", "segment", ((0.08, 0.10), (0.38, 0.10))),
        (
            "center_rectangle",
            "polygon",
            ((0.35, 0.35), (0.65, 0.35), (0.65, 0.65), (0.35, 0.65)),
        ),
        ("edge_triangle", "polygon", ((0.06, 0.12), (0.30, 0.14), (0.14, 0.38))),
        (
            "irregular_quadrilateral",
            "polygon",
            ((0.58, 0.58), (0.90, 0.62), (0.82, 0.90), (0.55, 0.80)),
        ),
    )
    return tuple(
        MetricControlDefinition(
            name=name,
            kind=kind,
            frame_points_px=_fractional_points_to_frame_pixels(
                fractional_points,
                frame_width_pixels=frame_width_pixels,
                frame_height_pixels=frame_height_pixels,
            ),
        )
        for name, kind, fractional_points in raw_definitions
    )


def evaluate_metric_controls(
    controls: tuple[MetricControlDefinition, ...],
    *,
    estimated_homography_frame_to_reference: Homography,
    true_homography_frame_to_reference: Homography,
    reference_transform: Affine,
    reference_crs: object,
    reference_width_pixels: int,
    reference_height_pixels: int,
) -> tuple[MetricControlEvaluation, ...]:
    """Сравнивает рабочее измерение с веткой скрытой геометрической истины."""

    evaluations: list[MetricControlEvaluation] = []
    for control in controls:
        common_arguments = {
            "reference_transform": reference_transform,
            "reference_crs": reference_crs,
            "reference_width_pixels": reference_width_pixels,
            "reference_height_pixels": reference_height_pixels,
        }
        if control.kind == "segment":
            estimated_measurement = measure_segment(
                control.frame_points_px,
                estimated_homography_frame_to_reference,
                **common_arguments,
            )
            true_measurement = measure_segment(
                control.frame_points_px,
                true_homography_frame_to_reference,
                **common_arguments,
            )
            estimated_value = estimated_measurement.length_meters
            true_value = true_measurement.length_meters
            unit = "m"
        elif control.kind == "polygon":
            estimated_measurement = measure_polygon(
                control.frame_points_px,
                estimated_homography_frame_to_reference,
                **common_arguments,
            )
            true_measurement = measure_polygon(
                control.frame_points_px,
                true_homography_frame_to_reference,
                **common_arguments,
            )
            estimated_value = estimated_measurement.area_square_meters
            true_value = true_measurement.area_square_meters
            unit = "m²"
        else:
            raise ValueError(f"Неизвестный вид контрольной фигуры: {control.kind}")

        estimated_mapped = estimated_measurement.mapped_points
        true_mapped = true_measurement.mapped_points
        vertex_errors = np.linalg.norm(
            estimated_mapped.world_points_m - true_mapped.world_points_m,
            axis=1,
        )
        absolute_error = abs(estimated_value - true_value)
        evaluations.append(
            MetricControlEvaluation(
                name=control.name,
                kind=control.kind,
                frame_points_px=control.frame_points_px,
                true_reference_points_px=true_mapped.reference_points_px,
                estimated_reference_points_px=estimated_mapped.reference_points_px,
                true_world_points_m=true_mapped.world_points_m,
                estimated_world_points_m=estimated_mapped.world_points_m,
                true_value=true_value,
                estimated_value=estimated_value,
                unit=unit,
                absolute_error=absolute_error,
                relative_error_percent=absolute_error / true_value * 100.0,
                mean_vertex_position_error_meters=float(np.mean(vertex_errors)),
                max_vertex_position_error_meters=float(np.max(vertex_errors)),
            )
        )
    return tuple(evaluations)


def metric_evaluation_to_dict(
    evaluation: MetricControlEvaluation,
    *,
    decimals: int | None = None,
) -> dict[str, object]:
    """Преобразует результат в сериализуемую строку JSON/CSV."""

    def maybe_round(value: float) -> float:
        return value if decimals is None else round(value, decimals)

    return {
        "name": evaluation.name,
        "kind": evaluation.kind,
        "unit": evaluation.unit,
        "true_value": maybe_round(evaluation.true_value),
        "estimated_value": maybe_round(evaluation.estimated_value),
        "absolute_error": maybe_round(evaluation.absolute_error),
        "relative_error_percent": maybe_round(evaluation.relative_error_percent),
        "mean_vertex_position_error_meters": maybe_round(
            evaluation.mean_vertex_position_error_meters
        ),
        "max_vertex_position_error_meters": maybe_round(
            evaluation.max_vertex_position_error_meters
        ),
    }
