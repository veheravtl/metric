"""Визуализации для проверки геометрии синтетического эксперимента."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from aerial_mapper.alignment import AlignmentResult
from aerial_mapper.measurement_evaluation import (
    MetricControlDefinition,
    MetricControlEvaluation,
)
from aerial_mapper.synthetic import FloatPoints, Homography, RgbImage


@dataclass(frozen=True)
class ReverseOverlayResult:
    """Результат обратного наложения кадра на полный эталон."""

    comparison_rgb: RgbImage
    mean_absolute_error: float
    covered_pixels: int


def draw_metric_controls_on_frame(
    frame_rgb: RgbImage,
    controls: tuple[MetricControlDefinition, ...],
) -> RgbImage:
    """Показывает независимые контрольные фигуры, измеряемые после привязки."""

    annotated = frame_rgb.copy()
    line_thickness = max(2, round(min(frame_rgb.shape[:2]) / 400))
    for index, control in enumerate(controls, start=1):
        points = np.rint(control.frame_points_px).astype(np.int32)
        color = (255, 220, 40) if control.kind == "segment" else (255, 80, 220)
        cv2.polylines(
            annotated,
            [points],
            isClosed=control.kind == "polygon",
            color=color,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )
        for point in points:
            cv2.circle(
                annotated,
                tuple(int(value) for value in point),
                radius=line_thickness * 2,
                color=color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        label_point = tuple(int(value) for value in points[0])
        cv2.putText(
            annotated,
            str(index),
            (label_point[0] + 8, label_point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            thickness=4,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            str(index),
            (label_point[0] + 8, label_point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    cv2.rectangle(annotated, (12, 12), (660, 82), (20, 20, 20), thickness=-1)
    cv2.putText(
        annotated,
        "YELLOW: distance controls",
        (28, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 220, 40),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "MAGENTA: area controls",
        (28, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 80, 220),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return annotated


def draw_metric_evaluations_on_reference(
    reference_rgb: RgbImage,
    evaluations: tuple[MetricControlEvaluation, ...],
) -> RgbImage:
    """Накладывает истинные и измеренные положения контрольных фигур."""

    annotated = reference_rgb.copy()
    line_thickness = max(2, round(min(reference_rgb.shape[:2]) / 900))
    for evaluation in evaluations:
        true_points = np.rint(evaluation.true_reference_points_px).astype(np.int32)
        estimated_points = np.rint(evaluation.estimated_reference_points_px).astype(
            np.int32
        )
        is_closed = evaluation.kind == "polygon"
        cv2.polylines(
            annotated,
            [true_points],
            isClosed=is_closed,
            color=(255, 128, 0),
            thickness=line_thickness + 2,
            lineType=cv2.LINE_AA,
        )
        cv2.polylines(
            annotated,
            [estimated_points],
            isClosed=is_closed,
            color=(0, 255, 255),
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

    reference_height = reference_rgb.shape[0]
    legend_top = reference_height - 95
    cv2.rectangle(
        annotated,
        (18, legend_top),
        (650, reference_height - 18),
        (20, 20, 20),
        thickness=-1,
    )
    cv2.putText(
        annotated,
        "METRIC CONTROLS: ORANGE truth / CYAN estimate",
        (35, legend_top + 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return annotated


def draw_reference_footprint(
    reference_rgb: RgbImage,
    corners_reference_px: FloatPoints,
) -> RgbImage:
    """Рисует на копии эталона след виртуальной камеры и номера углов."""

    annotated = reference_rgb.copy()
    polygon = np.rint(corners_reference_px).astype(np.int32)

    # Оранжевый контур хорошо заметен и на зелёной растительности, и на сером
    # асфальте. Толщина масштабируется вместе с размером эталонного изображения.
    line_thickness = max(2, round(min(reference_rgb.shape[:2]) / 700))
    cv2.polylines(
        annotated,
        [polygon],
        isClosed=True,
        color=(255, 128, 0),
        thickness=line_thickness,
        lineType=cv2.LINE_AA,
    )

    for point_number, (x_coordinate, y_coordinate) in enumerate(polygon, start=1):
        cv2.circle(
            annotated,
            (int(x_coordinate), int(y_coordinate)),
            radius=line_thickness * 3,
            color=(255, 255, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            annotated,
            (int(x_coordinate), int(y_coordinate)),
            radius=line_thickness * 3,
            color=(255, 128, 0),
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            str(point_number),
            (int(x_coordinate) + 12, int(y_coordinate) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            thickness=4,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            str(point_number),
            (int(x_coordinate) + 12, int(y_coordinate) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (180, 70, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return annotated


def draw_alignment_footprints(
    reference_rgb: RgbImage,
    true_corners_reference_px: FloatPoints,
    estimated_corners_reference_px: FloatPoints,
) -> RgbImage:
    """Показывает истинный и независимо найденный следы на одном эталоне."""

    annotated = draw_reference_footprint(
        reference_rgb,
        true_corners_reference_px,
    )
    estimated_polygon = np.rint(estimated_corners_reference_px).astype(np.int32)
    line_thickness = max(2, round(min(reference_rgb.shape[:2]) / 900))
    cv2.polylines(
        annotated,
        [estimated_polygon],
        isClosed=True,
        color=(0, 255, 255),
        thickness=line_thickness,
        lineType=cv2.LINE_AA,
    )

    # Легенда наносится прямо на изображение, чтобы смысл цветов сохранялся и
    # после скачивания PNG отдельно от веб-интерфейса.
    cv2.rectangle(annotated, (18, 18), (570, 105), (20, 20, 20), thickness=-1)
    cv2.putText(
        annotated,
        "ORANGE: ground truth",
        (35, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 128, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "CYAN: SIFT + RANSAC estimate",
        (35, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return annotated


def _resize_to_height(
    image_rgb: RgbImage, target_height: int
) -> tuple[RgbImage, float]:
    """Масштабирует изображение для компактной диагностической композиции."""

    scale = target_height / image_rgb.shape[0]
    target_width = max(1, round(image_rgb.shape[1] * scale))
    resized = cv2.resize(
        image_rgb,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    return resized, scale


def _sample_indices(indices: np.ndarray, maximum_count: int) -> np.ndarray:
    """Детерминированно выбирает точки по всему списку для читаемого рисунка."""

    if indices.size <= maximum_count:
        return indices
    positions = np.linspace(0, indices.size - 1, maximum_count, dtype=np.int64)
    return indices[positions]


def draw_alignment_matches(
    reference_rgb: RgbImage,
    frame_rgb: RgbImage,
    alignment: AlignmentResult,
    *,
    target_height: int = 900,
    maximum_inlier_lines: int = 120,
    maximum_outlier_lines: int = 30,
    reference_label: str = "REFERENCE",
    frame_label: str = "SYNTHETIC FRAME",
) -> RgbImage:
    """Рисует часть SIFT-пар: зелёные inlier и красные outlier RANSAC.

    Все сотни линий сделали бы изображение нечитаемым, поэтому визуализация
    показывает детерминированную выборку. Числа в диагностике при этом относятся
    ко всем соответствиям, а не только к нарисованным.
    """

    reference_display, reference_scale = _resize_to_height(
        reference_rgb,
        target_height,
    )
    frame_display, frame_scale = _resize_to_height(frame_rgb, target_height)
    separator_width = 6
    reference_width = reference_display.shape[1]
    frame_offset_x = reference_width + separator_width

    canvas = np.zeros(
        (
            target_height,
            reference_width + separator_width + frame_display.shape[1],
            3,
        ),
        dtype=np.uint8,
    )
    canvas[:, :reference_width] = reference_display
    canvas[:, reference_width:frame_offset_x] = 235
    canvas[:, frame_offset_x:] = frame_display

    all_indices = np.arange(alignment.ratio_match_count, dtype=np.int64)
    inlier_indices = _sample_indices(
        all_indices[alignment.inlier_mask],
        maximum_inlier_lines,
    )
    outlier_indices = _sample_indices(
        all_indices[~alignment.inlier_mask],
        maximum_outlier_lines,
    )

    def draw_pairs(indices: np.ndarray, color: tuple[int, int, int]) -> None:
        for index in indices:
            reference_point = alignment.reference_points_px[index]
            frame_point = alignment.frame_points_px[index]
            reference_xy = (
                round(float(reference_point[0]) * reference_scale),
                round(float(reference_point[1]) * reference_scale),
            )
            frame_xy = (
                frame_offset_x + round(float(frame_point[0]) * frame_scale),
                round(float(frame_point[1]) * frame_scale),
            )
            cv2.line(
                canvas,
                reference_xy,
                frame_xy,
                color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(canvas, reference_xy, 3, color, thickness=-1)
            cv2.circle(canvas, frame_xy, 3, color, thickness=-1)

    # Сначала рисуем ошибочные пары, чтобы основные зелёные связи оставались
    # видимыми поверх них.
    draw_pairs(outlier_indices, (255, 70, 70))
    draw_pairs(inlier_indices, (50, 255, 100))

    cv2.rectangle(
        canvas,
        (12, 12),
        (canvas.shape[1] - 12, 86),
        (15, 15, 15),
        thickness=-1,
    )
    cv2.putText(
        canvas,
        reference_label,
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        frame_label,
        (frame_offset_x + 18, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "GREEN: RANSAC inlier    RED: rejected match",
        (25, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (210, 255, 220),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return canvas


def build_reverse_overlay(
    reference_rgb: RgbImage,
    frame_rgb: RgbImage,
    homography_frame_to_reference: Homography,
    corners_reference_px: FloatPoints,
) -> ReverseOverlayResult:
    """Возвращает красно-бирюзовое сравнение после обратной проекции.

    Внутри следа камеры красный канал берётся из возвращённого кадра, а зелёный
    и синий — из эталона. При точном совпадении изображение выглядит серым.
    Геометрическая ошибка проявляется парными красными и бирюзовыми контурами,
    которые визуально заметнее обычного полупрозрачного наложения.
    """

    reference_height, reference_width = reference_rgb.shape[:2]
    warped_frame = cv2.warpPerspective(
        frame_rgb,
        homography_frame_to_reference,
        (reference_width, reference_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    frame_mask = np.full(frame_rgb.shape[:2], 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(
        frame_mask,
        homography_frame_to_reference,
        (reference_width, reference_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid_mask = warped_mask > 0
    covered_pixels = int(np.count_nonzero(valid_mask))
    if covered_pixels == 0:
        raise ValueError("Обратная проекция кадра не пересекает эталон")

    reference_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
    warped_gray = cv2.cvtColor(warped_frame, cv2.COLOR_RGB2GRAY)
    comparison = cv2.cvtColor(reference_gray, cv2.COLOR_GRAY2RGB)
    comparison[valid_mask, 0] = warped_gray[valid_mask]
    comparison[valid_mask, 1] = reference_gray[valid_mask]
    comparison[valid_mask, 2] = reference_gray[valid_mask]

    absolute_difference = cv2.absdiff(reference_rgb, warped_frame)
    mean_absolute_error = float(absolute_difference[valid_mask].mean())

    polygon = np.rint(corners_reference_px).astype(np.int32)
    cv2.polylines(
        comparison,
        [polygon],
        isClosed=True,
        color=(255, 180, 0),
        thickness=max(2, round(min(reference_rgb.shape[:2]) / 900)),
        lineType=cv2.LINE_AA,
    )

    return ReverseOverlayResult(
        comparison_rgb=comparison,
        mean_absolute_error=mean_absolute_error,
        covered_pixels=covered_pixels,
    )
