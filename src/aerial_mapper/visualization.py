"""Визуализации для проверки геометрии синтетического эксперимента."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from aerial_mapper.synthetic import FloatPoints, Homography, RgbImage


@dataclass(frozen=True)
class ReverseOverlayResult:
    """Результат обратного наложения кадра на полный эталон."""

    comparison_rgb: RgbImage
    mean_absolute_error: float
    covered_pixels: int


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
