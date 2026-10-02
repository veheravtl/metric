"""Количественная оценка найденной гомографии по скрытой истине."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from aerial_mapper.synthetic import FloatPoints, Homography


@dataclass(frozen=True)
class HomographyEvaluation:
    """Ошибки переноса независимой регулярной сетки контрольных точек."""

    control_point_count: int
    errors_pixels: NDArray[np.float64]
    errors_meters: NDArray[np.float64]
    corner_errors_pixels: NDArray[np.float64]
    mean_error_pixels: float
    median_error_pixels: float
    max_error_pixels: float
    root_mean_square_error_pixels: float
    mean_error_meters: float
    median_error_meters: float
    max_error_meters: float


def _project_points(points_px: FloatPoints, homography: Homography) -> FloatPoints:
    """Проецирует двумерные точки и запрещает нечисловой результат."""

    projected = cv2.perspectiveTransform(
        points_px.reshape(1, -1, 2),
        homography,
    ).reshape(-1, 2)
    if not np.isfinite(projected).all():
        raise ValueError("Гомография переводит контрольные точки в бесконечность")
    return projected.astype(np.float32)


def evaluate_homography(
    estimated_frame_to_reference: Homography,
    true_frame_to_reference: Homography,
    *,
    frame_width_pixels: int,
    frame_height_pixels: int,
    reference_resolution_m_per_pixel: float,
    grid_columns: int = 9,
    grid_rows: int = 7,
) -> HomographyEvaluation:
    """Сравнивает две гомографии по результату, а не по элементам матриц.

    Гомография определена с точностью до общего ненулевого множителя, поэтому
    прямое вычитание матриц не является содержательной метрикой. Вместо этого
    обе матрицы переносят одинаковую сетку точек кадра в систему эталона.
    Расстояние между полученными координатами измеряет практическую ошибку
    привязки по всей площади, включая области между найденными SIFT-точками.
    """

    if frame_width_pixels <= 1 or frame_height_pixels <= 1:
        raise ValueError("Размер кадра должен быть больше одного пикселя")
    if reference_resolution_m_per_pixel <= 0:
        raise ValueError("Разрешение эталона должно быть положительным")
    if grid_columns < 2 or grid_rows < 2:
        raise ValueError("Контрольная сетка должна иметь хотя бы 2 × 2 точки")

    x_coordinates = np.linspace(0, frame_width_pixels - 1, grid_columns)
    y_coordinates = np.linspace(0, frame_height_pixels - 1, grid_rows)
    grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)
    control_points = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(
        np.float32
    )

    estimated_reference_points = _project_points(
        control_points,
        estimated_frame_to_reference,
    )
    true_reference_points = _project_points(
        control_points,
        true_frame_to_reference,
    )
    errors_pixels = np.linalg.norm(
        estimated_reference_points.astype(np.float64)
        - true_reference_points.astype(np.float64),
        axis=1,
    )
    errors_meters = errors_pixels * reference_resolution_m_per_pixel

    corner_indices = np.array(
        [0, grid_columns - 1, -grid_columns, -1],
        dtype=np.int64,
    )
    corner_errors = errors_pixels[corner_indices]

    return HomographyEvaluation(
        control_point_count=int(control_points.shape[0]),
        errors_pixels=errors_pixels,
        errors_meters=errors_meters,
        corner_errors_pixels=corner_errors,
        mean_error_pixels=float(errors_pixels.mean()),
        median_error_pixels=float(np.median(errors_pixels)),
        max_error_pixels=float(errors_pixels.max()),
        root_mean_square_error_pixels=float(np.sqrt(np.mean(errors_pixels**2))),
        mean_error_meters=float(errors_meters.mean()),
        median_error_meters=float(np.median(errors_meters)),
        max_error_meters=float(errors_meters.max()),
    )
