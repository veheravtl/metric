"""Оценка плоской гомографии по плотной 3D-истине реального кадра.

OrthoLoC хранит для каждого пикселя UAV-кадра мировую XYZ-координату видимой
поверхности. Это существенно сильнее единственной координаты центра: найденную
гомографию можно проверить во многих точках, причём истинные данные не передаются
алгоритму сопоставления и используются только после его завершения.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from aerial_mapper.synthetic import FloatPoints, Homography


@dataclass(frozen=True)
class DenseHomographyEvaluation:
    """Ошибки одной гомографии относительно неплоской плотной истины.

    ``query_points_px`` — контрольные точки в UAV-кадре. Для каждой из них
    ``true_reference_points_px`` задаёт истинное плановое положение видимой
    3D-точки на ортографическом растре, а ``estimated_reference_points_px`` —
    положение после переноса проверяемой гомографией.
    """

    query_points_px: FloatPoints
    true_reference_points_px: FloatPoints
    estimated_reference_points_px: FloatPoints
    errors_meters: np.ndarray
    mean_error_meters: float
    median_error_meters: float
    p95_error_meters: float
    max_error_meters: float


def point_map_to_reference_pixels(
    point_map_xyz: np.ndarray,
    *,
    reference_offset_xy: np.ndarray,
    reference_scale_xy_m_per_pixel: np.ndarray,
) -> np.ndarray:
    """Переводит плотную XYZ-карту кадра в пиксели ортографического растра.

    В OrthoLoC первые два канала ``point_map`` уже выражены в локальных мировых
    координатах X/Y. ``reference_offset_xy`` задаёт мировую координату пикселя
    ``(0, 0)`` DOP, а signed scale — изменение X/Y на один пиксель. Масштаб по Y
    обычно отрицательный, потому что строки изображения растут вниз, тогда как
    мировая ось Y направлена вверх.
    """

    point_map = np.asarray(point_map_xyz, dtype=np.float64)
    offset = np.asarray(reference_offset_xy, dtype=np.float64)
    scale = np.asarray(reference_scale_xy_m_per_pixel, dtype=np.float64)
    if point_map.ndim != 3 or point_map.shape[2] != 3:
        raise ValueError("Плотная карта должна иметь форму H × W × 3")
    if offset.shape != (2,):
        raise ValueError("Смещение ортографического растра должно иметь форму (2,)")
    if scale.shape != (2,) or not np.isfinite(scale).all():
        raise ValueError("Метрический масштаб растра должен иметь два числа")
    if np.any(np.isclose(scale, 0.0)):
        raise ValueError("Метрический масштаб растра не может быть нулевым")

    reference_map = np.full((*point_map.shape[:2], 2), np.nan, dtype=np.float64)
    valid_mask = np.isfinite(point_map).all(axis=2)
    reference_map[valid_mask] = (
        point_map[valid_mask, :2] - offset[np.newaxis, :]
    ) / scale[np.newaxis, :]
    return reference_map


def sample_dense_map_bilinearly(
    dense_map: np.ndarray,
    query_points_px: FloatPoints,
) -> tuple[FloatPoints, np.ndarray]:
    """Билинейно считывает двумерную карту в дробных пиксельных координатах.

    SIFT возвращает координаты с субпиксельной точностью. Округление до целого
    пикселя само внесло бы заметную ошибку: при масштабе около 0,18 м/пиксель
    она могла бы достигнуть нескольких сантиметров. Поэтому используются четыре
    соседних отсчёта. Точка считается невалидной, если хотя бы один из них
    отсутствует или содержит NaN.
    """

    values_map = np.asarray(dense_map, dtype=np.float64)
    points = np.asarray(query_points_px, dtype=np.float64)
    if values_map.ndim != 3 or values_map.shape[2] != 2:
        raise ValueError("Плотная двумерная карта должна иметь форму H × W × 2")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Точки запроса должны иметь форму N × 2")

    height, width = values_map.shape[:2]
    x = points[:, 0]
    y = points[:, 1]
    inside = (
        np.isfinite(points).all(axis=1)
        & (x >= 0.0)
        & (y >= 0.0)
        & (x < width - 1)
        & (y < height - 1)
    )
    sampled = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    if not np.any(inside):
        return sampled.astype(np.float32), inside

    valid_indices = np.flatnonzero(inside)
    valid_x = x[inside]
    valid_y = y[inside]
    x0 = np.floor(valid_x).astype(np.int64)
    y0 = np.floor(valid_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    neighbours = np.stack(
        (
            values_map[y0, x0],
            values_map[y0, x1],
            values_map[y1, x0],
            values_map[y1, x1],
        ),
        axis=1,
    )
    finite_neighbours = np.isfinite(neighbours).all(axis=(1, 2))
    inside[valid_indices[~finite_neighbours]] = False
    if not np.any(finite_neighbours):
        return sampled.astype(np.float32), inside

    dx = (valid_x[finite_neighbours] - x0[finite_neighbours])[:, np.newaxis]
    dy = (valid_y[finite_neighbours] - y0[finite_neighbours])[:, np.newaxis]
    finite_values = neighbours[finite_neighbours]
    top = finite_values[:, 0] * (1.0 - dx) + finite_values[:, 1] * dx
    bottom = finite_values[:, 2] * (1.0 - dx) + finite_values[:, 3] * dx
    interpolated = top * (1.0 - dy) + bottom * dy
    sampled[valid_indices[finite_neighbours]] = interpolated
    return sampled.astype(np.float32), inside


def build_regular_query_grid(
    *,
    frame_width_pixels: int,
    frame_height_pixels: int,
    columns: int = 41,
    rows: int = 31,
) -> FloatPoints:
    """Создаёт регулярную сетку внутри кадра, не касаясь крайней границы."""

    if frame_width_pixels < 3 or frame_height_pixels < 3:
        raise ValueError("Кадр должен иметь размер не менее 3 × 3 пикселей")
    if columns < 2 or rows < 2:
        raise ValueError("Контрольная сетка должна иметь хотя бы 2 × 2 точки")
    x_coordinates = np.linspace(0.5, frame_width_pixels - 1.5, columns)
    y_coordinates = np.linspace(0.5, frame_height_pixels - 1.5, rows)
    grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)
    return np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)


def select_valid_dense_correspondences(
    query_points_px: FloatPoints,
    dense_reference_map_px: np.ndarray,
    *,
    reference_width_pixels: int,
    reference_height_pixels: int,
) -> tuple[FloatPoints, FloatPoints]:
    """Оставляет точки с известной истиной внутри ортографического растра."""

    true_points, valid_mask = sample_dense_map_bilinearly(
        dense_reference_map_px,
        query_points_px,
    )
    inside_reference = (
        valid_mask
        & (true_points[:, 0] >= 0.0)
        & (true_points[:, 1] >= 0.0)
        & (true_points[:, 0] < reference_width_pixels)
        & (true_points[:, 1] < reference_height_pixels)
    )
    return (
        np.asarray(query_points_px, dtype=np.float32)[inside_reference],
        true_points[inside_reference],
    )


def fit_best_single_homography(
    query_points_px: FloatPoints,
    true_reference_points_px: FloatPoints,
) -> Homography:
    """Подбирает описательную least-squares гомографию по плотной истине.

    Это не результат алгоритма локализации и не допустимая подсказка ему.
    Матрица строится только после эксперимента и показывает, насколько хорошо
    одна плоская модель вообще способна описать видимую неплоскую сцену.
    """

    query_points = np.asarray(query_points_px, dtype=np.float32)
    true_points = np.asarray(true_reference_points_px, dtype=np.float32)
    if query_points.shape != true_points.shape or query_points.shape[0] < 4:
        raise ValueError("Для подбора гомографии нужны минимум четыре пары точек")
    homography, _ = cv2.findHomography(query_points, true_points, method=0)
    if homography is None or not np.isfinite(homography).all():
        raise ValueError("Не удалось подобрать описательную гомографию")
    return homography.astype(np.float64)


def evaluate_homography_against_dense_truth(
    homography_query_to_reference: Homography,
    query_points_px: FloatPoints,
    true_reference_points_px: FloatPoints,
    *,
    reference_scale_xy_m_per_pixel: np.ndarray,
) -> DenseHomographyEvaluation:
    """Считает плановую ошибку гомографии в метрах по известной 3D-истине."""

    query_points = np.asarray(query_points_px, dtype=np.float32)
    true_points = np.asarray(true_reference_points_px, dtype=np.float32)
    scale = np.abs(np.asarray(reference_scale_xy_m_per_pixel, dtype=np.float64))
    if query_points.shape != true_points.shape or query_points.shape[0] == 0:
        raise ValueError("Контрольные пары должны иметь одинаковую непустую форму")
    if scale.shape != (2,) or np.any(scale <= 0.0):
        raise ValueError("Метрический масштаб должен содержать два ненулевых числа")

    estimated_points = cv2.perspectiveTransform(
        query_points.reshape(1, -1, 2),
        np.asarray(homography_query_to_reference, dtype=np.float64),
    ).reshape(-1, 2)
    if not np.isfinite(estimated_points).all():
        raise ValueError("Гомография переносит контрольные точки в бесконечность")
    errors_meters = np.linalg.norm(
        (estimated_points.astype(np.float64) - true_points.astype(np.float64))
        * scale[np.newaxis, :],
        axis=1,
    )
    return DenseHomographyEvaluation(
        query_points_px=query_points,
        true_reference_points_px=true_points,
        estimated_reference_points_px=estimated_points.astype(np.float32),
        errors_meters=errors_meters,
        mean_error_meters=float(np.mean(errors_meters)),
        median_error_meters=float(np.median(errors_meters)),
        p95_error_meters=float(np.percentile(errors_meters, 95)),
        max_error_meters=float(np.max(errors_meters)),
    )
