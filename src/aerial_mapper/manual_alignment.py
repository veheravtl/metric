"""Ручная оценка проективного совмещения кадра дрона с картой."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

FloatPoints = NDArray[np.float64]
Homography = NDArray[np.float64]


class ManualAlignmentFailure(RuntimeError):
    """Ручные точки не задают конечное и проверяемое преобразование."""


@dataclass(frozen=True)
class ManualAlignmentResult:
    """Гомография и диагностика согласованности ручных соответствий.

    ``fit_errors_map_px`` измеряет остаток на тех же точках, по которым строилась
    итоговая матрица. При четырёх парах этот остаток почти неизбежно будет
    нулевым и потому не является независимой проверкой.

    ``leave_one_out_errors_map_px`` строится честнее: каждая точка по очереди
    исключается, матрица оценивается по остальным, затем исключённая точка
    прогнозируется. Такая проверка доступна начиная с пяти пар. Значение NaN
    означает, что оставшиеся точки в конкретном испытании выродились.
    """

    homography_frame_to_map: Homography
    frame_points_px: FloatPoints
    map_points_px: FloatPoints
    fit_errors_map_px: NDArray[np.float64]
    leave_one_out_errors_map_px: NDArray[np.float64]
    frame_coverage_fraction: float
    map_coverage_fraction: float
    projected_frame_corners_map_px: FloatPoints

    @property
    def fit_rmse_map_px(self) -> float:
        """Среднеквадратичная ошибка подгонки в пикселях карты."""

        return float(np.sqrt(np.mean(self.fit_errors_map_px**2)))

    @property
    def fit_max_map_px(self) -> float:
        """Наибольшая ошибка подгонки в пикселях карты."""

        return float(np.max(self.fit_errors_map_px))

    @property
    def leave_one_out_median_map_px(self) -> float | None:
        """Медианная независимая ошибка либо ``None`` при четырёх точках."""

        finite = self.leave_one_out_errors_map_px[
            np.isfinite(self.leave_one_out_errors_map_px)
        ]
        if finite.size == 0:
            return None
        return float(np.median(finite))

    @property
    def leave_one_out_max_map_px(self) -> float | None:
        """Худшая независимая ошибка либо ``None`` при четырёх точках."""

        finite = self.leave_one_out_errors_map_px[
            np.isfinite(self.leave_one_out_errors_map_px)
        ]
        if finite.size == 0:
            return None
        return float(np.max(finite))


@dataclass(frozen=True)
class ManualAlignmentQuality:
    """Решение защитного порога и диагностика одной перепутанной пары.

    Пределы заданы в пикселях карты и масштабируются с её диагональю. Если
    удаление одной пары возвращает ошибку под предел и улучшает её минимум
    втрое, индекс этой пары сообщается оператору как вероятная причина.
    """

    is_acceptable: bool
    fit_rmse_limit_map_px: float
    fit_max_limit_map_px: float
    suspected_outlier_index: int | None
    fit_rmse_without_suspected_map_px: float | None
    fit_max_without_suspected_map_px: float | None


def estimate_manual_alignment(
    frame_points_px: NDArray[np.floating] | list[tuple[float, float]],
    map_points_px: NDArray[np.floating] | list[tuple[float, float]],
    *,
    frame_size_pixels: tuple[int, int],
    map_size_pixels: tuple[int, int],
) -> ManualAlignmentResult:
    """Оценить преобразование плоскости кадра в плоскость карты.

    Размеры задаются в порядке ``(width, height)``. Точки должны относиться к
    поверхности земли. Крыши, деревья и другие высотные объекты нарушают модель
    одной плоскости и способны дать систематическую ошибку даже при аккуратных
    кликах.
    """

    frame = _validated_points(frame_points_px, name="Точки кадра")
    map_points = _validated_points(map_points_px, name="Точки карты")
    if frame.shape != map_points.shape:
        raise ManualAlignmentFailure(
            "Число точек кадра и карты должно совпадать попарно"
        )

    frame_width, frame_height = _validated_size(frame_size_pixels, name="Размер кадра")
    map_width, map_height = _validated_size(map_size_pixels, name="Размер карты")
    _validate_inside(frame, width=frame_width, height=frame_height, name="Кадр")
    _validate_inside(
        map_points,
        width=map_width,
        height=map_height,
        name="Карта",
    )

    homography = _find_homography(frame, map_points)
    predicted_map_points = _transform_points(frame, homography)
    fit_errors = np.linalg.norm(predicted_map_points - map_points, axis=1)
    leave_one_out_errors = _leave_one_out_errors(frame, map_points)

    frame_corners = np.asarray(
        [
            [0.0, 0.0],
            [frame_width - 1.0, 0.0],
            [frame_width - 1.0, frame_height - 1.0],
            [0.0, frame_height - 1.0],
        ],
        dtype=np.float64,
    )
    projected_corners = _transform_points(frame_corners, homography)

    return ManualAlignmentResult(
        homography_frame_to_map=homography,
        frame_points_px=frame,
        map_points_px=map_points,
        fit_errors_map_px=fit_errors,
        leave_one_out_errors_map_px=leave_one_out_errors,
        frame_coverage_fraction=_point_coverage(frame, frame_width, frame_height),
        map_coverage_fraction=_point_coverage(map_points, map_width, map_height),
        projected_frame_corners_map_px=projected_corners,
    )


def assess_manual_alignment_quality(
    result: ManualAlignmentResult,
    *,
    frame_size_pixels: tuple[int, int],
    map_size_pixels: tuple[int, int],
) -> ManualAlignmentQuality:
    """Решить, безопасно ли показывать оператору рассчитанное наложение.

    Ошибка измеряется в пикселях карты. Допустимый средний остаток равен
    половине процента диагонали карты, а худший — одному проценту. Нижние
    пределы 3 и 6 пикселей не дают малой карте отвергать обычную погрешность
    ручного клика.

    Если результат плох, дополнительно проверяется простая гипотеза «ошибочна
    ровно одна пара». Эта диагностика только указывает оператору кандидата:
    подозрительная пара не удаляется и исправленная матрица не принимается
    автоматически.
    """

    frame_width, frame_height = _validated_size(frame_size_pixels, name="Размер кадра")
    map_width, map_height = _validated_size(map_size_pixels, name="Размер карты")
    map_diagonal_px = float(np.hypot(map_width - 1.0, map_height - 1.0))
    rmse_limit = max(3.0, 0.005 * map_diagonal_px)
    max_limit = max(6.0, 0.01 * map_diagonal_px)
    is_acceptable = (
        result.fit_rmse_map_px <= rmse_limit and result.fit_max_map_px <= max_limit
    )

    if is_acceptable or result.frame_points_px.shape[0] < 5:
        return ManualAlignmentQuality(
            is_acceptable=is_acceptable,
            fit_rmse_limit_map_px=rmse_limit,
            fit_max_limit_map_px=max_limit,
            suspected_outlier_index=None,
            fit_rmse_without_suspected_map_px=None,
            fit_max_without_suspected_map_px=None,
        )

    best_index: int | None = None
    best_result: ManualAlignmentResult | None = None
    point_count = result.frame_points_px.shape[0]
    for excluded_index in range(point_count):
        included = np.arange(point_count) != excluded_index
        try:
            candidate = estimate_manual_alignment(
                result.frame_points_px[included],
                result.map_points_px[included],
                frame_size_pixels=(frame_width, frame_height),
                map_size_pixels=(map_width, map_height),
            )
        except ManualAlignmentFailure:
            continue
        if (
            best_result is None
            or candidate.fit_rmse_map_px < best_result.fit_rmse_map_px
        ):
            best_index = excluded_index
            best_result = candidate

    suspected_index: int | None = None
    rmse_without_suspected: float | None = None
    max_without_suspected: float | None = None
    if best_result is not None:
        candidate_passes = (
            best_result.fit_rmse_map_px <= rmse_limit
            and best_result.fit_max_map_px <= max_limit
        )
        improves_at_least_threefold = (
            best_result.fit_rmse_map_px * 3.0 <= result.fit_rmse_map_px
        )
        if candidate_passes and improves_at_least_threefold:
            suspected_index = best_index
            rmse_without_suspected = best_result.fit_rmse_map_px
            max_without_suspected = best_result.fit_max_map_px

    return ManualAlignmentQuality(
        is_acceptable=False,
        fit_rmse_limit_map_px=rmse_limit,
        fit_max_limit_map_px=max_limit,
        suspected_outlier_index=suspected_index,
        fit_rmse_without_suspected_map_px=rmse_without_suspected,
        fit_max_without_suspected_map_px=max_without_suspected,
    )


def _validated_points(
    values: NDArray[np.floating] | list[tuple[float, float]], *, name: str
) -> FloatPoints:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ManualAlignmentFailure(f"{name} должны иметь форму (N, 2)")
    if points.shape[0] < 4:
        raise ManualAlignmentFailure(
            f"{name}: для гомографии нужны минимум четыре точки"
        )
    if not np.all(np.isfinite(points)):
        raise ManualAlignmentFailure(f"{name} содержат NaN или бесконечность")
    return points


def _validated_size(size: tuple[int, int], *, name: str) -> tuple[int, int]:
    width, height = size
    if width <= 1 or height <= 1:
        raise ManualAlignmentFailure(f"{name} должен быть больше одного пикселя")
    return int(width), int(height)


def _validate_inside(
    points: FloatPoints, *, width: int, height: int, name: str
) -> None:
    outside = (
        (points[:, 0] < 0.0)
        | (points[:, 0] > width - 1.0)
        | (points[:, 1] < 0.0)
        | (points[:, 1] > height - 1.0)
    )
    if np.any(outside):
        raise ManualAlignmentFailure(f"Хотя бы одна точка выходит за {name}")


def _find_homography(source: FloatPoints, destination: FloatPoints) -> Homography:
    homography, _ = cv2.findHomography(source, destination, method=0)
    if homography is None or not np.all(np.isfinite(homography)):
        raise ManualAlignmentFailure(
            "Точки не задают гомографию; проверьте, не лежат ли они на одной линии"
        )
    homography = homography.astype(np.float64)
    if np.isclose(homography[2, 2], 0.0):
        raise ManualAlignmentFailure("Получена вырожденная гомография")
    homography /= homography[2, 2]
    if np.linalg.matrix_rank(homography) < 3:
        raise ManualAlignmentFailure("Получена гомография неполного ранга")
    return homography


def _transform_points(points: FloatPoints, homography: Homography) -> FloatPoints:
    transformed = cv2.perspectiveTransform(
        points.reshape(1, -1, 2), homography
    ).reshape(-1, 2)
    if not np.all(np.isfinite(transformed)):
        raise ManualAlignmentFailure("Гомография переносит точки в бесконечность")
    return transformed.astype(np.float64)


def _leave_one_out_errors(
    frame_points: FloatPoints, map_points: FloatPoints
) -> NDArray[np.float64]:
    errors = np.full(frame_points.shape[0], np.nan, dtype=np.float64)
    if frame_points.shape[0] < 5:
        return errors

    for excluded_index in range(frame_points.shape[0]):
        included = np.arange(frame_points.shape[0]) != excluded_index
        try:
            homography = _find_homography(frame_points[included], map_points[included])
            predicted = _transform_points(
                frame_points[excluded_index : excluded_index + 1], homography
            )[0]
        except ManualAlignmentFailure:
            continue
        errors[excluded_index] = np.linalg.norm(predicted - map_points[excluded_index])
    return errors


def _point_coverage(points: FloatPoints, width: int, height: int) -> float:
    hull = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2))
    return float(cv2.contourArea(hull)) / float((width - 1) * (height - 1))
