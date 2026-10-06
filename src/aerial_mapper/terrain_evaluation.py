# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Независимая оценка плоской модели на общей видимой части рельефа."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

FloatPoints = NDArray[np.float64]


@dataclass(frozen=True)
class PointErrorSummary:
    """Распределение плановой ошибки X, Y в метрах."""

    count: int
    median_m: float
    p95_m: float
    maximum_m: float


def estimate_homography(source: FloatPoints, destination: FloatPoints) -> FloatPoints:
    """Оценивает least-squares гомографию по точным парам без RANSAC."""

    source_array = _points(source, name="Исходные точки", minimum=4)
    destination_array = _points(destination, name="Целевые точки", minimum=4)
    if source_array.shape != destination_array.shape:
        raise ValueError("Наборы соответствий должны иметь одинаковую форму")
    homography, _ = cv2.findHomography(source_array, destination_array, method=0)
    if homography is None or not np.isfinite(homography).all():
        raise ValueError("Точные соответствия не задали конечную гомографию")
    if np.isclose(homography[2, 2], 0.0):
        raise ValueError("Получена вырожденная гомография")
    result = homography.astype(np.float64)
    return result / result[2, 2]


def transform_points(points: FloatPoints, homography: FloatPoints) -> FloatPoints:
    """Применяет гомографию и явно отклоняет бесконечный результат."""

    values = _points(points, name="Преобразуемые точки", minimum=1)
    transformed = cv2.perspectiveTransform(
        values.reshape(1, -1, 2), np.asarray(homography, dtype=np.float64)
    ).reshape(-1, 2)
    if not np.isfinite(transformed).all():
        raise ValueError("Гомография перенесла точки в бесконечность")
    return transformed.astype(np.float64)


def point_error_summary(
    estimated_xy_m: FloatPoints, true_xy_m: FloatPoints
) -> PointErrorSummary:
    """Считает медиану, 95-й процентиль и максимум евклидовой ошибки."""

    estimated = _points(estimated_xy_m, name="Оценённые точки", minimum=1)
    truth = _points(true_xy_m, name="Истинные точки", minimum=1)
    if estimated.shape != truth.shape:
        raise ValueError("Оценённые и истинные точки должны иметь одинаковую форму")
    errors = np.linalg.norm(estimated - truth, axis=1)
    return PointErrorSummary(
        count=int(errors.size),
        median_m=float(np.median(errors)),
        p95_m=float(np.percentile(errors, 95)),
        maximum_m=float(np.max(errors)),
    )


def spatial_group_labels(
    world_xy_m: FloatPoints, *, group_count: int = 3
) -> NDArray[np.int64]:
    """Назначает пространственно перемешанные группы без случайного seed."""

    if group_count < 2:
        raise ValueError("Число пространственных групп должно быть не меньше двух")
    points = _points(world_xy_m, name="Мировые точки группировки", minimum=group_count)
    x_rank = np.unique(np.round(points[:, 0], decimals=9), return_inverse=True)[1]
    y_rank = np.unique(np.round(points[:, 1], decimals=9), return_inverse=True)[1]
    labels = (x_rank + (group_count - 1) * y_rank) % group_count
    if np.unique(labels).size != group_count:
        raise ValueError("Геометрия точек не заполнила все пространственные группы")
    return labels.astype(np.int64)


def spatial_checkerboard_split_indices(
    world_xy_m: FloatPoints,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Разделяет пространственную сетку на fit/evaluation точки.

    Делить только по номеру записи нельзя: если точки сохранены построчно,
    одна часть способна случайно оказаться на одной прямой и не задать
    гомографию. Для X и Y вычисляются ранги координат, после чего цвет
    шахматной клетки определяется чётностью суммы рангов.
    """

    points = _points(world_xy_m, name="Мировые точки разбиения", minimum=8)
    if points.shape[0] < 8:
        raise ValueError("Для независимой подгонки и оценки нужны минимум 8 точек")
    labels = spatial_group_labels(points, group_count=2)
    fit_mask = labels == 0
    indices = np.arange(points.shape[0], dtype=np.int64)
    fit = indices[fit_mask]
    evaluation = indices[~fit_mask]
    if fit.size < 4 or evaluation.size < 4:
        raise ValueError("Каждая часть должна содержать минимум четыре точки")
    return fit, evaluation


def select_grid_anchor_indices(
    world_xy_m: FloatPoints,
    *,
    columns: int,
    rows: int,
    inset_fraction: float = 0.1,
) -> NDArray[np.int64]:
    """Выбирает разнесённые реперы около узлов регулярной сетки.

    Функция работает только с координатами скрытой истины и нужна оценщику
    синтетического опыта. Она имитирует набор хорошо распределённых ориентиров
    на масштабной карте, не выдавая рабочему алгоритму остальные точки.
    """

    required = columns * rows
    if columns < 2 or rows < 2:
        raise ValueError("Сетка реперов должна иметь минимум 2 x 2 узла")
    if not 0.0 <= inset_fraction < 0.5:
        raise ValueError("Отступ сетки реперов должен лежать в диапазоне [0, 0.5)")
    points = _points(world_xy_m, name="Кандидаты реперов", minimum=required)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    span = maximum - minimum
    if np.any(span <= 0.0):
        raise ValueError("Реперы должны покрывать ненулевой диапазон X и Y")

    x_targets = np.linspace(
        minimum[0] + inset_fraction * span[0],
        maximum[0] - inset_fraction * span[0],
        columns,
    )
    y_targets = np.linspace(
        minimum[1] + inset_fraction * span[1],
        maximum[1] - inset_fraction * span[1],
        rows,
    )
    selected: list[int] = []
    used: set[int] = set()
    for y_target in y_targets:
        for x_target in x_targets:
            target = np.asarray([x_target, y_target], dtype=np.float64)
            nearest_first = np.argsort(np.linalg.norm(points - target, axis=1))
            index = next(
                (
                    int(candidate)
                    for candidate in nearest_first
                    if candidate not in used
                ),
                None,
            )
            if index is None:
                raise ValueError("Не удалось выбрать уникальные реперы")
            used.add(index)
            selected.append(index)
    return np.asarray(selected, dtype=np.int64)


def select_displacement_pairs(
    world_xy_m: FloatPoints,
    desired_delta_xy_m: NDArray[np.float64],
    *,
    maximum_count: int,
    tolerance_m: float,
) -> NDArray[np.int64]:
    """Выбирает пары цель/попадание с заданным направленным смещением.

    Индекс в первом столбце обозначает цель, во втором — попадание. В отличие
    от выбора обычных отрезков порядок нельзя сортировать: знак компонент
    является частью проверяемого сообщения оператору.
    """

    points = _points(world_xy_m, name="Кандидаты цели и попадания", minimum=2)
    delta = np.asarray(desired_delta_xy_m, dtype=np.float64)
    if delta.shape != (2,) or not np.isfinite(delta).all():
        raise ValueError("Желаемый вектор должен иметь форму (2,) и конечные числа")
    if np.linalg.norm(delta) == 0.0:
        raise ValueError("Желаемый вектор промаха не может быть нулевым")
    if maximum_count <= 0 or tolerance_m < 0.0:
        raise ValueError(
            "Число пар должно быть положительным, а допуск неотрицательным"
        )

    spatial_order = np.lexsort((points[:, 0], points[:, 1]))
    candidates: list[tuple[int, int]] = []
    for start in spatial_order:
        residuals = np.linalg.norm(points - (points[start] + delta), axis=1)
        stop = int(np.argmin(residuals))
        if stop == int(start) or residuals[stop] > tolerance_m:
            continue
        candidates.append((int(start), stop))
    if not candidates:
        raise ValueError("Не найдены пары с требуемым вектором промаха")
    if len(candidates) <= maximum_count:
        return np.asarray(candidates, dtype=np.int64)

    positions = np.linspace(
        0,
        len(candidates) - 1,
        num=maximum_count,
        dtype=np.int64,
    )
    return np.asarray([candidates[index] for index in positions], dtype=np.int64)


@dataclass(frozen=True)
class SegmentErrorSummary:
    """Максимальные ошибки независимых горизонтальных отрезков."""

    count: int
    maximum_absolute_m: float
    maximum_relative_percent: float


def select_segment_pairs(
    world_xy_m: FloatPoints,
    *,
    target_length_m: float = 6.0,
    tolerance_m: float = 1.5,
    maximum_count: int = 24,
) -> NDArray[np.int64]:
    """Детерминированно выбирает распределённые отрезки близкой длины.

    Выбор выполняется только по истинным X, Y до применения оцениваемой
    гомографии. Для каждого равномерно взятого начала ищется ближайшая к
    целевой длине конечная точка. Так итог не зависит от ошибки алгоритма.
    """

    points = _points(world_xy_m, name="Точки отрезков", minimum=2)
    if target_length_m <= 0.0 or tolerance_m <= 0.0 or maximum_count <= 0:
        raise ValueError("Параметры выбора отрезков должны быть положительными")

    starts = np.linspace(
        0,
        points.shape[0] - 1,
        num=min(maximum_count * 3, points.shape[0]),
        dtype=np.int64,
    )
    pairs: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for start in starts:
        distances = np.linalg.norm(points - points[start], axis=1)
        valid = np.flatnonzero(
            (distances >= target_length_m - tolerance_m)
            & (distances <= target_length_m + tolerance_m)
        )
        if valid.size == 0:
            continue
        stop = int(valid[np.argmin(np.abs(distances[valid] - target_length_m))])
        pair = (min(int(start), stop), max(int(start), stop))
        if pair[0] == pair[1] or pair in used:
            continue
        used.add(pair)
        pairs.append(pair)
        if len(pairs) >= maximum_count:
            break
    if not pairs:
        raise ValueError("Не удалось выбрать отрезки требуемой длины")
    return np.asarray(pairs, dtype=np.int64)


def segment_error_summary(
    estimated_xy_m: FloatPoints,
    true_xy_m: FloatPoints,
    pairs: NDArray[np.int64],
) -> SegmentErrorSummary:
    """Сравнивает длины заранее выбранных отрезков в плане X, Y."""

    estimated = _points(estimated_xy_m, name="Оценённые точки", minimum=2)
    truth = _points(true_xy_m, name="Истинные точки", minimum=2)
    pair_indices = np.asarray(pairs, dtype=np.int64)
    if estimated.shape != truth.shape:
        raise ValueError("Оценённые и истинные точки должны иметь одинаковую форму")
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError("Индексы отрезков должны иметь форму (N, 2)")
    if pair_indices.size == 0 or np.any(pair_indices < 0):
        raise ValueError("Нужен непустой набор неотрицательных индексов")
    if int(np.max(pair_indices)) >= truth.shape[0]:
        raise ValueError("Индекс отрезка выходит за пределы массива точек")

    true_lengths = np.linalg.norm(
        truth[pair_indices[:, 1]] - truth[pair_indices[:, 0]], axis=1
    )
    estimated_lengths = np.linalg.norm(
        estimated[pair_indices[:, 1]] - estimated[pair_indices[:, 0]], axis=1
    )
    if np.any(true_lengths <= 0.0):
        raise ValueError("Истинная длина отрезка должна быть положительной")
    absolute = np.abs(estimated_lengths - true_lengths)
    relative = absolute / true_lengths * 100.0
    return SegmentErrorSummary(
        count=int(pair_indices.shape[0]),
        maximum_absolute_m=float(np.max(absolute)),
        maximum_relative_percent=float(np.max(relative)),
    )


def _points(values: FloatPoints, *, name: str, minimum: int) -> FloatPoints:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < minimum:
        raise ValueError(f"{name} должны иметь форму (N, 2), N >= {minimum}")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} содержат NaN или бесконечность")
    return points
