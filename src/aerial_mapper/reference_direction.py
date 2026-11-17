# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Проекция локального направления карты в короткий луч видеокадра.

Гомография надёжно связывает только видимую область Repeat-кадра с небольшим
участком карты. Поэтому удалённый репер нельзя преобразовывать обратной
гомографией напрямую: это была бы экстраполяция на тысячи пикселей. Вместо
этого модуль переносит две соседние точки карты, получает локальное направление
в кадре и обрезает луч точной границей изображения.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class ReferenceDirectionProjectionFailure(ValueError):
    """Ожидаемый отказ при вырожденной или противоречивой геометрии луча."""


@dataclass(frozen=True)
class FrameDirectionSegment:
    """Видимый фрагмент луча в пикселях кадра: начало и точка на границе."""

    origin_px: tuple[float, float]
    boundary_px: tuple[float, float]


def project_map_direction_to_frame_edge(
    origin_frame_px: tuple[float, float],
    origin_map_px: tuple[float, float] | NDArray[np.floating],
    map_direction: tuple[float, float] | NDArray[np.floating],
    homography_frame_to_map: NDArray[np.floating],
    *,
    frame_width_pixels: int,
    frame_height_pixels: int,
) -> FrameDirectionSegment:
    """Продолжить направление от точки кадра ровно до его первой границы.

    Матрица homography_frame_to_map переводит пиксели кадра в пиксели карты.
    map_direction задаёт только направление и не обязано иметь единичную длину.
    Для переноса направления используется локальная пробная точка на расстоянии
    одного пикселя карты от origin_map_px. Сам далёкий репер в преобразование
    не попадает.
    """

    width = int(frame_width_pixels)
    height = int(frame_height_pixels)
    if width <= 1 or height <= 1:
        raise ReferenceDirectionProjectionFailure(
            "Для направляющей нужен кадр размером не менее 2×2 px"
        )

    frame_origin = _finite_pair(origin_frame_px, name="Начало луча в кадре")
    if not (
        0.0 <= frame_origin[0] <= width - 1.0 and 0.0 <= frame_origin[1] <= height - 1.0
    ):
        raise ReferenceDirectionProjectionFailure(
            "Начало направляющей лежит за границей кадра"
        )
    map_origin = _finite_pair(origin_map_px, name="Начало луча на карте")
    direction = _finite_pair(map_direction, name="Направление на карте")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ReferenceDirectionProjectionFailure(
            "Нельзя перенести нулевое направление на репер"
        )
    direction /= direction_norm

    homography = np.asarray(homography_frame_to_map, dtype=np.float64)
    if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
        raise ReferenceDirectionProjectionFailure(
            "Гомография кадр → карта должна быть конечной матрицей 3×3"
        )
    try:
        map_to_frame = np.linalg.inv(homography)
    except np.linalg.LinAlgError as error:
        raise ReferenceDirectionProjectionFailure(
            "Гомография кадр → карта вырождена"
        ) from error

    # Обе точки находятся рядом в карте. Их разность после обратной проекции —
    # локальное направление в перспективе кадра, а не попытка показать репер.
    map_points = np.vstack((map_origin, map_origin + direction))
    homogeneous_map_points = np.column_stack((map_points, np.ones(2, dtype=np.float64)))
    homogeneous_frame_points = (map_to_frame @ homogeneous_map_points.T).T
    denominators = homogeneous_frame_points[:, 2]
    if np.any(np.abs(denominators) <= 1e-12):
        raise ReferenceDirectionProjectionFailure(
            "Локальное направление попало на горизонт проективного преобразования"
        )
    frame_points = homogeneous_frame_points[:, :2] / denominators[:, np.newaxis]
    frame_direction = frame_points[1] - frame_points[0]
    frame_direction_norm = float(np.linalg.norm(frame_direction))
    if not np.all(np.isfinite(frame_direction)) or frame_direction_norm <= 1e-12:
        raise ReferenceDirectionProjectionFailure(
            "Гомография схлопнула локальное направление на репер"
        )
    frame_direction /= frame_direction_norm

    x, y = float(frame_origin[0]), float(frame_origin[1])
    dx, dy = float(frame_direction[0]), float(frame_direction[1])
    boundary_times: list[float] = []
    if abs(dx) > 1e-12:
        boundary_x = float(width - 1) if dx > 0.0 else 0.0
        boundary_times.append((boundary_x - x) / dx)
    if abs(dy) > 1e-12:
        boundary_y = float(height - 1) if dy > 0.0 else 0.0
        boundary_times.append((boundary_y - y) / dy)
    positive_times = [value for value in boundary_times if value >= 0.0]
    if not positive_times:
        raise ReferenceDirectionProjectionFailure(
            "Не удалось пересечь направляющей границу кадра"
        )
    distance_to_boundary = min(positive_times)
    if distance_to_boundary <= 1e-9:
        raise ReferenceDirectionProjectionFailure(
            "Начало луча уже находится на выходной границе кадра"
        )

    boundary_x = min(max(x + dx * distance_to_boundary, 0.0), float(width - 1))
    boundary_y = min(max(y + dy * distance_to_boundary, 0.0), float(height - 1))
    return FrameDirectionSegment(
        origin_px=(x, y),
        boundary_px=(boundary_x, boundary_y),
    )


def _finite_pair(
    value: tuple[float, float] | NDArray[np.floating],
    *,
    name: str,
) -> NDArray[np.float64]:
    """Преобразовать пару в вектор float64 и дать понятный отказ."""

    pair = np.array(value, dtype=np.float64, copy=True)
    if pair.shape != (2,) or not np.all(np.isfinite(pair)):
        raise ReferenceDirectionProjectionFailure(
            f"{name} должно быть конечной парой (x, y)"
        )
    return pair
