# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Камеры и лучи для учебного моста Blender → NeRF/3DGS.

NeRF и формат Synthetic NeRF используют соглашение OpenGL/Blender: локальная
ось камеры +X направлена вправо, +Y вверх, +Z назад, а камера смотрит вдоль -Z.
Матрица хранит преобразование camera-to-world, то есть переносит координаты из
камеры в мировую систему.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Параметры идеальной pinhole-камеры в пикселях.

    Координаты ``cx`` и ``cy`` заданы в графическом соглашении: границы кадра
    идут от 0 до width/height, а центр верхнего левого пикселя равен
    ``(0.5, 0.5)``. Поэтому для симметричного кадра главная точка равна
    ``(width/2, height/2)``.
    """

    width_px: int
    height_px: int
    focal_x_px: float
    focal_y_px: float
    principal_x_px: float
    principal_y_px: float

    @property
    def horizontal_field_of_view_rad(self) -> float:
        """Возвращает горизонтальный угол обзора в радианах."""

        return float(2.0 * np.arctan(self.width_px / (2.0 * self.focal_x_px)))


def blender_horizontal_sensor_intrinsics(
    *,
    width_px: int,
    height_px: int,
    focal_length_mm: float,
    sensor_width_mm: float,
) -> PinholeIntrinsics:
    """Переводит настройки Blender в pinhole intrinsics.

    Формула соответствует используемым синтетическим сценам: perspective,
    горизонтальная подгонка сенсора, квадратные пиксели, нулевой camera shift
    и рендер 100%. Для другой подгонки сенсора или pixel aspect ratio эту
    функцию нельзя применять молча.
    """

    if width_px <= 0 or height_px <= 0:
        raise ValueError("Размер кадра должен быть положительным")
    if focal_length_mm <= 0.0 or sensor_width_mm <= 0.0:
        raise ValueError("Фокус и ширина сенсора должны быть положительными")
    focal_px = float(focal_length_mm * width_px / sensor_width_mm)
    return PinholeIntrinsics(
        width_px=width_px,
        height_px=height_px,
        focal_x_px=focal_px,
        focal_y_px=focal_px,
        principal_x_px=width_px / 2.0,
        principal_y_px=height_px / 2.0,
    )


def validate_camera_to_world(
    matrix_camera_to_world: NDArray[np.float64],
    *,
    tolerance: float = 1e-6,
) -> None:
    """Проверяет форму и жёсткость camera-to-world преобразования."""

    matrix = np.asarray(matrix_camera_to_world, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Camera-to-world должна быть конечной матрицей 4 x 4")
    if tolerance <= 0.0:
        raise ValueError("Допуск должен быть положительным")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=tolerance):
        raise ValueError("Последняя строка camera-to-world должна быть [0,0,0,1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=tolerance):
        raise ValueError("Поворот camera-to-world не ортонормален")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=tolerance):
        raise ValueError("Поворот camera-to-world должен иметь determinant +1")


def project_world_points_opengl(
    world_xyz: NDArray[np.float64],
    matrix_camera_to_world: NDArray[np.float64],
    intrinsics: PinholeIntrinsics,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Проецирует мировые точки в пиксельные центры формата NeRF.

    Вход ``world_xyz`` имеет форму ``N x 3`` в метрах. Выходные координаты
    имеют форму ``N x 2`` и отсчитываются от верхней левой границы кадра;
    центр пикселя (0, 0) расположен в (0.5, 0.5). Булев массив отмечает точки
    перед камерой, то есть с отрицательной локальной координатой Z.
    """

    points = np.asarray(world_xyz, dtype=np.float64)
    matrix = np.asarray(matrix_camera_to_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Мировые точки должны иметь конечную форму N x 3")
    validate_camera_to_world(matrix)
    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    camera_xyz = (np.linalg.inv(matrix) @ homogeneous.T).T[:, :3]
    depth = -camera_xyz[:, 2]
    in_front = depth > 0.0
    pixels = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    pixels[in_front, 0] = (
        intrinsics.focal_x_px * camera_xyz[in_front, 0] / depth[in_front]
        + intrinsics.principal_x_px
    )
    pixels[in_front, 1] = (
        intrinsics.principal_y_px
        - intrinsics.focal_y_px * camera_xyz[in_front, 1] / depth[in_front]
    )
    return pixels, in_front


def pixel_rays_world_opengl(
    pixel_centres_xy: NDArray[np.float64],
    matrix_camera_to_world: NDArray[np.float64],
    intrinsics: PinholeIntrinsics,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Строит мировые начала и единичные направления лучей пикселей.

    Все лучи одной pinhole-камеры начинаются в переносе camera-to-world.
    Направление сначала строится как ``[x, y, -1]`` в локальной системе,
    затем поворачивается в мировую систему. Перенос к направлению не
    применяется.
    """

    pixels = np.asarray(pixel_centres_xy, dtype=np.float64)
    matrix = np.asarray(matrix_camera_to_world, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
        raise ValueError("Пиксельные центры должны иметь конечную форму N x 2")
    validate_camera_to_world(matrix)
    local = np.column_stack(
        (
            (pixels[:, 0] - intrinsics.principal_x_px) / intrinsics.focal_x_px,
            -(pixels[:, 1] - intrinsics.principal_y_px) / intrinsics.focal_y_px,
            -np.ones(pixels.shape[0]),
        )
    )
    local /= np.linalg.norm(local, axis=1, keepdims=True)
    directions = (matrix[:3, :3] @ local.T).T
    origins = np.repeat(matrix[None, :3, 3], pixels.shape[0], axis=0)
    return origins, directions
