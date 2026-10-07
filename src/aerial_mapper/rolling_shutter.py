# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Плоскостная модель построчного считывания движущейся камерой.

Каждая строка получает собственную pinhole-позу. Номинальная camera-to-world
матрица относится к середине чтения кадра, а движение задаёт полную разницу
поз между верхней и нижней строками. Модуль намеренно не моделирует выдержку,
смаз, рельеф или объёмные объекты.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from aerial_mapper.radiance_camera import (
    PinholeIntrinsics,
    validate_camera_to_world,
)


@dataclass(frozen=True)
class RollingShutterMotion:
    """Полное локальное движение камеры от верхней строки к нижней."""

    translation_local_m: tuple[float, float, float]
    rotation_local_deg: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Запрещает нечисловую или бесконечную траекторию."""

        translation = np.asarray(self.translation_local_m, dtype=np.float64)
        rotation = np.asarray(self.rotation_local_deg, dtype=np.float64)
        if translation.shape != (3,) or rotation.shape != (3,):
            raise ValueError("Движение должно содержать по три компоненты")
        if not np.isfinite(translation).all() or not np.isfinite(rotation).all():
            raise ValueError("Движение должно быть конечным")


@dataclass(frozen=True)
class RollingShutterWarp:
    """Результат перепроекции и диагностическая карта координат."""

    image_rgb: NDArray[np.uint8]
    valid_mask: NDArray[np.bool_]
    source_xy: NDArray[np.float32]
    maximum_displacement_px: float


@dataclass(frozen=True)
class RollingShutterProjection:
    """Самосогласованная проекция мировых точек в построчный кадр."""

    pixel_xy: NDArray[np.float64]
    in_front: NDArray[np.bool_]
    converged: NDArray[np.bool_]
    row_residual_px: NDArray[np.float64]
    displacement_from_centre_pose_px: NDArray[np.float64]


def scan_fraction(
    row_y: NDArray[np.float64] | float, height_px: int
) -> NDArray[np.float64]:
    """Переводит координату центра строки во время от −0,5 до +0,5.

    В массивах OpenCV центр первой строки имеет координату 0, поэтому добавка
    0,5 переводит индекс центра в долю полной высоты сенсора.
    """

    if height_px <= 0:
        raise ValueError("Высота кадра должна быть положительной")
    return (np.asarray(row_y, dtype=np.float64) + 0.5) / height_px - 0.5


def pose_at_scan_fraction(
    matrix_camera_to_world: NDArray[np.float64],
    motion: RollingShutterMotion,
    fraction: float,
) -> NDArray[np.float64]:
    """Возвращает позу строки на линейной локальной траектории SE(3)."""

    centre = np.asarray(matrix_camera_to_world, dtype=np.float64)
    validate_camera_to_world(centre)
    if not np.isfinite(fraction):
        raise ValueError("Доля чтения должна быть конечной")
    translation = np.asarray(motion.translation_local_m, dtype=np.float64)
    rotation_rad = np.deg2rad(np.asarray(motion.rotation_local_deg, dtype=np.float64))
    rotation_increment, _ = cv2.Rodrigues(rotation_rad * float(fraction))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = centre[:3, :3] @ rotation_increment
    result[:3, 3] = centre[:3, 3] + centre[:3, :3] @ (translation * float(fraction))
    return result


def _project_from_fraction(
    world_xyz_m: NDArray[np.float64],
    fractions: NDArray[np.float64],
    matrix_camera_to_world: NDArray[np.float64],
    intrinsics: PinholeIntrinsics,
    motion: RollingShutterMotion,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Векторно проецирует точки, когда доля чтения каждой уже известна."""

    points = np.asarray(world_xyz_m, dtype=np.float64)
    centre = np.asarray(matrix_camera_to_world, dtype=np.float64)
    base_camera = (centre[:3, :3].T @ (points - centre[:3, 3]).T).T
    translation = np.asarray(motion.translation_local_m, dtype=np.float64)
    shifted = base_camera - fractions[:, None] * translation[None, :]

    rotation_vector = np.deg2rad(
        np.asarray(motion.rotation_local_deg, dtype=np.float64)
    )
    rotated = np.empty_like(shifted)
    for index, (value, vector) in enumerate(zip(fractions, shifted, strict=True)):
        increment, _ = cv2.Rodrigues(rotation_vector * float(value))
        rotated[index] = increment.T @ vector

    depth = -rotated[:, 2]
    in_front = depth > 0.0
    pixels = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    pixels[in_front, 0] = (
        intrinsics.focal_x_px * rotated[in_front, 0] / depth[in_front]
        + intrinsics.principal_x_px
    )
    pixels[in_front, 1] = (
        intrinsics.principal_y_px
        - intrinsics.focal_y_px * rotated[in_front, 1] / depth[in_front]
    )
    return pixels, in_front


def project_world_points_rolling_shutter(
    world_xyz_m: NDArray[np.float64],
    matrix_camera_to_world: NDArray[np.float64],
    intrinsics: PinholeIntrinsics,
    motion: RollingShutterMotion,
    *,
    maximum_iterations: int = 30,
    tolerance_px: float = 1e-9,
) -> RollingShutterProjection:
    """Решает неявное условие: поза точки задаётся её итоговой строкой.

    Сначала точка проецируется центральной позой. Затем её строка задаёт новую
    позу, из которой точка проецируется снова. Итерации продолжаются до
    согласования номера строки и времени её чтения.
    """

    points = np.asarray(world_xyz_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Мировые точки должны иметь конечную форму N x 3")
    if maximum_iterations <= 0 or tolerance_px <= 0.0:
        raise ValueError("Параметры итераций должны быть положительными")
    validate_camera_to_world(matrix_camera_to_world)

    zero = np.zeros(points.shape[0], dtype=np.float64)
    centre_pixels, centre_front = _project_from_fraction(
        points,
        zero,
        matrix_camera_to_world,
        intrinsics,
        motion,
    )
    rows = centre_pixels[:, 1].copy()
    converged = np.zeros(points.shape[0], dtype=bool)
    residual = np.full(points.shape[0], np.inf, dtype=np.float64)
    pixels = centre_pixels.copy()
    in_front = centre_front.copy()

    for _ in range(maximum_iterations):
        fractions = scan_fraction(rows, intrinsics.height_px)
        projected, current_front = _project_from_fraction(
            points,
            fractions,
            matrix_camera_to_world,
            intrinsics,
            motion,
        )
        residual = np.abs(projected[:, 1] - rows)
        pixels = projected
        in_front &= current_front
        converged |= residual <= tolerance_px
        rows = projected[:, 1]
        if np.all(converged | ~in_front):
            break

    final_fractions = scan_fraction(rows, intrinsics.height_px)
    pixels, final_front = _project_from_fraction(
        points,
        final_fractions,
        matrix_camera_to_world,
        intrinsics,
        motion,
    )
    residual = np.abs(pixels[:, 1] - rows)
    in_front &= final_front
    converged = in_front & (residual <= tolerance_px)
    displacement = np.linalg.norm(pixels - centre_pixels, axis=1)
    return RollingShutterProjection(
        pixel_xy=pixels,
        in_front=in_front,
        converged=converged,
        row_residual_px=residual,
        displacement_from_centre_pose_px=displacement,
    )


def warp_planar_rolling_shutter(
    image_rgb: NDArray[np.uint8],
    matrix_camera_to_world: NDArray[np.float64],
    intrinsics: PinholeIntrinsics,
    motion: RollingShutterMotion,
    *,
    plane_z_m: float = 0.0,
) -> RollingShutterWarp:
    """Перепроецирует плоский RGB через отдельную позу каждой строки.

    Для выходного пикселя строится луч текущей строки, пересекается с плоскостью
    ``Z=plane_z_m``, а найденная точка проецируется в центральную исходную
    камеру. Поэтому карта используется в обратном направлении и не оставляет
    дыр между соседними строками.
    """

    image = np.asarray(image_rgb)
    if image.shape != (intrinsics.height_px, intrinsics.width_px, 3):
        raise ValueError("RGB не совпадает с размерами intrinsics")
    if image.dtype != np.uint8:
        raise ValueError("RGB должен иметь dtype uint8")
    centre = np.asarray(matrix_camera_to_world, dtype=np.float64)
    validate_camera_to_world(centre)

    height, width = image.shape[:2]
    translation = np.asarray(motion.translation_local_m, dtype=np.float64)
    rotation = np.asarray(motion.rotation_local_deg, dtype=np.float64)
    if np.count_nonzero(translation) == 0 and np.count_nonzero(rotation) == 0:
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        return RollingShutterWarp(
            image_rgb=image.copy(),
            valid_mask=np.ones((height, width), dtype=bool),
            source_xy=np.stack((grid_x, grid_y), axis=2),
            maximum_displacement_px=0.0,
        )

    source_xy = np.full((height, width, 2), -1.0, dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)
    x_values = np.arange(width, dtype=np.float64)
    maximum_displacement = 0.0

    for row in range(height):
        fraction = float(scan_fraction(float(row), height))
        pose = pose_at_scan_fraction(centre, motion, fraction)
        local_rays = np.column_stack(
            (
                (x_values - intrinsics.principal_x_px) / intrinsics.focal_x_px,
                np.full(
                    width,
                    -(row - intrinsics.principal_y_px) / intrinsics.focal_y_px,
                ),
                -np.ones(width),
            )
        )
        world_directions = (pose[:3, :3] @ local_rays.T).T
        denominators = world_directions[:, 2]
        scale = np.divide(
            plane_z_m - pose[2, 3],
            denominators,
            out=np.full(width, np.nan),
            where=np.abs(denominators) > 1e-12,
        )
        world = pose[:3, 3] + scale[:, None] * world_directions
        centre_camera = (centre[:3, :3].T @ (world - centre[:3, 3]).T).T
        depth = -centre_camera[:, 2]
        source_x = (
            intrinsics.focal_x_px * centre_camera[:, 0] / depth
            + intrinsics.principal_x_px
        )
        source_y = (
            intrinsics.principal_y_px
            - intrinsics.focal_y_px * centre_camera[:, 1] / depth
        )
        row_valid = (
            np.isfinite(source_x)
            & np.isfinite(source_y)
            & (scale > 0.0)
            & (depth > 0.0)
            & (source_x >= 0.0)
            & (source_x <= width - 1)
            & (source_y >= 0.0)
            & (source_y <= height - 1)
        )
        source_xy[row, :, 0] = source_x.astype(np.float32)
        source_xy[row, :, 1] = source_y.astype(np.float32)
        valid[row] = row_valid
        displacement = np.hypot(source_x - x_values, source_y - row)
        if np.any(row_valid):
            maximum_displacement = max(
                maximum_displacement, float(np.max(displacement[row_valid]))
            )

    warped = cv2.remap(
        image,
        source_xy[:, :, 0],
        source_xy[:, :, 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return RollingShutterWarp(
        image_rgb=warped,
        valid_mask=valid,
        source_xy=source_xy,
        maximum_displacement_px=maximum_displacement,
    )
