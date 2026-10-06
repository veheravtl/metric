# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Детерминированные классы синтетической текстуры поверхности земли.

Модуль не зависит от Blender. Поэтому свойства исходной текстуры можно
проверить быстрыми тестами до дорогого рендера, а Blender получает уже готовый
RGB-массив с диапазоном значений 0..1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

GroundTextureClass = Literal[
    "rich_irregular",
    "low_detail",
    "periodic_rows",
]
SUPPORTED_GROUND_TEXTURE_CLASSES: tuple[GroundTextureClass, ...] = (
    "rich_irregular",
    "low_detail",
    "periodic_rows",
)


def _validate_dimensions(specification: Mapping[str, Any]) -> tuple[int, int]:
    """Проверяет размер и возвращает ширину и высоту в пикселях."""

    width = int(specification["width_pixels"])
    height = int(specification["height_pixels"])
    if width < 64 or height < 64:
        raise ValueError("Метрическая текстура должна быть не меньше 64 x 64")
    return width, height


def _rich_irregular_texture(
    width: int,
    height: int,
    *,
    random_generator: np.random.Generator,
) -> NDArray[np.float32]:
    """Воспроизводит богатую непериодическую текстуру прежних G1--G12."""

    y_coordinates, x_coordinates = np.indices((height, width), dtype=np.float32)
    x_fraction = x_coordinates / max(width - 1, 1)
    y_fraction = y_coordinates / max(height - 1, 1)

    background = np.empty((height, width, 3), dtype=np.float32)
    background[..., 0] = 0.18 + 0.07 * np.sin(
        17.0 * x_fraction + 3.0 * y_fraction
    )
    background[..., 1] = 0.34 + 0.09 * np.sin(
        11.0 * y_fraction - 5.0 * x_fraction
    )
    background[..., 2] = 0.14 + 0.05 * np.cos(
        13.0 * (x_fraction + y_fraction)
    )

    cell_size = 12
    grid_height = math.ceil(height / cell_size)
    grid_width = math.ceil(width / cell_size)
    cell_noise = random_generator.normal(
        0.0,
        0.055,
        (grid_height, grid_width, 1),
    )
    cell_noise = np.repeat(
        np.repeat(cell_noise, cell_size, axis=0),
        cell_size,
        axis=1,
    )
    background += cell_noise[:height, :width]
    image_rgb = np.clip(background, 0.03, 0.92)

    horizontal_road = np.abs(y_fraction - 0.37) < 0.055
    diagonal_road = np.abs(y_fraction - (0.83 * x_fraction + 0.05)) < 0.035
    image_rgb[horizontal_road | diagonal_road] = (0.115, 0.125, 0.14)
    horizontal_marking = (np.abs(y_fraction - 0.37) < 0.004) & (
        (x_coordinates.astype(np.int32) // 55) % 2 == 0
    )
    diagonal_distance = np.abs(y_fraction - (0.83 * x_fraction + 0.05))
    diagonal_marking = (diagonal_distance < 0.003) & (
        ((x_coordinates + y_coordinates).astype(np.int32) // 70) % 2 == 0
    )
    image_rgb[horizontal_marking | diagonal_marking] = (0.92, 0.78, 0.18)

    for index in range(34):
        center_x = int(random_generator.integers(35, width - 35))
        center_y = int(random_generator.integers(35, height - 35))
        half_width = int(random_generator.integers(9, 31))
        half_height = int(random_generator.integers(8, 27))
        color = random_generator.uniform(0.12, 0.92, size=3)
        x_start = max(0, center_x - half_width)
        x_stop = min(width, center_x + half_width)
        y_start = max(0, center_y - half_height)
        y_stop = min(height, center_y + half_height)
        image_rgb[y_start:y_stop, x_start:x_stop] = color
        border = 3 + index % 4
        image_rgb[y_start : min(y_stop, y_start + border), x_start:x_stop] = 0.96
        image_rgb[max(y_start, y_stop - border) : y_stop, x_start:x_stop] = 0.04

    return image_rgb.astype(np.float32, copy=False)


def _low_detail_texture(
    width: int,
    height: int,
    *,
    random_generator: np.random.Generator,
) -> NDArray[np.float32]:
    """Создаёт гладкую поверхность без резких локальных ориентиров.

    Несколько широких волн и пятен сохраняют реалистичную неравномерность
    яркости, но не добавляют искусственных углов. Такой класс проверяет,
    способен ли gate честно отказаться при нехватке локальных признаков.
    """

    y_coordinates, x_coordinates = np.indices((height, width), dtype=np.float32)
    x_fraction = x_coordinates / max(width - 1, 1)
    y_fraction = y_coordinates / max(height - 1, 1)
    phase = random_generator.uniform(0.0, 2.0 * np.pi, size=3)

    smooth_field = (
        0.018 * np.sin(2.0 * np.pi * (0.72 * x_fraction + 0.31 * y_fraction) + phase[0])
        + 0.012
        * np.cos(2.0 * np.pi * (-0.28 * x_fraction + 0.61 * y_fraction) + phase[1])
        + 0.008
        * np.sin(2.0 * np.pi * (0.19 * x_fraction - 0.23 * y_fraction) + phase[2])
    )
    for _ in range(4):
        center_x, center_y = random_generator.uniform(0.1, 0.9, size=2)
        sigma = float(random_generator.uniform(0.16, 0.29))
        amplitude = float(random_generator.uniform(-0.018, 0.018))
        squared_radius = (
            (x_fraction - center_x) ** 2 + (y_fraction - center_y) ** 2
        )
        smooth_field += amplitude * np.exp(
            -squared_radius / (2.0 * sigma * sigma)
        )

    image_rgb = np.empty((height, width, 3), dtype=np.float32)
    image_rgb[..., 0] = 0.205 + 0.75 * smooth_field
    image_rgb[..., 1] = 0.335 + smooth_field
    image_rgb[..., 2] = 0.175 + 0.55 * smooth_field
    return np.clip(image_rgb, 0.03, 0.92)


def _periodic_rows_texture(
    width: int,
    height: int,
    *,
    random_generator: np.random.Generator,
    specification: Mapping[str, Any],
) -> NDArray[np.float32]:
    """Создаёт регулярные ряды одинаковых прямоугольных элементов.

    Резких углов здесь много, но локальные окрестности повторяются. Это
    принципиально другой трудный случай, чем гладкая земля: признаков может
    быть достаточно, однако их соответствия неоднозначны.
    """

    y_coordinates, x_coordinates = np.indices((height, width), dtype=np.float32)
    angle_degrees = float(specification.get("row_angle_degrees", 11.0))
    angle = np.deg2rad(angle_degrees)
    along = x_coordinates * np.cos(angle) + y_coordinates * np.sin(angle)
    across = -x_coordinates * np.sin(angle) + y_coordinates * np.cos(angle)

    along_period = float(specification.get("along_period_pixels", 48.0))
    row_period = float(specification.get("row_period_pixels", 36.0))
    if along_period < 12.0 or row_period < 12.0:
        raise ValueError("Периоды рядов должны быть не меньше 12 пикселей")

    phase_along = float(random_generator.uniform(0.0, along_period))
    phase_across = float(random_generator.uniform(0.0, row_period))
    local_along = np.mod(along + phase_along, along_period)
    local_across = np.mod(across + phase_across, row_period)

    rows = np.abs(local_across - row_period / 2.0) <= 4.5
    repeated_blocks = rows & (
        np.abs(local_along - along_period / 2.0) <= 13.0
    )
    narrow_centres = rows & (
        np.abs(local_along - along_period / 2.0) <= 2.0
    )

    image_rgb = np.empty((height, width, 3), dtype=np.float32)
    image_rgb[...] = (0.20, 0.34, 0.16)
    image_rgb[rows] = (0.15, 0.265, 0.115)
    image_rgb[repeated_blocks] = (0.29, 0.46, 0.20)
    image_rgb[narrow_centres] = (0.36, 0.52, 0.23)
    return image_rgb


def generate_ground_texture(
    specification: Mapping[str, Any],
    *,
    seed: int,
) -> NDArray[np.float32]:
    """Возвращает RGB-текстуру формы (height, width, 3) в диапазоне 0..1.

    Поле class отсутствовало в протоколах G1--G12. Его отсутствие намеренно
    означает rich_irregular, поэтому старые конфигурации сохраняют прежний
    пиксельный генератор и остаются воспроизводимыми.
    """

    width, height = _validate_dimensions(specification)
    texture_class = str(specification.get("class", "rich_irregular"))
    if texture_class not in SUPPORTED_GROUND_TEXTURE_CLASSES:
        supported = ", ".join(SUPPORTED_GROUND_TEXTURE_CLASSES)
        raise ValueError(
            f"Неизвестный класс текстуры {texture_class!r}; доступны: {supported}"
        )

    random_generator = np.random.default_rng(seed)
    if texture_class == "rich_irregular":
        return _rich_irregular_texture(
            width,
            height,
            random_generator=random_generator,
        )
    if texture_class == "low_detail":
        return _low_detail_texture(
            width,
            height,
            random_generator=random_generator,
        )
    return _periodic_rows_texture(
        width,
        height,
        random_generator=random_generator,
        specification=specification,
    )
