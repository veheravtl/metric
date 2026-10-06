# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Детерминированные поверхности для проверки границы одной гомографии.

Модуль не зависит от Blender. Одна и та же формула используется генератором
сцены, unit-тестами и оценщиком, поэтому параметры рельефа имеют явный
метрический смысл и могут быть проверены без дорогостоящего рендера.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

TerrainKind = Literal["flat", "plane", "gaussian", "wave"]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TerrainSpecification:
    """Параметры однозначной поверхности ``Z = f(X, Y)`` в метрах.

    ``plane`` задаётся уклонами по X и Y в градусах. Для ``gaussian`` знак
    амплитуды различает холм и яму, а sigma задаёт характерный радиус. Волна
    меняется вдоль X; её длина волны фиксирует пространственный масштаб, чтобы
    амплитуда не смешивалась с частотой рельефа.
    """

    kind: TerrainKind
    amplitude_m: float = 0.0
    sigma_m: float = 10.0
    wavelength_m: float = 40.0
    slope_x_degrees: float = 0.0
    slope_y_degrees: float = 0.0
    center_x_m: float = 0.0
    center_y_m: float = 0.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> TerrainSpecification:
        """Читает JSON-представление и сразу проверяет физические диапазоны."""

        specification = cls(
            kind=values["kind"],
            amplitude_m=float(values.get("amplitude_m", 0.0)),
            sigma_m=float(values.get("sigma_m", 10.0)),
            wavelength_m=float(values.get("wavelength_m", 40.0)),
            slope_x_degrees=float(values.get("slope_x_degrees", 0.0)),
            slope_y_degrees=float(values.get("slope_y_degrees", 0.0)),
            center_x_m=float(values.get("center_x_m", 0.0)),
            center_y_m=float(values.get("center_y_m", 0.0)),
        )
        specification.validate()
        return specification

    def validate(self) -> None:
        """Отклоняет неизвестную либо численно опасную поверхность."""

        if self.kind not in {"flat", "plane", "gaussian", "wave"}:
            raise ValueError(f"Неизвестный вид рельефа: {self.kind!r}")
        numeric_values = np.asarray(
            [
                self.amplitude_m,
                self.sigma_m,
                self.wavelength_m,
                self.slope_x_degrees,
                self.slope_y_degrees,
                self.center_x_m,
                self.center_y_m,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(numeric_values).all():
            raise ValueError("Параметры рельефа должны быть конечными")
        if self.kind == "gaussian" and self.sigma_m <= 0.0:
            raise ValueError("Радиус gaussian-рельефа должен быть положительным")
        if self.kind == "wave" and self.wavelength_m <= 0.0:
            raise ValueError("Длина волны должна быть положительной")
        if abs(self.slope_x_degrees) >= 80.0 or abs(self.slope_y_degrees) >= 80.0:
            raise ValueError("Уклон поверхности должен быть меньше 80 градусов")


def terrain_height_m(
    x_m: FloatArray | float,
    y_m: FloatArray | float,
    specification: TerrainSpecification,
) -> FloatArray:
    """Возвращает высоту Z для мировых координат X, Y в метрах."""

    specification.validate()
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    x, y = np.broadcast_arrays(x, y)

    if specification.kind == "flat":
        return np.zeros_like(x)
    if specification.kind == "plane":
        return np.tan(np.deg2rad(specification.slope_x_degrees)) * (
            x - specification.center_x_m
        ) + np.tan(np.deg2rad(specification.slope_y_degrees)) * (
            y - specification.center_y_m
        )
    if specification.kind == "gaussian":
        radius_squared = (x - specification.center_x_m) ** 2 + (
            y - specification.center_y_m
        ) ** 2
        return specification.amplitude_m * np.exp(
            -radius_squared / (2.0 * specification.sigma_m**2)
        )
    return specification.amplitude_m * np.sin(
        2.0 * np.pi * (x - specification.center_x_m) / specification.wavelength_m
    )


def terrain_vertices(
    *,
    width_m: float,
    height_m: float,
    columns: int,
    rows: int,
    specification: TerrainSpecification,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    """Строит вершины, UV и четырёхугольные грани регулярной mesh-сетки."""

    if width_m <= 0.0 or height_m <= 0.0:
        raise ValueError("Размеры поверхности должны быть положительными")
    if columns < 2 or rows < 2:
        raise ValueError("Сетка рельефа должна иметь минимум 2 x 2 вершины")

    x = np.linspace(-width_m / 2.0, width_m / 2.0, columns)
    y = np.linspace(-height_m / 2.0, height_m / 2.0, rows)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_z = terrain_height_m(grid_x, grid_y, specification)
    vertices = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))
    uv = np.column_stack(
        (
            (grid_x.ravel() + width_m / 2.0) / width_m,
            (grid_y.ravel() + height_m / 2.0) / height_m,
        )
    )
    faces = np.asarray(
        [
            (
                row * columns + column,
                row * columns + column + 1,
                (row + 1) * columns + column + 1,
                (row + 1) * columns + column,
            )
            for row in range(rows - 1)
            for column in range(columns - 1)
        ],
        dtype=np.int64,
    )
    return vertices, uv, faces


def triangle_centroid_samples(
    vertices: FloatArray,
    faces: NDArray[np.int64],
    *,
    stride: int,
) -> FloatArray:
    """Выбирает центроиды треугольников, лежащие на реальной mesh."""

    if stride <= 0:
        raise ValueError("Шаг выборки должен быть положительным")
    selected = np.asarray(faces, dtype=np.int64)[::stride]
    if selected.size == 0:
        raise ValueError("Сетка не содержит точек для контрольной выборки")
    return np.mean(np.asarray(vertices, dtype=np.float64)[selected[:, :3]], axis=1)
