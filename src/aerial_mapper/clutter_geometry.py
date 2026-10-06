# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Детерминированное размещение выступающих объектов для синтетического гейта.

Модуль не импортирует Blender. Он отвечает только за геометрию в плане:
выбирает тип, размеры, положение и цвет объектов в метрах. Поэтому правила
размещения проверяются быстрыми unit-тестами, а Blender остаётся лишь
визуализатором уже зафиксированной сцены.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

ClutterKind = Literal["rock", "stump", "shrub"]


@dataclass(frozen=True)
class ClutterInstance:
    """Один объект над землёй в мировой системе X east, Y north, Z up."""

    identifier: str
    kind: ClutterKind
    x_m: float
    y_m: float
    radius_x_m: float
    radius_y_m: float
    height_m: float
    rotation_z_degrees: float
    color_srgb: tuple[float, float, float]

    @property
    def footprint_radius_m(self) -> float:
        """Возвращает консервативный радиус занимаемого места на земле."""

        return max(self.radius_x_m, self.radius_y_m)


_BASE_COLORS: dict[ClutterKind, np.ndarray] = {
    "rock": np.asarray([0.38, 0.36, 0.33], dtype=np.float64),
    "stump": np.asarray([0.34, 0.18, 0.08], dtype=np.float64),
    "shrub": np.asarray([0.12, 0.31, 0.09], dtype=np.float64),
}


def generate_clutter_instances(
    *,
    width_m: float,
    depth_m: float,
    specification: dict[str, Any],
    seed: int,
) -> tuple[ClutterInstance, ...]:
    """Создаёт воспроизводимую смесь объектов без пересечения их оснований.

    `minimum_gap_m` относится к расстоянию между краями консервативных
    круговых оснований. Если заданную плотность физически нельзя разместить,
    функция явно отказывается вместо молчаливого уменьшения числа объектов.
    """

    count = int(specification.get("count", 0))
    edge_margin = float(specification.get("edge_margin_m", 0.0))
    minimum_gap = float(specification.get("minimum_gap_m", 0.0))
    maximum_attempts = int(specification.get("maximum_attempts", max(1, count * 500)))
    if width_m <= 0.0 or depth_m <= 0.0:
        raise ValueError("Размеры участка должны быть положительными")
    if count < 0:
        raise ValueError("Число объектов не может быть отрицательным")
    if edge_margin < 0.0 or 2.0 * edge_margin >= min(width_m, depth_m):
        raise ValueError("Краевой отступ не помещается внутри участка")
    if minimum_gap < 0.0:
        raise ValueError("Минимальный зазор не может быть отрицательным")
    if maximum_attempts <= 0:
        raise ValueError("Число попыток размещения должно быть положительным")
    if count == 0:
        return ()

    kinds = specification.get("kinds")
    if not isinstance(kinds, dict) or not kinds:
        raise ValueError("Для ненулевого мусора нужны параметры типов объектов")
    unknown = set(kinds) - set(_BASE_COLORS)
    if unknown:
        raise ValueError(f"Неизвестные типы объектов: {sorted(unknown)}")

    kind_names = tuple(kinds)
    weights = np.asarray(
        [float(kinds[name].get("weight", 1.0)) for name in kind_names],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError(
            "Веса типов должны быть конечными, неотрицательными и ненулевыми"
        )
    probabilities = weights / weights.sum()
    random_generator = np.random.default_rng(seed)
    selected: list[ClutterInstance] = []

    for object_index in range(count):
        kind = str(random_generator.choice(kind_names, p=probabilities))
        parameters = kinds[kind]
        height_range = _positive_range(parameters, "height_m")
        radius_range = _positive_range(parameters, "radius_m")
        aspect_range = _positive_range(parameters, "xy_aspect_ratio")
        height = float(random_generator.uniform(*height_range))
        base_radius = float(random_generator.uniform(*radius_range))
        aspect = float(random_generator.uniform(*aspect_range))
        radius_x = base_radius * aspect**0.5
        radius_y = base_radius / aspect**0.5
        footprint_radius = max(radius_x, radius_y)
        x_limit = width_m / 2.0 - edge_margin - footprint_radius
        y_limit = depth_m / 2.0 - edge_margin - footprint_radius
        if x_limit <= 0.0 or y_limit <= 0.0:
            raise ValueError("Объект с заданным отступом не помещается на участке")

        position: tuple[float, float] | None = None
        for _ in range(maximum_attempts):
            candidate = (
                float(random_generator.uniform(-x_limit, x_limit)),
                float(random_generator.uniform(-y_limit, y_limit)),
            )
            if all(
                np.hypot(candidate[0] - item.x_m, candidate[1] - item.y_m)
                >= footprint_radius + item.footprint_radius_m + minimum_gap
                for item in selected
            ):
                position = candidate
                break
        if position is None:
            raise ValueError(
                f"Не удалось разместить {count} объектов без пересечений; "
                f"остановка на индексе {object_index}"
            )

        color_noise = random_generator.uniform(-0.055, 0.055, size=3)
        color = np.clip(_BASE_COLORS[kind] + color_noise, 0.03, 0.95)
        selected.append(
            ClutterInstance(
                identifier=f"clutter_{object_index:03d}_{kind}",
                kind=kind,
                x_m=position[0],
                y_m=position[1],
                radius_x_m=radius_x,
                radius_y_m=radius_y,
                height_m=height,
                rotation_z_degrees=float(random_generator.uniform(0.0, 360.0)),
                color_srgb=tuple(float(value) for value in color),
            )
        )
    return tuple(selected)


def _positive_range(
    parameters: dict[str, Any],
    name: str,
) -> tuple[float, float]:
    """Читает закрытый положительный диапазон из двух чисел."""

    values = np.asarray(parameters.get(name), dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError(f"{name} должен содержать два конечных числа")
    minimum, maximum = (float(value) for value in values)
    if minimum <= 0.0 or maximum < minimum:
        raise ValueError(f"Недопустимый диапазон {name}: {values.tolist()}")
    return minimum, maximum
