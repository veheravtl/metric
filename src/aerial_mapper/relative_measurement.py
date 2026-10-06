# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Измерение промаха в системе координат с центром в цели.

Модуль не знает, откуда получены метрические координаты. Ими могут быть
локальные координаты бумажной карты, проекционная система цифровой карты или
синтетическая истина Blender. Обязательное свойство одно: цель и попадание
должны быть выражены в одной декартовой системе с единицами в метрах.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot

import numpy as np
from numpy.typing import NDArray

FloatPoint = NDArray[np.float64]


@dataclass(frozen=True)
class TargetCenteredDisplacement:
    """Вектор от цели к попаданию в осях метрического Teach."""

    delta_x_m: float
    delta_y_m: float
    distance_m: float


@dataclass(frozen=True)
class DisplacementError:
    """Ошибка восстановленного вектора относительно скрытой истины."""

    error_x_m: float
    error_y_m: float
    vector_error_m: float
    absolute_distance_error_m: float
    direction_error_degrees: float | None
    x_sign_correct: bool | None
    y_sign_correct: bool | None


def target_centered_displacement(
    target_xy_m: FloatPoint | list[float] | tuple[float, float],
    impact_xy_m: FloatPoint | list[float] | tuple[float, float],
) -> TargetCenteredDisplacement:
    """Вычитает цель из попадания и возвращает компоненты и длину.

    Начало координат результата всегда находится в цели. Ориентация осей
    наследуется от метрического Teach или карты. Положение стрелка для этого
    базового измерения не требуется.
    """

    target = _point(target_xy_m, name="Цель")
    impact = _point(impact_xy_m, name="Попадание")
    delta = impact - target
    return TargetCenteredDisplacement(
        delta_x_m=float(delta[0]),
        delta_y_m=float(delta[1]),
        distance_m=float(np.linalg.norm(delta)),
    )


def displacement_error(
    estimated: TargetCenteredDisplacement,
    truth: TargetCenteredDisplacement,
    *,
    zero_component_tolerance_m: float = 1e-9,
) -> DisplacementError:
    """Сравнивает компоненты, длину, направление и знаки двух векторов.

    Знак нулевой истинной компоненты не имеет физического смысла, поэтому для
    неё возвращается None. Направление также не определено для точного
    попадания в цель.
    """

    if zero_component_tolerance_m < 0.0:
        raise ValueError("Допуск нулевой компоненты не может быть отрицательным")

    error_x = estimated.delta_x_m - truth.delta_x_m
    error_y = estimated.delta_y_m - truth.delta_y_m
    direction_error = _direction_error_degrees(estimated, truth)
    return DisplacementError(
        error_x_m=float(error_x),
        error_y_m=float(error_y),
        vector_error_m=hypot(error_x, error_y),
        absolute_distance_error_m=abs(estimated.distance_m - truth.distance_m),
        direction_error_degrees=direction_error,
        x_sign_correct=_sign_correct(
            estimated.delta_x_m,
            truth.delta_x_m,
            tolerance=zero_component_tolerance_m,
        ),
        y_sign_correct=_sign_correct(
            estimated.delta_y_m,
            truth.delta_y_m,
            tolerance=zero_component_tolerance_m,
        ),
    )


def _point(
    value: FloatPoint | list[float] | tuple[float, float],
    *,
    name: str,
) -> FloatPoint:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,):
        raise ValueError(f"{name} должна иметь форму (2,)")
    if not np.isfinite(point).all():
        raise ValueError(f"{name} содержит NaN или бесконечность")
    return point


def _direction_error_degrees(
    estimated: TargetCenteredDisplacement,
    truth: TargetCenteredDisplacement,
) -> float | None:
    if truth.distance_m == 0.0 or estimated.distance_m == 0.0:
        return None
    cross = (
        truth.delta_x_m * estimated.delta_y_m - truth.delta_y_m * estimated.delta_x_m
    )
    dot = truth.delta_x_m * estimated.delta_x_m + truth.delta_y_m * estimated.delta_y_m
    return abs(degrees(atan2(cross, dot)))


def _sign_correct(estimated: float, truth: float, *, tolerance: float) -> bool | None:
    if abs(truth) <= tolerance:
        return None
    if abs(estimated) <= tolerance:
        return False
    return bool(np.signbit(estimated) == np.signbit(truth))
