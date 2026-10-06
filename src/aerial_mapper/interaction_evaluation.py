# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Классификация продуктового вектора в комбинированных стресс-тестах.

Модуль не знает изображений, позы камеры и скрытой геометрии Blender. Он
сопоставляет наблюдаемое решение gate с независимой ошибкой вектора
«цель → попадание». Такое разделение не позволяет объявить безопасным
геометрически устойчивое, но метрически неправильное совмещение.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProductGateSummary:
    """Итог одной группы направленных векторов при одном решении gate."""

    count: int
    vector_error_p95_m: float
    vector_error_maximum_m: float
    x_sign_error_count: int
    y_sign_error_count: int
    product_within_limits: bool
    classification: str


def summarize_product_gate(
    vector_errors_m: Iterable[float],
    *,
    x_sign_correct: Iterable[bool | None],
    y_sign_correct: Iterable[bool | None],
    gate_accepted: bool,
    maximum_vector_error_p95_m: float,
) -> ProductGateSummary:
    """Сопоставляет допуск вектора с принятием или отказом рабочего gate.

    Знак проверяется только для ненулевой истинной компоненты. Поэтому None
    означает «знак неприменим», а False — реальную перестановку
    «лево/право» либо «перелёт/недолёт».
    """

    errors = np.asarray(tuple(vector_errors_m), dtype=np.float64)
    x_signs = tuple(x_sign_correct)
    y_signs = tuple(y_sign_correct)
    if errors.ndim != 1 or errors.size == 0:
        raise ValueError("Нужен непустой одномерный набор ошибок вектора")
    if not np.isfinite(errors).all() or np.any(errors < 0.0):
        raise ValueError("Ошибки вектора должны быть конечными и неотрицательными")
    if len(x_signs) != errors.size or len(y_signs) != errors.size:
        raise ValueError("Для каждой ошибки нужны признаки знака X и Y")
    if maximum_vector_error_p95_m <= 0.0:
        raise ValueError("Предел p95 ошибки вектора должен быть положительным")
    if any(value not in (True, False, None) for value in (*x_signs, *y_signs)):
        raise ValueError("Признак знака должен быть True, False или None")

    p95 = float(np.percentile(errors, 95))
    maximum = float(np.max(errors))
    x_errors = sum(value is False for value in x_signs)
    y_errors = sum(value is False for value in y_signs)
    within = bool(
        p95 <= maximum_vector_error_p95_m
        and x_errors == 0
        and y_errors == 0
    )
    if gate_accepted and within:
        classification = "accepted_correct"
    elif gate_accepted:
        classification = "false_accept"
    elif within:
        classification = "rejected_valid"
    else:
        classification = "rejected_invalid"

    return ProductGateSummary(
        count=int(errors.size),
        vector_error_p95_m=p95,
        vector_error_maximum_m=maximum,
        x_sign_error_count=x_errors,
        y_sign_error_count=y_errors,
        product_within_limits=within,
        classification=classification,
    )
