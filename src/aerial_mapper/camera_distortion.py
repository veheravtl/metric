# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Контролируемая дисторсия pinhole-камеры для синтетических опытов.

Модуль использует ту же модель Brown--Conrady, что и OpenCV: координаты
сначала нормируются фокусом и главной точкой, затем получают радиальную и
тангенциальную поправки. Поза камеры и метрическая истина сюда не передаются.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from aerial_mapper.radiance_camera import PinholeIntrinsics
from aerial_mapper.synthetic import FloatPoints, RgbImage


@dataclass(frozen=True)
class BrownConradyDistortion:
    """Пять коэффициентов стандартной модели дисторсии OpenCV.

    ``k1``, ``k2`` и ``k3`` задают радиальное смещение. ``p1`` и ``p2``
    задают тангенциальное смещение от несовпадения осей линз и сенсора. Все
    коэффициенты безразмерны, потому что применяются к нормированным
    координатам ``x=(u-cx)/fx`` и ``y=(v-cy)/fy``.
    """

    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def validate(self) -> None:
        """Запрещает нечисловую конфигурацию до построения карт remap."""

        values = np.asarray(
            [self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError("Коэффициенты дисторсии должны быть конечными")

    def as_opencv(self) -> NDArray[np.float64]:
        """Возвращает порядок ``k1, k2, p1, p2, k3``, ожидаемый OpenCV."""

        self.validate()
        return np.asarray(
            [self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64
        )


def camera_matrix(intrinsics: PinholeIntrinsics) -> NDArray[np.float64]:
    """Строит матрицу внутренних параметров 3 × 3 в пикселях."""

    if intrinsics.width_px <= 0 or intrinsics.height_px <= 0:
        raise ValueError("Размер кадра должен быть положительным")
    values = np.asarray(
        [
            intrinsics.focal_x_px,
            intrinsics.focal_y_px,
            intrinsics.principal_x_px,
            intrinsics.principal_y_px,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values[:2] <= 0.0):
        raise ValueError("Внутренние параметры камеры должны быть конечными")
    return np.asarray(
        [
            [intrinsics.focal_x_px, 0.0, intrinsics.principal_x_px],
            [0.0, intrinsics.focal_y_px, intrinsics.principal_y_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _validate_points(points_px: FloatPoints) -> NDArray[np.float64]:
    """Проверяет форму пиксельных координат на общей границе функций."""

    points = np.asarray(points_px, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Пиксельные точки должны иметь форму N x 2")
    if not np.isfinite(points).all():
        raise ValueError("Пиксельные точки должны быть конечными")
    return points


def _validate_image(image_rgb: RgbImage, intrinsics: PinholeIntrinsics) -> None:
    """Проверяет формат и согласованность RGB с калибровкой."""

    expected = (intrinsics.height_px, intrinsics.width_px, 3)
    if image_rgb.shape != expected or image_rgb.dtype != np.uint8:
        raise ValueError(
            f"Ожидается RGB uint8 формы {expected}, получено {image_rgb.shape}"
        )


def distort_points(
    undistorted_points_px: FloatPoints,
    *,
    intrinsics: PinholeIntrinsics,
    distortion: BrownConradyDistortion,
) -> NDArray[np.float64]:
    """Переносит идеальные pinhole-пиксели в координаты искажённого кадра.

    Точки интерпретируются как лучи ``[x, y, 1]``. ``cv2.projectPoints``
    применяет ту же модель, что функции калибровки и исправления OpenCV.
    """

    points = _validate_points(undistorted_points_px)
    matrix = camera_matrix(intrinsics)
    normalized = np.column_stack(
        (
            (points[:, 0] - intrinsics.principal_x_px)
            / intrinsics.focal_x_px,
            (points[:, 1] - intrinsics.principal_y_px)
            / intrinsics.focal_y_px,
            np.ones(points.shape[0], dtype=np.float64),
        )
    )
    projected, _ = cv2.projectPoints(
        normalized,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        matrix,
        distortion.as_opencv(),
    )
    return projected.reshape(-1, 2).astype(np.float64)


def undistort_points(
    distorted_points_px: FloatPoints,
    *,
    intrinsics: PinholeIntrinsics,
    distortion: BrownConradyDistortion,
) -> NDArray[np.float64]:
    """Возвращает наблюдаемые пиксели в исходную pinhole-систему.

    Python-binding установленного OpenCV не открывает вариант с настраиваемым
    числом итераций. Его стандартная остановка даёт до нескольких тысячных
    пикселя на краю G14-A. Поэтому здесь явно решается та же система
    Brown--Conrady методом Ньютона до строгого численного допуска.
    """

    points = _validate_points(distorted_points_px)
    camera_matrix(intrinsics)
    k1, k2, p1, p2, k3 = distortion.as_opencv()
    target_x = (points[:, 0] - intrinsics.principal_x_px) / intrinsics.focal_x_px
    target_y = (points[:, 1] - intrinsics.principal_y_px) / intrinsics.focal_y_px
    x = target_x.copy()
    y = target_y.copy()
    for _ in range(20):
        radius2 = x * x + y * y
        radius4 = radius2 * radius2
        radial = 1.0 + k1 * radius2 + k2 * radius4 + k3 * radius4 * radius2
        radial_derivative = k1 + 2.0 * k2 * radius2 + 3.0 * k3 * radius4
        radial_x = 2.0 * x * radial_derivative
        radial_y = 2.0 * y * radial_derivative
        predicted_x = x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
        predicted_y = y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
        residual_x = predicted_x - target_x
        residual_y = predicted_y - target_y
        if np.max(np.hypot(residual_x, residual_y), initial=0.0) < 1e-14:
            break
        jacobian_xx = radial + x * radial_x + 2.0 * p1 * y + 6.0 * p2 * x
        jacobian_xy = x * radial_y + 2.0 * p1 * x + 2.0 * p2 * y
        jacobian_yx = y * radial_x + 2.0 * p1 * x + 2.0 * p2 * y
        jacobian_yy = radial + y * radial_y + 6.0 * p1 * y + 2.0 * p2 * x
        determinant = jacobian_xx * jacobian_yy - jacobian_xy * jacobian_yx
        if np.any(np.abs(determinant) < 1e-12):
            raise ValueError("Модель дисторсии необратима в контрольных точках")
        x -= (jacobian_yy * residual_x - jacobian_xy * residual_y) / determinant
        y -= (-jacobian_yx * residual_x + jacobian_xx * residual_y) / determinant
    corrected_x = x * intrinsics.focal_x_px + intrinsics.principal_x_px
    corrected_y = y * intrinsics.focal_y_px + intrinsics.principal_y_px
    corrected = np.column_stack((corrected_x, corrected_y))
    # Метод не имеет права молча вернуть правдоподобную точку, если заданная
    # модель оказалась плохо обратимой. Проверяем результат независимым прямым
    # преобразованием в пикселях, а не только внутренним условием остановки.
    redistorted = distort_points(
        corrected,
        intrinsics=intrinsics,
        distortion=distortion,
    )
    if np.max(np.linalg.norm(redistorted - points, axis=1), initial=0.0) > 1e-6:
        raise ValueError("Обращение модели дисторсии не сошлось")
    return corrected


def distort_image(
    image_rgb: RgbImage,
    *,
    intrinsics: PinholeIntrinsics,
    distortion: BrownConradyDistortion,
) -> RgbImage:
    """Создаёт искажённый RGB из идеального pinhole-рендера.

    Обратная карта для каждого пикселя искажённого кадра указывает, откуда
    взять цвет в pinhole-изображении. Чёрная граница означает неизвестный луч.
    """

    _validate_image(image_rgb, intrinsics)
    matrix = camera_matrix(intrinsics)
    map_x, map_y = cv2.initInverseRectificationMap(
        matrix,
        distortion.as_opencv(),
        np.eye(3, dtype=np.float64),
        matrix,
        (intrinsics.width_px, intrinsics.height_px),
        cv2.CV_32FC1,
    )
    return cv2.remap(
        image_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def undistort_image(
    image_rgb: RgbImage,
    *,
    intrinsics: PinholeIntrinsics,
    distortion: BrownConradyDistortion,
) -> RgbImage:
    """Исправляет RGB в pinhole-систему при известной калибровке."""

    _validate_image(image_rgb, intrinsics)
    matrix = camera_matrix(intrinsics)
    return cv2.undistort(
        image_rgb, matrix, distortion.as_opencv(), None, matrix
    )
