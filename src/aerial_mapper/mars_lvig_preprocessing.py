"""Подготовка кадров MARS-LVIG к поиску и геометрической проверке."""

from __future__ import annotations

import cv2
import numpy as np

from aerial_mapper.camera_distortion import (
    BrownConradyDistortion,
    undistort_image,
)
from aerial_mapper.radiance_camera import PinholeIntrinsics
from aerial_mapper.synthetic import RgbImage


def prepare_mars_lvig_image(
    image_rgb: RgbImage,
    *,
    intrinsics: PinholeIntrinsics,
    distortion: BrownConradyDistortion,
    maximum_side_pixels: int = 672,
) -> RgbImage:
    """Исправляет дисторсию и уменьшает кадр без изменения пропорций.

    Сначала применяется опубликованная модель исходной камеры 2448 x 2048.
    Только затем изображение масштабируется: если сначала уменьшить кадр, пришлось
    бы пересчитывать фокус и главную точку, что создаёт лишний источник ошибки.

    Длинная сторона результата равна maximum_side_pixels. Увеличивать малые
    изображения функция не будет: интерполяция не создаёт новых деталей, важных
    для SIFT или DINOv2.
    """

    if maximum_side_pixels <= 0:
        raise ValueError("Максимальная сторона должна быть положительной")

    corrected_rgb = undistort_image(
        image_rgb,
        intrinsics=intrinsics,
        distortion=distortion,
    )
    height, width = corrected_rgb.shape[:2]
    longest_side = max(width, height)
    if longest_side <= maximum_side_pixels:
        return corrected_rgb.copy()

    scale = maximum_side_pixels / longest_side
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        corrected_rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    if resized.dtype != np.uint8 or resized.ndim != 3 or resized.shape[2] != 3:
        raise RuntimeError("OpenCV вернул изображение неожиданного формата")
    return resized
