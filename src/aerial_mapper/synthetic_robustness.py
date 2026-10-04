"""Детерминированные искажения RGB для синтетических G4--G6.

Blender задаёт геометрию и базовый рендер. Здесь меняется только готовый
Repeat-кадр, чтобы однофакторный опыт не смешивал причины ошибки. Функции не
получают позу камеры или метрическую истину.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from aerial_mapper.synthetic import RgbImage


@dataclass(frozen=True)
class ImageDegradation:
    """Параметры контролируемого ухудшения одного RGB-кадра.

    Экспозиция измеряется в фотографических ступенях: +1 удваивает линейный
    свет. Доля тени задаёт площадь простой прямолинейной тени, а пропускание --
    долю света внутри неё. Размытие измеряется стандартным отклонением
    Gaussian blur в пикселях, шум -- стандартным отклонением 8-битного канала.
    Масштаб разрешения сначала уменьшает растр, затем возвращает исходный размер.
    """

    exposure_stops: float = 0.0
    shadow_fraction: float = 0.0
    shadow_transmission: float = 0.22
    blur_sigma_px: float = 0.0
    noise_standard_deviation: float = 0.0
    resolution_scale: float = 1.0
    shadow_angle_degrees: float = 27.0

    def validate(self) -> None:
        """Отвергает значения без физического или численного смысла."""

        if not -12.0 <= self.exposure_stops <= 12.0:
            raise ValueError("Экспозиция должна лежать в диапазоне -12..12")
        if not 0.0 <= self.shadow_fraction <= 1.0:
            raise ValueError("Доля тени должна лежать в диапазоне 0..1")
        if not 0.0 <= self.shadow_transmission <= 1.0:
            raise ValueError("Пропускание тени должно лежать в диапазоне 0..1")
        if self.blur_sigma_px < 0.0:
            raise ValueError("Размытие не может быть отрицательным")
        if self.noise_standard_deviation < 0.0:
            raise ValueError("Шум не может быть отрицательным")
        if not 0.05 <= self.resolution_scale <= 1.0:
            raise ValueError("Масштаб разрешения должен лежать в 0.05..1")


def _srgb_to_linear(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Переводит нормированный sRGB в линейный свет."""

    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Возвращает линейный свет в sRGB с ограничением диапазона сенсора."""

    image = np.clip(image, 0.0, 1.0)
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * image ** (1.0 / 2.4) - 0.055,
    )


def _shadow_mask(
    height: int,
    width: int,
    *,
    fraction: float,
    angle_degrees: float,
) -> NDArray[np.bool_]:
    """Строит полуплоскость с заданной долей площади кадра."""

    if fraction <= 0.0:
        return np.zeros((height, width), dtype=bool)
    if fraction >= 1.0:
        return np.ones((height, width), dtype=bool)
    y, x = np.indices((height, width), dtype=np.float32)
    x = (x - (width - 1) / 2.0) / max(width - 1, 1)
    y = (y - (height - 1) / 2.0) / max(height - 1, 1)
    angle = np.deg2rad(angle_degrees)
    coordinate = x * np.cos(angle) + y * np.sin(angle)
    return coordinate <= float(np.quantile(coordinate, fraction))


def apply_image_degradation(
    image_rgb: RgbImage,
    degradation: ImageDegradation,
    *,
    random_seed: int,
) -> RgbImage:
    """Применяет свет, оптику, дискретизацию и шум в фиксированном порядке."""

    degradation.validate()
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Ожидается RGB-изображение с тремя каналами")
    if image_rgb.dtype != np.uint8:
        raise ValueError("Ожидается RGB-изображение типа uint8")

    linear = _srgb_to_linear(image_rgb.astype(np.float32) / 255.0)
    linear *= np.float32(2.0**degradation.exposure_stops)
    height, width = image_rgb.shape[:2]
    shadow = _shadow_mask(
        height,
        width,
        fraction=degradation.shadow_fraction,
        angle_degrees=degradation.shadow_angle_degrees,
    )
    linear[shadow] *= np.float32(degradation.shadow_transmission)
    result = np.clip(
        np.rint(_linear_to_srgb(linear) * 255.0), 0.0, 255.0
    ).astype(np.uint8)

    if degradation.blur_sigma_px > 0.0:
        result = cv2.GaussianBlur(
            result,
            (0, 0),
            sigmaX=degradation.blur_sigma_px,
            sigmaY=degradation.blur_sigma_px,
        )
    if degradation.resolution_scale < 1.0:
        reduced_size = (
            max(2, int(round(width * degradation.resolution_scale))),
            max(2, int(round(height * degradation.resolution_scale))),
        )
        result = cv2.resize(result, reduced_size, interpolation=cv2.INTER_AREA)
        result = cv2.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)
    if degradation.noise_standard_deviation > 0.0:
        generator = np.random.default_rng(random_seed)
        noise = generator.normal(
            0.0,
            degradation.noise_standard_deviation,
            size=result.shape,
        )
        result = np.clip(
            np.rint(result.astype(np.float32) + noise), 0.0, 255.0
        ).astype(np.uint8)
    return result


def perturb_click_points(
    points_px: NDArray[np.float64],
    *,
    standard_deviation_px: float,
    random_seed: int,
) -> NDArray[np.float64]:
    """Эмулирует независимую нормальную ошибку ручных кликов."""

    points = np.asarray(points_px, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Точки должны иметь форму (N, 2)")
    if standard_deviation_px < 0.0:
        raise ValueError("Ошибка клика не может быть отрицательной")
    if standard_deviation_px == 0.0:
        return points.copy()
    generator = np.random.default_rng(random_seed)
    return points + generator.normal(0.0, standard_deviation_px, size=points.shape)
