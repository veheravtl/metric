"""Проверяемые признаки пригодности видеокадра для совмещения.

Модуль намеренно не пытается свести качество к одному красивому числу.
Экспозиция и резкость описывают разные причины отказа, поэтому сохраняются
отдельно. Порог резкости сравнивает соседние кадры одной последовательности:
абсолютная дисперсия границ слишком сильно зависит от фактуры местности.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FrameImageQuality:
    """Диагностика изображения текущего кадра относительно предыдущего.

    ``laplacian_variance`` — дисперсия второй производной яркости. Резкие
    границы дают большой разброс значений, размазанные — малый. Это не мера
    физического фокуса и не универсальная шкала качества, а дешёвый индикатор
    внезапной потери деталей между соседними кадрами.

    Доли тёмных и светлых пикселей лежат в диапазоне 0…1. Они обнаруживают
    почти пустой кадр, но не пытаются угадать параметры затвора или диафрагмы.
    """

    previous_laplacian_variance: float
    current_laplacian_variance: float
    sharpness_ratio_to_previous: float
    mean_luminance: float
    dark_pixel_fraction: float
    bright_pixel_fraction: float
    warning_reasons: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    @property
    def is_rejected(self) -> bool:
        """Есть ли причина не строить совмещение этого кадра."""

        return bool(self.rejection_reasons)


def assess_frame_image_quality(
    previous_frame_bgr: NDArray[np.uint8],
    current_frame_bgr: NDArray[np.uint8],
) -> FrameImageQuality:
    """Оценить очевидные дефекты изображения без эталонного хорошего кадра.

    Отказ сделан намеренно консервативным:

    - не менее 98% почти чёрных или почти белых пикселей означает фактически
      потерянное изображение;
    - резкость должна упасть более чем в 20 раз относительно достаточно
      детального предыдущего кадра, чтобы автоматика сама остановилась.

    Более мягкие границы дают предупреждение, но не отказ. Эти исходные пороги
    являются инженерной политикой первого эксперимента и должны калиброваться
    на размеченных кадрах реального пролёта.
    """

    previous_gray = _as_gray_u8(previous_frame_bgr, name="Предыдущий кадр")
    current_gray = _as_gray_u8(current_frame_bgr, name="Текущий кадр")
    if previous_gray.shape != current_gray.shape:
        raise ValueError("Соседние кадры должны иметь одинаковый размер")

    previous_sharpness = _laplacian_variance(previous_gray)
    current_sharpness = _laplacian_variance(current_gray)
    if previous_sharpness <= np.finfo(np.float64).eps:
        sharpness_ratio = 1.0 if current_sharpness <= 0.0 else float("inf")
    else:
        sharpness_ratio = current_sharpness / previous_sharpness

    dark_fraction = float(np.mean(current_gray <= 8))
    bright_fraction = float(np.mean(current_gray >= 247))
    mean_luminance = float(np.mean(current_gray))

    warnings: list[str] = []
    rejections: list[str] = []
    if dark_fraction >= 0.98:
        rejections.append("кадр почти полностью чёрный")
    elif dark_fraction >= 0.65:
        warnings.append("большая часть кадра недоэкспонирована")

    if bright_fraction >= 0.98:
        rejections.append("кадр почти полностью пересвечен")
    elif bright_fraction >= 0.65:
        warnings.append("большая часть кадра пересвечена")

    # На почти однородном предыдущем изображении отношение резкости не имеет
    # физического смысла. Минимум 25 исключает деление двух шумовых величин.
    if previous_sharpness >= 25.0:
        if sharpness_ratio < 0.05:
            rejections.append("резкость упала более чем в 20 раз")
        elif sharpness_ratio < 0.35:
            warnings.append("резкость заметно ниже соседнего кадра")

    return FrameImageQuality(
        previous_laplacian_variance=previous_sharpness,
        current_laplacian_variance=current_sharpness,
        sharpness_ratio_to_previous=float(sharpness_ratio),
        mean_luminance=mean_luminance,
        dark_pixel_fraction=dark_fraction,
        bright_pixel_fraction=bright_fraction,
        warning_reasons=tuple(warnings),
        rejection_reasons=tuple(rejections),
    )


def _laplacian_variance(gray: NDArray[np.uint8]) -> float:
    """Вернуть дисперсию лапласиана яркости в квадратных уровнях серого."""

    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return float(np.var(laplacian))


def _as_gray_u8(image: NDArray[np.uint8], *, name: str) -> NDArray[np.uint8]:
    """Проверить массив изображения и привести его к непрерывному grayscale."""

    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"{name} должен иметь тип uint8")
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim == 3 and array.shape[2] == 4:
        gray = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"{name} должен быть серым, BGR или BGRA-изображением")
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        raise ValueError(f"{name} слишком мал для оценки качества")
    return np.ascontiguousarray(gray)
