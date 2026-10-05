"""Контролируемые ошибки разметки перспективного Teach-кадра.

G8 не обучает сегментатор и не пытается угадать реалистичное распределение
ошибок конкретного человека. Вместо этого модуль по одной причине изменяет
известную бинарную маску: сдвигает границу, удаляет связную часть допустимой
области либо добавляет часть заведомо неплоского объекта. Такой причинный
эксперимент позволяет связать изменение метрик с одним типом ошибки.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

BinaryMask = NDArray[np.uint8]


@dataclass(frozen=True)
class MaskAgreement:
    """Качество рабочей маски относительно скрытой истинной маски земли.

    ``precision`` отвечает на вопрос, какая доля разрешённых рабочей маской
    пикселей действительно является землёй. ``recall`` показывает, какая доля
    всей видимой земли сохранилась. ``intersection_over_union`` (IoU) — доля
    пересечения в объединении двух масок; она одновременно штрафует пропуски и
    ошибочные включения.
    """

    predicted_fraction: float
    true_ground_fraction: float
    true_positive_pixels: int
    false_positive_pixels: int
    false_negative_pixels: int
    precision: float
    recall: float
    intersection_over_union: float


def _binary_mask(mask: BinaryMask, *, name: str) -> BinaryMask:
    """Проверяет двумерную непустую маску и нормализует её к значениям 0/255."""

    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError(f"{name} должна быть двумерной")
    if values.dtype != np.uint8:
        raise ValueError(f"{name} должна иметь тип uint8")
    result = np.where(values != 0, 255, 0).astype(np.uint8)
    if not np.any(result):
        raise ValueError(f"{name} не содержит ни одного разрешённого пикселя")
    return result


def offset_mask_boundary(mask: BinaryMask, *, offset_pixels: int) -> BinaryMask:
    """Сдвигает границу бинарной маски на целое число пикселей.

    Отрицательное значение выполняет эрозию — консервативно отступает внутрь
    земли. Положительное выполняет дилатацию и тем самым может захватить крышу
    или другую соседнюю поверхность. Ноль возвращает точную копию.
    """

    source = _binary_mask(mask, name="Исходная маска")
    radius = abs(int(offset_pixels))
    if radius == 0:
        return source.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    if offset_pixels < 0:
        result = cv2.erode(source, kernel, iterations=1)
    else:
        result = cv2.dilate(source, kernel, iterations=1)
    return _binary_mask(result, name="Маска после сдвига границы")


def retain_connected_fraction(
    mask: BinaryMask,
    *,
    retained_fraction: float,
) -> BinaryMask:
    """Оставляет заданную долю маски связным диагональным срезом.

    Это модель грубой неполной разметки: аннотатор закончил только часть
    видимой земли. Срез детерминирован и не создаёт россыпь случайных дыр,
    которая одновременно проверяла бы иной тип ошибки границы.
    """

    source = _binary_mask(mask, name="Исходная маска")
    if not 0.0 < retained_fraction <= 1.0:
        raise ValueError("Сохраняемая доля должна лежать в диапазоне (0, 1]")
    if retained_fraction == 1.0:
        return source.copy()

    y, x = np.indices(source.shape, dtype=np.float64)
    height, width = source.shape
    score = x / max(width - 1, 1) + 0.37 * y / max(height - 1, 1)
    active_scores = score[source != 0]
    threshold = float(np.quantile(active_scores, retained_fraction))
    result = np.where((source != 0) & (score <= threshold), 255, 0).astype(np.uint8)
    return _binary_mask(result, name="Неполная маска")


def enclosed_non_ground(mask: BinaryMask) -> BinaryMask:
    """Находит неплоские области, полностью окружённые видимой землёй.

    Компоненты фона, касающиеся рамки изображения, исключаются: они могут быть
    внешней областью сцены, а не крышей. В текущей G7/G8-сцене оставшийся
    внутренний компонент соответствует проекции здания вместе с его границей.
    """

    ground = _binary_mask(mask, name="Истинная маска земли")
    non_ground = (ground == 0).astype(np.uint8)
    component_count, labels = cv2.connectedComponents(non_ground, connectivity=8)
    enclosed = np.zeros_like(ground)
    for label in range(1, component_count):
        component = labels == label
        touches_border = bool(
            np.any(component[0, :])
            or np.any(component[-1, :])
            or np.any(component[:, 0])
            or np.any(component[:, -1])
        )
        if not touches_border:
            enclosed[component] = 255
    if not np.any(enclosed):
        raise ValueError("В маске не найдена внутренняя неплоская область")
    return enclosed


def add_contaminant_fraction(
    base_mask: BinaryMask,
    contaminant_mask: BinaryMask,
    *,
    added_fraction: float,
) -> BinaryMask:
    """Добавляет связную долю известной неплоской области к маске земли."""

    base = _binary_mask(base_mask, name="Базовая маска")
    contaminant = _binary_mask(contaminant_mask, name="Маска загрязнителя")
    if base.shape != contaminant.shape:
        raise ValueError("Маски земли и загрязнителя должны иметь одинаковый размер")
    if np.any((base != 0) & (contaminant != 0)):
        raise ValueError("Маска загрязнителя должна быть отделена от базовой маски")
    if not 0.0 < added_fraction <= 1.0:
        raise ValueError("Добавляемая доля должна лежать в диапазоне (0, 1]")

    if added_fraction == 1.0:
        selected = contaminant != 0
    else:
        y, x = np.indices(base.shape, dtype=np.float64)
        height, width = base.shape
        score = x / max(width - 1, 1) - 0.23 * y / max(height - 1, 1)
        active_scores = score[contaminant != 0]
        threshold = float(np.quantile(active_scores, added_fraction))
        selected = (contaminant != 0) & (score <= threshold)
    return np.where((base != 0) | selected, 255, 0).astype(np.uint8)


def mask_agreement(
    predicted_mask: BinaryMask,
    true_ground_mask: BinaryMask,
) -> MaskAgreement:
    """Считает чистоту, полноту и IoU маски без использования RGB."""

    predicted = _binary_mask(predicted_mask, name="Рабочая маска") != 0
    truth = _binary_mask(true_ground_mask, name="Истинная маска земли") != 0
    if predicted.shape != truth.shape:
        raise ValueError("Рабочая и истинная маски должны иметь одинаковый размер")

    true_positive = int(np.count_nonzero(predicted & truth))
    false_positive = int(np.count_nonzero(predicted & ~truth))
    false_negative = int(np.count_nonzero(~predicted & truth))
    predicted_count = true_positive + false_positive
    truth_count = true_positive + false_negative
    union_count = true_positive + false_positive + false_negative
    pixel_count = predicted.size
    return MaskAgreement(
        predicted_fraction=predicted_count / pixel_count,
        true_ground_fraction=truth_count / pixel_count,
        true_positive_pixels=true_positive,
        false_positive_pixels=false_positive,
        false_negative_pixels=false_negative,
        precision=true_positive / predicted_count,
        recall=true_positive / truth_count,
        intersection_over_union=true_positive / union_count,
    )
