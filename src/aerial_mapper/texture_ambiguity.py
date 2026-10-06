# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Наблюдаемая проверка самопохожести опорного Teach-кадра.

Обычная привязка спрашивает: «нашлась ли одна согласованная гомография?».
Периодический узор способен дать убедительный, но неверный ответ на этот
вопрос. Здесь задаётся дополнительный вопрос: «есть ли у локальных признаков
почти одинаковые копии далеко в том же Teach-кадре?». Для него не нужны
скрытая геометрия Blender, класс текстуры или Repeat-кадр.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TextureAmbiguityThresholds:
    """Пороги fail-closed проверки повторяющейся текстуры.

    ``remote_distance_px`` исключает соседние признаки одного объекта.
    ``maximum_descriptor_distance`` задаёт, насколько близкими должны быть
    два 128-мерных SIFT-дескриптора, чтобы считаться почти копиями.
    """

    remote_distance_px: float = 64.0
    maximum_descriptor_distance: float = 80.0
    minimum_keypoint_count: int = 200
    maximum_remote_duplicate_fraction: float = 0.20

    def __post_init__(self) -> None:
        """Не позволяет бессмысленным порогам молча попасть в отчёт."""

        if self.remote_distance_px <= 0.0:
            raise ValueError("Удалённость признаков должна быть положительной")
        if self.maximum_descriptor_distance <= 0.0:
            raise ValueError("Порог расстояния дескрипторов должен быть положительным")
        if self.minimum_keypoint_count < 2:
            raise ValueError("Для проверки неоднозначности нужны хотя бы 2 признака")
        if not 0.0 <= self.maximum_remote_duplicate_fraction <= 1.0:
            raise ValueError("Доля удалённых копий должна лежать в диапазоне 0..1")


@dataclass(frozen=True)
class TextureAmbiguityAssessment:
    """Результат проверки, полностью вычисленный по рабочим входам."""

    keypoint_count: int
    remotely_comparable_keypoint_count: int
    remote_duplicate_count: int
    remote_duplicate_fraction: float
    rejected_as_ambiguous: bool


def _remote_nearest_descriptor_distances(
    descriptors: NDArray[np.float32],
    points_xy: NDArray[np.float32],
    *,
    remote_distance_px: float,
    chunk_size: int = 128,
) -> NDArray[np.float32]:
    """Ищет ближайшую по SIFT копию вне локальной окрестности.

    Возвращается по одному расстоянию на признак. ``inf`` означает, что в
    кадре нет ни одного достаточно удалённого кандидата. Расчёт выполняется
    блоками, чтобы не создавать массив ``N x N x 128`` для большого кадра.
    """

    descriptor_array = np.asarray(descriptors, dtype=np.float32)
    point_array = np.asarray(points_xy, dtype=np.float32)
    if descriptor_array.ndim != 2 or descriptor_array.shape[0] == 0:
        raise ValueError("Нужен непустой двумерный массив дескрипторов")
    if point_array.shape != (descriptor_array.shape[0], 2):
        raise ValueError("Каждому дескриптору нужна одна точка (x, y)")
    if not np.isfinite(descriptor_array).all() or not np.isfinite(point_array).all():
        raise ValueError("Дескрипторы и координаты должны быть конечными")
    if remote_distance_px <= 0.0 or chunk_size < 1:
        raise ValueError("Удалённость и размер блока должны быть положительными")

    descriptor_norms = np.einsum(
        "ij,ij->i", descriptor_array, descriptor_array
    )
    nearest_squared = np.full(descriptor_array.shape[0], np.inf, dtype=np.float32)
    minimum_spatial_squared = np.float32(remote_distance_px * remote_distance_px)

    for start in range(0, descriptor_array.shape[0], chunk_size):
        stop = min(start + chunk_size, descriptor_array.shape[0])
        query = descriptor_array[start:stop]
        # ||a-b||² = ||a||² + ||b||² - 2a·b. Небольшое отрицательное
        # значение возможно только из-за округления float32.
        descriptor_squared = (
            descriptor_norms[start:stop, None]
            + descriptor_norms[None, :]
            - 2.0 * query @ descriptor_array.T
        )
        np.maximum(descriptor_squared, 0.0, out=descriptor_squared)
        spatial_delta = point_array[start:stop, None, :] - point_array[None, :, :]
        spatial_squared = np.einsum("ijk,ijk->ij", spatial_delta, spatial_delta)
        descriptor_squared[spatial_squared < minimum_spatial_squared] = np.inf
        nearest_squared[start:stop] = np.min(descriptor_squared, axis=1)

    return np.sqrt(nearest_squared).astype(np.float32, copy=False)


def assess_reference_texture_ambiguity(
    reference_rgb: NDArray[np.uint8],
    reference_feature_mask: NDArray[np.uint8],
    *,
    thresholds: TextureAmbiguityThresholds | None = None,
) -> TextureAmbiguityAssessment:
    """Оценивает риск далёких повторов по RGB Teach и его рабочей маске.

    Маска имеет ту же семантику, что и у основного matcher: ненулевые пиксели
    разрешают искать SIFT-признаки. Класс поверхности и истинные координаты в
    функцию намеренно не передаются.
    """

    limits = thresholds or TextureAmbiguityThresholds()
    image = np.asarray(reference_rgb)
    mask = np.asarray(reference_feature_mask)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Teach RGB должен иметь форму H x W x 3 и dtype uint8")
    if mask.shape != image.shape[:2] or mask.dtype != np.uint8:
        raise ValueError("Teach-маска должна иметь форму H x W и dtype uint8")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = cv2.SIFT_create().detectAndCompute(gray, mask)
    keypoint_count = len(keypoints)
    if descriptors is None or keypoint_count < 2:
        return TextureAmbiguityAssessment(
            keypoint_count=keypoint_count,
            remotely_comparable_keypoint_count=0,
            remote_duplicate_count=0,
            remote_duplicate_fraction=0.0,
            rejected_as_ambiguous=False,
        )

    points_xy = np.asarray([item.pt for item in keypoints], dtype=np.float32)
    remote_distances = _remote_nearest_descriptor_distances(
        np.asarray(descriptors, dtype=np.float32),
        points_xy,
        remote_distance_px=limits.remote_distance_px,
    )
    comparable = np.isfinite(remote_distances)
    comparable_count = int(np.count_nonzero(comparable))
    duplicate_count = int(
        np.count_nonzero(
            comparable
            & (remote_distances <= limits.maximum_descriptor_distance)
        )
    )
    duplicate_fraction = (
        duplicate_count / comparable_count if comparable_count else 0.0
    )
    rejected = bool(
        keypoint_count >= limits.minimum_keypoint_count
        and duplicate_fraction > limits.maximum_remote_duplicate_fraction
    )
    return TextureAmbiguityAssessment(
        keypoint_count=keypoint_count,
        remotely_comparable_keypoint_count=comparable_count,
        remote_duplicate_count=duplicate_count,
        remote_duplicate_fraction=duplicate_fraction,
        rejected_as_ambiguous=rejected,
    )
