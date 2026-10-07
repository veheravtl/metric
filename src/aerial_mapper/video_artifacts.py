# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Детерминированные цифровые приближения JPEG и экранной OSD-графики."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class JpegRoundTrip:
    """Декодированный RGB и размер реально полученного JPEG-буфера."""

    image_rgb: NDArray[np.uint8]
    encoded_bytes: int


@dataclass(frozen=True)
class OsdOcclusion:
    """RGB с псевдоглифами и точная маска заменённых пикселей."""

    image_rgb: NDArray[np.uint8]
    occluded_mask: NDArray[np.bool_]
    actual_fraction: float


def _validate_rgb(image_rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Проверяет общий контракт RGB uint8 без неявного преобразования."""

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Ожидается RGB uint8 с формой H x W x 3")
    return image


def jpeg_round_trip(
    image_rgb: NDArray[np.uint8], *, quality: int
) -> JpegRoundTrip:
    """Выполняет настоящий JPEG encode/decode с заданным quality 0..100."""

    image = _validate_rgb(image_rgb)
    if isinstance(quality, bool) or not isinstance(quality, int):
        raise ValueError("JPEG quality должен быть целым")
    if not 0 <= quality <= 100:
        raise ValueError("JPEG quality должен лежать в 0..100")
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(
        ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not success:
        raise RuntimeError("OpenCV не смог закодировать JPEG")
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise RuntimeError("OpenCV не смог декодировать свой JPEG")
    return JpegRoundTrip(
        image_rgb=cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB),
        encoded_bytes=int(encoded.size),
    )


def apply_osd_occlusion(
    image_rgb: NDArray[np.uint8],
    *,
    fraction: float,
    random_seed: int,
) -> OsdOcclusion:
    """Заменяет точную долю пикселей контрастными псевдоглифами OSD.

    Кандидаты образуют штрихи условной сетки 30 x 13, как у аналогового OSD.
    Сначала выбираются клетки у краёв, затем ближе к центру. Seed меняет
    конкретный набор штрихов, но не требуемую площадь перекрытия.
    """

    image = _validate_rgb(image_rgb)
    if not 0.0 <= fraction <= 0.5:
        raise ValueError("Доля OSD должна лежать в 0..0.5")
    height, width = image.shape[:2]
    target_count = int(round(fraction * height * width))
    if target_count == 0:
        return OsdOcclusion(
            image_rgb=image.copy(),
            occluded_mask=np.zeros((height, width), dtype=bool),
            actual_fraction=0.0,
        )

    y, x = np.indices((height, width))
    cell_x = np.minimum(29, x * 30 // width)
    cell_y = np.minimum(12, y * 13 // height)
    local_x = x * 30 % width
    local_y = y * 13 % height
    stroke_x = max(1, width // 320)
    stroke_y = max(1, height // 240)
    glyph_candidate = (
        (local_x < stroke_x * 3)
        | ((local_x > width // 60) & (local_x < width // 60 + stroke_x * 2))
        | (local_y < stroke_y * 3)
        | ((local_y > height // 26) & (local_y < height // 26 + stroke_y * 2))
    )
    candidate_indices = np.flatnonzero(glyph_candidate)
    if candidate_indices.size < target_count:
        candidate_indices = np.arange(height * width)

    edge_distance = np.minimum.reduce(
        (cell_x, 29 - cell_x, cell_y, 12 - cell_y)
    ).ravel()[candidate_indices]
    generator = np.random.default_rng(random_seed)
    tie_break = generator.random(candidate_indices.size)
    order = np.lexsort((tie_break, edge_distance))
    selected = candidate_indices[order[:target_count]]
    mask = np.zeros(height * width, dtype=bool)
    mask[selected] = True
    mask = mask.reshape(height, width)

    result = image.copy()
    white = ((cell_x + cell_y) % 2 == 0)[mask]
    result[mask] = np.where(white[:, None], 255, 0).astype(np.uint8)
    return OsdOcclusion(
        image_rgb=result,
        occluded_mask=mask,
        actual_fraction=float(np.mean(mask)),
    )
