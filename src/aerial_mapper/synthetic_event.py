# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Детерминированное наложение синтетического события на видеопоследовательность.

Модуль решает две независимые задачи. Сначала неподвижная точка земли
переносится между соседними чистыми кадрами по фоновым признакам. Затем из
футажа со статичной камерой выделяется изменившаяся область и накладывается в
эту точку. Разделение важно для эксперимента: сам взрыв не участвует в расчёте
своей контрольной траектории и потому не может незаметно облегчить проверяемому
алгоритму задачу.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

GrayImage = NDArray[np.uint8]
BgrImage = NDArray[np.uint8]
FloatImage = NDArray[np.float32]
FloatPoints = NDArray[np.float64]


class SyntheticEventFailure(RuntimeError):
    """Ожидаемый отказ, когда контрольную траекторию нельзя подтвердить."""


@dataclass(frozen=True)
class OpticalFlowAnchorConfig:
    """Параметры переноса наземного якоря по чистым соседним кадрам.

    Изображение уменьшается только для поиска движения. Координаты результата
    всегда возвращаются в исходных пикселях. Проверка потока вперёд-назад
    удаляет точки, которые не возвращаются в исходное место, а RANSAC оставляет
    движение, согласованное с одной плоской проективной моделью земли.
    """

    processing_scale: float = 0.5
    maximum_corners: int = 1_500
    corner_quality: float = 0.01
    minimum_corner_distance_px: float = 12.0
    lk_window_size_px: int = 31
    lk_pyramid_levels: int = 4
    maximum_forward_backward_error_px: float = 1.5
    ransac_threshold_px: float = 2.0
    minimum_inlier_count: int = 100
    minimum_inlier_ratio: float = 0.70

    def validate(self) -> None:
        """Отклоняет параметры, при которых проверка движения теряет смысл."""

        if not 0.0 < self.processing_scale <= 1.0:
            raise ValueError("Масштаб обработки должен быть в диапазоне (0, 1]")
        if self.maximum_corners < 4 or self.minimum_inlier_count < 4:
            raise ValueError("Для гомографии необходимы хотя бы четыре точки")
        if not 0.0 < self.corner_quality <= 1.0:
            raise ValueError("Качество угла должно быть в диапазоне (0, 1]")
        if self.minimum_corner_distance_px <= 0.0:
            raise ValueError("Минимальное расстояние углов должно быть положительным")
        if self.lk_window_size_px < 3 or self.lk_window_size_px % 2 == 0:
            raise ValueError("Окно Лукаса—Канаде должно быть нечётным и не меньше 3")
        if self.lk_pyramid_levels < 0:
            raise ValueError("Число уровней пирамиды не может быть отрицательным")
        if self.maximum_forward_backward_error_px <= 0.0:
            raise ValueError("Порог ошибки вперёд-назад должен быть положительным")
        if self.ransac_threshold_px <= 0.0:
            raise ValueError("Порог RANSAC должен быть положительным")
        if not 0.0 < self.minimum_inlier_ratio <= 1.0:
            raise ValueError("Минимальная доля inlier должна быть в диапазоне (0, 1]")


@dataclass(frozen=True)
class AnchorTrackingStep:
    """Диагностика одного перехода между соседними чистыми кадрами."""

    frame_index: int
    checked_track_count: int
    inlier_count: int
    inlier_ratio: float
    median_forward_backward_error_processing_px: float


@dataclass(frozen=True)
class AnchorTrajectory:
    """Контрольные координаты одной точки земли в исходных пикселях кадров."""

    frame_indices: NDArray[np.int64]
    points_px: FloatPoints
    steps: tuple[AnchorTrackingStep, ...]


def _read_scaled_gray(path: Path, *, scale: float) -> GrayImage:
    """Читает кадр и уменьшает его для быстрого расчёта движения."""

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise SyntheticEventFailure(f"Не удалось прочитать чистый кадр: {path}")
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def track_ground_anchor(
    frame_paths: Sequence[Path],
    *,
    first_frame_index: int,
    initial_anchor_px: tuple[float, float],
    config: OpticalFlowAnchorConfig | None = None,
) -> AnchorTrajectory:
    """Переносит наземный якорь по последовательности чистых кадров.

    ``frame_paths[0]`` соответствует ``first_frame_index``. Для каждого
    перехода признаки ищутся заново по всему предыдущему кадру. Это дороже
    переноса одного пикселя, но даёт сотни независимых фоновых свидетельств и
    позволяет явно отказаться от результата при слабой геометрии.
    """

    effective = config or OpticalFlowAnchorConfig()
    effective.validate()
    if not frame_paths:
        raise ValueError("Нужен хотя бы один чистый кадр")
    anchor = np.asarray(initial_anchor_px, dtype=np.float64)
    if anchor.shape != (2,) or not np.all(np.isfinite(anchor)):
        raise ValueError("Начальный якорь должен быть конечной парой (x, y)")

    previous = _read_scaled_gray(Path(frame_paths[0]), scale=effective.processing_scale)
    points = [anchor.copy()]
    steps: list[AnchorTrackingStep] = []
    lk_parameters = {
        "winSize": (effective.lk_window_size_px, effective.lk_window_size_px),
        "maxLevel": effective.lk_pyramid_levels,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }

    for offset, path in enumerate(frame_paths[1:], start=1):
        current = _read_scaled_gray(Path(path), scale=effective.processing_scale)
        if current.shape != previous.shape:
            raise SyntheticEventFailure(
                "Размер чистых кадров изменился внутри фрагмента"
            )
        previous_points = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=effective.maximum_corners,
            qualityLevel=effective.corner_quality,
            minDistance=effective.minimum_corner_distance_px,
            blockSize=7,
        )
        if previous_points is None or len(previous_points) < 4:
            raise SyntheticEventFailure("На чистом кадре недостаточно фоновых углов")

        current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous, current, previous_points, None, **lk_parameters
        )
        if current_points is None or forward_status is None:
            raise SyntheticEventFailure("Оптический поток вперёд не вычислен")
        returned_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current, previous, current_points, None, **lk_parameters
        )
        if returned_points is None or backward_status is None:
            raise SyntheticEventFailure("Оптический поток назад не вычислен")

        previous_flat = previous_points.reshape(-1, 2)
        current_flat = current_points.reshape(-1, 2)
        returned_flat = returned_points.reshape(-1, 2)
        forward_backward_error = np.linalg.norm(
            previous_flat - returned_flat,
            axis=1,
        )
        accepted = (
            (forward_status.ravel() == 1)
            & (backward_status.ravel() == 1)
            & np.isfinite(forward_backward_error)
            & (
                forward_backward_error
                < effective.maximum_forward_backward_error_px
            )
        )
        source = previous_flat[accepted]
        target = current_flat[accepted]
        if source.shape[0] < effective.minimum_inlier_count:
            raise SyntheticEventFailure(
                "После проверки потока осталось слишком мало фоновых точек: "
                f"{source.shape[0]}"
            )

        homography, ransac_mask = cv2.findHomography(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=effective.ransac_threshold_px,
            maxIters=3_000,
            confidence=0.999,
        )
        if homography is None or ransac_mask is None:
            raise SyntheticEventFailure("RANSAC не оценил движение между кадрами")
        inlier_count = int(np.count_nonzero(ransac_mask))
        inlier_ratio = inlier_count / source.shape[0]
        if inlier_count < effective.minimum_inlier_count:
            raise SyntheticEventFailure(
                f"RANSAC подтвердил только {inlier_count} фоновых точек"
            )
        if inlier_ratio < effective.minimum_inlier_ratio:
            raise SyntheticEventFailure(
                f"Доля согласованного фонового движения мала: {inlier_ratio:.3f}"
            )

        scaled_anchor = anchor * effective.processing_scale
        moved_scaled = cv2.perspectiveTransform(
            scaled_anchor.reshape(1, 1, 2), homography
        ).reshape(2)
        anchor = moved_scaled / effective.processing_scale
        if not np.all(np.isfinite(anchor)):
            raise SyntheticEventFailure("Якорь ушёл в бесконечность при переносе")
        points.append(anchor.copy())
        steps.append(
            AnchorTrackingStep(
                frame_index=first_frame_index + offset,
                checked_track_count=int(source.shape[0]),
                inlier_count=inlier_count,
                inlier_ratio=float(inlier_ratio),
                median_forward_backward_error_processing_px=float(
                    np.median(forward_backward_error[accepted])
                ),
            )
        )
        previous = current

    return AnchorTrajectory(
        frame_indices=np.arange(
            first_frame_index,
            first_frame_index + len(frame_paths),
            dtype=np.int64,
        ),
        points_px=np.asarray(points, dtype=np.float64),
        steps=tuple(steps),
    )


def _edge_feather(
    height: int,
    width: int,
    *,
    horizontal_fraction: float,
    vertical_fraction: float,
) -> FloatImage:
    """Создаёт плавное окно, скрывающее границы прямоугольного crop."""

    if not 0.0 < horizontal_fraction <= 0.5:
        raise ValueError("Горизонтальное затухание должно быть в диапазоне (0, 0.5]")
    if not 0.0 < vertical_fraction <= 0.5:
        raise ValueError("Вертикальное затухание должно быть в диапазоне (0, 0.5]")
    x_distance = np.minimum(np.arange(width), np.arange(width)[::-1])
    y_distance = np.minimum(np.arange(height), np.arange(height)[::-1])
    x = np.clip(x_distance / (width * horizontal_fraction), 0.0, 1.0)
    y = np.clip(y_distance / (height * vertical_fraction), 0.0, 1.0)
    window = np.minimum(y[:, None], x[None, :]).astype(np.float32)
    return window * window * (3.0 - 2.0 * window)


def extract_changed_foreground_alpha(
    current_bgr: BgrImage,
    background_bgr: BgrImage,
    *,
    hard_difference_threshold: int = 30,
    soft_difference_start: int = 24,
    soft_difference_span: int = 42,
    minimum_component_area_px: int = 250,
    maximum_components: int = 5,
    horizontal_edge_feather_fraction: float = 0.16,
    vertical_edge_feather_fraction: float = 0.18,
) -> FloatImage:
    """Оценивает мягкую маску изменения относительно чистого фона.

    Это не нейросетевая сегментация. Метод рассчитан на футаж со статичной
    камерой: большое связное изменение считается эффектом, мелкие изменения
    кодека отбрасываются. Края маски плавно гасятся, чтобы обрезанный crop не
    проявлялся на целевом кадре прямоугольником.
    """

    if current_bgr.shape != background_bgr.shape:
        raise ValueError("Текущий кадр и фон должны иметь одинаковый размер")
    if current_bgr.ndim != 3 or current_bgr.shape[2] != 3:
        raise ValueError("Футаж должен быть BGR-изображением с тремя каналами")
    if current_bgr.dtype != np.uint8 or background_bgr.dtype != np.uint8:
        raise ValueError("Футаж и фон должны иметь тип uint8")
    if hard_difference_threshold <= 0 or soft_difference_span <= 0:
        raise ValueError("Пороги разности должны быть положительными")
    if minimum_component_area_px <= 0 or maximum_components <= 0:
        raise ValueError("Параметры связных областей должны быть положительными")

    difference = cv2.absdiff(current_bgr, background_bgr).max(axis=2)
    difference = cv2.GaussianBlur(difference, (0, 0), 2.0)
    hard_mask = (difference > hard_difference_threshold).astype(np.uint8) * 255
    hard_mask = cv2.morphologyEx(
        hard_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    hard_mask = cv2.morphologyEx(
        hard_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        hard_mask
    )
    ranked_components = sorted(
        (
            (int(statistics[index, cv2.CC_STAT_AREA]), index)
            for index in range(1, component_count)
            if statistics[index, cv2.CC_STAT_AREA] >= minimum_component_area_px
        ),
        reverse=True,
    )
    retained = np.zeros_like(hard_mask)
    for _, component_index in ranked_components[:maximum_components]:
        retained[labels == component_index] = 255
    retained = cv2.dilate(
        retained,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    connected_soft_mask = cv2.GaussianBlur(retained, (0, 0), 7.0) / 255.0
    difference_alpha = np.clip(
        (difference.astype(np.float32) - soft_difference_start)
        / soft_difference_span,
        0.0,
        1.0,
    )
    height, width = difference.shape
    edge_window = _edge_feather(
        height,
        width,
        horizontal_fraction=horizontal_edge_feather_fraction,
        vertical_fraction=vertical_edge_feather_fraction,
    )
    return (difference_alpha * connected_soft_mask * edge_window).astype(np.float32)


def composite_layer_at_anchor(
    background_bgr: BgrImage,
    layer_bgr: BgrImage,
    alpha: NDArray[np.floating],
    *,
    layer_anchor_px: tuple[float, float],
    target_anchor_px: tuple[float, float],
) -> BgrImage:
    """Накладывает слой так, чтобы его якорь совпал с якорем целевого кадра."""

    if background_bgr.ndim != 3 or background_bgr.shape[2] != 3:
        raise ValueError("Фон должен быть BGR-изображением")
    if layer_bgr.ndim != 3 or layer_bgr.shape[2] != 3:
        raise ValueError("Слой должен быть BGR-изображением")
    alpha_float = np.asarray(alpha, dtype=np.float32)
    if alpha_float.shape != layer_bgr.shape[:2]:
        raise ValueError("Форма alpha должна совпадать с высотой и шириной слоя")
    if not np.all(np.isfinite(alpha_float)):
        raise ValueError("Alpha содержит нечисловые значения")
    if np.min(alpha_float) < 0.0 or np.max(alpha_float) > 1.0:
        raise ValueError("Alpha должна находиться в диапазоне [0, 1]")

    layer_anchor = np.asarray(layer_anchor_px, dtype=np.float64)
    target_anchor = np.asarray(target_anchor_px, dtype=np.float64)
    if layer_anchor.shape != (2,) or target_anchor.shape != (2,):
        raise ValueError("Якоря должны быть парами (x, y)")
    top_left = np.rint(target_anchor - layer_anchor).astype(np.int64)
    layer_height, layer_width = layer_bgr.shape[:2]
    background_height, background_width = background_bgr.shape[:2]
    target_x0 = max(0, int(top_left[0]))
    target_y0 = max(0, int(top_left[1]))
    target_x1 = min(background_width, int(top_left[0]) + layer_width)
    target_y1 = min(background_height, int(top_left[1]) + layer_height)
    if target_x0 >= target_x1 or target_y0 >= target_y1:
        raise SyntheticEventFailure("Слой целиком оказался за пределами кадра")

    layer_x0 = target_x0 - int(top_left[0])
    layer_y0 = target_y0 - int(top_left[1])
    layer_x1 = layer_x0 + target_x1 - target_x0
    layer_y1 = layer_y0 + target_y1 - target_y0
    result = background_bgr.copy()
    local_alpha = alpha_float[layer_y0:layer_y1, layer_x0:layer_x1, None]
    foreground = layer_bgr[layer_y0:layer_y1, layer_x0:layer_x1].astype(np.float32)
    background = result[target_y0:target_y1, target_x0:target_x1].astype(
        np.float32
    )
    result[target_y0:target_y1, target_x0:target_x1] = np.clip(
        foreground * local_alpha + background * (1.0 - local_alpha),
        0.0,
        255.0,
    ).astype(np.uint8)
    return result
