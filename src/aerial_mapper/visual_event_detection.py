# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Лёгкий детектор локального резкого изменения в движущемся видеопотоке.

Модуль намеренно не распознаёт внешний вид взрыва. Он проверяет более общую
гипотезу: после компенсации движения камеры в кадре появилась связная область,
изменение которой заметно сильнее обычного остатка совмещения.

Расчёт выполняется на уменьшенной полутоновой копии кадра. Движение камеры
оценивается по разреженным точкам методом Лукаса—Канаде, поэтому плотный
оптический поток по всем пикселям не нужен. Ненадёжная геометрия приводит к
явному состоянию ``invalid_geometry``, а не к правдоподобной тревоге.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

BgrImage = NDArray[np.uint8]
GrayImage = NDArray[np.uint8]
DetectionStatus = Literal[
    "warming_up",
    "monitoring",
    "candidate",
    "confirmed",
    "invalid_geometry",
]


@dataclass(frozen=True)
class VisualEventDetectorConfig:
    """Параметры первого, намеренно компактного эксперимента.

    Все геометрические пороги, кроме явно отмеченных долей, задаются в
    пикселях уменьшенного изображения. Координаты наружу возвращаются в
    пикселях исходного кадра.
    """

    analysis_width_px: int = 640
    maximum_features: int = 500
    minimum_tracked_features: int = 35
    minimum_inlier_ratio: float = 0.55
    ransac_reprojection_threshold_px: float = 2.0
    maximum_forward_backward_error_px: float = 1.5
    minimum_residual_threshold: float = 20.0
    residual_mad_multiplier: float = 6.0
    minimum_component_area_px: int = 70
    maximum_component_fraction: float = 0.15
    maximum_global_change_fraction: float = 0.25
    minimum_component_mean_to_threshold_ratio: float = 1.5
    confirmation_window_frames: int = 3
    confirmation_hits: int = 2

    def validate(self) -> None:
        """Не допустить неоднозначных либо физически невозможных настроек."""

        if self.analysis_width_px < 160:
            raise ValueError("Ширина анализа должна быть не меньше 160 px")
        if self.maximum_features < self.minimum_tracked_features:
            raise ValueError("Число искомых точек меньше требуемого минимума")
        if self.minimum_tracked_features < 4:
            raise ValueError("Для гомографии требуется хотя бы четыре точки")
        if not 0.0 < self.minimum_inlier_ratio <= 1.0:
            raise ValueError("Доля inlier должна лежать в интервале (0, 1]")
        if self.ransac_reprojection_threshold_px <= 0.0:
            raise ValueError("Порог RANSAC должен быть положительным")
        if self.maximum_forward_backward_error_px <= 0.0:
            raise ValueError("Порог ошибки потока должен быть положительным")
        if self.minimum_residual_threshold <= 0.0:
            raise ValueError("Минимальный порог остатка должен быть положительным")
        if self.residual_mad_multiplier <= 0.0:
            raise ValueError("Множитель MAD должен быть положительным")
        if self.minimum_component_area_px <= 0:
            raise ValueError("Минимальная площадь компоненты должна быть положительной")
        for name, value in (
            ("maximum_component_fraction", self.maximum_component_fraction),
            ("maximum_global_change_fraction", self.maximum_global_change_fraction),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} должен лежать в интервале (0, 1)")
        if self.minimum_component_mean_to_threshold_ratio <= 1.0:
            raise ValueError(
                "Средний остаток компоненты должен превышать порог более чем в 1 раз"
            )
        if not 1 <= self.confirmation_hits <= self.confirmation_window_frames:
            raise ValueError("Число подтверждений несовместимо с временным окном")


@dataclass(frozen=True)
class EventCandidate:
    """Одна связная область необычного изменения на текущем кадре."""

    frame_index: int
    center_px: tuple[float, float]
    bounding_box_px: tuple[float, float, float, float]
    area_fraction: float
    mean_residual: float
    threshold: float


@dataclass(frozen=True)
class ConfirmedVisualEvent:
    """Событие, подтверждённое несколькими близкими кандидатами.

    Точка и прямоугольник выражены в геометрии кадра подтверждения, на котором
    поток будет остановлен. Момент начала при этом сохраняется отдельно.
    """
    onset_frame_index: int
    confirmation_frame_index: int
    point_px: tuple[float, float]
    bounding_box_px: tuple[float, float, float, float]


@dataclass(frozen=True)
class VisualEventDetectionStep:
    """Диагностика одного перехода между соседними кадрами."""

    frame_index: int
    status: DetectionStatus
    candidate: EventCandidate | None
    confirmed_event: ConfirmedVisualEvent | None
    tracked_feature_count: int
    inlier_count: int
    inlier_ratio: float
    residual_threshold: float | None
    changed_fraction: float | None
    processing_time_ms: float
    reason: str | None = None


@dataclass(frozen=True)
class _AnalysisFrame:
    gray: GrayImage
    scale_from_source: float
    source_size: tuple[int, int]


class VisualEventDetector:
    """Найти первое локальное резкое изменение в последовательном потоке.

    Детектор хранит только предыдущий уменьшенный кадр и короткое окно
    кандидатов. Он не обучается на синтетическом взрыве и не использует цвет,
    форму огня либо заранее заданный номер кадра.
    """

    def __init__(self, config: VisualEventDetectorConfig | None = None) -> None:
        self.config = config or VisualEventDetectorConfig()
        self.config.validate()
        self._previous: _AnalysisFrame | None = None
        self._previous_frame_index: int | None = None
        self._two_back: _AnalysisFrame | None = None
        self._two_back_to_previous_homography: NDArray[np.float64] | None = None
        self._recent_candidates: deque[EventCandidate | None] = deque(
            maxlen=self.config.confirmation_window_frames
        )
        self._confirmed_event: ConfirmedVisualEvent | None = None

    @property
    def confirmed_event(self) -> ConfirmedVisualEvent | None:
        """Вернуть уже подтверждённое первое событие, если оно найдено."""

        return self._confirmed_event

    def reset(self) -> None:
        """Удалить временную историю перед новым источником или перемоткой."""

        self._previous = None
        self._previous_frame_index = None
        self._two_back = None
        self._two_back_to_previous_homography = None
        self._recent_candidates.clear()
        self._confirmed_event = None

    def start(
        self, frame_bgr: BgrImage, *, frame_index: int
    ) -> VisualEventDetectionStep:
        """Принять первый кадр, для которого ещё нет предыдущего наблюдения."""

        started = time.perf_counter()
        if frame_index < 0:
            raise ValueError("Индекс кадра не может быть отрицательным")
        self._previous = _prepare_analysis_frame(frame_bgr, self.config)
        self._previous_frame_index = frame_index
        self._two_back = None
        self._two_back_to_previous_homography = None
        self._recent_candidates.clear()
        self._confirmed_event = None
        return VisualEventDetectionStep(
            frame_index=frame_index,
            status="warming_up",
            candidate=None,
            confirmed_event=None,
            tracked_feature_count=0,
            inlier_count=0,
            inlier_ratio=0.0,
            residual_threshold=None,
            changed_fraction=None,
            processing_time_ms=_elapsed_ms(started),
            reason="Нужен следующий последовательный кадр",
        )

    def advance(
        self, frame_bgr: BgrImage, *, frame_index: int
    ) -> VisualEventDetectionStep:
        """Обработать строго следующий кадр и вернуть объяснимый результат."""

        if self._previous is None or self._previous_frame_index is None:
            return self.start(frame_bgr, frame_index=frame_index)
        if frame_index != self._previous_frame_index + 1:
            self.reset()
            step = self.start(frame_bgr, frame_index=frame_index)
            return VisualEventDetectionStep(
                **{
                    **step.__dict__,
                    "reason": "Последовательность кадров прервана; история сброшена",
                }
            )

        started = time.perf_counter()
        current = _prepare_analysis_frame(frame_bgr, self.config)
        previous = self._previous

        if current.gray.shape != previous.gray.shape:
            self._recent_candidates.clear()
            self._remember_transition(
                previous, current, frame_index, homography=None
            )
            return self._invalid_step(
                frame_index,
                started,
                "Размер кадра изменился во время потока",
            )

        geometry = _estimate_camera_motion(previous.gray, current.gray, self.config)
        if geometry.homography is None:
            self._recent_candidates.clear()
            self._remember_transition(
                previous, current, frame_index, homography=None
            )
            return self._invalid_step(
                frame_index,
                started,
                geometry.reason or "Движение камеры не оценено",
                tracked_feature_count=geometry.tracked_count,
                inlier_count=geometry.inlier_count,
                inlier_ratio=geometry.inlier_ratio,
            )

        candidate, threshold, changed_fraction = _find_local_change(
            previous.gray,
            current.gray,
            geometry.homography,
            frame_index=frame_index,
            source_scale=current.scale_from_source,
            config=self.config,
        )
        # Если изменение на следующем кадре стало почти неподвижным, разность
        # соседних кадров уже не видит его. Только после первого кандидата
        # сравниваем текущий кадр с кадром до события. Две соседние гомографии
        # компонуются, поэтому повторно искать движение не требуется.
        if (
            candidate is None
            and any(item is not None for item in self._recent_candidates)
            and self._two_back is not None
            and self._two_back_to_previous_homography is not None
        ):
            two_back_to_current = (
                geometry.homography @ self._two_back_to_previous_homography
            )
            candidate, threshold, changed_fraction = _find_local_change(
                self._two_back.gray,
                current.gray,
                two_back_to_current,
                frame_index=frame_index,
                source_scale=current.scale_from_source,
                config=self.config,
            )
        self._recent_candidates.append(candidate)
        self._remember_transition(
            previous,
            current,
            frame_index,
            homography=geometry.homography,
        )

        status: DetectionStatus = "candidate" if candidate is not None else "monitoring"
        if self._confirmed_event is None:
            self._confirmed_event = _confirm_event(
                self._recent_candidates,
                confirmation_frame_index=frame_index,
                required_hits=self.config.confirmation_hits,
            )
        if self._confirmed_event is not None:
            status = "confirmed"

        return VisualEventDetectionStep(
            frame_index=frame_index,
            status=status,
            candidate=candidate,
            confirmed_event=self._confirmed_event,
            tracked_feature_count=geometry.tracked_count,
            inlier_count=geometry.inlier_count,
            inlier_ratio=geometry.inlier_ratio,
            residual_threshold=threshold,
            changed_fraction=changed_fraction,
            processing_time_ms=_elapsed_ms(started),
        )

    def _invalid_step(
        self,
        frame_index: int,
        started: float,
        reason: str,
        *,
        tracked_feature_count: int = 0,
        inlier_count: int = 0,
        inlier_ratio: float = 0.0,
    ) -> VisualEventDetectionStep:
        """Сформировать явный отказ без кандидата и без ложной тревоги."""

        return VisualEventDetectionStep(
            frame_index=frame_index,
            status="invalid_geometry",
            candidate=None,
            confirmed_event=self._confirmed_event,
            tracked_feature_count=tracked_feature_count,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            residual_threshold=None,
            changed_fraction=None,
            processing_time_ms=_elapsed_ms(started),
            reason=reason,
        )

    def _remember_transition(
        self,
        previous: _AnalysisFrame,
        current: _AnalysisFrame,
        frame_index: int,
        *,
        homography: NDArray[np.float64] | None,
    ) -> None:
        """Сохранить два последних кадра и известный переход между ними."""

        self._two_back = previous
        self._two_back_to_previous_homography = homography
        self._previous = current
        self._previous_frame_index = frame_index


@dataclass(frozen=True)
class _GeometryEstimate:
    homography: NDArray[np.float64] | None
    tracked_count: int
    inlier_count: int
    inlier_ratio: float
    reason: str | None = None


def _prepare_analysis_frame(
    frame_bgr: BgrImage, config: VisualEventDetectorConfig
) -> _AnalysisFrame:
    """Проверить исходный кадр и уменьшить его без изменения пропорций."""

    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Ожидается BGR-кадр uint8 формы H×W×3")
    source_height, source_width = frame_bgr.shape[:2]
    if source_height == 0 or source_width == 0:
        raise ValueError("Пустой кадр нельзя анализировать")
    scale = min(1.0, config.analysis_width_px / float(source_width))
    if scale < 1.0:
        analysis_size = (
            max(1, round(source_width * scale)),
            max(1, round(source_height * scale)),
        )
        resized = cv2.resize(frame_bgr, analysis_size, interpolation=cv2.INTER_AREA)
    else:
        resized = frame_bgr
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return _AnalysisFrame(
        gray=gray,
        scale_from_source=scale,
        source_size=(source_width, source_height),
    )


def _estimate_camera_motion(
    previous_gray: GrayImage,
    current_gray: GrayImage,
    config: VisualEventDetectorConfig,
) -> _GeometryEstimate:
    """Оценить гомографию предыдущий→текущий по проверенному редкому потоку."""

    source = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=config.maximum_features,
        qualityLevel=0.01,
        minDistance=8.0,
        blockSize=7,
    )
    if source is None or len(source) < config.minimum_tracked_features:
        count = 0 if source is None else len(source)
        return _GeometryEstimate(
            None,
            count,
            0,
            0.0,
            "Недостаточно текстурных точек для оценки движения камеры",
        )

    parameters = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }
    target, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, source, None, **parameters
    )
    if target is None or forward_status is None:
        return _GeometryEstimate(None, 0, 0, 0.0, "Прямой поток не вычислен")
    returned, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, target, None, **parameters
    )
    if returned is None or backward_status is None:
        return _GeometryEstimate(None, 0, 0, 0.0, "Обратный поток не вычислен")

    forward_ok = forward_status.reshape(-1).astype(bool)
    backward_ok = backward_status.reshape(-1).astype(bool)
    errors = np.linalg.norm(
        returned.reshape(-1, 2) - source.reshape(-1, 2), axis=1
    )
    valid = (
        forward_ok
        & backward_ok
        & np.isfinite(errors)
        & (errors <= config.maximum_forward_backward_error_px)
    )
    tracked_count = int(np.count_nonzero(valid))
    if tracked_count < config.minimum_tracked_features:
        return _GeometryEstimate(
            None,
            tracked_count,
            0,
            0.0,
            "Слишком мало точек прошло проверку потока вперёд-назад",
        )

    source_valid = source.reshape(-1, 2)[valid]
    target_valid = target.reshape(-1, 2)[valid]
    homography, inlier_mask = cv2.findHomography(
        source_valid,
        target_valid,
        cv2.RANSAC,
        config.ransac_reprojection_threshold_px,
    )
    if homography is None or inlier_mask is None or not np.all(np.isfinite(homography)):
        return _GeometryEstimate(
            None, tracked_count, 0, 0.0, "RANSAC не нашёл конечную гомографию"
        )
    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_ratio = inlier_count / tracked_count
    if (
        inlier_count < config.minimum_tracked_features
        or inlier_ratio < config.minimum_inlier_ratio
    ):
        return _GeometryEstimate(
            None,
            tracked_count,
            inlier_count,
            inlier_ratio,
            "Гомография поддерживается недостаточной долей точек",
        )
    homography = homography.astype(np.float64)
    if abs(float(homography[2, 2])) < 1e-12:
        return _GeometryEstimate(
            None, tracked_count, inlier_count, inlier_ratio, "Гомография вырождена"
        )
    homography /= homography[2, 2]
    return _GeometryEstimate(
        homography, tracked_count, inlier_count, inlier_ratio
    )


def _find_local_change(
    previous_gray: GrayImage,
    current_gray: GrayImage,
    homography: NDArray[np.float64],
    *,
    frame_index: int,
    source_scale: float,
    config: VisualEventDetectorConfig,
) -> tuple[EventCandidate | None, float, float]:
    """Совместить кадры и выбрать наиболее выраженную локальную компоненту."""

    height, width = current_gray.shape
    warped = cv2.warpPerspective(
        previous_gray,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = cv2.warpPerspective(
        np.full_like(previous_gray, 255),
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # Эрозия исключает полосу интерполяции у края преобразованного изображения.
    valid = cv2.erode(valid, np.ones((5, 5), dtype=np.uint8))
    valid_pixels = valid > 0
    if np.count_nonzero(valid_pixels) < 0.5 * current_gray.size:
        return None, config.minimum_residual_threshold, 0.0

    # Медианный сдвиг компенсирует общий скачок экспозиции. Он не подгоняет
    # локальное событие, поскольку событие занимает меньшинство кадра.
    photometric_shift = float(
        np.median(
            current_gray[valid_pixels].astype(np.float32)
            - warped[valid_pixels].astype(np.float32)
        )
    )
    corrected = np.clip(warped.astype(np.float32) + photometric_shift, 0.0, 255.0)
    residual = np.abs(current_gray.astype(np.float32) - corrected)
    residual[~valid_pixels] = 0.0
    residual = cv2.GaussianBlur(residual, (5, 5), 0.0)

    valid_residual = residual[valid_pixels]
    median = float(np.median(valid_residual))
    mad = float(np.median(np.abs(valid_residual - median)))
    robust_sigma = 1.4826 * mad
    threshold = max(
        config.minimum_residual_threshold,
        median + config.residual_mad_multiplier * robust_sigma,
    )
    changed = np.zeros_like(current_gray, dtype=np.uint8)
    changed[(residual >= threshold) & valid_pixels] = 255
    changed = cv2.morphologyEx(
        changed, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
    )
    changed = cv2.morphologyEx(
        changed, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)
    )
    changed_fraction = float(np.count_nonzero(changed)) / float(
        np.count_nonzero(valid_pixels)
    )
    if changed_fraction > config.maximum_global_change_fraction:
        return None, threshold, changed_fraction

    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        changed, connectivity=8
    )
    valid_area = float(np.count_nonzero(valid_pixels))
    best_label: int | None = None
    best_strength = -np.inf
    for label in range(1, component_count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        area_fraction = area / valid_area
        if area < config.minimum_component_area_px:
            continue
        if area_fraction > config.maximum_component_fraction:
            continue
        component_values = residual[labels == label]
        mean_residual = float(component_values.mean())
        if (
            mean_residual
            < threshold * config.minimum_component_mean_to_threshold_ratio
        ):
            continue
        strength = mean_residual * area
        if strength > best_strength:
            best_strength = strength
            best_label = label
    if best_label is None:
        return None, threshold, changed_fraction

    x = int(statistics[best_label, cv2.CC_STAT_LEFT])
    y = int(statistics[best_label, cv2.CC_STAT_TOP])
    box_width = int(statistics[best_label, cv2.CC_STAT_WIDTH])
    box_height = int(statistics[best_label, cv2.CC_STAT_HEIGHT])
    area = int(statistics[best_label, cv2.CC_STAT_AREA])
    component_mask = labels == best_label
    weights = residual[component_mask].astype(np.float64)
    rows, columns = np.nonzero(component_mask)
    if float(weights.sum()) > 0.0:
        center_x = float(np.average(columns, weights=weights))
        center_y = float(np.average(rows, weights=weights))
    else:
        center_x = x + 0.5 * box_width
        center_y = y + 0.5 * box_height
    inverse_scale = 1.0 / source_scale
    return (
        EventCandidate(
            frame_index=frame_index,
            center_px=(center_x * inverse_scale, center_y * inverse_scale),
            bounding_box_px=(
                x * inverse_scale,
                y * inverse_scale,
                box_width * inverse_scale,
                box_height * inverse_scale,
            ),
            area_fraction=area / valid_area,
            mean_residual=float(residual[component_mask].mean()),
            threshold=threshold,
        ),
        threshold,
        changed_fraction,
    )


def _confirm_event(
    candidates: deque[EventCandidate | None],
    *,
    confirmation_frame_index: int,
    required_hits: int,
) -> ConfirmedVisualEvent | None:
    """Подтвердить кандидатов, центры которых согласуются в коротком окне."""

    present = [candidate for candidate in candidates if candidate is not None]
    if len(present) < required_hits:
        return None
    earliest = present[0]
    compatible = [earliest]
    earliest_width = earliest.bounding_box_px[2]
    earliest_height = earliest.bounding_box_px[3]
    for candidate in present[1:]:
        distance = float(
            np.linalg.norm(
                np.asarray(candidate.center_px) - np.asarray(earliest.center_px)
            )
        )
        candidate_scale = max(
            earliest_width,
            earliest_height,
            candidate.bounding_box_px[2],
            candidate.bounding_box_px[3],
            40.0,
        )
        if distance <= candidate_scale:
            compatible.append(candidate)
    if len(compatible) < required_hits:
        return None
    confirmed = compatible[-1]
    return ConfirmedVisualEvent(
        onset_frame_index=earliest.frame_index,
        confirmation_frame_index=confirmation_frame_index,
        point_px=confirmed.center_px,
        bounding_box_px=confirmed.bounding_box_px,
    )


def _elapsed_ms(started: float) -> float:
    """Перевести монотонную длительность в миллисекунды."""

    return (time.perf_counter() - started) * 1000.0
