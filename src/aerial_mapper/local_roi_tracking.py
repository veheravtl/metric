# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Сопровождение выбранной оператором локальной области между кадрами.

Локальная область интереса — Region of Interest, ROI — задаётся центром и
квадратным окном в пикселях исходного кадра. Пирамидальный метод
Лукаса—Канаде переносит центр на следующий кадр. Затем выполняется обратный
перенос: если точка не возвращается почти в исходное место, сопровождение
считается потерянным и правдоподобная, но неподтверждённая координата наружу
не выдаётся.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

BgrImage = NDArray[np.uint8]
GrayImage = NDArray[np.uint8]


class LocalRoiTrackingFailure(RuntimeError):
    """Выбранная область больше не подтверждается изображением."""


@dataclass(frozen=True)
class LocalRoiTrackerConfig:
    """Параметры локального оптического потока в исходных пикселях."""

    roi_size_px: int = 41
    pyramid_levels: int = 3
    maximum_forward_backward_error_px: float = 1.5
    minimum_eigenvalue: float = 1e-4

    def validate(self) -> None:
        """Проверить параметры до начала интерактивного сопровождения."""

        if self.roi_size_px < 3 or self.roi_size_px % 2 == 0:
            raise ValueError("Размер ROI должен быть нечётным и не меньше 3 px")
        if self.pyramid_levels < 0:
            raise ValueError("Число уровней пирамиды не может быть отрицательным")
        if self.maximum_forward_backward_error_px <= 0.0:
            raise ValueError("Порог ошибки вперёд-назад должен быть положительным")
        if self.minimum_eigenvalue <= 0.0:
            raise ValueError("Порог текстуры должен быть положительным")

    @property
    def roi_radius_px(self) -> int:
        """Радиус квадратной области относительно её центрального пикселя."""

        return self.roi_size_px // 2


@dataclass(frozen=True)
class LocalRoiTrackingStep:
    """Подтверждённый перенос центра ROI на один следующий кадр."""

    frame_index: int
    point_px: tuple[float, float]
    forward_backward_error_px: float


class LocalRoiTracker:
    """Хранить единственный захват и переносить его по соседним кадрам."""

    def __init__(self, config: LocalRoiTrackerConfig | None = None) -> None:
        self.config = config or LocalRoiTrackerConfig()
        self.config.validate()
        self._previous_gray: GrayImage | None = None
        self._frame_index: int | None = None
        self._point_px: tuple[float, float] | None = None

    @property
    def is_active(self) -> bool:
        """Есть ли подтверждённая область, готовая к следующему переходу."""

        return (
            self._previous_gray is not None
            and self._frame_index is not None
            and self._point_px is not None
        )

    @property
    def frame_index(self) -> int | None:
        """Индекс кадра, на котором подтверждена текущая координата."""

        return self._frame_index

    @property
    def point_px(self) -> tuple[float, float] | None:
        """Центр области в пикселях текущего исходного кадра."""

        return self._point_px

    def capture(
        self,
        frame_bgr: BgrImage,
        *,
        frame_index: int,
        point_px: tuple[float, float],
    ) -> None:
        """Заменить прежний захват областью вокруг выбранного пикселя."""

        gray = _as_gray_u8(frame_bgr)
        point = _validated_point(point_px)
        _require_roi_inside(
            point,
            width=gray.shape[1],
            height=gray.shape[0],
            radius=self.config.roi_radius_px,
        )
        if frame_index < 0:
            raise ValueError("Индекс кадра не может быть отрицательным")
        self._previous_gray = gray
        self._frame_index = frame_index
        self._point_px = point

    def advance(
        self,
        frame_bgr: BgrImage,
        *,
        frame_index: int,
    ) -> LocalRoiTrackingStep:
        """Перенести захват строго на один следующий кадр.

        При любом неподтверждённом результате внутреннее состояние очищается.
        Вызывающая сторона должна убрать маркер и может продолжить показ видео.
        """

        if not self.is_active:
            raise LocalRoiTrackingFailure("Нет активной области для сопровождения")
        assert self._previous_gray is not None
        assert self._frame_index is not None
        assert self._point_px is not None
        expected_frame_index = self._frame_index + 1
        if frame_index != expected_frame_index:
            self.clear()
            raise LocalRoiTrackingFailure(
                "Нарушена последовательность кадров: ожидался "
                f"{expected_frame_index}, получен {frame_index}"
            )

        current_gray = _as_gray_u8(frame_bgr)
        if current_gray.shape != self._previous_gray.shape:
            self.clear()
            raise LocalRoiTrackingFailure(
                "Размер кадра изменился во время сопровождения"
            )

        try:
            point, forward_backward_error = _track_one_step(
                self._previous_gray,
                current_gray,
                self._point_px,
                self.config,
            )
            _require_roi_inside(
                point,
                width=current_gray.shape[1],
                height=current_gray.shape[0],
                radius=self.config.roi_radius_px,
            )
        except (LocalRoiTrackingFailure, cv2.error) as error:
            self.clear()
            if isinstance(error, LocalRoiTrackingFailure):
                raise
            raise LocalRoiTrackingFailure(
                f"OpenCV не вычислил локальный оптический поток: {error}"
            ) from error

        self._previous_gray = current_gray
        self._frame_index = frame_index
        self._point_px = point
        return LocalRoiTrackingStep(
            frame_index=frame_index,
            point_px=point,
            forward_backward_error_px=forward_backward_error,
        )

    def clear(self) -> None:
        """Удалить захват и все данные предыдущего кадра."""

        self._previous_gray = None
        self._frame_index = None
        self._point_px = None


def _as_gray_u8(frame: BgrImage) -> GrayImage:
    """Проверить BGR-кадр и создать независимое полутоновое изображение."""

    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Ожидается BGR-кадр uint8 формы H×W×3")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("Пустой кадр нельзя использовать для сопровождения")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _validated_point(point_px: tuple[float, float]) -> tuple[float, float]:
    """Преобразовать координату в конечную пару чисел."""

    point = np.asarray(point_px, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("Центр ROI должен быть конечной парой (x, y)")
    return float(point[0]), float(point[1])


def _require_roi_inside(
    point_px: tuple[float, float],
    *,
    width: int,
    height: int,
    radius: int,
) -> None:
    """Не допустить неполную область у края кадра."""

    x, y = point_px
    if (
        x - radius < 0.0
        or y - radius < 0.0
        or x + radius > width - 1.0
        or y + radius > height - 1.0
    ):
        raise LocalRoiTrackingFailure(
            f"Область {2 * radius + 1}×{2 * radius + 1} px не помещается внутри кадра"
        )


def _track_one_step(
    previous_gray: GrayImage,
    current_gray: GrayImage,
    point_px: tuple[float, float],
    config: LocalRoiTrackerConfig,
) -> tuple[tuple[float, float], float]:
    """Вычислить прямой и обратный перенос одной локальной области."""

    source = np.asarray(point_px, dtype=np.float32).reshape(1, 1, 2)
    parameters = {
        "winSize": (config.roi_size_px, config.roi_size_px),
        "maxLevel": config.pyramid_levels,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
        "minEigThreshold": config.minimum_eigenvalue,
    }
    target, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, source, None, **parameters
    )
    if target is None or forward_status is None or forward_status.item() != 1:
        raise LocalRoiTrackingFailure(
            "Область интереса потеряна: прямой оптический поток не найден"
        )
    returned, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, target, None, **parameters
    )
    if returned is None or backward_status is None or backward_status.item() != 1:
        raise LocalRoiTrackingFailure(
            "Область интереса потеряна: обратная проверка не пройдена"
        )

    candidate = target.reshape(2).astype(np.float64)
    returned_point = returned.reshape(2).astype(np.float64)
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(returned_point)):
        raise LocalRoiTrackingFailure(
            "Область интереса потеряна: получены неконечные координаты"
        )
    forward_backward_error = float(
        np.linalg.norm(source.reshape(2).astype(np.float64) - returned_point)
    )
    if forward_backward_error > config.maximum_forward_backward_error_px:
        raise LocalRoiTrackingFailure(
            "Область интереса потеряна: ошибка вперёд-назад "
            f"{forward_backward_error:.2f} px превышает "
            f"{config.maximum_forward_backward_error_px:.2f} px"
        )
    return (float(candidate[0]), float(candidate[1])), forward_backward_error
