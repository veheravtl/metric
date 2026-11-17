# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Проверки локального сопровождения выбранной оператором области."""

import cv2
import numpy as np
import pytest

from aerial_mapper.local_roi_tracking import (
    LocalRoiTracker,
    LocalRoiTrackingFailure,
)


def _textured_frame() -> np.ndarray:
    """Создать воспроизводимую текстуру, пригодную для оптического потока."""

    generator = np.random.default_rng(20261113)
    return generator.integers(0, 256, (120, 160, 3), dtype=np.uint8)


def test_tracker_follows_known_translation_of_local_surface() -> None:
    """Центр ROI должен повторить известный сдвиг текстуры между кадрами."""

    first = _textured_frame()
    translation_x = 4.0
    translation_y = -3.0
    second = cv2.warpAffine(
        first,
        np.float32(
            [
                [1.0, 0.0, translation_x],
                [0.0, 1.0, translation_y],
            ]
        ),
        (first.shape[1], first.shape[0]),
    )
    tracker = LocalRoiTracker()
    tracker.capture(first, frame_index=7, point_px=(80.0, 60.0))

    step = tracker.advance(second, frame_index=8)

    assert step.frame_index == 8
    assert step.point_px == pytest.approx((84.0, 57.0), abs=0.05)
    assert step.forward_backward_error_px < 0.1
    assert tracker.is_active
    assert tracker.point_px == pytest.approx(step.point_px)


def test_tracker_rejects_roi_that_does_not_fit_inside_frame() -> None:
    """Полуобрезанная область у края не должна выглядеть успешным захватом."""

    tracker = LocalRoiTracker()

    with pytest.raises(LocalRoiTrackingFailure, match="не помещается"):
        tracker.capture(_textured_frame(), frame_index=0, point_px=(10.0, 10.0))

    assert not tracker.is_active


def test_tracker_clears_state_when_texture_is_lost() -> None:
    """Исчезновение текстуры должно дать явный отказ вместо дрейфа точки."""

    tracker = LocalRoiTracker()
    tracker.capture(_textured_frame(), frame_index=0, point_px=(80.0, 60.0))
    blank = np.zeros((120, 160, 3), dtype=np.uint8)

    with pytest.raises(LocalRoiTrackingFailure, match="потеряна"):
        tracker.advance(blank, frame_index=1)

    assert not tracker.is_active


def test_tracker_rejects_discontinuous_frame_sequence() -> None:
    """После ручного скачка по времени прежняя область больше не подтверждена."""

    tracker = LocalRoiTracker()
    tracker.capture(_textured_frame(), frame_index=3, point_px=(80.0, 60.0))

    with pytest.raises(LocalRoiTrackingFailure, match="ожидался 4, получен 9"):
        tracker.advance(_textured_frame(), frame_index=9)

    assert not tracker.is_active
