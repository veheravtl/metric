# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Проверки минимального детектора локального изменения сцены."""

import cv2
import numpy as np
import pytest

from aerial_mapper.visual_event_detection import (
    VisualEventDetector,
    VisualEventDetectorConfig,
)


def _textured_scene() -> np.ndarray:
    """Создать воспроизводимую текстуру с признаками во всём кадре."""

    generator = np.random.default_rng(20261113)
    noise = generator.integers(0, 256, (240, 320), dtype=np.uint8)
    gray = cv2.GaussianBlur(noise, (0, 0), 1.2)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _translated(scene: np.ndarray, frame_index: int) -> np.ndarray:
    """Имитировать равномерное движение камеры без изменения сцены."""

    return cv2.warpAffine(
        scene,
        np.float32([[1.0, 0.0, 2.0 * frame_index], [0.0, 1.0, frame_index]]),
        (scene.shape[1], scene.shape[0]),
    )


def _detector() -> VisualEventDetector:
    return VisualEventDetector(
        VisualEventDetectorConfig(
            analysis_width_px=320,
            minimum_component_area_px=50,
            minimum_residual_threshold=18.0,
        )
    )


def test_camera_translation_does_not_create_event() -> None:
    """Компенсированное движение камеры не должно считаться событием."""

    scene = _textured_scene()
    detector = _detector()
    statuses = []
    for frame_index in range(6):
        step = detector.advance(
            _translated(scene, frame_index), frame_index=frame_index
        )
        statuses.append(step.status)

    assert statuses[0] == "warming_up"
    assert set(statuses[1:]) == {"monitoring"}
    assert detector.confirmed_event is None


def test_local_change_is_confirmed_and_backdated_to_first_candidate() -> None:
    """Два локальных изменения подтверждают событие с моментом первого из них."""

    scene = _textured_scene()
    detector = _detector()
    steps = []
    for frame_index in range(4):
        frame = _translated(scene, frame_index)
        if frame_index >= 2:
            cv2.circle(
                frame,
                (164 + 2 * frame_index, 122 + frame_index),
                18,
                (255, 255, 255),
                -1,
            )
        steps.append(detector.advance(frame, frame_index=frame_index))

    assert steps[2].status == "candidate"
    assert steps[3].status == "confirmed"
    event = steps[3].confirmed_event
    assert event is not None
    assert event.onset_frame_index == 2
    assert event.confirmation_frame_index == 3
    assert steps[3].candidate is not None
    assert event.point_px == pytest.approx(steps[3].candidate.center_px)
    assert event.point_px == pytest.approx((170.0, 125.0), abs=3.0)


def test_global_brightness_jump_is_removed_as_exposure_change() -> None:
    """Общий сдвиг яркости не должен порождать локальную тревогу."""

    scene = _textured_scene()
    detector = _detector()
    detector.advance(scene, frame_index=0)
    brighter = np.clip(scene.astype(np.int16) + 35, 0, 255).astype(np.uint8)

    step = detector.advance(brighter, frame_index=1)

    assert step.status == "monitoring"
    assert step.candidate is None


def test_texture_loss_gives_explicit_geometry_failure() -> None:
    """Без подтверждённой геометрии запрещено выдавать событие."""

    detector = _detector()
    detector.advance(_textured_scene(), frame_index=0)

    step = detector.advance(np.zeros((240, 320, 3), dtype=np.uint8), frame_index=1)

    assert step.status == "invalid_geometry"
    assert step.candidate is None
    assert step.reason


def test_nonsequential_frame_resets_temporal_confirmation() -> None:
    """Перемотка не должна склеивать кандидатов из разных участков времени."""

    detector = _detector()
    detector.advance(_textured_scene(), frame_index=2)

    step = detector.advance(_textured_scene(), frame_index=9)

    assert step.status == "warming_up"
    assert step.reason is not None
    assert "сброшена" in step.reason
