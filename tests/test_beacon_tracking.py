"""Проверки сопровождения операторских маяков между соседними кадрами."""

import cv2
import numpy as np
import pytest

from aerial_mapper.beacon_tracking import (
    BeaconTrackingFailure,
    track_beacons_to_map,
    track_beacons_with_machine_support_to_map,
)
from aerial_mapper.manual_alignment import estimate_manual_alignment


def _textured_frame() -> np.ndarray:
    """Создать воспроизводимую текстуру с углами для оптического потока."""

    random_generator = np.random.default_rng(20261105)
    gray = random_generator.integers(0, 256, size=(240, 320), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
    for x, y in ((45, 45), (160, 40), (275, 50), (55, 180), (165, 190), (270, 175)):
        cv2.drawMarker(gray, (x, y), 255, cv2.MARKER_CROSS, 15, 2)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_tracks_translation_and_recovers_current_frame_to_map_homography() -> None:
    """Небольшой перенос не должен заставлять повторно размечать карту."""

    previous = _textured_frame()
    motion_previous_to_current = np.asarray(
        [[1.0, 0.0, 4.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    current = cv2.warpPerspective(previous, motion_previous_to_current, (320, 240))
    # Оператор должен ставить маяк на локальный угол, а не просто на заметный
    # объект. Здесь такие точки воспроизводимо выбирает стандартный детектор
    # углов; сам тестируемый алгоритм всё равно получает только готовые точки.
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    detected = cv2.goodFeaturesToTrack(
        previous_gray, maxCorners=20, qualityLevel=0.05, minDistance=20
    )
    assert detected is not None
    previous_points = detected.reshape(-1, 2).astype(np.float64)
    frame_to_map_at_anchor = np.asarray(
        [[1.15, 0.03, 30.0], [-0.02, 1.10, 25.0], [0.0001, -0.0001, 1.0]],
        dtype=np.float64,
    )
    map_points = cv2.perspectiveTransform(
        previous_points.reshape(1, -1, 2), frame_to_map_at_anchor
    ).reshape(-1, 2)

    result = track_beacons_to_map(
        previous,
        current,
        previous_points,
        map_points,
        map_size_pixels=(500, 400),
    )

    expected_current_points = cv2.perspectiveTransform(
        previous_points.reshape(1, -1, 2), motion_previous_to_current
    ).reshape(-1, 2)
    expected_current_to_map = frame_to_map_at_anchor @ np.linalg.inv(
        motion_previous_to_current
    )
    expected_current_to_map /= expected_current_to_map[2, 2]

    assert result.tracked_count >= 10
    assert result.inlier_count >= 10
    accepted = result.homography_inlier_mask
    assert result.current_frame_points_px[accepted] == pytest.approx(
        expected_current_points[accepted], abs=0.15
    )
    # Коэффициенты гомографии чувствительны к малой субпиксельной ошибке и сами
    # по себе не являются физической метрикой. Сравниваем положение независимых
    # точек по всей площади кадра — именно оно определяет качество наложения.
    probes = np.asarray(
        [[0.0, 0.0], [319.0, 0.0], [319.0, 239.0], [0.0, 239.0], [160.0, 120.0]]
    )
    predicted_map = cv2.perspectiveTransform(
        probes.reshape(1, -1, 2), result.alignment.homography_frame_to_map
    ).reshape(-1, 2)
    expected_map = cv2.perspectiveTransform(
        probes.reshape(1, -1, 2), expected_current_to_map
    ).reshape(-1, 2)
    independent_errors = np.linalg.norm(predicted_map - expected_map, axis=1)
    assert np.max(independent_errors) < 1.0
    assert result.accepted_forward_backward_max_px < 0.5


def test_blank_frame_is_rejected_instead_of_returning_alignment() -> None:
    """Исчезновение текстуры должно дать явный отказ, а не старую матрицу."""

    previous = _textured_frame()
    current = np.zeros_like(previous)
    points = np.asarray(
        [[45, 45], [160, 40], [275, 50], [55, 180], [165, 190], [270, 175]],
        dtype=np.float64,
    )

    with pytest.raises(BeaconTrackingFailure, match="Сопровождение потеряно") as caught:
        track_beacons_to_map(
            previous,
            current,
            points,
            points,
            map_size_pixels=(320, 240),
        )

    assert caught.value.current_frame_points_px.shape == (6, 2)
    assert caught.value.retained_mask.shape == (6,)
    assert caught.value.failure_stage == "flow_validation"
    assert caught.value.forward_backward_errors_px.shape == (6,)


def test_machine_support_keeps_geometry_with_one_bad_manual_beacon() -> None:
    """Машинные углы помогают, но ошибочный ручной маяк остаётся видимым."""

    previous = _textured_frame()
    motion_previous_to_current = np.asarray(
        [[1.0, 0.0, 4.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    current = cv2.warpPerspective(previous, motion_previous_to_current, (320, 240))
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    detected = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=40,
        qualityLevel=0.05,
        minDistance=15,
        blockSize=7,
    )
    assert detected is not None
    candidate_points = detected.reshape(-1, 2).astype(np.float64)
    anchor_homography = np.asarray(
        [[1.15, 0.03, 30.0], [-0.02, 1.10, 25.0], [0.0001, -0.0001, 1.0]],
        dtype=np.float64,
    )
    candidate_map_points = cv2.perspectiveTransform(
        candidate_points.reshape(1, -1, 2), anchor_homography
    ).reshape(-1, 2)

    # Сначала оставляем точки, воспроизводимо прошедшие обычную проверку потока.
    # Затем жадно выбираем четыре наиболее разнесённые тестовые опоры.
    candidate_tracking = track_beacons_to_map(
        previous,
        current,
        candidate_points,
        candidate_map_points,
        map_size_pixels=(500, 400),
    )
    valid_candidates = candidate_points[candidate_tracking.homography_inlier_mask]
    selected = [valid_candidates[np.argmin(np.sum(valid_candidates, axis=1))]]
    for _ in range(3):
        distances = np.min(
            np.linalg.norm(
                valid_candidates[:, None, :] - np.asarray(selected)[None, :, :],
                axis=2,
            ),
            axis=1,
        )
        selected.append(valid_candidates[np.argmax(distances)])
    manual_points = np.asarray(selected)
    correct_map_points = cv2.perspectiveTransform(
        manual_points.reshape(1, -1, 2), anchor_homography
    ).reshape(-1, 2)
    anchor_alignment = estimate_manual_alignment(
        manual_points,
        correct_map_points,
        frame_size_pixels=(320, 240),
        map_size_pixels=(500, 400),
    )
    one_bad_map_point = correct_map_points.copy()
    one_bad_map_point[0] += [80.0, -60.0]

    result = track_beacons_with_machine_support_to_map(
        previous,
        current,
        manual_points,
        one_bad_map_point,
        previous_alignment=anchor_alignment,
        map_size_pixels=(500, 400),
    )

    assert result.inlier_count == 3
    assert result.automatic_detected_count >= 20
    assert result.automatic_inlier_count >= 20
    expected_current_to_map = anchor_homography @ np.linalg.inv(
        motion_previous_to_current
    )
    expected_current_to_map /= expected_current_to_map[2, 2]
    probes = np.asarray([[0.0, 0.0], [319.0, 0.0], [319.0, 239.0], [0.0, 239.0]])
    actual_map = cv2.perspectiveTransform(
        probes.reshape(1, -1, 2), result.alignment.homography_frame_to_map
    ).reshape(-1, 2)
    expected_map = cv2.perspectiveTransform(
        probes.reshape(1, -1, 2), expected_current_to_map
    ).reshape(-1, 2)
    assert np.max(np.linalg.norm(actual_map - expected_map, axis=1)) < 1.0
