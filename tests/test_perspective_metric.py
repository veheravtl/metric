"""Тесты метрической разметки перспективного Teach-кадра."""

import cv2
import numpy as np
import pytest

from aerial_mapper.measurement import MeasurementFailure
from aerial_mapper.perspective_metric import (
    calibrate_perspective_reference,
    measure_ground_segment_via_perspective_reference,
)


def _project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        points.astype(np.float64).reshape(1, -1, 2),
        homography,
    ).reshape(-1, 2)


def test_perspective_teach_calibration_recovers_metric_segment() -> None:
    """Две неизвестные перспективы не должны разрушать длину на одной плоскости."""

    world_to_teach = np.array(
        [[31.0, 3.0, 410.0], [-2.0, -27.0, 330.0], [0.012, -0.006, 1.0]],
        dtype=np.float64,
    )
    repeat_to_world = np.array(
        [[0.055, 0.006, -18.0], [-0.004, -0.061, 15.0], [0.00008, -0.00004, 1.0]],
        dtype=np.float64,
    )
    calibration_world = np.array(
        [[-12.0, -9.0], [12.0, -9.0], [13.0, 10.0], [-13.0, 10.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    teach_pixels = _project(calibration_world, world_to_teach)
    calibration = calibrate_perspective_reference(teach_pixels, calibration_world)

    segment_repeat = np.array([[300.0, 220.0], [360.0, 260.0]], dtype=np.float64)
    frame_to_teach = world_to_teach @ repeat_to_world
    expected_world = _project(segment_repeat, repeat_to_world)
    measurement = measure_ground_segment_via_perspective_reference(
        segment_repeat,
        frame_to_teach,
        calibration=calibration,
        reference_width_pixels=1000,
        reference_height_pixels=800,
    )

    expected_length = np.linalg.norm(expected_world[1] - expected_world[0])
    assert calibration.control_reprojection_max_m < 1e-6
    assert measurement.mapped_points.world_points_m == pytest.approx(
        expected_world, abs=1e-6
    )
    assert measurement.length_meters == pytest.approx(expected_length, abs=1e-6)


def test_perspective_teach_calibration_rejects_too_few_points() -> None:
    """Три пары не определяют восемь степеней свободы гомографии."""

    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(MeasurementFailure, match="не менее 4"):
        calibrate_perspective_reference(points, points)


def test_perspective_measurement_rejects_point_outside_teach() -> None:
    """Экстраполяция за размеченный кадр должна давать отказ."""

    control_px = np.array(
        [[0.0, 0.0], [99.0, 0.0], [99.0, 99.0], [0.0, 99.0]],
    )
    calibration = calibrate_perspective_reference(control_px, control_px)
    with pytest.raises(MeasurementFailure, match="за границы Teach"):
        measure_ground_segment_via_perspective_reference(
            np.array([[20.0, 20.0], [120.0, 20.0]]),
            np.eye(3),
            calibration=calibration,
            reference_width_pixels=100,
            reference_height_pixels=100,
        )
