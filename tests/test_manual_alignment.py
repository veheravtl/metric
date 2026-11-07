"""Проверки ручного совмещения кадра с картой по парным точкам."""

import cv2
import numpy as np
import pytest

from aerial_mapper.manual_alignment import (
    ManualAlignmentFailure,
    assess_manual_alignment_quality,
    estimate_manual_alignment,
)


def _transform(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        points.astype(np.float64).reshape(1, -1, 2), homography
    ).reshape(-1, 2)


def test_manual_points_recover_known_projective_transform() -> None:
    """Разнесённые точные пары должны восстановить скрытую гомографию."""

    frame_points = np.asarray(
        [
            [80.0, 70.0],
            [500.0, 60.0],
            [920.0, 90.0],
            [100.0, 390.0],
            [510.0, 370.0],
            [900.0, 410.0],
            [130.0, 710.0],
            [870.0, 690.0],
        ]
    )
    true_homography = np.asarray(
        [
            [0.95, 0.08, 120.0],
            [-0.04, 0.92, 90.0],
            [0.00008, -0.00005, 1.0],
        ]
    )
    map_points = _transform(frame_points, true_homography)

    result = estimate_manual_alignment(
        frame_points,
        map_points,
        frame_size_pixels=(1000, 800),
        map_size_pixels=(1300, 1000),
    )

    expected_corners = _transform(
        np.asarray([[0.0, 0.0], [999.0, 0.0], [999.0, 799.0], [0.0, 799.0]]),
        true_homography,
    )
    assert result.projected_frame_corners_map_px == pytest.approx(
        expected_corners, abs=5e-5
    )
    assert result.fit_rmse_map_px < 1e-5
    assert result.leave_one_out_median_map_px is not None
    assert result.leave_one_out_median_map_px < 1e-4
    assert result.frame_coverage_fraction > 0.6


def test_four_points_have_no_independent_leave_one_out_check() -> None:
    """После исключения одной из четырёх пар гомографию оценить уже нельзя."""

    points = np.asarray([[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]])
    result = estimate_manual_alignment(
        points,
        points,
        frame_size_pixels=(100, 100),
        map_size_pixels=(100, 100),
    )

    assert result.fit_rmse_map_px < 1e-8
    assert result.leave_one_out_median_map_px is None
    assert result.leave_one_out_max_map_px is None


def test_leave_one_out_exposes_inconsistent_manual_click() -> None:
    """Одна ошибочная пара не должна скрываться почти хорошей общей подгонкой."""

    frame_points = np.asarray(
        [
            [10.0, 10.0],
            [90.0, 10.0],
            [90.0, 90.0],
            [10.0, 90.0],
            [50.0, 20.0],
            [50.0, 80.0],
        ]
    )
    map_points = frame_points.copy()
    map_points[-1] += [12.0, -8.0]

    result = estimate_manual_alignment(
        frame_points,
        map_points,
        frame_size_pixels=(100, 100),
        map_size_pixels=(120, 120),
    )

    assert result.leave_one_out_max_map_px is not None
    assert result.leave_one_out_max_map_px > 10.0


def test_quality_gate_rejects_one_mismatched_pair_and_identifies_it() -> None:
    """Одна перепутанная пара не должна превращаться в искажённое наложение."""

    frame_points = np.asarray(
        [
            [80.0, 70.0],
            [500.0, 60.0],
            [920.0, 90.0],
            [100.0, 390.0],
            [510.0, 370.0],
            [900.0, 410.0],
            [130.0, 710.0],
            [870.0, 690.0],
        ]
    )
    true_homography = np.asarray(
        [
            [0.95, 0.08, 120.0],
            [-0.04, 0.92, 90.0],
            [0.00008, -0.00005, 1.0],
        ]
    )
    map_points = _transform(frame_points, true_homography)
    mismatched_index = 6
    map_points[mismatched_index] += [120.0, -180.0]

    result = estimate_manual_alignment(
        frame_points,
        map_points,
        frame_size_pixels=(1000, 800),
        map_size_pixels=(1300, 1000),
    )
    quality = assess_manual_alignment_quality(
        result,
        frame_size_pixels=(1000, 800),
        map_size_pixels=(1300, 1000),
    )

    assert not quality.is_acceptable
    assert quality.suspected_outlier_index == mismatched_index
    assert quality.fit_rmse_without_suspected_map_px is not None
    assert quality.fit_rmse_without_suspected_map_px < 1e-4


def test_collinear_points_are_rejected() -> None:
    """Точки на одной линии не задают преобразование плоскости."""

    points = np.asarray([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])

    with pytest.raises(ManualAlignmentFailure, match="гомограф"):
        estimate_manual_alignment(
            points,
            points,
            frame_size_pixels=(100, 100),
            map_size_pixels=(100, 100),
        )
