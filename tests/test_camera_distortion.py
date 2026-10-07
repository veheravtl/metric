"""Проверки синтетической модели дисторсии камеры."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from aerial_mapper.camera_distortion import (
    BrownConradyDistortion,
    distort_image,
    distort_points,
    undistort_image,
    undistort_points,
)
from aerial_mapper.radiance_camera import blender_horizontal_sensor_intrinsics


@pytest.fixture
def intrinsics():  # type: ignore[no-untyped-def]
    """Возвращает ту же камеру 960 × 720, что используется в G12--G14."""

    return blender_horizontal_sensor_intrinsics(
        width_px=960,
        height_px=720,
        focal_length_mm=24.0,
        sensor_width_mm=36.0,
    )


def test_zero_distortion_keeps_points_exact(intrinsics) -> None:  # type: ignore[no-untyped-def]
    """Нулевой контроль не должен менять систему координат пикселей."""

    points = np.asarray([[0.0, 0.0], [480.0, 360.0], [959.0, 719.0]])
    distorted = distort_points(
        points,
        intrinsics=intrinsics,
        distortion=BrownConradyDistortion(),
    )
    assert distorted == pytest.approx(points, abs=1e-10)


def test_barrel_distortion_moves_corner_towards_principal_point(intrinsics) -> None:  # type: ignore[no-untyped-def]
    """Отрицательный k1 обязан уменьшать радиус точки, а не увеличивать его."""

    centre = np.asarray([intrinsics.principal_x_px, intrinsics.principal_y_px])
    point = np.asarray([[959.0, 719.0]])
    distorted = distort_points(
        point,
        intrinsics=intrinsics,
        distortion=BrownConradyDistortion(k1=-0.1),
    )
    assert np.linalg.norm(distorted[0] - centre) < np.linalg.norm(point[0] - centre)


@pytest.mark.parametrize("k1", [-0.1, -0.025, 0.025, 0.1])
def test_point_round_trip_is_sub_millipixel(intrinsics, k1: float) -> None:  # type: ignore[no-untyped-def]
    """Скрытая истина и RGB должны использовать взаимно согласованную модель."""

    x, y = np.meshgrid(np.linspace(40, 920, 9), np.linspace(40, 680, 7))
    points = np.column_stack((x.ravel(), y.ravel()))
    distortion = BrownConradyDistortion(k1=k1)
    distorted = distort_points(
        points, intrinsics=intrinsics, distortion=distortion
    )
    restored = undistort_points(
        distorted, intrinsics=intrinsics, distortion=distortion
    )
    maximum_error = np.linalg.norm(restored - points, axis=1).max()
    assert maximum_error < 0.001


def test_image_warp_matches_distorted_point(intrinsics) -> None:  # type: ignore[no-untyped-def]
    """Яркая метка должна оказаться там же, куда модель переносит её центр."""

    image = np.zeros((720, 960, 3), dtype=np.uint8)
    source_point = np.asarray([[820.0, 600.0]])
    cv2.circle(image, (820, 600), 8, (255, 255, 255), thickness=-1)
    distortion = BrownConradyDistortion(k1=-0.1)
    warped = distort_image(
        image, intrinsics=intrinsics, distortion=distortion
    )
    expected = distort_points(
        source_point, intrinsics=intrinsics, distortion=distortion
    )[0]
    luminance = warped.mean(axis=2)
    observed_y, observed_x = np.unravel_index(np.argmax(luminance), luminance.shape)
    assert [observed_x, observed_y] == pytest.approx(expected, abs=8.0)


def test_distort_then_undistort_preserves_shape_and_type(intrinsics) -> None:  # type: ignore[no-untyped-def]
    """Калибровочная коррекция не должна менять контракт RGB-массива."""

    image = np.zeros((720, 960, 3), dtype=np.uint8)
    image[200:500, 300:700] = [40, 120, 220]
    distortion = BrownConradyDistortion(k1=0.05)
    distorted = distort_image(
        image, intrinsics=intrinsics, distortion=distortion
    )
    corrected = undistort_image(
        distorted, intrinsics=intrinsics, distortion=distortion
    )
    assert corrected.shape == image.shape
    assert corrected.dtype == np.uint8


def test_non_finite_distortion_is_rejected(intrinsics) -> None:  # type: ignore[no-untyped-def]
    """NaN в протоколе нельзя молча превратить в пустое изображение."""

    with pytest.raises(ValueError, match="конечными"):
        distort_points(
            np.asarray([[1.0, 2.0]]),
            intrinsics=intrinsics,
            distortion=BrownConradyDistortion(k1=float("nan")),
        )


def test_g14a_protocol_keeps_control_and_distortion_sweep_frozen() -> None:
    """Число попыток и уровни k1 нельзя незаметно менять после результата."""

    path = Path("experiments/configs/synthetic_3d_g14a_lens_distortion.json")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    coefficients = [case["k1"] for case in protocol["distortion_cases"]]

    assert coefficients == [0.0, -0.025, -0.05, -0.1, 0.025, 0.05, 0.1]
    assert protocol["pipelines"] == ["raw", "calibrated"]
    assert len(protocol["surface_ids"]) == 3
    assert len(protocol["repeat_camera_ids"]) == 3
    nominal_attempts = len(protocol["surface_ids"]) * len(
        protocol["repeat_camera_ids"]
    )
    nonzero_attempts = (
        nominal_attempts
        * (len(coefficients) - 1)
        * len(protocol["pipelines"])
    )
    assert nominal_attempts + nonzero_attempts == 117
    assert protocol["maximum_point_position_error_m"] == 0.2
    assert (
        protocol["preregistered_raw_safety"]["maximum_false_accepts"] == 0
    )
def test_imperfect_k1_correction_leaves_measurable_residual(intrinsics) -> None:  # type: ignore[no-untyped-def]
    """Неточный k1 не должен случайно использовать идеальные pinhole-точки."""

    nominal = np.asarray([[40.0, 40.0], [920.0, 680.0], [480.0, 360.0]])
    observed = distort_points(
        nominal,
        intrinsics=intrinsics,
        distortion=BrownConradyDistortion(k1=0.05),
    )
    exact = undistort_points(
        observed,
        intrinsics=intrinsics,
        distortion=BrownConradyDistortion(k1=0.05),
    )
    imperfect = undistort_points(
        observed,
        intrinsics=intrinsics,
        distortion=BrownConradyDistortion(k1=0.04),
    )

    assert exact == pytest.approx(nominal, abs=0.001)
    assert np.linalg.norm(imperfect - nominal, axis=1).max() > 1.0


def test_g14a2_protocol_keeps_attempt_matrix_frozen() -> None:
    """G14-A2 нельзя после результата сузить удалением неудобного уровня."""

    path = Path(
        "experiments/configs/synthetic_3d_g14a2_calibration_uncertainty.json"
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))

    assert protocol["raw_boundary_k1"] == [
        -0.025,
        -0.02,
        -0.015,
        -0.01,
        -0.005,
        0.005,
        0.01,
        0.015,
        0.02,
        0.025,
    ]
    assert protocol["calibration_true_k1"] == [-0.05, -0.025, 0.025, 0.05]
    assert protocol["calibration_error_k1"] == [
        -0.02,
        -0.01,
        -0.005,
        -0.0025,
        0.0,
        0.0025,
        0.005,
        0.01,
        0.02,
    ]
    scene_pose_count = len(protocol["surface_ids"]) * len(
        protocol["repeat_camera_ids"]
    )
    attempts_per_scene_pose = (
        1
        + len(protocol["raw_boundary_k1"])
        + len(protocol["calibration_true_k1"])
        * len(protocol["calibration_error_k1"])
    )
    assert scene_pose_count * attempts_per_scene_pose == 423
    assert protocol["maximum_point_position_error_m"] == 0.2
    assert (
        protocol["preregistered_imperfect_calibration_safety"][
            "maximum_false_accepts"
        ]
        == 0
    )
