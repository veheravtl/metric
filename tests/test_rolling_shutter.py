"""Проверки физической согласованности плоскостной rolling-shutter модели."""

import numpy as np

from aerial_mapper.radiance_camera import PinholeIntrinsics
from aerial_mapper.rolling_shutter import (
    RollingShutterMotion,
    pose_at_scan_fraction,
    project_world_points_rolling_shutter,
    scan_fraction,
    warp_planar_rolling_shutter,
)


def test_scan_fraction_uses_pixel_centres() -> None:
    """Первая и последняя строки должны симметрично окружать середину."""

    values = scan_fraction(np.array([0.0, 3.0]), 4)
    np.testing.assert_allclose(values, [-0.375, 0.375])


def test_pose_uses_total_top_to_bottom_local_translation() -> None:
    """Разница поз при долях ±0,5 равна заданному полному движению."""

    centre = np.eye(4)
    motion = RollingShutterMotion((2.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    top = pose_at_scan_fraction(centre, motion, -0.5)
    bottom = pose_at_scan_fraction(centre, motion, 0.5)
    np.testing.assert_allclose(bottom[:3, 3] - top[:3, 3], [2.0, 0.0, 0.0])


def test_identity_warp_is_pixel_exact() -> None:
    """Нулевая траектория не должна менять RGB или координаты remap."""

    intrinsics = PinholeIntrinsics(8, 6, 8.0, 8.0, 4.0, 3.0)
    image = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(6, 8, 3)
    centre = np.eye(4)
    centre[2, 3] = 10.0
    motion = RollingShutterMotion((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    result = warp_planar_rolling_shutter(image, centre, intrinsics, motion)

    np.testing.assert_array_equal(result.image_rgb, image)
    assert result.valid_mask.all()
    assert result.maximum_displacement_px < 1e-12


def test_projected_point_satisfies_its_own_row_pose() -> None:
    """Найденная строка должна повторно проецироваться в саму себя."""

    intrinsics = PinholeIntrinsics(960, 720, 640.0, 640.0, 480.0, 360.0)
    centre = np.eye(4)
    centre[2, 3] = 45.0
    points = np.array([[-20.0, -15.0, 0.0], [0.0, 0.0, 0.0], [15.0, 20.0, 0.0]])
    motion = RollingShutterMotion((1.0, 0.0, 0.0), (0.0, 0.0, 2.0))

    result = project_world_points_rolling_shutter(
        points,
        centre,
        intrinsics,
        motion,
        tolerance_px=1e-8,
    )

    assert result.converged.all()
    assert np.max(result.row_residual_px) <= 1e-8
    assert np.max(result.displacement_from_centre_pose_px) > 1.0
