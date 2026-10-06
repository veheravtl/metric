"""Проверки соглашений камеры для подготовительного NeRF/3DGS-набора."""

import numpy as np
import pytest

from aerial_mapper.radiance_camera import (
    blender_horizontal_sensor_intrinsics,
    pixel_rays_world_opengl,
    project_world_points_opengl,
    validate_camera_to_world,
)


def test_blender_intrinsics_match_frozen_synthetic_camera() -> None:
    """24 мм на сенсоре 36 мм и ширине 960 px должны дать фокус 640 px."""

    intrinsics = blender_horizontal_sensor_intrinsics(
        width_px=960,
        height_px=720,
        focal_length_mm=24.0,
        sensor_width_mm=36.0,
    )

    assert intrinsics.focal_x_px == pytest.approx(640.0)
    assert intrinsics.focal_y_px == pytest.approx(640.0)
    assert intrinsics.principal_x_px == pytest.approx(480.0)
    assert intrinsics.principal_y_px == pytest.approx(360.0)
    assert np.degrees(intrinsics.horizontal_field_of_view_rad) == pytest.approx(
        73.739795,
        abs=1e-6,
    )


def test_nadir_camera_projects_world_origin_to_image_centre() -> None:
    """Камера над началом мира должна видеть его центральным лучом вниз."""

    intrinsics = blender_horizontal_sensor_intrinsics(
        width_px=960,
        height_px=720,
        focal_length_mm=24.0,
        sensor_width_mm=36.0,
    )
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[2, 3] = 45.0

    pixels, in_front = project_world_points_opengl(
        np.asarray([[0.0, 0.0, 0.0]]),
        camera_to_world,
        intrinsics,
    )

    assert in_front.tolist() == [True]
    assert pixels[0].tolist() == pytest.approx([480.0, 360.0])


def test_central_pixel_ray_uses_minus_camera_z() -> None:
    """Главный луч OpenGL обязан смотреть по -Z, а не по +Z или оси Y."""

    intrinsics = blender_horizontal_sensor_intrinsics(
        width_px=960,
        height_px=720,
        focal_length_mm=24.0,
        sensor_width_mm=36.0,
    )
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, 3] = [2.0, 3.0, 45.0]

    origins, directions = pixel_rays_world_opengl(
        np.asarray([[480.0, 360.0]]),
        camera_to_world,
        intrinsics,
    )

    assert origins[0].tolist() == pytest.approx([2.0, 3.0, 45.0])
    assert directions[0].tolist() == pytest.approx([0.0, 0.0, -1.0])


def test_right_image_pixel_has_positive_local_x_direction() -> None:
    """Проверка ловит зеркальный экспорт горизонтальной оси камеры."""

    intrinsics = blender_horizontal_sensor_intrinsics(
        width_px=960,
        height_px=720,
        focal_length_mm=24.0,
        sensor_width_mm=36.0,
    )
    _, directions = pixel_rays_world_opengl(
        np.asarray([[580.0, 360.0]]),
        np.eye(4, dtype=np.float64),
        intrinsics,
    )

    assert directions[0, 0] > 0.0
    assert directions[0, 2] < 0.0
    assert np.linalg.norm(directions[0]) == pytest.approx(1.0)


def test_non_rigid_camera_matrix_is_rejected() -> None:
    """Масштаб в pose нельзя незаметно принять за корректный поворот камеры."""

    invalid = np.eye(4, dtype=np.float64)
    invalid[0, 0] = 2.0

    with pytest.raises(ValueError, match="не ортонормален"):
        validate_camera_to_world(invalid)
