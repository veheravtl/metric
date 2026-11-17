"""Тесты локальной проекции направления на репер в границы кадра."""

import numpy as np
import pytest

from aerial_mapper.reference_direction import (
    ReferenceDirectionProjectionFailure,
    project_map_direction_to_frame_edge,
)


def test_identity_projection_clips_rightward_ray_at_last_frame_pixel() -> None:
    """Единичная гомография сохраняет направление и обрезает его по X=99."""

    segment = project_map_direction_to_frame_edge(
        (20.0, 30.0),
        (20.0, 30.0),
        (1.0, 0.0),
        np.eye(3, dtype=np.float64),
        frame_width_pixels=100,
        frame_height_pixels=80,
    )

    assert segment.origin_px == pytest.approx((20.0, 30.0))
    assert segment.boundary_px == pytest.approx((99.0, 30.0))


def test_inverse_homography_rotates_local_map_direction_into_frame() -> None:
    """Направление переносится локально, а не проекцией далёкой точки."""

    # Кадр → карта: поворот на 90° против часовой стрелки с переносом.
    # Поэтому восток карты соответствует направлению вверх в кадре.
    homography = np.array(
        [
            [0.0, -1.0, 79.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    segment = project_map_direction_to_frame_edge(
        (20.0, 30.0),
        (49.0, 20.0),
        (1.0, 0.0),
        homography,
        frame_width_pixels=100,
        frame_height_pixels=80,
    )

    assert segment.boundary_px == pytest.approx((20.0, 0.0), abs=1e-12)


def test_projective_scale_still_ends_on_visible_frame_boundary() -> None:
    """Перспективная матрица не должна увести конечную точку за изображение."""

    homography = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.002, 0.001, 1.0],
        ],
        dtype=np.float64,
    )
    segment = project_map_direction_to_frame_edge(
        (40.0, 30.0),
        (40.0 / 1.11, 30.0 / 1.11),
        (1.0, 1.0),
        homography,
        frame_width_pixels=100,
        frame_height_pixels=80,
    )

    x, y = segment.boundary_px
    assert 0.0 <= x <= 99.0
    assert 0.0 <= y <= 79.0
    assert x == pytest.approx(99.0) or y == pytest.approx(79.0)


def test_singular_homography_is_rejected_explicitly() -> None:
    """Вырожденная матрица не должна давать правдоподобную линию."""

    with pytest.raises(ReferenceDirectionProjectionFailure, match="вырождена"):
        project_map_direction_to_frame_edge(
            (20.0, 30.0),
            (20.0, 30.0),
            (1.0, 0.0),
            np.zeros((3, 3), dtype=np.float64),
            frame_width_pixels=100,
            frame_height_pixels=80,
        )
