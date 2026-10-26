"""Тесты единого препроцессинга MARS-LVIG для DINOv2 и SIFT."""

from __future__ import annotations

import numpy as np
import pytest

from aerial_mapper.camera_distortion import BrownConradyDistortion
from aerial_mapper.mars_lvig_preprocessing import prepare_mars_lvig_image
from aerial_mapper.radiance_camera import PinholeIntrinsics


def test_prepare_undistorts_before_aspect_preserving_resize() -> None:
    """Нулевая дисторсия оставляет геометрию и даёт ожидаемые 672 x 562."""

    height = 2048
    width = 2448
    horizontal_ramp = np.linspace(0, 255, width, dtype=np.uint8)
    image = np.repeat(horizontal_ramp[np.newaxis, :, np.newaxis], height, axis=0)
    image = np.repeat(image, 3, axis=2)
    intrinsics = PinholeIntrinsics(
        width_px=width,
        height_px=height,
        focal_x_px=1444.0,
        focal_y_px=1444.0,
        principal_x_px=1178.0,
        principal_y_px=1044.0,
    )

    prepared = prepare_mars_lvig_image(
        image,
        intrinsics=intrinsics,
        distortion=BrownConradyDistortion(),
    )

    assert prepared.shape == (562, 672, 3)
    assert prepared.dtype == np.uint8
    assert prepared[281, 0, 0] < prepared[281, -1, 0]


def test_prepare_rejects_nonpositive_target_size() -> None:
    """Нулевой размер не должен доходить до неоднозначной ошибки OpenCV."""

    image = np.zeros((2, 3, 3), dtype=np.uint8)
    intrinsics = PinholeIntrinsics(3, 2, 1.0, 1.0, 1.5, 1.0)

    with pytest.raises(ValueError, match="положительной"):
        prepare_mars_lvig_image(
            image,
            intrinsics=intrinsics,
            distortion=BrownConradyDistortion(),
            maximum_side_pixels=0,
        )
