"""Тесты разделения плоской земли и неплоских поверхностей."""

import numpy as np
import pytest
from rasterio import Affine

from aerial_mapper.measurement import map_reference_points_to_world
from aerial_mapper.surface_evaluation import (
    SurfaceSegmentControl,
    estimate_oracle_homography,
    evaluate_surface_segments,
)

TRANSFORM = Affine(0.1, 0.0, 1000.0, 0.0, -0.1, 2000.0)
CRS = "EPSG:32636"


def world_from_reference(points: np.ndarray) -> np.ndarray:
    """Переводит тестовые пиксели в ту же метрическую истину, что рабочий код."""

    return map_reference_points_to_world(
        points,
        reference_transform=TRANSFORM,
        reference_crs=CRS,
    )


def test_ground_oracle_is_exact_but_exposes_nonplanar_control() -> None:
    """Одна H должна быть точной для земли и ошибаться на смещённой крыше."""

    ground_a = np.asarray([[10.0, 10.0], [40.0, 10.0]])
    ground_b = np.asarray([[15.0, 35.0], [45.0, 55.0]])
    roof_frame = np.asarray([[60.0, 30.0], [90.0, 30.0]])
    roof_reference = np.asarray([[66.0, 35.0], [99.0, 35.0]])
    controls = (
        SurfaceSegmentControl(
            "ground_a",
            "ground",
            ground_a,
            ground_a,
            world_from_reference(ground_a),
        ),
        SurfaceSegmentControl(
            "ground_b",
            "ground",
            ground_b,
            ground_b,
            world_from_reference(ground_b),
        ),
        SurfaceSegmentControl(
            "roof",
            "roof",
            roof_frame,
            roof_reference,
            world_from_reference(roof_reference),
        ),
    )

    ground_oracle = estimate_oracle_homography(controls, surface="ground")
    evaluations = evaluate_surface_segments(
        controls,
        homography_frame_to_reference=ground_oracle,
        reference_transform=TRANSFORM,
        reference_crs=CRS,
        reference_width_pixels=200,
        reference_height_pixels=200,
    )

    assert ground_oracle == pytest.approx(np.eye(3), abs=1e-10)
    assert evaluations[0].maximum_endpoint_position_error_m == pytest.approx(0.0)
    assert evaluations[1].maximum_endpoint_position_error_m == pytest.approx(0.0)
    assert evaluations[2].maximum_endpoint_position_error_m > 0.5
    assert evaluations[2].relative_length_error_percent == pytest.approx(
        100.0 / 11.0,
        abs=1e-9,
    )
