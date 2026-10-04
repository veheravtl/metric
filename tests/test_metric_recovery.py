"""Тесты рабочего API восстановления метрик без сведений о позе камеры."""

import cv2
import numpy as np
import pytest
from rasterio import Affine

from aerial_mapper.measurement import measure_segment
from aerial_mapper.metric_recovery import (
    MetricRecoveryFailure,
    MetricRecoveryThresholds,
    ObservedSegment,
    recover_metric_segments,
)
from aerial_mapper.synthetic import SyntheticFrameSpec, generate_synthetic_frame


def make_metric_reference(size: int = 700) -> np.ndarray:
    """Создаёт уникальную локальную текстуру без внешнего файла данных."""

    random_generator = np.random.default_rng(20261004)
    noise = random_generator.integers(
        0,
        256,
        size=(size, size, 3),
        dtype=np.uint8,
    )
    reference = cv2.GaussianBlur(noise, (0, 0), sigmaX=1.1)
    cv2.rectangle(reference, (70, 90), (310, 270), (245, 210, 35), 10)
    cv2.circle(reference, (520, 180), 70, (30, 220, 240), 12)
    cv2.line(reference, (60, 610), (640, 410), (250, 250, 250), 14)
    return reference


def test_recovery_uses_only_images_and_map_to_measure_segment() -> None:
    """Найденная H должна дать ту же длину, что скрытая H генератора."""

    resolution_m_per_pixel = 0.1
    reference = make_metric_reference()
    synthetic = generate_synthetic_frame(
        reference,
        reference_resolution_m_per_pixel=resolution_m_per_pixel,
        spec=SyntheticFrameSpec(
            footprint_width_m=42.0,
            footprint_height_m=31.0,
            output_width_pixels=480,
            output_height_pixels=360,
            rotation_degrees=13.0,
            perspective_strength=0.35,
        ),
    )
    frame_points = np.asarray([[95.0, 105.0], [370.0, 245.0]])
    transform = Affine(0.1, 0.0, 1000.0, 0.0, -0.1, 2000.0)
    common = {
        "reference_transform": transform,
        "reference_crs": "EPSG:32636",
        "reference_width_pixels": reference.shape[1],
        "reference_height_pixels": reference.shape[0],
    }

    hidden_truth = measure_segment(
        frame_points,
        synthetic.homography_frame_to_reference,
        **common,
    )
    recovered = recover_metric_segments(
        reference,
        synthetic.image_rgb,
        (ObservedSegment("test_segment", frame_points, surface="ground"),),
        reference_transform=transform,
        reference_crs="EPSG:32636",
        thresholds=MetricRecoveryThresholds(
            minimum_inlier_count=15,
            minimum_inlier_ratio=0.6,
            minimum_coverage_fraction=0.2,
            maximum_reprojection_p95_px=2.0,
            maximum_stability_p95_corner_shift_px=5.0,
        ),
        random_seed=17,
    )

    assert recovered.alignment.inlier_count >= 15
    assert recovered.segments[0].measurement.length_meters == pytest.approx(
        hidden_truth.length_meters,
        abs=0.05,
    )


def test_recovery_rejects_unconfirmed_surface_before_alignment() -> None:
    """Без модели поверхности нельзя выдавать крышу как наземное измерение."""

    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    segment = ObservedSegment(
        "unknown_surface",
        np.asarray([[10.0, 10.0], [20.0, 20.0]]),
    )

    with pytest.raises(
        MetricRecoveryFailure, match="только подтверждённые точки земли"
    ):
        recover_metric_segments(
            blank,
            blank,
            (segment,),
            reference_transform=Affine.identity(),
            reference_crs="EPSG:32636",
            thresholds=MetricRecoveryThresholds(
                minimum_inlier_count=4,
                minimum_inlier_ratio=0.0,
                minimum_coverage_fraction=0.0,
                maximum_reprojection_p95_px=3.0,
                maximum_stability_p95_corner_shift_px=5.0,
            ),
        )
