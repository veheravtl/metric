"""Тесты контролируемого синтетического кадра и обратного наложения."""

import cv2
import numpy as np
import pytest

from aerial_mapper.synthetic import SyntheticFrameSpec, generate_synthetic_frame
from aerial_mapper.visualization import (
    build_reverse_overlay,
    draw_reference_footprint,
)


def make_test_reference(size: int = 1000) -> np.ndarray:
    """Создаёт цветной градиент с линиями для проверки преобразований."""

    y_coordinates, x_coordinates = np.indices((size, size))
    reference = np.stack(
        [
            x_coordinates % 256,
            y_coordinates % 256,
            (x_coordinates + y_coordinates) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)

    # Линии разных направлений делают ошибку гомографии визуально и численно
    # заметнее, чем один гладкий градиент.
    cv2.line(reference, (100, 100), (900, 800), (255, 255, 255), 8)
    cv2.rectangle(reference, (300, 250), (700, 650), (20, 240, 40), 10)
    return reference


def test_known_homography_maps_footprint_to_frame_corners() -> None:
    """Истинная гомография должна точно переводить четыре заданных угла."""

    reference = make_test_reference()
    spec = SyntheticFrameSpec(
        footprint_width_m=60.0,
        footprint_height_m=45.0,
        output_width_pixels=640,
        output_height_pixels=480,
    )
    result = generate_synthetic_frame(
        reference,
        reference_resolution_m_per_pixel=0.1,
        spec=spec,
    )

    projected_corners = cv2.perspectiveTransform(
        result.source_corners_reference_px.reshape(1, -1, 2),
        result.homography_reference_to_frame,
    ).reshape(-1, 2)

    assert result.image_rgb.shape == (480, 640, 3)
    assert result.image_rgb.dtype == np.uint8
    assert float(result.image_rgb.std()) > 1.0
    assert projected_corners == pytest.approx(
        result.destination_corners_frame_px,
        abs=1e-3,
    )


def test_forward_and_inverse_homographies_cancel_each_other() -> None:
    """Произведение прямой и обратной матриц должно давать единичную матрицу."""

    result = generate_synthetic_frame(
        make_test_reference(),
        reference_resolution_m_per_pixel=0.1,
        spec=SyntheticFrameSpec(
            footprint_width_m=60.0,
            footprint_height_m=45.0,
            output_width_pixels=640,
            output_height_pixels=480,
        ),
    )

    round_trip = (
        result.homography_frame_to_reference @ result.homography_reference_to_frame
    )
    round_trip /= round_trip[2, 2]

    assert round_trip == pytest.approx(np.eye(3), abs=1e-9)


def test_visualizations_keep_reference_dimensions() -> None:
    """Разметка и обратное наложение не должны менять размер эталона."""

    reference = make_test_reference()
    result = generate_synthetic_frame(
        reference,
        reference_resolution_m_per_pixel=0.1,
        spec=SyntheticFrameSpec(
            footprint_width_m=60.0,
            footprint_height_m=45.0,
            output_width_pixels=640,
            output_height_pixels=480,
        ),
    )

    annotated = draw_reference_footprint(
        reference,
        result.source_corners_reference_px,
    )
    overlay = build_reverse_overlay(
        reference,
        result.image_rgb,
        result.homography_frame_to_reference,
        result.source_corners_reference_px,
    )

    assert annotated.shape == reference.shape
    assert overlay.comparison_rgb.shape == reference.shape
    assert overlay.covered_pixels > 0
    assert 0.0 <= overlay.mean_absolute_error < 30.0


def test_footprint_outside_reference_is_rejected() -> None:
    """Слишком большой участок нельзя незаметно дополнить чёрными полями."""

    with pytest.raises(ValueError, match="выходит за границы"):
        generate_synthetic_frame(
            make_test_reference(),
            reference_resolution_m_per_pixel=0.1,
            spec=SyntheticFrameSpec(
                footprint_width_m=200.0,
                footprint_height_m=200.0,
            ),
        )
