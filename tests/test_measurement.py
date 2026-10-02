"""Тесты цепочки кадр → GeoTIFF → метрические расстояния и площади."""

import numpy as np
import pytest
from rasterio import Affine

from aerial_mapper.measurement import (
    MeasurementFailure,
    map_frame_points_to_reference,
    map_frame_points_to_world,
    map_reference_points_to_world,
    measure_polygon,
    measure_segment,
)
from aerial_mapper.measurement_evaluation import (
    build_poc_control_definitions,
    evaluate_metric_controls,
)

REFERENCE_TRANSFORM = Affine(0.1, 0.0, 100.0, 0.0, -0.1, 200.0)
REFERENCE_CRS = "EPSG:32636"
IDENTITY_HOMOGRAPHY = np.eye(3, dtype=np.float64)
REFERENCE_SIZE = 1000


def measurement_arguments() -> dict[str, object]:
    """Возвращает общие параметры маленького метрического тестового растра."""

    return {
        "reference_transform": REFERENCE_TRANSFORM,
        "reference_crs": REFERENCE_CRS,
        "reference_width_pixels": REFERENCE_SIZE,
        "reference_height_pixels": REFERENCE_SIZE,
    }


def test_reference_pixel_centers_map_to_expected_easting_and_northing() -> None:
    """Проверяет полупиксель и противоположные направления Y растра и карты."""

    world_points = map_reference_points_to_world(
        np.asarray([[0.0, 0.0], [10.0, 10.0]]),
        reference_transform=REFERENCE_TRANSFORM,
        reference_crs=REFERENCE_CRS,
    )

    # Центр пикселя (0, 0) расположен на полпикселя восточнее и южнее угла
    # GeoTIFF: +0,05 м по Easting и -0,05 м по Northing.
    assert world_points == pytest.approx(
        np.asarray([[100.05, 199.95], [101.05, 198.95]]),
        abs=1e-12,
    )


def test_segment_uses_metric_crs_and_recovers_three_four_five_triangle() -> None:
    """Сдвиг 30 × 40 пикселей при 0,1 м/px должен иметь длину ровно 5 м."""

    measurement = measure_segment(
        np.asarray([[100.0, 100.0], [130.0, 140.0]]),
        IDENTITY_HOMOGRAPHY,
        **measurement_arguments(),
    )

    assert measurement.length_meters == pytest.approx(5.0, abs=1e-12)


def test_polygon_area_is_measured_in_square_meters() -> None:
    """Квадрат 100 × 100 пикселей должен стать квадратом 10 × 10 метров."""

    measurement = measure_polygon(
        np.asarray(
            [
                [100.0, 100.0],
                [200.0, 100.0],
                [200.0, 200.0],
                [100.0, 200.0],
            ]
        ),
        IDENTITY_HOMOGRAPHY,
        **measurement_arguments(),
    )

    assert measurement.area_square_meters == pytest.approx(100.0, abs=1e-10)


def test_homography_common_scale_does_not_change_world_coordinates() -> None:
    """Одна проекция, записанная с другим общим масштабом, эквивалентна."""

    frame_points = np.asarray([[20.0, 30.0], [80.0, 90.0]])
    base = map_frame_points_to_world(
        frame_points,
        IDENTITY_HOMOGRAPHY,
        **measurement_arguments(),
    )
    scaled = map_frame_points_to_world(
        frame_points,
        IDENTITY_HOMOGRAPHY * 1e-18,
        **measurement_arguments(),
    )

    assert scaled.reference_points_px == pytest.approx(base.reference_points_px)
    assert scaled.world_points_m == pytest.approx(base.world_points_m)


def test_self_intersecting_polygon_is_rejected_instead_of_silently_repaired() -> None:
    """Контур-«бабочка» не должен превращаться в правдоподобную площадь."""

    bow_tie = np.asarray(
        [[100.0, 100.0], [200.0, 200.0], [100.0, 200.0], [200.0, 100.0]]
    )

    with pytest.raises(MeasurementFailure, match="Полигон невалиден"):
        measure_polygon(
            bow_tie,
            IDENTITY_HOMOGRAPHY,
            **measurement_arguments(),
        )


def test_geographic_crs_in_degrees_is_rejected() -> None:
    """Евклидово расстояние в градусах нельзя выдавать пользователю как метры."""

    with pytest.raises(MeasurementFailure, match="проецированная CRS"):
        map_reference_points_to_world(
            np.asarray([[10.0, 20.0]]),
            reference_transform=REFERENCE_TRANSFORM,
            reference_crs="EPSG:4326",
        )


def test_point_outside_reference_is_rejected() -> None:
    """Измерение за границей карты не должно продолжаться по affine-формуле."""

    with pytest.raises(MeasurementFailure, match="за границы эталона"):
        map_frame_points_to_world(
            np.asarray([[1001.0, 500.0]]),
            IDENTITY_HOMOGRAPHY,
            **measurement_arguments(),
        )


def test_point_at_infinity_is_rejected() -> None:
    """Нулевой знаменатель перспективного деления должен давать явный отказ."""

    degenerate_homography = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    with pytest.raises(MeasurementFailure, match="в бесконечность"):
        map_frame_points_to_reference(
            np.asarray([[10.0, 20.0]]),
            degenerate_homography,
        )


def test_control_evaluation_detects_deliberately_perturbed_scale() -> None:
    """Оценщик должен отличать точную матрицу от намеренно искажённой."""

    controls = build_poc_control_definitions(
        frame_width_pixels=640,
        frame_height_pixels=480,
    )
    # Эта матрица увеличивает координаты относительно начала на 1%. Это не
    # общий множитель H, а реальное изменение отображаемого масштаба.
    perturbed_homography = np.asarray(
        [[1.01, 0.0, 0.0], [0.0, 1.01, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    evaluations = evaluate_metric_controls(
        controls,
        estimated_homography_frame_to_reference=perturbed_homography,
        true_homography_frame_to_reference=IDENTITY_HOMOGRAPHY,
        reference_transform=REFERENCE_TRANSFORM,
        reference_crs=REFERENCE_CRS,
        reference_width_pixels=REFERENCE_SIZE,
        reference_height_pixels=REFERENCE_SIZE,
    )

    segment_errors = [
        result.relative_error_percent
        for result in evaluations
        if result.kind == "segment"
    ]
    polygon_errors = [
        result.relative_error_percent
        for result in evaluations
        if result.kind == "polygon"
    ]
    assert segment_errors == pytest.approx([1.0] * len(segment_errors), abs=1e-10)
    assert polygon_errors == pytest.approx([2.01] * len(polygon_errors), abs=1e-10)
