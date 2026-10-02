"""Тесты оценки реальной сцены по плотной 3D-истине."""

import cv2
import numpy as np
import pytest

from aerial_mapper.dense_ground_truth import (
    build_regular_query_grid,
    evaluate_homography_against_dense_truth,
    fit_best_single_homography,
    point_map_to_reference_pixels,
    sample_dense_map_bilinearly,
    select_valid_dense_correspondences,
)


def test_point_map_respects_signed_vertical_scale() -> None:
    """Отрицательный масштаб Y должен правильно учитывать строки вниз."""

    point_map = np.array(
        [
            [[10.0, 20.0, 2.0], [10.5, 20.0, 3.0]],
            [[10.0, 19.5, 1.0], [10.5, 19.5, 4.0]],
        ],
        dtype=np.float32,
    )
    reference_map = point_map_to_reference_pixels(
        point_map,
        reference_offset_xy=np.array([10.0, 20.0]),
        reference_scale_xy_m_per_pixel=np.array([0.5, -0.5]),
    )

    assert reference_map == pytest.approx(
        np.array(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 1.0]],
            ]
        )
    )


def test_bilinear_sampling_preserves_subpixel_truth() -> None:
    """Субпиксельная выборка не должна добавлять ошибку округления SIFT-точек."""

    x_coordinates, y_coordinates = np.meshgrid(
        np.arange(4, dtype=np.float32),
        np.arange(4, dtype=np.float32),
    )
    dense_map = np.stack((2.0 * x_coordinates, 3.0 * y_coordinates), axis=2)
    sampled, valid = sample_dense_map_bilinearly(
        dense_map,
        np.array([[1.25, 2.5], [3.0, 1.0]], dtype=np.float32),
    )

    assert valid.tolist() == [True, False]
    assert sampled[0] == pytest.approx([2.5, 7.5])
    assert np.isnan(sampled[1]).all()


def test_dense_evaluation_recovers_known_projective_mapping() -> None:
    """Истинная плоская сцена должна давать нулевую ошибку fitted-гомографии."""

    height, width = 24, 32
    query_grid = build_regular_query_grid(
        frame_width_pixels=width,
        frame_height_pixels=height,
        columns=9,
        rows=7,
    )
    true_homography = np.array(
        [[1.2, 0.05, 7.0], [-0.03, 0.9, 11.0], [0.0004, -0.0002, 1.0]],
        dtype=np.float64,
    )
    all_x, all_y = np.meshgrid(np.arange(width), np.arange(height))
    all_query_points = np.column_stack((all_x.ravel(), all_y.ravel())).astype(
        np.float32
    )
    dense_map = cv2.perspectiveTransform(
        all_query_points.reshape(1, -1, 2),
        true_homography,
    ).reshape(height, width, 2)
    valid_query, valid_truth = select_valid_dense_correspondences(
        query_grid,
        dense_map,
        reference_width_pixels=100,
        reference_height_pixels=100,
    )
    fitted = fit_best_single_homography(valid_query, valid_truth)
    evaluation = evaluate_homography_against_dense_truth(
        fitted,
        valid_query,
        valid_truth,
        reference_scale_xy_m_per_pixel=np.array([0.2, -0.2]),
    )

    # Билинейная выборка дискретной карты перспективного преобразования не
    # воспроизводит дробную проекцию с машинным нулём. Допуск 20 микрометров
    # проверяет практически нулевую ошибку и не маскирует сантиметровый сдвиг.
    assert evaluation.max_error_meters == pytest.approx(0.0, abs=2e-5)
