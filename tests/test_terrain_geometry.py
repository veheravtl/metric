"""Проверки форм рельефа и независимого оценщика G9."""

import numpy as np
import pytest

from aerial_mapper.terrain_evaluation import (
    estimate_homography,
    point_error_summary,
    segment_error_summary,
    select_displacement_pairs,
    select_grid_anchor_indices,
    select_segment_pairs,
    spatial_checkerboard_split_indices,
    spatial_group_labels,
    transform_points,
)
from aerial_mapper.terrain_geometry import (
    TerrainSpecification,
    terrain_height_m,
    terrain_vertices,
    triangle_centroid_samples,
)


def test_inclined_plane_uses_angle_in_degrees() -> None:
    """Уклон 45 градусов должен давать один метр высоты на метр X."""

    specification = TerrainSpecification(kind="plane", slope_x_degrees=45.0)
    height = terrain_height_m(np.array([0.0, 1.0]), 0.0, specification)
    assert height == pytest.approx([0.0, 1.0])


def test_gaussian_sign_distinguishes_hill_and_pit() -> None:
    """Холм и яма одной формы должны отличаться только знаком высоты."""

    hill = TerrainSpecification(kind="gaussian", amplitude_m=3.0, sigma_m=8.0)
    pit = TerrainSpecification(kind="gaussian", amplitude_m=-3.0, sigma_m=8.0)
    coordinates = np.array([-4.0, 0.0, 5.0])
    hill_height = terrain_height_m(coordinates, 0.0, hill)
    pit_height = terrain_height_m(coordinates, 0.0, pit)
    assert pit_height == pytest.approx(-hill_height)
    assert hill_height[1] == pytest.approx(3.0)


def test_triangle_samples_lie_on_actual_planar_mesh() -> None:
    """Центроиды граней не должны отходить от наклонной mesh-поверхности."""

    specification = TerrainSpecification(
        kind="plane", slope_x_degrees=7.0, slope_y_degrees=-4.0
    )
    vertices, _, faces = terrain_vertices(
        width_m=20.0,
        height_m=16.0,
        columns=11,
        rows=9,
        specification=specification,
    )
    samples = triangle_centroid_samples(vertices, faces, stride=3)
    expected_z = terrain_height_m(samples[:, 0], samples[:, 1], specification)
    assert samples[:, 2] == pytest.approx(expected_z, abs=1e-12)


def test_exact_planar_chain_generalizes_to_held_out_points() -> None:
    """Раздельные точки подгонки и оценки должны восстановить плоское H точно."""

    grid_x, grid_y = np.meshgrid(
        np.linspace(20.0, 900.0, 8), np.linspace(100.0, 600.0, 6)
    )
    source = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    true_homography = np.array(
        [[0.04, 0.002, -18.0], [-0.001, 0.05, -14.0], [1e-5, -2e-5, 1.0]]
    )
    destination = transform_points(source, true_homography)
    fit, evaluation = spatial_checkerboard_split_indices(source)
    estimated_homography = estimate_homography(source[fit], destination[fit])
    estimated = transform_points(source[evaluation], estimated_homography)
    summary = point_error_summary(estimated, destination[evaluation])
    assert summary.maximum_m < 1e-6


def test_checkerboard_split_rejects_too_few_points() -> None:
    """Оценщик не должен переиспользовать минимальный набор как проверочный."""

    with pytest.raises(ValueError, match="N >= 8"):
        spatial_checkerboard_split_indices(np.zeros((7, 2)))


def test_three_spatial_groups_cover_regular_grid_evenly() -> None:
    """Калибровка, подгонка и оценка не должны делить сетку полосами."""

    grid_x, grid_y = np.meshgrid(np.arange(6.0), np.arange(6.0))
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    labels = spatial_group_labels(points, group_count=3)
    counts = np.bincount(labels, minlength=3)
    assert counts.tolist() == [12, 12, 12]


def test_exact_mapping_preserves_selected_segment_lengths() -> None:
    """Точный перенос должен давать нулевую ошибку выбранных отрезков."""

    truth = np.column_stack(
        (
            np.arange(13.0),
            np.zeros(13),
        )
    )
    pairs = select_segment_pairs(
        truth,
        target_length_m=6.0,
        tolerance_m=0.1,
        maximum_count=4,
    )
    summary = segment_error_summary(truth.copy(), truth, pairs)
    assert summary.count > 0
    assert summary.maximum_absolute_m == pytest.approx(0.0)
    assert summary.maximum_relative_percent == pytest.approx(0.0)


def test_grid_anchor_selection_spans_both_map_axes() -> None:
    """Синтетические реперы должны окружать область, а не собираться полосой."""

    grid_x, grid_y = np.meshgrid(np.arange(5.0), np.arange(5.0))
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    indices = select_grid_anchor_indices(
        points,
        columns=3,
        rows=2,
        inset_fraction=0.0,
    )
    selected = points[indices]
    assert len(np.unique(indices)) == 6
    assert selected[:, 0].min() == pytest.approx(0.0)
    assert selected[:, 0].max() == pytest.approx(4.0)
    assert selected[:, 1].min() == pytest.approx(0.0)
    assert selected[:, 1].max() == pytest.approx(4.0)


def test_displacement_pair_selection_preserves_direction_and_sign() -> None:
    """Пара цель/попадание должна сохранять порядок и заданный знак вектора."""

    grid_x, grid_y = np.meshgrid(np.arange(7.0), np.arange(7.0))
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    pairs = select_displacement_pairs(
        points,
        np.array([-2.0, 3.0]),
        maximum_count=5,
        tolerance_m=1e-9,
    )
    deltas = points[pairs[:, 1]] - points[pairs[:, 0]]
    assert pairs.shape == (5, 2)
    assert deltas == pytest.approx(np.tile([-2.0, 3.0], (5, 1)))
