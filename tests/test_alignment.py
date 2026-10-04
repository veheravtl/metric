"""Тесты независимой оценки гомографии и её количественной проверки."""

import cv2
import numpy as np
import pytest

from aerial_mapper.alignment import (
    AlignmentFailure,
    SiftRansacConfig,
    align_frame_to_reference,
)
from aerial_mapper.evaluation import evaluate_homography
from aerial_mapper.quality import analyze_alignment_quality
from aerial_mapper.synthetic import SyntheticFrameSpec, generate_synthetic_frame


def make_unique_textured_reference(size: int = 900) -> np.ndarray:
    """Создаёт детерминированную карту с уникальной локальной текстурой.

    Псевдослучайная составляющая здесь не имитирует реальную местность. Она
    нужна, чтобы unit-тест проверял геометрию и не зависел от внешнего GeoTIFF.
    Seed фиксирован, поэтому входные данные полностью воспроизводимы.
    """

    random_generator = np.random.default_rng(20261002)
    noise = random_generator.integers(
        0,
        256,
        size=(size, size, 3),
        dtype=np.uint8,
    )
    reference = cv2.GaussianBlur(noise, (0, 0), sigmaX=1.2)

    # Крупные фигуры добавляют признаки на нескольких масштабах и делают
    # возможную грубую ошибку положения понятной при диагностике упавшего теста.
    cv2.rectangle(reference, (120, 100), (370, 310), (245, 220, 30), 12)
    cv2.circle(reference, (650, 230), 95, (20, 220, 245), 14)
    cv2.line(reference, (80, 760), (820, 500), (250, 250, 250), 18)
    cv2.putText(
        reference,
        "POC",
        (310, 650),
        cv2.FONT_HERSHEY_SIMPLEX,
        3.0,
        (30, 30, 30),
        thickness=12,
        lineType=cv2.LINE_AA,
    )
    return reference


def test_sift_ransac_recovers_hidden_homography() -> None:
    """Оценщик должен восстановить H, не получая углы и матрицу генератора."""

    reference = make_unique_textured_reference()
    synthetic = generate_synthetic_frame(
        reference,
        reference_resolution_m_per_pixel=0.1,
        spec=SyntheticFrameSpec(
            footprint_width_m=58.0,
            footprint_height_m=43.0,
            output_width_pixels=640,
            output_height_pixels=480,
            rotation_degrees=17.0,
            perspective_strength=0.45,
        ),
    )

    # В интерфейс функции намеренно передаются только изображения. Истинная
    # матрица появляется впервые ниже, уже в независимом оценщике ошибки.
    alignment = align_frame_to_reference(
        reference,
        synthetic.image_rgb,
        config=SiftRansacConfig(max_features_per_image=3_000),
    )
    evaluation = evaluate_homography(
        alignment.homography_frame_to_reference,
        synthetic.homography_frame_to_reference,
        frame_width_pixels=synthetic.spec.output_width_pixels,
        frame_height_pixels=synthetic.spec.output_height_pixels,
        reference_resolution_m_per_pixel=0.1,
    )

    assert alignment.ratio_match_count >= 20
    assert alignment.inlier_count >= 15
    assert alignment.inlier_ratio > 0.7
    assert alignment.inlier_spatial_coverage_fraction > 0.25
    assert evaluation.mean_error_pixels < 0.5
    assert evaluation.max_error_pixels < 1.0

    quality = analyze_alignment_quality(
        alignment,
        frame_width_pixels=synthetic.spec.output_width_pixels,
        frame_height_pixels=synthetic.spec.output_height_pixels,
        random_seed=17,
    )
    repeated_quality = analyze_alignment_quality(
        alignment,
        frame_width_pixels=synthetic.spec.output_width_pixels,
        frame_height_pixels=synthetic.spec.output_height_pixels,
        random_seed=17,
    )

    # Здесь нет порога «хорошей привязки»: тест проверяет диапазоны, единицы и
    # воспроизводимость самого измерительного инструмента. Допустимые границы
    # будут определяться только на независимой валидационной серии.
    assert 0 < quality.occupied_grid_cells <= quality.grid_rows * quality.grid_columns
    assert 0.0 < quality.grid_occupancy_fraction <= 1.0
    assert 0.0 <= quality.horizontal_span_fraction <= 1.0
    assert 0.0 <= quality.vertical_span_fraction <= 1.0
    assert quality.inlier_reprojection_median_reference_px >= 0.0
    assert (
        quality.inlier_reprojection_p95_reference_px
        >= quality.inlier_reprojection_median_reference_px
    )
    assert (
        quality.inlier_reprojection_max_reference_px
        >= quality.inlier_reprojection_p95_reference_px
    )
    assert quality.stability_trials_succeeded > 0
    assert quality.stability_p95_max_corner_shift_reference_px is not None
    assert quality == repeated_quality


def test_evaluation_is_independent_of_homography_matrix_scale() -> None:
    """Умножение H на число не должно создавать геометрическую ошибку."""

    homography = np.array(
        [
            [0.8, -0.15, 320.0],
            [0.12, 0.75, 180.0],
            [0.0001, -0.0002, 1.0],
        ],
        dtype=np.float64,
    )
    evaluation = evaluate_homography(
        homography * 7.0,
        homography,
        frame_width_pixels=640,
        frame_height_pixels=480,
        reference_resolution_m_per_pixel=0.1,
    )

    assert evaluation.max_error_pixels == pytest.approx(0.0, abs=1e-6)
    assert evaluation.max_error_meters == pytest.approx(0.0, abs=1e-7)


def test_alignment_refuses_blank_images() -> None:
    """Отсутствие признаков должно давать явный отказ, а не случайную матрицу."""

    blank_reference = np.zeros((500, 500, 3), dtype=np.uint8)
    blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)

    with pytest.raises(AlignmentFailure, match="недостаточно SIFT-признаков"):
        align_frame_to_reference(blank_reference, blank_frame)


def test_reference_feature_mask_limits_teach_keypoints() -> None:
    """Разметка Teach должна ограничивать признаки без маски Repeat-кадра."""

    reference = make_unique_textured_reference(size=700)
    frame = reference.copy()
    mask = np.zeros(reference.shape[:2], dtype=np.uint8)
    mask[:, :350] = 255

    alignment = align_frame_to_reference(
        reference,
        frame,
        config=SiftRansacConfig(max_features_per_image=2_000),
        reference_feature_mask=mask,
    )

    rounded = np.rint(alignment.reference_points_px).astype(np.int64)
    assert alignment.inlier_count > 20
    assert np.all(mask[rounded[:, 1], rounded[:, 0]] != 0)
    assert alignment.reference_keypoint_count < alignment.frame_keypoint_count


@pytest.mark.parametrize(
    "mask",
    [
        np.ones((100, 100), dtype=np.uint8),
        np.ones((500, 500), dtype=np.bool_),
        np.zeros((500, 500), dtype=np.uint8),
    ],
)
def test_reference_feature_mask_rejects_invalid_input(mask: np.ndarray) -> None:
    """Неверная или пустая Teach-маска не должна молча игнорироваться."""

    image = np.zeros((500, 500, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Маска признаков эталона"):
        align_frame_to_reference(image, image, reference_feature_mask=mask)
