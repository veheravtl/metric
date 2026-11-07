"""Проверки экспериментального автоматического поиска кадра на карте."""

import cv2
import numpy as np
import pytest

from aerial_mapper.automatic_map_alignment import (
    AutomaticMapAlignmentConfig,
    AutomaticMapAlignmentFailure,
    _filter_matches_by_scale_and_orientation,
    _MutualMatch,
    align_frame_to_map_automatically,
)
from aerial_mapper.evaluation import evaluate_homography
from aerial_mapper.synthetic import SyntheticFrameSpec, generate_synthetic_frame


def _textured_reference(size: int = 900) -> np.ndarray:
    """Создать детерминированную карту с уникальными деталями разных размеров."""

    random_generator = np.random.default_rng(20261105)
    noise = random_generator.integers(0, 256, (size, size, 3), dtype=np.uint8)
    reference = cv2.GaussianBlur(noise, (0, 0), sigmaX=1.1)
    cv2.rectangle(reference, (90, 110), (390, 320), (240, 220, 30), 10)
    cv2.circle(reference, (680, 210), 100, (20, 210, 245), 14)
    cv2.line(reference, (80, 790), (840, 500), (245, 245, 245), 17)
    cv2.putText(
        reference,
        "MAP",
        (270, 680),
        cv2.FONT_HERSHEY_SIMPLEX,
        3.0,
        (25, 25, 25),
        12,
        cv2.LINE_AA,
    )
    return reference


def test_automatic_alignment_recovers_known_homography() -> None:
    """Алгоритм должен найти скрытое положение перспективного кадра.

    Истинная матрица передаётся только независимому оценщику после завершения
    поиска. Таким образом тест обнаружит код, который случайно воспользовался
    параметрами генератора вместо содержимого изображений.
    """

    reference = _textured_reference()
    synthetic = generate_synthetic_frame(
        reference,
        reference_resolution_m_per_pixel=0.1,
        spec=SyntheticFrameSpec(
            footprint_width_m=58.0,
            footprint_height_m=43.0,
            output_width_pixels=640,
            output_height_pixels=480,
            rotation_degrees=19.0,
            perspective_strength=0.35,
        ),
    )
    result = align_frame_to_map_automatically(
        reference,
        synthetic.image_rgb,
        config=AutomaticMapAlignmentConfig(
            analysis_maximum_side_pixels=900,
            maximum_features_per_image=5_000,
        ),
    )
    evaluation = evaluate_homography(
        result.homography_frame_to_reference,
        synthetic.homography_frame_to_reference,
        frame_width_pixels=synthetic.spec.output_width_pixels,
        frame_height_pixels=synthetic.spec.output_height_pixels,
        reference_resolution_m_per_pixel=0.1,
    )

    assert result.inlier_count >= 20
    assert result.inlier_ratio > 0.5
    assert result.frame_coverage_fraction > 0.1
    assert result.map_overlap_fraction == pytest.approx(1.0)
    assert result.inlier_reprojection_p95_reference_px < 1.0
    assert evaluation.max_error_pixels < 1.0


def test_homography_is_lifted_back_from_analysis_resolution() -> None:
    """Уменьшение больших входов не должно менять систему координат ответа."""

    reference = _textured_reference(size=1_200)
    synthetic = generate_synthetic_frame(
        reference,
        reference_resolution_m_per_pixel=0.1,
        spec=SyntheticFrameSpec(
            footprint_width_m=70.0,
            footprint_height_m=52.0,
            output_width_pixels=800,
            output_height_pixels=600,
            rotation_degrees=-13.0,
            perspective_strength=0.2,
        ),
    )
    result = align_frame_to_map_automatically(
        reference,
        synthetic.image_rgb,
        config=AutomaticMapAlignmentConfig(
            analysis_maximum_side_pixels=700,
            maximum_features_per_image=4_000,
        ),
    )
    evaluation = evaluate_homography(
        result.homography_frame_to_reference,
        synthetic.homography_frame_to_reference,
        frame_width_pixels=800,
        frame_height_pixels=600,
        reference_resolution_m_per_pixel=0.1,
    )
    assert evaluation.max_error_pixels < 1.5


def test_rejected_hypothesis_is_preserved_for_diagnostics() -> None:
    """Защитный отказ не должен уничтожать данные, нужные для разбора."""

    reference = _textured_reference(size=700)
    synthetic = generate_synthetic_frame(
        reference,
        reference_resolution_m_per_pixel=0.1,
        spec=SyntheticFrameSpec(
            footprint_width_m=45.0,
            footprint_height_m=34.0,
            output_width_pixels=500,
            output_height_pixels=380,
            rotation_degrees=11.0,
            perspective_strength=0.2,
        ),
    )
    # Требование покрыть 100% кадра заведомо строже реального распределения
    # SIFT-точек. Поиск должен найти правильную геометрию, но gate её отклонит.
    config = AutomaticMapAlignmentConfig(
        analysis_maximum_side_pixels=700,
        maximum_features_per_image=4_000,
        minimum_frame_coverage_fraction=1.0,
    )
    with pytest.raises(AutomaticMapAlignmentFailure) as failure:
        align_frame_to_map_automatically(reference, synthetic.image_rgb, config=config)

    diagnostics = failure.value.diagnostics
    assert diagnostics is not None
    assert diagnostics.best_candidate is not None
    assert diagnostics.best_candidate.inlier_count >= 8
    assert diagnostics.ratio_match_counts
    assert diagnostics.rejection_reasons
    candidate = diagnostics.best_candidate
    assert (
        f"{candidate.inlier_count}/{candidate.mutual_match_count} inlier"
        in str(failure.value)
    )
    assert all(reason in str(failure.value) for reason in diagnostics.rejection_reasons)


def test_blank_images_produce_explicit_refusal() -> None:
    """Пустые изображения не должны порождать правдоподобную случайную матрицу."""

    blank_map = np.zeros((400, 500, 3), dtype=np.uint8)
    blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)

    with pytest.raises(AutomaticMapAlignmentFailure, match="недостаточно SIFT"):
        align_frame_to_map_automatically(blank_map, blank_frame)


def test_scale_orientation_voting_rejects_larger_weak_false_cluster() -> None:
    """Сильный согласованный кластер должен пережить многочисленные выбросы.

    Пять правильных пар имеют один поворот и масштаб. Шестая почти совпадает с
    первой и имитирует несколько SIFT-детекторов на одной детали. Двенадцать
    ложных пар образуют другой, но существенно менее однозначный кластер.
    Тест проверяет одновременно взвешенное голосование и удаление дубля.
    """

    frame_keypoints: list[cv2.KeyPoint] = []
    reference_keypoints: list[cv2.KeyPoint] = []
    matches: list[_MutualMatch] = []

    def add_pair(
        *,
        frame_xy: tuple[float, float],
        reference_xy: tuple[float, float],
        frame_size: float,
        reference_size: float,
        frame_angle: float,
        reference_angle: float,
        ratio: float,
    ) -> None:
        index = len(matches)
        frame_keypoints.append(
            cv2.KeyPoint(*frame_xy, frame_size, frame_angle)
        )
        reference_keypoints.append(
            cv2.KeyPoint(*reference_xy, reference_size, reference_angle)
        )
        matches.append(
            _MutualMatch(
                dmatch=cv2.DMatch(
                    _queryIdx=index,
                    _trainIdx=index,
                    _distance=ratio,
                ),
                forward_ratio=ratio,
                reverse_ratio=ratio,
            )
        )

    for index in range(5):
        add_pair(
            frame_xy=(20.0 + 30.0 * index, 25.0 + 18.0 * index),
            reference_xy=(100.0 + 60.0 * index, 120.0 + 36.0 * index),
            frame_size=10.0,
            reference_size=20.0,
            frame_angle=10.0 + index,
            reference_angle=40.0 + index,
            ratio=0.2,
        )
    add_pair(
        frame_xy=(21.0, 26.0),
        reference_xy=(101.0, 121.0),
        frame_size=10.0,
        reference_size=20.0,
        frame_angle=10.0,
        reference_angle=40.0,
        ratio=0.2,
    )
    for index in range(12):
        add_pair(
            frame_xy=(250.0 + 12.0 * index, 50.0 + 17.0 * index),
            reference_xy=(700.0 - 13.0 * index, 80.0 + 19.0 * index),
            frame_size=20.0,
            reference_size=10.0,
            frame_angle=20.0,
            reference_angle=300.0,
            ratio=0.7,
        )

    filtered = _filter_matches_by_scale_and_orientation(
        matches,
        frame_keypoints=tuple(frame_keypoints),
        reference_keypoints=tuple(reference_keypoints),
        config=AutomaticMapAlignmentConfig(),
    )

    assert filtered.unfiltered_match_count == 18
    assert len(filtered.matches) == 5
    assert filtered.center_degrees == pytest.approx(30.0)
    assert filtered.center_log2_scale == pytest.approx(1.0)
    assert all(match.dmatch.queryIdx < 6 for match in filtered.matches)
