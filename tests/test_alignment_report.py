"""Проверки передаваемого пакета ручного/автоматического сравнения."""

import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from aerial_mapper.alignment_report import export_alignment_comparison_package
from aerial_mapper.automatic_map_alignment import (
    AutomaticMapAlignmentDiagnostics,
    AutomaticMapAlignmentResult,
)
from aerial_mapper.manual_alignment import estimate_manual_alignment


def _manual_and_automatic(
    *, automatic_shift_x: float = 0.0
) -> tuple[np.ndarray, np.ndarray, object, AutomaticMapAlignmentResult]:
    """Создать два результата с известным расхождением по оси карты."""

    frame = np.full((80, 100, 3), (40, 130, 220), dtype=np.uint8)
    reference = np.full((140, 160, 3), (80, 80, 80), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (88, 68), (240, 240, 240), 3)
    cv2.line(reference, (10, 100), (145, 30), (220, 220, 220), 3)
    frame_points = np.asarray(
        [[5.0, 5.0], [90.0, 8.0], [92.0, 70.0], [7.0, 72.0], [50.0, 40.0]],
        dtype=np.float64,
    )
    manual_h = np.asarray(
        [[1.2, 0.03, 15.0], [-0.02, 1.1, 12.0], [0.0002, -0.0001, 1.0]],
        dtype=np.float64,
    )
    map_points = cv2.perspectiveTransform(
        frame_points.reshape(1, -1, 2), manual_h
    ).reshape(-1, 2)
    manual = estimate_manual_alignment(
        frame_points,
        map_points,
        frame_size_pixels=(100, 80),
        map_size_pixels=(160, 140),
    )
    automatic_h = manual.homography_frame_to_map.copy()
    automatic_h[0, 2] += automatic_shift_x
    automatic_map_points = cv2.perspectiveTransform(
        frame_points.reshape(1, -1, 2), automatic_h
    ).reshape(-1, 2)
    corners = np.asarray(
        [[0.0, 0.0], [99.0, 0.0], [99.0, 79.0], [0.0, 79.0]],
        dtype=np.float64,
    )
    projected_corners = cv2.perspectiveTransform(
        corners.reshape(1, -1, 2), automatic_h
    ).reshape(-1, 2)
    automatic = AutomaticMapAlignmentResult(
        homography_frame_to_reference=automatic_h,
        frame_points_px=frame_points,
        reference_points_px=automatic_map_points,
        inlier_mask=np.ones(frame_points.shape[0], dtype=bool),
        selected_ratio_threshold=0.8,
        tested_hypothesis_count=2,
        reference_keypoint_count=100,
        frame_keypoint_count=80,
        frame_coverage_fraction=0.7,
        projected_frame_corners_reference_px=projected_corners,
        projected_area_fraction=0.45,
        map_overlap_fraction=1.0,
        projected_edge_ratio=1.4,
        inlier_reprojection_median_reference_px=0.2,
        inlier_reprojection_p95_reference_px=0.4,
        inlier_reprojection_max_reference_px=0.5,
        processing_time_seconds=0.25,
        unfiltered_mutual_match_count=17,
        scale_orientation_center_degrees=12.5,
        scale_orientation_center_log2_scale=-0.75,
        scale_orientation_vote_score=1.25,
    )
    return frame, reference, manual, automatic


def test_accepted_report_contains_images_points_and_strict_json(
    tmp_path: Path,
) -> None:
    """ZIP должен сам содержать все входы и независимую ошибку к ручным точкам."""

    frame, reference, manual, automatic = _manual_and_automatic()
    output = export_alignment_comparison_package(
        tmp_path / "accepted.zip",
        raw_frame_bgr=frame,
        automatic_frame_bgr=frame,
        reference_map_bgr=reference,
        manual_result=manual,
        manual_points_in_automatic_frame_px=manual.frame_points_px,
        automatic_status="accepted",
        automatic_candidate=automatic,
        automatic_diagnostics=None,
        automatic_message="accepted in test",
        correction_note="identity",
        video_source_name="flight-test",
        map_source_name="map-test.png",
        frame_index=12,
    )

    with ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "report.json",
            "frame_raw.png",
            "frame_automatic_input.png",
            "reference_map.png",
            "comparison.png",
        }
        report_text = archive.read("report.json").decode("utf-8")
        # Стандарт JSON не знает NaN: отсутствие этой строки защищает обмен с
        # анализаторами, которые строже стандартного json-модуля Python.
        assert "NaN" not in report_text
        report = json.loads(report_text)
        comparison = report["comparison_to_manual"]
        assert report["automatic"]["status"] == "accepted"
        assert report["automatic"]["candidate_is_safe_to_use"] is True
        assert report["schema_version"] == 2
        candidate = report["automatic"]["candidate"]
        assert candidate["unfiltered_mutual_match_count"] == 17
        assert candidate["mutual_match_count"] == 5
        assert candidate["scale_orientation_cluster"]["center_degrees"] == 12.5
        assert report["sources"]["frame_index_zero_based"] == 12
        assert comparison["max_error_map_px"] < 1e-5
        assert "/home/" not in report_text

        diagnostic = cv2.imdecode(
            np.frombuffer(archive.read("comparison.png"), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert diagnostic is not None
        assert diagnostic.shape == (192, 320, 3)


def test_rejected_candidate_is_exported_but_marked_unsafe(tmp_path: Path) -> None:
    """Ложная гипотеза нужна для разбора, но не должна выглядеть принятой."""

    frame, reference, manual, automatic = _manual_and_automatic(
        automatic_shift_x=11.0
    )
    diagnostics = AutomaticMapAlignmentDiagnostics(
        reference_keypoint_count=100,
        frame_keypoint_count=80,
        ratio_match_counts=((0.76, 4), (0.80, 12)),
        tested_hypothesis_count=1,
        best_candidate=automatic,
        rejection_reasons=("покрытие кадра слишком мало",),
        processing_time_seconds=0.3,
        filtered_match_counts=((0.76, 0), (0.80, 8)),
    )
    output = export_alignment_comparison_package(
        tmp_path / "rejected.zip",
        raw_frame_bgr=frame,
        automatic_frame_bgr=frame,
        reference_map_bgr=reference,
        manual_result=manual,
        manual_points_in_automatic_frame_px=manual.frame_points_px,
        automatic_status="rejected",
        automatic_candidate=automatic,
        automatic_diagnostics=diagnostics,
        automatic_message="геометрическая проверка отклонила гипотезу",
        correction_note="identity",
        video_source_name="flight-test",
        map_source_name="map-test.png",
        frame_index=13,
    )

    with ZipFile(output) as archive:
        report = json.loads(archive.read("report.json"))
    assert report["automatic"]["status"] == "rejected"
    assert report["automatic"]["candidate_is_safe_to_use"] is False
    assert report["automatic"]["candidate"] is not None
    assert report["automatic"]["rejection_diagnostics"][
        "rejection_reasons"
    ] == ["покрытие кадра слишком мало"]
    assert report["automatic"]["rejection_diagnostics"]["ratio_match_counts"] == [
        {
            "ratio_threshold": 0.76,
            "mutual_match_count_before_filter": 4,
            "mutual_match_count_after_filter": 0,
        },
        {
            "ratio_threshold": 0.8,
            "mutual_match_count_before_filter": 12,
            "mutual_match_count_after_filter": 8,
        },
    ]

    # При проективном преобразовании добавка к H[0, 2] делится на зависящий
    # от точки знаменатель. Поэтому ожидается близкое к 11, но не ровно 11 px.
    median_error = report["comparison_to_manual"]["median_error_map_px"]
    assert 10.8 < median_error < 11.1
