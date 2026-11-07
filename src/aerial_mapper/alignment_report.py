"""Экспорт воспроизводимого сравнения ручного и автоматического совмещения."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
from numpy.typing import NDArray

from aerial_mapper import __version__
from aerial_mapper.automatic_map_alignment import (
    AutomaticMapAlignmentDiagnostics,
    AutomaticMapAlignmentResult,
)
from aerial_mapper.manual_alignment import ManualAlignmentResult


def export_alignment_comparison_package(
    destination: str | Path,
    *,
    raw_frame_bgr: NDArray[np.uint8],
    automatic_frame_bgr: NDArray[np.uint8],
    reference_map_bgr: NDArray[np.uint8],
    manual_result: ManualAlignmentResult,
    manual_points_in_automatic_frame_px: NDArray[np.float64],
    automatic_status: str,
    automatic_candidate: AutomaticMapAlignmentResult | None,
    automatic_diagnostics: AutomaticMapAlignmentDiagnostics | None,
    automatic_message: str,
    correction_note: str,
    video_source_name: str,
    map_source_name: str,
    frame_index: int,
) -> Path:
    """Сохранить один ZIP для передачи и повторного анализа.

    ``manual_points_in_automatic_frame_px`` — те же физические лучи камеры,
    что ручные ``frame_points_px``, но в системе координат входа автоматики.
    Для обычного видео массивы совпадают. Для MARS-LVIG точки исправляются от
    дисторсии, как и изображение перед SIFT. Без этого сравнение двух матриц
    выглядело бы численно точным, но смешивало бы разные системы координат.

    Допустимые состояния автоматики: ``accepted``, ``rejected`` и ``error``.
    Отвергнутая гипотеза сохраняется только как диагностический кандидат.
    """

    if automatic_status not in {"accepted", "rejected", "error"}:
        raise ValueError("Неизвестное состояние автоматического совмещения")
    _validate_bgr(raw_frame_bgr, name="Сырой кадр")
    _validate_bgr(automatic_frame_bgr, name="Вход автоматики")
    _validate_bgr(reference_map_bgr, name="Карта")
    automatic_manual_points = np.asarray(
        manual_points_in_automatic_frame_px, dtype=np.float64
    )
    if automatic_manual_points.shape != manual_result.frame_points_px.shape:
        raise ValueError("Ручные точки в системе автоматики имеют неверную форму")
    if not np.isfinite(automatic_manual_points).all():
        raise ValueError("Ручные точки в системе автоматики должны быть конечными")
    if frame_index < 0:
        raise ValueError("Индекс кадра не может быть отрицательным")

    comparison, comparison_metrics = _make_comparison_image(
        raw_frame_bgr=raw_frame_bgr,
        automatic_frame_bgr=automatic_frame_bgr,
        reference_map_bgr=reference_map_bgr,
        manual_result=manual_result,
        manual_points_in_automatic_frame_px=automatic_manual_points,
        automatic_status=automatic_status,
        automatic_candidate=automatic_candidate,
    )
    image_arrays = {
        "frame_raw.png": raw_frame_bgr,
        "frame_automatic_input.png": automatic_frame_bgr,
        "reference_map.png": reference_map_bgr,
        "comparison.png": comparison,
    }
    image_payloads = {
        name: _encode_png(image) for name, image in image_arrays.items()
    }

    report = {
        "schema_version": 2,
        "created_utc": datetime.now(UTC).isoformat(),
        "application": {
            "name": "aerial-map-measurement",
            "version": __version__,
        },
        "sources": {
            # В отчёт намеренно не пишутся абсолютные локальные пути: имена и
            # сами изображения достаточны для воспроизведения, а структура
            # домашнего каталога не должна утекать при передаче ZIP.
            "video_name": video_source_name,
            "map_name": map_source_name,
            "frame_index_zero_based": frame_index,
            "frame_number_one_based": frame_index + 1,
            "raw_frame_size_px": _image_size(raw_frame_bgr),
            "automatic_frame_size_px": _image_size(automatic_frame_bgr),
            "map_size_px": _image_size(reference_map_bgr),
        },
        "coordinate_systems": {
            "manual_frame": "pixels of frame_raw.png",
            "automatic_frame": "pixels of frame_automatic_input.png",
            "map": "pixels of reference_map.png",
            "pixel_origin": "top-left; x right; y down",
            "automatic_input_preprocessing": correction_note,
        },
        "manual": _manual_report(manual_result),
        "automatic": _automatic_report(
            status=automatic_status,
            candidate=automatic_candidate,
            diagnostics=automatic_diagnostics,
            message=automatic_message,
        ),
        "comparison_to_manual": comparison_metrics,
        "files": {
            name: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
            for name, payload in image_payloads.items()
        },
    }
    report_payload = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")

    output_path = Path(destination).expanduser().resolve()
    if output_path.suffix.lower() != ".zip":
        output_path = output_path.with_suffix(".zip")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("report.json", report_payload)
        for name, payload in image_payloads.items():
            archive.writestr(name, payload)
    return output_path


def _manual_report(result: ManualAlignmentResult) -> dict[str, object]:
    return {
        "homography_frame_to_map": _array_values(
            result.homography_frame_to_map
        ),
        "frame_points_px": _array_values(result.frame_points_px),
        "map_points_px": _array_values(result.map_points_px),
        "fit_errors_map_px": _array_values(result.fit_errors_map_px),
        "leave_one_out_errors_map_px": _array_values(
            result.leave_one_out_errors_map_px
        ),
        "fit_rmse_map_px": result.fit_rmse_map_px,
        "fit_max_map_px": result.fit_max_map_px,
        "leave_one_out_median_map_px": result.leave_one_out_median_map_px,
        "leave_one_out_max_map_px": result.leave_one_out_max_map_px,
        "frame_coverage_fraction": result.frame_coverage_fraction,
        "map_coverage_fraction": result.map_coverage_fraction,
        "projected_frame_corners_map_px": _array_values(
            result.projected_frame_corners_map_px
        ),
    }


def _automatic_report(
    *,
    status: str,
    candidate: AutomaticMapAlignmentResult | None,
    diagnostics: AutomaticMapAlignmentDiagnostics | None,
    message: str,
) -> dict[str, object]:
    report: dict[str, object] = {
        "status": status,
        "message": message,
        "candidate_is_safe_to_use": status == "accepted",
        "candidate": _automatic_candidate_report(candidate),
    }
    if diagnostics is not None:
        filtered_counts = dict(diagnostics.filtered_match_counts)
        report["rejection_diagnostics"] = {
            "reference_keypoint_count": diagnostics.reference_keypoint_count,
            "frame_keypoint_count": diagnostics.frame_keypoint_count,
            "ratio_match_counts": [
                {
                    "ratio_threshold": ratio,
                    "mutual_match_count_before_filter": count,
                    "mutual_match_count_after_filter": filtered_counts.get(ratio),
                }
                for ratio, count in diagnostics.ratio_match_counts
            ],
            "tested_hypothesis_count": diagnostics.tested_hypothesis_count,
            "rejection_reasons": list(diagnostics.rejection_reasons),
            "processing_time_seconds": diagnostics.processing_time_seconds,
        }
    return report


def _automatic_candidate_report(
    result: AutomaticMapAlignmentResult | None,
) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "homography_frame_to_map": _array_values(
            result.homography_frame_to_reference
        ),
        "frame_points_px": _array_values(result.frame_points_px),
        "map_points_px": _array_values(result.reference_points_px),
        "inlier_mask": result.inlier_mask.tolist(),
        "selected_ratio_threshold": result.selected_ratio_threshold,
        "tested_hypothesis_count": result.tested_hypothesis_count,
        "reference_keypoint_count": result.reference_keypoint_count,
        "frame_keypoint_count": result.frame_keypoint_count,
        "mutual_match_count": result.mutual_match_count,
        "unfiltered_mutual_match_count": result.unfiltered_mutual_match_count,
        "scale_orientation_cluster": {
            "center_degrees": result.scale_orientation_center_degrees,
            "center_log2_scale": result.scale_orientation_center_log2_scale,
            "vote_score": result.scale_orientation_vote_score,
        },
        "inlier_count": result.inlier_count,
        "inlier_ratio": result.inlier_ratio,
        "frame_coverage_fraction": result.frame_coverage_fraction,
        "projected_frame_corners_map_px": _array_values(
            result.projected_frame_corners_reference_px
        ),
        "projected_area_fraction": result.projected_area_fraction,
        "map_overlap_fraction": result.map_overlap_fraction,
        "projected_edge_ratio": result.projected_edge_ratio,
        "inlier_reprojection_median_map_px": (
            result.inlier_reprojection_median_reference_px
        ),
        "inlier_reprojection_p95_map_px": (
            result.inlier_reprojection_p95_reference_px
        ),
        "inlier_reprojection_max_map_px": (
            result.inlier_reprojection_max_reference_px
        ),
        "processing_time_seconds": result.processing_time_seconds,
    }


def _make_comparison_image(
    *,
    raw_frame_bgr: NDArray[np.uint8],
    automatic_frame_bgr: NDArray[np.uint8],
    reference_map_bgr: NDArray[np.uint8],
    manual_result: ManualAlignmentResult,
    manual_points_in_automatic_frame_px: NDArray[np.float64],
    automatic_status: str,
    automatic_candidate: AutomaticMapAlignmentResult | None,
) -> tuple[NDArray[np.uint8], dict[str, object]]:
    manual_panel = _warp_and_blend(
        reference_map_bgr,
        raw_frame_bgr,
        manual_result.homography_frame_to_map,
    )
    for index, point in enumerate(manual_result.map_points_px):
        _draw_point(manual_panel, point, (80, 230, 80), index + 1)
    manual_panel = _add_header(manual_panel, "MANUAL REFERENCE", (80, 230, 80))

    if automatic_candidate is None:
        automatic_panel = reference_map_bgr.copy()
        automatic_panel = _add_header(
            automatic_panel,
            "AUTO: NO GEOMETRIC CANDIDATE",
            (40, 80, 240),
        )
        comparison_metrics: dict[str, object] = {
            "available": False,
            "reason": "automatic candidate is unavailable",
        }
    else:
        automatic_panel = _warp_and_blend(
            reference_map_bgr,
            automatic_frame_bgr,
            automatic_candidate.homography_frame_to_reference,
        )
        predicted = cv2.perspectiveTransform(
            manual_points_in_automatic_frame_px.reshape(1, -1, 2),
            automatic_candidate.homography_frame_to_reference,
        ).reshape(-1, 2)
        errors = np.linalg.norm(predicted - manual_result.map_points_px, axis=1)
        for truth, estimate in zip(
            manual_result.map_points_px, predicted, strict=True
        ):
            truth_px = tuple(np.rint(truth).astype(int))
            estimate_px = tuple(np.rint(estimate).astype(int))
            cv2.line(automatic_panel, truth_px, estimate_px, (0, 215, 255), 2)
            cv2.circle(automatic_panel, truth_px, 6, (80, 230, 80), 2)
            cv2.drawMarker(
                automatic_panel,
                estimate_px,
                (230, 80, 230),
                cv2.MARKER_TILTED_CROSS,
                14,
                2,
            )
        inlier_points = automatic_candidate.reference_points_px[
            automatic_candidate.inlier_mask
        ]
        for point in inlier_points:
            cv2.circle(
                automatic_panel,
                tuple(np.rint(point).astype(int)),
                2,
                (230, 80, 230),
                -1,
            )
        state_label = "ACCEPTED" if automatic_status == "accepted" else "REJECTED"
        state_color = (
            (80, 230, 80) if automatic_status == "accepted" else (40, 80, 240)
        )
        automatic_panel = _add_header(
            automatic_panel,
            f"AUTO {state_label} | error to manual: median {np.median(errors):.1f}px",
            state_color,
        )
        comparison_metrics = {
            "available": True,
            "note": (
                "Automatic homography evaluated at manually labelled camera rays; "
                "errors are measured in map pixels."
            ),
            "manual_points_in_automatic_frame_px": _array_values(
                manual_points_in_automatic_frame_px
            ),
            "automatic_predictions_map_px": _array_values(predicted),
            "errors_map_px": _array_values(errors),
            "median_error_map_px": float(np.median(errors)),
            "p95_error_map_px": float(np.percentile(errors, 95)),
            "max_error_map_px": float(np.max(errors)),
        }
    return np.hstack((manual_panel, automatic_panel)), comparison_metrics


def _warp_and_blend(
    map_bgr: NDArray[np.uint8],
    frame_bgr: NDArray[np.uint8],
    homography: NDArray[np.float64],
) -> NDArray[np.uint8]:
    height, width = map_bgr.shape[:2]
    warped = cv2.warpPerspective(frame_bgr, homography, (width, height))
    mask = cv2.warpPerspective(
        np.full(frame_bgr.shape[:2], 255, dtype=np.uint8),
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
    )
    mixed = cv2.addWeighted(map_bgr, 0.55, warped, 0.45, 0.0)
    output = map_bgr.copy()
    output[mask > 0] = mixed[mask > 0]
    return output


def _draw_point(
    image: NDArray[np.uint8],
    point: NDArray[np.float64],
    color: tuple[int, int, int],
    label: int,
) -> None:
    pixel = tuple(np.rint(point).astype(int))
    cv2.circle(image, pixel, 6, color, 2)
    cv2.putText(
        image,
        str(label),
        (pixel[0] + 7, pixel[1] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def _add_header(
    image: NDArray[np.uint8], text: str, color: tuple[int, int, int]
) -> NDArray[np.uint8]:
    panel = cv2.copyMakeBorder(
        image,
        52,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(24, 24, 24),
    )
    cv2.putText(
        panel,
        text,
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )
    return panel


def _encode_png(image: NDArray[np.uint8]) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("OpenCV не смог закодировать PNG для отчёта")
    return encoded.tobytes()


def _validate_bgr(image: NDArray[np.uint8], *, name: str) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} должен быть BGR uint8 с тремя каналами")
    if min(image.shape[:2]) < 2:
        raise ValueError(f"{name} слишком мал")


def _image_size(image: NDArray[np.uint8]) -> list[int]:
    return [int(image.shape[1]), int(image.shape[0])]


def _array_values(array: NDArray[np.generic]) -> list[object]:
    values = np.asarray(array)
    if values.ndim == 1:
        return [float(value) if np.isfinite(value) else None for value in values]
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in values
    ]
