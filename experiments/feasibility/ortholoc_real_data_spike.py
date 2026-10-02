#!/usr/bin/env python3
"""Проверяет текущий SIFT baseline на реальных demo-кадрах OrthoLoC.

Алгоритм привязки получает только RGB UAV-кадра и RGB-ортофото. Плотная
3D-карта, DSM и параметры камеры открываются оценщику лишь после завершения
SIFT + ratio test + RANSAC. Поэтому визуализация может сравнить найденное
положение с истиной, но истина не помогает поиску.

Два demo-образца являются исследовательской выборкой. Они способны быстро
опровергнуть слабую гипотезу, но не могут подтвердить общую робастность и не
используются для ретроспективного назначения порога «годен/не годен».
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from aerial_mapper.alignment import (
    AlignmentFailure,
    AlignmentResult,
    align_frame_to_reference,
)
from aerial_mapper.dense_ground_truth import (
    DenseHomographyEvaluation,
    build_regular_query_grid,
    evaluate_homography_against_dense_truth,
    fit_best_single_homography,
    point_map_to_reference_pixels,
    sample_dense_map_bilinearly,
    select_valid_dense_correspondences,
)
from aerial_mapper.quality import analyze_alignment_quality
from aerial_mapper.synthetic import FloatPoints, Homography
from aerial_mapper.visualization import draw_alignment_matches

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data/queries/ortholoc"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs/ortholoc_real_data_spike.csv"
OUTPUT_SUMMARY_PATH = PROJECT_ROOT / "outputs/ortholoc_real_data_spike_summary.json"
OUTPUT_REPORT_TEMPLATE = "ortholoc_{sample_id}_report.png"

CONTROL_GRID_COLUMNS = 41
CONTROL_GRID_ROWS = 31


@dataclass(frozen=True)
class OrthoLocSample:
    """Минимальный набор полей официального NPZ, нужный нашему эксперименту."""

    sample_id: str
    query_rgb: np.ndarray
    reference_rgb: np.ndarray
    point_map_xyz: np.ndarray
    dsm_xyz: np.ndarray
    scale_xy_m_per_pixel: np.ndarray


def load_sample(path: Path) -> OrthoLocSample:
    """Загружает NPZ без зависимости от тяжёлого Python-пакета OrthoLoC."""

    with np.load(path, allow_pickle=False) as data:
        required_keys = {
            "sample_id",
            "image_query",
            "image_dop",
            "point_map",
            "dsm",
            "scale",
        }
        missing_keys = required_keys.difference(data.files)
        if missing_keys:
            raise ValueError(
                f"{path.name} не содержит обязательные поля: {sorted(missing_keys)}"
            )
        return OrthoLocSample(
            sample_id=str(data["sample_id"].item()),
            query_rgb=np.asarray(data["image_query"], dtype=np.uint8),
            reference_rgb=np.asarray(data["image_dop"], dtype=np.uint8),
            point_map_xyz=np.asarray(data["point_map"], dtype=np.float64),
            dsm_xyz=np.asarray(data["dsm"], dtype=np.float64),
            scale_xy_m_per_pixel=np.asarray(data["scale"], dtype=np.float64),
        )


def build_boundary_points(
    width: int, height: int, count_per_side: int = 120
) -> FloatPoints:
    """Возвращает упорядоченный контур кадра с точками вдоль каждой стороны."""

    left = 0.5
    right = width - 1.5
    top = 0.5
    bottom = height - 1.5
    top_edge = np.column_stack(
        (np.linspace(left, right, count_per_side), np.full(count_per_side, top))
    )
    right_edge = np.column_stack(
        (
            np.full(count_per_side, right),
            np.linspace(top, bottom, count_per_side),
        )
    )
    bottom_edge = np.column_stack(
        (
            np.linspace(right, left, count_per_side),
            np.full(count_per_side, bottom),
        )
    )
    left_edge = np.column_stack(
        (
            np.full(count_per_side, left),
            np.linspace(bottom, top, count_per_side),
        )
    )
    return np.vstack((top_edge, right_edge, bottom_edge, left_edge)).astype(np.float32)


def project_points(points_px: FloatPoints, homography: Homography) -> FloatPoints:
    """Проецирует набор двумерных точек проверяемой гомографией."""

    return cv2.perspectiveTransform(
        points_px.reshape(1, -1, 2),
        homography,
    ).reshape(-1, 2)


def calculate_match_errors_meters(
    alignment: AlignmentResult,
    dense_reference_map_px: np.ndarray,
    scale_xy_m_per_pixel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Сравнивает каждую принятую SIFT-пару с плотной геометрической истиной."""

    true_reference_points, valid_mask = sample_dense_map_bilinearly(
        dense_reference_map_px,
        alignment.frame_points_px,
    )
    predicted_reference_points = alignment.reference_points_px[valid_mask]
    true_reference_points = true_reference_points[valid_mask]
    errors = np.linalg.norm(
        (predicted_reference_points.astype(np.float64) - true_reference_points)
        * np.abs(scale_xy_m_per_pixel)[np.newaxis, :],
        axis=1,
    )
    return errors, alignment.inlier_mask[valid_mask]


def evaluation_to_metrics(
    prefix: str,
    evaluation: DenseHomographyEvaluation,
) -> dict[str, float | int]:
    """Преобразует оценку в плоские поля CSV/JSON с явными единицами."""

    return {
        f"{prefix}_control_point_count": int(evaluation.errors_meters.size),
        f"{prefix}_mean_error_meters": evaluation.mean_error_meters,
        f"{prefix}_median_error_meters": evaluation.median_error_meters,
        f"{prefix}_p95_error_meters": evaluation.p95_error_meters,
        f"{prefix}_max_error_meters": evaluation.max_error_meters,
    }


def draw_report(
    sample: OrthoLocSample,
    alignment: AlignmentResult,
    estimated_evaluation: DenseHomographyEvaluation,
    oracle_evaluation: DenseHomographyEvaluation,
    oracle_homography: Homography,
    dense_reference_map_px: np.ndarray,
    output_path: Path,
) -> None:
    """Создаёт единый учебный отчёт: входы, соответствия и карту ошибок."""

    query_height, query_width = sample.query_rgb.shape[:2]
    boundary_query = build_boundary_points(query_width, query_height)
    true_boundary, true_boundary_mask = sample_dense_map_bilinearly(
        dense_reference_map_px,
        boundary_query,
    )
    estimated_boundary = project_points(
        boundary_query,
        alignment.homography_frame_to_reference,
    )
    oracle_boundary = project_points(boundary_query, oracle_homography)
    matches_rgb = draw_alignment_matches(
        sample.reference_rgb,
        sample.query_rgb,
        alignment,
        target_height=650,
        reference_label="ORTHOPHOTO",
        frame_label="REAL UAV QUERY",
    )

    figure = plt.figure(figsize=(20, 13))
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 1.05))
    query_axis = figure.add_subplot(grid[0, 0])
    reference_axis = figure.add_subplot(grid[0, 1])
    dsm_axis = figure.add_subplot(grid[0, 2])
    matches_axis = figure.add_subplot(grid[1, :2])
    error_axis = figure.add_subplot(grid[1, 2])

    query_axis.imshow(sample.query_rgb)
    query_axis.set_title("Реальный UAV-кадр — вход алгоритма")
    query_axis.axis("off")

    reference_axis.imshow(sample.reference_rgb)
    valid_true_boundary = true_boundary[true_boundary_mask]
    reference_axis.scatter(
        valid_true_boundary[:, 0],
        valid_true_boundary[:, 1],
        s=7,
        c="#ff8c00",
        label="Плотная 3D-истина",
    )
    reference_axis.plot(
        estimated_boundary[:, 0],
        estimated_boundary[:, 1],
        color="#00e5ff",
        linewidth=2.0,
        label="SIFT + RANSAC, одна H",
    )
    reference_axis.plot(
        oracle_boundary[:, 0],
        oracle_boundary[:, 1],
        color="#ff4dff",
        linewidth=1.7,
        linestyle="--",
        label="Лучшая одна H по истине",
    )
    reference_axis.set_xlim(0, sample.reference_rgb.shape[1])
    reference_axis.set_ylim(sample.reference_rgb.shape[0], 0)
    reference_axis.set_title("Где кадр должен находиться на ортофото")
    reference_axis.legend(loc="lower right", fontsize=8)
    reference_axis.axis("off")

    elevation = np.ma.masked_invalid(sample.dsm_xyz[:, :, 2])
    elevation_image = dsm_axis.imshow(elevation, cmap="terrain")
    dsm_axis.set_title("DSM: высота видимых поверхностей, м")
    dsm_axis.axis("off")
    figure.colorbar(elevation_image, ax=dsm_axis, shrink=0.78, label="Высота, м")

    matches_axis.imshow(matches_rgb)
    matches_axis.set_title(
        "SIFT-пары: зелёные поддержали найденную гомографию, красные отклонены"
    )
    matches_axis.axis("off")

    error_axis.imshow(sample.query_rgb)
    color_limit = max(estimated_evaluation.p95_error_meters, 1e-6)
    error_points = error_axis.scatter(
        estimated_evaluation.query_points_px[:, 0],
        estimated_evaluation.query_points_px[:, 1],
        c=estimated_evaluation.errors_meters,
        s=18,
        cmap="magma",
        norm=Normalize(vmin=0.0, vmax=color_limit, clip=True),
        alpha=0.88,
    )
    error_axis.set_title(
        "Ошибка планового положения\n"
        f"median={estimated_evaluation.median_error_meters:.2f} м, "
        f"P95={estimated_evaluation.p95_error_meters:.2f} м"
    )
    error_axis.axis("off")
    figure.colorbar(
        error_points,
        ax=error_axis,
        shrink=0.78,
        label="Ошибка, м (цвет ограничен P95)",
    )

    figure.suptitle(
        f"OrthoLoC {sample.sample_id}: реальный кадр против ортографической карты\n"
        f"SIFT inlier: {alignment.inlier_count}/{alignment.ratio_match_count}; "
        f"SIFT median={estimated_evaluation.median_error_meters:.2f} м; "
        f"лучшая одна H median={oracle_evaluation.median_error_meters:.2f} м",
        fontsize=16,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_sample(path: Path) -> tuple[dict[str, Any], Path | None]:
    """Выполняет независимую привязку и только затем открывает истинную геометрию."""

    sample = load_sample(path)
    common: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "source_filename": path.name,
        "query_width_pixels": sample.query_rgb.shape[1],
        "query_height_pixels": sample.query_rgb.shape[0],
        "reference_width_pixels": sample.reference_rgb.shape[1],
        "reference_height_pixels": sample.reference_rgb.shape[0],
        "reference_scale_x_m_per_pixel": sample.scale_xy_m_per_pixel[0],
        "reference_scale_y_m_per_pixel": sample.scale_xy_m_per_pixel[1],
    }
    try:
        # До этой строки алгоритм видит только две RGB-картинки. Point map, DSM,
        # scale и camera pose намеренно не передаются функции привязки.
        alignment = align_frame_to_reference(
            sample.reference_rgb,
            sample.query_rgb,
        )
    except AlignmentFailure as error:
        return (
            {
                **common,
                "estimation_returned": False,
                "failure_reason": str(error),
            },
            None,
        )

    reference_offset_xy = sample.dsm_xyz[0, 0, :2]
    dense_reference_map = point_map_to_reference_pixels(
        sample.point_map_xyz,
        reference_offset_xy=reference_offset_xy,
        reference_scale_xy_m_per_pixel=sample.scale_xy_m_per_pixel,
    )
    control_grid = build_regular_query_grid(
        frame_width_pixels=sample.query_rgb.shape[1],
        frame_height_pixels=sample.query_rgb.shape[0],
        columns=CONTROL_GRID_COLUMNS,
        rows=CONTROL_GRID_ROWS,
    )
    query_points, true_reference_points = select_valid_dense_correspondences(
        control_grid,
        dense_reference_map,
        reference_width_pixels=sample.reference_rgb.shape[1],
        reference_height_pixels=sample.reference_rgb.shape[0],
    )
    estimated_evaluation = evaluate_homography_against_dense_truth(
        alignment.homography_frame_to_reference,
        query_points,
        true_reference_points,
        reference_scale_xy_m_per_pixel=sample.scale_xy_m_per_pixel,
    )

    # Эта матрица строится из правильных ответов только после SIFT-прогона. Она
    # не является альтернативным локализатором: это оптимистичная характеристика
    # того, насколько одну H вообще можно натянуть на неплоскую видимую сцену.
    oracle_homography = fit_best_single_homography(
        query_points,
        true_reference_points,
    )
    oracle_evaluation = evaluate_homography_against_dense_truth(
        oracle_homography,
        query_points,
        true_reference_points,
        reference_scale_xy_m_per_pixel=sample.scale_xy_m_per_pixel,
    )
    match_errors, valid_match_inliers = calculate_match_errors_meters(
        alignment,
        dense_reference_map,
        sample.scale_xy_m_per_pixel,
    )
    inlier_match_errors = match_errors[valid_match_inliers]
    quality = analyze_alignment_quality(
        alignment,
        frame_width_pixels=sample.query_rgb.shape[1],
        frame_height_pixels=sample.query_rgb.shape[0],
    )
    report_path = (
        PROJECT_ROOT
        / "outputs"
        / OUTPUT_REPORT_TEMPLATE.format(sample_id=sample.sample_id)
    )
    draw_report(
        sample,
        alignment,
        estimated_evaluation,
        oracle_evaluation,
        oracle_homography,
        dense_reference_map,
        report_path,
    )

    row: dict[str, Any] = {
        **common,
        "estimation_returned": True,
        "failure_reason": "",
        "reference_keypoints": alignment.reference_keypoint_count,
        "query_keypoints": alignment.frame_keypoint_count,
        "ratio_matches": alignment.ratio_match_count,
        "ransac_inliers": alignment.inlier_count,
        "inlier_ratio": alignment.inlier_ratio,
        "inlier_spatial_coverage_fraction": (
            alignment.inlier_spatial_coverage_fraction
        ),
        "grid_occupancy_fraction": quality.grid_occupancy_fraction,
        "stability_trials_succeeded": quality.stability_trials_succeeded,
        "stability_p95_max_corner_shift_reference_px": (
            quality.stability_p95_max_corner_shift_reference_px
        ),
        "valid_sift_matches_with_dense_truth": int(match_errors.size),
        "all_sift_match_median_error_meters": float(np.median(match_errors)),
        "ransac_inlier_match_median_error_meters": (
            float(np.median(inlier_match_errors)) if inlier_match_errors.size else None
        ),
        "ransac_inlier_match_p95_error_meters": (
            float(np.percentile(inlier_match_errors, 95))
            if inlier_match_errors.size
            else None
        ),
        "processing_time_seconds": alignment.processing_time_seconds,
        "report_path": str(report_path.relative_to(PROJECT_ROOT)),
        **evaluation_to_metrics("estimated_homography", estimated_evaluation),
        **evaluation_to_metrics("oracle_single_homography", oracle_evaluation),
    }
    return row, report_path


def save_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Сохраняет подробную таблицу и честную сводку без порога прохождения."""

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    field_names = sorted({key for row in rows for key in row})
    with OUTPUT_CSV_PATH.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)

    successful_rows = [row for row in rows if row["estimation_returned"]]
    summary = {
        "experiment_role": (
            "Первый исследовательский прогон на независимых реальных UAV-кадрах; "
            "два demo-образца не являются достаточной валидационной выборкой."
        ),
        "source": "OrthoLoC official demo, CC BY-NC-SA 4.0",
        "sample_count": len(rows),
        "estimation_returned_count": len(successful_rows),
        "explicit_failure_count": len(rows) - len(successful_rows),
        "control_grid_shape": [CONTROL_GRID_ROWS, CONTROL_GRID_COLUMNS],
        "results": rows,
        "interpretation_warning": (
            "RANSAC inlier означает согласие пары с найденной гомографией, а не "
            "её согласие с плотной 3D-истиной. Численный порог годности заранее "
            "не задан и по этим двум просмотренным сценам не подбирается."
        ),
    }
    with OUTPUT_SUMMARY_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    return summary


def main() -> None:
    """Запускает два demo-сценария и печатает пути к визуальным артефактам."""

    sample_paths = sorted(DATA_DIRECTORY.glob("*.npz"))
    if not sample_paths:
        raise FileNotFoundError(
            "Demo OrthoLoC не найдены. Выполните: "
            "uv run python scripts/download_ortholoc_demo.py"
        )

    rows: list[dict[str, Any]] = []
    report_paths: list[Path] = []
    for sample_path in sample_paths:
        row, report_path = run_sample(sample_path)
        rows.append(row)
        if report_path is not None:
            report_paths.append(report_path)

    summary = save_results(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Таблица: {OUTPUT_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Сводка: {OUTPUT_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    for report_path in report_paths:
        print(f"Визуальный отчёт: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
