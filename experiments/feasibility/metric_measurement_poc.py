#!/usr/bin/env python3
"""Проверяет конечные расстояния и площади после независимой привязки P2.

Контрольные фигуры задаются в координатах кадра и не участвуют в поиске SIFT-
соответствий. Рабочая ветка переносит их найденной гомографией. Отдельная ветка
с истинной матрицей генератора открывается только после привязки и даёт точные
значения для оценки ошибки.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import Affine

from aerial_mapper.alignment import align_frame_to_reference
from aerial_mapper.measurement import map_frame_points_to_reference
from aerial_mapper.measurement_evaluation import (
    MetricControlEvaluation,
    build_poc_control_definitions,
    evaluate_metric_controls,
    metric_evaluation_to_dict,
)
from aerial_mapper.synthetic import SyntheticFrameSpec, generate_synthetic_frame
from aerial_mapper.visualization import (
    draw_alignment_footprints,
    draw_metric_controls_on_frame,
    draw_metric_evaluations_on_reference,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = PROJECT_ROOT / "data/reference/cherkasy_2021_poc.tif"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs/metric_measurement_poc.csv"
OUTPUT_SUMMARY_PATH = PROJECT_ROOT / "outputs/metric_measurement_poc_summary.json"
OUTPUT_PLOT_PATH = PROJECT_ROOT / "outputs/metric_measurement_poc.png"


def load_reference() -> tuple[np.ndarray, float, Affine, str]:
    """Загружает RGB, разрешение, affine transform и CRS эталона."""

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            "Эталон не найден. Выполните: uv run python scripts/download_reference.py"
        )

    with rasterio.open(REFERENCE_PATH) as dataset:
        if dataset.crs is None:
            raise ValueError("GeoTIFF не содержит CRS")
        resolution_x = float(dataset.res[0])
        resolution_y = float(dataset.res[1])
        if not np.isclose(resolution_x, resolution_y, atol=1e-9):
            raise ValueError("Эксперимент ожидает квадратные пиксели эталона")
        reference_rgb = np.moveaxis(dataset.read((1, 2, 3)), 0, -1)
        return (
            reference_rgb,
            resolution_x,
            dataset.transform,
            dataset.crs.to_string(),
        )


def run_experiment() -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[MetricControlEvaluation, ...],
    dict[str, Any],
]:
    """Выполняет одну честную привязку и семь метрических проверок."""

    reference_rgb, resolution_m_per_pixel, reference_transform, reference_crs = (
        load_reference()
    )
    spec = SyntheticFrameSpec()
    synthetic = generate_synthetic_frame(
        reference_rgb,
        reference_resolution_m_per_pixel=resolution_m_per_pixel,
        spec=spec,
    )
    alignment = align_frame_to_reference(reference_rgb, synthetic.image_rgb)
    controls = build_poc_control_definitions(
        frame_width_pixels=spec.output_width_pixels,
        frame_height_pixels=spec.output_height_pixels,
    )
    evaluations = evaluate_metric_controls(
        controls,
        estimated_homography_frame_to_reference=(
            alignment.homography_frame_to_reference
        ),
        true_homography_frame_to_reference=(synthetic.homography_frame_to_reference),
        reference_transform=reference_transform,
        reference_crs=reference_crs,
        reference_width_pixels=reference_rgb.shape[1],
        reference_height_pixels=reference_rgb.shape[0],
    )

    frame_visualization = draw_metric_controls_on_frame(
        synthetic.image_rgb,
        controls,
    )
    # Для следа камеры используем четыре угла полного кадра. Контрольные фигуры
    # расположены внутри изображения и не описывают границу видимой области.
    frame_corners = np.asarray(
        [
            [0.0, 0.0],
            [spec.output_width_pixels - 1.0, 0.0],
            [spec.output_width_pixels - 1.0, spec.output_height_pixels - 1.0],
            [0.0, spec.output_height_pixels - 1.0],
        ],
        dtype=np.float64,
    )
    estimated_corners = map_frame_points_to_reference(
        frame_corners,
        alignment.homography_frame_to_reference,
    ).astype(np.float32)
    reference_visualization = draw_alignment_footprints(
        reference_rgb,
        synthetic.source_corners_reference_px,
        estimated_corners,
    )
    reference_visualization = draw_metric_evaluations_on_reference(
        reference_visualization,
        evaluations,
    )

    segment_rows = [result for result in evaluations if result.kind == "segment"]
    polygon_rows = [result for result in evaluations if result.kind == "polygon"]
    summary: dict[str, Any] = {
        "experiment": "P2: метрические измерения на одном чистом кадре",
        "reference_crs": reference_crs,
        "reference_resolution_m_per_pixel": resolution_m_per_pixel,
        "control_count": len(evaluations),
        "segment_count": len(segment_rows),
        "polygon_count": len(polygon_rows),
        "maximum_vertex_position_error_meters": max(
            result.max_vertex_position_error_meters for result in evaluations
        ),
        "largest_absolute_segment_error": metric_evaluation_to_dict(
            max(segment_rows, key=lambda result: result.absolute_error)
        ),
        "largest_relative_segment_error": metric_evaluation_to_dict(
            max(segment_rows, key=lambda result: result.relative_error_percent)
        ),
        "largest_absolute_polygon_error": metric_evaluation_to_dict(
            max(polygon_rows, key=lambda result: result.absolute_error)
        ),
        "largest_relative_polygon_error": metric_evaluation_to_dict(
            max(polygon_rows, key=lambda result: result.relative_error_percent)
        ),
        "note": (
            "Порога прохождения нет. Истинные значения используются только "
            "после рабочей ветки для количественной оценки ошибки."
        ),
    }
    return frame_visualization, reference_visualization, evaluations, summary


def save_results(
    frame_visualization: np.ndarray,
    reference_visualization: np.ndarray,
    evaluations: tuple[MetricControlEvaluation, ...],
    summary: dict[str, Any],
) -> None:
    """Сохраняет таблицу, сводку и картинку с контрольными фигурами."""

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = [metric_evaluation_to_dict(result) for result in evaluations]
    with OUTPUT_CSV_PATH.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with OUTPUT_SUMMARY_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    figure, axes = plt.subplots(1, 2, figsize=(18, 9))
    axes[0].imshow(frame_visualization)
    axes[0].set_title("Контрольные фигуры в кадре")
    axes[1].imshow(reference_visualization)
    axes[1].set_title("Истина (оранжевый) и измерение (бирюзовый)")
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle("P2: расстояния и площади после независимой привязки")
    figure.tight_layout()
    figure.savefig(OUTPUT_PLOT_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    """Запускает опыт и печатает сводку вместе с путями артефактов."""

    frame, reference, evaluations, summary = run_experiment()
    save_results(frame, reference, evaluations, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Подробные строки: {OUTPUT_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Сводка: {OUTPUT_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Визуализация: {OUTPUT_PLOT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
