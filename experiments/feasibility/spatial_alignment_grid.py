#!/usr/bin/env python3
"""Исследует влияние положения чистого плоского кадра на привязку P1.

Эксперимент перемещает неизменный прямоугольный след по сетке 5 × 5. Масштаб,
разрешение, поворот, перспектива и фотометрия фиксированы. Благодаря этому
изменение результата можно связать с содержимым разных участков эталона, а не
со смесью нескольких факторов.

Серия является исследовательской: после просмотра её результатов нельзя
подбирать пороги качества и на тех же строках объявлять их подтверждёнными.
Для будущей проверки порогов потребуется отдельная карта или заранее скрытая
пространственная выборка.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.axes import Axes
from rasterio import Affine

from aerial_mapper.alignment import AlignmentFailure, align_frame_to_reference
from aerial_mapper.evaluation import evaluate_homography
from aerial_mapper.measurement_evaluation import (
    build_poc_control_definitions,
    evaluate_metric_controls,
    metric_evaluation_to_dict,
)
from aerial_mapper.quality import analyze_alignment_quality
from aerial_mapper.synthetic import (
    SyntheticFrameSpec,
    calculate_valid_axis_centers,
    generate_synthetic_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = PROJECT_ROOT / "data/reference/cherkasy_2021_poc.tif"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs/spatial_alignment_grid.csv"
OUTPUT_METRIC_CSV_PATH = PROJECT_ROOT / "outputs/spatial_metric_controls.csv"
OUTPUT_SUMMARY_PATH = PROJECT_ROOT / "outputs/spatial_alignment_grid_summary.json"
OUTPUT_PLOT_PATH = PROJECT_ROOT / "outputs/spatial_alignment_grid.png"
OUTPUT_METRIC_PLOT_PATH = PROJECT_ROOT / "outputs/spatial_metric_grid.png"

FOOTPRINT_WIDTH_M = 120.0
FOOTPRINT_HEIGHT_M = 90.0
OUTPUT_WIDTH_PIXELS = 1280
OUTPUT_HEIGHT_PIXELS = 960

# Эти доли относятся не ко всему изображению, а к допустимому пути центра.
# 0 означает, что левый/верхний край следа касается края эталона; 1 — что
# правый/нижний край касается противоположного края. Это параметры дизайна
# эксперимента, а не пороги качества алгоритма.
TRAVEL_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_reference() -> tuple[np.ndarray, float, Affine, str]:
    """Загружает RGB-эталон вместе с полной метрической геопривязкой."""

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


def run_grid() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Выполняет 25 привязок и 175 последующих метрических измерений."""

    (
        reference_rgb,
        reference_resolution_m_per_pixel,
        reference_transform,
        reference_crs,
    ) = load_reference()
    reference_height, reference_width = reference_rgb.shape[:2]
    footprint_width_px = FOOTPRINT_WIDTH_M / reference_resolution_m_per_pixel
    footprint_height_px = FOOTPRINT_HEIGHT_M / reference_resolution_m_per_pixel
    center_x_positions = calculate_valid_axis_centers(
        image_size_pixels=reference_width,
        footprint_size_pixels=footprint_width_px,
        travel_fractions=TRAVEL_FRACTIONS,
    )
    center_y_positions = calculate_valid_axis_centers(
        image_size_pixels=reference_height,
        footprint_size_pixels=footprint_height_px,
        travel_fractions=TRAVEL_FRACTIONS,
    )
    controls = build_poc_control_definitions(
        frame_width_pixels=OUTPUT_WIDTH_PIXELS,
        frame_height_pixels=OUTPUT_HEIGHT_PIXELS,
    )

    rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for grid_y_index, (travel_y_fraction, center_y_px) in enumerate(
        zip(TRAVEL_FRACTIONS, center_y_positions, strict=True)
    ):
        for grid_x_index, (travel_x_fraction, center_x_px) in enumerate(
            zip(TRAVEL_FRACTIONS, center_x_positions, strict=True)
        ):
            center_x_fraction = center_x_px / reference_width
            center_y_fraction = center_y_px / reference_height
            common_values: dict[str, Any] = {
                "grid_x_index": grid_x_index,
                "grid_y_index": grid_y_index,
                "travel_x_fraction": travel_x_fraction,
                "travel_y_fraction": travel_y_fraction,
                "center_reference_x_px": center_x_px,
                "center_reference_y_px": center_y_px,
                "center_x_fraction": center_x_fraction,
                "center_y_fraction": center_y_fraction,
                "footprint_width_m": FOOTPRINT_WIDTH_M,
                "footprint_height_m": FOOTPRINT_HEIGHT_M,
                "rotation_degrees": 0.0,
                "perspective_strength": 0.0,
            }
            spec = SyntheticFrameSpec(
                footprint_width_m=FOOTPRINT_WIDTH_M,
                footprint_height_m=FOOTPRINT_HEIGHT_M,
                output_width_pixels=OUTPUT_WIDTH_PIXELS,
                output_height_pixels=OUTPUT_HEIGHT_PIXELS,
                rotation_degrees=0.0,
                perspective_strength=0.0,
                center_x_fraction=center_x_fraction,
                center_y_fraction=center_y_fraction,
            )
            synthetic = generate_synthetic_frame(
                reference_rgb,
                reference_resolution_m_per_pixel=reference_resolution_m_per_pixel,
                spec=spec,
            )

            try:
                alignment = align_frame_to_reference(
                    reference_rgb,
                    synthetic.image_rgb,
                )
            except AlignmentFailure as error:
                rows.append(
                    {
                        **common_values,
                        "estimation_returned": False,
                        "failure_reason": str(error),
                    }
                )
                continue

            evaluation = evaluate_homography(
                alignment.homography_frame_to_reference,
                synthetic.homography_frame_to_reference,
                frame_width_pixels=OUTPUT_WIDTH_PIXELS,
                frame_height_pixels=OUTPUT_HEIGHT_PIXELS,
                reference_resolution_m_per_pixel=reference_resolution_m_per_pixel,
            )
            quality = analyze_alignment_quality(
                alignment,
                frame_width_pixels=OUTPUT_WIDTH_PIXELS,
                frame_height_pixels=OUTPUT_HEIGHT_PIXELS,
            )
            metric_evaluations = evaluate_metric_controls(
                controls,
                estimated_homography_frame_to_reference=(
                    alignment.homography_frame_to_reference
                ),
                true_homography_frame_to_reference=(
                    synthetic.homography_frame_to_reference
                ),
                reference_transform=reference_transform,
                reference_crs=reference_crs,
                reference_width_pixels=reference_width,
                reference_height_pixels=reference_height,
            )
            metric_result_rows = [
                metric_evaluation_to_dict(result) for result in metric_evaluations
            ]
            for metric_row in metric_result_rows:
                control_rows.append(
                    {
                        "grid_x_index": grid_x_index,
                        "grid_y_index": grid_y_index,
                        "travel_x_fraction": travel_x_fraction,
                        "travel_y_fraction": travel_y_fraction,
                        "center_reference_x_px": center_x_px,
                        "center_reference_y_px": center_y_px,
                        **metric_row,
                    }
                )

            segment_metric_rows = [
                row for row in metric_result_rows if row["kind"] == "segment"
            ]
            polygon_metric_rows = [
                row for row in metric_result_rows if row["kind"] == "polygon"
            ]
            rows.append(
                {
                    **common_values,
                    "estimation_returned": True,
                    "failure_reason": "",
                    "ratio_matches": alignment.ratio_match_count,
                    "ransac_inliers": alignment.inlier_count,
                    "inlier_ratio": alignment.inlier_ratio,
                    "inlier_spatial_coverage_fraction": (
                        alignment.inlier_spatial_coverage_fraction
                    ),
                    "grid_occupancy_fraction": quality.grid_occupancy_fraction,
                    "horizontal_inlier_span_fraction": (
                        quality.horizontal_span_fraction
                    ),
                    "vertical_inlier_span_fraction": quality.vertical_span_fraction,
                    "stability_trials_succeeded": (quality.stability_trials_succeeded),
                    "stability_p95_max_corner_shift_reference_px": (
                        quality.stability_p95_max_corner_shift_reference_px
                    ),
                    "mean_transfer_error_reference_px": evaluation.mean_error_pixels,
                    "max_transfer_error_reference_px": evaluation.max_error_pixels,
                    "mean_transfer_error_meters": evaluation.mean_error_meters,
                    "max_transfer_error_meters": evaluation.max_error_meters,
                    "max_control_vertex_error_meters": max(
                        row["max_vertex_position_error_meters"]
                        for row in metric_result_rows
                    ),
                    "max_segment_absolute_error_meters": max(
                        row["absolute_error"] for row in segment_metric_rows
                    ),
                    "max_polygon_absolute_error_square_meters": max(
                        row["absolute_error"] for row in polygon_metric_rows
                    ),
                    "max_control_relative_error_percent": max(
                        row["relative_error_percent"] for row in metric_result_rows
                    ),
                    "processing_time_seconds": alignment.processing_time_seconds,
                }
            )

    return rows, control_rows


def _scenario_location(row: dict[str, Any]) -> dict[str, Any]:
    """Извлекает координаты сценария для компактной JSON-сводки."""

    return {
        "grid_index_xy": [row["grid_x_index"], row["grid_y_index"]],
        "travel_fraction_xy": [
            row["travel_x_fraction"],
            row["travel_y_fraction"],
        ],
        "center_reference_px": [
            row["center_reference_x_px"],
            row["center_reference_y_px"],
        ],
    }


def _metric_control_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Сокращает подробную метрическую строку для JSON-сводки."""

    return {
        "grid_index_xy": [row["grid_x_index"], row["grid_y_index"]],
        "travel_fraction_xy": [
            row["travel_x_fraction"],
            row["travel_y_fraction"],
        ],
        "control_name": row["name"],
        "kind": row["kind"],
        "true_value": row["true_value"],
        "estimated_value": row["estimated_value"],
        "unit": row["unit"],
        "absolute_error": row["absolute_error"],
        "relative_error_percent": row["relative_error_percent"],
        "max_vertex_position_error_meters": row["max_vertex_position_error_meters"],
    }


def build_summary(
    rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Создаёт фактическую сводку без ретроспективного критерия успеха."""

    estimated_rows = [row for row in rows if row["estimation_returned"]]
    failed_rows = [row for row in rows if not row["estimation_returned"]]
    summary: dict[str, Any] = {
        "experiment_role": (
            "Исследовательская выборка; не использовать одновременно для "
            "настройки и окончательной проверки порогов качества."
        ),
        "controlled_variables": {
            "footprint_m": [FOOTPRINT_WIDTH_M, FOOTPRINT_HEIGHT_M],
            "output_shape_pixels": [OUTPUT_HEIGHT_PIXELS, OUTPUT_WIDTH_PIXELS],
            "rotation_degrees": 0.0,
            "perspective_strength": 0.0,
            "photometric_degradations": "отсутствуют",
        },
        "grid_shape": [len(TRAVEL_FRACTIONS), len(TRAVEL_FRACTIONS)],
        "travel_fractions": list(TRAVEL_FRACTIONS),
        "scenario_count": len(rows),
        "estimation_returned_count": len(estimated_rows),
        "explicit_failure_count": len(failed_rows),
        "failed_locations": [_scenario_location(row) for row in failed_rows],
        "metric_control_row_count": len(control_rows),
        "note": (
            "Факт возврата гомографии не равен прохождению. Численный порог "
            "годности в этом эксперименте намеренно отсутствует."
        ),
    }
    if not estimated_rows:
        return summary

    worst_error_row = max(
        estimated_rows,
        key=lambda row: row["max_transfer_error_reference_px"],
    )
    fewest_inliers_row = min(
        estimated_rows,
        key=lambda row: row["ransac_inliers"],
    )
    rows_with_stability = [
        row
        for row in estimated_rows
        if row["stability_p95_max_corner_shift_reference_px"] is not None
    ]
    summary.update(
        {
            "worst_returned_error": {
                **_scenario_location(worst_error_row),
                "mean_transfer_error_reference_px": worst_error_row[
                    "mean_transfer_error_reference_px"
                ],
                "max_transfer_error_reference_px": worst_error_row[
                    "max_transfer_error_reference_px"
                ],
                "ransac_inliers": worst_error_row["ransac_inliers"],
            },
            "fewest_inliers": {
                **_scenario_location(fewest_inliers_row),
                "ransac_inliers": fewest_inliers_row["ransac_inliers"],
                "inlier_spatial_coverage_fraction": fewest_inliers_row[
                    "inlier_spatial_coverage_fraction"
                ],
            },
            "minimum_grid_occupancy_fraction": min(
                row["grid_occupancy_fraction"] for row in estimated_rows
            ),
        }
    )
    if rows_with_stability:
        most_unstable_row = max(
            rows_with_stability,
            key=lambda row: row["stability_p95_max_corner_shift_reference_px"],
        )
        summary["largest_observed_instability"] = {
            **_scenario_location(most_unstable_row),
            "stability_p95_max_corner_shift_reference_px": most_unstable_row[
                "stability_p95_max_corner_shift_reference_px"
            ],
            "max_transfer_error_reference_px": most_unstable_row[
                "max_transfer_error_reference_px"
            ],
        }
    if control_rows:
        segment_rows = [row for row in control_rows if row["kind"] == "segment"]
        polygon_rows = [row for row in control_rows if row["kind"] == "polygon"]
        summary["metric_measurements"] = {
            "largest_vertex_position_error": _metric_control_summary(
                max(
                    control_rows,
                    key=lambda row: row["max_vertex_position_error_meters"],
                )
            ),
            "largest_absolute_segment_error": _metric_control_summary(
                max(segment_rows, key=lambda row: row["absolute_error"])
            ),
            "largest_relative_segment_error": _metric_control_summary(
                max(segment_rows, key=lambda row: row["relative_error_percent"])
            ),
            "largest_absolute_polygon_error": _metric_control_summary(
                max(polygon_rows, key=lambda row: row["absolute_error"])
            ),
            "largest_relative_polygon_error": _metric_control_summary(
                max(polygon_rows, key=lambda row: row["relative_error_percent"])
            ),
        }
    return summary


def save_table_and_summary(
    rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Сохраняет сценарии, измерения и сводку в каталог ``outputs``."""

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    field_names = sorted({key for row in rows for key in row})
    with OUTPUT_CSV_PATH.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)

    metric_field_names = sorted({key for row in control_rows for key in row})
    with OUTPUT_METRIC_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        # В нормальном прогоне здесь 175 строк. Условие оставляет функцию
        # корректной и при полном отказе всех привязок: создаётся пустой файл,
        # а не CSV с искусственно придуманной схемой.
        if metric_field_names:
            writer = csv.DictWriter(output_file, fieldnames=metric_field_names)
            writer.writeheader()
            writer.writerows(control_rows)

    with OUTPUT_SUMMARY_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def _metric_matrix(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    """Перекладывает плоские строки в матрицу 5 × 5 для тепловой карты."""

    side_length = len(TRAVEL_FRACTIONS)
    matrix = np.full((side_length, side_length), np.nan, dtype=np.float64)
    for row in rows:
        value = row.get(key)
        if value is not None:
            matrix[row["grid_y_index"], row["grid_x_index"]] = float(value)
    return matrix


def _draw_heatmap(
    axis: Axes,
    matrix: np.ndarray,
    *,
    title: str,
    number_format: str,
    color_map: str,
) -> None:
    """Рисует одну подписанную карту метрики без порога good/bad."""

    image = axis.imshow(matrix, cmap=color_map, origin="upper")
    axis.set_title(title)
    axis.set_xlabel("Доля допустимого хода центра по X")
    axis.set_ylabel("Доля допустимого хода центра по Y")
    axis.set_xticks(range(len(TRAVEL_FRACTIONS)), TRAVEL_FRACTIONS)
    axis.set_yticks(range(len(TRAVEL_FRACTIONS)), TRAVEL_FRACTIONS)
    axis.figure.colorbar(image, ax=axis, shrink=0.82)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            label = "отказ" if np.isnan(value) else format(value, number_format)
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5},
            )


def save_plot(rows: list[dict[str, Any]]) -> None:
    """Сохраняет четыре карты, позволяющие увидеть зависимость от местности."""

    figure, axes = plt.subplots(2, 2, figsize=(14, 12))
    plot_specs = (
        (
            "max_transfer_error_reference_px",
            "Максимальная ошибка, px эталона",
            ".3f",
            "magma",
        ),
        ("ransac_inliers", "Число inlier-пар RANSAC", ".0f", "viridis"),
        (
            "inlier_spatial_coverage_fraction",
            "Покрытие кадра выпуклой оболочкой",
            ".2f",
            "viridis",
        ),
        (
            "stability_p95_max_corner_shift_reference_px",
            "P95 сдвига худшего угла, px эталона",
            ".3f",
            "magma",
        ),
    )
    for axis, (key, title, number_format, color_map) in zip(
        axes.ravel(),
        plot_specs,
        strict=True,
    ):
        _draw_heatmap(
            axis,
            _metric_matrix(rows, key),
            title=title,
            number_format=number_format,
            color_map=color_map,
        )

    figure.suptitle(
        "Чистая пространственная сетка 5 × 5: меняется только участок эталона",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_PLOT_PATH, dpi=160)
    plt.close(figure)


def save_metric_plot(rows: list[dict[str, Any]]) -> None:
    """Показывает, как метрическая ошибка меняется по пространственной сетке.

    Каждая ячейка агрегирует семь контрольных фигур соответствующего кадра.
    Цвета здесь означают только численную величину: они намеренно не кодируют
    категории «хорошо» и «плохо», потому что допустимый порог ещё не задан.
    """

    figure, axes = plt.subplots(2, 2, figsize=(14, 12))
    plot_specs = (
        (
            "max_control_vertex_error_meters",
            "Максимальная ошибка вершины, м",
            ".5f",
            "magma",
        ),
        (
            "max_segment_absolute_error_meters",
            "Максимальная абсолютная ошибка длины, м",
            ".5f",
            "magma",
        ),
        (
            "max_polygon_absolute_error_square_meters",
            "Максимальная абсолютная ошибка площади, м²",
            ".4f",
            "magma",
        ),
        (
            "max_control_relative_error_percent",
            "Максимальная относительная ошибка фигуры, %",
            ".5f",
            "magma",
        ),
    )
    for axis, (key, title, number_format, color_map) in zip(
        axes.ravel(),
        plot_specs,
        strict=True,
    ):
        _draw_heatmap(
            axis,
            _metric_matrix(rows, key),
            title=title,
            number_format=number_format,
            color_map=color_map,
        )

    figure.suptitle(
        "Метрический контроль 5 × 5: семь фигур в каждом положении кадра",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_METRIC_PLOT_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    """Выполняет эксперимент и сохраняет воспроизводимые артефакты."""

    rows, control_rows = run_grid()
    summary = build_summary(rows, control_rows)
    save_table_and_summary(rows, control_rows, summary)
    save_plot(rows)
    save_metric_plot(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Подробные строки: {OUTPUT_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Подробные измерения: {OUTPUT_METRIC_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Сводка: {OUTPUT_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Тепловая карта: {OUTPUT_PLOT_PATH.relative_to(PROJECT_ROOT)}")
    print(
        "Метрическая тепловая карта: "
        f"{OUTPUT_METRIC_PLOT_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
