#!/usr/bin/env python3
"""Проверяет влияние только масштаба на чистую плоскую привязку.

Во всех сценариях центр участка, поворот, перспективный перекос, выходное
разрешение и пиксели исходного эталона остаются неизменными. Меняется только
физический размер видимого участка. При фиксированных 1280 × 960 пикселях это
эквивалентно изменению высоты полёта или поля зрения идеальной плоской камеры.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from aerial_mapper.alignment import AlignmentFailure, align_frame_to_reference
from aerial_mapper.evaluation import evaluate_homography
from aerial_mapper.synthetic import SyntheticFrameSpec, generate_synthetic_frame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = PROJECT_ROOT / "data/reference/cherkasy_2021_poc.tif"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs/scale_sweep.csv"
OUTPUT_SUMMARY_PATH = PROJECT_ROOT / "outputs/scale_sweep_summary.json"
OUTPUT_PLOT_PATH = PROJECT_ROOT / "outputs/scale_sweep.png"

BASE_FOOTPRINT_WIDTH_M = 120.0
BASE_FOOTPRINT_HEIGHT_M = 90.0
OUTPUT_WIDTH_PIXELS = 1280
OUTPUT_HEIGHT_PIXELS = 960

# Логарифмически-подобная шкала подробнее исследует область сильного
# увеличения, где предварительный опыт показал переход от отказа к нестабильной
# оценке. Максимальный участок 240 × 180 м остаётся внутри эталона 250 × 250 м.
SCALE_MULTIPLIERS = (
    0.08,
    0.10,
    0.125,
    0.16,
    0.20,
    0.25,
    0.35,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
)


def load_reference() -> tuple[np.ndarray, float]:
    """Загружает локальный RGB-эталон и его разрешение в метрах на пиксель."""

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            "Эталон не найден. Выполните: uv run python scripts/download_reference.py"
        )

    with rasterio.open(REFERENCE_PATH) as dataset:
        resolution_x = float(dataset.res[0])
        resolution_y = float(dataset.res[1])
        if not np.isclose(resolution_x, resolution_y, atol=1e-9):
            raise ValueError("Эксперимент ожидает квадратные пиксели эталона")
        reference_rgb = np.moveaxis(dataset.read((1, 2, 3)), 0, -1)
    return reference_rgb, resolution_x


def run_sweep() -> list[dict[str, Any]]:
    """Выполняет масштабную шкалу без поворота и иных искажений."""

    reference_rgb, reference_resolution_m_per_pixel = load_reference()
    rows: list[dict[str, Any]] = []

    for scale_multiplier in SCALE_MULTIPLIERS:
        footprint_width_m = BASE_FOOTPRINT_WIDTH_M * scale_multiplier
        footprint_height_m = BASE_FOOTPRINT_HEIGHT_M * scale_multiplier
        frame_resolution_m_per_pixel = footprint_width_m / OUTPUT_WIDTH_PIXELS

        # Отношение показывает, во сколько раз линейная деталь эталона станет
        # крупнее или мельче в кадре. Например, 4 означает четырёхкратное
        # увеличение относительно эталонного растрового разрешения.
        frame_pixels_per_reference_pixel = (
            reference_resolution_m_per_pixel / frame_resolution_m_per_pixel
        )
        spec = SyntheticFrameSpec(
            footprint_width_m=footprint_width_m,
            footprint_height_m=footprint_height_m,
            output_width_pixels=OUTPUT_WIDTH_PIXELS,
            output_height_pixels=OUTPUT_HEIGHT_PIXELS,
            rotation_degrees=0.0,
            perspective_strength=0.0,
        )
        synthetic = generate_synthetic_frame(
            reference_rgb,
            reference_resolution_m_per_pixel=reference_resolution_m_per_pixel,
            spec=spec,
        )

        common_values: dict[str, Any] = {
            "scale_multiplier": scale_multiplier,
            "footprint_width_m": footprint_width_m,
            "footprint_height_m": footprint_height_m,
            "frame_resolution_m_per_pixel": frame_resolution_m_per_pixel,
            "frame_pixels_per_reference_pixel": frame_pixels_per_reference_pixel,
            "rotation_degrees": 0.0,
            "perspective_strength": 0.0,
        }

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
                "mean_transfer_error_reference_px": evaluation.mean_error_pixels,
                "max_transfer_error_reference_px": evaluation.max_error_pixels,
                "mean_transfer_error_meters": evaluation.mean_error_meters,
                "max_transfer_error_meters": evaluation.max_error_meters,
                "processing_time_seconds": alignment.processing_time_seconds,
            }
        )

    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Формирует фактическую сводку без ретроспективного порога успеха."""

    estimated_rows = [row for row in rows if row["estimation_returned"]]
    failed_rows = [row for row in rows if not row["estimation_returned"]]
    summary: dict[str, Any] = {
        "controlled_variables": {
            "rotation_degrees": 0.0,
            "perspective_strength": 0.0,
            "output_shape_pixels": [OUTPUT_HEIGHT_PIXELS, OUTPUT_WIDTH_PIXELS],
            "footprint_center": "центр эталона",
            "photometric_degradations": "отсутствуют",
        },
        "scenario_count": len(rows),
        "estimation_returned_count": len(estimated_rows),
        "explicit_failure_count": len(failed_rows),
        "failed_scale_multipliers": [row["scale_multiplier"] for row in failed_rows],
        "note": (
            "Факт возврата матрицы не означает достаточную точность. Порог "
            "прохождения намеренно не назначается после просмотра этих данных."
        ),
    }
    if estimated_rows:
        worst_row = max(
            estimated_rows,
            key=lambda row: row["max_transfer_error_reference_px"],
        )
        summary.update(
            {
                "worst_returned_estimate": {
                    "scale_multiplier": worst_row["scale_multiplier"],
                    "footprint_m": [
                        worst_row["footprint_width_m"],
                        worst_row["footprint_height_m"],
                    ],
                    "ratio_matches": worst_row["ratio_matches"],
                    "ransac_inliers": worst_row["ransac_inliers"],
                    "inlier_spatial_coverage_fraction": worst_row[
                        "inlier_spatial_coverage_fraction"
                    ],
                    "mean_transfer_error_reference_px": worst_row[
                        "mean_transfer_error_reference_px"
                    ],
                    "max_transfer_error_reference_px": worst_row[
                        "max_transfer_error_reference_px"
                    ],
                },
                "minimum_inlier_spatial_coverage_fraction": min(
                    row["inlier_spatial_coverage_fraction"] for row in estimated_rows
                ),
            }
        )
    return summary


def save_table_and_summary(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Сохраняет подробные результаты и сводку в игнорируемый outputs."""

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    field_names = sorted({key for row in rows for key in row})
    with OUTPUT_CSV_PATH.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)

    with OUTPUT_SUMMARY_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def save_plot(rows: list[dict[str, Any]]) -> None:
    """Строит три графика: ошибку, число пар и покрытие кадра."""

    estimated_rows = [row for row in rows if row["estimation_returned"]]
    failed_rows = [row for row in rows if not row["estimation_returned"]]
    multipliers = np.asarray(
        [row["scale_multiplier"] for row in estimated_rows],
        dtype=np.float64,
    )

    figure, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    error_axis, count_axis, coverage_axis = axes

    error_axis.plot(
        multipliers,
        [row["mean_transfer_error_reference_px"] for row in estimated_rows],
        marker="o",
        label="Средняя ошибка",
    )
    error_axis.plot(
        multipliers,
        [row["max_transfer_error_reference_px"] for row in estimated_rows],
        marker="o",
        label="Максимальная ошибка",
    )
    error_axis.set_ylabel("Ошибка на эталоне, px")
    error_axis.set_yscale("log")
    error_axis.grid(True, which="both", alpha=0.3)
    error_axis.legend()

    count_axis.plot(
        multipliers,
        [row["ratio_matches"] for row in estimated_rows],
        marker="o",
        label="Пары после ratio test",
    )
    count_axis.plot(
        multipliers,
        [row["ransac_inliers"] for row in estimated_rows],
        marker="o",
        label="Inlier-пары RANSAC",
    )
    count_axis.set_ylabel("Количество пар")
    count_axis.set_yscale("log")
    count_axis.grid(True, which="both", alpha=0.3)
    count_axis.legend()

    coverage_axis.plot(
        multipliers,
        [row["inlier_ratio"] for row in estimated_rows],
        marker="o",
        label="Доля inlier-пар",
    )
    coverage_axis.plot(
        multipliers,
        [row["inlier_spatial_coverage_fraction"] for row in estimated_rows],
        marker="o",
        label="Покрытие площади кадра",
    )
    coverage_axis.set_ylabel("Доля от 0 до 1")
    coverage_axis.set_xlabel("Множитель физического охвата относительно 120 × 90 м")
    coverage_axis.set_ylim(-0.02, 1.02)
    coverage_axis.grid(True, alpha=0.3)
    coverage_axis.legend()

    for failed_row in failed_rows:
        for axis in axes:
            axis.axvline(
                failed_row["scale_multiplier"],
                color="red",
                linestyle="--",
                alpha=0.55,
            )
    axes[0].set_title("Чистый однофакторный тест масштаба: rotation=0, perspective=0")
    axes[-1].set_xscale("log")
    figure.tight_layout()
    figure.savefig(OUTPUT_PLOT_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    """Выполняет серию и сохраняет все воспроизводимые артефакты."""

    rows = run_sweep()
    summary = build_summary(rows)
    save_table_and_summary(rows, summary)
    save_plot(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Подробные строки: {OUTPUT_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Сводка: {OUTPUT_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"График: {OUTPUT_PLOT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
