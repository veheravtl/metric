#!/usr/bin/env python3
"""Запускает воспроизводимую серию чистых плоских привязок P1.

Эксперимент меняет только поворот и условную силу перспективного перекоса.
Помехи изображения ещё не добавляются. Для каждого кадра алгоритм получает
только два изображения, после чего отдельный оценщик открывает истинную
гомографию и вычисляет ошибку на регулярной сетке контрольных точек.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from aerial_mapper.alignment import AlignmentFailure, align_frame_to_reference
from aerial_mapper.evaluation import evaluate_homography
from aerial_mapper.synthetic import SyntheticFrameSpec, generate_synthetic_frame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = PROJECT_ROOT / "data/reference/cherkasy_2021_poc.tif"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs/clean_alignment_grid.csv"
OUTPUT_SUMMARY_PATH = PROJECT_ROOT / "outputs/clean_alignment_grid_summary.json"

ROTATIONS_DEGREES = (-35.0, -20.0, 0.0, 20.0, 35.0)
PERSPECTIVE_STRENGTHS = (0.0, 0.4, 0.8)


def load_reference() -> tuple[np.ndarray, float]:
    """Загружает RGB-эталон и проверяет квадратность его пикселей."""

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


def run_grid() -> list[dict[str, Any]]:
    """Выполняет все 15 сценариев и возвращает плоские строки для CSV."""

    reference_rgb, resolution_m_per_pixel = load_reference()
    rows: list[dict[str, Any]] = []

    for rotation_degrees in ROTATIONS_DEGREES:
        for perspective_strength in PERSPECTIVE_STRENGTHS:
            spec = SyntheticFrameSpec(
                rotation_degrees=rotation_degrees,
                perspective_strength=perspective_strength,
            )
            synthetic = generate_synthetic_frame(
                reference_rgb,
                reference_resolution_m_per_pixel=resolution_m_per_pixel,
                spec=spec,
            )

            try:
                alignment = align_frame_to_reference(
                    reference_rgb,
                    synthetic.image_rgb,
                )
            except AlignmentFailure as error:
                # Отказ является измеряемым результатом эксперимента, а не
                # причиной потерять остальные сценарии серии.
                rows.append(
                    {
                        "rotation_degrees": rotation_degrees,
                        "perspective_strength": perspective_strength,
                        "success": False,
                        "failure_reason": str(error),
                    }
                )
                continue

            evaluation = evaluate_homography(
                alignment.homography_frame_to_reference,
                synthetic.homography_frame_to_reference,
                frame_width_pixels=spec.output_width_pixels,
                frame_height_pixels=spec.output_height_pixels,
                reference_resolution_m_per_pixel=resolution_m_per_pixel,
            )
            rows.append(
                {
                    "rotation_degrees": rotation_degrees,
                    "perspective_strength": perspective_strength,
                    "success": True,
                    "failure_reason": "",
                    "ratio_matches": alignment.ratio_match_count,
                    "ransac_inliers": alignment.inlier_count,
                    "inlier_ratio": alignment.inlier_ratio,
                    "inlier_spatial_coverage_fraction": (
                        alignment.inlier_spatial_coverage_fraction
                    ),
                    "mean_transfer_error_reference_px": (evaluation.mean_error_pixels),
                    "max_transfer_error_reference_px": evaluation.max_error_pixels,
                    "mean_transfer_error_meters": evaluation.mean_error_meters,
                    "max_transfer_error_meters": evaluation.max_error_meters,
                    "processing_time_seconds": alignment.processing_time_seconds,
                }
            )

    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Сводит серию к нескольким проверяемым показателям."""

    successful_rows = [row for row in rows if row["success"]]
    summary: dict[str, Any] = {
        "scenario_count": len(rows),
        "success_count": len(successful_rows),
        "failure_count": len(rows) - len(successful_rows),
        "rotations_degrees": list(ROTATIONS_DEGREES),
        "perspective_strengths": list(PERSPECTIVE_STRENGTHS),
        "limitations": (
            "Все кадры пока центрированы, созданы из того же эталона и не "
            "содержат фотометрических помех."
        ),
    }
    if not successful_rows:
        # Полный отказ должен быть корректным результатом серии, а не вторичной
        # ошибкой max()/min(), скрывающей исходные причины из подробного CSV.
        return summary

    summary.update(
        {
            "worst_mean_transfer_error_reference_px": max(
                row["mean_transfer_error_reference_px"] for row in successful_rows
            ),
            "worst_max_transfer_error_reference_px": max(
                row["max_transfer_error_reference_px"] for row in successful_rows
            ),
            "minimum_inlier_ratio": min(row["inlier_ratio"] for row in successful_rows),
            "minimum_inlier_spatial_coverage_fraction": min(
                row["inlier_spatial_coverage_fraction"] for row in successful_rows
            ),
        }
    )
    return summary


def save_results(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Сохраняет подробные строки и краткую сводку в игнорируемый outputs."""

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    field_names = sorted({key for row in rows for key in row})
    with OUTPUT_CSV_PATH.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)

    with OUTPUT_SUMMARY_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def main() -> None:
    """Запускает эксперимент и печатает пути к воспроизводимым артефактам."""

    rows = run_grid()
    summary = build_summary(rows)
    save_results(rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Подробные строки: {OUTPUT_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Сводка: {OUTPUT_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
