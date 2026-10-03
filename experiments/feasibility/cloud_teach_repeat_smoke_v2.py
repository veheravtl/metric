#!/usr/bin/env python3
"""Smoke test v2 текущей привязки на реальной паре полётов CLOUD.

Для каждого из заранее выбранных Repeat-кадров телеметрия определяет два
контроля: ближайший Teach-кадр и самый дальний Teach-кадр. Функция привязки
получает только пиксели двух изображений. GPS используется исключительно как
приближённая разметка и никогда не попадает в SIFT + RANSAC.

Эксперимент намеренно не вводит порог «принять/отказать» после просмотра этих
данных. Он отвечает на более ранний вопрос: отличается ли геометрическое
свидетельство правильных пар от заведомо неправильных на реальных кадрах.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from aerial_mapper.alignment import (
    AlignmentFailure,
    AlignmentResult,
    SiftRansacConfig,
    align_frame_to_reference,
)
from aerial_mapper.cloud_dataset import (
    TeachRepeatPair,
    load_positioned_images,
    select_teach_repeat_pairs,
)
from aerial_mapper.quality import analyze_alignment_quality
from aerial_mapper.visualization import draw_alignment_matches

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data/queries/cloud/utiasfield_trial1"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs/cloud_smoke_v2_pairs.csv"
OUTPUT_SUMMARY_PATH = PROJECT_ROOT / "outputs/cloud_smoke_v2_summary.json"
OUTPUT_METRICS_PATH = PROJECT_ROOT / "outputs/cloud_smoke_v2_metrics.png"
OUTPUT_MATCHES_PATH = PROJECT_ROOT / "outputs/cloud_smoke_v2_matches.png"
QUERY_COUNT = 11
PROJECTED_CRS = "EPSG:32617"


def load_rgb_image(path: Path) -> np.ndarray:
    """Читает PNG как RGB uint8 и явно отказывается при повреждении файла."""

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def image_path(flight_name: str, image_id: int) -> Path:
    """Строит имя кадра по принятому в CLOUD шестизначному номеру."""

    return DATA_DIRECTORY / flight_name / "images" / f"{image_id:06d}.png"


def evaluate_pair(
    pair: TeachRepeatPair,
    *,
    pair_kind: str,
    teach_image_id: int,
    telemetry_distance_meters: float,
    config: SiftRansacConfig,
) -> tuple[dict[str, Any], AlignmentResult | None, np.ndarray, np.ndarray]:
    """Запускает одну слепую проверку и возвращает измерения либо явный отказ."""

    repeat_rgb = load_rgb_image(image_path("repeat", pair.repeat_image_id))
    teach_rgb = load_rgb_image(image_path("teach", teach_image_id))
    started_at = perf_counter()
    try:
        alignment = align_frame_to_reference(
            teach_rgb,
            repeat_rgb,
            config=config,
        )
    except AlignmentFailure as error:
        return (
            {
                "repeat_image_id": pair.repeat_image_id,
                "pair_kind": pair_kind,
                "teach_image_id": teach_image_id,
                "telemetry_distance_meters": telemetry_distance_meters,
                "matrix_returned": False,
                "rejection_reason": str(error),
                "reference_keypoint_count": None,
                "repeat_keypoint_count": None,
                "candidate_match_count": None,
                "ratio_match_count": None,
                "inlier_count": None,
                "inlier_ratio": None,
                "inlier_spatial_coverage_fraction": None,
                "occupied_grid_cells": None,
                "grid_occupancy_fraction": None,
                "horizontal_span_fraction": None,
                "vertical_span_fraction": None,
                "inlier_reprojection_median_reference_px": None,
                "inlier_reprojection_p95_reference_px": None,
                "inlier_reprojection_max_reference_px": None,
                "stability_trials_succeeded": None,
                "stability_p95_corner_shift_reference_px": None,
                "processing_time_seconds": perf_counter() - started_at,
            },
            None,
            teach_rgb,
            repeat_rgb,
        )

    repeat_height, repeat_width = repeat_rgb.shape[:2]
    quality = analyze_alignment_quality(
        alignment,
        frame_width_pixels=repeat_width,
        frame_height_pixels=repeat_height,
        random_seed=config.random_seed,
    )
    return (
        {
            "repeat_image_id": pair.repeat_image_id,
            "pair_kind": pair_kind,
            "teach_image_id": teach_image_id,
            "telemetry_distance_meters": telemetry_distance_meters,
            "matrix_returned": True,
            "rejection_reason": "",
            "reference_keypoint_count": alignment.reference_keypoint_count,
            "repeat_keypoint_count": alignment.frame_keypoint_count,
            "candidate_match_count": alignment.candidate_match_count,
            "ratio_match_count": alignment.ratio_match_count,
            "inlier_count": alignment.inlier_count,
            "inlier_ratio": alignment.inlier_ratio,
            "inlier_spatial_coverage_fraction": (
                alignment.inlier_spatial_coverage_fraction
            ),
            "occupied_grid_cells": quality.occupied_grid_cells,
            "grid_occupancy_fraction": quality.grid_occupancy_fraction,
            "horizontal_span_fraction": quality.horizontal_span_fraction,
            "vertical_span_fraction": quality.vertical_span_fraction,
            "inlier_reprojection_median_reference_px": (
                quality.inlier_reprojection_median_reference_px
            ),
            "inlier_reprojection_p95_reference_px": (
                quality.inlier_reprojection_p95_reference_px
            ),
            "inlier_reprojection_max_reference_px": (
                quality.inlier_reprojection_max_reference_px
            ),
            "stability_trials_succeeded": quality.stability_trials_succeeded,
            "stability_p95_corner_shift_reference_px": (
                quality.stability_p95_max_corner_shift_reference_px
            ),
            "processing_time_seconds": alignment.processing_time_seconds,
        },
        alignment,
        teach_rgb,
        repeat_rgb,
    )


def _metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    """Возвращает только действительно измеренные значения выбранной метрики."""

    return [float(row[key]) for row in rows if row[key] is not None]


def _median_or_none(rows: list[dict[str, Any]], key: str) -> float | None:
    """Считает медиану без подмены отсутствующего измерения нулём."""

    values = _metric_values(rows, key)
    return float(np.median(values)) if values else None


def build_summary(
    rows: list[dict[str, Any]],
    pairs: list[TeachRepeatPair],
    config: SiftRansacConfig,
) -> dict[str, Any]:
    """Собирает описательную сводку без ретроспективного порога качества."""

    positive_rows = [row for row in rows if row["pair_kind"] == "positive_nearest"]
    negative_rows = [row for row in rows if row["pair_kind"] == "negative_farthest"]
    rows_by_key = {
        (int(row["repeat_image_id"]), str(row["pair_kind"])): row for row in rows
    }

    positive_more_inliers = 0
    positive_more_coverage = 0
    both_matrices_returned = 0
    for pair in pairs:
        positive = rows_by_key[(pair.repeat_image_id, "positive_nearest")]
        negative = rows_by_key[(pair.repeat_image_id, "negative_farthest")]
        if positive["matrix_returned"] and negative["matrix_returned"]:
            both_matrices_returned += 1
        positive_inliers = float(positive["inlier_count"] or 0)
        negative_inliers = float(negative["inlier_count"] or 0)
        positive_coverage = float(positive["inlier_spatial_coverage_fraction"] or 0.0)
        negative_coverage = float(negative["inlier_spatial_coverage_fraction"] or 0.0)
        positive_more_inliers += positive_inliers > negative_inliers
        positive_more_coverage += positive_coverage > negative_coverage

    def group_summary(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        returned_count = sum(bool(row["matrix_returned"]) for row in group_rows)
        return {
            "pair_count": len(group_rows),
            "matrix_returned_count": returned_count,
            "matrix_returned_fraction": returned_count / len(group_rows),
            "median_telemetry_distance_meters": _median_or_none(
                group_rows, "telemetry_distance_meters"
            ),
            "median_inlier_count_when_returned": _median_or_none(
                group_rows, "inlier_count"
            ),
            "median_inlier_ratio_when_returned": _median_or_none(
                group_rows, "inlier_ratio"
            ),
            "median_coverage_when_returned": _median_or_none(
                group_rows, "inlier_spatial_coverage_fraction"
            ),
            "median_p95_reprojection_error_reference_px_when_returned": (
                _median_or_none(group_rows, "inlier_reprojection_p95_reference_px")
            ),
            "median_processing_time_seconds": _median_or_none(
                group_rows, "processing_time_seconds"
            ),
        }

    return {
        "experiment": "CLOUD Teach/Repeat smoke test v2",
        "dataset": "UTIAS Field trial 1",
        "query_selection": (
            "11 равномерно разнесённых Repeat-кадров; ближайший Teach по GPS "
            "как положительная пара; самый дальний Teach как лёгкий "
            "отрицательный контроль"
        ),
        "algorithm_input": "Только два изображения; телеметрия скрыта от алгоритма",
        "algorithm_config": asdict(config),
        "positive_nearest": group_summary(positive_rows),
        "negative_farthest": group_summary(negative_rows),
        "paired_comparison": {
            "query_count": len(pairs),
            "both_matrices_returned_count": both_matrices_returned,
            "positive_has_more_inliers_count": positive_more_inliers,
            "positive_has_more_coverage_count": positive_more_coverage,
        },
        "interpretation_limits": [
            (
                "Одна пара полётов — один коррелированный опыт, а не 11 "
                "независимых доказательств."
            ),
            (
                "GPS даёт приблизительную близость камер, но не точную "
                "попиксельную истину."
            ),
            (
                "Самый дальний Teach-кадр является лёгким, а не трудным "
                "отрицательным примером."
            ),
            (
                "Возврат матрицы означает техническую оценку, а не решение "
                "о допустимости измерений."
            ),
            (
                "Пороги принятия нельзя выбирать и оценивать на этой же "
                "исследовательской серии."
            ),
        ],
    }


def save_metrics_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Показывает разделимость пар по числу inlier и покрытию кадра."""

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    styles = {
        "positive_nearest": ("Ближайшая Teach-пара", "#178f49", "o"),
        "negative_farthest": ("Дальний отрицательный контроль", "#c23b3b", "x"),
    }
    for pair_kind, (label, color, marker) in styles.items():
        group = [row for row in rows if row["pair_kind"] == pair_kind]
        x_values = np.arange(len(group))
        inliers = [float(row["inlier_count"] or 0.0) for row in group]
        coverage = [
            float(row["inlier_spatial_coverage_fraction"] or 0.0) for row in group
        ]
        axes[0].plot(x_values, inliers, marker=marker, color=color, label=label)
        axes[1].plot(x_values, coverage, marker=marker, color=color, label=label)

    axes[0].set_ylabel("Геометрически согласованные пары, шт.")
    axes[1].set_ylabel("Покрытие кадра согласованными точками")
    axes[1].set_xlabel("Номер разнесённого Repeat-запроса вдоль маршрута")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(
        "CLOUD smoke v2: ноль также означает технический отказ до оценки матрицы"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _rejection_visual(
    teach_rgb: np.ndarray,
    repeat_rgb: np.ndarray,
    reason: str,
) -> np.ndarray:
    """Создаёт понятную панель для пары, отклонённой до оценки матрицы."""

    height = 420
    images: list[np.ndarray] = []
    for image in (teach_rgb, repeat_rgb):
        width = round(image.shape[1] * height / image.shape[0])
        images.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    canvas = np.hstack(images)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 72), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        "TECHNICAL REJECTION",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 210, 80),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        reason[:90],
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def save_match_report(
    visuals: dict[tuple[int, str], np.ndarray],
    selected_repeat_ids: list[int],
    output_path: Path,
) -> None:
    """Сохраняет начало, середину и конец маршрута без выбора лучших случаев."""

    figure, axes = plt.subplots(3, 2, figsize=(18, 13))
    kinds = ("positive_nearest", "negative_farthest")
    labels = ("Ближайшая Teach-пара", "Дальний отрицательный контроль")
    for row_index, repeat_id in enumerate(selected_repeat_ids):
        for column_index, (kind, label) in enumerate(zip(kinds, labels, strict=True)):
            axis = axes[row_index, column_index]
            axis.imshow(visuals[(repeat_id, kind)])
            axis.set_title(f"Repeat {repeat_id:06d}: {label}")
            axis.axis("off")
    figure.suptitle(
        "Зелёные линии согласованы одной гомографией; красные ею отвергнуты",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def main() -> None:
    """Выполняет зафиксированный опыт и сохраняет таблицу, сводку и рисунки."""

    if not DATA_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Нет распакованного CLOUD trial: {DATA_DIRECTORY}. "
            "Сначала запустите scripts/download_cloud_trial.py"
        )

    teach = load_positioned_images(
        DATA_DIRECTORY / "teach", projected_crs=PROJECTED_CRS
    )
    repeat = load_positioned_images(
        DATA_DIRECTORY / "repeat", projected_crs=PROJECTED_CRS
    )
    pairs = select_teach_repeat_pairs(teach, repeat, query_count=QUERY_COUNT)
    config = SiftRansacConfig()
    selected_repeat_ids = [
        pairs[0].repeat_image_id,
        pairs[len(pairs) // 2].repeat_image_id,
        pairs[-1].repeat_image_id,
    ]

    rows: list[dict[str, Any]] = []
    visuals: dict[tuple[int, str], np.ndarray] = {}
    for pair in pairs:
        controls = (
            (
                "positive_nearest",
                pair.positive_teach_image_id,
                pair.positive_distance_meters,
            ),
            (
                "negative_farthest",
                pair.negative_teach_image_id,
                pair.negative_distance_meters,
            ),
        )
        for pair_kind, teach_image_id, distance_meters in controls:
            row, alignment, teach_rgb, repeat_rgb = evaluate_pair(
                pair,
                pair_kind=pair_kind,
                teach_image_id=teach_image_id,
                telemetry_distance_meters=distance_meters,
                config=config,
            )
            rows.append(row)
            print(
                f"Repeat {pair.repeat_image_id:06d}, {pair_kind}: "
                f"matrix_returned={row['matrix_returned']}, "
                f"inliers={row['inlier_count']}, "
                f"coverage={row['inlier_spatial_coverage_fraction']}"
            )

            if pair.repeat_image_id in selected_repeat_ids:
                if alignment is None:
                    visual = _rejection_visual(
                        teach_rgb, repeat_rgb, str(row["rejection_reason"])
                    )
                else:
                    visual = draw_alignment_matches(
                        teach_rgb,
                        repeat_rgb,
                        alignment,
                        target_height=420,
                        reference_label=f"TEACH {teach_image_id:06d}",
                        frame_label=f"REPEAT {pair.repeat_image_id:06d}",
                    )
                visuals[(pair.repeat_image_id, pair_kind)] = visual

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = build_summary(rows, pairs, config)
    OUTPUT_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_metrics_plot(rows, OUTPUT_METRICS_PATH)
    save_match_report(visuals, selected_repeat_ids, OUTPUT_MATCHES_PATH)

    print(f"Таблица: {OUTPUT_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Сводка: {OUTPUT_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"График: {OUTPUT_METRICS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Соответствия: {OUTPUT_MATCHES_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
