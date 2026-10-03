#!/usr/bin/env python3
"""Сравнение покадрового и последовательного поиска места на CLOUD.

Оба варианта получают одну и ту же матрицу косинусного сходства DINOv2-S +
VLAD. Покадровый baseline ранжирует каждую строку независимо. Последовательный
вариант усредняет сходство вдоль коротких допустимых траекторий из пяти кадров.
GPS используется только после обоих ранжирований для вычисления метрик.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from aerial_mapper.cloud_dataset import (
    PositionedImages,
    load_positioned_images,
    select_spaced_image_indices,
)
from aerial_mapper.place_retrieval import (
    DinoV2SmallPatchExtractor,
    aggregate_vlad,
    fit_visual_vocabulary,
    score_similarity_sequences,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data/queries/cloud/utiasfield_trial1"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "data/manifests/dinov2_vits14.json"
TORCH_HOME = PROJECT_ROOT / "data/models/torch"
CACHE_ROOT_DIRECTORY = PROJECT_ROOT / "outputs/cache/cloud_dinov2_s_vlad"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs/cloud_sequence_retrieval_queries.csv"
OUTPUT_SUMMARY_PATH = PROJECT_ROOT / "outputs/cloud_sequence_retrieval_summary.json"
OUTPUT_HEATMAP_PATH = PROJECT_ROOT / "outputs/cloud_sequence_retrieval_heatmap.png"
OUTPUT_TRACK_PATH = PROJECT_ROOT / "outputs/cloud_sequence_retrieval_track.png"

PROJECTED_CRS = "EPSG:32617"
KEYFRAME_SPACING_METERS = 1.0
POSITIVE_RADIUS_METERS = 3.0
CATASTROPHIC_DISTANCE_METERS = 10.0
LARGE_DATABASE_JUMP = 5
WINDOW_LENGTH = 5
VELOCITY_RATIOS = (0.0, 0.5, 1.0, 1.5, 2.0)
DIRECTIONS = (-1, 1)
CLUSTER_COUNT = 8
MAXIMUM_VOCABULARY_DESCRIPTORS = 20_000
RANDOM_SEED = 0


def load_rgb_image(path: Path) -> np.ndarray:
    """Читает PNG как RGB uint8 и явно отказывается при повреждении."""

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def image_path(flight_name: str, image_id: int) -> Path:
    """Строит путь к кадру CLOUD по имени полёта и Image Id."""

    return DATA_DIRECTORY / flight_name / "images" / f"{image_id:06d}.png"


def sha256_file(path: Path) -> str:
    """Потоково вычисляет SHA-256 файла весов."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_weights(manifest: dict[str, Any]) -> None:
    """Проверяет размер и контрольную сумму загруженных весов DINOv2."""

    model = manifest["model"]
    weights_path = TORCH_HOME / "hub/checkpoints" / model["weights_filename"]
    if not weights_path.exists():
        return
    if weights_path.stat().st_size != int(model["weights_size_bytes"]):
        raise RuntimeError(f"Не совпадает размер весов DINOv2: {weights_path}")
    if sha256_file(weights_path) != str(model["weights_sha256"]):
        raise RuntimeError(f"Не совпадает SHA-256 весов DINOv2: {weights_path}")


def load_or_extract_patches(
    extractor: DinoV2SmallPatchExtractor,
    *,
    flight_name: str,
    image_id: int,
) -> tuple[np.ndarray, float, bool]:
    """Возвращает patch-признаки, время нового расчёта и признак cache hit."""

    cache_directory = CACHE_ROOT_DIRECTORY / extractor.cache_signature
    cache_path = cache_directory / f"{flight_name}_{image_id:06d}_r000.npy"
    if cache_path.exists():
        descriptors = np.load(cache_path, allow_pickle=False)
        if descriptors.ndim != 2 or not np.all(np.isfinite(descriptors)):
            raise RuntimeError(f"Некорректный кэш DINOv2: {cache_path}")
        return descriptors.astype(np.float32), 0.0, True

    image_rgb = load_rgb_image(image_path(flight_name, image_id))
    result = extractor.extract(image_rgb)
    cache_directory.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, result.patch_descriptors, allow_pickle=False)
    return result.patch_descriptors, result.processing_time_seconds, False


def selected_flight(
    positioned: PositionedImages,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает индексы, Image Id и координаты метровых keyframe."""

    indices = select_spaced_image_indices(
        positioned,
        minimum_distance_meters=KEYFRAME_SPACING_METERS,
    )
    return indices, positioned.image_ids[indices], positioned.xy_meters[indices]


def first_positive_rank(
    ranked_database_indices: np.ndarray,
    distances_meters: np.ndarray,
) -> int | None:
    """Возвращает единичный ранг первого кандидата не дальше 3 м."""

    positive = np.flatnonzero(
        distances_meters[ranked_database_indices] <= POSITIVE_RADIUS_METERS
    )
    return int(positive[0] + 1) if positive.size else None


def evaluate_variant(
    similarities: np.ndarray,
    distance_matrix_meters: np.ndarray,
    evaluable_mask: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[int | None]]:
    """Считает retrieval-метрики для одного способа ранжирования."""

    ranked = np.argsort(-similarities, axis=1, kind="stable")
    top1_indices = ranked[:, 0]
    row_indices = np.arange(ranked.shape[0])
    top1_distances = distance_matrix_meters[row_indices, top1_indices]
    ranks = [
        first_positive_rank(ranked[row], distance_matrix_meters[row])
        for row in range(ranked.shape[0])
    ]
    evaluable_ranks = np.asarray(
        [ranks[index] for index in np.flatnonzero(evaluable_mask)],
        dtype=np.float64,
    )
    evaluable_top1_distances = top1_distances[evaluable_mask]

    adjacent_evaluable = evaluable_mask[:-1] & evaluable_mask[1:]
    top1_jumps = np.abs(np.diff(top1_indices))
    evaluated_jumps = top1_jumps[adjacent_evaluable]
    metrics: dict[str, Any] = {
        "evaluable_query_count": int(np.count_nonzero(evaluable_mask)),
        "recall_at_1": float(np.mean(evaluable_ranks <= 1)),
        "recall_at_5": float(np.mean(evaluable_ranks <= 5)),
        "recall_at_10": float(np.mean(evaluable_ranks <= 10)),
        "median_first_positive_rank": float(np.median(evaluable_ranks)),
        "p95_first_positive_rank": float(np.percentile(evaluable_ranks, 95)),
        "median_top1_distance_meters": float(np.median(evaluable_top1_distances)),
        "p95_top1_distance_meters": float(np.percentile(evaluable_top1_distances, 95)),
        "top1_farther_than_10m_count": int(
            np.count_nonzero(evaluable_top1_distances > CATASTROPHIC_DISTANCE_METERS)
        ),
        "adjacent_evaluable_transition_count": int(evaluated_jumps.size),
        "top1_database_jump_over_5_count": int(
            np.count_nonzero(evaluated_jumps > LARGE_DATABASE_JUMP)
        ),
        "median_top1_database_jump": (
            float(np.median(evaluated_jumps)) if evaluated_jumps.size else None
        ),
        "p95_top1_database_jump": (
            float(np.percentile(evaluated_jumps, 95)) if evaluated_jumps.size else None
        ),
    }
    return metrics, ranked, top1_distances, ranks


def save_heatmap(
    frame_scores: np.ndarray,
    sequence_scores: np.ndarray,
    nearest_teach_indices: np.ndarray,
    repeat_ids: np.ndarray,
    teach_ids: np.ndarray,
) -> None:
    """Сохраняет покадровую и последовательную матрицы в одном масштабе."""

    minimum = float(min(np.min(frame_scores), np.min(sequence_scores)))
    maximum = float(max(np.max(frame_scores), np.max(sequence_scores)))
    figure, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    for axis, scores, title in (
        (axes[0], frame_scores, "Покадровое косинусное сходство"),
        (axes[1], sequence_scores, "Среднее по лучшей траектории из 5 кадров"),
    ):
        image = axis.imshow(
            scores,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=minimum,
            vmax=maximum,
        )
        axis.scatter(
            nearest_teach_indices,
            np.arange(repeat_ids.size),
            marker="x",
            s=18,
            linewidths=0.8,
            color="red",
            label="ближайший Teach по GPS",
        )
        axis.set_title(title)
        axis.set_ylabel("Repeat-keyframe")
        axis.legend(loc="upper right")
        figure.colorbar(image, ax=axis, label="сходство")
    repeat_ticks = np.linspace(0, repeat_ids.size - 1, 7, dtype=int)
    axes[0].set_yticks(repeat_ticks, [f"{repeat_ids[i]:06d}" for i in repeat_ticks])
    axes[1].set_yticks(repeat_ticks, [f"{repeat_ids[i]:06d}" for i in repeat_ticks])
    teach_ticks = np.linspace(0, teach_ids.size - 1, 7, dtype=int)
    axes[1].set_xticks(teach_ticks, [f"{teach_ids[i]:06d}" for i in teach_ticks])
    axes[1].set_xlabel("Teach-keyframe в порядке полёта")
    figure.suptitle("CLOUD: влияние пятикадровой последовательности")
    figure.tight_layout()
    figure.savefig(OUTPUT_HEATMAP_PATH, dpi=160)
    plt.close(figure)


def save_track_plot(
    frame_top1: np.ndarray,
    sequence_top1: np.ndarray,
    nearest_teach_indices: np.ndarray,
    repeat_ids: np.ndarray,
) -> None:
    """Показывает скачки предсказанного положения вдоль Repeat-маршрута."""

    x_axis = np.arange(repeat_ids.size)
    figure, axis = plt.subplots(figsize=(15, 5))
    axis.plot(
        x_axis, nearest_teach_indices, color="black", linewidth=2, label="GPS-контроль"
    )
    axis.plot(x_axis, frame_top1, color="#e76f51", alpha=0.8, label="покадровый Top-1")
    axis.plot(
        x_axis,
        sequence_top1,
        color="#2a9d8f",
        alpha=0.9,
        label="последовательный Top-1",
    )
    tick_positions = np.linspace(0, repeat_ids.size - 1, 8, dtype=int)
    axis.set_xticks(
        tick_positions,
        [f"{repeat_ids[index]:06d}" for index in tick_positions],
    )
    axis.set_xlabel("Repeat Image Id в порядке полёта")
    axis.set_ylabel("индекс Teach-keyframe")
    axis.set_title("Красная и зелёная линии не получают GPS при поиске")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_TRACK_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    """Выполняет зафиксированное сравнение и сохраняет сырые результаты."""

    if not DATA_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Нет CLOUD trial: {DATA_DIRECTORY}. "
            "Сначала запустите scripts/download_cloud_trial.py"
        )
    manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    verify_model_weights(manifest)
    extractor = DinoV2SmallPatchExtractor(
        repository=manifest["source"]["torch_hub_repository"],
        revision=manifest["source"]["repository_revision"],
        model_name=manifest["model"]["torch_hub_name"],
        torch_home=TORCH_HOME,
    )
    verify_model_weights(manifest)

    teach = load_positioned_images(
        DATA_DIRECTORY / "teach", projected_crs=PROJECTED_CRS
    )
    repeat = load_positioned_images(
        DATA_DIRECTORY / "repeat", projected_crs=PROJECTED_CRS
    )
    _, teach_ids, teach_xy = selected_flight(teach)
    _, repeat_ids, repeat_xy = selected_flight(repeat)

    extraction_times: list[float] = []
    cache_hits = 0
    patch_sets: dict[str, list[np.ndarray]] = {"teach": [], "repeat": []}
    for flight_name, image_ids in (("teach", teach_ids), ("repeat", repeat_ids)):
        for item_number, image_id in enumerate(image_ids, start=1):
            patches, elapsed, from_cache = load_or_extract_patches(
                extractor,
                flight_name=flight_name,
                image_id=int(image_id),
            )
            patch_sets[flight_name].append(patches)
            extraction_times.append(elapsed)
            cache_hits += int(from_cache)
            if item_number % 25 == 0 or item_number == image_ids.size:
                print(f"{flight_name}: {item_number}/{image_ids.size}")

    vocabulary_started_at = perf_counter()
    vocabulary = fit_visual_vocabulary(
        patch_sets["teach"],
        cluster_count=CLUSTER_COUNT,
        maximum_training_descriptors=MAXIMUM_VOCABULARY_DESCRIPTORS,
        random_seed=RANDOM_SEED,
    )
    vocabulary_seconds = perf_counter() - vocabulary_started_at
    teach_descriptors = np.stack(
        [aggregate_vlad(patches, vocabulary) for patches in patch_sets["teach"]]
    )
    repeat_descriptors = np.stack(
        [aggregate_vlad(patches, vocabulary) for patches in patch_sets["repeat"]]
    )
    frame_similarities = repeat_descriptors @ teach_descriptors.T

    sequence_started_at = perf_counter()
    sequence_result = score_similarity_sequences(
        frame_similarities,
        window_length=WINDOW_LENGTH,
        velocity_ratios=VELOCITY_RATIOS,
        directions=DIRECTIONS,
    )
    sequence_seconds = perf_counter() - sequence_started_at
    query_indices = sequence_result.query_center_indices
    eligible_repeat_ids = repeat_ids[query_indices]
    eligible_repeat_xy = repeat_xy[query_indices]
    eligible_frame_scores = frame_similarities[query_indices]
    distance_matrix = np.linalg.norm(
        eligible_repeat_xy[:, None, :] - teach_xy[None, :, :],
        axis=2,
    )
    evaluable_mask = np.any(distance_matrix <= POSITIVE_RADIUS_METERS, axis=1)
    nearest_teach_indices = np.argmin(distance_matrix, axis=1)

    frame_metrics, frame_ranked, frame_top1_distances, frame_ranks = evaluate_variant(
        eligible_frame_scores,
        distance_matrix,
        evaluable_mask,
    )
    sequence_metrics, sequence_ranked, sequence_top1_distances, sequence_ranks = (
        evaluate_variant(
            sequence_result.similarities,
            distance_matrix,
            evaluable_mask,
        )
    )
    frame_top1 = frame_ranked[:, 0]
    sequence_top1 = sequence_ranked[:, 0]

    rows: list[dict[str, Any]] = []
    for row_index, repeat_id in enumerate(eligible_repeat_ids):
        for variant, scores, ranked, distances, ranks in (
            (
                "single_frame",
                eligible_frame_scores,
                frame_ranked,
                frame_top1_distances,
                frame_ranks,
            ),
            (
                "sequence_5",
                sequence_result.similarities,
                sequence_ranked,
                sequence_top1_distances,
                sequence_ranks,
            ),
        ):
            top1_index = int(ranked[row_index, 0])
            is_sequence = variant == "sequence_5"
            rows.append(
                {
                    "repeat_image_id": int(repeat_id),
                    "variant": variant,
                    "positive_exists_within_3m": bool(evaluable_mask[row_index]),
                    "nearest_teach_image_id": int(
                        teach_ids[nearest_teach_indices[row_index]]
                    ),
                    "nearest_teach_distance_meters": float(
                        distance_matrix[row_index, nearest_teach_indices[row_index]]
                    ),
                    "top1_teach_image_id": int(teach_ids[top1_index]),
                    "top1_distance_meters": float(distances[row_index]),
                    "top1_score": float(scores[row_index, top1_index]),
                    "first_positive_rank": ranks[row_index],
                    "positive_in_top5": bool(
                        np.any(
                            distance_matrix[row_index, ranked[row_index, :5]]
                            <= POSITIVE_RADIUS_METERS
                        )
                    ),
                    "positive_in_top10": bool(
                        np.any(
                            distance_matrix[row_index, ranked[row_index, :10]]
                            <= POSITIVE_RADIUS_METERS
                        )
                    ),
                    "best_direction_for_top1": (
                        int(sequence_result.best_directions[row_index, top1_index])
                        if is_sequence
                        else None
                    ),
                    "best_velocity_for_top1": (
                        float(
                            sequence_result.best_velocity_ratios[row_index, top1_index]
                        )
                        if is_sequence
                        else None
                    ),
                }
            )

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    delta_recall_at_1 = sequence_metrics["recall_at_1"] - frame_metrics["recall_at_1"]
    delta_recall_at_5 = sequence_metrics["recall_at_5"] - frame_metrics["recall_at_5"]
    criterion_met = bool(
        delta_recall_at_1 >= 0.05
        and sequence_metrics["recall_at_5"] >= frame_metrics["recall_at_5"]
        and sequence_metrics["top1_farther_than_10m_count"]
        <= frame_metrics["top1_farther_than_10m_count"]
    )
    noncached_times = [value for value in extraction_times if value > 0.0]
    sequence_top1_directions = [
        int(sequence_result.best_directions[row, database_index])
        for row, database_index in enumerate(sequence_top1)
    ]
    sequence_top1_velocities = [
        float(sequence_result.best_velocity_ratios[row, database_index])
        for row, database_index in enumerate(sequence_top1)
    ]
    summary = {
        "experiment": "cloud_sequence_retrieval_smoke",
        "status": "exploratory_not_a_validation",
        "pre_registered_plan": "docs/cloud-sequence-retrieval-plan.md",
        "configuration": {
            "projected_crs": PROJECTED_CRS,
            "keyframe_spacing_meters": KEYFRAME_SPACING_METERS,
            "positive_radius_meters": POSITIVE_RADIUS_METERS,
            "catastrophic_distance_meters": CATASTROPHIC_DISTANCE_METERS,
            "large_database_jump": LARGE_DATABASE_JUMP,
            "window_length": WINDOW_LENGTH,
            "velocity_ratios": list(VELOCITY_RATIOS),
            "directions": list(DIRECTIONS),
            "vlad_cluster_count": CLUSTER_COUNT,
            "vlad_dimensions": CLUSTER_COUNT
            * int(manifest["model"]["patch_descriptor_dimensions"]),
            "random_seed": RANDOM_SEED,
        },
        "data": {
            "teach_keyframe_count": int(teach_ids.size),
            "repeat_keyframe_count": int(repeat_ids.size),
            "sequence_center_query_count": int(eligible_repeat_ids.size),
            "evaluable_query_count": int(np.count_nonzero(evaluable_mask)),
            "queries_without_positive_within_3m": [
                int(image_id) for image_id in eligible_repeat_ids[~evaluable_mask]
            ],
        },
        "results": {
            "single_frame": frame_metrics,
            "sequence_5": sequence_metrics,
            "difference_sequence_minus_single": {
                "recall_at_1": delta_recall_at_1,
                "recall_at_5": delta_recall_at_5,
                "p95_top1_distance_meters": (
                    sequence_metrics["p95_top1_distance_meters"]
                    - frame_metrics["p95_top1_distance_meters"]
                ),
                "top1_farther_than_10m_count": (
                    sequence_metrics["top1_farther_than_10m_count"]
                    - frame_metrics["top1_farther_than_10m_count"]
                ),
                "top1_database_jump_over_5_count": (
                    sequence_metrics["top1_database_jump_over_5_count"]
                    - frame_metrics["top1_database_jump_over_5_count"]
                ),
            },
            "pre_registered_utility_criterion_met": criterion_met,
            "sequence_top1_direction_counts": dict(
                sorted(Counter(sequence_top1_directions).items())
            ),
            "sequence_top1_velocity_counts": dict(
                sorted(Counter(sequence_top1_velocities).items())
            ),
        },
        "timing": {
            "noncached_extraction_count": len(noncached_times),
            "cache_hit_count": cache_hits,
            "median_dinov2_extraction_seconds": (
                float(np.median(noncached_times)) if noncached_times else None
            ),
            "total_dinov2_extraction_seconds": float(sum(noncached_times)),
            "visual_vocabulary_fit_seconds": vocabulary_seconds,
            "sequence_scoring_seconds": sequence_seconds,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "device": "cpu",
        },
        "limitations": [
            "Одна пара коррелированных CLOUD-полётов.",
            "Радиус 3 м измеряет GPS-близость, а не точное видимое перекрытие.",
            "Центрированное окно использует будущие кадры и является офлайн-методом.",
            "Параметры нельзя повторно подгонять на этих запросах "
            "как на независимом тесте.",
        ],
    }
    OUTPUT_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_heatmap(
        eligible_frame_scores,
        sequence_result.similarities,
        nearest_teach_indices,
        eligible_repeat_ids,
        teach_ids,
    )
    save_track_plot(
        frame_top1,
        sequence_top1,
        nearest_teach_indices,
        eligible_repeat_ids,
    )

    print(
        f"single: R@1={frame_metrics['recall_at_1']:.3f}, "
        f"R@5={frame_metrics['recall_at_5']:.3f}, "
        f">10m={frame_metrics['top1_farther_than_10m_count']}"
    )
    print(
        f"sequence: R@1={sequence_metrics['recall_at_1']:.3f}, "
        f"R@5={sequence_metrics['recall_at_5']:.3f}, "
        f">10m={sequence_metrics['top1_farther_than_10m_count']}"
    )
    print(f"Заранее заданный критерий выполнен: {criterion_met}")
    print(f"Таблица: {OUTPUT_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Сводка: {OUTPUT_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Матрицы: {OUTPUT_HEATMAP_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Траектория: {OUTPUT_TRACK_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
