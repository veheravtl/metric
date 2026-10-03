#!/usr/bin/env python3
"""Поиск места Repeat-кадра в разреженной Teach-базе CLOUD.

Опыт проверяет двухступенчатую гипотезу:

1. DINOv2-S + VLAD должен поместить географически близкий Teach-кадр в
   короткий список кандидатов, не видя GPS-координаты запроса.
2. Уже существующая геометрическая проверка SIFT + RANSAC должна отличить
   пригодного кандидата внутри этого списка от визуально похожей ошибки.

GPS используется только после ранжирования как приблизительная контрольная
истина. Порог в три метра не является готовым эксплуатационным допуском: он
учитывает метровый шаг разреживания и наблюдавшуюся ранее погрешность
сопоставления траекторий двух полётов.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from aerial_mapper.alignment import (
    AlignmentFailure,
    SiftRansacConfig,
    align_frame_to_reference,
)
from aerial_mapper.cloud_dataset import (
    PositionedImages,
    load_positioned_images,
    select_spaced_image_indices,
    select_teach_repeat_pairs,
)
from aerial_mapper.place_retrieval import (
    DinoV2SmallPatchExtractor,
    aggregate_vlad,
    fit_visual_vocabulary,
)
from aerial_mapper.quality import analyze_alignment_quality

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data/queries/cloud/utiasfield_trial1"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "data/manifests/dinov2_vits14.json"
TORCH_HOME = PROJECT_ROOT / "data/models/torch"
CACHE_ROOT_DIRECTORY = PROJECT_ROOT / "outputs/cache/cloud_dinov2_s_vlad"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs/cloud_dinov2_vlad_candidates.csv"
OUTPUT_SUMMARY_PATH = PROJECT_ROOT / "outputs/cloud_dinov2_vlad_summary.json"
OUTPUT_HEATMAP_PATH = PROJECT_ROOT / "outputs/cloud_dinov2_vlad_heatmap.png"
OUTPUT_TOP1_PATH = PROJECT_ROOT / "outputs/cloud_dinov2_vlad_top1.png"

PROJECTED_CRS = "EPSG:32617"
QUERY_COUNT = 11
KEYFRAME_SPACING_METERS = 1.0
POSITIVE_RADIUS_METERS = 3.0
TOP_K = 5
CLUSTER_COUNT = 8
MAXIMUM_VOCABULARY_DESCRIPTORS = 20_000
RANDOM_SEED = 0
QUERY_ROTATIONS_DEGREES = (0, 90, 180, 270)


def load_rgb_image(path: Path) -> np.ndarray:
    """Читает изображение как RGB uint8 и явно сообщает о повреждении."""

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def image_path(flight_name: str, image_id: int) -> Path:
    """Строит путь к кадру CLOUD по полёту и шестизначному Image Id."""

    return DATA_DIRECTORY / flight_name / "images" / f"{image_id:06d}.png"


def sha256_file(path: Path) -> str:
    """Потоково вычисляет SHA-256, не загружая весь файл в память."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_weights(manifest: dict[str, Any]) -> None:
    """Проверяет размер и контрольную сумму уже загруженных весов модели."""

    model = manifest["model"]
    weights_path = TORCH_HOME / "hub/checkpoints" / model["weights_filename"]
    if not weights_path.exists():
        # Первый вызов torch.hub загрузит файл. После него функция вызывается
        # повторно, и эксперимент продолжится только с проверенными весами.
        return
    if weights_path.stat().st_size != int(model["weights_size_bytes"]):
        raise RuntimeError(f"Не совпадает размер весов DINOv2: {weights_path}")
    if sha256_file(weights_path) != str(model["weights_sha256"]):
        raise RuntimeError(f"Не совпадает SHA-256 весов DINOv2: {weights_path}")


def position_index_by_image_id(positioned: PositionedImages) -> dict[int, int]:
    """Создаёт однозначное соответствие Image Id индексу строки телеметрии."""

    result = {
        int(image_id): index for index, image_id in enumerate(positioned.image_ids)
    }
    if len(result) != positioned.image_ids.size:
        raise ValueError("В телеметрии встретились повторяющиеся Image Id")
    return result


def load_or_extract_patches(
    extractor: DinoV2SmallPatchExtractor,
    *,
    flight_name: str,
    image_id: int,
    rotation_degrees: int,
) -> tuple[np.ndarray, float, bool]:
    """Извлекает patch-признаки либо читает их из воспроизводимого кэша.

    Возвращаемый флаг сообщает, было ли значение прочитано из кэша. Время для
    кэшированного значения равно нулю и не включается в оценку скорости модели.
    """

    if rotation_degrees not in QUERY_ROTATIONS_DEGREES:
        raise ValueError("Разрешены только повороты, кратные 90 градусам")
    cache_directory = CACHE_ROOT_DIRECTORY / extractor.cache_signature
    cache_path = (
        cache_directory / f"{flight_name}_{image_id:06d}_r{rotation_degrees:03d}.npy"
    )
    if cache_path.exists():
        descriptors = np.load(cache_path, allow_pickle=False)
        if descriptors.ndim != 2 or not np.all(np.isfinite(descriptors)):
            raise RuntimeError(f"Некорректный кэш DINOv2: {cache_path}")
        return descriptors.astype(np.float32), 0.0, True

    image_rgb = load_rgb_image(image_path(flight_name, image_id))
    rotation_quarters = rotation_degrees // 90
    if rotation_quarters:
        image_rgb = np.ascontiguousarray(np.rot90(image_rgb, k=rotation_quarters))
    result = extractor.extract(image_rgb)
    cache_directory.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, result.patch_descriptors, allow_pickle=False)
    return result.patch_descriptors, result.processing_time_seconds, False


def geometry_metrics(teach_image_id: int, repeat_image_id: int) -> dict[str, Any]:
    """Проверяет одну найденную пару независимым SIFT + RANSAC."""

    teach_rgb = load_rgb_image(image_path("teach", teach_image_id))
    repeat_rgb = load_rgb_image(image_path("repeat", repeat_image_id))
    try:
        alignment = align_frame_to_reference(
            teach_rgb,
            repeat_rgb,
            config=SiftRansacConfig(),
        )
    except AlignmentFailure as error:
        return {
            "geometry_matrix_returned": False,
            "geometry_rejection_reason": str(error),
            "geometry_inlier_count": None,
            "geometry_inlier_ratio": None,
            "geometry_coverage_fraction": None,
            "geometry_reprojection_p95_px": None,
            "geometry_stability_p95_corner_shift_px": None,
        }

    height, width = repeat_rgb.shape[:2]
    quality = analyze_alignment_quality(
        alignment,
        frame_width_pixels=width,
        frame_height_pixels=height,
        random_seed=RANDOM_SEED,
    )
    return {
        "geometry_matrix_returned": True,
        "geometry_rejection_reason": "",
        "geometry_inlier_count": alignment.inlier_count,
        "geometry_inlier_ratio": alignment.inlier_ratio,
        "geometry_coverage_fraction": alignment.inlier_spatial_coverage_fraction,
        "geometry_reprojection_p95_px": (quality.inlier_reprojection_p95_reference_px),
        "geometry_stability_p95_corner_shift_px": (
            quality.stability_p95_max_corner_shift_reference_px
        ),
    }


def first_positive_rank(distances_meters: np.ndarray) -> int | None:
    """Возвращает ранг первого кандидата в радиусе, считая ранги с единицы."""

    positive_indices = np.flatnonzero(distances_meters <= POSITIVE_RADIUS_METERS)
    if positive_indices.size == 0:
        return None
    return int(positive_indices[0] + 1)


def retrieval_summary(
    query_results: list[dict[str, Any]],
    *,
    variant: str,
) -> dict[str, Any]:
    """Считает Recall@K и геометрическую сводку для одного варианта поиска."""

    selected = [result for result in query_results if result["variant"] == variant]
    first_ranks = [
        int(result["first_positive_rank"])
        for result in selected
        if result["first_positive_rank"] is not None
    ]
    top1_distances = [float(result["top1_distance_meters"]) for result in selected]
    positive_in_top5 = sum(bool(result["positive_in_top5"]) for result in selected)
    geometry_found_positive = sum(
        bool(result["geometry_found_positive_in_top5"]) for result in selected
    )
    return {
        "query_count": len(selected),
        "recall_at_1": sum(bool(result["positive_at_top1"]) for result in selected)
        / len(selected),
        "recall_at_5": positive_in_top5 / len(selected),
        "median_first_positive_rank": (
            float(np.median(first_ranks)) if first_ranks else None
        ),
        "maximum_first_positive_rank": max(first_ranks) if first_ranks else None,
        "queries_without_positive_anywhere": len(selected) - len(first_ranks),
        "median_top1_distance_meters": float(np.median(top1_distances)),
        "geometry_returned_for_positive_in_top5_queries": geometry_found_positive,
    }


def save_heatmap(
    similarities_by_variant: dict[str, np.ndarray],
    nearest_indices: np.ndarray,
    query_ids: list[int],
    teach_ids: np.ndarray,
) -> None:
    """Показывает всю матрицу сходства и положение ближайшей GPS-разметки."""

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    titles = {
        "original": "Без перебора поворотов",
        "rotation_robust": "Максимум по поворотам запроса 0/90/180/270°",
    }
    for axis, variant in zip(axes, titles, strict=True):
        matrix = similarities_by_variant[variant]
        image = axis.imshow(
            matrix, aspect="auto", interpolation="nearest", cmap="viridis"
        )
        axis.scatter(
            nearest_indices,
            np.arange(len(query_ids)),
            marker="x",
            s=55,
            linewidths=1.7,
            color="red",
            label="ближайший Teach по GPS",
        )
        axis.set_title(titles[variant])
        axis.set_ylabel("номер Repeat-запроса")
        axis.set_yticks(
            np.arange(len(query_ids)), [f"{item:06d}" for item in query_ids]
        )
        axis.legend(loc="upper right")
        figure.colorbar(image, ax=axis, label="косинусное сходство")
    tick_positions = np.linspace(0, len(teach_ids) - 1, 7, dtype=int)
    axes[-1].set_xticks(tick_positions, [f"{teach_ids[i]:06d}" for i in tick_positions])
    axes[-1].set_xlabel("разреженная Teach-база в порядке полёта")
    figure.suptitle("CLOUD: DINOv2-S + VLAD, красный крест не участвовал в поиске")
    figure.tight_layout()
    figure.savefig(OUTPUT_HEATMAP_PATH, dpi=160)
    plt.close(figure)


def save_top1_montage(
    query_ids: list[int],
    top1_ids_by_variant: dict[str, list[int]],
) -> None:
    """Сохраняет все запросы и первые найденные кадры без отбора удачных."""

    figure, axes = plt.subplots(len(query_ids), 3, figsize=(12, 2.8 * len(query_ids)))
    for row_index, query_id in enumerate(query_ids):
        items = (
            ("repeat", query_id, f"Repeat {query_id:06d}"),
            (
                "teach",
                top1_ids_by_variant["original"][row_index],
                f"Top-1 без поворотов {top1_ids_by_variant['original'][row_index]:06d}",
            ),
            (
                "teach",
                top1_ids_by_variant["rotation_robust"][row_index],
                "Top-1 с поворотами "
                f"{top1_ids_by_variant['rotation_robust'][row_index]:06d}",
            ),
        )
        for column_index, (flight, image_id, title) in enumerate(items):
            axis = axes[row_index, column_index]
            axis.imshow(load_rgb_image(image_path(flight, image_id)))
            axis.set_title(title, fontsize=9)
            axis.axis("off")
    figure.suptitle("Все 11 запросов и первые кандидаты поиска")
    figure.tight_layout()
    figure.savefig(OUTPUT_TOP1_PATH, dpi=130)
    plt.close(figure)


def main() -> None:
    """Выполняет зафиксированный опыт и сохраняет сырые результаты."""

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
    teach_keyframe_indices = select_spaced_image_indices(
        teach,
        minimum_distance_meters=KEYFRAME_SPACING_METERS,
    )
    teach_ids = teach.image_ids[teach_keyframe_indices]
    teach_xy = teach.xy_meters[teach_keyframe_indices]
    query_pairs = select_teach_repeat_pairs(teach, repeat, query_count=QUERY_COUNT)
    query_ids = [pair.repeat_image_id for pair in query_pairs]
    repeat_indices = position_index_by_image_id(repeat)

    extraction_times: list[float] = []
    cache_hits = 0
    teach_patches: list[np.ndarray] = []
    for sequence_index, teach_id in enumerate(teach_ids, start=1):
        patches, elapsed, from_cache = load_or_extract_patches(
            extractor,
            flight_name="teach",
            image_id=int(teach_id),
            rotation_degrees=0,
        )
        teach_patches.append(patches)
        extraction_times.append(elapsed)
        cache_hits += int(from_cache)
        if sequence_index % 25 == 0 or sequence_index == len(teach_ids):
            print(f"Teach-признаки: {sequence_index}/{len(teach_ids)}")

    vocabulary_started_at = perf_counter()
    vocabulary = fit_visual_vocabulary(
        teach_patches,
        cluster_count=CLUSTER_COUNT,
        maximum_training_descriptors=MAXIMUM_VOCABULARY_DESCRIPTORS,
        random_seed=RANDOM_SEED,
    )
    vocabulary_seconds = perf_counter() - vocabulary_started_at
    database_descriptors = np.stack(
        [aggregate_vlad(patches, vocabulary) for patches in teach_patches]
    )

    geometry_cache: dict[tuple[int, int], dict[str, Any]] = {}
    candidate_rows: list[dict[str, Any]] = []
    query_results: list[dict[str, Any]] = []
    similarities_by_variant: dict[str, list[np.ndarray]] = {
        "original": [],
        "rotation_robust": [],
    }
    nearest_indices: list[int] = []
    top1_ids_by_variant: dict[str, list[int]] = {
        "original": [],
        "rotation_robust": [],
    }

    for query_number, repeat_id in enumerate(query_ids, start=1):
        rotation_similarities: list[np.ndarray] = []
        for rotation_degrees in QUERY_ROTATIONS_DEGREES:
            patches, elapsed, from_cache = load_or_extract_patches(
                extractor,
                flight_name="repeat",
                image_id=repeat_id,
                rotation_degrees=rotation_degrees,
            )
            extraction_times.append(elapsed)
            cache_hits += int(from_cache)
            query_descriptor = aggregate_vlad(patches, vocabulary)
            rotation_similarities.append(database_descriptors @ query_descriptor)

        similarity_stack = np.stack(rotation_similarities)
        variant_similarities = {
            "original": similarity_stack[0],
            "rotation_robust": np.max(similarity_stack, axis=0),
        }
        best_rotation_indices = np.argmax(similarity_stack, axis=0)
        query_xy = repeat.xy_meters[repeat_indices[repeat_id]]
        distances = np.linalg.norm(teach_xy - query_xy, axis=1)
        nearest_indices.append(int(np.argmin(distances)))

        for variant, similarities in variant_similarities.items():
            ranked_indices = np.argsort(-similarities, kind="stable")
            ranked_distances = distances[ranked_indices]
            positive_rank = first_positive_rank(ranked_distances)
            top_indices = ranked_indices[:TOP_K]
            similarities_by_variant[variant].append(similarities)
            top1_ids_by_variant[variant].append(int(teach_ids[top_indices[0]]))
            geometry_positive_found = False

            for rank, database_index in enumerate(top_indices, start=1):
                teach_id = int(teach_ids[database_index])
                geometry_key = (teach_id, repeat_id)
                if geometry_key not in geometry_cache:
                    geometry_cache[geometry_key] = geometry_metrics(teach_id, repeat_id)
                is_positive = bool(distances[database_index] <= POSITIVE_RADIUS_METERS)
                if (
                    is_positive
                    and geometry_cache[geometry_key]["geometry_matrix_returned"]
                ):
                    geometry_positive_found = True
                chosen_rotation = (
                    0
                    if variant == "original"
                    else QUERY_ROTATIONS_DEGREES[
                        int(best_rotation_indices[database_index])
                    ]
                )
                candidate_rows.append(
                    {
                        "repeat_image_id": repeat_id,
                        "variant": variant,
                        "rank": rank,
                        "teach_image_id": teach_id,
                        "cosine_similarity": float(similarities[database_index]),
                        "chosen_query_rotation_degrees": chosen_rotation,
                        "telemetry_distance_meters": float(distances[database_index]),
                        "within_exploratory_positive_radius": is_positive,
                        **geometry_cache[geometry_key],
                    }
                )

            query_results.append(
                {
                    "repeat_image_id": repeat_id,
                    "variant": variant,
                    "positive_at_top1": bool(
                        ranked_distances[0] <= POSITIVE_RADIUS_METERS
                    ),
                    "positive_in_top5": bool(
                        np.any(ranked_distances[:TOP_K] <= POSITIVE_RADIUS_METERS)
                    ),
                    "first_positive_rank": positive_rank,
                    "top1_distance_meters": float(ranked_distances[0]),
                    "geometry_found_positive_in_top5": geometry_positive_found,
                }
            )
        print(f"Repeat-запросы: {query_number}/{len(query_ids)}")

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)

    noncached_times = [value for value in extraction_times if value > 0.0]
    summary = {
        "experiment": "cloud_dinov2_s_vlad_retrieval_smoke",
        "status": "exploratory_not_a_validation",
        "model_manifest": str(MODEL_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "model": manifest,
        "configuration": {
            "projected_crs": PROJECTED_CRS,
            "query_count": QUERY_COUNT,
            "teach_keyframe_spacing_meters": KEYFRAME_SPACING_METERS,
            "exploratory_positive_radius_meters": POSITIVE_RADIUS_METERS,
            "top_k": TOP_K,
            "vlad_cluster_count": CLUSTER_COUNT,
            "vlad_dimensions": CLUSTER_COUNT
            * int(manifest["model"]["patch_descriptor_dimensions"]),
            "maximum_vocabulary_training_descriptors": MAXIMUM_VOCABULARY_DESCRIPTORS,
            "query_rotations_degrees": list(QUERY_ROTATIONS_DEGREES),
            "random_seed": RANDOM_SEED,
            "sift_ransac": asdict(SiftRansacConfig()),
        },
        "data": {
            "teach_positioned_frame_count": int(teach.image_ids.size),
            "teach_keyframe_count": int(teach_ids.size),
            "repeat_positioned_frame_count": int(repeat.image_ids.size),
            "repeat_query_ids": query_ids,
        },
        "results": {
            variant: retrieval_summary(query_results, variant=variant)
            for variant in ("original", "rotation_robust")
        },
        "per_query_results": query_results,
        "timing": {
            "noncached_extraction_count": len(noncached_times),
            "cache_hit_count": cache_hits,
            "median_dinov2_extraction_seconds": (
                float(np.median(noncached_times)) if noncached_times else None
            ),
            "total_dinov2_extraction_seconds": float(sum(noncached_times)),
            "visual_vocabulary_fit_seconds": vocabulary_seconds,
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
            "Один участок и одна пара Teach/Repeat-полётов CLOUD.",
            "Те же 11 запросов ранее использовались в smoke v2.",
            "GPS задаёт лишь приблизительную контрольную близость, "
            "а не совпадение поля зрения.",
            "Радиус 3 м выбран до запуска, но ещё не валидирован на отдельной выборке.",
            "Использованы финальные DINOv2 patch-токены и облегчённый VLAD, "
            "а не точное воспроизведение AnyLoc.",
            "Максимум по четырём поворотам может усиливать как правильные, "
            "так и ложные совпадения.",
            "Порог автоматического принятия результата на этих 11 запросах "
            "не подбирался.",
        ],
    }
    OUTPUT_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    similarity_arrays = {
        variant: np.stack(values) for variant, values in similarities_by_variant.items()
    }
    save_heatmap(
        similarity_arrays,
        np.asarray(nearest_indices),
        query_ids,
        teach_ids,
    )
    save_top1_montage(query_ids, top1_ids_by_variant)

    for variant, result in summary["results"].items():
        print(
            f"{variant}: Recall@1={result['recall_at_1']:.3f}, "
            f"Recall@5={result['recall_at_5']:.3f}, "
            f"median top-1 distance={result['median_top1_distance_meters']:.2f} m"
        )
    print(f"Кандидаты: {OUTPUT_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Сводка: {OUTPUT_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Матрица поиска: {OUTPUT_HEATMAP_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Top-1 кадры: {OUTPUT_TOP1_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
