#!/usr/bin/env python3
"""Сквозной CLOUD-гейт: поиск, геометрия, выбор или безопасный отказ.

Эксперимент разделён на два режима. ``develop`` перебирает небольшой заранее
заданный набор понятных порогов только на UTIAS Field trial 1. ``validate``
применяет уже зафиксированные пороги к другому trial без перенастройки.

Рабочий алгоритм видит только изображения. GPS-координаты подключаются после
выбора кандидата и служат приблизительной контрольной истиной эксперимента.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch

from aerial_mapper.alignment import (
    AlignmentFailure,
    SiftRansacConfig,
    align_frame_to_reference,
)
from aerial_mapper.cloud_dataset import (
    load_positioned_images,
    select_spaced_image_indices,
)
from aerial_mapper.place_retrieval import (
    DinoV2SmallPatchExtractor,
    aggregate_vlad,
    fit_visual_vocabulary,
    rank_database,
)
from aerial_mapper.place_verification import (
    CandidateGeometry,
    VerificationDecision,
    VerificationThresholds,
    verify_candidates,
)
from aerial_mapper.quality import analyze_alignment_quality

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_MANIFEST_PATH = PROJECT_ROOT / "data/manifests/dinov2_vits14.json"
TORCH_HOME = PROJECT_ROOT / "data/models/torch"
CACHE_ROOT = PROJECT_ROOT / "outputs/cache/cloud_end_to_end"

PROJECTED_CRS = "EPSG:32617"
KEYFRAME_SPACING_METERS = 1.0
POSITIVE_RADIUS_METERS = 3.0
CATASTROPHIC_DISTANCE_METERS = 10.0
TOP_K = 10
CLUSTER_COUNT = 8
MAXIMUM_VOCABULARY_DESCRIPTORS = 20_000
RANDOM_SEED = 0

# Эти критерии относятся к итоговому trial целиком. Радиус 3 м строгий и
# чувствителен к GPS, поэтому отдельно запрещается более надёжно различимая
# ошибка свыше 10 м. Пороги зафиксированы до просмотра результата trial 2.
GATE_MINIMUM_RECALL_AT_10 = 0.90
GATE_MINIMUM_ACCEPT_WITHIN_10M_RATE = 0.80
GATE_MAXIMUM_CATASTROPHIC_FALSE_ACCEPTS = 0
GATE_MINIMUM_HARD_NEGATIVE_CASES = 20
GATE_MAXIMUM_HARD_NEGATIVE_ACCEPTS = 0

# Пороги выбраны на trial 1 до распаковки и первого запуска trial 2. Проверка
# отношения 1.0 означает, что среди пригодных кандидатов берётся лидер по числу
# inlier-пар без дополнительного требования отрыва: соседние правильные
# Teach-кадры часто видят одну область и закономерно дают близкие результаты.
FROZEN_THRESHOLDS = VerificationThresholds(
    minimum_inlier_count=20,
    minimum_inlier_ratio=0.5,
    minimum_coverage_fraction=0.1,
    maximum_reprojection_p95_px=3.0,
    maximum_stability_p95_corner_shift_px=20.0,
    minimum_winner_to_runner_up_inlier_ratio=1.0,
)


def parse_args() -> argparse.Namespace:
    """Читает режим, каталог trial и отдельный префикс файлов результата."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("develop", "validate"), required=True)
    parser.add_argument("--trial-directory", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def load_rgb_image(path: Path) -> np.ndarray:
    """Читает кадр CLOUD в принятом проектом порядке каналов RGB."""

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def sha256_file(path: Path) -> str:
    """Потоково проверяет зафиксированные веса, не копируя файл в память."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_weights(manifest: dict[str, Any]) -> None:
    """Не допускает тихую замену весов DINOv2 между двумя trial."""

    model = manifest["model"]
    weights_path = TORCH_HOME / "hub/checkpoints" / model["weights_filename"]
    if not weights_path.exists():
        return
    if weights_path.stat().st_size != int(model["weights_size_bytes"]):
        raise RuntimeError(f"Не совпадает размер весов DINOv2: {weights_path}")
    if sha256_file(weights_path) != str(model["weights_sha256"]):
        raise RuntimeError(f"Не совпадает SHA-256 весов DINOv2: {weights_path}")


def image_path(trial_directory: Path, flight_name: str, image_id: int) -> Path:
    """Строит путь по имени полёта и шестизначному номеру кадра."""

    return trial_directory / flight_name / "images" / f"{image_id:06d}.png"


def load_or_extract_patches(
    extractor: DinoV2SmallPatchExtractor,
    *,
    trial_directory: Path,
    flight_name: str,
    image_id: int,
) -> tuple[np.ndarray, float, bool]:
    """Возвращает DINOv2-признаки с безопасным кэшем по имени trial.

    Номера кадров повторяются в разных архивах CLOUD. Поэтому имя trial — часть
    ключа кэша; без него validation мог бы незаметно прочитать признаки trial 1.
    """

    cache_directory = CACHE_ROOT / extractor.cache_signature / trial_directory.name
    cache_path = cache_directory / f"{flight_name}_{image_id:06d}.npy"
    if cache_path.exists():
        descriptors = np.load(cache_path, allow_pickle=False)
        if descriptors.ndim != 2 or not np.all(np.isfinite(descriptors)):
            raise RuntimeError(f"Некорректный кэш DINOv2: {cache_path}")
        return descriptors.astype(np.float32), 0.0, True

    result = extractor.extract(
        load_rgb_image(image_path(trial_directory, flight_name, image_id))
    )
    cache_directory.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, result.patch_descriptors, allow_pickle=False)
    return result.patch_descriptors, result.processing_time_seconds, False


def geometry_evidence(
    *,
    trial_directory: Path,
    teach_image_id: int,
    repeat_image_id: int,
    database_index: int,
    retrieval_rank: int,
    retrieval_similarity: float,
) -> CandidateGeometry:
    """Строит геометрические доказательства для одной предложенной пары."""

    teach_rgb = load_rgb_image(image_path(trial_directory, "teach", teach_image_id))
    repeat_rgb = load_rgb_image(image_path(trial_directory, "repeat", repeat_image_id))
    try:
        alignment = align_frame_to_reference(
            teach_rgb,
            repeat_rgb,
            config=SiftRansacConfig(),
        )
    except AlignmentFailure:
        return CandidateGeometry(
            database_index=database_index,
            retrieval_rank=retrieval_rank,
            retrieval_similarity=retrieval_similarity,
            inlier_count=None,
            inlier_ratio=None,
            coverage_fraction=None,
            reprojection_p95_px=None,
            stability_p95_corner_shift_px=None,
        )

    height, width = repeat_rgb.shape[:2]
    quality = analyze_alignment_quality(
        alignment,
        frame_width_pixels=width,
        frame_height_pixels=height,
        random_seed=RANDOM_SEED,
    )
    return CandidateGeometry(
        database_index=database_index,
        retrieval_rank=retrieval_rank,
        retrieval_similarity=retrieval_similarity,
        inlier_count=alignment.inlier_count,
        inlier_ratio=alignment.inlier_ratio,
        coverage_fraction=alignment.inlier_spatial_coverage_fraction,
        reprojection_p95_px=quality.inlier_reprojection_p95_reference_px,
        stability_p95_corner_shift_px=(
            quality.stability_p95_max_corner_shift_reference_px
        ),
    )


def threshold_grid() -> list[VerificationThresholds]:
    """Создаёт небольшой интерпретируемый набор вариантов для trial 1."""

    return [
        VerificationThresholds(inliers, ratio, coverage, 3.0, stability, dominance)
        for inliers in (12, 20, 30, 40)
        for ratio in (0.4, 0.5, 0.6)
        for coverage in (0.10, 0.20, 0.30)
        for stability in (5.0, 10.0, 20.0)
        for dominance in (1.0, 1.1, 1.25, 1.5)
    ]


def summarize_decisions(
    query_records: list[dict[str, Any]],
    *,
    thresholds: VerificationThresholds,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Применяет пороги и считает результат по GPS после принятия решения."""

    rows: list[dict[str, Any]] = []
    correct = 0
    near = 0
    catastrophic = 0
    rejected = 0
    accepted_distances: list[float] = []
    hard_negative_case_count = 0
    hard_negative_accept_count = 0
    evaluable_count = sum(
        record["has_positive_in_database"] for record in query_records
    )
    recall_at_10_count = sum(record["positive_in_top_k"] for record in query_records)

    for record in query_records:
        far_candidates = [
            candidate
            for candidate in record["candidates"]
            if record["distances_meters"][candidate.database_index]
            > CATASTROPHIC_DISTANCE_METERS
        ]
        if far_candidates:
            hard_negative_case_count += 1
            hard_negative_decision = verify_candidates(
                far_candidates,
                thresholds=thresholds,
            )
            hard_negative_accepted = hard_negative_decision.accepted
            hard_negative_accept_count += int(hard_negative_accepted)
        else:
            hard_negative_accepted = None

        decision: VerificationDecision = verify_candidates(
            record["candidates"],
            thresholds=thresholds,
        )
        if decision.accepted_database_index is None:
            selected_distance = None
            outcome = "rejected"
            rejected += 1
        else:
            selected_distance = float(
                record["distances_meters"][decision.accepted_database_index]
            )
            accepted_distances.append(selected_distance)
            if selected_distance <= POSITIVE_RADIUS_METERS:
                outcome = "correct_within_3m"
                correct += 1
            elif selected_distance <= CATASTROPHIC_DISTANCE_METERS:
                outcome = "near_3m_to_10m"
                near += 1
            else:
                outcome = "false_accept_over_10m"
                catastrophic += 1
        rows.append(
            {
                "repeat_image_id": record["repeat_image_id"],
                "nearest_teach_distance_meters": record[
                    "nearest_teach_distance_meters"
                ],
                "positive_in_top_k": record["positive_in_top_k"],
                "outcome": outcome,
                "selected_teach_image_id": (
                    record["teach_ids"][decision.accepted_database_index]
                    if decision.accepted_database_index is not None
                    else None
                ),
                "selected_distance_meters": selected_distance,
                "selected_retrieval_rank": decision.accepted_retrieval_rank,
                "eligible_candidate_count": decision.eligible_candidate_count,
                "winner_to_runner_up_inlier_ratio": (
                    decision.winner_to_runner_up_inlier_ratio
                ),
                "decision_reason": decision.reason,
                "hard_negative_candidate_count": len(far_candidates),
                "hard_negative_accepted": hard_negative_accepted,
            }
        )

    accepted = correct + near + catastrophic
    summary = {
        "query_count": len(query_records),
        "evaluable_query_count": evaluable_count,
        "recall_at_10": (
            recall_at_10_count / evaluable_count if evaluable_count else None
        ),
        "correct_accept_within_3m_count": correct,
        "near_accept_3m_to_10m_count": near,
        "catastrophic_false_accept_over_10m_count": catastrophic,
        "rejection_count": rejected,
        "accepted_within_10m_rate": (correct + near) / len(query_records),
        "correct_accept_rate": correct / evaluable_count if evaluable_count else None,
        "accepted_precision_within_3m": correct / accepted if accepted else None,
        "hard_negative_case_count": hard_negative_case_count,
        "hard_negative_accept_count": hard_negative_accept_count,
        "median_accepted_distance_meters": (
            float(np.median(accepted_distances)) if accepted_distances else None
        ),
        "p95_accepted_distance_meters": (
            float(np.percentile(accepted_distances, 95)) if accepted_distances else None
        ),
    }
    return summary, rows


def choose_development_thresholds(
    query_records: list[dict[str, Any]],
) -> tuple[VerificationThresholds, dict[str, Any], list[dict[str, Any]]]:
    """Максимизирует полезные ответы при первичном запрете грубых ошибок.

    Сначала минимизируется принятие трудных отрицательных контролей и обычных
    ответов дальше 10 м. Затем максимизируется число полезных ответов в пределах
    10 м. Радиус 3 м остаётся строгой диагностикой: соседний Teach-кадр дальше
    этой границы всё ещё может видеть ту же область и давать лучшую матрицу.
    """

    evaluated: list[
        tuple[
            tuple[int, int, int, int, int, float],
            VerificationThresholds,
            dict[str, Any],
            list[dict[str, Any]],
        ]
    ] = []
    for thresholds in threshold_grid():
        summary, rows = summarize_decisions(query_records, thresholds=thresholds)
        objective = (
            int(summary["hard_negative_accept_count"]),
            int(summary["catastrophic_false_accept_over_10m_count"]),
            -int(summary["correct_accept_within_3m_count"])
            - int(summary["near_accept_3m_to_10m_count"]),
            -int(summary["correct_accept_within_3m_count"]),
            int(summary["rejection_count"]),
            abs(thresholds.minimum_inlier_ratio - 0.5),
        )
        evaluated.append((objective, thresholds, summary, rows))
    _, thresholds, summary, rows = min(evaluated, key=lambda item: item[0])
    return thresholds, summary, rows


def gate_result(summary: dict[str, Any]) -> dict[str, Any]:
    """Сопоставляет итоговые метрики с заранее объявленным CLOUD-гейтом."""

    checks = {
        "recall_at_10": bool(
            summary["recall_at_10"] is not None
            and summary["recall_at_10"] >= GATE_MINIMUM_RECALL_AT_10
        ),
        "accepted_within_10m_rate": bool(
            summary["accepted_within_10m_rate"] >= GATE_MINIMUM_ACCEPT_WITHIN_10M_RATE
        ),
        "catastrophic_false_accepts": bool(
            summary["catastrophic_false_accept_over_10m_count"]
            <= GATE_MAXIMUM_CATASTROPHIC_FALSE_ACCEPTS
        ),
        "enough_hard_negative_cases": bool(
            summary["hard_negative_case_count"] >= GATE_MINIMUM_HARD_NEGATIVE_CASES
        ),
        "hard_negative_accepts": bool(
            summary["hard_negative_accept_count"] <= GATE_MAXIMUM_HARD_NEGATIVE_ACCEPTS
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Сохраняет прямоугольную таблицу с устойчивым порядком столбцов."""

    if not rows:
        raise ValueError("Нельзя записать пустую таблицу")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Выполняет сквозной поиск и сохраняет все данные для аудита."""

    args = parse_args()
    trial_directory = args.trial_directory.resolve()
    if not (trial_directory / "teach/images").is_dir():
        raise FileNotFoundError(f"Нет Teach-изображений: {trial_directory}")
    if not (trial_directory / "repeat/images").is_dir():
        raise FileNotFoundError(f"Нет Repeat-изображений: {trial_directory}")
    model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    verify_model_weights(model_manifest)
    extractor = DinoV2SmallPatchExtractor(
        repository=model_manifest["source"]["torch_hub_repository"],
        revision=model_manifest["source"]["repository_revision"],
        model_name=model_manifest["model"]["torch_hub_name"],
        torch_home=TORCH_HOME,
    )
    verify_model_weights(model_manifest)

    teach = load_positioned_images(
        trial_directory / "teach", projected_crs=PROJECTED_CRS
    )
    repeat = load_positioned_images(
        trial_directory / "repeat", projected_crs=PROJECTED_CRS
    )
    teach_indices = select_spaced_image_indices(
        teach, minimum_distance_meters=KEYFRAME_SPACING_METERS
    )
    repeat_indices = select_spaced_image_indices(
        repeat, minimum_distance_meters=KEYFRAME_SPACING_METERS
    )
    teach_ids = teach.image_ids[teach_indices]
    teach_xy = teach.xy_meters[teach_indices]
    repeat_ids = repeat.image_ids[repeat_indices]
    repeat_xy = repeat.xy_meters[repeat_indices]

    extraction_seconds: list[float] = []
    cache_hit_count = 0
    teach_patches: list[np.ndarray] = []
    for number, teach_id in enumerate(teach_ids, start=1):
        patches, elapsed, from_cache = load_or_extract_patches(
            extractor,
            trial_directory=trial_directory,
            flight_name="teach",
            image_id=int(teach_id),
        )
        teach_patches.append(patches)
        extraction_seconds.append(elapsed)
        cache_hit_count += int(from_cache)
        if number % 25 == 0 or number == teach_ids.size:
            print(f"Teach DINOv2: {number}/{teach_ids.size}")

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

    candidate_rows: list[dict[str, Any]] = []
    query_records: list[dict[str, Any]] = []
    geometry_started_at = perf_counter()
    for query_number, (repeat_id, query_xy) in enumerate(
        zip(repeat_ids, repeat_xy, strict=True), start=1
    ):
        query_patches, elapsed, from_cache = load_or_extract_patches(
            extractor,
            trial_directory=trial_directory,
            flight_name="repeat",
            image_id=int(repeat_id),
        )
        extraction_seconds.append(elapsed)
        cache_hit_count += int(from_cache)
        query_descriptor = aggregate_vlad(query_patches, vocabulary)
        retrieval = rank_database(
            database_descriptors,
            query_descriptor,
            maximum_results=min(TOP_K, teach_ids.size),
        )
        distances = np.linalg.norm(teach_xy - query_xy, axis=1)
        evidence: list[CandidateGeometry] = []
        for rank, (database_index, similarity) in enumerate(
            zip(
                retrieval.database_indices,
                retrieval.similarities,
                strict=True,
            ),
            start=1,
        ):
            candidate = geometry_evidence(
                trial_directory=trial_directory,
                teach_image_id=int(teach_ids[database_index]),
                repeat_image_id=int(repeat_id),
                database_index=int(database_index),
                retrieval_rank=rank,
                retrieval_similarity=float(similarity),
            )
            evidence.append(candidate)
            candidate_rows.append(
                {
                    "repeat_image_id": int(repeat_id),
                    "rank": rank,
                    "teach_image_id": int(teach_ids[database_index]),
                    "telemetry_distance_meters": float(distances[database_index]),
                    "within_3m": bool(
                        distances[database_index] <= POSITIVE_RADIUS_METERS
                    ),
                    **asdict(candidate),
                }
            )
        query_records.append(
            {
                "repeat_image_id": int(repeat_id),
                "teach_ids": teach_ids,
                "distances_meters": distances,
                "nearest_teach_distance_meters": float(np.min(distances)),
                "has_positive_in_database": bool(
                    np.any(distances <= POSITIVE_RADIUS_METERS)
                ),
                "positive_in_top_k": bool(
                    np.any(
                        distances[retrieval.database_indices] <= POSITIVE_RADIUS_METERS
                    )
                ),
                "candidates": evidence,
            }
        )
        if query_number % 10 == 0 or query_number == repeat_ids.size:
            print(f"Repeat запросы и геометрия: {query_number}/{repeat_ids.size}")
    geometry_seconds = perf_counter() - geometry_started_at

    if args.mode == "develop":
        thresholds, decision_summary, decision_rows = choose_development_thresholds(
            query_records
        )
        status = "development_threshold_selection"
    else:
        thresholds = FROZEN_THRESHOLDS
        decision_summary, decision_rows = summarize_decisions(
            query_records, thresholds=thresholds
        )
        status = "held_out_validation"

    output_prefix = args.output_prefix.resolve()
    candidates_path = output_prefix.with_name(output_prefix.name + "_candidates.csv")
    decisions_path = output_prefix.with_name(output_prefix.name + "_decisions.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    write_csv(candidates_path, candidate_rows)
    write_csv(decisions_path, decision_rows)

    noncached_times = [value for value in extraction_seconds if value > 0.0]
    summary = {
        "experiment": "cloud_end_to_end_gate",
        "status": status,
        "mode": args.mode,
        "trial": trial_directory.name,
        "configuration": {
            "projected_crs": PROJECTED_CRS,
            "keyframe_spacing_meters": KEYFRAME_SPACING_METERS,
            "positive_radius_meters": POSITIVE_RADIUS_METERS,
            "catastrophic_distance_meters": CATASTROPHIC_DISTANCE_METERS,
            "top_k": TOP_K,
            "vlad_cluster_count": CLUSTER_COUNT,
            "maximum_vocabulary_training_descriptors": (MAXIMUM_VOCABULARY_DESCRIPTORS),
            "random_seed": RANDOM_SEED,
            "sift_ransac": asdict(SiftRansacConfig()),
            "verification_thresholds": asdict(thresholds),
        },
        "gate_criteria": {
            "minimum_recall_at_10": GATE_MINIMUM_RECALL_AT_10,
            "minimum_accepted_within_10m_rate": (GATE_MINIMUM_ACCEPT_WITHIN_10M_RATE),
            "maximum_catastrophic_false_accepts": (
                GATE_MAXIMUM_CATASTROPHIC_FALSE_ACCEPTS
            ),
            "minimum_hard_negative_cases": GATE_MINIMUM_HARD_NEGATIVE_CASES,
            "maximum_hard_negative_accepts": GATE_MAXIMUM_HARD_NEGATIVE_ACCEPTS,
        },
        "data": {
            "teach_positioned_frame_count": int(teach.image_ids.size),
            "teach_keyframe_count": int(teach_ids.size),
            "repeat_positioned_frame_count": int(repeat.image_ids.size),
            "repeat_query_count": int(repeat_ids.size),
        },
        "results": decision_summary,
        "gate": gate_result(decision_summary),
        "timing": {
            "noncached_dinov2_extraction_count": len(noncached_times),
            "cache_hit_count": cache_hit_count,
            "median_dinov2_extraction_seconds": (
                float(np.median(noncached_times)) if noncached_times else None
            ),
            "total_dinov2_extraction_seconds": float(sum(noncached_times)),
            "visual_vocabulary_fit_seconds": vocabulary_seconds,
            "retrieval_and_geometry_seconds": geometry_seconds,
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
            "GPS задаёт приблизительную близость, а не точное перекрытие поля зрения.",
            "Радиусы 3 и 10 м являются экспериментальными границами, а не "
            "заявленной метрической точностью системы.",
            "Гомография проверяет внутреннюю согласованность одной плоской модели, "
            "но CLOUD не даёт плотную метрическую 3D-истину.",
            "Итог относится к маршруту UTIAS Field и не доказывает переносимость "
            "на другой город, сезон или камеру.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["results"], ensure_ascii=False, indent=2))
    print(f"Пороги: {asdict(thresholds)}")
    print(f"Гейт: {summary['gate']}")
    print(f"Сводка: {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
