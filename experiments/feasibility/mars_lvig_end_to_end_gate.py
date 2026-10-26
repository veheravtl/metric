#!/usr/bin/env python3
"""MARS-LVIG development/held-out гейт визуальной Teach/Repeat-привязки.

Рабочий алгоритм получает только подготовленные RGB. RTK используется отдельно:
для разрежения экспериментальной выборки и оценки уже выданного ответа.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
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
from aerial_mapper.camera_distortion import BrownConradyDistortion
from aerial_mapper.cloud_dataset import (
    load_positioned_images,
    select_spaced_image_indices,
)
from aerial_mapper.mars_lvig_preprocessing import prepare_mars_lvig_image
from aerial_mapper.place_retrieval import (
    DinoV2SmallPatchExtractor,
    aggregate_vlad,
    fit_visual_vocabulary,
    rank_database,
)
from aerial_mapper.place_verification import (
    CandidateGeometry,
    VerificationThresholds,
    verify_candidates,
)
from aerial_mapper.quality import analyze_alignment_quality
from aerial_mapper.radiance_camera import PinholeIntrinsics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data/queries/mars_lvig/extracted"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/mars_lvig_hkairport.json"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "data/manifests/dinov2_vits14.json"
TORCH_HOME = PROJECT_ROOT / "data/models/torch"
CACHE_ROOT = PROJECT_ROOT / "outputs/cache/mars_lvig_gate"

TEACH_SEQUENCE = "HKairport01"
DEVELOPMENT_SEQUENCE = "HKairport02"
HELD_OUT_SEQUENCE = "HKairport03"
KEYFRAME_SPACING_METERS = 5.0
POSITIVE_RADIUS_METERS = 10.0
CATASTROPHIC_DISTANCE_METERS = 200.0
PREPARED_LONG_SIDE_PIXELS = 672
DINO_INPUT_SIZE_PIXELS = (336, 280)
TOP_K = 10
CLUSTER_COUNT = 8
MAXIMUM_VOCABULARY_DESCRIPTORS = 20_000
RANDOM_SEED = 0

GATE_MINIMUM_EVALUABLE_QUERIES = 100
GATE_MINIMUM_RECALL_AT_10 = 0.90
GATE_MINIMUM_ACCEPT_WITHIN_200M_RATE = 0.80
GATE_MAXIMUM_CATASTROPHIC_FALSE_ACCEPTS = 0
GATE_MINIMUM_HARD_NEGATIVE_CASES = 50
GATE_MAXIMUM_HARD_NEGATIVE_ACCEPTS = 0

# Эти значения выбраны только по HKairport02. Среди 54 наборов с одинаковыми
# решениями взят наиболее строгий набор; HKairport03 на выбор не влиял.
FROZEN_THRESHOLDS: VerificationThresholds | None = VerificationThresholds(
    minimum_inlier_count=60,
    minimum_inlier_ratio=0.5,
    minimum_coverage_fraction=0.20,
    maximum_reprojection_p95_px=4.0,
    maximum_stability_p95_corner_shift_px=10.0,
    minimum_winner_to_runner_up_inlier_ratio=1.0,
)


@dataclass(frozen=True)
class PreparedImageStore:
    """Загружает сырой кадр и кэширует единую исправленную PNG-версию."""

    flight_directory: Path
    cache_directory: Path
    intrinsics: PinholeIntrinsics
    distortion: BrownConradyDistortion

    def load(self, image_id: int) -> np.ndarray:
        """Возвращает RGB uint8 после коррекции дисторсии и масштаба."""

        cache_path = self.cache_directory / f"{image_id:06d}.png"
        if cache_path.exists():
            cached_bgr = cv2.imread(str(cache_path), cv2.IMREAD_COLOR)
            if cached_bgr is None:
                raise RuntimeError(f"Не удалось прочитать кэш: {cache_path}")
            return cv2.cvtColor(cached_bgr, cv2.COLOR_BGR2RGB)

        candidates = [
            self.flight_directory / "images" / f"{image_id:06d}.jpg",
            self.flight_directory / "images" / f"{image_id:06d}.png",
        ]
        source_path = next((path for path in candidates if path.is_file()), None)
        if source_path is None:
            raise FileNotFoundError(f"Не найден исходный кадр {image_id}")
        source_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source_bgr is None:
            raise RuntimeError(f"OpenCV не декодировал {source_path}")
        source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        prepared_rgb = prepare_mars_lvig_image(
            source_rgb,
            intrinsics=self.intrinsics,
            distortion=self.distortion,
            maximum_side_pixels=PREPARED_LONG_SIDE_PIXELS,
        )

        success, encoded = cv2.imencode(
            ".png",
            cv2.cvtColor(prepared_rgb, cv2.COLOR_RGB2BGR),
        )
        if not success:
            raise RuntimeError(f"Не удалось закодировать кэш кадра {image_id}")
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".png.part")
        temporary_path.write_bytes(encoded.tobytes())
        temporary_path.replace(cache_path)
        return prepared_rgb


def parse_args() -> argparse.Namespace:
    """Разделяет development и ещё не разрешённый held-out запуск."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("develop", "validate"), required=True)
    parser.add_argument("--repeat-sequence", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Считает SHA-256 потоково для проверки зафиксированных весов."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_weights(manifest: dict[str, Any]) -> None:
    """Не допускает тихую замену DINOv2 между development и held-out."""

    model = manifest["model"]
    weights_path = TORCH_HOME / "hub/checkpoints" / model["weights_filename"]
    if not weights_path.exists():
        return
    if weights_path.stat().st_size != int(model["weights_size_bytes"]):
        raise RuntimeError(f"Не совпадает размер весов DINOv2: {weights_path}")
    if sha256_file(weights_path) != str(model["weights_sha256"]):
        raise RuntimeError(f"Не совпадает SHA-256 весов DINOv2: {weights_path}")


def camera_from_manifest(
    manifest: dict[str, Any],
) -> tuple[PinholeIntrinsics, BrownConradyDistortion, str]:
    """Создаёт калибровку и подпись кэша из зафиксированного JSON."""

    camera = manifest["camera"]
    intrinsics = PinholeIntrinsics(
        width_px=int(camera["width_pixels"]),
        height_px=int(camera["height_pixels"]),
        focal_x_px=float(camera["fx_pixels"]),
        focal_y_px=float(camera["fy_pixels"]),
        principal_x_px=float(camera["cx_pixels"]),
        principal_y_px=float(camera["cy_pixels"]),
    )
    coefficients = [float(value) for value in camera["distortion"]]
    if len(coefficients) != 4:
        raise ValueError("Ожидались коэффициенты k1, k2, p1, p2")
    distortion = BrownConradyDistortion(
        k1=coefficients[0],
        k2=coefficients[1],
        p1=coefficients[2],
        p2=coefficients[3],
    )
    serialized = json.dumps(camera, sort_keys=True).encode()
    signature = (
        f"long{PREPARED_LONG_SIDE_PIXELS}_"
        f"{hashlib.sha256(serialized).hexdigest()[:12]}"
    )
    return intrinsics, distortion, signature


def load_or_extract_patches(
    extractor: DinoV2SmallPatchExtractor,
    image_store: PreparedImageStore,
    *,
    cache_signature: str,
    sequence_name: str,
    image_id: int,
) -> tuple[np.ndarray, float, bool]:
    """Извлекает DINOv2 patch-токены или проверяет безопасный numpy-кэш."""

    cache_directory = (
        CACHE_ROOT
        / "dinov2"
        / extractor.cache_signature
        / cache_signature
        / sequence_name
    )
    cache_path = cache_directory / f"{image_id:06d}.npy"
    if cache_path.exists():
        descriptors = np.load(cache_path, allow_pickle=False)
        if descriptors.ndim != 2 or not np.all(np.isfinite(descriptors)):
            raise RuntimeError(f"Некорректный DINOv2-кэш: {cache_path}")
        return descriptors.astype(np.float32), 0.0, True

    result = extractor.extract(image_store.load(image_id))
    cache_directory.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, result.patch_descriptors, allow_pickle=False)
    return result.patch_descriptors, result.processing_time_seconds, False


def geometry_evidence(
    *,
    teach_store: PreparedImageStore,
    repeat_store: PreparedImageStore,
    teach_image_id: int,
    repeat_image_id: int,
    database_index: int,
    retrieval_rank: int,
    retrieval_similarity: float,
) -> CandidateGeometry:
    """Проверяет одну визуально предложенную пару независимой геометрией."""

    teach_rgb = teach_store.load(teach_image_id)
    repeat_rgb = repeat_store.load(repeat_image_id)
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
    """Перебирает только заранее объявленные интерпретируемые пороги."""

    return [
        VerificationThresholds(
            minimum_inlier_count=inliers,
            minimum_inlier_ratio=ratio,
            minimum_coverage_fraction=coverage,
            maximum_reprojection_p95_px=reprojection,
            maximum_stability_p95_corner_shift_px=stability,
            minimum_winner_to_runner_up_inlier_ratio=dominance,
        )
        for inliers in (20, 40, 60)
        for ratio in (0.3, 0.5, 0.7)
        for coverage in (0.05, 0.10, 0.20)
        for reprojection in (2.0, 4.0, 6.0)
        for stability in (10.0, 20.0, 40.0)
        for dominance in (1.0, 1.1, 1.25)
    ]


def summarize_decisions(
    query_records: list[dict[str, Any]],
    *,
    thresholds: VerificationThresholds,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Применяет визуальные пороги, затем отдельно оценивает ответ по RTK."""

    rows: list[dict[str, Any]] = []
    within_10m = 0
    ambiguous_10m_to_200m = 0
    catastrophic = 0
    rejected = 0
    hard_negative_cases = 0
    hard_negative_accepts = 0
    accepted_distances: list[float] = []
    evaluable_records = [
        record for record in query_records if record["has_positive_in_database"]
    ]
    recall_count = sum(record["positive_in_top_k"] for record in evaluable_records)

    for record in query_records:
        hard_candidates = record["hard_negative_candidates"]
        hard_accepted: bool | None = None
        if hard_candidates:
            hard_negative_cases += 1
            hard_decision = verify_candidates(hard_candidates, thresholds=thresholds)
            hard_accepted = hard_decision.accepted
            hard_negative_accepts += int(hard_accepted)

        decision = verify_candidates(record["candidates"], thresholds=thresholds)
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
                outcome = "accepted_within_10m"
                within_10m += 1
            elif selected_distance <= CATASTROPHIC_DISTANCE_METERS:
                outcome = "accepted_ambiguous_10m_to_200m"
                ambiguous_10m_to_200m += 1
            else:
                outcome = "false_accept_over_200m"
                catastrophic += 1

        rows.append(
            {
                "repeat_image_id": record["repeat_image_id"],
                "nearest_teach_distance_meters": record[
                    "nearest_teach_distance_meters"
                ],
                "has_positive_in_database": record["has_positive_in_database"],
                "positive_in_top_k": record["positive_in_top_k"],
                "outcome": outcome,
                "selected_teach_image_id": (
                    int(record["teach_ids"][decision.accepted_database_index])
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
                "hard_negative_candidate_count": len(hard_candidates),
                "hard_negative_accepted": hard_accepted,
            }
        )

    accepted_within_200m = within_10m + ambiguous_10m_to_200m
    summary = {
        "query_count": len(query_records),
        "evaluable_query_count": len(evaluable_records),
        "recall_at_10": (
            recall_count / len(evaluable_records) if evaluable_records else None
        ),
        "accepted_within_10m_count": within_10m,
        "accepted_ambiguous_10m_to_200m_count": ambiguous_10m_to_200m,
        "catastrophic_false_accept_over_200m_count": catastrophic,
        "rejection_count": rejected,
        "accepted_within_200m_rate": accepted_within_200m / len(query_records),
        "hard_negative_case_count": hard_negative_cases,
        "hard_negative_accept_count": hard_negative_accepts,
        "median_accepted_distance_meters": (
            float(np.median(accepted_distances)) if accepted_distances else None
        ),
        "p95_accepted_distance_meters": (
            float(np.percentile(accepted_distances, 95))
            if accepted_distances
            else None
        ),
        "maximum_accepted_distance_meters": (
            float(np.max(accepted_distances)) if accepted_distances else None
        ),
    }
    return summary, rows


def choose_development_thresholds(
    query_records: list[dict[str, Any]],
) -> tuple[VerificationThresholds, dict[str, Any], list[dict[str, Any]]]:
    """Сначала запрещает опасные ответы, затем максимизирует доступность.

    Если несколько вариантов дают абсолютно одинаковые наблюдаемые решения,
    последние компоненты ключа выбирают более строгий вариант. Это устраняет
    зависимость результата от порядка сетки и не выдаёт строгость за отдельное
    улучшение качества на development-данных.
    """

    evaluated: list[
        tuple[
            tuple[int | float, ...],
            VerificationThresholds,
            dict[str, Any],
            list[dict[str, Any]],
        ]
    ] = []
    for thresholds in threshold_grid():
        summary, rows = summarize_decisions(query_records, thresholds=thresholds)
        objective = (
            int(summary["hard_negative_accept_count"]),
            int(summary["catastrophic_false_accept_over_200m_count"]),
            -int(summary["accepted_within_10m_count"])
            - int(summary["accepted_ambiguous_10m_to_200m_count"]),
            -int(summary["accepted_within_10m_count"]),
            int(summary["rejection_count"]),
            abs(thresholds.minimum_inlier_ratio - 0.5),
            -thresholds.minimum_inlier_count,
            -thresholds.minimum_coverage_fraction,
            thresholds.maximum_reprojection_p95_px,
            thresholds.maximum_stability_p95_corner_shift_px,
            -thresholds.minimum_winner_to_runner_up_inlier_ratio,
        )
        evaluated.append((objective, thresholds, summary, rows))
    _, thresholds, summary, rows = min(evaluated, key=lambda item: item[0])
    return thresholds, summary, rows


def gate_result(summary: dict[str, Any]) -> dict[str, Any]:
    """Сопоставляет метрики только с заранее записанным held-out-гейтом."""

    checks = {
        "enough_evaluable_queries": (
            summary["evaluable_query_count"] >= GATE_MINIMUM_EVALUABLE_QUERIES
        ),
        "recall_at_10": bool(
            summary["recall_at_10"] is not None
            and summary["recall_at_10"] >= GATE_MINIMUM_RECALL_AT_10
        ),
        "accepted_within_200m_rate": (
            summary["accepted_within_200m_rate"]
            >= GATE_MINIMUM_ACCEPT_WITHIN_200M_RATE
        ),
        "catastrophic_false_accepts": (
            summary["catastrophic_false_accept_over_200m_count"]
            <= GATE_MAXIMUM_CATASTROPHIC_FALSE_ACCEPTS
        ),
        "enough_hard_negative_cases": (
            summary["hard_negative_case_count"]
            >= GATE_MINIMUM_HARD_NEGATIVE_CASES
        ),
        "hard_negative_accepts": (
            summary["hard_negative_accept_count"]
            <= GATE_MAXIMUM_HARD_NEGATIVE_ACCEPTS
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Сохраняет таблицу аудита с устойчивым порядком столбцов."""

    if not rows:
        raise ValueError("Нельзя записать пустую таблицу")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Выполняет поиск, геометрическую проверку и независимую RTK-оценку."""

    args = parse_args()
    if args.mode == "develop" and args.repeat_sequence != DEVELOPMENT_SEQUENCE:
        raise ValueError("develop разрешён только для HKairport02")
    if args.mode == "validate" and args.repeat_sequence != HELD_OUT_SEQUENCE:
        raise ValueError("validate разрешён только для HKairport03")
    if args.mode == "validate" and FROZEN_THRESHOLDS is None:
        raise RuntimeError(
            "Held-out запуск запрещён: development-пороги ещё не зафиксированы"
        )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    projected_crs = str(manifest["projected_crs"])
    intrinsics, distortion, preparation_signature = camera_from_manifest(manifest)
    teach_directory = DATA_ROOT / TEACH_SEQUENCE
    repeat_directory = DATA_ROOT / args.repeat_sequence
    for directory in (teach_directory, repeat_directory):
        if not (directory / "images").is_dir():
            raise FileNotFoundError(f"Нет извлечённых изображений: {directory}")

    teach_store = PreparedImageStore(
        teach_directory,
        CACHE_ROOT / "prepared" / preparation_signature / TEACH_SEQUENCE,
        intrinsics,
        distortion,
    )
    repeat_store = PreparedImageStore(
        repeat_directory,
        CACHE_ROOT / "prepared" / preparation_signature / args.repeat_sequence,
        intrinsics,
        distortion,
    )

    model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    verify_model_weights(model_manifest)
    extractor = DinoV2SmallPatchExtractor(
        repository=model_manifest["source"]["torch_hub_repository"],
        revision=model_manifest["source"]["repository_revision"],
        model_name=model_manifest["model"]["torch_hub_name"],
        torch_home=TORCH_HOME,
        landscape_size_pixels=DINO_INPUT_SIZE_PIXELS,
    )
    verify_model_weights(model_manifest)

    teach = load_positioned_images(teach_directory, projected_crs=projected_crs)
    repeat = load_positioned_images(repeat_directory, projected_crs=projected_crs)
    teach_indices = select_spaced_image_indices(
        teach,
        minimum_distance_meters=KEYFRAME_SPACING_METERS,
    )
    repeat_indices = select_spaced_image_indices(
        repeat,
        minimum_distance_meters=KEYFRAME_SPACING_METERS,
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
            teach_store,
            cache_signature=preparation_signature,
            sequence_name=TEACH_SEQUENCE,
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
        zip(repeat_ids, repeat_xy, strict=True),
        start=1,
    ):
        query_patches, elapsed, from_cache = load_or_extract_patches(
            extractor,
            repeat_store,
            cache_signature=preparation_signature,
            sequence_name=args.repeat_sequence,
            image_id=int(repeat_id),
        )
        extraction_seconds.append(elapsed)
        cache_hit_count += int(from_cache)
        query_descriptor = aggregate_vlad(query_patches, vocabulary)
        retrieval = rank_database(
            database_descriptors,
            query_descriptor,
            maximum_results=teach_ids.size,
        )
        distances = np.linalg.norm(teach_xy - query_xy, axis=1)
        normal_indices = retrieval.database_indices[:TOP_K]
        far_indices = np.asarray(
            [
                index
                for index in retrieval.database_indices
                if distances[index] > CATASTROPHIC_DISTANCE_METERS
            ][:TOP_K],
            dtype=np.int64,
        )
        rank_lookup = {
            int(database_index): rank
            for rank, database_index in enumerate(
                retrieval.database_indices,
                start=1,
            )
        }
        similarity_lookup = {
            int(database_index): float(similarity)
            for database_index, similarity in zip(
                retrieval.database_indices,
                retrieval.similarities,
                strict=True,
            )
        }
        evidence_by_index: dict[int, CandidateGeometry] = {}
        ordered_unique_indices = list(
            dict.fromkeys([*normal_indices.tolist(), *far_indices.tolist()])
        )
        for database_index in ordered_unique_indices:
            candidate = geometry_evidence(
                teach_store=teach_store,
                repeat_store=repeat_store,
                teach_image_id=int(teach_ids[database_index]),
                repeat_image_id=int(repeat_id),
                database_index=int(database_index),
                retrieval_rank=rank_lookup[database_index],
                retrieval_similarity=similarity_lookup[database_index],
            )
            evidence_by_index[database_index] = candidate
            candidate_rows.append(
                {
                    "repeat_image_id": int(repeat_id),
                    "teach_image_id": int(teach_ids[database_index]),
                    "telemetry_distance_meters": float(distances[database_index]),
                    "in_normal_top_10": database_index in normal_indices,
                    "in_hard_negative_top_10": database_index in far_indices,
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
                    np.any(distances[normal_indices] <= POSITIVE_RADIUS_METERS)
                ),
                "candidates": [
                    evidence_by_index[int(index)] for index in normal_indices
                ],
                "hard_negative_candidates": [
                    evidence_by_index[int(index)] for index in far_indices
                ],
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
        assert FROZEN_THRESHOLDS is not None
        thresholds = FROZEN_THRESHOLDS
        decision_summary, decision_rows = summarize_decisions(
            query_records,
            thresholds=thresholds,
        )
        status = "held_out_validation"

    output_prefix = args.output_prefix.resolve()
    candidates_path = output_prefix.with_name(output_prefix.name + "_candidates.csv")
    decisions_path = output_prefix.with_name(output_prefix.name + "_decisions.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    write_csv(candidates_path, candidate_rows)
    write_csv(decisions_path, decision_rows)

    noncached_times = [value for value in extraction_seconds if value > 0.0]
    reported_gate = (
        gate_result(decision_summary)
        if args.mode == "validate"
        else {
            "passed": None,
            "checks": {},
            "reason": "development не является held-out проверкой",
        }
    )

    summary = {
        "experiment": "mars_lvig_end_to_end_gate",
        "status": status,
        "mode": args.mode,
        "teach_sequence": TEACH_SEQUENCE,
        "repeat_sequence": args.repeat_sequence,
        "configuration": {
            "projected_crs": projected_crs,
            "keyframe_spacing_meters": KEYFRAME_SPACING_METERS,
            "positive_radius_meters": POSITIVE_RADIUS_METERS,
            "catastrophic_distance_meters": CATASTROPHIC_DISTANCE_METERS,
            "prepared_long_side_pixels": PREPARED_LONG_SIDE_PIXELS,
            "dinov2_input_size_pixels": DINO_INPUT_SIZE_PIXELS,
            "top_k": TOP_K,
            "vlad_cluster_count": CLUSTER_COUNT,
            "maximum_vocabulary_training_descriptors": (
                MAXIMUM_VOCABULARY_DESCRIPTORS
            ),
            "random_seed": RANDOM_SEED,
            "sift_ransac": asdict(SiftRansacConfig()),
            "verification_thresholds": asdict(thresholds),
        },
        "gate_criteria": {
            "minimum_evaluable_queries": GATE_MINIMUM_EVALUABLE_QUERIES,
            "minimum_recall_at_10": GATE_MINIMUM_RECALL_AT_10,
            "minimum_accepted_within_200m_rate": (
                GATE_MINIMUM_ACCEPT_WITHIN_200M_RATE
            ),
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
        "gate": reported_gate,
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
            "RTK-позиция антенны не является полной шестимерной позой камеры.",
            "Дистанция между камерами не доказывает точное перекрытие кадров.",
            "Результат проверяет поиск места, но не точность наземного измерения.",
            "Глобальный затвор MARS-LVIG не воспроизводит rolling-shutter камеры.",
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
