#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""G12--G13: совместная оценка позы, изображения и класса текстуры."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from aerial_mapper.alignment import AlignmentFailure, align_frame_to_reference
from aerial_mapper.interaction_evaluation import (
    classification_limit_failures,
    summarize_product_gate,
)
from aerial_mapper.measurement import MeasurementFailure
from aerial_mapper.metric_recovery import (
    MetricRecoveryThresholds,
    alignment_gate_failures,
)
from aerial_mapper.perspective_metric import (
    PerspectiveReferenceCalibration,
    calibrate_perspective_reference,
    map_frame_points_via_perspective_reference,
)
from aerial_mapper.quality import analyze_alignment_quality
from aerial_mapper.relative_measurement import (
    displacement_error,
    target_centered_displacement,
)
from aerial_mapper.synthetic_robustness import (
    ImageDegradation,
    apply_image_degradation,
)
from aerial_mapper.teach_annotation import (
    add_contaminant_fraction,
    enclosed_non_ground,
    mask_agreement,
    offset_mask_boundary,
)
from aerial_mapper.terrain_evaluation import (
    point_error_summary,
    select_displacement_pairs,
    select_grid_anchor_indices,
    spatial_group_labels,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "experiments/configs/synthetic_3d_g12_pose_interactions.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/synthetic_3d/g12_pose_interactions"


def parse_arguments() -> argparse.Namespace:
    """Читает только пути; все научные параметры хранятся в JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def project_path(value: str) -> Path:
    """Разрешает путь протокола относительно корня проекта."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    """Вычисляет отпечаток фактически использованной конфигурации."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    """Загружает RGB uint8 и явно сообщает об отсутствующем артефакте."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать RGB: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    """Загружает контрольную маску как двоичный uint8."""

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Не удалось прочитать маску: {path}")
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def load_truth(scene_directory: Path) -> dict[str, np.ndarray]:
    """Читает плотную скрытую истину Blender для одной поверхности."""

    metadata = json.loads(
        (scene_directory / "generation_metadata.json").read_text(encoding="utf-8")
    )
    with np.load(scene_directory / metadata["terrain_truth"]["path"]) as source:
        return {
            "world_xyz_m": np.asarray(source["world_xyz_m"], dtype=np.float64),
            "camera_ids": np.asarray(source["camera_ids"]),
            "pixel_xy": np.asarray(source["pixel_xy"], dtype=np.float64),
            "visible": np.asarray(source["visible"], dtype=bool),
        }


def camera_index(camera_ids: np.ndarray, camera_id: str) -> int:
    """Возвращает индекс единственной камеры с указанным id."""

    matches = np.flatnonzero(camera_ids == camera_id)
    if matches.size != 1:
        raise RuntimeError(f"Не найдена единственная камера {camera_id}")
    return int(matches[0])


def degradation_from_case(case: dict[str, Any]) -> ImageDegradation:
    """Берёт из case только поля существующей модели искажений G4--G6."""

    names = {field.name for field in fields(ImageDegradation)}
    return ImageDegradation(
        **{name: float(case[name]) for name in names if name in case}
    )


def feature_mask_for_case(
    case: dict[str, Any],
    *,
    safe_mask: np.ndarray,
    contaminant_mask: np.ndarray,
) -> np.ndarray:
    """Строит безопасную либо слегка загрязнённую Teach-маску."""

    fraction = float(case.get("mask_contamination_fraction", 0.0))
    if fraction == 0.0:
        return safe_mask.copy()
    return add_contaminant_fraction(
        safe_mask,
        contaminant_mask,
        added_fraction=fraction,
    )


def build_map_calibration(
    *,
    protocol: dict[str, Any],
    experiment: dict[str, Any],
    truth: dict[str, np.ndarray],
) -> tuple[PerspectiveReferenceCalibration, np.ndarray, np.ndarray]:
    """Имитирует перенос масштаба карты по 12 разнесённым Teach-реперам."""

    world_xyz = truth["world_xyz_m"]
    camera_ids = truth["camera_ids"]
    pixels = truth["pixel_xy"]
    visible = truth["visible"]
    groups = spatial_group_labels(
        world_xyz[:, :2],
        group_count=int(experiment["point_group_count"]),
    )
    teach = next(
        camera for camera in protocol["cameras"] if camera["role"] == "reference"
    )
    teach_index = camera_index(camera_ids, teach["id"])
    candidates = np.flatnonzero(visible[teach_index] & (groups == 0))
    anchor_grid = experiment["map_anchor_grid"]
    local_indices = select_grid_anchor_indices(
        world_xyz[candidates, :2],
        columns=int(anchor_grid["columns"]),
        rows=int(anchor_grid["rows"]),
        inset_fraction=float(anchor_grid["inset_fraction"]),
    )
    anchor_indices = candidates[local_indices]
    calibration = calibrate_perspective_reference(
        pixels[teach_index, anchor_indices],
        world_xyz[anchor_indices, :2],
    )
    return calibration, groups, anchor_indices


def estimate_alignment(
    *,
    teach_rgb: np.ndarray,
    repeat_rgb: np.ndarray,
    feature_mask: np.ndarray,
    thresholds: MetricRecoveryThresholds,
    random_seed: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Строит рабочую гомографию и сохраняет причины любого отказа gate."""

    try:
        alignment = align_frame_to_reference(
            teach_rgb,
            repeat_rgb,
            reference_feature_mask=feature_mask,
        )
        quality = analyze_alignment_quality(
            alignment,
            frame_width_pixels=repeat_rgb.shape[1],
            frame_height_pixels=repeat_rgb.shape[0],
            random_seed=random_seed,
        )
        failures = alignment_gate_failures(
            alignment,
            quality,
            thresholds=thresholds,
        )
        report = {
            "alignment_constructed": True,
            "gate_accepted": not failures,
            "gate_failures": list(failures),
            "ratio_matches": alignment.ratio_match_count,
            "inliers": alignment.inlier_count,
            "inlier_ratio": alignment.inlier_ratio,
            "coverage_fraction": alignment.inlier_spatial_coverage_fraction,
            "reprojection_p95_reference_px": (
                quality.inlier_reprojection_p95_reference_px
            ),
            "stability_p95_corner_shift_reference_px": (
                quality.stability_p95_max_corner_shift_reference_px
            ),
        }
        return alignment.homography_frame_to_reference, report
    except AlignmentFailure as error:
        return None, {
            "alignment_constructed": False,
            "gate_accepted": False,
            "gate_failures": [str(error)],
            "ratio_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "coverage_fraction": 0.0,
            "reprojection_p95_reference_px": None,
            "stability_p95_corner_shift_reference_px": None,
        }


def absolute_point_metric(
    *,
    homography: np.ndarray,
    camera_index_value: int,
    truth: dict[str, np.ndarray],
    groups: np.ndarray,
    calibration: PerspectiveReferenceCalibration,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Измеряет абсолютную X,Y-ошибку на независимой видимой группе земли."""

    visible = truth["visible"]
    teach_index = camera_index(
        truth["camera_ids"],
        next(
            camera["id"]
            for camera in protocol["cameras"]
            if camera["role"] == "reference"
        ),
    )
    indices = np.flatnonzero(
        visible[teach_index] & visible[camera_index_value] & (groups == 2)
    )
    if indices.size < 8:
        raise RuntimeError("Недостаточно независимых точек для абсолютной оценки")
    render = protocol["render"]
    estimated = map_frame_points_via_perspective_reference(
        truth["pixel_xy"][camera_index_value, indices],
        homography,
        calibration=calibration,
        reference_width_pixels=int(render["width_pixels"]),
        reference_height_pixels=int(render["height_pixels"]),
    ).world_points_m
    summary = point_error_summary(
        estimated,
        truth["world_xyz_m"][indices, :2],
    )
    limit = float(protocol["acceptance_thresholds"]["maximum_point_position_error_m"])
    return {
        **asdict(summary),
        "within_point_limit": summary.maximum_m <= limit,
        "point_limit_m": limit,
    }


def select_pairs_for_repeat(
    *,
    repeat_index: int,
    target_index: int,
    teach_index: int,
    truth: dict[str, np.ndarray],
    groups: np.ndarray,
    experiment: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Выбирает пары и отдельно сообщает о недостаточной контрольной выборке."""

    visible = truth["visible"]
    candidates = np.flatnonzero(
        visible[teach_index]
        & visible[target_index]
        & visible[repeat_index]
        & (groups >= 2)
    )
    pairs: dict[str, np.ndarray] = {}
    shortfalls: list[dict[str, Any]] = []
    minimum = int(experiment["minimum_pairs_per_vector"])
    for vector in experiment["displacement_vectors_xy_m"]:
        try:
            local_pairs = select_displacement_pairs(
                truth["world_xyz_m"][candidates, :2],
                np.asarray(vector["delta_xy_m"], dtype=np.float64),
                maximum_count=int(experiment["maximum_pairs_per_vector"]),
                tolerance_m=float(experiment["pair_vector_tolerance_m"]),
            )
            selected = candidates[local_pairs]
        except ValueError:
            selected = np.empty((0, 2), dtype=np.int64)
        if selected.shape[0] < minimum:
            shortfalls.append(
                {
                    "vector_id": vector["id"],
                    "pair_count": int(selected.shape[0]),
                    "required_pair_count": minimum,
                }
            )
        if selected.shape[0] > 0:
            pairs[vector["id"]] = selected
    return pairs, shortfalls


def evaluate_vector_pairs(
    *,
    surface_id: str,
    repeat_id: str,
    case_id: str,
    pipeline: str,
    gate_accepted: bool,
    pair_indices: np.ndarray,
    vector_id: str,
    truth: dict[str, np.ndarray],
    target_camera_index: int,
    impact_camera_index: int,
    target_homography: np.ndarray,
    impact_homography: np.ndarray,
    calibration: PerspectiveReferenceCalibration,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Переносит цель и попадание в метры для одной комбинации факторов."""

    render = protocol["render"]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for pair_number, (target_truth_index, impact_truth_index) in enumerate(
        pair_indices
    ):
        try:
            estimated_target = map_frame_points_via_perspective_reference(
                truth["pixel_xy"][
                    target_camera_index,
                    [target_truth_index],
                ],
                target_homography,
                calibration=calibration,
                reference_width_pixels=int(render["width_pixels"]),
                reference_height_pixels=int(render["height_pixels"]),
            ).world_points_m[0]
            estimated_impact = map_frame_points_via_perspective_reference(
                truth["pixel_xy"][
                    impact_camera_index,
                    [impact_truth_index],
                ],
                impact_homography,
                calibration=calibration,
                reference_width_pixels=int(render["width_pixels"]),
                reference_height_pixels=int(render["height_pixels"]),
            ).world_points_m[0]
        except MeasurementFailure as error:
            failures.append(
                {
                    "surface_id": surface_id,
                    "repeat_id": repeat_id,
                    "case_id": case_id,
                    "pipeline": pipeline,
                    "vector_id": vector_id,
                    "pair_number": pair_number,
                    "reason": str(error),
                }
            )
            continue

        true_target = truth["world_xyz_m"][target_truth_index, :2]
        true_impact = truth["world_xyz_m"][impact_truth_index, :2]
        true_displacement = target_centered_displacement(true_target, true_impact)
        estimated_displacement = target_centered_displacement(
            estimated_target,
            estimated_impact,
        )
        error = displacement_error(estimated_displacement, true_displacement)
        rows.append(
            {
                "surface_id": surface_id,
                "repeat_id": repeat_id,
                "case_id": case_id,
                "pipeline": pipeline,
                "vector_id": vector_id,
                "pair_number": pair_number,
                "gate_accepted": gate_accepted,
                "truth": asdict(true_displacement),
                "estimated": asdict(estimated_displacement),
                "error": asdict(error),
            }
        )
    return rows, failures


def summarize_vector_rows(
    rows: list[dict[str, Any]],
    *,
    maximum_vector_error_p95_m: float,
) -> list[dict[str, Any]]:
    """Агрегирует продуктовую ошибку отдельно для каждой factor-комбинации."""

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["surface_id"],
            row["repeat_id"],
            row["case_id"],
            row["pipeline"],
        )
        grouped[key].append(row)

    summaries: list[dict[str, Any]] = []
    for key, selected in sorted(grouped.items()):
        gate_values = {bool(row["gate_accepted"]) for row in selected}
        if len(gate_values) != 1:
            raise RuntimeError("Внутри группы изменилось решение gate")
        summary = summarize_product_gate(
            (row["error"]["vector_error_m"] for row in selected),
            x_sign_correct=(
                row["error"]["x_sign_correct"] for row in selected
            ),
            y_sign_correct=(
                row["error"]["y_sign_correct"] for row in selected
            ),
            gate_accepted=gate_values.pop(),
            maximum_vector_error_p95_m=maximum_vector_error_p95_m,
        )
        summaries.append(
            {
                "surface_id": key[0],
                "repeat_id": key[1],
                "case_id": key[2],
                "pipeline": key[3],
                **asdict(summary),
            }
        )
    return summaries


def classification_counts(
    rows: list[dict[str, Any]],
    *,
    field: str = "classification",
) -> dict[str, int]:
    """Считает все исходы, включая нулевые, в стабильном порядке."""

    labels = (
        "accepted_correct",
        "false_accept",
        "rejected_valid",
        "rejected_invalid",
        "alignment_failure",
    )
    return {label: sum(row.get(field) == label for row in rows) for label in labels}


def main() -> None:
    """Выполняет зафиксированные взаимодействия без повторного рендера."""

    arguments = parse_arguments()
    experiment = json.loads(arguments.config.read_text(encoding="utf-8"))
    source_protocol_path = project_path(experiment["source_protocol"])
    source_output = project_path(experiment["source_output"])
    protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    arguments.output.mkdir(parents=True, exist_ok=True)

    thresholds = MetricRecoveryThresholds.from_mapping(
        protocol["acceptance_thresholds"]
    )
    product_limit = float(
        experiment["product_thresholds"]["maximum_vector_error_p95_m"]
    )
    teach_id = next(
        camera["id"]
        for camera in protocol["cameras"]
        if camera["role"] == "reference"
    )
    repeat_ids = [
        camera["id"] for camera in protocol["cameras"] if camera["role"] == "repeat"
    ]
    target_id = experiment["target_frame_camera_id"]

    alignment_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    measurement_failures: list[dict[str, Any]] = []
    calibration_reports: dict[str, Any] = {}
    pair_reports: dict[str, Any] = {}
    pair_shortfalls: list[dict[str, Any]] = []
    mask_reports: dict[str, Any] = {}
    surface_metadata: dict[str, dict[str, Any]] = {}

    for surface_number, surface in enumerate(protocol["surfaces"]):
        surface_id = surface["id"]
        texture_class = str(surface.get("texture_class", "rich_irregular"))
        surface_metadata[surface_id] = {
            "scene_seed": int(surface.get("scene_seed", protocol["seed"])),
            "texture_class": texture_class,
        }
        scene_directory = source_output / "scenes" / surface_id
        truth = load_truth(scene_directory)
        calibration, groups, anchor_indices = build_map_calibration(
            protocol=protocol,
            experiment=experiment,
            truth=truth,
        )
        calibration_reports[surface_id] = {
            "anchor_count": int(anchor_indices.size),
            "control_reprojection_rmse_m": calibration.control_reprojection_rmse_m,
            "control_reprojection_max_m": calibration.control_reprojection_max_m,
        }

        teach_index = camera_index(truth["camera_ids"], teach_id)
        target_index = camera_index(truth["camera_ids"], target_id)
        teach_rgb = load_rgb(scene_directory / "reference/perspective_rgb.png")
        true_ground_mask = load_mask(scene_directory / "reference/ground_mask.png")
        safe_mask = offset_mask_boundary(
            true_ground_mask,
            offset_pixels=-int(experiment["safe_mask_erosion_px"]),
        )
        contaminant_mask = enclosed_non_ground(true_ground_mask)

        mask_cache: dict[float, np.ndarray] = {}
        target_cache: dict[float, tuple[np.ndarray | None, dict[str, Any]]] = {}
        for case in experiment["cases"]:
            fraction = float(case.get("mask_contamination_fraction", 0.0))
            if fraction in mask_cache:
                continue
            feature_mask = feature_mask_for_case(
                case,
                safe_mask=safe_mask,
                contaminant_mask=contaminant_mask,
            )
            mask_cache[fraction] = feature_mask
            mask_reports[f"{surface_id}:{fraction:.3f}"] = {
                "surface_id": surface_id,
                "mask_contamination_fraction": fraction,
                **asdict(mask_agreement(feature_mask, true_ground_mask)),
            }
            target_rgb = load_rgb(
                scene_directory / "repeat" / target_id / "rgb.png"
            )
            target_cache[fraction] = estimate_alignment(
                teach_rgb=teach_rgb,
                repeat_rgb=target_rgb,
                feature_mask=feature_mask,
                thresholds=thresholds,
                random_seed=int(experiment["seed"]) + surface_number * 10_000,
            )

        surface_pair_report: dict[str, Any] = {}
        for repeat_number, repeat_id in enumerate(repeat_ids):
            repeat_index = camera_index(truth["camera_ids"], repeat_id)
            selected_pairs, shortfalls = select_pairs_for_repeat(
                repeat_index=repeat_index,
                target_index=target_index,
                teach_index=teach_index,
                truth=truth,
                groups=groups,
                experiment=experiment,
            )
            pair_shortfalls.extend(
                {
                    "surface_id": surface_id,
                    "repeat_id": repeat_id,
                    **item,
                }
                for item in shortfalls
            )
            surface_pair_report[repeat_id] = {
                vector_id: int(pairs.shape[0])
                for vector_id, pairs in selected_pairs.items()
            }
            nominal_repeat = load_rgb(
                scene_directory / "repeat" / repeat_id / "rgb.png"
            )

            for case_number, case in enumerate(experiment["cases"]):
                case_id = case["id"]
                fraction = float(case.get("mask_contamination_fraction", 0.0))
                degradation = degradation_from_case(case)
                degraded_repeat = apply_image_degradation(
                    nominal_repeat,
                    degradation,
                    random_seed=(
                        int(experiment["seed"])
                        + surface_number * 100_000
                        + repeat_number * 1000
                        + case_number
                    ),
                )
                impact_h, impact_report = estimate_alignment(
                    teach_rgb=teach_rgb,
                    repeat_rgb=degraded_repeat,
                    feature_mask=mask_cache[fraction],
                    thresholds=thresholds,
                    random_seed=(
                        int(experiment["seed"])
                        + surface_number * 100_000
                        + repeat_number * 1000
                        + case_number
                    ),
                )
                target_h, target_report = target_cache[fraction]

                absolute_metric = None
                absolute_measurement_failure = None
                absolute_within = False
                if impact_h is not None:
                    try:
                        absolute_metric = absolute_point_metric(
                            homography=impact_h,
                            camera_index_value=repeat_index,
                            truth=truth,
                            groups=groups,
                            calibration=calibration,
                            protocol=protocol,
                        )
                        absolute_within = bool(
                            absolute_metric["within_point_limit"]
                        )
                    except MeasurementFailure as error:
                        # Выход контрольной точки за Teach — тоже метрический
                        # провал. Gate обязан отклонить такую гомографию.
                        absolute_measurement_failure = str(error)

                if impact_h is None:
                    absolute_classification = "alignment_failure"
                elif impact_report["gate_accepted"] and absolute_within:
                    absolute_classification = "accepted_correct"
                elif impact_report["gate_accepted"]:
                    absolute_classification = "false_accept"
                elif absolute_within:
                    absolute_classification = "rejected_valid"
                else:
                    absolute_classification = "rejected_invalid"

                alignment_rows.append(
                    {
                        "surface_id": surface_id,
                        "scene_seed": int(
                            surface.get("scene_seed", protocol["seed"])
                        ),
                        "texture_class": texture_class,
                        "repeat_id": repeat_id,
                        "case_id": case_id,
                        "mask_contamination_fraction": fraction,
                        "degradation": asdict(degradation),
                        "impact_alignment": impact_report,
                        "target_alignment": target_report,
                        "absolute_point_metric": absolute_metric,
                        "absolute_measurement_failure": (
                            absolute_measurement_failure
                        ),
                        "absolute_classification": absolute_classification,
                    }
                )

                if impact_h is None:
                    continue
                for vector_id, pairs in selected_pairs.items():
                    rows, failures = evaluate_vector_pairs(
                        surface_id=surface_id,
                        repeat_id=repeat_id,
                        case_id=case_id,
                        pipeline="same_frame",
                        gate_accepted=bool(impact_report["gate_accepted"]),
                        pair_indices=pairs,
                        vector_id=vector_id,
                        truth=truth,
                        target_camera_index=repeat_index,
                        impact_camera_index=repeat_index,
                        target_homography=impact_h,
                        impact_homography=impact_h,
                        calibration=calibration,
                        protocol=protocol,
                    )
                    vector_rows.extend(rows)
                    measurement_failures.extend(failures)

                    if target_h is not None:
                        rows, failures = evaluate_vector_pairs(
                            surface_id=surface_id,
                            repeat_id=repeat_id,
                            case_id=case_id,
                            pipeline="cross_frame",
                            gate_accepted=bool(
                                impact_report["gate_accepted"]
                                and target_report["gate_accepted"]
                            ),
                            pair_indices=pairs,
                            vector_id=vector_id,
                            truth=truth,
                            target_camera_index=target_index,
                            impact_camera_index=repeat_index,
                            target_homography=target_h,
                            impact_homography=impact_h,
                            calibration=calibration,
                            protocol=protocol,
                        )
                        vector_rows.extend(rows)
                        measurement_failures.extend(failures)

        pair_reports[surface_id] = surface_pair_report
        print(f"completed_surface={surface_id}")

    vector_summaries = summarize_vector_rows(
        vector_rows,
        maximum_vector_error_p95_m=product_limit,
    )
    for row in vector_summaries:
        row["texture_class"] = surface_metadata[row["surface_id"]]["texture_class"]

    absolute_counts = classification_counts(
        alignment_rows,
        field="absolute_classification",
    )
    product_counts = {
        pipeline: classification_counts(
            [row for row in vector_summaries if row["pipeline"] == pipeline]
        )
        for pipeline in ("same_frame", "cross_frame")
    }
    # Если гомография не построена, строк векторной ошибки закономерно нет.
    # Такие безопасные отказы всё равно входят в полный счётчик 195 попыток.
    product_counts["same_frame"]["alignment_failure"] = sum(
        not row["impact_alignment"]["alignment_constructed"]
        for row in alignment_rows
    )
    product_counts["cross_frame"]["alignment_failure"] = sum(
        not row["impact_alignment"]["alignment_constructed"]
        or not row["target_alignment"]["alignment_constructed"]
        for row in alignment_rows
    )
    texture_class_counts = {
        texture_class: classification_counts(
            [
                row
                for row in vector_summaries
                if row["pipeline"] == "cross_frame"
                and row["texture_class"] == texture_class
            ]
        )
        for texture_class in sorted(
            {metadata["texture_class"] for metadata in surface_metadata.values()}
        )
    }
    for texture_class, counts in texture_class_counts.items():
        counts["alignment_failure"] = sum(
            row["texture_class"] == texture_class
            and (
                not row["impact_alignment"]["alignment_constructed"]
                or not row["target_alignment"]["alignment_constructed"]
            )
            for row in alignment_rows
        )
    class_success_failures: list[str] = []
    for texture_class, criteria in experiment.get(
        "texture_class_success",
        {},
    ).items():
        failures = classification_limit_failures(
            texture_class_counts.get(texture_class, {}),
            maximum_false_accepts=int(criteria["maximum_false_accepts"]),
            minimum_accepted_correct=int(
                criteria.get("minimum_accepted_correct", 0)
            ),
        )
        class_success_failures.extend(
            f"{texture_class}: {failure}" for failure in failures
        )

    accepted = [
        row for row in vector_summaries if row["classification"] == "accepted_correct"
    ]
    accepted_p95_max = max(
        (row["vector_error_p95_m"] for row in accepted),
        default=None,
    )
    accepted_sign_errors = sum(
        row["x_sign_error_count"] + row["y_sign_error_count"] for row in accepted
    )
    false_accept_count = sum(
        row["classification"] == "false_accept" for row in vector_summaries
    )
    observed_pair_counts = [
        count
        for surface in pair_reports.values()
        for repeat in surface.values()
        for count in repeat.values()
    ]
    observed_pair_counts.extend(
        item["pair_count"] for item in pair_shortfalls if item["pair_count"] == 0
    )
    minimum_pair_count = min(observed_pair_counts)
    success = experiment["preregistered_success"]
    maximum_absolute_false_accepts = success.get(
        "maximum_absolute_false_accepts"
    )
    preregistered_passed = bool(
        false_accept_count <= int(success["maximum_false_accepts"])
        and (
            maximum_absolute_false_accepts is None
            or absolute_counts["false_accept"]
            <= int(maximum_absolute_false_accepts)
        )
        and not class_success_failures
        and len(measurement_failures)
        <= int(success["maximum_measurement_failures"])
        and minimum_pair_count >= int(success["minimum_pairs_per_vector"])
        and (
            accepted_p95_max is None
            or accepted_p95_max
            <= float(success["maximum_accepted_vector_error_p95_m"])
        )
        and accepted_sign_errors <= int(success["maximum_accepted_sign_errors"])
    )

    report = {
        "schema_version": 1,
        "experiment": experiment["experiment"],
        "configuration": str(arguments.config),
        "configuration_sha256": sha256_file(arguments.config),
        "source_protocol": str(source_protocol_path),
        "source_protocol_sha256": sha256_file(source_protocol_path),
        "source_output": str(source_output),
        "rerendered_blender_scenes": False,
        "design": {
            "surface_count": len(protocol["surfaces"]),
            "repeat_pose_count": len(repeat_ids),
            "case_count": len(experiment["cases"]),
            "attempted_alignment_count": len(alignment_rows),
            "targeted_interactions_not_full_factorial": True,
            "primary_sources": experiment["primary_sources"],
        },
        "input_contract": {
            "working_algorithm_receives": [
                "Teach RGB",
                "operational Teach feature mask",
                "12 Teach pixels paired with local map XY metres",
                "degraded Repeat RGB",
            ],
            "evaluator_only": [
                "camera poses",
                "scene seed and clutter geometry",
                "exact ground mask and contaminant pixels",
                "dense world coordinates and visibility",
            ],
        },
        "frozen_alignment_thresholds": protocol["acceptance_thresholds"],
        "frozen_product_thresholds": experiment["product_thresholds"],
        "preregistered_success": success,
        "preregistered_passed": preregistered_passed,
        "surface_metadata": surface_metadata,
        "absolute_classification_counts": absolute_counts,
        "product_classification_counts": product_counts,
        "texture_class_classification_counts": texture_class_counts,
        "texture_class_success_failures": class_success_failures,
        "false_accept_count": false_accept_count,
        "accepted_vector_error_p95_max_m": accepted_p95_max,
        "accepted_sign_error_count": accepted_sign_errors,
        "minimum_pair_count": minimum_pair_count,
        "measurement_row_count": len(vector_rows),
        "measurement_failure_count": len(measurement_failures),
        "calibrations": calibration_reports,
        "masks": mask_reports,
        "pair_selection": pair_reports,
        "pair_selection_shortfalls": pair_shortfalls,
        "alignment_rows": alignment_rows,
        "vector_summaries": vector_summaries,
        "measurement_failures": measurement_failures,
        "limitations": [
            "Три seed принадлежат одной процедурной семье богатой текстуры.",
            "Мусор статичен между Teach и Repeat.",
            "Искажения применяются к готовому кадру и не моделируют rolling shutter.",
            "Картографические реперы и клики цели/попадания точны.",
            "Поверхность земли плоская.",
        ],
    }
    report_path = arguments.output / experiment.get(
        "report_filename",
        "g12_pose_interactions_report.json",
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(product_counts, ensure_ascii=False))
    print(f"preregistered_passed={preregistered_passed}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
