#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""G9-R: ошибка вектора от цели к попаданию на существующих сценах G9."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from aerial_mapper.alignment import AlignmentFailure, align_frame_to_reference
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
from aerial_mapper.terrain_evaluation import (
    estimate_homography,
    select_displacement_pairs,
    select_grid_anchor_indices,
    spatial_group_labels,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "experiments/configs/synthetic_3d_g9r_relative_displacement.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/synthetic_3d/g9r_relative_displacement"

PIPELINE_ORDER = (
    "teach_direct",
    "oracle_same_frame",
    "sift_same_frame",
    "oracle_cross_frame",
    "sift_cross_frame",
)


def parse_arguments() -> argparse.Namespace:
    """Читает пути конфигурации и результата без запуска Blender."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def project_path(value: str) -> Path:
    """Разрешает зафиксированный в JSON путь относительно корня проекта."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    """Вычисляет отпечаток фактически использованного входного файла."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    """Читает PNG как RGB uint8 и явно сообщает об отсутствующем артефакте."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать RGB: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    """Читает двоичную Teach-маску земли."""

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Не удалось прочитать маску: {path}")
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def camera_index(camera_ids: np.ndarray, camera_id: str) -> int:
    """Находит единственную камеру с заданным идентификатором."""

    matches = np.flatnonzero(camera_ids == camera_id)
    if matches.size != 1:
        raise RuntimeError(f"Не найдена единственная камера {camera_id}")
    return int(matches[0])


def load_truth(scene_directory: Path) -> dict[str, np.ndarray]:
    """Загружает плотную истину, созданную и проверенную генератором G9."""

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
        world_xyz[:, :2], group_count=int(experiment["point_group_count"])
    )
    teach = next(
        camera for camera in protocol["cameras"] if camera["role"] == "reference"
    )
    teach_index = camera_index(camera_ids, teach["id"])
    candidate_indices = np.flatnonzero(visible[teach_index] & (groups == 0))
    anchor_grid = experiment["map_anchor_grid"]
    local_indices = select_grid_anchor_indices(
        world_xyz[candidate_indices, :2],
        columns=int(anchor_grid["columns"]),
        rows=int(anchor_grid["rows"]),
        inset_fraction=float(anchor_grid["inset_fraction"]),
    )
    anchor_indices = candidate_indices[local_indices]
    calibration = calibrate_perspective_reference(
        pixels[teach_index, anchor_indices],
        world_xyz[anchor_indices, :2],
    )
    return calibration, groups, anchor_indices


def build_alignment_models(
    *,
    protocol: dict[str, Any],
    scene_directory: Path,
    truth: dict[str, np.ndarray],
    groups: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Готовит oracle и рабочее Repeat→Teach преобразования один раз на кадр."""

    camera_ids = truth["camera_ids"]
    pixels = truth["pixel_xy"]
    visible = truth["visible"]
    teach = next(
        camera for camera in protocol["cameras"] if camera["role"] == "reference"
    )
    repeats = [camera for camera in protocol["cameras"] if camera["role"] == "repeat"]
    teach_index = camera_index(camera_ids, teach["id"])
    teach_visible = visible[teach_index]
    teach_rgb = load_rgb(scene_directory / "reference/perspective_rgb.png")
    teach_mask = load_mask(scene_directory / "reference/ground_mask.png")
    erosion = int(protocol["teach_mask_erosion_px"])
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * erosion + 1, 2 * erosion + 1),
    )
    feature_mask = cv2.erode(teach_mask, kernel, iterations=1)
    thresholds = MetricRecoveryThresholds.from_mapping(
        protocol["acceptance_thresholds"]
    )

    models: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {}
    for repeat_number, repeat in enumerate(repeats):
        repeat_id = repeat["id"]
        repeat_index = camera_index(camera_ids, repeat_id)
        common = teach_visible & visible[repeat_index]
        fit_mask = common & (groups == 1)
        if np.count_nonzero(fit_mask) < 8:
            raise RuntimeError(f"Недостаточно oracle-точек для {repeat_id}")
        oracle_h = estimate_homography(
            pixels[repeat_index, fit_mask],
            pixels[teach_index, fit_mask],
        )

        repeat_rgb = load_rgb(scene_directory / "repeat" / repeat_id / "rgb.png")
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
                random_seed=int(protocol["seed"]) + repeat_number,
            )
            failures = alignment_gate_failures(
                alignment,
                quality,
                thresholds=thresholds,
            )
            sift_h = alignment.homography_frame_to_reference if not failures else None
            sift_report: dict[str, Any] = {
                "accepted": not failures,
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
        except AlignmentFailure as error:
            sift_h = None
            sift_report = {
                "accepted": False,
                "gate_failures": [str(error)],
                "ratio_matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "coverage_fraction": 0.0,
                "reprojection_p95_reference_px": None,
                "stability_p95_corner_shift_reference_px": None,
            }

        models[repeat_id] = {
            "camera_index": repeat_index,
            "oracle_h": oracle_h,
            "sift_h": sift_h,
        }
        report[repeat_id] = {
            "common_visible_points": int(np.count_nonzero(common)),
            "oracle_fit_points": int(np.count_nonzero(fit_mask)),
            "sift": sift_report,
        }
    return models, report


def map_one_point(
    pixel_xy: np.ndarray,
    homography_frame_to_teach: np.ndarray,
    *,
    calibration: PerspectiveReferenceCalibration,
    protocol: dict[str, Any],
) -> np.ndarray:
    """Переводит одну отмеченную точку кадра в локальные метры карты."""

    render = protocol["render"]
    mapped = map_frame_points_via_perspective_reference(
        np.asarray(pixel_xy, dtype=np.float64).reshape(1, 2),
        homography_frame_to_teach,
        calibration=calibration,
        reference_width_pixels=int(render["width_pixels"]),
        reference_height_pixels=int(render["height_pixels"]),
    )
    return mapped.world_points_m[0]


def evaluate_pairs(
    *,
    surface_id: str,
    pipeline: str,
    repeat_id: str,
    vector_id: str,
    pair_indices: np.ndarray,
    world_xyz: np.ndarray,
    pixels: np.ndarray,
    target_camera_index: int,
    impact_camera_index: int,
    target_homography: np.ndarray,
    impact_homography: np.ndarray,
    calibration: PerspectiveReferenceCalibration,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Оценивает один способ переноса на фиксированных парах цель/попадание."""

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for pair_number, (target_index, impact_index) in enumerate(pair_indices):
        true_target = world_xyz[target_index, :2]
        true_impact = world_xyz[impact_index, :2]
        try:
            estimated_target = map_one_point(
                pixels[target_camera_index, target_index],
                target_homography,
                calibration=calibration,
                protocol=protocol,
            )
            estimated_impact = map_one_point(
                pixels[impact_camera_index, impact_index],
                impact_homography,
                calibration=calibration,
                protocol=protocol,
            )
        except MeasurementFailure as error:
            failures.append(
                {
                    "surface_id": surface_id,
                    "pipeline": pipeline,
                    "repeat_id": repeat_id,
                    "vector_id": vector_id,
                    "pair_number": pair_number,
                    "reason": str(error),
                }
            )
            continue

        truth_displacement = target_centered_displacement(true_target, true_impact)
        estimated_displacement = target_centered_displacement(
            estimated_target,
            estimated_impact,
        )
        error = displacement_error(estimated_displacement, truth_displacement)
        target_error = float(np.linalg.norm(estimated_target - true_target))
        impact_error = float(np.linalg.norm(estimated_impact - true_impact))
        endpoint_maximum = max(target_error, impact_error)
        cancellation_ratio = (
            error.vector_error_m / endpoint_maximum if endpoint_maximum > 1e-12 else 0.0
        )
        rows.append(
            {
                "surface_id": surface_id,
                "pipeline": pipeline,
                "repeat_id": repeat_id,
                "vector_id": vector_id,
                "pair_number": pair_number,
                "target_truth_index": int(target_index),
                "impact_truth_index": int(impact_index),
                "truth": asdict(truth_displacement),
                "estimated": asdict(estimated_displacement),
                "error": asdict(error),
                "target_absolute_error_m": target_error,
                "impact_absolute_error_m": impact_error,
                "maximum_endpoint_absolute_error_m": endpoint_maximum,
                "cancellation_ratio": cancellation_ratio,
            }
        )
    return rows, failures


def summarize_rows(
    rows: list[dict[str, Any]],
    error_bands_m: list[float],
    *,
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Агрегирует ошибки без объявления произвольного проходного порога."""

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in group_fields)].append(row)

    summaries: list[dict[str, Any]] = []
    for key, selected in sorted(grouped.items()):
        vector_errors = np.asarray(
            [row["error"]["vector_error_m"] for row in selected],
            dtype=np.float64,
        )
        distance_errors = np.asarray(
            [row["error"]["absolute_distance_error_m"] for row in selected],
            dtype=np.float64,
        )
        endpoint_errors = np.asarray(
            [row["maximum_endpoint_absolute_error_m"] for row in selected],
            dtype=np.float64,
        )
        direction_errors = np.asarray(
            [
                row["error"]["direction_error_degrees"]
                for row in selected
                if row["error"]["direction_error_degrees"] is not None
            ],
            dtype=np.float64,
        )
        component_errors = np.asarray(
            [
                [
                    abs(row["error"]["error_x_m"]),
                    abs(row["error"]["error_y_m"]),
                ]
                for row in selected
            ],
            dtype=np.float64,
        )
        summary: dict[str, Any] = {
            field: value for field, value in zip(group_fields, key, strict=True)
        }
        summary.update(
            {
                "count": len(selected),
                "vector_error_median_m": float(np.median(vector_errors)),
                "vector_error_p95_m": float(np.percentile(vector_errors, 95)),
                "vector_error_maximum_m": float(np.max(vector_errors)),
                "maximum_component_error_p95_m": float(
                    np.percentile(np.max(component_errors, axis=1), 95)
                ),
                "distance_error_p95_m": float(np.percentile(distance_errors, 95)),
                "distance_error_maximum_m": float(np.max(distance_errors)),
                "direction_error_p95_degrees": (
                    float(np.percentile(direction_errors, 95))
                    if direction_errors.size
                    else None
                ),
                "endpoint_absolute_error_p95_m": float(
                    np.percentile(endpoint_errors, 95)
                ),
                "x_sign_error_count": sum(
                    row["error"]["x_sign_correct"] is False for row in selected
                ),
                "y_sign_error_count": sum(
                    row["error"]["y_sign_correct"] is False for row in selected
                ),
            }
        )
        for band in error_bands_m:
            label = str(band).replace(".", "_")
            summary[f"within_{label}m_fraction"] = float(np.mean(vector_errors <= band))
        summaries.append(summary)
    return summaries


def save_summary_plot(
    summaries: list[dict[str, Any]],
    surface_ids: list[str],
    output_path: Path,
) -> None:
    """Показывает p95 ошибки вектора по поверхности и пути измерения."""

    figure, axes = plt.subplots(
        len(PIPELINE_ORDER),
        1,
        figsize=(13, 3.2 * len(PIPELINE_ORDER)),
    )
    for axis, pipeline in zip(axes, PIPELINE_ORDER, strict=True):
        selected_pipeline = [row for row in summaries if row["pipeline"] == pipeline]
        repeat_ids = sorted({row["repeat_id"] for row in selected_pipeline})
        for repeat_id in repeat_ids:
            values = []
            for surface_id in surface_ids:
                match = next(
                    (
                        row
                        for row in selected_pipeline
                        if row["surface_id"] == surface_id
                        and row["repeat_id"] == repeat_id
                    ),
                    None,
                )
                values.append(
                    match["vector_error_p95_m"] if match is not None else np.nan
                )
            axis.plot(surface_ids, values, marker="o", label=repeat_id)
        for band, color in ((0.25, "green"), (0.5, "orange"), (1.0, "red")):
            axis.axhline(band, color=color, linestyle="--", alpha=0.45)
        axis.set_title(pipeline)
        axis.set_ylabel("p95 ошибки вектора, м")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(alpha=0.25)
        if repeat_ids:
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Переоценивает G9 по продуктовой метрике без повторного рендера."""

    arguments = parse_arguments()
    experiment = json.loads(arguments.config.read_text(encoding="utf-8"))
    source_protocol_path = project_path(experiment["source_protocol"])
    source_output = project_path(experiment["source_output"])
    protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    arguments.output.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    alignment_reports: dict[str, Any] = {}
    calibration_reports: dict[str, Any] = {}
    pair_reports: dict[str, Any] = {}
    target_frame_id = experiment["target_frame_camera_id"]
    surface_ids = [surface["id"] for surface in protocol["surfaces"]]

    for surface in protocol["surfaces"]:
        surface_id = surface["id"]
        scene_directory = source_output / "scenes" / surface_id
        truth = load_truth(scene_directory)
        world_xyz = truth["world_xyz_m"]
        pixels = truth["pixel_xy"]
        visible = truth["visible"]
        camera_ids = truth["camera_ids"]
        calibration, groups, anchor_indices = build_map_calibration(
            protocol=protocol,
            experiment=experiment,
            truth=truth,
        )
        models, alignment_report = build_alignment_models(
            protocol=protocol,
            scene_directory=scene_directory,
            truth=truth,
            groups=groups,
        )
        alignment_reports[surface_id] = alignment_report
        calibration_reports[surface_id] = {
            "anchor_count": int(anchor_indices.size),
            "anchor_truth_indices": anchor_indices.tolist(),
            "control_reprojection_rmse_m": (calibration.control_reprojection_rmse_m),
            "control_reprojection_max_m": calibration.control_reprojection_max_m,
        }

        all_visible = np.all(visible, axis=0)
        render = protocol["render"]
        width = int(render["width_pixels"])
        height = int(render["height_pixels"])
        inside_pixel_centres = np.all(
            (pixels[:, :, 0] >= 0.0)
            & (pixels[:, :, 0] <= width - 1.0)
            & (pixels[:, :, 1] >= 0.0)
            & (pixels[:, :, 1] <= height - 1.0),
            axis=0,
        )
        candidate_indices = np.flatnonzero(
            all_visible & inside_pixel_centres & (groups >= 2)
        )
        teach_id = next(
            camera["id"]
            for camera in protocol["cameras"]
            if camera["role"] == "reference"
        )
        teach_index = camera_index(camera_ids, teach_id)
        target_frame_index = camera_index(camera_ids, target_frame_id)
        target_model = models[target_frame_id]
        surface_pair_report: dict[str, Any] = {}

        for vector in experiment["displacement_vectors_xy_m"]:
            vector_id = vector["id"]
            local_pairs = select_displacement_pairs(
                world_xyz[candidate_indices, :2],
                np.asarray(vector["delta_xy_m"], dtype=np.float64),
                maximum_count=int(experiment["maximum_pairs_per_vector"]),
                tolerance_m=float(experiment["pair_vector_tolerance_m"]),
            )
            pair_indices = candidate_indices[local_pairs]
            actual_vectors = (
                world_xyz[pair_indices[:, 1], :2] - world_xyz[pair_indices[:, 0], :2]
            )
            surface_pair_report[vector_id] = {
                "pair_count": int(pair_indices.shape[0]),
                "requested_delta_xy_m": vector["delta_xy_m"],
                "maximum_selection_residual_m": float(
                    np.max(
                        np.linalg.norm(
                            actual_vectors
                            - np.asarray(vector["delta_xy_m"], dtype=np.float64),
                            axis=1,
                        )
                    )
                ),
            }

            rows, failures = evaluate_pairs(
                surface_id=surface_id,
                pipeline="teach_direct",
                repeat_id=teach_id,
                vector_id=vector_id,
                pair_indices=pair_indices,
                world_xyz=world_xyz,
                pixels=pixels,
                target_camera_index=teach_index,
                impact_camera_index=teach_index,
                target_homography=np.eye(3, dtype=np.float64),
                impact_homography=np.eye(3, dtype=np.float64),
                calibration=calibration,
                protocol=protocol,
            )
            all_rows.extend(rows)
            all_failures.extend(failures)

            for repeat_id, model in models.items():
                repeat_index = int(model["camera_index"])
                for pipeline, target_h, impact_h, target_camera, impact_camera in (
                    (
                        "oracle_same_frame",
                        model["oracle_h"],
                        model["oracle_h"],
                        repeat_index,
                        repeat_index,
                    ),
                    (
                        "oracle_cross_frame",
                        target_model["oracle_h"],
                        model["oracle_h"],
                        target_frame_index,
                        repeat_index,
                    ),
                ):
                    rows, failures = evaluate_pairs(
                        surface_id=surface_id,
                        pipeline=pipeline,
                        repeat_id=repeat_id,
                        vector_id=vector_id,
                        pair_indices=pair_indices,
                        world_xyz=world_xyz,
                        pixels=pixels,
                        target_camera_index=target_camera,
                        impact_camera_index=impact_camera,
                        target_homography=target_h,
                        impact_homography=impact_h,
                        calibration=calibration,
                        protocol=protocol,
                    )
                    all_rows.extend(rows)
                    all_failures.extend(failures)

                sift_h = model["sift_h"]
                target_sift_h = target_model["sift_h"]
                if sift_h is not None:
                    rows, failures = evaluate_pairs(
                        surface_id=surface_id,
                        pipeline="sift_same_frame",
                        repeat_id=repeat_id,
                        vector_id=vector_id,
                        pair_indices=pair_indices,
                        world_xyz=world_xyz,
                        pixels=pixels,
                        target_camera_index=repeat_index,
                        impact_camera_index=repeat_index,
                        target_homography=sift_h,
                        impact_homography=sift_h,
                        calibration=calibration,
                        protocol=protocol,
                    )
                    all_rows.extend(rows)
                    all_failures.extend(failures)
                if sift_h is not None and target_sift_h is not None:
                    rows, failures = evaluate_pairs(
                        surface_id=surface_id,
                        pipeline="sift_cross_frame",
                        repeat_id=repeat_id,
                        vector_id=vector_id,
                        pair_indices=pair_indices,
                        world_xyz=world_xyz,
                        pixels=pixels,
                        target_camera_index=target_frame_index,
                        impact_camera_index=repeat_index,
                        target_homography=target_sift_h,
                        impact_homography=sift_h,
                        calibration=calibration,
                        protocol=protocol,
                    )
                    all_rows.extend(rows)
                    all_failures.extend(failures)

        pair_reports[surface_id] = surface_pair_report
        print(f"completed_surface={surface_id}")

    error_bands = [float(value) for value in experiment["error_bands_m"]]
    summaries_by_surface = summarize_rows(
        all_rows,
        error_bands,
        group_fields=("pipeline", "surface_id", "repeat_id"),
    )
    summaries_by_vector = summarize_rows(
        all_rows,
        error_bands,
        group_fields=("pipeline", "surface_id", "repeat_id", "vector_id"),
    )
    plot_path = arguments.output / "g9r_relative_summary.png"
    save_summary_plot(summaries_by_surface, surface_ids, plot_path)

    report = {
        "schema_version": 1,
        "experiment": experiment["experiment"],
        "configuration": str(arguments.config),
        "configuration_sha256": sha256_file(arguments.config),
        "source_protocol": str(source_protocol_path),
        "source_protocol_sha256": sha256_file(source_protocol_path),
        "source_output": str(source_output),
        "rerendered_blender_scenes": False,
        "input_contract": {
            "working_algorithm_receives": [
                "Teach RGB and trusted ground mask",
                "12 Teach pixels paired with local map XY metres",
                "Repeat RGB",
                "exact operator clicks for target and impact in this first slice",
            ],
            "not_required": [
                "shooter position",
                "GPS",
                "camera pose",
                "object dimensions",
            ],
            "evaluator_only": [
                "dense Blender world coordinates",
                "exact camera projections and visibility",
                "five-way independent point-group assignment",
            ],
        },
        "error_bands_m": error_bands,
        "surface_count": len(surface_ids),
        "measurement_row_count": len(all_rows),
        "measurement_failure_count": len(all_failures),
        "calibrations": calibration_reports,
        "alignments": alignment_reports,
        "pair_selection": pair_reports,
        "summaries_by_surface": summaries_by_surface,
        "summaries_by_vector": summaries_by_vector,
        "measurement_failures": all_failures,
        "rows": all_rows,
        "visualization": str(plot_path),
        "limitations": [
            "Координаты картографических реперов и клики пока точны.",
            "Сетка истины имеет шаг 4 м по X и 1 м по Y.",
            "Бумажная карта представлена координатами реперов, а не её изображением.",
            "Положение стрелка и поворот в стрелковые оси не проверяются.",
        ],
    }
    report_path = arguments.output / "g9r_relative_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"measurement_rows={len(all_rows)}")
    print(f"measurement_failures={len(all_failures)}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
