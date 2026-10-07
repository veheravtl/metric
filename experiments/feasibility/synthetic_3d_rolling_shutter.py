#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""G14-B: отдельные позы строк при движении rolling-shutter камеры."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from aerial_mapper.metric_recovery import MetricRecoveryThresholds
from aerial_mapper.paired_stability import stability_seeds_for_pair
from aerial_mapper.rolling_shutter import (
    RollingShutterMotion,
    project_world_points_rolling_shutter,
    warp_planar_rolling_shutter,
)
from aerial_mapper.synthetic_3d import run_synthetic_3d_smoke
from aerial_mapper.teach_annotation import offset_mask_boundary
from experiments.feasibility.synthetic_3d_calibration_uncertainty_repair import (
    PhysicalCase,
    control_summary,
    evaluate_physical_case,
    safety_summary,
)
from experiments.feasibility.synthetic_3d_lens_distortion import (
    build_map_calibration,
    camera_index,
    intrinsics_for_camera,
    load_mask,
    load_rgb,
    load_truth,
    save_rgb,
    sha256_file,
)
from experiments.feasibility.synthetic_3d_terrain_pose import (
    GENERATOR_SCRIPT,
    build_scene_description,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "experiments/configs/synthetic_3d_g14b_rolling_shutter.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/synthetic_3d/g14b_rolling_shutter"
DEFAULT_BLENDER = PROJECT_ROOT / ".tools/blender-4.5.14-linux-x64/blender"


def parse_arguments() -> argparse.Namespace:
    """Читает только пути и явное разрешение переиспользовать сцены."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--reuse-scenes", action="store_true")
    return parser.parse_args()


def project_path(value: str) -> Path:
    """Разрешает путь frozen-конфигурации относительно корня проекта."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def selected_scene_description(
    protocol: dict[str, Any], experiment: dict[str, Any], surface: dict[str, Any]
) -> dict[str, Any]:
    """Строит чистую плоскую сцену только с реально проверяемыми камерами."""

    description = build_scene_description(protocol, surface)
    selected_ids = set(experiment["repeat_camera_ids"])
    description["scenario_id"] = f"g14b_{surface['id']}"
    description["cameras"] = [
        camera
        for camera in description["cameras"]
        if camera["role"] == "reference" or camera["id"] in selected_ids
    ]
    return description


def ensure_scenes(
    *,
    experiment: dict[str, Any],
    protocol: dict[str, Any],
    output: Path,
    blender: Path,
    reuse_scenes: bool,
) -> dict[str, str]:
    """Генерирует новые плоские seed либо строго сверяет frozen-описание."""

    hashes: dict[str, str] = {}
    for surface in experiment["surfaces"]:
        scene_directory = output / "scenes" / surface["id"]
        scene_directory.mkdir(parents=True, exist_ok=True)
        description = selected_scene_description(protocol, experiment, surface)
        frozen_path = scene_directory / "frozen_scene_config.json"
        reusable = bool(
            frozen_path.exists()
            and json.loads(frozen_path.read_text(encoding="utf-8")) == description
            and (scene_directory / "generation_metadata.json").exists()
        )
        frozen_path.write_text(
            json.dumps(description, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not (reuse_scenes and reusable):
            smoke = run_synthetic_3d_smoke(
                description_path=frozen_path,
                output_directory=scene_directory,
                blender_executable=blender,
                generator_script=GENERATOR_SCRIPT,
            )
            if not smoke["passed"]:
                raise RuntimeError(f"Blender smoke не пройден: {surface['id']}")
        hashes[surface["id"]] = sha256_file(frozen_path)
    return hashes


def camera_matrix(scene_directory: Path, camera_id: str) -> np.ndarray:
    """Читает фактическую camera-to-world матрицу из metadata Blender."""

    metadata = json.loads(
        (scene_directory / "generation_metadata.json").read_text(encoding="utf-8")
    )
    matches = [camera for camera in metadata["cameras"] if camera["id"] == camera_id]
    if len(matches) != 1:
        raise RuntimeError(f"Не найдена единственная поза {camera_id}")
    return np.asarray(matches[0]["matrix_camera_to_world_blender"], dtype=np.float64)


def motion_from_case(case: dict[str, Any]) -> RollingShutterMotion:
    """Переводит JSON-векторы в проверенный неизменяемый объект движения."""

    return RollingShutterMotion(
        tuple(float(value) for value in case["total_translation_local_m"]),
        tuple(float(value) for value in case["total_rotation_local_deg"]),
    )


def build_summary_figure(
    rows: list[dict[str, Any]], *, limit_m: float, output_path: Path
) -> None:
    """Сопоставляет силу строковой деформации, ошибку и решения gate."""

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    scatter = None
    for axis, mode, title in zip(
        axes,
        ("translation_x", "rotation_z"),
        ("локальный перенос X", "вращение вокруг оптической оси"),
        strict=True,
    ):
        selected = [
            row
            for row in rows
            if row["mode"] == mode and row["evaluation"]["metric"] is not None
        ]
        scatter = axis.scatter(
            [row["maximum_grid_displacement_px"] for row in selected],
            [row["evaluation"]["metric"]["maximum_m"] for row in selected],
            c=[row["evaluation"]["gate_accept_fraction"] for row in selected],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            alpha=0.8,
        )
        axis.axhline(limit_m, color="red", linestyle="--", linewidth=1)
        axis.set_xlabel("максимальный сдвиг контрольной сетки, px")
        axis.set_title(title)
        axis.set_yscale("symlog", linthresh=0.01)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("максимальная ошибка точки, м")
    if scatter is not None:
        figure.colorbar(scatter, ax=axes, label="доля seed, принявших случай")
    figure.suptitle("G14-B: поза камеры зависит от строки")
    figure.subplots_adjust(top=0.86, bottom=0.12, wspace=0.18)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def main() -> None:
    """Генерирует чистые сцены и выполняет frozen G14-B sweep."""

    arguments = parse_arguments()
    experiment = json.loads(arguments.config.read_text(encoding="utf-8"))
    protocol_path = project_path(experiment["source_protocol"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    arguments.output.mkdir(parents=True, exist_ok=True)
    scene_hashes = ensure_scenes(
        experiment=experiment,
        protocol=protocol,
        output=arguments.output,
        blender=arguments.blender,
        reuse_scenes=arguments.reuse_scenes,
    )
    thresholds = MetricRecoveryThresholds.from_mapping(
        protocol["acceptance_thresholds"]
    )
    limit_m = float(experiment["maximum_point_position_error_m"])
    teach_id = next(
        camera["id"] for camera in protocol["cameras"] if camera["role"] == "reference"
    )
    rows: list[dict[str, Any]] = []
    calibration_reports: dict[str, Any] = {}
    identity_errors: list[float] = []
    row_residuals: list[float] = []

    for surface_number, surface in enumerate(experiment["surfaces"]):
        scene_directory = arguments.output / "scenes" / surface["id"]
        truth = load_truth(scene_directory)
        calibration, groups, anchor_indices = build_map_calibration(
            protocol=protocol, experiment=experiment, truth=truth
        )
        calibration_reports[surface["id"]] = {
            "anchor_count": int(anchor_indices.size),
            "control_reprojection_rmse_m": calibration.control_reprojection_rmse_m,
            "control_reprojection_max_m": calibration.control_reprojection_max_m,
        }
        teach_index = camera_index(truth["camera_ids"], teach_id)
        teach_rgb = load_rgb(scene_directory / "reference/perspective_rgb.png")
        safe_mask = offset_mask_boundary(
            load_mask(scene_directory / "reference/ground_mask.png"),
            offset_pixels=-int(experiment["safe_mask_erosion_px"]),
        )

        for repeat_number, repeat_id in enumerate(experiment["repeat_camera_ids"]):
            repeat_index = camera_index(truth["camera_ids"], repeat_id)
            intrinsics = intrinsics_for_camera(protocol, repeat_id)
            centre_matrix = camera_matrix(scene_directory, repeat_id)
            nominal_rgb = load_rgb(scene_directory / "repeat" / repeat_id / "rgb.png")
            common_indices = np.flatnonzero(
                truth["visible"][teach_index]
                & truth["visible"][repeat_index]
                & (groups == 2)
            )
            stability_seeds = stability_seeds_for_pair(
                experiment_seed=int(experiment["seed"]),
                surface_number=surface_number,
                repeat_number=repeat_number,
                offsets=[int(value) for value in experiment["stability_seed_offsets"]],
            )

            for case in experiment["motion_cases"]:
                motion = motion_from_case(case)
                warp = warp_planar_rolling_shutter(
                    nominal_rgb, centre_matrix, intrinsics, motion
                )
                projection = project_world_points_rolling_shutter(
                    truth["world_xyz_m"][common_indices],
                    centre_matrix,
                    intrinsics,
                    motion,
                )
                in_frame = (
                    projection.converged
                    & (projection.pixel_xy[:, 0] >= 0.0)
                    & (projection.pixel_xy[:, 0] < intrinsics.width_px)
                    & (projection.pixel_xy[:, 1] >= 0.0)
                    & (projection.pixel_xy[:, 1] < intrinsics.height_px)
                )
                selected_indices = common_indices[in_frame]
                selected_points = projection.pixel_xy[in_frame]
                maximum_grid_displacement = float(
                    np.max(projection.displacement_from_centre_pose_px[in_frame])
                )
                maximum_row_residual = float(
                    np.max(projection.row_residual_px[in_frame])
                )
                row_residuals.append(maximum_row_residual)
                if case["mode"] == "nominal":
                    identity_errors.append(warp.maximum_displacement_px)

                physical_case = PhysicalCase(
                    sweep=str(case["mode"]),
                    case_id=str(case["id"]),
                    true_k1=0.0,
                    estimated_k1=None,
                    calibration_error_k1=None,
                    true_edge_shift_px=warp.maximum_displacement_px,
                    residual_grid_shift_px=maximum_grid_displacement,
                    repeat_rgb=warp.image_rgb,
                    query_points_px=selected_points,
                    truth_indices=selected_indices,
                )
                evaluation = evaluate_physical_case(
                    case=physical_case,
                    teach_rgb=teach_rgb,
                    safe_mask=safe_mask,
                    thresholds=thresholds,
                    stability_seeds=stability_seeds,
                    stability_trials=int(experiment["stability_trials_per_seed"]),
                    stability_subsample_fraction=float(
                        experiment["stability_subsample_fraction"]
                    ),
                    truth=truth,
                    calibration=calibration,
                    protocol=protocol,
                    limit_m=limit_m,
                )
                rows.append(
                    {
                        "surface_id": surface["id"],
                        "repeat_id": repeat_id,
                        "case_id": case["id"],
                        "mode": case["mode"],
                        "total_translation_local_m": case["total_translation_local_m"],
                        "total_rotation_local_deg": case["total_rotation_local_deg"],
                        "valid_warp_fraction": float(np.mean(warp.valid_mask)),
                        "maximum_warp_displacement_px": warp.maximum_displacement_px,
                        "maximum_grid_displacement_px": maximum_grid_displacement,
                        "maximum_row_residual_px": maximum_row_residual,
                        "evaluation_point_count": int(selected_indices.size),
                        "evaluation": evaluation,
                    }
                )
                if (
                    surface_number == 0
                    and repeat_id == "repeat_same"
                    and case["id"]
                    in {"translate_x_plus_2_00m", "rotate_z_plus_2_00deg"}
                ):
                    save_rgb(
                        arguments.output / "diagnostics" / f"{case['id']}.png",
                        warp.image_rgb,
                    )
            print(f"completed={surface['id']}:{repeat_id}", flush=True)

    expected_physical = int(
        experiment["preregistered_validity"]["expected_physical_case_count"]
    )
    nominal_rows = [row for row in rows if row["mode"] == "nominal"]
    translation_rows = [row for row in rows if row["mode"] == "translation_x"]
    rotation_rows = [row for row in rows if row["mode"] == "rotation_z"]
    nominal_control = control_summary(nominal_rows, limit_m=limit_m)
    translation_safety = safety_summary(translation_rows)
    rotation_safety = safety_summary(rotation_rows)
    decision_count = sum(len(row["evaluation"]["seed_decisions"]) for row in rows)
    validity = experiment["preregistered_validity"]
    validity_failures: list[str] = []
    if len(rows) != expected_physical:
        validity_failures.append("Неверное число физических случаев")
    if decision_count != int(validity["expected_gate_decision_count"]):
        validity_failures.append("Неверное число решений gate")
    if max(identity_errors) > float(validity["maximum_identity_warp_error_px"]):
        validity_failures.append("Тождественный warp изменяет координаты")
    if max(row_residuals) > float(validity["maximum_row_equation_residual_px"]):
        validity_failures.append("Не сошлось уравнение строки")
    if nominal_control["alignment_constructed"] < int(
        validity["minimum_nominal_alignment_constructed"]
    ):
        validity_failures.append("Не построен номинальный alignment")
    if nominal_control["metric_failures"] > int(
        validity["maximum_nominal_metric_failures"]
    ):
        validity_failures.append("Есть номинальный метрический сбой")
    if nominal_control["over_limit"] > int(validity["maximum_nominal_over_limit"]):
        validity_failures.append("Номинал превысил 0,2 м")

    safety_limits = experiment["preregistered_safety"]
    safety_failures: list[str] = []
    checks = (
        (
            translation_safety["any_seed_false_accept_cases"],
            safety_limits["maximum_any_seed_translation_false_accept_cases"],
            "Перенос имеет any-seed ложные принятия",
        ),
        (
            translation_safety["all_seed_false_accept_cases"],
            safety_limits["maximum_all_seed_translation_false_accept_cases"],
            "Перенос имеет all-seed ложные принятия",
        ),
        (
            rotation_safety["any_seed_false_accept_cases"],
            safety_limits["maximum_any_seed_rotation_false_accept_cases"],
            "Вращение имеет any-seed ложные принятия",
        ),
        (
            rotation_safety["all_seed_false_accept_cases"],
            safety_limits["maximum_all_seed_rotation_false_accept_cases"],
            "Вращение имеет all-seed ложные принятия",
        ),
    )
    for observed, maximum, message in checks:
        if int(observed) > int(maximum):
            safety_failures.append(message)

    report = {
        "schema_version": 1,
        "experiment": experiment["experiment"],
        "configuration": str(arguments.config),
        "configuration_sha256": sha256_file(arguments.config),
        "source_protocol": str(protocol_path),
        "source_protocol_sha256": sha256_file(protocol_path),
        "scene_configuration_sha256": scene_hashes,
        "physical_case_count": len(rows),
        "gate_decision_count": decision_count,
        "maximum_identity_warp_error_px": max(identity_errors),
        "maximum_row_equation_residual_px": max(row_residuals),
        "nominal_control": nominal_control,
        "translation_safety": translation_safety,
        "rotation_safety": rotation_safety,
        "preregistered_validity": validity,
        "validity_failures": validity_failures,
        "experiment_valid": not validity_failures,
        "preregistered_safety": safety_limits,
        "safety_failures": safety_failures,
        "safety_passed": not safety_failures,
        "calibrations": calibration_reports,
        "rows": rows,
        "primary_sources": experiment["primary_sources"],
        "limitations": [
            "Статичная плоскость без объёмных объектов и рельефа.",
            "Мгновенная экспозиция строки без motion blur.",
            "Движение задаётся за кадр, поскольку readout time камеры неизвестен.",
            "Проверены только локальный перенос X и вращение вокруг оптической оси.",
        ],
    }
    report_path = arguments.output / "g14b_rolling_shutter_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    build_summary_figure(
        rows,
        limit_m=limit_m,
        output_path=arguments.output / "g14b_rolling_shutter_summary.png",
    )
    print(
        json.dumps(
            {
                "nominal_control": nominal_control,
                "translation_safety": translation_safety,
                "rotation_safety": rotation_safety,
            },
            ensure_ascii=False,
        )
    )
    print(f"experiment_valid={not validity_failures}")
    print(f"safety_passed={not safety_failures}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
