#!/usr/bin/env python3
"""G8: устойчивость Teach-маски и метрической разметки к ошибкам."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from synthetic_3d_teach_repeat_gate import (
    control_layers_by_camera,
    degradation_from_mapping,
    evaluate,
    ground_truth_controls,
    load_mask,
    load_rgb,
    metric_errors,
    projected_points,
    save_rgb,
    within_metric_limits,
)

from aerial_mapper.alignment import AlignmentFailure, align_frame_to_reference
from aerial_mapper.metric_recovery import (
    MetricRecoveryThresholds,
    alignment_gate_failures,
)
from aerial_mapper.perspective_metric import calibrate_perspective_reference
from aerial_mapper.quality import analyze_alignment_quality
from aerial_mapper.synthetic_3d import run_synthetic_3d_smoke
from aerial_mapper.synthetic_robustness import (
    apply_image_degradation,
    perturb_click_points,
)
from aerial_mapper.teach_annotation import (
    add_contaminant_fraction,
    enclosed_non_ground,
    mask_agreement,
    offset_mask_boundary,
    retain_connected_fraction,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "experiments/configs/synthetic_3d_g8_annotation_robustness.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/synthetic_3d/g8_annotation_robustness"
DEFAULT_BLENDER = PROJECT_ROOT / ".tools/blender-4.5.14-linux-x64/blender"
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts/blender_generate_scene.py"


def parse_arguments() -> argparse.Namespace:
    """Разбирает только пути; научные параметры хранятся в JSON-протоколе."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    """Разрешает путь из протокола относительно корня репозитория."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_feature_mask(
    case: dict[str, Any],
    *,
    true_ground_mask: np.ndarray,
    safe_base_mask: np.ndarray,
    roof_mask: np.ndarray,
) -> np.ndarray:
    """Создаёт один вариант маски, изменяя ровно указанный тип ошибки."""

    kind = case["kind"]
    if kind == "boundary":
        return offset_mask_boundary(
            true_ground_mask,
            offset_pixels=int(case["offset_px"]),
        )
    if kind == "completeness":
        return retain_connected_fraction(
            safe_base_mask,
            retained_fraction=float(case["retained_fraction"]),
        )
    if kind == "contamination":
        return add_contaminant_fraction(
            safe_base_mask,
            roof_mask,
            added_fraction=float(case["added_fraction"]),
        )
    raise ValueError(f"Неизвестный тип масочной ошибки: {kind!r}")


def calibration_points(
    controls: list[dict[str, Any]],
    *,
    teach_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает восемь пар Teach-пиксель ↔ координата земли в метрах."""

    pixels: list[np.ndarray] = []
    world: list[np.ndarray] = []
    for control in controls:
        if control["purpose"] != "calibration":
            continue
        pixels.append(projected_points(control, teach_id))
        world.append(np.asarray(control["points_world_m"], dtype=np.float64)[:, :2])
    if not pixels:
        raise RuntimeError("В сцене нет метрических Teach-контролей")
    return np.concatenate(pixels), np.concatenate(world)


def classification_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Считает исходы одного масочного варианта."""

    result: dict[str, int] = {}
    for row in rows:
        classification = row["result"]["classification"]
        result[classification] = result.get(classification, 0) + 1
    return result


def accepted_metric_summary(
    rows: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    """Сводит ошибки только выданных системой измерений.

    Отклонённая оценка остаётся диагностикой, но не является пользовательским
    ответом. Поэтому процентиль рабочей ошибки считается по принятым случаям,
    включая ложные принятия.
    """

    point_errors: list[float] = []
    absolute_length_errors: list[float] = []
    relative_length_errors: list[float] = []
    accepted_case_count = 0
    for row in rows:
        result = row["result"]
        errors = result["metric_errors"]
        if not result["gate_accepted"] or errors is None:
            continue
        accepted_case_count += 1
        for control in errors["controls"]:
            point_errors.append(
                float(control["maximum_endpoint_position_error_m"])
            )
            absolute_length_errors.append(float(control["absolute_length_error_m"]))
            relative_length_errors.append(
                float(control["relative_length_error_percent"])
            )
    if not point_errors:
        return {
            "accepted_case_count": accepted_case_count,
            "accepted_point_position_error_p95_m": None,
            "accepted_point_position_error_max_m": None,
            "accepted_segment_absolute_error_p95_m": None,
            "accepted_segment_relative_error_p95_percent": None,
        }
    return {
        "accepted_case_count": accepted_case_count,
        "accepted_point_position_error_p95_m": percentile(point_errors),
        "accepted_point_position_error_max_m": max(point_errors),
        "accepted_segment_absolute_error_p95_m": percentile(absolute_length_errors),
        "accepted_segment_relative_error_p95_percent": percentile(
            relative_length_errors
        ),
    }


def percentile(values: list[float], level: float = 95.0) -> float:
    """Считает процентиль и явно отвергает пустую выборку."""

    if not values:
        raise RuntimeError("Нельзя вычислить процентиль пустого набора")
    return float(np.percentile(np.asarray(values, dtype=np.float64), level))


def prepare_repeat_cases(
    *,
    scene_description: dict[str, Any],
    generator: dict[str, Any],
    output_directory: Path,
) -> list[dict[str, Any]]:
    """Один раз готовит те же четыре позы и три RGB-условия, что использовал G7."""

    layers = control_layers_by_camera(generator)
    repeat_ids = [
        item["id"] for item in scene_description["cameras"] if item["role"] == "repeat"
    ]
    scene_directory = output_directory / "scene"
    prepared: list[dict[str, Any]] = []
    for repeat_index, repeat_id in enumerate(repeat_ids):
        repeat_rgb = load_rgb(scene_directory / "repeat" / repeat_id / "rgb.png")
        repeat_ground_mask = load_mask(
            scene_directory / layers[repeat_id]["ground_mask"]
        )
        controls = ground_truth_controls(
            generator["metric_controls"],
            repeat_id=repeat_id,
            repeat_ground_mask=repeat_ground_mask,
        )
        for condition_index, condition in enumerate(
            scene_description["g7"]["conditions"]
        ):
            random_seed = (
                int(scene_description["seed"]) + repeat_index * 100 + condition_index
            )
            degradation = degradation_from_mapping(condition)
            degraded = apply_image_degradation(
                repeat_rgb,
                degradation,
                random_seed=random_seed,
            )
            image_path = (
                output_directory
                / "repeat_conditions"
                / repeat_id
                / f"{condition['id']}.png"
            )
            save_rgb(image_path, degraded)
            prepared.append(
                {
                    "repeat_id": repeat_id,
                    "condition_id": condition["id"],
                    "degradation": asdict(degradation),
                    "random_seed": random_seed,
                    "image_path": image_path,
                    "image_rgb": degraded,
                    "controls": controls,
                }
            )
    return prepared


def evaluate_masks(
    *,
    protocol: dict[str, Any],
    scene_description: dict[str, Any],
    teach_rgb: np.ndarray,
    true_ground_mask: np.ndarray,
    feature_masks: dict[str, np.ndarray],
    mask_metadata: dict[str, dict[str, Any]],
    repeat_cases: list[dict[str, Any]],
    calibration: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Прогоняет все маски при неизменных RGB, калибровке и порогах gate."""

    thresholds = MetricRecoveryThresholds.from_mapping(
        scene_description["acceptance_thresholds"]
    )
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for mask_index, case in enumerate(protocol["mask_cases"]):
        case_rows: list[dict[str, Any]] = []
        feature_mask = feature_masks[case["id"]]
        for repeat_index, repeat_case in enumerate(repeat_cases):
            result = evaluate(
                teach_rgb,
                repeat_case["image_rgb"],
                reference_mask=feature_mask,
                teach_ground_mask=true_ground_mask,
                controls=repeat_case["controls"],
                calibration=calibration,
                thresholds=thresholds,
                limits=scene_description["acceptance_thresholds"],
                random_seed=(int(protocol["seed"]) + mask_index * 1000 + repeat_index),
            )
            row = {
                "mask_id": case["id"],
                "repeat_id": repeat_case["repeat_id"],
                "condition_id": repeat_case["condition_id"],
                "result": result,
            }
            rows.append(row)
            case_rows.append(row)
        summaries.append(
            {
                **mask_metadata[case["id"]],
                "classification_counts": classification_counts(case_rows),
                **accepted_metric_summary(case_rows),
            }
        )
    return rows, summaries


def accepted_alignments(
    *,
    teach_rgb: np.ndarray,
    base_mask: np.ndarray,
    repeat_cases: list[dict[str, Any]],
    thresholds: MetricRecoveryThresholds,
) -> list[dict[str, Any]]:
    """Повторяет nominal alignment и оставляет только реально принятые случаи."""

    accepted: list[dict[str, Any]] = []
    for repeat_case in repeat_cases:
        try:
            alignment = align_frame_to_reference(
                teach_rgb,
                repeat_case["image_rgb"],
                reference_feature_mask=base_mask,
            )
            quality = analyze_alignment_quality(
                alignment,
                frame_width_pixels=repeat_case["image_rgb"].shape[1],
                frame_height_pixels=repeat_case["image_rgb"].shape[0],
                random_seed=int(repeat_case["random_seed"]),
            )
            failures = alignment_gate_failures(
                alignment,
                quality,
                thresholds=thresholds,
            )
            if not failures:
                accepted.append({**repeat_case, "alignment": alignment})
        except AlignmentFailure:
            continue
    return accepted


def evaluate_calibration_noise(
    *,
    protocol: dict[str, Any],
    scene_description: dict[str, Any],
    teach_pixels: np.ndarray,
    world_points: np.ndarray,
    accepted_cases: list[dict[str, Any]],
    reference_shape: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Измеряет перенос шума Teach-кликов в независимые наземные отрезки."""

    noise_spec = protocol["calibration_pixel_noise"]
    trial_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for level_index, sigma in enumerate(noise_spec["standard_deviations_px"]):
        position_errors: list[float] = []
        absolute_length_errors: list[float] = []
        relative_length_errors: list[float] = []
        calibration_rmse: list[float] = []
        false_accept_count = 0
        trials_with_false_accept = 0
        for trial_index in range(int(noise_spec["trials_per_level"])):
            random_seed = (
                int(protocol["seed"]) + 100_000 + level_index * 1000 + trial_index
            )
            noisy_pixels = perturb_click_points(
                teach_pixels,
                standard_deviation_px=float(sigma),
                random_seed=random_seed,
            )
            calibration = calibrate_perspective_reference(noisy_pixels, world_points)
            calibration_rmse.append(calibration.control_reprojection_rmse_m)
            case_results = []
            for accepted_case in accepted_cases:
                errors = metric_errors(
                    accepted_case["controls"],
                    accepted_case["alignment"].homography_frame_to_reference,
                    calibration=calibration,
                    reference_shape=reference_shape,
                )
                valid = within_metric_limits(
                    errors,
                    scene_description["acceptance_thresholds"],
                )
                if not valid:
                    false_accept_count += 1
                for control in errors["controls"]:
                    position_errors.append(
                        float(control["maximum_endpoint_position_error_m"])
                    )
                    absolute_length_errors.append(
                        float(control["absolute_length_error_m"])
                    )
                    relative_length_errors.append(
                        float(control["relative_length_error_percent"])
                    )
                case_results.append(
                    {
                        "repeat_id": accepted_case["repeat_id"],
                        "condition_id": accepted_case["condition_id"],
                        "within_metric_limits": valid,
                        "metric_errors": errors,
                    }
                )
            if any(not item["within_metric_limits"] for item in case_results):
                trials_with_false_accept += 1
            trial_rows.append(
                {
                    "standard_deviation_px": float(sigma),
                    "trial_index": trial_index,
                    "random_seed": random_seed,
                    "calibration_rmse_m": calibration.control_reprojection_rmse_m,
                    "calibration_max_residual_m": (
                        calibration.control_reprojection_max_m
                    ),
                    "cases": case_results,
                }
            )
        summaries.append(
            {
                "standard_deviation_px": float(sigma),
                "trial_count": int(noise_spec["trials_per_level"]),
                "accepted_alignment_count_per_trial": len(accepted_cases),
                "evaluated_segment_count": len(position_errors),
                "false_accept_count": false_accept_count,
                "trials_with_false_accept": trials_with_false_accept,
                "calibration_self_rmse_p95_m": percentile(calibration_rmse),
                "point_position_error_p95_m": percentile(position_errors),
                "point_position_error_max_m": max(position_errors),
                "segment_absolute_error_p95_m": percentile(absolute_length_errors),
                "segment_absolute_error_max_m": max(absolute_length_errors),
                "segment_relative_error_p95_percent": percentile(
                    relative_length_errors
                ),
                "segment_relative_error_max_percent": max(relative_length_errors),
            }
        )
    return trial_rows, summaries


def make_summary_figure(
    *,
    mask_summaries: list[dict[str, Any]],
    calibration_summaries: list[dict[str, Any]],
    success_spec: dict[str, Any],
    path: Path,
) -> None:
    """Сохраняет компактный обзор доступности, качества маски и метрики."""

    figure, axes = plt.subplots(3, 1, figsize=(13, 14), constrained_layout=True)
    labels = [item["id"] for item in mask_summaries]
    positions = np.arange(len(labels))
    accepted = [
        item["classification_counts"].get("accepted_correct", 0)
        for item in mask_summaries
    ]
    false_accepts = [
        item["classification_counts"].get("false_accept", 0) for item in mask_summaries
    ]
    axes[0].bar(positions - 0.2, accepted, width=0.4, label="корректно принято")
    axes[0].bar(positions + 0.2, false_accepts, width=0.4, label="ложно принято")
    axes[0].set_ylabel("случаев из 12")
    axes[0].set_xticks(positions, labels, rotation=35, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    precision = [item["agreement"]["precision"] for item in mask_summaries]
    recall = [item["agreement"]["recall"] for item in mask_summaries]
    axes[1].plot(positions, precision, "o-", label="чистота (precision)")
    axes[1].plot(positions, recall, "s-", label="полнота (recall)")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(positions, labels, rotation=35, ha="right")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    sigma = [item["standard_deviation_px"] for item in calibration_summaries]
    point_p95 = [item["point_position_error_p95_m"] for item in calibration_summaries]
    axes[2].plot(sigma, point_p95, "o-", label="p95 ошибки положения")
    axes[2].axhline(
        success_spec["maximum_calibration_p95_point_position_error_m"],
        color="tab:red",
        linestyle="--",
        label="предел 0,2 м",
    )
    axes[2].set_xlabel("стандартное отклонение ошибки клика, px")
    axes[2].set_ylabel("метры")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    figure.suptitle("G8: устойчивость Teach-разметки")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Выполняет зафиксированный G8 и сохраняет полный машиночитаемый отчёт."""

    arguments = parse_arguments()
    protocol_bytes = arguments.config.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    scene_config = resolve_project_path(protocol["scene_config"])
    scene_description = json.loads(scene_config.read_text(encoding="utf-8"))
    arguments.output.mkdir(parents=True, exist_ok=True)

    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    frozen_protocol = {
        "sha256": protocol_hash,
        "source": str(arguments.config),
        "protocol": protocol,
    }
    (arguments.output / "frozen_protocol.json").write_text(
        json.dumps(frozen_protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    smoke = run_synthetic_3d_smoke(
        description_path=scene_config,
        output_directory=arguments.output / "scene",
        blender_executable=arguments.blender,
        generator_script=GENERATOR_SCRIPT,
    )
    generator = smoke["generator"]
    layers = control_layers_by_camera(generator)
    teach_id = next(
        item["id"]
        for item in scene_description["cameras"]
        if item["role"] == "reference"
    )
    scene_directory = arguments.output / "scene"
    teach_rgb = load_rgb(scene_directory / "reference/perspective_rgb.png")
    true_ground_mask = load_mask(scene_directory / layers[teach_id]["ground_mask"])
    safe_base_mask = offset_mask_boundary(
        true_ground_mask,
        offset_pixels=-int(protocol["base_mask_erosion_px"]),
    )
    roof_mask = enclosed_non_ground(true_ground_mask)
    cv2.imwrite(str(arguments.output / "enclosed_non_ground_mask.png"), roof_mask)

    feature_masks: dict[str, np.ndarray] = {}
    mask_metadata: dict[str, dict[str, Any]] = {}
    mask_directory = arguments.output / "teach_masks"
    mask_directory.mkdir(parents=True, exist_ok=True)
    for case in protocol["mask_cases"]:
        feature_mask = build_feature_mask(
            case,
            true_ground_mask=true_ground_mask,
            safe_base_mask=safe_base_mask,
            roof_mask=roof_mask,
        )
        feature_masks[case["id"]] = feature_mask
        mask_path = mask_directory / f"{case['id']}.png"
        if not cv2.imwrite(str(mask_path), feature_mask):
            raise RuntimeError(f"Не удалось сохранить маску {mask_path}")
        agreement = mask_agreement(feature_mask, true_ground_mask)
        mask_metadata[case["id"]] = {
            **case,
            "path": str(mask_path),
            "agreement": asdict(agreement),
        }

    teach_pixels, world_points = calibration_points(
        generator["metric_controls"],
        teach_id=teach_id,
    )
    exact_calibration = calibrate_perspective_reference(teach_pixels, world_points)
    repeat_cases = prepare_repeat_cases(
        scene_description=scene_description,
        generator=generator,
        output_directory=arguments.output,
    )
    mask_rows, mask_summaries = evaluate_masks(
        protocol=protocol,
        scene_description=scene_description,
        teach_rgb=teach_rgb,
        true_ground_mask=true_ground_mask,
        feature_masks=feature_masks,
        mask_metadata=mask_metadata,
        repeat_cases=repeat_cases,
        calibration=exact_calibration,
    )

    thresholds = MetricRecoveryThresholds.from_mapping(
        scene_description["acceptance_thresholds"]
    )
    nominal_accepted = accepted_alignments(
        teach_rgb=teach_rgb,
        base_mask=feature_masks["baseline_erode_9"],
        repeat_cases=repeat_cases,
        thresholds=thresholds,
    )
    calibration_rows, calibration_summaries = evaluate_calibration_noise(
        protocol=protocol,
        scene_description=scene_description,
        teach_pixels=teach_pixels,
        world_points=world_points,
        accepted_cases=nominal_accepted,
        reference_shape=teach_rgb.shape[:2],
    )

    success_spec = protocol["preregistered_success"]
    baseline = next(item for item in mask_summaries if item["id"] == "baseline_erode_9")
    baseline_counts = baseline["classification_counts"]
    baseline_reproduced = bool(
        baseline_counts.get("accepted_correct", 0)
        == success_spec["baseline_expected_accepted_correct"]
        and baseline_counts.get("rejected_valid", 0)
        == success_spec["baseline_expected_rejected_valid"]
    )
    mask_false_accepts = sum(
        item["classification_counts"].get("false_accept", 0) for item in mask_summaries
    )
    mask_safety_passed = bool(
        mask_false_accepts <= success_spec["maximum_mask_false_accepts_all_cases"]
    )
    operational_availability_passed = all(
        item["classification_counts"].get("accepted_correct", 0)
        >= success_spec["minimum_accepted_correct_per_operational_mask"]
        for item in mask_summaries
        if item["operational"]
    )
    operational_sigma_limit = protocol["calibration_pixel_noise"][
        "maximum_operational_standard_deviation_px"
    ]
    operational_calibration = [
        item
        for item in calibration_summaries
        if item["standard_deviation_px"] <= operational_sigma_limit
    ]
    calibration_passed = all(
        item["false_accept_count"]
        <= success_spec["maximum_calibration_false_accepts_operational"]
        and item["point_position_error_p95_m"]
        <= success_spec["maximum_calibration_p95_point_position_error_m"]
        and item["segment_absolute_error_p95_m"]
        <= success_spec["maximum_calibration_p95_segment_absolute_error_m"]
        and item["segment_relative_error_p95_percent"]
        <= success_spec["maximum_calibration_p95_segment_relative_error_percent"]
        for item in operational_calibration
    )
    preregistered_passed = bool(
        smoke["passed"]
        and baseline_reproduced
        and mask_safety_passed
        and operational_availability_passed
        and calibration_passed
    )

    figure_path = arguments.output / "g8_summary.png"
    make_summary_figure(
        mask_summaries=mask_summaries,
        calibration_summaries=calibration_summaries,
        success_spec=success_spec,
        path=figure_path,
    )
    report = {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "protocol_sha256": protocol_hash,
        "configuration": str(arguments.config),
        "scene_configuration": str(scene_config),
        "smoke_passed": smoke["passed"],
        "frozen_alignment_thresholds": scene_description["acceptance_thresholds"],
        "input_contract": {
            "working_algorithm_receives": [
                "perspective Teach RGB",
                "one possibly imperfect Teach feature mask",
                "eight possibly imperfect Teach pixel annotations with exact world XY",
                "Repeat RGB",
            ],
            "working_algorithm_does_not_receive": [
                "Repeat pose, intrinsics, depth or ground mask",
                "true Repeat-to-Teach homography",
                "world coordinates of evaluation segments",
            ],
        },
        "mask_experiment": {
            "case_count": len(mask_summaries),
            "repeat_condition_count_per_case": len(repeat_cases),
            "summaries": mask_summaries,
            "rows": mask_rows,
            "false_accept_count": mask_false_accepts,
            "safety_passed": mask_safety_passed,
            "operational_availability_passed": operational_availability_passed,
        },
        "calibration_experiment": {
            "control_point_count": int(teach_pixels.shape[0]),
            "nominal_accepted_alignment_count": len(nominal_accepted),
            "summaries": calibration_summaries,
            "trials": calibration_rows,
            "operational_passed": calibration_passed,
        },
        "baseline_reproduced": baseline_reproduced,
        "preregistered_success": success_spec,
        "preregistered_passed": preregistered_passed,
        "visualization": str(figure_path),
        "sources": [
            "https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html",
            "https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html",
            "https://arxiv.org/abs/1803.03025",
        ],
        "limitations": [
            (
                "One synthetic scene and twelve Repeat RGB cases cannot estimate "
                "rare field failures."
            ),
            (
                "Mask corruptions are controlled geometric models, not errors "
                "sampled from real annotators or a trained segmenter."
            ),
            (
                "Pixel click noise is isotropic Gaussian and world coordinates "
                "are exact."
            ),
            (
                "Only ground measurements are evaluated; roofs and facades "
                "remain unsupported."
            ),
        ],
    }
    report_path = arguments.output / "g8_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "mask_false_accepts": mask_false_accepts,
                "baseline_reproduced": baseline_reproduced,
                "operational_availability_passed": operational_availability_passed,
                "calibration_passed": calibration_passed,
                "preregistered_passed": preregistered_passed,
            },
            ensure_ascii=False,
        )
    )
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
