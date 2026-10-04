#!/usr/bin/env python3
"""G7: перспективный размеченный Teach → неизвестный Repeat RGB."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
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
    measure_ground_segment_via_perspective_reference,
)
from aerial_mapper.quality import analyze_alignment_quality
from aerial_mapper.synthetic_3d import run_synthetic_3d_smoke
from aerial_mapper.synthetic_robustness import (
    ImageDegradation,
    apply_image_degradation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "experiments/configs/synthetic_3d_g7_teach_repeat.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/synthetic_3d/g7_teach_repeat"
DEFAULT_BLENDER = PROJECT_ROOT / ".tools/blender-4.5.14-linux-x64/blender"
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts/blender_generate_scene.py"


def parse_arguments() -> argparse.Namespace:
    """Разбирает пути воспроизводимого запуска."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    """Читает PNG как RGB uint8."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать RGB: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    """Читает контрольную маску как бинарный uint8-массив."""

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Не удалось прочитать маску: {path}")
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    """Сохраняет деградированный Repeat для ручного аудита."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Не удалось записать RGB: {path}")


def degradation_from_mapping(values: dict[str, Any]) -> ImageDegradation:
    """Отделяет параметры изображения от идентификатора условия."""

    names = set(ImageDegradation.__dataclass_fields__)
    return ImageDegradation(**{name: values[name] for name in names if name in values})


def control_layers_by_camera(generator: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Индексирует пути скрытых слоёв по камере."""

    return {item["camera_id"]: item for item in generator["rendered_control_layers"]}


def points_are_on_mask(points_px: np.ndarray, mask: np.ndarray) -> bool:
    """Проверяет, что центры контрольных пикселей относятся к видимой земле."""

    rounded = np.rint(points_px).astype(np.int64)
    height, width = mask.shape
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    if not np.all(inside):
        return False
    return bool(np.all(mask[rounded[:, 1], rounded[:, 0]] != 0))


def projected_points(control: dict[str, Any], camera_id: str) -> np.ndarray:
    """Возвращает две скрытые пиксельные проекции одного контроля."""

    projection = control["projections"][camera_id]
    if not all(point["visible"] for point in projection):
        raise RuntimeError(
            f"Контроль {control['id']} не виден целиком камерой {camera_id}"
        )
    return np.asarray([point["pixel_xy"] for point in projection], dtype=np.float64)


def calibrate_teach(
    controls: list[dict[str, Any]],
    *,
    teach_id: str,
    teach_ground_mask: np.ndarray,
) -> PerspectiveReferenceCalibration:
    """Строит разрешённую метрическую калибровку из разметки Teach."""

    pixels: list[np.ndarray] = []
    world: list[np.ndarray] = []
    for control in controls:
        if control["purpose"] != "calibration":
            continue
        points = projected_points(control, teach_id)
        if not points_are_on_mask(points, teach_ground_mask):
            raise RuntimeError(
                f"Калибровочный контроль {control['id']} не лежит на видимой земле"
            )
        pixels.append(points)
        world.append(np.asarray(control["points_world_m"], dtype=np.float64)[:, :2])
    if not pixels:
        raise RuntimeError("В сцене нет Teach-контролей purpose=calibration")
    return calibrate_perspective_reference(
        np.concatenate(pixels),
        np.concatenate(world),
    )


def ground_truth_controls(
    controls: list[dict[str, Any]],
    *,
    repeat_id: str,
    repeat_ground_mask: np.ndarray,
) -> list[tuple[dict[str, Any], np.ndarray]]:
    """Собирает независимые ground-контроли, физически видимые в Repeat."""

    selected: list[tuple[dict[str, Any], np.ndarray]] = []
    for control in controls:
        if control["surface"] != "ground" or control["purpose"] != "evaluation":
            continue
        points = projected_points(control, repeat_id)
        if not points_are_on_mask(points, repeat_ground_mask):
            raise RuntimeError(
                f"Оценочный контроль {control['id']} перекрыт в {repeat_id}"
            )
        selected.append((control, points))
    if not selected:
        raise RuntimeError(f"Для {repeat_id} нет ground-контролей оценки")
    return selected


def metric_errors(
    controls: list[tuple[dict[str, Any], np.ndarray]],
    homography: np.ndarray,
    *,
    calibration: PerspectiveReferenceCalibration,
    reference_shape: tuple[int, int],
) -> dict[str, Any]:
    """Считает ошибки положения концов и длины только после оценки H."""

    evaluations = []
    reference_height, reference_width = reference_shape
    for control, frame_points in controls:
        measurement = measure_ground_segment_via_perspective_reference(
            frame_points,
            homography,
            calibration=calibration,
            reference_width_pixels=reference_width,
            reference_height_pixels=reference_height,
        )
        true_world = np.asarray(control["points_world_m"], dtype=np.float64)[:, :2]
        estimated_world = measurement.mapped_points.world_points_m
        endpoint_errors = np.linalg.norm(estimated_world - true_world, axis=1)
        true_length = float(np.linalg.norm(true_world[1] - true_world[0]))
        absolute_length_error = abs(measurement.length_meters - true_length)
        evaluations.append(
            {
                "id": control["id"],
                "true_length_m": true_length,
                "estimated_length_m": measurement.length_meters,
                "maximum_endpoint_position_error_m": float(np.max(endpoint_errors)),
                "absolute_length_error_m": absolute_length_error,
                "relative_length_error_percent": absolute_length_error
                / true_length
                * 100.0,
            }
        )
    return {
        "controls": evaluations,
        "maximum_endpoint_position_error_m": max(
            item["maximum_endpoint_position_error_m"] for item in evaluations
        ),
        "maximum_absolute_length_error_m": max(
            item["absolute_length_error_m"] for item in evaluations
        ),
        "maximum_relative_length_error_percent": max(
            item["relative_length_error_percent"] for item in evaluations
        ),
    }


def inlier_ground_fraction(
    reference_points_px: np.ndarray,
    inlier_mask: np.ndarray,
    teach_ground_mask: np.ndarray,
) -> float:
    """Диагностирует, какую поверхность фактически поддержал RANSAC."""

    points = reference_points_px[inlier_mask]
    rounded = np.rint(points).astype(np.int64)
    height, width = teach_ground_mask.shape
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    on_ground = np.zeros(points.shape[0], dtype=bool)
    on_ground[inside] = teach_ground_mask[rounded[inside, 1], rounded[inside, 0]] != 0
    return float(np.mean(on_ground))


def within_metric_limits(errors: dict[str, Any], limits: dict[str, Any]) -> bool:
    """Проверяет истинные ошибки; эта функция существует только в оценщике."""

    return bool(
        errors["maximum_endpoint_position_error_m"]
        <= limits["maximum_point_position_error_m"]
        and errors["maximum_absolute_length_error_m"]
        <= limits["maximum_segment_absolute_error_m"]
        and errors["maximum_relative_length_error_percent"]
        <= limits["maximum_segment_relative_error_percent"]
    )


def evaluate(
    teach_rgb: np.ndarray,
    repeat_rgb: np.ndarray,
    *,
    reference_mask: np.ndarray | None,
    teach_ground_mask: np.ndarray,
    controls: list[tuple[dict[str, Any], np.ndarray]],
    calibration: PerspectiveReferenceCalibration,
    thresholds: MetricRecoveryThresholds,
    limits: dict[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    """Запускает один вариант, не передавая Repeat-истину в alignment."""

    try:
        alignment = align_frame_to_reference(
            teach_rgb,
            repeat_rgb,
            reference_feature_mask=reference_mask,
        )
        quality = analyze_alignment_quality(
            alignment,
            frame_width_pixels=repeat_rgb.shape[1],
            frame_height_pixels=repeat_rgb.shape[0],
            random_seed=random_seed,
        )
        gate_failures = alignment_gate_failures(
            alignment,
            quality,
            thresholds=thresholds,
        )
        try:
            errors = metric_errors(
                controls,
                alignment.homography_frame_to_reference,
                calibration=calibration,
                reference_shape=teach_rgb.shape[:2],
            )
            truth_valid = within_metric_limits(errors, limits)
            measurement_failure = None
        except MeasurementFailure as error:
            errors = None
            truth_valid = False
            measurement_failure = str(error)

        accepted = not gate_failures and measurement_failure is None
        if accepted and truth_valid:
            classification = "accepted_correct"
        elif accepted:
            classification = "false_accept"
        elif truth_valid:
            classification = "rejected_valid"
        else:
            classification = "rejected_invalid"
        return {
            "classification": classification,
            "gate_accepted": accepted,
            "gate_failures": list(gate_failures),
            "truth_valid_if_estimated": truth_valid,
            "measurement_failure": measurement_failure,
            "alignment": {
                "reference_keypoints": alignment.reference_keypoint_count,
                "frame_keypoints": alignment.frame_keypoint_count,
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
                "inlier_fraction_on_teach_ground": inlier_ground_fraction(
                    alignment.reference_points_px,
                    alignment.inlier_mask,
                    teach_ground_mask,
                ),
            },
            "metric_errors": errors,
        }
    except AlignmentFailure as error:
        return {
            "classification": "alignment_failure",
            "gate_accepted": False,
            "gate_failures": [str(error)],
            "truth_valid_if_estimated": None,
            "measurement_failure": None,
            "alignment": None,
            "metric_errors": None,
        }


def counts(rows: list[dict[str, Any]], method: str) -> dict[str, int]:
    """Подсчитывает исходы одного метода."""

    result: dict[str, int] = {}
    for row in rows:
        classification = row["results"][method]["classification"]
        result[classification] = result.get(classification, 0) + 1
    return result


def main() -> None:
    """Генерирует сцену, сравнивает baseline и Teach-mask, пишет JSON."""

    arguments = parse_arguments()
    description = json.loads(arguments.config.read_text(encoding="utf-8"))
    arguments.output.mkdir(parents=True, exist_ok=True)
    smoke = run_synthetic_3d_smoke(
        description_path=arguments.config,
        output_directory=arguments.output / "scene",
        blender_executable=arguments.blender,
        generator_script=GENERATOR_SCRIPT,
    )
    generator = smoke["generator"]
    cameras = description["cameras"]
    teach_id = next(item["id"] for item in cameras if item["role"] == "reference")
    repeat_ids = [item["id"] for item in cameras if item["role"] == "repeat"]
    layers = control_layers_by_camera(generator)
    scene_directory = arguments.output / "scene"

    teach_rgb = load_rgb(scene_directory / "reference/perspective_rgb.png")
    teach_ground_mask = load_mask(scene_directory / layers[teach_id]["ground_mask"])
    erosion = int(description["g7"]["teach_mask_erosion_px"])
    kernel = np.ones((erosion * 2 + 1, erosion * 2 + 1), dtype=np.uint8)
    feature_mask = cv2.erode(teach_ground_mask, kernel, iterations=1)
    feature_mask_path = arguments.output / "teach_ground_feature_mask.png"
    cv2.imwrite(str(feature_mask_path), feature_mask)

    calibration = calibrate_teach(
        generator["metric_controls"],
        teach_id=teach_id,
        teach_ground_mask=teach_ground_mask,
    )
    thresholds = MetricRecoveryThresholds.from_mapping(
        description["acceptance_thresholds"]
    )
    rows = []
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
        for condition_index, condition in enumerate(description["g7"]["conditions"]):
            degradation = degradation_from_mapping(condition)
            random_seed = (
                int(description["seed"]) + repeat_index * 100 + condition_index
            )
            degraded = apply_image_degradation(
                repeat_rgb,
                degradation,
                random_seed=random_seed,
            )
            image_path = (
                arguments.output
                / "repeat_conditions"
                / repeat_id
                / f"{condition['id']}.png"
            )
            save_rgb(image_path, degraded)
            results = {
                "all_features": evaluate(
                    teach_rgb,
                    degraded,
                    reference_mask=None,
                    teach_ground_mask=teach_ground_mask,
                    controls=controls,
                    calibration=calibration,
                    thresholds=thresholds,
                    limits=description["acceptance_thresholds"],
                    random_seed=random_seed,
                ),
                "teach_ground_mask": evaluate(
                    teach_rgb,
                    degraded,
                    reference_mask=feature_mask,
                    teach_ground_mask=teach_ground_mask,
                    controls=controls,
                    calibration=calibration,
                    thresholds=thresholds,
                    limits=description["acceptance_thresholds"],
                    random_seed=random_seed,
                ),
            }
            rows.append(
                {
                    "repeat_id": repeat_id,
                    "condition_id": condition["id"],
                    "degradation": asdict(degradation),
                    "image": str(image_path),
                    "results": results,
                }
            )

    baseline_false = [
        row
        for row in rows
        if row["results"]["all_features"]["classification"] == "false_accept"
    ]
    corrected = all(
        row["results"]["teach_ground_mask"]["classification"] != "false_accept"
        for row in baseline_false
    )
    masked_counts = counts(rows, "teach_ground_mask")
    success_spec = description["g7"]["preregistered_success"]
    preregistered_pass = bool(
        masked_counts.get("false_accept", 0)
        <= success_spec["maximum_masked_false_accepts"]
        and masked_counts.get("accepted_correct", 0)
        >= success_spec["minimum_masked_accepted_correct"]
        and (
            corrected
            or not success_spec["baseline_false_accepts_must_be_corrected_or_rejected"]
        )
    )
    report = {
        "schema_version": 1,
        "experiment": "G7 perspective annotated Teach to unknown Repeat RGB",
        "smoke_passed": smoke["passed"],
        "input_contract": {
            "working_algorithm_receives": [
                "perspective Teach RGB",
                "Teach ground mask eroded before SIFT",
                "Teach pixel-to-ground-metre calibration",
                "Repeat RGB",
            ],
            "working_algorithm_does_not_receive": [
                "Repeat camera pose or intrinsics",
                "Repeat depth or ground mask",
                "true Repeat-to-Teach homography",
                "world coordinates of evaluation segments",
            ],
            "evaluator_only": [
                "Blender projections",
                "Repeat ground mask",
                "world coordinates of evaluation segments",
            ],
        },
        "configuration": str(arguments.config),
        "teach_mask_erosion_px": erosion,
        "teach_ground_fraction": float(np.mean(teach_ground_mask != 0)),
        "teach_feature_mask_fraction": float(np.mean(feature_mask != 0)),
        "calibration": {
            "control_point_count": calibration.control_point_count,
            "control_reprojection_rmse_m": calibration.control_reprojection_rmse_m,
            "control_reprojection_max_m": calibration.control_reprojection_max_m,
            "homography_reference_to_world_xy": (
                calibration.homography_reference_to_world_xy.tolist()
            ),
        },
        "frozen_thresholds": description["acceptance_thresholds"],
        "rows": rows,
        "classification_counts": {
            "all_features": counts(rows, "all_features"),
            "teach_ground_mask": masked_counts,
        },
        "baseline_false_accept_count": len(baseline_false),
        "baseline_false_accepts_corrected_or_rejected": corrected,
        "preregistered_success": success_spec,
        "preregistered_passed": preregistered_pass,
        "limitations": [
            (
                "Teach ground mask is exact synthetic ground truth, "
                "not a noisy manual mask."
            ),
            "Only one scene, four Repeat poses and three RGB conditions are tested.",
            "Evaluation segments are guaranteed to lie on visible ground.",
            "Synthetic textures do not establish transfer to real aerial imagery.",
        ],
    }
    report_path = arguments.output / "g7_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["classification_counts"], ensure_ascii=False))
    print(f"preregistered_passed={preregistered_pass}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
