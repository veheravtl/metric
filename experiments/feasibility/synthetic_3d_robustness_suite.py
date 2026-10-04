#!/usr/bin/env python3
"""G3--G6: измеряет область устойчивости плоского RGB baseline на синтетике."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from rasterio import Affine

from aerial_mapper.alignment import AlignmentFailure, align_frame_to_reference
from aerial_mapper.measurement import MeasurementFailure
from aerial_mapper.metric_recovery import (
    MetricRecoveryThresholds,
    alignment_gate_failures,
)
from aerial_mapper.quality import analyze_alignment_quality
from aerial_mapper.surface_evaluation import (
    SurfaceSegmentControl,
    estimate_oracle_homography,
    evaluate_surface_segments,
    surface_evaluation_to_dict,
)
from aerial_mapper.synthetic_3d import run_synthetic_3d_smoke
from aerial_mapper.synthetic_robustness import (
    ImageDegradation,
    apply_image_degradation,
    perturb_click_points,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "experiments/configs/synthetic_3d_g3_g6_robustness.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/synthetic_3d/g3_g6_robustness"
DEFAULT_BLENDER = PROJECT_ROOT / ".tools/blender-4.5.14-linux-x64/blender"
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts/blender_generate_scene.py"


def parse_arguments() -> argparse.Namespace:
    """Разбирает только воспроизводимые пути запуска полного набора."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    return parser.parse_args()


def slug(value: float) -> str:
    """Формирует стабильную часть имени из вещественного параметра."""

    return f"{value:.3f}".replace("-", "m").replace(".", "_")


def load_rgb(path: Path) -> np.ndarray:
    """Читает PNG как RGB uint8 и явно сообщает об отсутствии артефакта."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать RGB: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    """Сохраняет диагностический вариант кадра, не меняя исходный рендер."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"OpenCV не смог записать {path}")


def build_description(
    template: dict[str, Any],
    *,
    scenario_id: str,
    object_width_m: float,
    object_height_m: float,
    seed: int,
    camera_override: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Создаёт сцену с одной текстурированной крышей и наружными контролями."""

    description = copy.deepcopy(template)
    for key in ("g3", "g4", "g5", "g6"):
        description.pop(key, None)
    description["scenario_id"] = scenario_id
    description["seed"] = seed
    aspect = float(template["g3"]["object_aspect_ratio_y_to_x"])
    object_depth_m = object_width_m * aspect
    if object_width_m > 0.0:
        description["objects"] = [
            {
                "id": "central_textured_roof",
                "kind": "box",
                "center_m": [0.0, 0.0, object_height_m / 2.0],
                "size_m": [object_width_m, object_depth_m, object_height_m],
                "color_srgb": [0.72, 0.24, 0.14],
                "appearance": "procedural_texture",
            }
        ]
    else:
        description["objects"] = []

    # Даже нулевая сцена содержит метки roof на Z=0. Это проверяет, что само
    # разделение контролей не создаёт ошибку. При объекте точки лежат внутри
    # крыши и дают all-surface oracle одновременно две несовместимые плоскости.
    control_width = max(object_width_m, 6.0)
    control_depth = max(object_depth_m, 4.0)
    description["metric_controls"].extend(
        [
            {
                "id": "roof_horizontal",
                "surface": "roof",
                "points_world_m": [
                    [-0.3 * control_width, 0.0, object_height_m],
                    [0.3 * control_width, 0.0, object_height_m],
                ],
            },
            {
                "id": "roof_vertical",
                "surface": "roof",
                "points_world_m": [
                    [0.0, -0.3 * control_depth, object_height_m],
                    [0.0, 0.3 * control_depth, object_height_m],
                ],
            },
        ]
    )
    if object_width_m == 0.0:
        for control in description["metric_controls"]:
            if control["surface"] == "roof":
                for point in control["points_world_m"]:
                    point[2] = 0.0

    if camera_override:
        repeat = next(
            camera for camera in description["cameras"] if camera["role"] == "repeat"
        )
        repeat["location_m"] = [
            camera_override["camera_x_m"],
            camera_override["camera_y_m"],
            camera_override["camera_z_m"],
        ]
        repeat["target_m"] = [
            camera_override["target_x_m"],
            camera_override["target_y_m"],
            0.0,
        ]
        repeat["focal_length_mm"] = camera_override["focal_length_mm"]
    return description


def build_surface_controls(
    generator: dict[str, Any],
    *,
    reference_camera_id: str,
    repeat_camera_id: str,
) -> tuple[SurfaceSegmentControl, ...]:
    """Преобразует скрытые Blender-проекции в независимую метрическую истину."""

    origin = np.asarray(
        generator["reference_map"]["origin_world_xy_in_crs_m"],
        dtype=np.float64,
    )
    controls = []
    for item in generator["metric_controls"]:
        repeat_projection = item["projections"][repeat_camera_id]
        reference_projection = item["projections"][reference_camera_id]
        if not all(point["visible"] for point in repeat_projection):
            raise RuntimeError(f"Контроль {item['id']} вышел из Repeat-кадра")
        if not all(point["visible"] for point in reference_projection):
            raise RuntimeError(f"Контроль {item['id']} вышел из эталона")
        controls.append(
            SurfaceSegmentControl(
                name=item["id"],
                surface=item["surface"],
                frame_points_px=np.asarray(
                    [point["pixel_xy"] for point in repeat_projection],
                    dtype=np.float64,
                ),
                reference_points_px=np.asarray(
                    [point["pixel_xy"] for point in reference_projection],
                    dtype=np.float64,
                ),
                true_world_points_m=(
                    origin
                    + np.asarray(item["points_world_m"], dtype=np.float64)[:, :2]
                ),
            )
        )
    return tuple(controls)


def summarize(
    evaluations: list[dict[str, Any]],
    surface: str,
) -> dict[str, float]:
    """Возвращает худшие метрические ошибки заданной поверхности."""

    selected = [row for row in evaluations if row["surface"] == surface]
    return {
        "maximum_endpoint_position_error_m": max(
            row["maximum_endpoint_position_error_m"] for row in selected
        ),
        "maximum_absolute_length_error_m": max(
            row["absolute_length_error_m"] for row in selected
        ),
        "maximum_relative_length_error_percent": max(
            row["relative_length_error_percent"] for row in selected
        ),
    }


def within_limits(summary: dict[str, float], limits: dict[str, Any]) -> bool:
    """Проверяет внешнюю истину, не используя внутренние метрики matcher-а."""

    return (
        summary["maximum_endpoint_position_error_m"]
        <= limits["maximum_point_position_error_m"]
        and summary["maximum_absolute_length_error_m"]
        <= limits["maximum_segment_absolute_error_m"]
        and summary["maximum_relative_length_error_percent"]
        <= limits["maximum_segment_relative_error_percent"]
    )


def evaluate_model(
    context: dict[str, Any],
    homography: np.ndarray,
) -> dict[str, Any]:
    """Оценивает одну матрицу на земле и крыше одной и той же цепочкой."""

    evaluations = []
    failures: dict[str, str] = {}
    for control in context["controls"]:
        try:
            evaluation = evaluate_surface_segments(
                (control,),
                homography_frame_to_reference=homography,
                **context["evaluation_arguments"],
            )[0]
        except MeasurementFailure as error:
            failures[control.name] = str(error)
        else:
            evaluations.append(surface_evaluation_to_dict(evaluation))

    result: dict[str, Any] = {
        "controls": evaluations,
        "evaluation_failures": failures,
    }
    for surface in ("ground", "roof"):
        surface_controls = [
            control for control in context["controls"] if control.surface == surface
        ]
        surface_failed = any(control.name in failures for control in surface_controls)
        if surface_failed:
            result[surface] = None
            result[f"{surface}_within_limits"] = False
        else:
            summary = summarize(evaluations, surface)
            result[surface] = summary
            result[f"{surface}_within_limits"] = within_limits(
                summary,
                context["limits"],
            )
    return result


def mask_diagnostics(
    context: dict[str, Any],
    inlier_points: np.ndarray | None,
) -> dict[str, Any]:
    """Измеряет фактическую долю земли и происхождение inlier-точек."""

    mask = cv2.imread(str(context["ground_mask_path"]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(context["ground_mask_path"])
    ground = mask >= 128
    result: dict[str, Any] = {
        "ground_pixel_fraction": float(np.mean(ground)),
        "ground_control_endpoint_fraction": None,
        "inlier_ground_fraction": None,
    }

    endpoints = np.concatenate(
        [
            control.frame_points_px
            for control in context["controls"]
            if control.surface == "ground"
        ],
        axis=0,
    )
    rounded = np.rint(endpoints).astype(np.int64)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, ground.shape[1] - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, ground.shape[0] - 1)
    result["ground_control_endpoint_fraction"] = float(
        np.mean(ground[rounded[:, 1], rounded[:, 0]])
    )
    if inlier_points is not None and inlier_points.size:
        rounded = np.rint(inlier_points).astype(np.int64)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, ground.shape[1] - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, ground.shape[0] - 1)
        result["inlier_ground_fraction"] = float(
            np.mean(ground[rounded[:, 1], rounded[:, 0]])
        )
    return result


def evaluate_frame(
    context: dict[str, Any],
    frame_rgb: np.ndarray,
    *,
    random_seed: int,
) -> dict[str, Any]:
    """Запускает RGB-only alignment, gate и независимую проверку метров."""

    ground_oracle = estimate_oracle_homography(context["controls"], surface="ground")
    all_surface_oracle = estimate_oracle_homography(context["controls"])
    models = {
        "ground_oracle": evaluate_model(context, ground_oracle),
        "all_surface_oracle": evaluate_model(context, all_surface_oracle),
    }
    try:
        alignment = align_frame_to_reference(context["reference_rgb"], frame_rgb)
    except AlignmentFailure as error:
        return {
            "gate_accepted": False,
            "gate_failures": [str(error)],
            "classification": "alignment_failure",
            "truth_valid_if_estimated": None,
            "false_accept": False,
            "alignment": None,
            "ground_mask": mask_diagnostics(context, None),
            "models": models,
        }

    height, width = frame_rgb.shape[:2]
    quality = analyze_alignment_quality(
        alignment,
        frame_width_pixels=width,
        frame_height_pixels=height,
        random_seed=random_seed,
    )
    failures = alignment_gate_failures(
        alignment,
        quality,
        thresholds=context["thresholds"],
    )
    sift = evaluate_model(context, alignment.homography_frame_to_reference)
    models["sift_ransac"] = sift
    accepted = not failures
    truth_valid = sift["ground_within_limits"]
    if accepted and truth_valid:
        classification = "accepted_correct"
    elif accepted:
        classification = "false_accept"
    elif truth_valid:
        classification = "rejected_valid"
    else:
        classification = "rejected_invalid"
    inlier_points = alignment.frame_points_px[alignment.inlier_mask]
    return {
        "gate_accepted": accepted,
        "gate_failures": list(failures),
        "classification": classification,
        "truth_valid_if_estimated": truth_valid,
        "false_accept": accepted and not truth_valid,
        "alignment": {
            "inliers": alignment.inlier_count,
            "inlier_ratio": alignment.inlier_ratio,
            "coverage_fraction": alignment.inlier_spatial_coverage_fraction,
            "reprojection_p95_reference_px": (
                quality.inlier_reprojection_p95_reference_px
            ),
            "stability_p95_corner_shift_reference_px": (
                quality.stability_p95_max_corner_shift_reference_px
            ),
        },
        "ground_mask": mask_diagnostics(context, inlier_points),
        "models": models,
    }


def generate_context(
    template: dict[str, Any],
    *,
    description: dict[str, Any],
    directory: Path,
    blender_executable: Path,
) -> dict[str, Any]:
    """Генерирует сцену и собирает данные, доступные только оценщику."""

    directory.mkdir(parents=True, exist_ok=True)
    description_path = directory / "input_scene_description.json"
    description_path.write_text(
        json.dumps(description, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    smoke = run_synthetic_3d_smoke(
        description_path=description_path,
        output_directory=directory,
        blender_executable=blender_executable,
        generator_script=GENERATOR_SCRIPT,
    )
    generator = smoke["generator"]
    reference_id = next(
        item["id"] for item in description["cameras"] if item["role"] == "reference"
    )
    repeat_id = next(
        item["id"] for item in description["cameras"] if item["role"] == "repeat"
    )
    controls = build_surface_controls(
        generator,
        reference_camera_id=reference_id,
        repeat_camera_id=repeat_id,
    )
    reference_map = generator["reference_map"]
    control_layer = next(
        item
        for item in generator["rendered_control_layers"]
        if item["camera_id"] == repeat_id
    )
    reference_rgb = load_rgb(directory / "reference/orthographic_rgb.png")
    repeat_path = directory / "repeat" / repeat_id / "rgb.png"
    limits = description["acceptance_thresholds"]
    return {
        "description": description,
        "directory": directory,
        "smoke_passed": smoke["passed"],
        "reference_rgb": reference_rgb,
        "repeat_rgb": load_rgb(repeat_path),
        "repeat_rgb_path": repeat_path,
        "ground_mask_path": directory / control_layer["ground_mask"],
        "depth_path": directory / control_layer["depth"],
        "controls": controls,
        "limits": limits,
        "thresholds": MetricRecoveryThresholds.from_mapping(limits),
        "evaluation_arguments": {
            "reference_transform": Affine(*reference_map["affine"]),
            "reference_crs": reference_map["crs"],
            "reference_width_pixels": reference_rgb.shape[1],
            "reference_height_pixels": reference_rgb.shape[0],
        },
    }


def degradation_from_mapping(mapping: dict[str, Any]) -> ImageDegradation:
    """Берёт только параметры модели изображения, игнорируя метаданные case."""

    names = {field.name for field in ImageDegradation.__dataclass_fields__.values()}
    return ImageDegradation(
        **{name: mapping[name] for name in names if name in mapping}
    )


def classification_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Считает исходы gate без сокрытия отказов и ложных принятий."""

    labels = (
        "accepted_correct",
        "false_accept",
        "rejected_valid",
        "rejected_invalid",
        "alignment_failure",
    )
    return {
        label: sum(row["result"]["classification"] == label for row in rows)
        for label in labels
    }


def run_g3(
    template: dict[str, Any],
    output: Path,
    blender: Path,
) -> tuple[dict[str, Any], dict[float, dict[str, Any]]]:
    """Меняет только размер одной крыши и фактическую долю видимой земли."""

    contexts = {}
    rows = []
    height = float(template["g3"]["object_height_m"])
    for width in map(float, template["g3"]["object_widths_m"]):
        description = build_description(
            template,
            scenario_id=f"g3_width_{slug(width)}m",
            object_width_m=width,
            object_height_m=height,
            seed=int(template["seed"]),
        )
        context = generate_context(
            template,
            description=description,
            directory=output / "g3" / f"width_{slug(width)}m",
            blender_executable=blender,
        )
        contexts[width] = context
        result = evaluate_frame(
            context,
            context["repeat_rgb"],
            random_seed=int(template["seed"]),
        )
        rows.append(
            {
                "object_width_m": width,
                "result": result,
                "scene": str(context["directory"] / "scene.blend"),
                "smoke_passed": context["smoke_passed"],
            }
        )

    return {
        "experiment": "G3 visible-ground sweep",
        "changed_factor": "actual visible-ground fraction via roof footprint",
        "fixed_factors": [
            "camera pose",
            "object height",
            "illumination",
            "SIFT and RANSAC",
            "gate thresholds",
        ],
        "rows": rows,
        "classification_counts": classification_counts(rows),
    }, contexts


def run_g4(
    template: dict[str, Any],
    output: Path,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Проверяет каждое искажение отдельно на одной фиксированной сцене."""

    axes = template["g4"]["axes"]
    image_rows = []
    for axis_name, values in axes.items():
        if axis_name == "click_standard_deviation_px":
            continue
        for index, value in enumerate(values):
            degradation = ImageDegradation(**{axis_name: float(value)})
            degraded = apply_image_degradation(
                context["repeat_rgb"],
                degradation,
                random_seed=int(template["seed"]) + index,
            )
            path = output / "g4" / axis_name / f"{slug(float(value))}.png"
            save_rgb(path, degraded)
            image_rows.append(
                {
                    "axis": axis_name,
                    "value": float(value),
                    "degradation": asdict(degradation),
                    "result": evaluate_frame(
                        context,
                        degraded,
                        random_seed=int(template["seed"]) + index,
                    ),
                    "image": str(path),
                }
            )

    baseline = evaluate_frame(
        context,
        context["repeat_rgb"],
        random_seed=int(template["seed"]),
    )
    if baseline["alignment"] is None:
        raise RuntimeError("G4 click sweep требует успешный baseline alignment")
    alignment = align_frame_to_reference(
        context["reference_rgb"],
        context["repeat_rgb"],
    )
    click_rows = []
    trials = int(template["g4"]["click_trials"])
    for sigma_index, sigma in enumerate(
        map(float, axes["click_standard_deviation_px"])
    ):
        trial_summaries = []
        for trial in range(trials):
            perturbed = tuple(
                replace(
                    control,
                    frame_points_px=perturb_click_points(
                        control.frame_points_px,
                        standard_deviation_px=sigma,
                        random_seed=(
                            int(template["seed"])
                            + sigma_index * 10_000
                            + trial * 100
                            + control_index
                        ),
                    ),
                )
                if control.surface == "ground"
                else control
                for control_index, control in enumerate(context["controls"])
            )
            trial_context = {**context, "controls": perturbed}
            trial_summaries.append(
                evaluate_model(
                    trial_context,
                    alignment.homography_frame_to_reference,
                )["ground"]
            )
        percentiles = {
            key: {
                "p50": float(np.percentile([row[key] for row in trial_summaries], 50)),
                "p95": float(np.percentile([row[key] for row in trial_summaries], 95)),
                "maximum": float(max(row[key] for row in trial_summaries)),
            }
            for key in trial_summaries[0]
        }
        p95_summary = {key: value["p95"] for key, value in percentiles.items()}
        valid = within_limits(p95_summary, context["limits"])
        click_rows.append(
            {
                "axis": "click_standard_deviation_px",
                "value": sigma,
                "trials": trials,
                "error_percentiles": percentiles,
                "p95_within_limits": valid,
                "gate_accepted": baseline["gate_accepted"],
                "false_accept_at_p95": baseline["gate_accepted"] and not valid,
            }
        )

    return {
        "experiment": "G4 one-factor image and click quality sweep",
        "base_object_width_m": template["g4"]["base_object_width_m"],
        "image_rows": image_rows,
        "click_rows": click_rows,
        "classification_counts": classification_counts(image_rows),
    }


def run_g5(
    template: dict[str, Any],
    output: Path,
    contexts: dict[float, dict[str, Any]],
) -> dict[str, Any]:
    """Запускает только заранее перечисленные комбинации факторов."""

    rows = []
    for index, case in enumerate(template["g5"]["cases"]):
        width = float(case["object_width_m"])
        context = contexts[width]
        degradation = degradation_from_mapping(case)
        frame = apply_image_degradation(
            context["repeat_rgb"],
            degradation,
            random_seed=int(template["seed"]) + 50_000 + index,
        )
        path = output / "g5" / f"{case['id']}.png"
        save_rgb(path, frame)
        rows.append(
            {
                "id": case["id"],
                "object_width_m": width,
                "degradation": asdict(degradation),
                "result": evaluate_frame(
                    context,
                    frame,
                    random_seed=int(template["seed"]) + index,
                ),
                "image": str(path),
            }
        )
    return {
        "experiment": "G5 preregistered factor combinations",
        "rows": rows,
        "classification_counts": classification_counts(rows),
    }


def uniform(
    generator: np.random.Generator,
    ranges: dict[str, list[float]],
    name: str,
) -> float:
    """Выбирает одно значение внутри заранее объявленного диапазона."""

    low, high = ranges[name]
    return float(generator.uniform(low, high))


def run_g6(
    template: dict[str, Any],
    output: Path,
    blender: Path,
) -> dict[str, Any]:
    """Оценивает правило без перенастройки на отложенных случайных сценах."""

    specification = template["g6"]
    ranges = specification["ranges"]
    generator = np.random.default_rng(int(specification["held_out_seed"]))
    plans = []
    for index in range(int(specification["case_count"])):
        plan = {
            "id": f"held_out_{index:03d}",
            "object_width_m": uniform(generator, ranges, "object_width_m"),
            "object_height_m": uniform(generator, ranges, "object_height_m"),
            "camera_x_m": uniform(generator, ranges, "camera_x_m"),
            "camera_y_m": uniform(generator, ranges, "camera_y_m"),
            "camera_z_m": uniform(generator, ranges, "camera_z_m"),
            "target_x_m": uniform(generator, ranges, "target_x_m"),
            "target_y_m": uniform(generator, ranges, "target_y_m"),
            "focal_length_mm": uniform(generator, ranges, "focal_length_mm"),
            "exposure_stops": uniform(generator, ranges, "exposure_stops"),
            "shadow_fraction": uniform(generator, ranges, "shadow_fraction"),
            "blur_sigma_px": uniform(generator, ranges, "blur_sigma_px"),
            "noise_standard_deviation": uniform(
                generator, ranges, "noise_standard_deviation"
            ),
            "resolution_scale": uniform(generator, ranges, "resolution_scale"),
        }
        plans.append(plan)
    g6_directory = output / "g6"
    g6_directory.mkdir(parents=True, exist_ok=True)
    (g6_directory / "held_out_plan.json").write_text(
        json.dumps(plans, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows = []
    for index, plan in enumerate(plans):
        seed = int(specification["held_out_seed"]) + index
        description = build_description(
            template,
            scenario_id=plan["id"],
            object_width_m=plan["object_width_m"],
            object_height_m=plan["object_height_m"],
            seed=seed,
            camera_override=plan,
        )
        context = generate_context(
            template,
            description=description,
            directory=g6_directory / plan["id"],
            blender_executable=blender,
        )
        degradation = degradation_from_mapping(plan)
        frame = apply_image_degradation(
            context["repeat_rgb"],
            degradation,
            random_seed=seed,
        )
        path = context["directory"] / "degraded_repeat_rgb.png"
        save_rgb(path, frame)
        rows.append(
            {
                "id": plan["id"],
                "plan": plan,
                "smoke_passed": context["smoke_passed"],
                "result": evaluate_frame(context, frame, random_seed=seed),
                "scene": str(context["directory"] / "scene.blend"),
                "image": str(path),
            }
        )
    counts = classification_counts(rows)
    accepted = counts["accepted_correct"] + counts["false_accept"]
    return {
        "experiment": "G6 held-out random scenes and poses",
        "held_out_seed": specification["held_out_seed"],
        "rows": rows,
        "classification_counts": counts,
        "accepted_precision": (
            counts["accepted_correct"] / accepted if accepted else None
        ),
        "false_accept_rate_all_cases": counts["false_accept"] / len(rows),
    }


def save_plots(report: dict[str, Any], output: Path) -> dict[str, str]:
    """Сохраняет только графики ошибок и решений, а не декоративные overlay."""

    plot_directory = output / "plots"
    plot_directory.mkdir(parents=True, exist_ok=True)

    g3_rows = report["g3"]["rows"]
    fractions = [
        row["result"]["ground_mask"]["ground_pixel_fraction"] for row in g3_rows
    ]
    errors = []
    for row in g3_rows:
        ground = row["result"]["models"].get("sift_ransac", {}).get("ground")
        errors.append(
            ground["maximum_endpoint_position_error_m"]
            if ground is not None
            else np.nan
        )
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(fractions, errors, "o-")
    axis.axhline(0.2, color="black", linestyle=":", label="порог 0,20 м")
    axis.invert_xaxis()
    axis.set_xlabel("Фактическая доля видимой земли")
    axis.set_ylabel("Максимальная ошибка точки земли, м")
    axis.set_title("G3: вытеснение признаков земли")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    g3_path = plot_directory / "g3_ground_fraction.png"
    figure.savefig(g3_path, dpi=160)
    plt.close(figure)

    g6_rows = report["g6"]["rows"]
    g6_fractions = [
        row["result"]["ground_mask"]["ground_pixel_fraction"] for row in g6_rows
    ]
    g6_errors = []
    for row in g6_rows:
        ground = row["result"]["models"].get("sift_ransac", {}).get("ground")
        g6_errors.append(
            ground["maximum_endpoint_position_error_m"]
            if ground is not None
            else np.nan
        )
    colors = [
        "tab:red" if row["result"]["false_accept"] else "tab:blue"
        for row in g6_rows
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(g6_fractions, g6_errors, c=colors)
    axis.axhline(0.2, color="black", linestyle=":")
    axis.set_xlabel("Фактическая доля видимой земли")
    axis.set_ylabel("Максимальная ошибка точки земли, м")
    axis.set_title("G6: отложенные случайные сцены")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    g6_path = plot_directory / "g6_held_out.png"
    figure.savefig(g6_path, dpi=160)
    plt.close(figure)
    return {"g3": str(g3_path), "g6": str(g6_path)}


def run_suite(
    *,
    config_path: Path,
    output: Path,
    blender: Path,
) -> dict[str, Any]:
    """Выполняет G3--G6 последовательно и сохраняет один машинный отчёт."""

    template = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    frozen_config = output / "frozen_protocol.json"
    frozen_config.write_text(
        json.dumps(template, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    g3, contexts = run_g3(template, output, blender)
    g4_width = float(template["g4"]["base_object_width_m"])
    g4 = run_g4(template, output, contexts[g4_width])
    g5 = run_g5(template, output, contexts)
    g6 = run_g6(template, output, blender)
    report = {
        "schema_version": 1,
        "experiment": "G3--G6 synthetic robustness suite",
        "thresholds": template["acceptance_thresholds"],
        "g3": g3,
        "g4": g4,
        "g5": g5,
        "g6": g6,
        "limitations": [
            "Граница относится только к этой синтетике и SIFT+RANSAC baseline.",
            "G4 shadows are controlled raster half-planes, not ray-traced weather.",
            "Отсутствуют rolling shutter, lens distortion and real sensor artifacts.",
            "G6 has twenty held-out cases and estimates only a preliminary rate.",
        ],
    }
    report["plots"] = save_plots(report, output)
    report["protocol_checks"] = {
        "g3_all_smoke_passed": all(row["smoke_passed"] for row in g3["rows"]),
        "g3_ground_controls_visible": all(
            row["result"]["ground_mask"]["ground_control_endpoint_fraction"] == 1.0
            for row in g3["rows"]
        ),
        "g4_baseline_present": any(
            row["axis"] == "blur_sigma_px" and row["value"] == 0.0
            for row in g4["image_rows"]
        ),
        "g5_case_count_matches_config": (
            len(g5["rows"]) == len(template["g5"]["cases"])
        ),
        "g6_case_count_matches_config": (
            len(g6["rows"]) == int(template["g6"]["case_count"])
        ),
        "g6_all_smoke_passed": all(row["smoke_passed"] for row in g6["rows"]),
        "g6_ground_controls_visible": all(
            row["result"]["ground_mask"]["ground_control_endpoint_fraction"] == 1.0
            for row in g6["rows"]
        ),
    }
    report["completed"] = all(report["protocol_checks"].values())
    report_path = output / "robustness_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    """Печатает компактный итог и завершает ненулевым кодом при сбое протокола."""

    arguments = parse_arguments()
    if not arguments.blender.is_file():
        raise FileNotFoundError(arguments.blender)
    report = run_suite(
        config_path=arguments.config,
        output=arguments.output,
        blender=arguments.blender,
    )
    print(
        json.dumps(
            {
                "completed": report["completed"],
                "protocol_checks": report["protocol_checks"],
                "g3": report["g3"]["classification_counts"],
                "g4": report["g4"]["classification_counts"],
                "g5": report["g5"]["classification_counts"],
                "g6": report["g6"]["classification_counts"],
                "g6_accepted_precision": report["g6"]["accepted_precision"],
                "report": report["report_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
