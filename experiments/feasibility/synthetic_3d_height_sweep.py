#!/usr/bin/env python3
"""G2: измеряет влияние высоты объекта на одну общую гомографию.

Для каждого уровня меняется только высота одного здания. Камеры, текстура
земли, освещение, SIFT, RANSAC и пороги остаются неизменными. Сравниваются три
матрицы: найденная по RGB, oracle только по земле и компромиссный oracle по
точкам земли и крыши.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from rasterio import Affine

from aerial_mapper.alignment import align_frame_to_reference
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = (
    PROJECT_ROOT / "experiments/configs/synthetic_3d_g2_height_sweep.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/synthetic_3d/g2_height_sweep"
DEFAULT_BLENDER = PROJECT_ROOT / ".tools/blender-4.5.14-linux-x64/blender"
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts/blender_generate_scene.py"


def parse_arguments() -> argparse.Namespace:
    """Разбирает пути к шаблону, Blender и каталогу производных артефактов."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    return parser.parse_args()


def height_slug(height_m: float) -> str:
    """Создаёт стабильную часть имени каталога без зависимости от locale."""

    return f"{height_m:.2f}".replace(".", "_")


def build_height_description(
    template: dict[str, Any],
    *,
    height_m: float,
) -> dict[str, Any]:
    """Подставляет одну высоту, не меняя остальные факторы сцены."""

    if height_m < 0.0:
        raise ValueError("Высота объекта не может быть отрицательной")
    description = copy.deepcopy(template)
    description.pop("sweep_heights_m", None)
    description["scenario_id"] = f"g2_height_{height_slug(height_m)}m"

    if height_m == 0.0:
        # Нулевая точка — прежняя единая плоскость. Крышные контроли остаются
        # на Z=0 и проверяют, что само деление на классы не создаёт ошибку.
        description["objects"] = []
    else:
        building = description["objects"][0]
        building["center_m"][2] = height_m / 2.0
        building["size_m"][2] = height_m

    for control in description["metric_controls"]:
        if control["surface"] == "roof":
            for point in control["points_world_m"]:
                point[2] = height_m
    return description


def load_rgb(path: Path) -> np.ndarray:
    """Читает рендер в порядке каналов RGB."""

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Не удалось прочитать RGB: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def build_surface_controls(
    generator: dict[str, Any],
    *,
    reference_camera_id: str,
    repeat_camera_id: str,
) -> tuple[SurfaceSegmentControl, ...]:
    """Преобразует скрытые проекции Blender в независимые контроли оценки."""

    origin = np.asarray(
        generator["reference_map"]["origin_world_xy_in_crs_m"],
        dtype=np.float64,
    )
    controls: list[SurfaceSegmentControl] = []
    for item in generator["metric_controls"]:
        repeat_projection = item["projections"][repeat_camera_id]
        reference_projection = item["projections"][reference_camera_id]
        if not all(point["visible"] for point in repeat_projection):
            raise RuntimeError(f"Контроль {item['id']} вышел из Repeat-кадра")
        if not all(point["visible"] for point in reference_projection):
            raise RuntimeError(f"Контроль {item['id']} вышел из эталона")
        true_local = np.asarray(item["points_world_m"], dtype=np.float64)[:, :2]
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
                true_world_points_m=origin + true_local,
            )
        )
    return tuple(controls)


def find_control_layer(
    generator: dict[str, Any],
    *,
    camera_id: str,
) -> dict[str, str]:
    """Возвращает пути контрольных слоёв заданной камеры."""

    return next(
        item
        for item in generator["rendered_control_layers"]
        if item["camera_id"] == camera_id
    )


def ground_mask_diagnostics(
    mask_path: Path,
    *,
    inlier_frame_points_px: np.ndarray,
) -> dict[str, float | int]:
    """Считает видимую долю земли и долю inlier, реально лежащих на земле."""

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Не удалось прочитать ground mask: {mask_path}")
    ground = mask >= 128
    height, width = ground.shape
    rounded = np.rint(inlier_frame_points_px).astype(np.int64)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
    inliers_on_ground = ground[rounded[:, 1], rounded[:, 0]]
    return {
        "ground_pixel_fraction": float(np.mean(ground)),
        "inlier_count_sampled": int(inliers_on_ground.size),
        "inlier_ground_fraction": float(np.mean(inliers_on_ground)),
    }


def surface_passes_thresholds(
    evaluations: list[dict[str, Any]],
    *,
    surface: str,
    configured: dict[str, Any],
) -> bool:
    """Проверяет все контроли поверхности по внешней метрической истине."""

    selected = [row for row in evaluations if row["surface"] == surface]
    if not selected:
        raise ValueError(f"Нет контролей поверхности {surface!r}")
    return all(
        row["maximum_endpoint_position_error_m"]
        <= configured["maximum_point_position_error_m"]
        and row["absolute_length_error_m"]
        <= configured["maximum_segment_absolute_error_m"]
        and row["relative_length_error_percent"]
        <= configured["maximum_segment_relative_error_percent"]
        for row in selected
    )


def summarize_surface(
    evaluations: list[dict[str, Any]],
    *,
    surface: str,
) -> dict[str, float]:
    """Возвращает худшие ошибки поверхности для компактной таблицы sweep."""

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


def run_height_case(
    *,
    template: dict[str, Any],
    height_m: float,
    case_directory: Path,
    blender_executable: Path,
) -> dict[str, Any]:
    """Генерирует и полностью оценивает один уровень высоты."""

    description = build_height_description(template, height_m=height_m)
    case_directory.mkdir(parents=True, exist_ok=True)
    description_path = case_directory / "input_scene_description.json"
    description_path.write_text(
        json.dumps(description, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    smoke = run_synthetic_3d_smoke(
        description_path=description_path,
        output_directory=case_directory,
        blender_executable=blender_executable,
        generator_script=GENERATOR_SCRIPT,
    )
    generator = smoke["generator"]
    reference_specification = next(
        item for item in description["cameras"] if item["role"] == "reference"
    )
    repeat_specification = next(
        item for item in description["cameras"] if item["role"] == "repeat"
    )
    reference_id = reference_specification["id"]
    repeat_id = repeat_specification["id"]

    reference_rgb = load_rgb(case_directory / "reference/orthographic_rgb.png")
    repeat_rgb = load_rgb(case_directory / "repeat" / repeat_id / "rgb.png")
    alignment = align_frame_to_reference(reference_rgb, repeat_rgb)
    frame_height, frame_width = repeat_rgb.shape[:2]
    quality = analyze_alignment_quality(
        alignment,
        frame_width_pixels=frame_width,
        frame_height_pixels=frame_height,
        random_seed=description["seed"],
    )
    thresholds = MetricRecoveryThresholds.from_mapping(
        description["acceptance_thresholds"]
    )
    gate_failures = alignment_gate_failures(
        alignment,
        quality,
        thresholds=thresholds,
    )

    controls = build_surface_controls(
        generator,
        reference_camera_id=reference_id,
        repeat_camera_id=repeat_id,
    )
    ground_oracle = estimate_oracle_homography(controls, surface="ground")
    all_surface_oracle = estimate_oracle_homography(controls)
    reference_map = generator["reference_map"]
    transform = Affine(*reference_map["affine"])
    common_evaluation_arguments = {
        "reference_transform": transform,
        "reference_crs": reference_map["crs"],
        "reference_width_pixels": reference_rgb.shape[1],
        "reference_height_pixels": reference_rgb.shape[0],
    }

    model_homographies = {
        "sift_ransac": alignment.homography_frame_to_reference,
        "ground_oracle": ground_oracle,
        "all_surface_oracle": all_surface_oracle,
    }
    models: dict[str, Any] = {}
    for model_name, homography in model_homographies.items():
        evaluations = [
            surface_evaluation_to_dict(item)
            for item in evaluate_surface_segments(
                controls,
                homography_frame_to_reference=homography,
                **common_evaluation_arguments,
            )
        ]
        models[model_name] = {
            "controls": evaluations,
            "ground": summarize_surface(evaluations, surface="ground"),
            "roof": summarize_surface(evaluations, surface="roof"),
            "ground_within_limits": surface_passes_thresholds(
                evaluations,
                surface="ground",
                configured=description["acceptance_thresholds"],
            ),
            "roof_within_limits": surface_passes_thresholds(
                evaluations,
                surface="roof",
                configured=description["acceptance_thresholds"],
            ),
        }

    control_layer = find_control_layer(generator, camera_id=repeat_id)
    mask_path = case_directory / control_layer["ground_mask"]
    depth_path = case_directory / control_layer["depth"]
    inlier_points = alignment.frame_points_px[alignment.inlier_mask]
    mask_diagnostics = ground_mask_diagnostics(
        mask_path,
        inlier_frame_points_px=inlier_points,
    )
    if not depth_path.is_file() or depth_path.stat().st_size == 0:
        raise RuntimeError(f"Blender не создал непустой Z-pass: {depth_path}")

    gate_accepted = not gate_failures
    sift_model = models["sift_ransac"]
    return {
        "height_m": height_m,
        "scenario_id": description["scenario_id"],
        "smoke_passed": smoke["passed"],
        "gate_accepted": gate_accepted,
        "gate_failures": list(gate_failures),
        "false_accept_roof": gate_accepted and not sift_model["roof_within_limits"],
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
        "ground_mask": {
            **mask_diagnostics,
            "path": str(mask_path),
        },
        "depth": {
            "path": str(depth_path),
            "file_size_bytes": depth_path.stat().st_size,
            "semantic": "Z-pass: distance to nearest visible surface",
        },
        "models": models,
        "artifacts": {
            "scene": str(case_directory / "scene.blend"),
            "reference_rgb": str(case_directory / "reference/orthographic_rgb.png"),
            "repeat_rgb": str(case_directory / "repeat" / repeat_id / "rgb.png"),
        },
    }


def first_height(
    rows: list[dict[str, Any]],
    predicate: Any,
) -> float | None:
    """Возвращает первую положительную высоту, удовлетворяющую условию."""

    for row in rows:
        if row["height_m"] > 0.0 and predicate(row):
            return float(row["height_m"])
    return None


def save_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Строит численную кривую ошибки, а не декоративный overlay."""

    heights = [row["height_m"] for row in rows]
    sift_ground = [
        row["models"]["sift_ransac"]["ground"]["maximum_endpoint_position_error_m"]
        for row in rows
    ]
    sift_roof = [
        row["models"]["sift_ransac"]["roof"]["maximum_endpoint_position_error_m"]
        for row in rows
    ]
    oracle_roof = [
        row["models"]["ground_oracle"]["roof"]["maximum_endpoint_position_error_m"]
        for row in rows
    ]

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(heights, sift_ground, "o-", label="SIFT: земля")
    axis.plot(heights, sift_roof, "o-", label="SIFT: крыша")
    axis.plot(heights, oracle_roof, "s--", label="Ground oracle: крыша")
    axis.axhline(0.2, color="black", linestyle=":", label="порог 0,20 м")
    axis.set_xlabel("Высота крыши, м")
    axis.set_ylabel("Максимальная ошибка конца отрезка, м")
    axis.set_title("G2: параллакс при одной гомографии")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_experiment(
    *,
    template_path: Path,
    output_directory: Path,
    blender_executable: Path,
) -> dict[str, Any]:
    """Запускает зафиксированный sweep и формирует причинный итог."""

    template = json.loads(template_path.read_text(encoding="utf-8"))
    heights = [float(value) for value in template["sweep_heights_m"]]
    if heights != sorted(set(heights)) or not heights or heights[0] != 0.0:
        raise ValueError("Sweep должен начинаться с 0 и содержать уникальные высоты")

    rows: list[dict[str, Any]] = []
    for height_m in heights:
        case_directory = output_directory / f"height_{height_slug(height_m)}m"
        rows.append(
            run_height_case(
                template=template,
                height_m=height_m,
                case_directory=case_directory,
                blender_executable=blender_executable,
            )
        )

    first_sift_roof_failure = first_height(
        rows,
        lambda row: not row["models"]["sift_ransac"]["roof_within_limits"],
    )
    first_oracle_roof_failure = first_height(
        rows,
        lambda row: not row["models"]["ground_oracle"]["roof_within_limits"],
    )
    first_false_accept = first_height(
        rows,
        lambda row: row["false_accept_roof"],
    )
    baseline = rows[0]
    all_ground_valid = all(
        row["models"]["sift_ransac"]["ground_within_limits"] for row in rows
    )
    protocol_checks = {
        "all_renders_passed_smoke": all(row["smoke_passed"] for row in rows),
        "baseline_ground_valid": (
            baseline["models"]["sift_ransac"]["ground_within_limits"]
        ),
        "baseline_roof_controls_valid_on_plane": (
            baseline["models"]["sift_ransac"]["roof_within_limits"]
        ),
        "ground_remained_valid_across_sweep": all_ground_valid,
        "roof_model_boundary_observed": first_oracle_roof_failure is not None,
        "control_layers_created": all(
            row["depth"]["file_size_bytes"] > 0
            and row["ground_mask"]["inlier_count_sampled"] > 0
            for row in rows
        ),
    }
    plot_path = output_directory / "height_sweep_errors.png"
    save_plot(rows, plot_path)

    report = {
        "schema_version": 1,
        "experiment": "G2: one-factor object-height sweep",
        "completed": all(protocol_checks.values()),
        "current_algorithm_safe_for_ground": all_ground_valid,
        "current_algorithm_safe_for_arbitrary_surfaces": not any(
            row["false_accept_roof"] for row in rows
        ),
        "fixed_factors": [
            "camera poses",
            "ground texture and illumination",
            "SIFT and RANSAC configuration",
            "acceptance thresholds",
            "building footprint",
        ],
        "changed_factor": "roof height in metres",
        "heights_m": heights,
        "thresholds": template["acceptance_thresholds"],
        "boundary": {
            "first_sift_roof_failure_height_m": first_sift_roof_failure,
            "first_ground_oracle_roof_failure_height_m": (first_oracle_roof_failure),
            "first_false_accept_roof_height_m": first_false_accept,
        },
        "protocol_checks": protocol_checks,
        "rows": rows,
        "artifacts": {
            "plot": str(plot_path),
            "report": str(output_directory / "height_sweep_report.json"),
        },
        "interpretation": (
            "Ground oracle isolates the limitation of one planar homography. "
            "If it fails on a roof while ground remains accurate, the cause is "
            "parallax rather than SIFT matching."
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "height_sweep_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    """Печатает границу применимости и не маскирует сбой протокола."""

    arguments = parse_arguments()
    if not arguments.blender.is_file():
        raise FileNotFoundError(
            f"Blender не найден: {arguments.blender}. "
            "Выполните ./scripts/install_blender.sh"
        )
    report = run_experiment(
        template_path=arguments.template,
        output_directory=arguments.output,
        blender_executable=arguments.blender,
    )
    compact_rows = []
    for row in report["rows"]:
        compact_rows.append(
            {
                "height_m": row["height_m"],
                "gate_accepted": row["gate_accepted"],
                "false_accept_roof": row["false_accept_roof"],
                "ground_error_m": row["models"]["sift_ransac"]["ground"][
                    "maximum_endpoint_position_error_m"
                ],
                "roof_error_m": row["models"]["sift_ransac"]["roof"][
                    "maximum_endpoint_position_error_m"
                ],
                "ground_oracle_roof_error_m": row["models"]["ground_oracle"]["roof"][
                    "maximum_endpoint_position_error_m"
                ],
                "inlier_ground_fraction": row["ground_mask"]["inlier_ground_fraction"],
            }
        )
    print(
        json.dumps(
            {
                "completed": report["completed"],
                "current_algorithm_safe_for_ground": (
                    report["current_algorithm_safe_for_ground"]
                ),
                "current_algorithm_safe_for_arbitrary_surfaces": (
                    report["current_algorithm_safe_for_arbitrary_surfaces"]
                ),
                "boundary": report["boundary"],
                "protocol_checks": report["protocol_checks"],
                "rows": compact_rows,
                "report": report["artifacts"]["report"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
