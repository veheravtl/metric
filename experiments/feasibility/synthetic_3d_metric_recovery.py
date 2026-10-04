#!/usr/bin/env python3
"""G1: проверяет восстановление расстояний из одного неизвестного RGB-кадра.

Первая поза — ортографический эталон с известной метрической привязкой.
Вторая поза — перспективная камера дрона. Рабочая функция получает от неё
только RGB и координаты концов измеряемых отрезков в этом RGB. Параметры камеры
и мировые координаты открываются лишь после ответа для независимой оценки.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rasterio import Affine

from aerial_mapper.measurement import map_reference_points_to_world
from aerial_mapper.metric_recovery import (
    MetricRecoveryThresholds,
    ObservedSegment,
    recover_metric_segments,
)
from aerial_mapper.synthetic_3d import run_synthetic_3d_smoke

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESCRIPTION = PROJECT_ROOT / "experiments/configs/synthetic_3d_g1_metric.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/synthetic_3d/g1_metric"
DEFAULT_BLENDER = PROJECT_ROOT / ".tools/blender-4.5.14-linux-x64/blender"
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts/blender_generate_scene.py"
# Blender возвращает экранные координаты float32. Допуск 10 микрометров
# отделяет численное округление от содержательной ошибки геопривязки.
REFERENCE_GEOREFERENCING_TOLERANCE_M = 1e-5


def parse_arguments() -> argparse.Namespace:
    """Позволяет воспроизвести опыт с явными путями на другой машине."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--description", type=Path, default=DEFAULT_DESCRIPTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    """Читает PNG в едином для ядра порядке каналов RGB."""

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def projected_control_points(
    generator_metadata: dict[str, Any],
    *,
    camera_id: str,
) -> tuple[ObservedSegment, ...]:
    """Берёт идеальные клики из скрытой ветки Blender.

    Эти пиксели имитируют безошибочный выбор концов отрезка человеком. Их
    мировые координаты здесь не читаются и в рабочую функцию не передаются.
    """

    observed: list[ObservedSegment] = []
    for control in generator_metadata["metric_controls"]:
        projections = control["projections"][camera_id]
        if not all(point["visible"] for point in projections):
            raise RuntimeError(
                f"Контроль {control['id']} не целиком виден в кадре {camera_id}"
            )
        observed.append(
            ObservedSegment(
                name=control["id"],
                frame_points_px=np.asarray(
                    [point["pixel_xy"] for point in projections],
                    dtype=np.float64,
                ),
                surface="ground",
            )
        )
    return tuple(observed)


def evaluate_reference_georeferencing(
    generator_metadata: dict[str, Any],
    *,
    reference_camera_id: str,
    reference_transform: Affine,
    reference_crs: object,
) -> float:
    """Проверяет отдельно, что пиксели первой позы действительно метрические."""

    origin = np.asarray(
        generator_metadata["reference_map"]["origin_world_xy_in_crs_m"],
        dtype=np.float64,
    )
    errors: list[float] = []
    for control in generator_metadata["metric_controls"]:
        reference_pixels = np.asarray(
            [
                point["pixel_xy"]
                for point in control["projections"][reference_camera_id]
            ],
            dtype=np.float64,
        )
        mapped = map_reference_points_to_world(
            reference_pixels,
            reference_transform=reference_transform,
            reference_crs=reference_crs,
        )
        expected = (
            origin + np.asarray(control["points_world_m"], dtype=np.float64)[:, :2]
        )
        errors.extend(np.linalg.norm(mapped - expected, axis=1).tolist())
    return max(errors, default=0.0)


def evaluate_metric_answer(
    generator_metadata: dict[str, Any],
    recovery: Any,
) -> list[dict[str, Any]]:
    """Открывает скрытые мировые координаты только после рабочего ответа."""

    origin = np.asarray(
        generator_metadata["reference_map"]["origin_world_xy_in_crs_m"],
        dtype=np.float64,
    )
    truth_by_name = {
        control["id"]: control for control in generator_metadata["metric_controls"]
    }
    rows: list[dict[str, Any]] = []
    for recovered in recovery.segments:
        truth = truth_by_name[recovered.name]
        true_local = np.asarray(truth["points_world_m"], dtype=np.float64)[:, :2]
        true_world = origin + true_local
        estimated_world = recovered.measurement.mapped_points.world_points_m
        endpoint_errors = np.linalg.norm(estimated_world - true_world, axis=1)
        true_length = float(np.linalg.norm(true_world[1] - true_world[0]))
        estimated_length = recovered.measurement.length_meters
        absolute_error = abs(estimated_length - true_length)
        rows.append(
            {
                "id": recovered.name,
                "frame_points_px": (
                    recovered.measurement.mapped_points.frame_points_px.tolist()
                ),
                "estimated_reference_points_px": (
                    recovered.measurement.mapped_points.reference_points_px.tolist()
                ),
                "true_world_points_m": true_world.tolist(),
                "estimated_world_points_m": estimated_world.tolist(),
                "true_length_m": true_length,
                "estimated_length_m": estimated_length,
                "absolute_length_error_m": absolute_error,
                "relative_length_error_percent": absolute_error / true_length * 100.0,
                "endpoint_position_errors_m": endpoint_errors.tolist(),
                "maximum_endpoint_position_error_m": float(np.max(endpoint_errors)),
            }
        )
    return rows


def save_visualization(
    reference_rgb: np.ndarray,
    drone_rgb: np.ndarray,
    generator_metadata: dict[str, Any],
    recovery: Any,
    *,
    reference_camera_id: str,
    repeat_camera_id: str,
    output_path: Path,
) -> None:
    """Рисует идеальные клики и результат их переноса на эталон."""

    reference = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2BGR)
    drone = cv2.cvtColor(drone_rgb, cv2.COLOR_RGB2BGR)
    colors = [(40, 220, 255), (255, 120, 40), (70, 230, 80), (220, 80, 220)]
    truth_by_name = {
        control["id"]: control for control in generator_metadata["metric_controls"]
    }

    for index, recovered in enumerate(recovery.segments):
        color = colors[index % len(colors)]
        truth = truth_by_name[recovered.name]
        frame_points = np.asarray(
            [point["pixel_xy"] for point in truth["projections"][repeat_camera_id]]
        )
        estimated_reference = recovered.measurement.mapped_points.reference_points_px
        true_reference = np.asarray(
            [point["pixel_xy"] for point in truth["projections"][reference_camera_id]]
        )

        frame_integer = np.rint(frame_points).astype(int)
        cv2.line(drone, tuple(frame_integer[0]), tuple(frame_integer[1]), color, 4)
        for point in frame_integer:
            cv2.circle(drone, tuple(point), 7, color, -1)

        estimated_integer = np.rint(estimated_reference).astype(int)
        true_integer = np.rint(true_reference).astype(int)
        cv2.line(
            reference,
            tuple(estimated_integer[0]),
            tuple(estimated_integer[1]),
            color,
            4,
        )
        for estimated_point, true_point in zip(
            estimated_integer,
            true_integer,
            strict=True,
        ):
            cv2.circle(reference, tuple(true_point), 9, (0, 140, 255), 3)
            cv2.circle(reference, tuple(estimated_point), 5, color, -1)

    cv2.putText(
        reference,
        "REFERENCE: orange=true, color=estimated",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        drone,
        "DRONE RGB: ideal segment clicks",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    montage = np.concatenate((reference, drone), axis=1)
    if not cv2.imwrite(str(output_path), montage):
        raise RuntimeError(f"Не удалось записать визуализацию: {output_path}")


def run_experiment(
    *,
    description_path: Path,
    output_directory: Path,
    blender_executable: Path,
) -> dict[str, Any]:
    """Выполняет генерацию, рабочую привязку и независимую оценку."""

    smoke_report = run_synthetic_3d_smoke(
        description_path=description_path,
        output_directory=output_directory,
        blender_executable=blender_executable,
        generator_script=GENERATOR_SCRIPT,
    )
    description = json.loads(description_path.read_text(encoding="utf-8"))
    generator = smoke_report["generator"]
    reference_specification = next(
        item for item in description["cameras"] if item["role"] == "reference"
    )
    repeat_specification = next(
        item for item in description["cameras"] if item["role"] == "repeat"
    )
    reference_id = reference_specification["id"]
    repeat_id = repeat_specification["id"]

    reference_path = output_directory / "reference/orthographic_rgb.png"
    repeat_path = output_directory / "repeat" / repeat_id / "rgb.png"
    reference_rgb = load_rgb(reference_path)
    repeat_rgb = load_rgb(repeat_path)

    reference_map = generator["reference_map"]
    reference_transform = Affine(*reference_map["affine"])
    thresholds = MetricRecoveryThresholds.from_mapping(
        description["acceptance_thresholds"]
    )
    observed_segments = projected_control_points(generator, camera_id=repeat_id)

    # Рабочая граница: ниже нет ни позы Repeat-камеры, ни высоты, ни мировых
    # координат контрольных точек. Передаются только разрешённые данные.
    recovery = recover_metric_segments(
        reference_rgb,
        repeat_rgb,
        observed_segments,
        reference_transform=reference_transform,
        reference_crs=reference_map["crs"],
        thresholds=thresholds,
        random_seed=description["seed"],
    )

    # Только после завершения рабочей ветки открывается ground truth Blender.
    rows = evaluate_metric_answer(generator, recovery)
    reference_map_error = evaluate_reference_georeferencing(
        generator,
        reference_camera_id=reference_id,
        reference_transform=reference_transform,
        reference_crs=reference_map["crs"],
    )
    configured = description["acceptance_thresholds"]
    maximum_point_error = max(row["maximum_endpoint_position_error_m"] for row in rows)
    maximum_absolute_error = max(row["absolute_length_error_m"] for row in rows)
    maximum_relative_error = max(row["relative_length_error_percent"] for row in rows)
    metric_checks = {
        "reference_georeferencing_within_numerical_tolerance": (
            reference_map_error <= REFERENCE_GEOREFERENCING_TOLERANCE_M
        ),
        "point_position_error_within_limit": (
            maximum_point_error <= configured["maximum_point_position_error_m"]
        ),
        "absolute_length_error_within_limit": (
            maximum_absolute_error <= configured["maximum_segment_absolute_error_m"]
        ),
        "relative_length_error_within_limit": (
            maximum_relative_error
            <= configured["maximum_segment_relative_error_percent"]
        ),
    }

    visualization_path = output_directory / "metric_recovery_visualization.png"
    save_visualization(
        reference_rgb,
        repeat_rgb,
        generator,
        recovery,
        reference_camera_id=reference_id,
        repeat_camera_id=repeat_id,
        output_path=visualization_path,
    )

    quality = recovery.quality
    report = {
        "schema_version": 1,
        "scenario_id": description["scenario_id"],
        "passed": smoke_report["passed"] and all(metric_checks.values()),
        "input_contract": {
            "reference_pose": [
                "reference RGB",
                "projected CRS",
                "affine pixel-corner to map-metre transform",
            ],
            "drone_pose": [
                "RGB only",
                "two pixel coordinates per requested segment",
            ],
            "explicitly_not_used_by_working_branch": [
                "drone camera pose",
                "drone altitude",
                "focal length and sensor width",
                "metric control world coordinates",
            ],
        },
        "preregistered_thresholds": configured,
        "alignment": {
            "reference_keypoints": recovery.alignment.reference_keypoint_count,
            "frame_keypoints": recovery.alignment.frame_keypoint_count,
            "ratio_matches": recovery.alignment.ratio_match_count,
            "inliers": recovery.alignment.inlier_count,
            "inlier_ratio": recovery.alignment.inlier_ratio,
            "coverage_fraction": (recovery.alignment.inlier_spatial_coverage_fraction),
            "reprojection_p95_reference_px": (
                quality.inlier_reprojection_p95_reference_px
            ),
            "stability_p95_corner_shift_reference_px": (
                quality.stability_p95_max_corner_shift_reference_px
            ),
            "processing_time_seconds": (recovery.alignment.processing_time_seconds),
        },
        "reference_georeferencing_max_error_m": reference_map_error,
        "reference_georeferencing_tolerance_m": (REFERENCE_GEOREFERENCING_TOLERANCE_M),
        "segments": rows,
        "summary": {
            "maximum_endpoint_position_error_m": maximum_point_error,
            "maximum_absolute_length_error_m": maximum_absolute_error,
            "maximum_relative_length_error_percent": maximum_relative_error,
        },
        "checks": {
            "render_smoke_passed": smoke_report["passed"],
            **metric_checks,
        },
        "artifacts": {
            "reference_rgb": str(reference_path),
            "drone_rgb": str(repeat_path),
            "scene_blend": str(output_directory / "scene.blend"),
            "visualization": str(visualization_path),
        },
        "scope_limit": (
            "Доказана только идеальная плоская сцена с одним заранее выбранным "
            "эталоном и безошибочными кликами. Рельеф, здания, смена освещения, "
            "дисторсия, шум клика и неоднозначный поиск места пока не проверены."
        ),
    }
    report_path = output_directory / "metric_recovery_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    """Печатает компактный итог и завершает процесс ошибкой при провале."""

    arguments = parse_arguments()
    if not arguments.blender.is_file():
        raise FileNotFoundError(
            f"Blender не найден: {arguments.blender}. "
            "Выполните ./scripts/install_blender.sh"
        )
    report = run_experiment(
        description_path=arguments.description,
        output_directory=arguments.output,
        blender_executable=arguments.blender,
    )
    compact = {
        "scenario_id": report["scenario_id"],
        "passed": report["passed"],
        "alignment": report["alignment"],
        "summary": report["summary"],
        "checks": report["checks"],
        "segments": [
            {
                "id": row["id"],
                "true_length_m": row["true_length_m"],
                "estimated_length_m": row["estimated_length_m"],
                "absolute_length_error_m": row["absolute_length_error_m"],
                "relative_length_error_percent": (row["relative_length_error_percent"]),
            }
            for row in report["segments"]
        ],
        "report": str(arguments.output / "metric_recovery_report.json"),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
