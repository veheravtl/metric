#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""G9 pilot: рельеф, изменение камеры и граница одной гомографии."""

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

from aerial_mapper.alignment import AlignmentFailure, align_frame_to_reference
from aerial_mapper.metric_recovery import (
    MetricRecoveryThresholds,
    alignment_gate_failures,
)
from aerial_mapper.quality import analyze_alignment_quality
from aerial_mapper.synthetic_3d import run_synthetic_3d_smoke
from aerial_mapper.terrain_evaluation import (
    estimate_homography,
    point_error_summary,
    segment_error_summary,
    select_segment_pairs,
    spatial_group_labels,
    transform_points,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "experiments/configs/synthetic_3d_g9_terrain_pose.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/synthetic_3d/g9_terrain_pose"
DEFAULT_BLENDER = PROJECT_ROOT / ".tools/blender-4.5.14-linux-x64/blender"
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts/blender_generate_terrain_scene.py"


def parse_arguments() -> argparse.Namespace:
    """Разбирает воспроизводимые пути и необязательный поднабор поверхностей."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--surfaces", nargs="*", default=None)
    parser.add_argument(
        "--reuse-scenes",
        action="store_true",
        help="Переиспользовать сцену только при точном совпадении frozen config.",
    )
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    """Читает PNG как RGB uint8."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать RGB: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    """Читает контрольную маску видимой земли как uint8."""

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Не удалось прочитать маску: {path}")
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def sha256_file(path: Path) -> str:
    """Вычисляет отпечаток замороженной конфигурации."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_scene_description(
    protocol: dict[str, Any],
    surface: dict[str, Any],
) -> dict[str, Any]:
    """Разворачивает один случай рельефа в формат Blender-генератора."""

    world = protocol["world"]
    return {
        "schema_version": 1,
        "scenario_id": f"g9_{surface['id']}",
        # Протоколы G12--G13 содержат независимые texture/clutter seed.
        # Старые конфигурации не задают scene_seed и сохраняют прежнее поведение.
        "seed": int(surface.get("scene_seed", protocol["seed"])),
        "world": {
            "units": world["units"],
            "axis_convention": world["axis_convention"],
            "ground_size_m": world["ground_size_m"],
            "appearance": "terrain_metric_texture",
            "show_control_points": False,
            "metric_texture": {
                **world["metric_texture"],
                **surface.get("metric_texture", {}),
            },
            "terrain": {
                **surface["terrain"],
                "grid_vertices_xy": world["grid_vertices_xy"],
                "truth_sample_stride": world["truth_sample_stride"],
            },
        },
        "render": protocol["render"],
        "cameras": protocol["cameras"],
        "objects": [],
        "clutter": {
            **protocol.get("clutter_defaults", {}),
            **surface.get("clutter", {}),
        },
    }


def metric_result(
    estimated_xy_m: np.ndarray,
    true_xy_m: np.ndarray,
    limits: dict[str, Any],
) -> dict[str, Any]:
    """Считает точечные и независимые шестиметровые ошибки."""

    points = point_error_summary(estimated_xy_m, true_xy_m)
    pairs = select_segment_pairs(true_xy_m)
    segments = segment_error_summary(estimated_xy_m, true_xy_m, pairs)
    within_limits = bool(
        points.maximum_m <= limits["maximum_point_position_error_m"]
        and segments.maximum_absolute_m <= limits["maximum_segment_absolute_error_m"]
        and segments.maximum_relative_percent
        <= limits["maximum_segment_relative_error_percent"]
    )
    return {
        "within_metric_limits": within_limits,
        "points": asdict(points),
        "segments": asdict(segments),
    }


def camera_index(camera_ids: np.ndarray, camera_id: str) -> int:
    """Находит камеру в NPZ и явно сообщает о несовпадении метаданных."""

    matches = np.flatnonzero(camera_ids == camera_id)
    if matches.size != 1:
        raise RuntimeError(f"В terrain_truth нет единственной камеры {camera_id}")
    return int(matches[0])


def evaluate_surface(
    protocol: dict[str, Any],
    surface: dict[str, Any],
    scene_directory: Path,
) -> list[dict[str, Any]]:
    """Оценивает все Repeat-позы одной уже сгенерированной поверхности."""

    metadata = json.loads(
        (scene_directory / "generation_metadata.json").read_text(encoding="utf-8")
    )
    truth_path = scene_directory / metadata["terrain_truth"]["path"]
    with np.load(truth_path) as truth:
        world_xyz = np.asarray(truth["world_xyz_m"], dtype=np.float64)
        camera_ids = np.asarray(truth["camera_ids"])
        pixels = np.asarray(truth["pixel_xy"], dtype=np.float64)
        visible = np.asarray(truth["visible"], dtype=bool)

    teach = next(
        camera for camera in protocol["cameras"] if camera["role"] == "reference"
    )
    repeats = [camera for camera in protocol["cameras"] if camera["role"] == "repeat"]
    teach_index = camera_index(camera_ids, teach["id"])
    groups = spatial_group_labels(world_xyz[:, :2])
    teach_visible = visible[teach_index]
    calibration_mask = teach_visible & (groups == 0)
    if np.count_nonzero(calibration_mask) < 8:
        raise RuntimeError("Недостаточно видимых Teach-точек для калибровки")
    calibration_h = estimate_homography(
        pixels[teach_index, calibration_mask],
        world_xyz[calibration_mask, :2],
    )

    teach_rgb = load_rgb(scene_directory / "reference/perspective_rgb.png")
    teach_mask = load_mask(scene_directory / "reference/ground_mask.png")
    erosion = int(protocol["teach_mask_erosion_px"])
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * erosion + 1, 2 * erosion + 1)
    )
    feature_mask = cv2.erode(teach_mask, kernel, iterations=1)
    thresholds = MetricRecoveryThresholds.from_mapping(
        protocol["acceptance_thresholds"]
    )

    rows: list[dict[str, Any]] = []
    for repeat_number, repeat in enumerate(repeats):
        repeat_index = camera_index(camera_ids, repeat["id"])
        common = teach_visible & visible[repeat_index]
        fit_mask = common & (groups == 1)
        evaluation_mask = common & (groups == 2)
        common_count = int(np.count_nonzero(common))
        if np.count_nonzero(fit_mask) < 8 or np.count_nonzero(evaluation_mask) < 8:
            raise RuntimeError(f"Недостаточно общей видимой земли для {repeat['id']}")

        true_xy = world_xyz[evaluation_mask, :2]
        calibration_xy = transform_points(
            pixels[teach_index, evaluation_mask],
            calibration_h,
        )
        calibration_result = metric_result(
            calibration_xy, true_xy, protocol["acceptance_thresholds"]
        )

        oracle_repeat_to_teach = estimate_homography(
            pixels[repeat_index, fit_mask],
            pixels[teach_index, fit_mask],
        )
        oracle_teach_px = transform_points(
            pixels[repeat_index, evaluation_mask],
            oracle_repeat_to_teach,
        )
        oracle_xy = transform_points(oracle_teach_px, calibration_h)
        oracle_result = metric_result(
            oracle_xy, true_xy, protocol["acceptance_thresholds"]
        )

        repeat_rgb = load_rgb(scene_directory / "repeat" / repeat["id"] / "rgb.png")
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
            gate_failures = alignment_gate_failures(
                alignment,
                quality,
                thresholds=thresholds,
            )
            sift_teach_px = transform_points(
                pixels[repeat_index, evaluation_mask],
                alignment.homography_frame_to_reference,
            )
            sift_xy = transform_points(sift_teach_px, calibration_h)
            sift_metric = metric_result(
                sift_xy, true_xy, protocol["acceptance_thresholds"]
            )
            accepted = not gate_failures
            if accepted and sift_metric["within_metric_limits"]:
                classification = "accepted_correct"
            elif accepted:
                classification = "false_accept"
            elif sift_metric["within_metric_limits"]:
                classification = "rejected_valid"
            else:
                classification = "rejected_invalid"
            sift_result: dict[str, Any] = {
                "classification": classification,
                "gate_accepted": accepted,
                "gate_failures": list(gate_failures),
                "metric": sift_metric,
                "alignment": {
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
                },
            }
        except AlignmentFailure as error:
            sift_result = {
                "classification": "alignment_failure",
                "gate_accepted": False,
                "gate_failures": [str(error)],
                "metric": None,
                "alignment": None,
            }

        union_count = int(np.count_nonzero(teach_visible | visible[repeat_index]))
        rows.append(
            {
                "surface_id": surface["id"],
                "terrain": surface["terrain"],
                "repeat_id": repeat["id"],
                "common_visible_points": common_count,
                "common_visibility_iou": common_count / union_count,
                "evaluation_point_count": int(np.count_nonzero(evaluation_mask)),
                "calibration_only": calibration_result,
                "oracle_single_homography": oracle_result,
                "sift_ransac": sift_result,
            }
        )
    return rows


def save_summary_plot(
    rows: list[dict[str, Any]],
    surfaces: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Сохраняет компактное сравнение p95 oracle и SIFT по поверхностям."""

    surface_ids = [surface["id"] for surface in surfaces]
    repeat_ids = sorted({row["repeat_id"] for row in rows})
    figure, axes = plt.subplots(len(repeat_ids), 1, figsize=(12, 3 * len(repeat_ids)))
    axes_array = np.atleast_1d(axes)
    for axis, repeat_id in zip(axes_array, repeat_ids, strict=True):
        selected = [
            next(
                row
                for row in rows
                if row["surface_id"] == surface_id and row["repeat_id"] == repeat_id
            )
            for surface_id in surface_ids
        ]
        oracle = [
            row["oracle_single_homography"]["points"]["p95_m"] for row in selected
        ]
        sift = [
            (
                row["sift_ransac"]["metric"]["points"]["p95_m"]
                if row["sift_ransac"]["metric"] is not None
                else np.nan
            )
            for row in selected
        ]
        axis.plot(surface_ids, oracle, marker="o", label="точная одна H")
        axis.plot(surface_ids, sift, marker="x", label="SIFT + RANSAC")
        axis.axhline(0.2, color="red", linestyle="--", label="предел точки 0,2 м")
        axis.set_title(repeat_id)
        axis.set_ylabel("p95 ошибки X,Y, м")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Генерирует сцены, запускает три уровня оценки и пишет итог pilot."""

    arguments = parse_arguments()
    protocol = json.loads(arguments.config.read_text(encoding="utf-8"))
    selected_ids = set(arguments.surfaces) if arguments.surfaces else None
    surfaces = [
        surface
        for surface in protocol["surfaces"]
        if selected_ids is None or surface["id"] in selected_ids
    ]
    if not surfaces:
        raise ValueError("Не выбрано ни одной поверхности G9")
    arguments.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    smoke_results: dict[str, bool] = {}
    for surface in surfaces:
        scene_directory = arguments.output / "scenes" / surface["id"]
        scene_directory.mkdir(parents=True, exist_ok=True)
        scene_config = build_scene_description(protocol, surface)
        config_path = scene_directory / "frozen_scene_config.json"
        existing_matches = bool(
            config_path.exists()
            and json.loads(config_path.read_text(encoding="utf-8")) == scene_config
            and (scene_directory / "generation_metadata.json").exists()
        )
        config_path.write_text(
            json.dumps(scene_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if arguments.reuse_scenes and existing_matches:
            print(f"reused_surface={surface['id']}")
        else:
            smoke = run_synthetic_3d_smoke(
                description_path=config_path,
                output_directory=scene_directory,
                blender_executable=arguments.blender,
                generator_script=GENERATOR_SCRIPT,
            )
            if not smoke["passed"]:
                raise RuntimeError(f"Blender smoke не пройден для {surface['id']}")
        smoke_results[surface["id"]] = True
        rows.extend(evaluate_surface(protocol, surface, scene_directory))
        print(f"completed_surface={surface['id']}")

    counts: dict[str, int] = {}
    for row in rows:
        classification = row["sift_ransac"]["classification"]
        counts[classification] = counts.get(classification, 0) + 1
    planar_rows = [row for row in rows if row["terrain"]["kind"] in {"flat", "plane"}]
    if not planar_rows:
        raise RuntimeError("В протоколе отсутствует плоский контроль")
    planar_oracle_maximum = max(
        row["oracle_single_homography"]["points"]["maximum_m"] for row in planar_rows
    )
    success = protocol["preregistered_success"]
    preregistered_passed = bool(
        counts.get("false_accept", 0) <= success["maximum_false_accepts"]
        and planar_oracle_maximum <= success["maximum_planar_oracle_point_error_m"]
        and min(row["common_visible_points"] for row in rows)
        >= success["minimum_common_visible_points"]
    )

    plot_path = arguments.output / "g9_pilot_summary.png"
    save_summary_plot(rows, surfaces, plot_path)
    report = {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "configuration": str(arguments.config),
        "configuration_sha256": sha256_file(arguments.config),
        "surface_count": len(surfaces),
        "pair_count": len(rows),
        "smoke_results": smoke_results,
        "input_contract": {
            "working_algorithm_receives": [
                "perspective Teach RGB",
                "trusted Teach ground mask",
                "Teach pixel-to-XY calibration",
                "Repeat RGB",
            ],
            "evaluator_only": [
                "camera poses",
                "terrain mesh",
                "ray-cast visibility",
                "dense world coordinates and exact projections",
            ],
        },
        "frozen_thresholds": protocol["acceptance_thresholds"],
        "classification_counts": counts,
        "planar_oracle_maximum_point_error_m": planar_oracle_maximum,
        "preregistered_success": success,
        "preregistered_passed": preregistered_passed,
        "rows": rows,
        "visualization": str(plot_path),
        "limitations": [
            "Это pilot одной текстуры и одного seed, а не полевая гарантия.",
            "Рельеф гладкий; кусты, пни, здания и высокая трава не включены.",
            "Проверяется горизонтальная длина в X,Y, а не длина вдоль склона.",
        ],
    }
    report_path = arguments.output / "g9_pilot_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, ensure_ascii=False))
    print(f"preregistered_passed={preregistered_passed}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
