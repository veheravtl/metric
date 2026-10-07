#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""G14-A: радиальная дисторсия Repeat и коррекция известной калибровкой."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from aerial_mapper.alignment import AlignmentFailure, align_frame_to_reference
from aerial_mapper.camera_distortion import (
    BrownConradyDistortion,
    distort_image,
    distort_points,
    undistort_image,
    undistort_points,
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
from aerial_mapper.radiance_camera import (
    PinholeIntrinsics,
    blender_horizontal_sensor_intrinsics,
)
from aerial_mapper.teach_annotation import offset_mask_boundary
from aerial_mapper.terrain_evaluation import (
    point_error_summary,
    select_grid_anchor_indices,
    spatial_group_labels,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "experiments/configs/synthetic_3d_g14a_lens_distortion.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/synthetic_3d/g14a_lens_distortion"


def parse_arguments() -> argparse.Namespace:
    """Читает пути запуска; научные параметры остаются в frozen JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def project_path(value: str) -> Path:
    """Разрешает путь конфигурации относительно корня репозитория."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    """Вычисляет отпечаток фактически использованного протокола."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    """Читает PNG и явно переводит порядок каналов BGR в RGB."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать RGB: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    """Сохраняет диагностический RGB без неявной перестановки цветов."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Не удалось записать изображение: {path}")


def load_mask(path: Path) -> np.ndarray:
    """Читает контрольную маску как двоичный uint8."""

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Не удалось прочитать маску: {path}")
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def load_truth(scene_directory: Path) -> dict[str, np.ndarray]:
    """Читает плотную истину поверхности, недоступную рабочему алгоритму."""

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


def camera_by_id(protocol: dict[str, Any], camera_id: str) -> dict[str, Any]:
    """Находит описание камеры и запрещает неоднозначный id."""

    matches = [camera for camera in protocol["cameras"] if camera["id"] == camera_id]
    if len(matches) != 1:
        raise RuntimeError(f"В протоколе не найдена единственная камера {camera_id}")
    return matches[0]


def intrinsics_for_camera(
    protocol: dict[str, Any], camera_id: str
) -> PinholeIntrinsics:
    """Восстанавливает проверенные pinhole-intrinsics Blender в пикселях."""

    camera = camera_by_id(protocol, camera_id)
    render = protocol["render"]
    return blender_horizontal_sensor_intrinsics(
        width_px=int(render["width_pixels"]),
        height_px=int(render["height_pixels"]),
        focal_length_mm=float(camera["focal_length_mm"]),
        sensor_width_mm=float(camera["sensor_width_mm"]),
    )


def build_map_calibration(
    *,
    protocol: dict[str, Any],
    experiment: dict[str, Any],
    truth: dict[str, np.ndarray],
) -> tuple[PerspectiveReferenceCalibration, np.ndarray, np.ndarray]:
    """Калибрует Teach по 12 реперам и отделяет контрольную группу точек."""

    groups = spatial_group_labels(
        truth["world_xyz_m"][:, :2],
        group_count=int(experiment["point_group_count"]),
    )
    teach_id = next(
        camera["id"] for camera in protocol["cameras"] if camera["role"] == "reference"
    )
    teach_index = camera_index(truth["camera_ids"], teach_id)
    candidates = np.flatnonzero(truth["visible"][teach_index] & (groups == 0))
    grid = experiment["map_anchor_grid"]
    local_indices = select_grid_anchor_indices(
        truth["world_xyz_m"][candidates, :2],
        columns=int(grid["columns"]),
        rows=int(grid["rows"]),
        inset_fraction=float(grid["inset_fraction"]),
    )
    anchor_indices = candidates[local_indices]
    calibration = calibrate_perspective_reference(
        truth["pixel_xy"][teach_index, anchor_indices],
        truth["world_xyz_m"][anchor_indices, :2],
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
    """Оценивает гомографию и сохраняет все наблюдаемые причины отказа."""

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
        return alignment.homography_frame_to_reference, {
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


def in_frame_mask(points_px: np.ndarray, intrinsics: PinholeIntrinsics) -> np.ndarray:
    """Отмечает пиксели, реально доступные пользователю в выходном растре."""

    return (
        (points_px[:, 0] >= 0.0)
        & (points_px[:, 0] < intrinsics.width_px)
        & (points_px[:, 1] >= 0.0)
        & (points_px[:, 1] < intrinsics.height_px)
    )


def evaluate_metric_points(
    *,
    homography: np.ndarray,
    query_points_px: np.ndarray,
    truth_indices: np.ndarray,
    truth: dict[str, np.ndarray],
    calibration: PerspectiveReferenceCalibration,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Оценивает гомографию на независимых точках земли в метрах."""

    if truth_indices.size < 8:
        raise RuntimeError("Недостаточно видимых независимых точек для оценки")
    render = protocol["render"]
    estimated = map_frame_points_via_perspective_reference(
        query_points_px,
        homography,
        calibration=calibration,
        reference_width_pixels=int(render["width_pixels"]),
        reference_height_pixels=int(render["height_pixels"]),
    ).world_points_m
    summary = point_error_summary(
        estimated,
        truth["world_xyz_m"][truth_indices, :2],
    )
    return {"point_count": int(truth_indices.size), **asdict(summary)}


def classify_attempt(
    *,
    alignment_report: dict[str, Any],
    metric_report: dict[str, Any] | None,
    metric_failure: str | None,
    limit_m: float,
) -> str:
    """Различает точный ответ, ложное принятие и безопасный отказ."""

    if not alignment_report["alignment_constructed"]:
        return "alignment_failure"
    within = bool(
        metric_failure is None
        and metric_report is not None
        and metric_report["maximum_m"] <= limit_m
    )
    if alignment_report["gate_accepted"] and within:
        return "accepted_correct"
    if alignment_report["gate_accepted"]:
        return "false_accept"
    return "rejected_valid" if within else "rejected_invalid"


def classification_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Считает исходы в стабильном порядке, включая нулевые."""

    counter = Counter(row["classification"] for row in rows)
    labels = (
        "accepted_correct",
        "false_accept",
        "rejected_valid",
        "rejected_invalid",
        "alignment_failure",
    )
    return {label: counter[label] for label in labels}


def point_round_trip_error(
    intrinsics: PinholeIntrinsics,
    distortion: BrownConradyDistortion,
) -> float:
    """Проверяет согласованность прямой и обратной модели на сетке кадра."""

    x, y = np.meshgrid(
        np.linspace(0.0, intrinsics.width_px - 1.0, 9),
        np.linspace(0.0, intrinsics.height_px - 1.0, 7),
    )
    points = np.column_stack((x.ravel(), y.ravel()))
    distorted = distort_points(
        points,
        intrinsics=intrinsics,
        distortion=distortion,
    )
    restored = undistort_points(
        distorted,
        intrinsics=intrinsics,
        distortion=distortion,
    )
    return float(np.linalg.norm(restored - points, axis=1).max())


def maximum_edge_shift(
    intrinsics: PinholeIntrinsics,
    distortion: BrownConradyDistortion,
) -> float:
    """Переводит абстрактный коэффициент в понятный сдвиг края в пикселях."""

    points = np.asarray(
        [
            [0.0, 0.0],
            [intrinsics.width_px - 1.0, 0.0],
            [0.0, intrinsics.height_px - 1.0],
            [intrinsics.width_px - 1.0, intrinsics.height_px - 1.0],
        ]
    )
    distorted = distort_points(
        points,
        intrinsics=intrinsics,
        distortion=distortion,
    )
    return float(np.linalg.norm(distorted - points, axis=1).max())


def build_summary_figure(
    rows: list[dict[str, Any]],
    *,
    limit_m: float,
    output_path: Path,
) -> None:
    """Строит численную карту ошибки по позам, k1 и конвейерам."""

    pose_ids = sorted({row["repeat_id"] for row in rows})
    figure, axes = plt.subplots(1, len(pose_ids), figsize=(16, 5), sharey=True)
    if len(pose_ids) == 1:
        axes = [axes]
    colours = {"raw": "#c43c39", "calibrated": "#2878b5"}
    markers = {True: "o", False: "x"}
    for axis, pose_id in zip(axes, pose_ids, strict=True):
        for pipeline in ("raw", "calibrated"):
            selected = [
                row
                for row in rows
                if row["repeat_id"] == pose_id
                and row["pipeline"] == pipeline
                and row["metric"] is not None
            ]
            for accepted in (True, False):
                subset = [
                    row
                    for row in selected
                    if row["alignment"]["gate_accepted"] is accepted
                ]
                if not subset:
                    continue
                axis.scatter(
                    [row["k1"] for row in subset],
                    [row["metric"]["maximum_m"] for row in subset],
                    color=colours[pipeline],
                    marker=markers[accepted],
                    alpha=0.72,
                    label=f"{pipeline}, {'принято' if accepted else 'отказ'}",
                )
        axis.axhline(limit_m, color="black", linestyle="--", linewidth=1)
        axis.set_title(pose_id)
        axis.set_xlabel("k1")
        axis.grid(alpha=0.25)
        axis.set_yscale("symlog", linthresh=0.01)
    axes[0].set_ylabel("максимальная ошибка точки, м")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4)
    figure.suptitle("G14-A: дисторсия, коррекция и решение gate")
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def main() -> None:
    """Выполняет frozen sweep без повторного запуска Blender."""

    arguments = parse_arguments()
    experiment = json.loads(arguments.config.read_text(encoding="utf-8"))
    source_protocol_path = project_path(experiment["source_protocol"])
    source_output = project_path(experiment["source_output"])
    protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    arguments.output.mkdir(parents=True, exist_ok=True)
    thresholds = MetricRecoveryThresholds.from_mapping(
        protocol["acceptance_thresholds"]
    )
    limit_m = float(experiment["maximum_point_position_error_m"])
    teach_id = next(
        camera["id"] for camera in protocol["cameras"] if camera["role"] == "reference"
    )
    rows: list[dict[str, Any]] = []
    calibration_reports: dict[str, Any] = {}
    round_trip_by_case: dict[str, float] = {}
    edge_shift_by_case: dict[str, float] = {}

    for surface_number, surface_id in enumerate(experiment["surface_ids"]):
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
        teach_rgb = load_rgb(scene_directory / "reference/perspective_rgb.png")
        safe_mask = offset_mask_boundary(
            load_mask(scene_directory / "reference/ground_mask.png"),
            offset_pixels=-int(experiment["safe_mask_erosion_px"]),
        )

        for repeat_number, repeat_id in enumerate(experiment["repeat_camera_ids"]):
            repeat_index = camera_index(truth["camera_ids"], repeat_id)
            intrinsics = intrinsics_for_camera(protocol, repeat_id)
            nominal_rgb = load_rgb(
                scene_directory / "repeat" / repeat_id / "rgb.png"
            )
            common_indices = np.flatnonzero(
                truth["visible"][teach_index]
                & truth["visible"][repeat_index]
                & (groups == 2)
            )
            nominal_points = truth["pixel_xy"][repeat_index, common_indices]

            for case_number, case in enumerate(experiment["distortion_cases"]):
                case_id = str(case["id"])
                distortion = BrownConradyDistortion(k1=float(case["k1"]))
                round_trip_by_case.setdefault(
                    case_id, point_round_trip_error(intrinsics, distortion)
                )
                edge_shift_by_case.setdefault(
                    case_id, maximum_edge_shift(intrinsics, distortion)
                )
                distorted_rgb = distort_image(
                    nominal_rgb,
                    intrinsics=intrinsics,
                    distortion=distortion,
                )
                distorted_points_px = distort_points(
                    nominal_points,
                    intrinsics=intrinsics,
                    distortion=distortion,
                )
                visible_in_distorted = in_frame_mask(distorted_points_px, intrinsics)
                evaluation_indices = common_indices[visible_in_distorted]
                pipelines = ["raw"] if distortion.k1 == 0.0 else experiment["pipelines"]

                for pipeline_number, pipeline in enumerate(pipelines):
                    if pipeline == "raw":
                        repeat_rgb = distorted_rgb
                        query_points = distorted_points_px[visible_in_distorted]
                    elif pipeline == "calibrated":
                        repeat_rgb = undistort_image(
                            distorted_rgb,
                            intrinsics=intrinsics,
                            distortion=distortion,
                        )
                        query_points = nominal_points[visible_in_distorted]
                    else:
                        raise ValueError(f"Неизвестный pipeline: {pipeline}")

                    random_seed = (
                        int(experiment["seed"])
                        + surface_number * 10_000
                        + repeat_number * 100
                        + case_number * 10
                        + pipeline_number
                    )
                    homography, alignment_report = estimate_alignment(
                        teach_rgb=teach_rgb,
                        repeat_rgb=repeat_rgb,
                        feature_mask=safe_mask,
                        thresholds=thresholds,
                        random_seed=random_seed,
                    )
                    metric_report = None
                    metric_failure = None
                    if homography is not None:
                        try:
                            metric_report = evaluate_metric_points(
                                homography=homography,
                                query_points_px=query_points,
                                truth_indices=evaluation_indices,
                                truth=truth,
                                calibration=calibration,
                                protocol=protocol,
                            )
                        except MeasurementFailure as error:
                            metric_failure = str(error)
                    classification = classify_attempt(
                        alignment_report=alignment_report,
                        metric_report=metric_report,
                        metric_failure=metric_failure,
                        limit_m=limit_m,
                    )
                    rows.append(
                        {
                            "surface_id": surface_id,
                            "repeat_id": repeat_id,
                            "case_id": case_id,
                            "k1": distortion.k1,
                            "pipeline": pipeline,
                            "edge_shift_px": edge_shift_by_case[case_id],
                            "evaluation_point_count": int(evaluation_indices.size),
                            "alignment": alignment_report,
                            "metric": metric_report,
                            "metric_failure": metric_failure,
                            "classification": classification,
                        }
                    )
                    if (
                        surface_number == 0
                        and repeat_id == "repeat_same"
                        and abs(distortion.k1) == 0.1
                    ):
                        save_rgb(
                            arguments.output
                            / "diagnostics"
                            / f"{case_id}_{pipeline}.png",
                            repeat_rgb,
                        )
            print(f"completed={surface_id}:{repeat_id}", flush=True)

    nominal_rows = [row for row in rows if row["k1"] == 0.0]
    calibrated_rows = [
        row for row in rows if row["k1"] != 0.0 and row["pipeline"] == "calibrated"
    ]
    raw_rows = [
        row for row in rows if row["k1"] != 0.0 and row["pipeline"] == "raw"
    ]
    nominal_counts = classification_counts(nominal_rows)
    calibrated_counts = classification_counts(calibrated_rows)
    raw_counts = classification_counts(raw_rows)
    validity = experiment["preregistered_validity"]
    maximum_round_trip = max(round_trip_by_case.values())
    validity_failures: list[str] = []
    if maximum_round_trip > float(validity["maximum_point_round_trip_error_px"]):
        validity_failures.append("Превышена ошибка цикла координат")
    if nominal_counts["accepted_correct"] < int(
        validity["minimum_nominal_accepted_correct"]
    ):
        validity_failures.append("Недостаточно принятых номинальных контролей")
    if nominal_counts["false_accept"] > int(
        validity["maximum_nominal_false_accepts"]
    ):
        validity_failures.append("Есть ложные принятия в номинальном контроле")
    if calibrated_counts["accepted_correct"] < int(
        validity["minimum_calibrated_accepted_correct"]
    ):
        validity_failures.append("Калибровка восстановила слишком мало случаев")
    if calibrated_counts["false_accept"] > int(
        validity["maximum_calibrated_false_accepts"]
    ):
        validity_failures.append("Калиброванный конвейер имеет ложные принятия")
    raw_safety_passed = raw_counts["false_accept"] <= int(
        experiment["preregistered_raw_safety"]["maximum_false_accepts"]
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
        "attempt_count": len(rows),
        "maximum_point_round_trip_error_px": maximum_round_trip,
        "point_round_trip_error_by_case_px": round_trip_by_case,
        "edge_shift_by_case_px": edge_shift_by_case,
        "nominal_classification_counts": nominal_counts,
        "raw_classification_counts": raw_counts,
        "calibrated_classification_counts": calibrated_counts,
        "preregistered_validity": validity,
        "validity_failures": validity_failures,
        "experiment_valid": not validity_failures,
        "preregistered_raw_safety": experiment["preregistered_raw_safety"],
        "raw_safety_passed": raw_safety_passed,
        "calibrations": calibration_reports,
        "rows": rows,
        "primary_sources": experiment["primary_sources"],
        "limitations": [
            "Изменялся только k1 при идеальных остальных intrinsics.",
            "Teach оставался идеальным pinhole-кадром.",
            "Точная коррекция является верхней границей качества.",
            "Три сцены принадлежат одной семье богатой синтетической текстуры.",
            "Построчное считывание и движение объектов не моделировались.",
        ],
    }
    report_path = arguments.output / "g14a_lens_distortion_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    build_summary_figure(
        rows,
        limit_m=limit_m,
        output_path=arguments.output / "g14a_lens_distortion_summary.png",
    )
    print(
        json.dumps(
            {
                "nominal": nominal_counts,
                "raw": raw_counts,
                "calibrated": calibrated_counts,
            },
            ensure_ascii=False,
        )
    )
    print(f"experiment_valid={not validity_failures}")
    print(f"raw_safety_passed={raw_safety_passed}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
