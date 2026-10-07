#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""G14-A2: слабая дисторсия Repeat и ошибка оценённого коэффициента k1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from aerial_mapper.camera_distortion import (
    BrownConradyDistortion,
    distort_image,
    distort_points,
    undistort_image,
    undistort_points,
)
from aerial_mapper.measurement import MeasurementFailure
from aerial_mapper.metric_recovery import MetricRecoveryThresholds
from aerial_mapper.radiance_camera import PinholeIntrinsics
from aerial_mapper.teach_annotation import offset_mask_boundary
from experiments.feasibility.synthetic_3d_lens_distortion import (
    build_map_calibration,
    camera_index,
    classification_counts,
    classify_attempt,
    estimate_alignment,
    evaluate_metric_points,
    in_frame_mask,
    intrinsics_for_camera,
    load_mask,
    load_rgb,
    load_truth,
    maximum_edge_shift,
    point_round_trip_error,
    project_path,
    save_rgb,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "experiments/configs/synthetic_3d_g14a2_calibration_uncertainty.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/synthetic_3d/g14a2_calibration_uncertainty"


def parse_arguments() -> argparse.Namespace:
    """Читает только пути; научные параметры остаются в frozen JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def coefficient_id(value: float) -> str:
    """Даёт коэффициенту устойчивое имя для отчётов и диагностик."""

    sign = "p" if value >= 0.0 else "m"
    magnitude = f"{abs(value):.4f}".replace(".", "_")
    return f"{sign}{magnitude}"


def calibration_grid(intrinsics: PinholeIntrinsics) -> np.ndarray:
    """Строит сетку по всему кадру для измерения остаточной геометрии."""

    x, y = np.meshgrid(
        np.linspace(0.0, intrinsics.width_px - 1.0, 17),
        np.linspace(0.0, intrinsics.height_px - 1.0, 13),
    )
    return np.column_stack((x.ravel(), y.ravel()))


def maximum_residual_shift(
    intrinsics: PinholeIntrinsics,
    *,
    true_distortion: BrownConradyDistortion,
    estimated_distortion: BrownConradyDistortion,
) -> float:
    """Измеряет остаток цикла «истинно исказить → оценённо исправить».

    Значение выражено в пикселях исходной pinhole-системы. Нулевая ошибка
    калибровки должна вернуть сетку практически точно; ненулевая оставляет
    систематическое смещение, которое затем пытается приблизить гомография.
    """

    points = calibration_grid(intrinsics)
    observed = distort_points(
        points,
        intrinsics=intrinsics,
        distortion=true_distortion,
    )
    corrected = undistort_points(
        observed,
        intrinsics=intrinsics,
        distortion=estimated_distortion,
    )
    return float(np.linalg.norm(corrected - points, axis=1).max())


def run_attempt(
    *,
    teach_rgb: np.ndarray,
    repeat_rgb: np.ndarray,
    safe_mask: np.ndarray,
    thresholds: MetricRecoveryThresholds,
    query_points_px: np.ndarray,
    truth_indices: np.ndarray,
    truth: dict[str, np.ndarray],
    calibration: Any,
    protocol: dict[str, Any],
    limit_m: float,
    random_seed: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None, str]:
    """Запускает неизменённые alignment, gate и метрическую оценку."""

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
                query_points_px=query_points_px,
                truth_indices=truth_indices,
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
    return alignment_report, metric_report, metric_failure, classification


def grouped_classifications(
    rows: list[dict[str, Any]], field: str
) -> dict[str, dict[str, int]]:
    """Считает исходы по одному параметру без потери нулевых категорий."""

    values = sorted({row[field] for row in rows})
    return {
        str(value): classification_counts(
            [row for row in rows if row[field] == value]
        )
        for value in values
    }


def first_unsafe_shift(rows: list[dict[str, Any]]) -> float | None:
    """Возвращает минимальный наблюдённый остаток среди ложных принятий."""

    shifts = [
        float(row["residual_grid_shift_px"])
        for row in rows
        if row["classification"] == "false_accept"
    ]
    return min(shifts) if shifts else None


def build_summary_figure(
    rows: list[dict[str, Any]], *, limit_m: float, output_path: Path
) -> None:
    """Показывает границу raw и остаток неточной коррекции."""

    colours = {
        "accepted_correct": "#2a8f4e",
        "false_accept": "#c43c39",
        "rejected_valid": "#d49a27",
        "rejected_invalid": "#777777",
        "alignment_failure": "#222222",
    }
    labels = {
        "accepted_correct": "корректно принято",
        "false_accept": "ложно принято",
        "rejected_valid": "отказ от допустимого",
        "rejected_invalid": "правильно отклонено",
        "alignment_failure": "alignment не построен",
    }
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    panels = (
        ("raw_boundary", "true_k1", "raw: истинный k1"),
        (
            "calibration_error",
            "residual_grid_shift_px",
            "коррекция: остаточный сдвиг, px",
        ),
    )
    for axis, (sweep, x_field, x_label) in zip(axes, panels, strict=True):
        selected = [
            row
            for row in rows
            if row["sweep"] == sweep and row["metric"] is not None
        ]
        for classification, colour in colours.items():
            subset = [
                row for row in selected if row["classification"] == classification
            ]
            if not subset:
                continue
            axis.scatter(
                [row[x_field] for row in subset],
                [row["metric"]["maximum_m"] for row in subset],
                color=colour,
                alpha=0.68,
                label=labels[classification],
            )
        axis.axhline(limit_m, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel(x_label)
        axis.grid(alpha=0.25)
        axis.set_yscale("symlog", linthresh=0.01)
    axes[0].set_title("Слабая неисправленная дисторсия")
    axes[1].set_title("Ошибка коэффициента калибровки")
    axes[0].set_ylabel("максимальная ошибка точки, м")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    second_handles, second_labels = axes[1].get_legend_handles_labels()
    for handle, label in zip(second_handles, second_labels, strict=True):
        if label not in legend_labels:
            handles.append(handle)
            legend_labels.append(label)
    figure.legend(handles, legend_labels, loc="upper center", ncol=5)
    figure.suptitle("G14-A2: слабая дисторсия и неточная коррекция")
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def main() -> None:
    """Выполняет два замороженных sweep без повторного запуска Blender."""

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
        camera["id"]
        for camera in protocol["cameras"]
        if camera["role"] == "reference"
    )
    rows: list[dict[str, Any]] = []
    calibration_reports: dict[str, Any] = {}
    exact_round_trip_by_k1: dict[str, float] = {}
    true_edge_shift_by_k1: dict[str, float] = {}

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
            nominal_visible = in_frame_mask(nominal_points, intrinsics)
            nominal_indices = common_indices[nominal_visible]
            nominal_query = nominal_points[nominal_visible]
            base_seed = (
                int(experiment["seed"])
                + surface_number * 100_000
                + repeat_number * 10_000
            )

            alignment, metric, failure, classification = run_attempt(
                teach_rgb=teach_rgb,
                repeat_rgb=nominal_rgb,
                safe_mask=safe_mask,
                thresholds=thresholds,
                query_points_px=nominal_query,
                truth_indices=nominal_indices,
                truth=truth,
                calibration=calibration,
                protocol=protocol,
                limit_m=limit_m,
                random_seed=base_seed,
            )
            rows.append(
                {
                    "surface_id": surface_id,
                    "repeat_id": repeat_id,
                    "sweep": "nominal",
                    "case_id": "nominal_k1_0",
                    "true_k1": 0.0,
                    "estimated_k1": 0.0,
                    "calibration_error_k1": 0.0,
                    "true_edge_shift_px": 0.0,
                    "residual_grid_shift_px": 0.0,
                    "evaluation_point_count": int(nominal_indices.size),
                    "alignment": alignment,
                    "metric": metric,
                    "metric_failure": failure,
                    "classification": classification,
                }
            )

            for case_number, true_k1 in enumerate(experiment["raw_boundary_k1"]):
                true_k1 = float(true_k1)
                true_distortion = BrownConradyDistortion(k1=true_k1)
                key = str(true_k1)
                exact_round_trip_by_k1.setdefault(
                    key, point_round_trip_error(intrinsics, true_distortion)
                )
                true_edge_shift_by_k1.setdefault(
                    key, maximum_edge_shift(intrinsics, true_distortion)
                )
                distorted_rgb = distort_image(
                    nominal_rgb,
                    intrinsics=intrinsics,
                    distortion=true_distortion,
                )
                distorted_points = distort_points(
                    nominal_points,
                    intrinsics=intrinsics,
                    distortion=true_distortion,
                )
                visible = in_frame_mask(distorted_points, intrinsics)
                evaluation_indices = common_indices[visible]
                query_points = distorted_points[visible]
                alignment, metric, failure, classification = run_attempt(
                    teach_rgb=teach_rgb,
                    repeat_rgb=distorted_rgb,
                    safe_mask=safe_mask,
                    thresholds=thresholds,
                    query_points_px=query_points,
                    truth_indices=evaluation_indices,
                    truth=truth,
                    calibration=calibration,
                    protocol=protocol,
                    limit_m=limit_m,
                    random_seed=base_seed + 100 + case_number,
                )
                rows.append(
                    {
                        "surface_id": surface_id,
                        "repeat_id": repeat_id,
                        "sweep": "raw_boundary",
                        "case_id": f"raw_true_{coefficient_id(true_k1)}",
                        "true_k1": true_k1,
                        "estimated_k1": None,
                        "calibration_error_k1": None,
                        "true_edge_shift_px": true_edge_shift_by_k1[key],
                        "residual_grid_shift_px": true_edge_shift_by_k1[key],
                        "evaluation_point_count": int(evaluation_indices.size),
                        "alignment": alignment,
                        "metric": metric,
                        "metric_failure": failure,
                        "classification": classification,
                    }
                )

            for true_number, true_k1 in enumerate(
                experiment["calibration_true_k1"]
            ):
                true_k1 = float(true_k1)
                true_distortion = BrownConradyDistortion(k1=true_k1)
                key = str(true_k1)
                exact_round_trip_by_k1.setdefault(
                    key, point_round_trip_error(intrinsics, true_distortion)
                )
                true_edge_shift_by_k1.setdefault(
                    key, maximum_edge_shift(intrinsics, true_distortion)
                )
                distorted_rgb = distort_image(
                    nominal_rgb,
                    intrinsics=intrinsics,
                    distortion=true_distortion,
                )
                distorted_points = distort_points(
                    nominal_points,
                    intrinsics=intrinsics,
                    distortion=true_distortion,
                )
                visible_in_distorted = in_frame_mask(distorted_points, intrinsics)

                for error_number, error_k1 in enumerate(
                    experiment["calibration_error_k1"]
                ):
                    error_k1 = float(error_k1)
                    estimated_k1 = true_k1 + error_k1
                    estimated_distortion = BrownConradyDistortion(k1=estimated_k1)
                    corrected_rgb = undistort_image(
                        distorted_rgb,
                        intrinsics=intrinsics,
                        distortion=estimated_distortion,
                    )
                    corrected_points = undistort_points(
                        distorted_points,
                        intrinsics=intrinsics,
                        distortion=estimated_distortion,
                    )
                    visible = visible_in_distorted & in_frame_mask(
                        corrected_points, intrinsics
                    )
                    evaluation_indices = common_indices[visible]
                    query_points = corrected_points[visible]
                    residual_shift = maximum_residual_shift(
                        intrinsics,
                        true_distortion=true_distortion,
                        estimated_distortion=estimated_distortion,
                    )
                    alignment, metric, failure, classification = run_attempt(
                        teach_rgb=teach_rgb,
                        repeat_rgb=corrected_rgb,
                        safe_mask=safe_mask,
                        thresholds=thresholds,
                        query_points_px=query_points,
                        truth_indices=evaluation_indices,
                        truth=truth,
                        calibration=calibration,
                        protocol=protocol,
                        limit_m=limit_m,
                        random_seed=(
                            base_seed + 1000 + true_number * 100 + error_number
                        ),
                    )
                    rows.append(
                        {
                            "surface_id": surface_id,
                            "repeat_id": repeat_id,
                            "sweep": "calibration_error",
                            "case_id": (
                                f"true_{coefficient_id(true_k1)}_"
                                f"error_{coefficient_id(error_k1)}"
                            ),
                            "true_k1": true_k1,
                            "estimated_k1": estimated_k1,
                            "calibration_error_k1": error_k1,
                            "true_edge_shift_px": true_edge_shift_by_k1[key],
                            "residual_grid_shift_px": residual_shift,
                            "evaluation_point_count": int(evaluation_indices.size),
                            "alignment": alignment,
                            "metric": metric,
                            "metric_failure": failure,
                            "classification": classification,
                        }
                    )
                    if (
                        surface_number == 0
                        and repeat_id == "repeat_same"
                        and true_k1 == 0.05
                        and error_k1 in {-0.02, 0.0, 0.02}
                    ):
                        save_rgb(
                            arguments.output
                            / "diagnostics"
                            / (
                                f"true_{coefficient_id(true_k1)}_"
                                f"error_{coefficient_id(error_k1)}.png"
                            ),
                            corrected_rgb,
                        )
            print(f"completed={surface_id}:{repeat_id}", flush=True)

    nominal_rows = [row for row in rows if row["sweep"] == "nominal"]
    raw_rows = [row for row in rows if row["sweep"] == "raw_boundary"]
    calibration_rows = [
        row for row in rows if row["sweep"] == "calibration_error"
    ]
    exact_rows = [
        row for row in calibration_rows if row["calibration_error_k1"] == 0.0
    ]
    imperfect_rows = [
        row for row in calibration_rows if row["calibration_error_k1"] != 0.0
    ]
    nominal_counts = classification_counts(nominal_rows)
    raw_counts = classification_counts(raw_rows)
    exact_counts = classification_counts(exact_rows)
    imperfect_counts = classification_counts(imperfect_rows)

    expected_attempts = (
        len(experiment["surface_ids"])
        * len(experiment["repeat_camera_ids"])
        * (
            1
            + len(experiment["raw_boundary_k1"])
            + len(experiment["calibration_true_k1"])
            * len(experiment["calibration_error_k1"])
        )
    )
    if len(rows) != expected_attempts:
        raise RuntimeError(
            f"Ожидалось {expected_attempts} попытки, получено {len(rows)}"
        )

    validity = experiment["preregistered_validity"]
    validity_failures: list[str] = []
    maximum_round_trip = max(exact_round_trip_by_k1.values())
    if maximum_round_trip > float(
        validity["maximum_exact_point_round_trip_error_px"]
    ):
        validity_failures.append("Превышена ошибка точного цикла координат")
    if nominal_counts["accepted_correct"] < int(
        validity["minimum_nominal_accepted_correct"]
    ):
        validity_failures.append("Недостаточно принятых номинальных контролей")
    if nominal_counts["false_accept"] > int(
        validity["maximum_nominal_false_accepts"]
    ):
        validity_failures.append("Есть ложные принятия в номинальном контроле")
    if exact_counts["accepted_correct"] < int(
        validity["minimum_exact_calibration_accepted_correct"]
    ):
        validity_failures.append("Точная коррекция восстановила слишком мало случаев")
    if exact_counts["false_accept"] > int(
        validity["maximum_exact_calibration_false_accepts"]
    ):
        validity_failures.append("Точная коррекция имеет ложные принятия")

    raw_safety = experiment["preregistered_raw_boundary_safety"]
    imperfect_safety = experiment["preregistered_imperfect_calibration_safety"]
    raw_safety_passed = raw_counts["false_accept"] <= int(
        raw_safety["maximum_false_accepts"]
    )
    imperfect_safety_passed = imperfect_counts["false_accept"] <= int(
        imperfect_safety["maximum_false_accepts"]
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
        "expected_attempt_count": expected_attempts,
        "maximum_exact_point_round_trip_error_px": maximum_round_trip,
        "exact_point_round_trip_error_by_k1_px": exact_round_trip_by_k1,
        "true_edge_shift_by_k1_px": true_edge_shift_by_k1,
        "nominal_classification_counts": nominal_counts,
        "raw_boundary_classification_counts": raw_counts,
        "exact_calibration_classification_counts": exact_counts,
        "imperfect_calibration_classification_counts": imperfect_counts,
        "raw_boundary_counts_by_k1": grouped_classifications(raw_rows, "true_k1"),
        "calibration_counts_by_error_k1": grouped_classifications(
            calibration_rows, "calibration_error_k1"
        ),
        "calibration_counts_by_true_k1": grouped_classifications(
            calibration_rows, "true_k1"
        ),
        "first_raw_false_accept_edge_shift_px": first_unsafe_shift(raw_rows),
        "first_imperfect_false_accept_residual_shift_px": first_unsafe_shift(
            imperfect_rows
        ),
        "preregistered_validity": validity,
        "validity_failures": validity_failures,
        "experiment_valid": not validity_failures,
        "preregistered_raw_boundary_safety": raw_safety,
        "raw_boundary_safety_passed": raw_safety_passed,
        "preregistered_imperfect_calibration_safety": imperfect_safety,
        "imperfect_calibration_safety_passed": imperfect_safety_passed,
        "calibrations": calibration_reports,
        "rows": rows,
        "primary_sources": experiment["primary_sources"],
        "limitations": [
            "Изменялись только истинный и оценённый k1.",
            "Teach оставался идеальным pinhole-кадром.",
            "Повторно использованы три сцены G12 из одной синтетической семьи.",
            "Нет rolling shutter, движения объектов, CVBS-артефактов и OSD.",
            "Синтетические ошибки ещё не характеризуют реальную камеру.",
        ],
    }
    report_path = arguments.output / "g14a2_calibration_uncertainty_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    build_summary_figure(
        rows,
        limit_m=limit_m,
        output_path=arguments.output / "g14a2_calibration_uncertainty_summary.png",
    )
    print(
        json.dumps(
            {
                "nominal": nominal_counts,
                "raw_boundary": raw_counts,
                "exact_calibration": exact_counts,
                "imperfect_calibration": imperfect_counts,
            },
            ensure_ascii=False,
        )
    )
    print(f"experiment_valid={not validity_failures}")
    print(f"raw_boundary_safety_passed={raw_safety_passed}")
    print(f"imperfect_calibration_safety_passed={imperfect_safety_passed}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
