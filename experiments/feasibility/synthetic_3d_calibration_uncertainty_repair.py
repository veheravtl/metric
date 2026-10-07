#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""G14-A2-R: парная многосидовая проверка stability для матрицы G14-A2."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
from aerial_mapper.paired_stability import (
    gate_safety_flags,
    stability_seeds_for_pair,
)
from aerial_mapper.quality import analyze_alignment_quality
from aerial_mapper.radiance_camera import PinholeIntrinsics
from aerial_mapper.teach_annotation import offset_mask_boundary
from experiments.feasibility.synthetic_3d_calibration_uncertainty import (
    coefficient_id,
    maximum_residual_shift,
)
from experiments.feasibility.synthetic_3d_lens_distortion import (
    build_map_calibration,
    camera_index,
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
    PROJECT_ROOT / "experiments/configs/synthetic_3d_g14a2r_paired_stability.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/synthetic_3d/g14a2r_paired_stability"


@dataclass(frozen=True)
class PhysicalCase:
    """Один RGB-вариант и согласованные с ним контрольные пиксели."""

    sweep: str
    case_id: str
    true_k1: float
    estimated_k1: float | None
    calibration_error_k1: float | None
    true_edge_shift_px: float
    residual_grid_shift_px: float
    repeat_rgb: np.ndarray
    query_points_px: np.ndarray
    truth_indices: np.ndarray


def parse_arguments() -> argparse.Namespace:
    """Читает пути запуска; параметры repair остаются в JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def generate_physical_cases(
    *,
    source_experiment: dict[str, Any],
    nominal_rgb: np.ndarray,
    nominal_points: np.ndarray,
    common_indices: np.ndarray,
    intrinsics: PinholeIntrinsics,
) -> Iterator[PhysicalCase]:
    """Воспроизводит ровно 47 вариантов G14-A2 для одной сцены и позы."""

    nominal_visible = in_frame_mask(nominal_points, intrinsics)
    yield PhysicalCase(
        sweep="nominal",
        case_id="nominal_k1_0",
        true_k1=0.0,
        estimated_k1=0.0,
        calibration_error_k1=0.0,
        true_edge_shift_px=0.0,
        residual_grid_shift_px=0.0,
        repeat_rgb=nominal_rgb,
        query_points_px=nominal_points[nominal_visible],
        truth_indices=common_indices[nominal_visible],
    )

    for true_k1_value in source_experiment["raw_boundary_k1"]:
        true_k1 = float(true_k1_value)
        true_distortion = BrownConradyDistortion(k1=true_k1)
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
        edge_shift = maximum_edge_shift(intrinsics, true_distortion)
        yield PhysicalCase(
            sweep="raw_boundary",
            case_id=f"raw_true_{coefficient_id(true_k1)}",
            true_k1=true_k1,
            estimated_k1=None,
            calibration_error_k1=None,
            true_edge_shift_px=edge_shift,
            residual_grid_shift_px=edge_shift,
            repeat_rgb=distorted_rgb,
            query_points_px=distorted_points[visible],
            truth_indices=common_indices[visible],
        )

    for true_k1_value in source_experiment["calibration_true_k1"]:
        true_k1 = float(true_k1_value)
        true_distortion = BrownConradyDistortion(k1=true_k1)
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
        edge_shift = maximum_edge_shift(intrinsics, true_distortion)

        for error_k1_value in source_experiment["calibration_error_k1"]:
            error_k1 = float(error_k1_value)
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
            yield PhysicalCase(
                sweep="calibration_error",
                case_id=(
                    f"true_{coefficient_id(true_k1)}_"
                    f"error_{coefficient_id(error_k1)}"
                ),
                true_k1=true_k1,
                estimated_k1=estimated_k1,
                calibration_error_k1=error_k1,
                true_edge_shift_px=edge_shift,
                residual_grid_shift_px=maximum_residual_shift(
                    intrinsics,
                    true_distortion=true_distortion,
                    estimated_distortion=estimated_distortion,
                ),
                repeat_rgb=corrected_rgb,
                query_points_px=corrected_points[visible],
                truth_indices=common_indices[visible],
            )


def evaluate_physical_case(
    *,
    case: PhysicalCase,
    teach_rgb: np.ndarray,
    safe_mask: np.ndarray,
    thresholds: MetricRecoveryThresholds,
    stability_seeds: list[int],
    stability_trials: int,
    stability_subsample_fraction: float,
    truth: dict[str, np.ndarray],
    calibration: Any,
    protocol: dict[str, Any],
    limit_m: float,
) -> dict[str, Any]:
    """Оценивает физическую геометрию один раз и gate несколькими seed."""

    try:
        alignment = align_frame_to_reference(
            teach_rgb,
            case.repeat_rgb,
            reference_feature_mask=safe_mask,
        )
    except AlignmentFailure as error:
        return {
            "alignment_constructed": False,
            "alignment_failure": str(error),
            "ratio_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "coverage_fraction": 0.0,
            "metric": None,
            "metric_failure": None,
            "metric_within_limit": False,
            "seed_decisions": [],
            "gate_accept_count": 0,
            "gate_accept_fraction": 0.0,
            "any_seed_false_accept": False,
            "all_seed_false_accept": False,
        }

    metric_report = None
    metric_failure = None
    try:
        metric_report = evaluate_metric_points(
            homography=alignment.homography_frame_to_reference,
            query_points_px=case.query_points_px,
            truth_indices=case.truth_indices,
            truth=truth,
            calibration=calibration,
            protocol=protocol,
        )
    except MeasurementFailure as error:
        metric_failure = str(error)
    metric_within_limit = bool(
        metric_failure is None
        and metric_report is not None
        and metric_report["maximum_m"] <= limit_m
    )

    seed_decisions: list[dict[str, Any]] = []
    for seed in stability_seeds:
        quality = analyze_alignment_quality(
            alignment,
            frame_width_pixels=case.repeat_rgb.shape[1],
            frame_height_pixels=case.repeat_rgb.shape[0],
            stability_trials=stability_trials,
            stability_subsample_fraction=stability_subsample_fraction,
            random_seed=seed,
        )
        failures = alignment_gate_failures(
            alignment,
            quality,
            thresholds=thresholds,
        )
        seed_decisions.append(
            {
                "seed": seed,
                "gate_accepted": not failures,
                "gate_failures": list(failures),
                "quality": asdict(quality),
            }
        )
    accepted_by_seed = [decision["gate_accepted"] for decision in seed_decisions]
    any_false_accept, all_false_accept = gate_safety_flags(
        metric_within_limit=metric_within_limit,
        accepted_by_seed=accepted_by_seed,
    )
    accept_count = sum(accepted_by_seed)
    return {
        "alignment_constructed": True,
        "alignment_failure": None,
        "ratio_matches": alignment.ratio_match_count,
        "inliers": alignment.inlier_count,
        "inlier_ratio": alignment.inlier_ratio,
        "coverage_fraction": alignment.inlier_spatial_coverage_fraction,
        "metric": metric_report,
        "metric_failure": metric_failure,
        "metric_within_limit": metric_within_limit,
        "seed_decisions": seed_decisions,
        "gate_accept_count": accept_count,
        "gate_accept_fraction": accept_count / len(seed_decisions),
        "any_seed_false_accept": any_false_accept,
        "all_seed_false_accept": all_false_accept,
    }


def control_summary(rows: list[dict[str, Any]], *, limit_m: float) -> dict[str, Any]:
    """Разделяет физическую корректность контроля и доступность gate."""

    constructed = [row for row in rows if row["evaluation"]["alignment_constructed"]]
    metric_failures = [
        row
        for row in constructed
        if row["evaluation"]["metric"] is None
        or row["evaluation"]["metric_failure"] is not None
    ]
    over_limit = [
        row
        for row in constructed
        if row["evaluation"]["metric"] is not None
        and row["evaluation"]["metric"]["maximum_m"] > limit_m
    ]
    gate_decisions = [
        decision["gate_accepted"]
        for row in rows
        for decision in row["evaluation"]["seed_decisions"]
    ]
    return {
        "physical_case_count": len(rows),
        "alignment_constructed": len(constructed),
        "metric_failures": len(metric_failures),
        "over_limit": len(over_limit),
        "maximum_metric_error_m": max(
            (
                float(row["evaluation"]["metric"]["maximum_m"])
                for row in constructed
                if row["evaluation"]["metric"] is not None
            ),
            default=None,
        ),
        "gate_decision_count": len(gate_decisions),
        "gate_accept_count": sum(gate_decisions),
        "gate_accept_fraction": (
            sum(gate_decisions) / len(gate_decisions) if gate_decisions else 0.0
        ),
    }


def safety_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Считает опасные физические случаи и отдельные решения gate."""

    any_seed_cases = [
        row for row in rows if row["evaluation"]["any_seed_false_accept"]
    ]
    all_seed_cases = [
        row for row in rows if row["evaluation"]["all_seed_false_accept"]
    ]
    false_decisions = sum(
        not row["evaluation"]["metric_within_limit"]
        and decision["gate_accepted"]
        for row in rows
        for decision in row["evaluation"]["seed_decisions"]
    )
    return {
        "physical_case_count": len(rows),
        "any_seed_false_accept_cases": len(any_seed_cases),
        "all_seed_false_accept_cases": len(all_seed_cases),
        "false_accept_gate_decisions": false_decisions,
        "first_any_seed_false_accept_residual_shift_px": min(
            (
                float(row["residual_grid_shift_px"])
                for row in any_seed_cases
            ),
            default=None,
        ),
        "first_all_seed_false_accept_residual_shift_px": min(
            (
                float(row["residual_grid_shift_px"])
                for row in all_seed_cases
            ),
            default=None,
        ),
    }


def decisions_by_error(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Группирует физические опасные случаи по ошибке коэффициента."""

    result: dict[str, dict[str, Any]] = {}
    for error in sorted({row["calibration_error_k1"] for row in rows}):
        selected = [row for row in rows if row["calibration_error_k1"] == error]
        result[str(error)] = safety_summary(selected)
    return result


def build_summary_figure(
    rows: list[dict[str, Any]], *, limit_m: float, output_path: Path
) -> None:
    """Показывает метрическую ошибку и долю seed, принявших каждый случай."""

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    panels = (
        ("raw_boundary", "true_k1", "raw: истинный k1"),
        (
            "calibration_error",
            "residual_grid_shift_px",
            "коррекция: остаточный сдвиг, px",
        ),
    )
    scatter = None
    for axis, (sweep, x_field, x_label) in zip(axes, panels, strict=True):
        selected = [
            row
            for row in rows
            if row["sweep"] == sweep and row["evaluation"]["metric"] is not None
        ]
        scatter = axis.scatter(
            [row[x_field] for row in selected],
            [row["evaluation"]["metric"]["maximum_m"] for row in selected],
            c=[row["evaluation"]["gate_accept_fraction"] for row in selected],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            alpha=0.78,
        )
        axis.axhline(limit_m, color="red", linestyle="--", linewidth=1)
        axis.set_xlabel(x_label)
        axis.grid(alpha=0.25)
        axis.set_yscale("symlog", linthresh=0.01)
    axes[0].set_title("Слабая неисправленная дисторсия")
    axes[1].set_title("Неточная коррекция")
    axes[0].set_ylabel("максимальная ошибка точки, м")
    if scatter is not None:
        figure.colorbar(scatter, ax=axes, label="доля seed, принявших случай")
    figure.suptitle("G14-A2-R: парная проверка stability")
    figure.subplots_adjust(top=0.86, bottom=0.12, wspace=0.18)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def main() -> None:
    """Выполняет frozen repair-протокол и сохраняет полный машинный отчёт."""

    arguments = parse_arguments()
    repair = json.loads(arguments.config.read_text(encoding="utf-8"))
    source_experiment_path = project_path(repair["source_experiment_config"])
    source_experiment = json.loads(
        source_experiment_path.read_text(encoding="utf-8")
    )
    source_protocol_path = project_path(source_experiment["source_protocol"])
    source_output = project_path(source_experiment["source_output"])
    protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    arguments.output.mkdir(parents=True, exist_ok=True)
    thresholds = MetricRecoveryThresholds.from_mapping(
        protocol["acceptance_thresholds"]
    )
    limit_m = float(source_experiment["maximum_point_position_error_m"])
    teach_id = next(
        camera["id"]
        for camera in protocol["cameras"]
        if camera["role"] == "reference"
    )
    rows: list[dict[str, Any]] = []
    exact_round_trip_by_k1: dict[str, float] = {}
    calibration_reports: dict[str, Any] = {}

    for surface_number, surface_id in enumerate(source_experiment["surface_ids"]):
        scene_directory = source_output / "scenes" / surface_id
        truth = load_truth(scene_directory)
        calibration, groups, anchor_indices = build_map_calibration(
            protocol=protocol,
            experiment=source_experiment,
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
            offset_pixels=-int(source_experiment["safe_mask_erosion_px"]),
        )

        for repeat_number, repeat_id in enumerate(
            source_experiment["repeat_camera_ids"]
        ):
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
            stability_seeds = stability_seeds_for_pair(
                experiment_seed=int(repair["seed"]),
                surface_number=surface_number,
                repeat_number=repeat_number,
                offsets=[int(value) for value in repair["stability_seed_offsets"]],
            )

            for true_k1 in set(
                [float(value) for value in source_experiment["raw_boundary_k1"]]
                + [
                    float(value)
                    for value in source_experiment["calibration_true_k1"]
                ]
            ):
                exact_round_trip_by_k1.setdefault(
                    str(true_k1),
                    point_round_trip_error(
                        intrinsics, BrownConradyDistortion(k1=true_k1)
                    ),
                )

            for case in generate_physical_cases(
                source_experiment=source_experiment,
                nominal_rgb=nominal_rgb,
                nominal_points=nominal_points,
                common_indices=common_indices,
                intrinsics=intrinsics,
            ):
                evaluation = evaluate_physical_case(
                    case=case,
                    teach_rgb=teach_rgb,
                    safe_mask=safe_mask,
                    thresholds=thresholds,
                    stability_seeds=stability_seeds,
                    stability_trials=int(repair["stability_trials_per_seed"]),
                    stability_subsample_fraction=float(
                        repair["stability_subsample_fraction"]
                    ),
                    truth=truth,
                    calibration=calibration,
                    protocol=protocol,
                    limit_m=limit_m,
                )
                rows.append(
                    {
                        "surface_id": surface_id,
                        "repeat_id": repeat_id,
                        "sweep": case.sweep,
                        "case_id": case.case_id,
                        "true_k1": case.true_k1,
                        "estimated_k1": case.estimated_k1,
                        "calibration_error_k1": case.calibration_error_k1,
                        "true_edge_shift_px": case.true_edge_shift_px,
                        "residual_grid_shift_px": case.residual_grid_shift_px,
                        "evaluation_point_count": int(case.truth_indices.size),
                        "evaluation": evaluation,
                    }
                )
                if (
                    surface_number == 0
                    and repeat_id == "repeat_same"
                    and case.sweep == "calibration_error"
                    and case.true_k1 == 0.05
                    and case.calibration_error_k1 in {-0.02, 0.0, 0.02}
                ):
                    save_rgb(
                        arguments.output / "diagnostics" / f"{case.case_id}.png",
                        case.repeat_rgb,
                    )
            print(f"completed={surface_id}:{repeat_id}", flush=True)

    expected_physical = (
        len(source_experiment["surface_ids"])
        * len(source_experiment["repeat_camera_ids"])
        * (
            1
            + len(source_experiment["raw_boundary_k1"])
            + len(source_experiment["calibration_true_k1"])
            * len(source_experiment["calibration_error_k1"])
        )
    )
    if len(rows) != expected_physical:
        raise RuntimeError(
            f"Ожидалось {expected_physical} физических случаев, получено {len(rows)}"
        )

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
    nominal_control = control_summary(nominal_rows, limit_m=limit_m)
    exact_control = control_summary(exact_rows, limit_m=limit_m)
    raw_safety = safety_summary(raw_rows)
    imperfect_safety = safety_summary(imperfect_rows)

    validity = repair["preregistered_validity"]
    validity_failures: list[str] = []
    maximum_round_trip = max(exact_round_trip_by_k1.values())
    if maximum_round_trip > float(
        validity["maximum_exact_point_round_trip_error_px"]
    ):
        validity_failures.append("Превышена ошибка точного цикла координат")
    if nominal_control["alignment_constructed"] < int(
        validity["minimum_nominal_alignment_constructed"]
    ):
        validity_failures.append("Не построен хотя бы один номинальный alignment")
    if nominal_control["metric_failures"] > int(
        validity["maximum_nominal_metric_failures"]
    ):
        validity_failures.append("Есть метрический сбой номинального контроля")
    if nominal_control["over_limit"] > int(
        validity["maximum_nominal_over_limit"]
    ):
        validity_failures.append("Номинальный контроль превысил 0,2 м")
    if exact_control["alignment_constructed"] < int(
        validity["minimum_exact_calibration_alignment_constructed"]
    ):
        validity_failures.append("Не построена часть точных коррекций")
    if exact_control["metric_failures"] > int(
        validity["maximum_exact_calibration_metric_failures"]
    ):
        validity_failures.append("Есть метрический сбой точной коррекции")
    if exact_control["over_limit"] > int(
        validity["maximum_exact_calibration_over_limit"]
    ):
        validity_failures.append("Точная коррекция превысила 0,2 м")

    safety_limits = repair["preregistered_safety"]
    safety_failures: list[str] = []
    checks = (
        (
            raw_safety["any_seed_false_accept_cases"],
            safety_limits["maximum_any_seed_raw_false_accept_cases"],
            "raw имеет any-seed ложные принятия",
        ),
        (
            raw_safety["all_seed_false_accept_cases"],
            safety_limits["maximum_all_seed_raw_false_accept_cases"],
            "raw имеет all-seed ложные принятия",
        ),
        (
            imperfect_safety["any_seed_false_accept_cases"],
            safety_limits["maximum_any_seed_imperfect_false_accept_cases"],
            "Неточная коррекция имеет any-seed ложные принятия",
        ),
        (
            imperfect_safety["all_seed_false_accept_cases"],
            safety_limits["maximum_all_seed_imperfect_false_accept_cases"],
            "Неточная коррекция имеет all-seed ложные принятия",
        ),
    )
    for observed, maximum, message in checks:
        if int(observed) > int(maximum):
            safety_failures.append(message)

    decision_count = sum(
        len(row["evaluation"]["seed_decisions"]) for row in rows
    )
    decision_outcomes = Counter()
    for row in rows:
        within = row["evaluation"]["metric_within_limit"]
        for decision in row["evaluation"]["seed_decisions"]:
            if decision["gate_accepted"]:
                decision_outcomes[
                    "accepted_correct" if within else "false_accept"
                ] += 1
            else:
                decision_outcomes[
                    "rejected_valid" if within else "rejected_invalid"
                ] += 1

    report = {
        "schema_version": 1,
        "experiment": repair["experiment"],
        "configuration": str(arguments.config),
        "configuration_sha256": sha256_file(arguments.config),
        "source_experiment_config": str(source_experiment_path),
        "source_experiment_config_sha256": sha256_file(source_experiment_path),
        "source_protocol": str(source_protocol_path),
        "source_protocol_sha256": sha256_file(source_protocol_path),
        "source_output": str(source_output),
        "rerendered_blender_scenes": False,
        "physical_case_count": len(rows),
        "gate_decision_count": decision_count,
        "expected_physical_case_count": expected_physical,
        "expected_gate_decision_count": expected_physical
        * len(repair["stability_seed_offsets"]),
        "maximum_exact_point_round_trip_error_px": maximum_round_trip,
        "exact_point_round_trip_error_by_k1_px": exact_round_trip_by_k1,
        "nominal_control": nominal_control,
        "exact_calibration_control": exact_control,
        "raw_boundary_safety": raw_safety,
        "imperfect_calibration_safety": imperfect_safety,
        "imperfect_safety_by_error_k1": decisions_by_error(imperfect_rows),
        "gate_decision_outcomes": dict(sorted(decision_outcomes.items())),
        "preregistered_validity": validity,
        "validity_failures": validity_failures,
        "experiment_valid": not validity_failures,
        "preregistered_safety": safety_limits,
        "safety_failures": safety_failures,
        "safety_passed": not safety_failures,
        "calibrations": calibration_reports,
        "rows": rows,
        "primary_sources": repair["primary_sources"],
        "limitations": [
            "Repair использует те же RGB и физические уровни, что G14-A2.",
            "Независимо меняются только seed случайной stability-диагностики.",
            "Моделируется только ошибка k1.",
            "Нет rolling shutter, движения объектов, CVBS-артефактов и OSD.",
        ],
    }
    report_path = arguments.output / "g14a2r_paired_stability_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    build_summary_figure(
        rows,
        limit_m=limit_m,
        output_path=arguments.output / "g14a2r_paired_stability_summary.png",
    )
    print(
        json.dumps(
            {
                "nominal_control": nominal_control,
                "exact_calibration_control": exact_control,
                "raw_boundary_safety": raw_safety,
                "imperfect_calibration_safety": imperfect_safety,
                "gate_decision_outcomes": dict(decision_outcomes),
            },
            ensure_ascii=False,
        )
    )
    print(f"experiment_valid={not validity_failures}")
    print(f"safety_passed={not safety_failures}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
