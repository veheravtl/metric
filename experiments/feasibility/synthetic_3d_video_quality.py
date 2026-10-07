#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""G14-C: однофакторный sweep качества видеотракта."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from aerial_mapper.metric_recovery import MetricRecoveryThresholds
from aerial_mapper.paired_stability import stability_seeds_for_pair
from aerial_mapper.synthetic_robustness import (
    ImageDegradation,
    apply_image_degradation,
)
from aerial_mapper.teach_annotation import offset_mask_boundary
from aerial_mapper.video_artifacts import apply_osd_occlusion, jpeg_round_trip
from experiments.feasibility.synthetic_3d_calibration_uncertainty_repair import (
    PhysicalCase,
    control_summary,
    evaluate_physical_case,
)
from experiments.feasibility.synthetic_3d_lens_distortion import (
    build_map_calibration,
    camera_index,
    load_mask,
    load_rgb,
    load_truth,
    save_rgb,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "experiments/configs/synthetic_3d_g14c_video_quality.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/synthetic_3d/g14c_video_quality"


def parse_arguments() -> argparse.Namespace:
    """Читает пути; научные параметры остаются во frozen JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def project_path(value: str) -> Path:
    """Разрешает путь относительно корня репозитория."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def iter_cases(experiment: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Разворачивает номинал и пять независимых однофакторных осей."""

    yield {"case_id": "nominal", "factor": "nominal", "value": 0.0}
    for factor, values in experiment["factors"].items():
        for index, value in enumerate(values):
            yield {
                "case_id": f"{factor}_{index:02d}",
                "factor": factor,
                "value": value,
            }


def apply_case(
    image_rgb: np.ndarray,
    case: dict[str, Any],
    *,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Применяет ровно один фактор и возвращает проверяемые параметры."""

    factor = str(case["factor"])
    value = case["value"]
    metadata: dict[str, Any] = {}
    if factor == "nominal":
        result = image_rgb.copy()
    elif factor == "resolution_scale":
        scale = float(value)
        result = apply_image_degradation(
            image_rgb,
            ImageDegradation(resolution_scale=scale),
            random_seed=random_seed,
        )
        metadata["intermediate_width_px"] = int(round(image_rgb.shape[1] * scale))
        metadata["intermediate_height_px"] = int(round(image_rgb.shape[0] * scale))
    elif factor == "noise_standard_deviation":
        result = apply_image_degradation(
            image_rgb,
            ImageDegradation(noise_standard_deviation=float(value)),
            random_seed=random_seed,
        )
    elif factor == "blur_sigma_px":
        result = apply_image_degradation(
            image_rgb,
            ImageDegradation(blur_sigma_px=float(value)),
            random_seed=random_seed,
        )
    elif factor == "jpeg_quality":
        jpeg = jpeg_round_trip(image_rgb, quality=int(value))
        result = jpeg.image_rgb
        metadata["jpeg_encoded_bytes"] = jpeg.encoded_bytes
    elif factor == "osd_occlusion_fraction":
        osd = apply_osd_occlusion(
            image_rgb,
            fraction=float(value),
            random_seed=random_seed,
        )
        result = osd.image_rgb
        metadata["actual_osd_occlusion_fraction"] = osd.actual_fraction
    else:
        raise ValueError(f"Неизвестный фактор G14-C: {factor}")
    metadata["maximum_channel_difference"] = int(
        np.max(np.abs(result.astype(np.int16) - image_rgb.astype(np.int16)))
    )
    metadata["mean_absolute_channel_difference"] = float(
        np.mean(np.abs(result.astype(np.float32) - image_rgb.astype(np.float32)))
    )
    return result, metadata


def summarize_factor(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Считает безопасность и доступность физической оси без смены порогов."""

    any_false = [row for row in rows if row["evaluation"]["any_seed_false_accept"]]
    all_false = [row for row in rows if row["evaluation"]["all_seed_false_accept"]]
    decisions = [
        decision["gate_accepted"]
        for row in rows
        for decision in row["evaluation"]["seed_decisions"]
    ]
    false_decisions = sum(
        not row["evaluation"]["metric_within_limit"]
        and decision["gate_accepted"]
        for row in rows
        for decision in row["evaluation"]["seed_decisions"]
    )
    valid_cases = sum(row["evaluation"]["metric_within_limit"] for row in rows)
    return {
        "physical_case_count": len(rows),
        "metric_within_limit_cases": valid_cases,
        "metric_invalid_cases": len(rows) - valid_cases,
        "alignment_constructed_cases": sum(
            row["evaluation"]["alignment_constructed"] for row in rows
        ),
        "gate_decision_count": len(decisions),
        "gate_accept_count": sum(decisions),
        "gate_accept_fraction": sum(decisions) / len(decisions),
        "any_seed_false_accept_cases": len(any_false),
        "all_seed_false_accept_cases": len(all_false),
        "false_accept_gate_decisions": false_decisions,
        "first_any_seed_false_accept_case": (
            any_false[0]["case_id"] if any_false else None
        ),
        "first_any_seed_false_accept_value": (
            any_false[0]["value"] if any_false else None
        ),
        "maximum_metric_error_m": max(
            (
                float(row["evaluation"]["metric"]["maximum_m"])
                for row in rows
                if row["evaluation"]["metric"] is not None
            ),
            default=None,
        ),
    }


def build_summary_figure(
    rows: list[dict[str, Any]], *, limit_m: float, output_path: Path
) -> None:
    """Строит отдельную кривую ошибки и принятия для каждого фактора."""

    factors = [
        "resolution_scale",
        "noise_standard_deviation",
        "jpeg_quality",
        "blur_sigma_px",
        "osd_occlusion_fraction",
    ]
    titles = ["разрешение", "шум", "JPEG", "размытие", "OSD"]
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)
    scatter = None
    for axis, factor, title in zip(axes.flat, factors, titles, strict=False):
        selected = [
            row
            for row in rows
            if row["factor"] == factor and row["evaluation"]["metric"] is not None
        ]
        scatter = axis.scatter(
            [float(row["value"]) for row in selected],
            [row["evaluation"]["metric"]["maximum_m"] for row in selected],
            c=[row["evaluation"]["gate_accept_fraction"] for row in selected],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            alpha=0.8,
        )
        axis.axhline(limit_m, color="red", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("уровень фактора")
        axis.set_yscale("symlog", linthresh=0.01)
        axis.grid(alpha=0.25)
    axes.flat[0].set_ylabel("максимальная ошибка точки, м")
    axes.flat[3].set_ylabel("максимальная ошибка точки, м")
    axes.flat[-1].axis("off")
    if scatter is not None:
        figure.colorbar(scatter, ax=axes, label="доля seed, принявших случай")
    figure.suptitle("G14-C: независимые ухудшения видеотракта")
    figure.subplots_adjust(top=0.91, bottom=0.08, hspace=0.32, wspace=0.22)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def main() -> None:
    """Выполняет frozen G14-C и сохраняет полный машинный отчёт."""

    arguments = parse_arguments()
    experiment = json.loads(arguments.config.read_text(encoding="utf-8"))
    protocol_path = project_path(experiment["source_protocol"])
    scene_output = project_path(experiment["source_scene_output"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    arguments.output.mkdir(parents=True, exist_ok=True)
    thresholds = MetricRecoveryThresholds.from_mapping(
        protocol["acceptance_thresholds"]
    )
    limit_m = float(experiment["maximum_point_position_error_m"])
    teach_id = next(
        camera["id"] for camera in protocol["cameras"] if camera["role"] == "reference"
    )
    cases = list(iter_cases(experiment))
    rows: list[dict[str, Any]] = []
    nominal_pixel_differences: list[int] = []
    calibration_reports: dict[str, Any] = {}

    for surface_number, surface_id in enumerate(experiment["surface_ids"]):
        scene_directory = scene_output / "scenes" / surface_id
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
            nominal_rgb = load_rgb(scene_directory / "repeat" / repeat_id / "rgb.png")
            common_indices = np.flatnonzero(
                truth["visible"][teach_index]
                & truth["visible"][repeat_index]
                & (groups == 2)
            )
            nominal_points = truth["pixel_xy"][repeat_index, common_indices]
            stability_seeds = stability_seeds_for_pair(
                experiment_seed=int(experiment["seed"]),
                surface_number=surface_number,
                repeat_number=repeat_number,
                offsets=[int(value) for value in experiment["stability_seed_offsets"]],
            )
            artifact_seed = (
                int(experiment["seed"])
                + surface_number * 100_000
                + repeat_number * 10_000
            )

            for case in cases:
                repeat_rgb, artifact_metadata = apply_case(
                    nominal_rgb,
                    case,
                    random_seed=artifact_seed,
                )
                if case["factor"] == "nominal":
                    nominal_pixel_differences.append(
                        artifact_metadata["maximum_channel_difference"]
                    )
                physical_case = PhysicalCase(
                    sweep=str(case["factor"]),
                    case_id=str(case["case_id"]),
                    true_k1=0.0,
                    estimated_k1=None,
                    calibration_error_k1=None,
                    true_edge_shift_px=0.0,
                    residual_grid_shift_px=0.0,
                    repeat_rgb=repeat_rgb,
                    query_points_px=nominal_points,
                    truth_indices=common_indices,
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
                        "surface_id": surface_id,
                        "repeat_id": repeat_id,
                        "case_id": case["case_id"],
                        "factor": case["factor"],
                        "value": case["value"],
                        "artifact": artifact_metadata,
                        "evaluation_point_count": int(common_indices.size),
                        "evaluation": evaluation,
                    }
                )
                if (
                    surface_number == 0
                    and repeat_id == "repeat_same"
                    and case["factor"] != "nominal"
                    and case is cases[-1]
                ):
                    save_rgb(
                        arguments.output / "diagnostics" / f"{case['case_id']}.png",
                        repeat_rgb,
                    )
            print(f"completed={surface_id}:{repeat_id}", flush=True)

    nominal_rows = [row for row in rows if row["factor"] == "nominal"]
    nominal_control = control_summary(nominal_rows, limit_m=limit_m)
    factor_summaries = {
        factor: summarize_factor([row for row in rows if row["factor"] == factor])
        for factor in experiment["factors"]
    }
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

    validity = experiment["preregistered_validity"]
    validity_failures: list[str] = []
    if len(cases) != int(validity["expected_cases_per_scene_pose"]):
        validity_failures.append("Неверное число случаев на пару")
    if len(rows) != int(validity["expected_physical_case_count"]):
        validity_failures.append("Неверное число физических случаев")
    if decision_count != int(validity["expected_gate_decision_count"]):
        validity_failures.append("Неверное число решений gate")
    if max(nominal_pixel_differences) > int(
        validity["maximum_identity_pixel_difference"]
    ):
        validity_failures.append("Номинальная обработка изменила RGB")
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

    safety = experiment["preregistered_safety"]
    safety_failures: list[str] = []
    for factor, summary in factor_summaries.items():
        if summary["any_seed_false_accept_cases"] > int(
            safety["maximum_any_seed_false_accept_cases_per_factor"]
        ):
            safety_failures.append(f"{factor}: any-seed ложные принятия")
        if summary["all_seed_false_accept_cases"] > int(
            safety["maximum_all_seed_false_accept_cases_per_factor"]
        ):
            safety_failures.append(f"{factor}: all-seed ложные принятия")

    report = {
        "schema_version": 1,
        "experiment": experiment["experiment"],
        "configuration": str(arguments.config),
        "configuration_sha256": sha256_file(arguments.config),
        "source_protocol": str(protocol_path),
        "source_protocol_sha256": sha256_file(protocol_path),
        "source_scene_output": str(scene_output),
        "rerendered_blender_scenes": False,
        "physical_case_count": len(rows),
        "gate_decision_count": decision_count,
        "maximum_identity_pixel_difference": max(nominal_pixel_differences),
        "nominal_control": nominal_control,
        "factor_summaries": factor_summaries,
        "gate_decision_outcomes": dict(sorted(decision_outcomes.items())),
        "preregistered_validity": validity,
        "validity_failures": validity_failures,
        "experiment_valid": not validity_failures,
        "preregistered_safety": safety,
        "safety_failures": safety_failures,
        "safety_passed": not safety_failures,
        "calibrations": calibration_reports,
        "rows": rows,
        "primary_sources": experiment["primary_sources"],
        "limitations": [
            "Факторы проверены отдельно, а не в комбинациях.",
            "JPEG и OSD цифровые; полный аналоговый CVBS не моделируется.",
            "OSD является псевдоглифами, а не точной раскладкой Betaflight.",
            "Нет rolling shutter, дисторсии, рельефа и движения объектов.",
        ],
    }
    report_path = arguments.output / "g14c_video_quality_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    build_summary_figure(
        rows,
        limit_m=limit_m,
        output_path=arguments.output / "g14c_video_quality_summary.png",
    )
    print(
        json.dumps(
            {
                "nominal_control": nominal_control,
                "factor_summaries": factor_summaries,
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
