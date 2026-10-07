#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""G14-C2/G14-D: агрегированный шумовой и смешанный видеогейт."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from aerial_mapper.metric_recovery import MetricRecoveryThresholds
from aerial_mapper.paired_stability import (
    aggregate_gate_decisions,
    failed_alignment_seed_decisions,
    stability_seeds_for_pair,
)
from aerial_mapper.teach_annotation import offset_mask_boundary
from aerial_mapper.video_artifacts import (
    VideoArtifactProfile,
    apply_video_artifact_profile,
)
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
    PROJECT_ROOT / "experiments/configs/synthetic_3d_g14c2_noise_consensus.json"
)
OUTPUTS_BY_MODE = {
    "noise_boundary": PROJECT_ROOT / "outputs/synthetic_3d/g14c2_noise_consensus",
    "mixed_profiles": PROJECT_ROOT / "outputs/synthetic_3d/g14d_video_mixtures",
}
REPORT_NAMES = {
    "noise_boundary": "g14c2_noise_consensus_report.json",
    "mixed_profiles": "g14d_video_mixtures_report.json",
}
FIGURE_NAMES = {
    "noise_boundary": "g14c2_noise_consensus_summary.png",
    "mixed_profiles": "g14d_video_mixtures_summary.png",
}


@dataclass(frozen=True)
class FollowupCase:
    """Один физический кадр и параметры его воспроизводимого искажения."""

    case_id: str
    profile_id: str
    level_value: float | str
    replicate_index: int | None
    artifact_seed_offset: int
    profile: VideoArtifactProfile


def parse_arguments() -> argparse.Namespace:
    """Читает пути; все научные параметры остаются во frozen JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def project_path(value: str) -> Path:
    """Разрешает путь относительно корня репозитория."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def profile_from_mapping(values: Mapping[str, object]) -> VideoArtifactProfile:
    """Преобразует JSON-профиль без неявного включения JPEG для номинала."""

    jpeg_value = values.get("jpeg_quality")
    return VideoArtifactProfile(
        blur_sigma_px=float(values.get("blur_sigma_px", 0.0)),
        resolution_scale=float(values.get("resolution_scale", 1.0)),
        noise_standard_deviation=float(values.get("noise_standard_deviation", 0.0)),
        osd_occlusion_fraction=float(values.get("osd_occlusion_fraction", 0.0)),
        jpeg_quality=None if jpeg_value is None else int(jpeg_value),
    )


def iter_cases(experiment: Mapping[str, Any]) -> Iterator[FollowupCase]:
    """Разворачивает номинал и frozen-матрицу C2 либо D."""

    yield FollowupCase(
        case_id="nominal",
        profile_id="nominal",
        level_value="nominal",
        replicate_index=None,
        artifact_seed_offset=0,
        profile=VideoArtifactProfile(),
    )
    offsets = [int(value) for value in experiment["artifact_replicate_offsets"]]
    mode = str(experiment["case_mode"])
    if mode == "noise_boundary":
        for level_index, level_value in enumerate(
            experiment["noise_standard_deviation_levels"]
        ):
            sigma = float(level_value)
            profile = VideoArtifactProfile(noise_standard_deviation=sigma)
            for replicate_index, offset in enumerate(offsets):
                yield FollowupCase(
                    case_id=f"noise_{level_index:02d}_rep_{replicate_index:02d}",
                    profile_id=f"noise_{level_index:02d}",
                    level_value=sigma,
                    replicate_index=replicate_index,
                    artifact_seed_offset=offset,
                    profile=profile,
                )
        return
    if mode == "mixed_profiles":
        for profile_values in experiment["profiles"]:
            profile_id = str(profile_values["id"])
            profile = profile_from_mapping(profile_values)
            for replicate_index, offset in enumerate(offsets):
                yield FollowupCase(
                    case_id=f"{profile_id}_rep_{replicate_index:02d}",
                    profile_id=profile_id,
                    level_value=profile_id,
                    replicate_index=replicate_index,
                    artifact_seed_offset=offset,
                    profile=profile,
                )
        return
    raise ValueError(f"Неизвестный case_mode: {mode}")


def attach_aggregate_decision(
    evaluation: dict[str, Any],
    *,
    minimum_accept_count: int,
) -> None:
    """Добавляет одно рабочее решение без обращения к метрической истине."""

    component_accepts = [
        bool(decision["gate_accepted"]) for decision in evaluation["seed_decisions"]
    ]
    aggregate = aggregate_gate_decisions(
        component_accepts,
        minimum_accept_count=minimum_accept_count,
    )
    evaluation["aggregate_gate"] = asdict(aggregate)
    evaluation["aggregate_false_accept"] = bool(
        aggregate.gate_accepted and not evaluation["metric_within_limit"]
    )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Разделяет компонентную диагностику, итоговый gate и физическую истину."""

    component_decisions = [
        decision for row in rows for decision in row["evaluation"]["seed_decisions"]
    ]
    aggregate_accepts = [
        bool(row["evaluation"]["aggregate_gate"]["gate_accepted"]) for row in rows
    ]
    invalid_rows = [row for row in rows if not row["evaluation"]["metric_within_limit"]]
    return {
        "physical_case_count": len(rows),
        "metric_within_limit_cases": len(rows) - len(invalid_rows),
        "metric_invalid_cases": len(invalid_rows),
        "alignment_constructed_cases": sum(
            bool(row["evaluation"]["alignment_constructed"]) for row in rows
        ),
        "component_decision_count": len(component_decisions),
        "component_accept_count": sum(
            bool(decision["gate_accepted"]) for decision in component_decisions
        ),
        "component_any_false_accept_cases": sum(
            any(
                bool(decision["gate_accepted"])
                for decision in row["evaluation"]["seed_decisions"]
            )
            for row in invalid_rows
        ),
        "aggregate_decision_count": len(rows),
        "aggregate_accept_count": sum(aggregate_accepts),
        "aggregate_accept_fraction": (
            sum(aggregate_accepts) / len(rows) if rows else 0.0
        ),
        "aggregate_false_accept_cases": sum(
            bool(row["evaluation"]["aggregate_false_accept"]) for row in rows
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


def grouped_summaries(
    rows: list[dict[str, Any]],
    *,
    ordered_profile_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Сохраняет порядок frozen-уровней, а не сортировку строковых идентификаторов."""

    return {
        profile_id: summarize_rows(
            [row for row in rows if row["profile_id"] == profile_id]
        )
        for profile_id in ordered_profile_ids
    }


def ordered_profile_ids(
    experiment: Mapping[str, Any],
    cases: list[FollowupCase],
) -> list[str]:
    """Возвращает человекочитаемый порядок групп для таблиц и графика."""

    if experiment["case_mode"] == "mixed_profiles":
        return [str(profile["id"]) for profile in experiment["profiles"]]
    result: list[str] = []
    for case in cases:
        if case.profile_id != "nominal" and case.profile_id not in result:
            result.append(case.profile_id)
    return result


def build_summary_figure(
    rows: list[dict[str, Any]],
    *,
    profile_ids: list[str],
    profile_labels: list[str],
    limit_m: float,
    title: str,
    output_path: Path,
) -> None:
    """Показывает физическую ошибку и доступность итогового gate."""

    figure, (error_axis, availability_axis) = plt.subplots(1, 2, figsize=(13, 5.5))
    for profile_index, profile_id in enumerate(profile_ids):
        selected = [row for row in rows if row["profile_id"] == profile_id]
        for row_index, row in enumerate(selected):
            metric = row["evaluation"]["metric"]
            if metric is None:
                continue
            accepted = row["evaluation"]["aggregate_gate"]["gate_accepted"]
            offset = ((row_index % 9) - 4) * 0.018
            error_axis.scatter(
                profile_index + offset,
                metric["maximum_m"],
                color="tab:green" if accepted else "tab:gray",
                marker="o" if accepted else "x",
                alpha=0.75,
            )
    error_axis.axhline(limit_m, color="red", linestyle="--", linewidth=1)
    error_axis.set_yscale("symlog", linthresh=0.01)
    error_axis.set_ylabel("максимальная ошибка точки, м")
    error_axis.set_xticks(range(len(profile_labels)), profile_labels, rotation=30)
    error_axis.grid(alpha=0.25)

    summaries = [
        summarize_rows([row for row in rows if row["profile_id"] == profile_id])
        for profile_id in profile_ids
    ]
    availability_axis.bar(
        range(len(profile_labels)),
        [summary["aggregate_accept_fraction"] for summary in summaries],
        color="tab:blue",
        alpha=0.8,
    )
    availability_axis.set_ylim(0.0, 1.0)
    availability_axis.set_ylabel("доля итоговых принятий 5/5")
    availability_axis.set_xticks(
        range(len(profile_labels)), profile_labels, rotation=30
    )
    availability_axis.grid(axis="y", alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def validity_failures(
    *,
    experiment: Mapping[str, Any],
    cases: list[FollowupCase],
    rows: list[dict[str, Any]],
    component_decision_count: int,
    nominal_pixel_differences: list[int],
    nominal_control: Mapping[str, Any],
    nominal_aggregate_accept_count: int,
) -> list[str]:
    """Проверяет только заранее зарегистрированные условия валидности."""

    validity = experiment["preregistered_validity"]
    failures: list[str] = []
    if len(cases) != int(validity["expected_cases_per_scene_pose"]):
        failures.append("Неверное число случаев на пару")
    if len(rows) != int(validity["expected_physical_case_count"]):
        failures.append("Неверное число физических случаев")
    if component_decision_count != int(validity["expected_component_decision_count"]):
        failures.append("Неверное число компонентных решений")
    if len(rows) != int(validity["expected_aggregate_decision_count"]):
        failures.append("Неверное число агрегированных решений")
    if max(nominal_pixel_differences) > int(
        validity["maximum_identity_pixel_difference"]
    ):
        failures.append("Номинальная обработка изменила RGB")
    if nominal_control["alignment_constructed"] < int(
        validity["minimum_nominal_alignment_constructed"]
    ):
        failures.append("Не построен номинальный alignment")
    if nominal_control["metric_failures"] > int(
        validity["maximum_nominal_metric_failures"]
    ):
        failures.append("Есть номинальный метрический сбой")
    if nominal_control["over_limit"] > int(validity["maximum_nominal_over_limit"]):
        failures.append("Номинал превысил метрический предел")
    if nominal_aggregate_accept_count < int(
        validity["minimum_nominal_aggregate_accept_count"]
    ):
        failures.append("Агрегированный gate отклонил номинальный контроль")
    return failures


def availability_failures(
    *,
    experiment: Mapping[str, Any],
    rows: list[dict[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    nominal_aggregate_reject_count: int,
) -> list[str]:
    """Не даёт безопасному, но всегда молчащему правилу считаться полезным."""

    registered = experiment["preregistered_availability"]
    failures: list[str] = []
    if nominal_aggregate_reject_count > int(
        registered["maximum_nominal_aggregate_reject_count"]
    ):
        failures.append("Слишком много отказов на номинале")

    if experiment["case_mode"] == "noise_boundary":
        degraded = [row for row in rows if row["profile_id"] != "nominal"]
        accepted = sum(
            bool(row["evaluation"]["aggregate_gate"]["gate_accepted"])
            for row in degraded
        )
        fraction = accepted / len(degraded)
        if fraction < float(registered["minimum_degraded_aggregate_accept_fraction"]):
            failures.append(
                f"Доступность шумных кадров {fraction:.3f} ниже frozen-порога"
            )
    else:
        for profile_id, minimum_fraction in registered[
            "minimum_profile_accept_fraction"
        ].items():
            actual = float(summaries[profile_id]["aggregate_accept_fraction"])
            if actual < float(minimum_fraction):
                failures.append(
                    f"{profile_id}: доступность {actual:.3f} ниже frozen-порога"
                )
    return failures


def upstream_context(experiment: Mapping[str, Any]) -> dict[str, Any] | None:
    """Для D фиксирует, имеет ли опыт подтверждающий или диагностический статус."""

    if experiment["case_mode"] != "mixed_profiles":
        return None
    report_path = project_path(str(experiment["source_g14c2_report"]))
    if not report_path.exists():
        return {
            "report": str(report_path),
            "present": False,
            "eligible_for_confirmation": False,
            "reason": "Отчёт C2 отсутствует",
        }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    eligible = bool(
        report["experiment_valid"]
        and report["safety_passed"]
        and report["availability_passed"]
    )
    return {
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "present": True,
        "c2_experiment_valid": report["experiment_valid"],
        "c2_safety_passed": report["safety_passed"],
        "c2_availability_passed": report["availability_passed"],
        "eligible_for_confirmation": eligible,
        "reason": (
            "C2 прошёл все три группы критериев"
            if eligible
            else "C2 не прошёл хотя бы одну группу критериев"
        ),
    }


def main() -> None:
    """Выполняет frozen C2 или D и сохраняет полный машинный отчёт."""

    arguments = parse_arguments()
    experiment = json.loads(arguments.config.read_text(encoding="utf-8"))
    mode = str(experiment["case_mode"])
    output = arguments.output or OUTPUTS_BY_MODE[mode]
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = project_path(str(experiment["source_protocol"]))
    scene_output = project_path(str(experiment["source_scene_output"]))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    thresholds = MetricRecoveryThresholds.from_mapping(
        protocol["acceptance_thresholds"]
    )
    limit_m = float(experiment["maximum_point_position_error_m"])
    teach_id = next(
        camera["id"] for camera in protocol["cameras"] if camera["role"] == "reference"
    )
    cases = list(iter_cases(experiment))
    minimum_accept_count = int(experiment["aggregate_gate"]["minimum_accept_count"])
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

            for case in cases:
                artifact_seed = (
                    int(experiment["seed"])
                    + surface_number * 1_000_000
                    + repeat_number * 100_000
                    + case.artifact_seed_offset
                )
                artifact = apply_video_artifact_profile(
                    nominal_rgb,
                    case.profile,
                    random_seed=artifact_seed,
                )
                if case.profile_id == "nominal":
                    nominal_pixel_differences.append(
                        int(artifact.metadata["maximum_channel_difference"])
                    )
                physical_case = PhysicalCase(
                    sweep=mode,
                    case_id=case.case_id,
                    true_k1=0.0,
                    estimated_k1=None,
                    calibration_error_k1=None,
                    true_edge_shift_px=0.0,
                    residual_grid_shift_px=0.0,
                    repeat_rgb=artifact.image_rgb,
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
                if not evaluation["alignment_constructed"]:
                    failure = evaluation["alignment_failure"]
                    if not isinstance(failure, str):
                        raise RuntimeError("Ранний отказ не сохранил причину")
                    evaluation["seed_decisions"] = failed_alignment_seed_decisions(
                        seeds=stability_seeds,
                        failure=failure,
                    )
                    evaluation["gate_accept_count"] = 0
                    evaluation["gate_accept_fraction"] = 0.0
                attach_aggregate_decision(
                    evaluation,
                    minimum_accept_count=minimum_accept_count,
                )
                rows.append(
                    {
                        "surface_id": surface_id,
                        "repeat_id": repeat_id,
                        "case_id": case.case_id,
                        "profile_id": case.profile_id,
                        "level_value": case.level_value,
                        "replicate_index": case.replicate_index,
                        "artifact_seed": artifact_seed,
                        "profile": asdict(case.profile),
                        "artifact": artifact.metadata,
                        "evaluation_point_count": int(common_indices.size),
                        "evaluation": evaluation,
                    }
                )
                if (
                    surface_number == 0
                    and repeat_number == 0
                    and case.replicate_index == 0
                ):
                    save_rgb(
                        output / "diagnostics" / f"{case.profile_id}.png",
                        artifact.image_rgb,
                    )
            print(f"completed={surface_id}:{repeat_id}", flush=True)

    nominal_rows = [row for row in rows if row["profile_id"] == "nominal"]
    nominal_control = control_summary(nominal_rows, limit_m=limit_m)
    nominal_aggregate_accept_count = sum(
        bool(row["evaluation"]["aggregate_gate"]["gate_accepted"])
        for row in nominal_rows
    )
    nominal_control["aggregate_accept_count"] = nominal_aggregate_accept_count
    nominal_control["aggregate_accept_fraction"] = nominal_aggregate_accept_count / len(
        nominal_rows
    )
    profile_ids = ordered_profile_ids(experiment, cases)
    summaries = grouped_summaries(rows, ordered_profile_ids=profile_ids)
    component_decision_count = sum(
        len(row["evaluation"]["seed_decisions"]) for row in rows
    )
    aggregate_outcomes: Counter[str] = Counter()
    for row in rows:
        accepted = bool(row["evaluation"]["aggregate_gate"]["gate_accepted"])
        within = bool(row["evaluation"]["metric_within_limit"])
        if accepted:
            aggregate_outcomes["accepted_correct" if within else "false_accept"] += 1
        else:
            aggregate_outcomes["rejected_valid" if within else "rejected_invalid"] += 1

    invalid_reasons = validity_failures(
        experiment=experiment,
        cases=cases,
        rows=rows,
        component_decision_count=component_decision_count,
        nominal_pixel_differences=nominal_pixel_differences,
        nominal_control=nominal_control,
        nominal_aggregate_accept_count=nominal_aggregate_accept_count,
    )
    aggregate_false_accept_count = sum(
        bool(row["evaluation"]["aggregate_false_accept"]) for row in rows
    )
    safety_limit = int(
        experiment["preregistered_safety"]["maximum_aggregate_false_accept_cases"]
    )
    safety_reasons = []
    if aggregate_false_accept_count > safety_limit:
        safety_reasons.append(
            f"Агрегированный gate ложно принял {aggregate_false_accept_count} случаев"
        )
    nominal_reject_count = len(nominal_rows) - nominal_aggregate_accept_count
    availability_reasons = availability_failures(
        experiment=experiment,
        rows=rows,
        summaries=summaries,
        nominal_aggregate_reject_count=nominal_reject_count,
    )
    upstream = upstream_context(experiment)
    experiment_valid = not invalid_reasons
    safety_passed = not safety_reasons
    availability_passed = not availability_reasons
    own_gate_passed = experiment_valid and safety_passed and availability_passed
    confirmation_passed = bool(
        own_gate_passed and (upstream is None or upstream["eligible_for_confirmation"])
    )

    report = {
        "schema_version": 1,
        "experiment": experiment["experiment"],
        "case_mode": mode,
        "configuration": str(arguments.config),
        "configuration_sha256": sha256_file(arguments.config),
        "source_protocol": str(protocol_path),
        "source_protocol_sha256": sha256_file(protocol_path),
        "source_scene_output": str(scene_output),
        "rerendered_blender_scenes": False,
        "physical_case_count": len(rows),
        "component_decision_count": component_decision_count,
        "aggregate_decision_count": len(rows),
        "maximum_identity_pixel_difference": max(nominal_pixel_differences),
        "nominal_control": nominal_control,
        "profile_summaries": summaries,
        "aggregate_gate_outcomes": dict(sorted(aggregate_outcomes.items())),
        "aggregate_false_accept_count": aggregate_false_accept_count,
        "preregistered_validity": experiment["preregistered_validity"],
        "validity_failures": invalid_reasons,
        "experiment_valid": experiment_valid,
        "preregistered_safety": experiment["preregistered_safety"],
        "safety_failures": safety_reasons,
        "safety_passed": safety_passed,
        "preregistered_availability": experiment["preregistered_availability"],
        "availability_failures": availability_reasons,
        "availability_passed": availability_passed,
        "own_gate_passed": own_gate_passed,
        "upstream_c2": upstream,
        "confirmation_passed": confirmation_passed,
        "calibrations": calibration_reports,
        "rows": rows,
        "primary_sources": experiment["primary_sources"],
        "limitations": [
            (
                "Все воздействия являются цифровыми приближениями, "
                "а не полной моделью CVBS."
            ),
            "Текстуры и позы унаследованы от G14-B; holdout относится "
            "к артефактам и seed.",
            "Правило 5/5 является инженерной политикой, а не вероятностной гарантией.",
            "Нет rolling shutter, дисторсии, рельефа и движения объектов.",
        ],
    }
    report_path = output / REPORT_NAMES[mode]
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    labels = (
        [str(value) for value in experiment["noise_standard_deviation_levels"]]
        if mode == "noise_boundary"
        else profile_ids
    )
    build_summary_figure(
        rows,
        profile_ids=profile_ids,
        profile_labels=labels,
        limit_m=limit_m,
        title=(
            "G14-C2: шум и единогласный gate"
            if mode == "noise_boundary"
            else "G14-D: смеси артефактов и единогласный gate"
        ),
        output_path=output / FIGURE_NAMES[mode],
    )
    print(
        json.dumps(
            {
                "nominal_control": nominal_control,
                "profile_summaries": summaries,
                "aggregate_gate_outcomes": dict(sorted(aggregate_outcomes.items())),
                "experiment_valid": experiment_valid,
                "safety_passed": safety_passed,
                "availability_passed": availability_passed,
                "confirmation_passed": confirmation_passed,
                "report": str(report_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
