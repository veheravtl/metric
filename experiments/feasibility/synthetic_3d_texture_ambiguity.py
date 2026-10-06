#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""G13-R: наблюдаемый guard самопохожести Teach-текстуры."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from aerial_mapper.synthetic_3d import run_synthetic_3d_smoke
from aerial_mapper.texture_ambiguity import (
    TextureAmbiguityThresholds,
    assess_reference_texture_ambiguity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "experiments/configs/synthetic_3d_g13r_ambiguity.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/synthetic_3d/g13r_ambiguity"
DEFAULT_BLENDER = PROJECT_ROOT / ".tools/blender-4.5.14-linux-x64/blender"
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts/blender_generate_terrain_scene.py"


def parse_arguments() -> argparse.Namespace:
    """Разбирает только пути; все численные решения живут в frozen config."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument(
        "--reuse-scenes",
        action="store_true",
        help="Переиспользовать holdout только при совпадении frozen config.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Возвращает воспроизводимый отпечаток конфигурации или отчёта."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    """Читает PNG и явно переводит порядок каналов OpenCV BGR в RGB."""

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Не удалось прочитать RGB: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_mask(path: Path, erosion_px: int) -> np.ndarray:
    """Читает Teach-маску и повторяет эрозию рабочего matcher."""

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Не удалось прочитать маску: {path}")
    binary = np.where(mask >= 128, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * erosion_px + 1, 2 * erosion_px + 1),
    )
    return cv2.erode(binary, kernel, iterations=1)


def build_scene_description(
    base_protocol: dict[str, Any],
    surface: dict[str, Any],
) -> dict[str, Any]:
    """Строит дешёвую holdout-сцену только с одной Teach-камерой."""

    world = base_protocol["world"]
    reference_camera = next(
        camera
        for camera in base_protocol["cameras"]
        if camera["role"] == "reference"
    )
    return {
        "schema_version": 1,
        "scenario_id": f"g13r_{surface['id']}",
        "seed": int(surface["scene_seed"]),
        "world": {
            "units": world["units"],
            "axis_convention": world["axis_convention"],
            "ground_size_m": world["ground_size_m"],
            "appearance": "terrain_metric_texture",
            "show_control_points": False,
            "metric_texture": {
                **world["metric_texture"],
                "class": surface["texture_class"],
            },
            "terrain": {
                "kind": "flat",
                "grid_vertices_xy": world["grid_vertices_xy"],
                "truth_sample_stride": world["truth_sample_stride"],
            },
        },
        "render": base_protocol["render"],
        "cameras": [reference_camera],
        "objects": [],
        "clutter": {
            **base_protocol["clutter_defaults"],
            "count": int(surface["clutter_count"]),
        },
    }


def assess_scene(
    scene_directory: Path,
    *,
    surface_id: str,
    texture_class: str,
    erosion_px: int,
    thresholds: TextureAmbiguityThresholds,
) -> dict[str, Any]:
    """Считает guard только по тем данным, которые доступны оператору."""

    assessment = assess_reference_texture_ambiguity(
        load_rgb(scene_directory / "reference/perspective_rgb.png"),
        load_mask(scene_directory / "reference/ground_mask.png", erosion_px),
        thresholds=thresholds,
    )
    return {
        "surface_id": surface_id,
        # Класс используется оценщиком только для группировки после решения.
        "texture_class": texture_class,
        **asdict(assessment),
    }


def reclassify(original: str, within_limits: bool, rejected: bool) -> str:
    """Показывает продуктовый эффект guard, не пересчитывая скрытую истину."""

    if not rejected or original == "alignment_failure":
        return original
    return "rejected_valid" if within_limits else "rejected_invalid"


def summarize_g13_effect(
    source_report: dict[str, Any],
    decisions: dict[str, bool],
) -> dict[str, Any]:
    """Переоценивает исходы G13 после предварительного Teach-guard."""

    absolute_counts: Counter[str] = Counter()
    absolute_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for row in source_report["alignment_rows"]:
        point_metric = row["absolute_point_metric"]
        within = bool(point_metric and point_metric["within_point_limit"])
        classification = reclassify(
            row["absolute_classification"],
            within,
            decisions[row["surface_id"]],
        )
        absolute_counts[classification] += 1
        absolute_by_class[row["texture_class"]][classification] += 1

    product_counts: dict[str, Counter[str]] = defaultdict(Counter)
    product_by_class: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for row in source_report["vector_summaries"]:
        classification = reclassify(
            row["classification"],
            bool(row["product_within_limits"]),
            decisions[row["surface_id"]],
        )
        product_counts[row["pipeline"]][classification] += 1
        product_by_class[row["texture_class"]][row["pipeline"]][
            classification
        ] += 1

    return {
        "absolute_classification_counts": dict(absolute_counts),
        "absolute_texture_class_counts": {
            key: dict(value) for key, value in absolute_by_class.items()
        },
        "product_classification_counts": {
            key: dict(value) for key, value in product_counts.items()
        },
        "product_texture_class_counts": {
            texture: {
                pipeline: dict(counts) for pipeline, counts in pipelines.items()
            }
            for texture, pipelines in product_by_class.items()
        },
    }


def validation_failures(
    rows: list[dict[str, Any]],
    expectations: dict[str, Any],
) -> list[str]:
    """Сверяет независимые seed с заранее зафиксированными исходами классов."""

    failures: list[str] = []
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[row["texture_class"]].append(row)
    for texture_class, limits in expectations.items():
        selected = by_class.get(texture_class, [])
        rejected = sum(row["rejected_as_ambiguous"] for row in selected)
        if len(selected) != int(limits["case_count"]):
            failures.append(
                f"{texture_class}: случаев {len(selected)} != {limits['case_count']}"
            )
        if rejected < int(limits["minimum_rejected"]):
            failures.append(
                f"{texture_class}: отказов {rejected} < {limits['minimum_rejected']}"
            )
        if rejected > int(limits["maximum_rejected"]):
            failures.append(
                f"{texture_class}: отказов {rejected} > {limits['maximum_rejected']}"
            )
    return failures


def main() -> None:
    """Проверяет development G13, рендерит holdout и сохраняет один JSON."""

    arguments = parse_arguments()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    base_protocol_path = PROJECT_ROOT / config["source_scene_protocol"]
    source_report_path = PROJECT_ROOT / config["source_evaluation_report"]
    source_output = PROJECT_ROOT / config["source_scene_output"]
    base_protocol = json.loads(base_protocol_path.read_text(encoding="utf-8"))
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    thresholds = TextureAmbiguityThresholds(**config["ambiguity_thresholds"])
    erosion_px = int(base_protocol["teach_mask_erosion_px"])

    development_rows = []
    for surface in base_protocol["surfaces"]:
        development_rows.append(
            assess_scene(
                source_output / "scenes" / surface["id"],
                surface_id=surface["id"],
                texture_class=surface["texture_class"],
                erosion_px=erosion_px,
                thresholds=thresholds,
            )
        )
    development_decisions = {
        row["surface_id"]: bool(row["rejected_as_ambiguous"])
        for row in development_rows
    }
    g13_effect = summarize_g13_effect(source_report, development_decisions)

    validation_rows = []
    for surface in config["validation_surfaces"]:
        scene_directory = arguments.output / "scenes" / surface["id"]
        scene_directory.mkdir(parents=True, exist_ok=True)
        scene_description = build_scene_description(base_protocol, surface)
        frozen_path = scene_directory / "frozen_scene_config.json"
        reusable = bool(
            frozen_path.exists()
            and json.loads(frozen_path.read_text(encoding="utf-8"))
            == scene_description
            and (scene_directory / "generation_metadata.json").exists()
        )
        frozen_path.write_text(
            json.dumps(scene_description, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not (arguments.reuse_scenes and reusable):
            smoke = run_synthetic_3d_smoke(
                description_path=frozen_path,
                output_directory=scene_directory,
                blender_executable=arguments.blender,
                generator_script=GENERATOR_SCRIPT,
                required_camera_roles=("reference",),
            )
            if not smoke["passed"]:
                raise RuntimeError(f"Blender smoke не пройден: {surface['id']}")
        validation_rows.append(
            assess_scene(
                scene_directory,
                surface_id=surface["id"],
                texture_class=surface["texture_class"],
                erosion_px=erosion_px,
                thresholds=thresholds,
            )
        )
        print(f"completed_surface={surface['id']}")

    failures = validation_failures(
        validation_rows,
        config["validation_expectations"],
    )
    if g13_effect["absolute_classification_counts"].get("false_accept", 0):
        failures.append("После guard осталось абсолютное ложное принятие G13")
    for pipeline, counts in g13_effect["product_classification_counts"].items():
        if counts.get("false_accept", 0):
            failures.append(f"После guard осталось ложное принятие {pipeline}")
    rich_counts = g13_effect["absolute_texture_class_counts"]["rich_irregular"]
    if rich_counts.get("accepted_correct", 0) != 12:
        failures.append("Guard нарушил богатый контроль 12/12")

    arguments.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "configuration": str(arguments.config),
        "configuration_sha256": sha256_file(arguments.config),
        "source_scene_protocol": str(base_protocol_path),
        "source_scene_protocol_sha256": sha256_file(base_protocol_path),
        "source_evaluation_report": str(source_report_path),
        "source_evaluation_report_sha256": sha256_file(source_report_path),
        "input_contract": ["Teach RGB", "conservative Teach ground mask"],
        "ambiguity_thresholds": asdict(thresholds),
        "development_rows": development_rows,
        "g13_guarded_effect": g13_effect,
        "validation_rows": validation_rows,
        "validation_failures": failures,
        "preregistered_passed": not failures,
        "limitations": config["limitations"],
    }
    report_path = arguments.output / config["report_filename"]
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"report={report_path}")
    print(f"preregistered_passed={not failures}")


if __name__ == "__main__":
    main()
