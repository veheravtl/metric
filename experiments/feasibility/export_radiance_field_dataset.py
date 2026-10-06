#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Экспортирует камеры Blender в учебные форматы NeRF/Nerfstudio/3DGS."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from aerial_mapper.radiance_camera import (
    PinholeIntrinsics,
    blender_horizontal_sensor_intrinsics,
    pixel_rays_world_opengl,
    project_world_points_opengl,
    validate_camera_to_world,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "experiments/configs/nerf_3dgs_prelesson_export.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/nerf_3dgs_prelesson"


def parse_arguments() -> argparse.Namespace:
    """Разбирает пути, не пряча параметры эксперимента в командной строке."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Считает SHA-256 входного файла блоками."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_source_paths(
    scene_directory: Path,
    camera_specification: dict[str, Any],
) -> tuple[Path, Path]:
    """Возвращает RGB и ground mask по роли камеры в Blender-сцене."""

    camera_id = camera_specification["id"]
    if camera_specification["role"] == "reference":
        base = scene_directory / "reference"
        return base / "perspective_rgb.png", base / "ground_mask.png"
    base = scene_directory / "repeat" / camera_id
    return base / "rgb.png", base / "ground_mask.png"


def validate_shared_intrinsics(
    camera_specifications: list[dict[str, Any]],
    render: dict[str, Any],
) -> PinholeIntrinsics:
    """Проверяет общий pinhole и вычисляет intrinsics выбранных камер."""

    if not camera_specifications:
        raise ValueError("Нужна хотя бы одна камера")
    first = camera_specifications[0]
    shared = (
        first["projection"],
        float(first["focal_length_mm"]),
        float(first["sensor_width_mm"]),
    )
    if shared[0] != "perspective":
        raise ValueError("Экспорт поддерживает только perspective-камеры")
    for camera in camera_specifications[1:]:
        candidate = (
            camera["projection"],
            float(camera["focal_length_mm"]),
            float(camera["sensor_width_mm"]),
        )
        if candidate != shared:
            raise ValueError("Выбранные камеры должны иметь общие intrinsics")
    return blender_horizontal_sensor_intrinsics(
        width_px=int(render["width_pixels"]),
        height_px=int(render["height_pixels"]),
        focal_length_mm=shared[1],
        sensor_width_mm=shared[2],
    )


def frame_record(
    camera_metadata: dict[str, Any],
    *,
    include_mask: bool,
    synthetic_nerf_path: bool,
) -> dict[str, Any]:
    """Создаёт запись frame с camera-to-world без смены осей."""

    camera_id = camera_metadata["id"]
    file_path = (
        f"./images/{camera_id}" if synthetic_nerf_path else f"images/{camera_id}.png"
    )
    record: dict[str, Any] = {
        "file_path": file_path,
        "transform_matrix": camera_metadata["matrix_camera_to_world_blender"],
        "camera_id": camera_id,
        "source_role": camera_metadata["role"],
    }
    if include_mask and not synthetic_nerf_path:
        record["mask_path"] = f"masks/{camera_id}.png"
    return record


def nerfstudio_document(
    frames: list[dict[str, Any]],
    intrinsics: PinholeIntrinsics,
) -> dict[str, Any]:
    """Создаёт transforms.json с явными intrinsics Nerfstudio."""

    return {
        "camera_model": "OPENCV",
        "fl_x": intrinsics.focal_x_px,
        "fl_y": intrinsics.focal_y_px,
        "cx": intrinsics.principal_x_px,
        "cy": intrinsics.principal_y_px,
        "w": intrinsics.width_px,
        "h": intrinsics.height_px,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": frames,
    }


def synthetic_nerf_document(
    frames: list[dict[str, Any]],
    intrinsics: PinholeIntrinsics,
) -> dict[str, Any]:
    """Создаёт формат оригинального Blender loader NeRF и 3DGS."""

    return {
        "camera_angle_x": intrinsics.horizontal_field_of_view_rad,
        "frames": frames,
    }


def projection_validation(
    *,
    camera_id: str,
    camera_metadata: dict[str, Any],
    intrinsics: PinholeIntrinsics,
    world_xyz_m: np.ndarray,
    truth_pixels_opencv: np.ndarray,
    truth_in_frame: np.ndarray,
) -> dict[str, Any]:
    """Сверяет собственную проекцию и главный луч с истиной Blender."""

    matrix = np.asarray(
        camera_metadata["matrix_camera_to_world_blender"],
        dtype=np.float64,
    )
    validate_camera_to_world(matrix)
    projected, in_front = project_world_points_opengl(
        world_xyz_m,
        matrix,
        intrinsics,
    )
    # Blender-метаданные проекта хранят OpenCV-координаты с центром верхнего
    # левого пикселя (0, 0). Nerfstudio считает тот же центр как (0.5, 0.5).
    expected_nerf_pixels = truth_pixels_opencv + 0.5
    selected = truth_in_frame & in_front
    residuals = np.linalg.norm(
        projected[selected] - expected_nerf_pixels[selected],
        axis=1,
    )
    if residuals.size == 0:
        raise RuntimeError(f"Нет контрольных проекций для {camera_id}")

    center = np.asarray(
        [[intrinsics.principal_x_px, intrinsics.principal_y_px]],
        dtype=np.float64,
    )
    origins, directions = pixel_rays_world_opengl(center, matrix, intrinsics)
    target = np.asarray(camera_metadata["target_m"], dtype=np.float64)
    expected_direction = target - origins[0]
    expected_direction /= np.linalg.norm(expected_direction)
    cosine = float(np.clip(directions[0] @ expected_direction, -1.0, 1.0))
    angle_degrees = float(np.degrees(np.arccos(cosine)))
    location_error = float(
        np.linalg.norm(origins[0] - np.asarray(camera_metadata["location_m"]))
    )
    return {
        "camera_id": camera_id,
        "control_point_count": int(residuals.size),
        "projection_error_p95_px": float(np.percentile(residuals, 95)),
        "projection_error_maximum_px": float(np.max(residuals)),
        "central_ray_target_angle_degrees": angle_degrees,
        "camera_origin_error_m": location_error,
    }


def main() -> None:
    """Копирует четыре кадра, пишет три transform-файла и проверяет камеры."""

    arguments = parse_arguments()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    scene_directory = PROJECT_ROOT / config["source_scene"]
    frozen_path = scene_directory / "frozen_scene_config.json"
    metadata_path = scene_directory / "generation_metadata.json"
    truth_path = scene_directory / "terrain_truth.npz"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    train_ids = list(config["train_camera_ids"])
    test_ids = list(config["test_camera_ids"])
    selected_ids = train_ids + test_ids
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Train/test camera id должны быть уникальными")
    specifications = {item["id"]: item for item in frozen["cameras"]}
    camera_metadata = {item["id"]: item for item in metadata["cameras"]}
    missing = sorted(set(selected_ids) - specifications.keys())
    if missing:
        raise ValueError(f"В сцене нет камер: {', '.join(missing)}")
    intrinsics = validate_shared_intrinsics(
        [specifications[camera_id] for camera_id in selected_ids],
        frozen["render"],
    )

    image_directory = arguments.output / "images"
    mask_directory = arguments.output / "masks"
    image_directory.mkdir(parents=True, exist_ok=True)
    include_masks = bool(config["copy_ground_masks"])
    if include_masks:
        mask_directory.mkdir(parents=True, exist_ok=True)
    for camera_id in selected_ids:
        rgb_source, mask_source = camera_source_paths(
            scene_directory,
            specifications[camera_id],
        )
        shutil.copy2(rgb_source, image_directory / f"{camera_id}.png")
        if include_masks:
            shutil.copy2(mask_source, mask_directory / f"{camera_id}.png")

    nerfstudio_frames = [
        frame_record(
            camera_metadata[camera_id],
            include_mask=include_masks,
            synthetic_nerf_path=False,
        )
        for camera_id in selected_ids
    ]
    train_frames = [
        frame_record(
            camera_metadata[camera_id],
            include_mask=False,
            synthetic_nerf_path=True,
        )
        for camera_id in train_ids
    ]
    test_frames = [
        frame_record(
            camera_metadata[camera_id],
            include_mask=False,
            synthetic_nerf_path=True,
        )
        for camera_id in test_ids
    ]
    documents = {
        "transforms.json": nerfstudio_document(nerfstudio_frames, intrinsics),
        "transforms_train.json": synthetic_nerf_document(train_frames, intrinsics),
        "transforms_test.json": synthetic_nerf_document(test_frames, intrinsics),
        "transforms_val.json": synthetic_nerf_document(test_frames, intrinsics),
    }
    for filename, document in documents.items():
        (arguments.output / filename).write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    with np.load(truth_path) as truth:
        world_xyz_m = np.asarray(truth["world_xyz_m"], dtype=np.float64)
        truth_camera_ids = list(truth["camera_ids"])
        truth_pixels = np.asarray(truth["pixel_xy"], dtype=np.float64)
        truth_in_frame = np.asarray(truth["in_frame"], dtype=bool)
    validation_rows = []
    for camera_id in selected_ids:
        truth_index = truth_camera_ids.index(camera_id)
        validation_rows.append(
            projection_validation(
                camera_id=camera_id,
                camera_metadata=camera_metadata[camera_id],
                intrinsics=intrinsics,
                world_xyz_m=world_xyz_m,
                truth_pixels_opencv=truth_pixels[truth_index],
                truth_in_frame=truth_in_frame[truth_index],
            )
        )

    thresholds = config["validation_thresholds"]
    failures = []
    for row in validation_rows:
        if row["projection_error_p95_px"] > thresholds["maximum_projection_p95_px"]:
            failures.append(f"{row['camera_id']}: превышен p95 проекции")
        if (
            row["projection_error_maximum_px"]
            > thresholds["maximum_projection_error_px"]
        ):
            failures.append(f"{row['camera_id']}: превышен максимум проекции")
        if (
            row["central_ray_target_angle_degrees"]
            > thresholds["maximum_central_ray_angle_degrees"]
        ):
            failures.append(f"{row['camera_id']}: главный луч не смотрит в target")
        if row["camera_origin_error_m"] > 1e-9:
            failures.append(f"{row['camera_id']}: перенос pose не равен location")

    report = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "configuration": str(arguments.config),
        "configuration_sha256": sha256_file(arguments.config),
        "source_scene": str(scene_directory),
        "source_frozen_config_sha256": sha256_file(frozen_path),
        "source_generation_metadata_sha256": sha256_file(metadata_path),
        "coordinate_convention": {
            "pose": "camera-to-world",
            "camera_axes": "+X right, +Y up, +Z back, looks along -Z",
            "world_axes": metadata["world_axis_convention"],
            "pixel_centres": "top-left pixel centre is (0.5, 0.5)",
            "world_units": metadata["world_units"],
        },
        "intrinsics": asdict(intrinsics),
        "train_camera_ids": train_ids,
        "test_camera_ids": test_ids,
        "validation_rows": validation_rows,
        "validation_failures": failures,
        "passed": not failures,
        "training_ready": False,
        "limitations": config["limitations"],
    }
    report_path = arguments.output / "camera_export_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"report={report_path}")
    print(f"passed={not failures}")
    print("training_ready=False")


if __name__ == "__main__":
    main()
