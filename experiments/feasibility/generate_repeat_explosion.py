#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Создать Repeat-фрагмент HKairport03 с контролируемым взрывом.

Точка события задаётся на размеченном Teach-кадре. Скрипт переносит её через
сохранённую карту в Repeat, строит траекторию по чистым Repeat-кадрам и только
после этого накладывает внешний футаж. Кадры до события копируются побайтно.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from aerial_mapper.alignment import SiftRansacConfig, align_frame_to_reference
from aerial_mapper.cloud_dataset import load_positioned_images
from aerial_mapper.synthetic_event import (
    OpticalFlowAnchorConfig,
    composite_layer_at_anchor,
    extract_changed_foreground_alpha,
    track_ground_anchor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACTED_ROOT = PROJECT_ROOT / "data/queries/mars_lvig/extracted"
DEFAULT_SOURCE = EXTRACTED_ROOT / "HKairport03"
DEFAULT_TEACH_SOURCE = EXTRACTED_ROOT / "HKairport01"
DEFAULT_OUTPUT = EXTRACTED_ROOT / "expl_01"
DEFAULT_PROJECT = PROJECT_ROOT / "outputs/operator_projects/project.aerial-project.json"

EVENT_START_FRAME = 25
FIRST_FRAME = 0
DEFAULT_LAST_FRAME = 100
TEACH_EVENT_FRAME = 25
TEACH_EVENT_POINT_PX = (907.0, 986.0)

EFFECT_BACKGROUND_SECONDS = 7.40
EFFECT_START_SECONDS = 7.75
EFFECT_CROP_XYXY = (40, 20, 1240, 715)
EFFECT_OUTPUT_SIZE_WH = (660, 382)
EFFECT_VISUAL_ANCHOR_SOURCE_PX = (640.0, 515.0)

FOOTAGE_SOURCE = {
    "title": "Explosion B-Roll",
    "page_url": "https://www.dvidshub.net/video/462158/explosion-b-roll",
    "author": "Paul Criner",
    "organization": "Bureau of Alcohol, Tobacco, Firearms and Explosives",
    "recorded_date": "2016-05-04",
    "license": "Public Domain",
}


def parse_args() -> argparse.Namespace:
    """Читает только параметры, меняющиеся между локальными запусками."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--teach-source", type=Path, default=DEFAULT_TEACH_SOURCE)
    parser.add_argument("--operator-project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument(
        "--effect-video",
        type=Path,
        default=Path("/tmp/aerial-map-explosion-reference.mp4"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--last-frame", type=int, default=DEFAULT_LAST_FRAME)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Считает контрольную сумму файла потоково."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def perspective_point(
    point_px: tuple[float, float], homography: np.ndarray
) -> np.ndarray:
    """Переносит одну точку проективной матрицей 3×3."""

    return cv2.perspectiveTransform(
        np.asarray(point_px, dtype=np.float64).reshape(1, 1, 2),
        homography,
    ).reshape(2)


def locate_repeat_event_anchor(
    *,
    teach_source: Path,
    repeat_source: Path,
    project_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Переносит Teach-точку в Repeat через общую систему пикселей карты."""

    project = json.loads(project_path.read_text(encoding="utf-8"))
    teach_frame = project["alignment_session"]["frames"][TEACH_EVENT_FRAME]
    alignment = teach_frame.get("alignment")
    if alignment is None:
        raise RuntimeError(f"Teach-кадр {TEACH_EVENT_FRAME} не привязан к карте")
    teach_to_map = np.asarray(
        alignment["homography_frame_to_map"], dtype=np.float64
    )
    event_map_px = perspective_point(TEACH_EVENT_POINT_PX, teach_to_map)

    teach_positions = load_positioned_images(teach_source)
    repeat_positions = load_positioned_images(repeat_source)
    repeat_matches = np.flatnonzero(
        repeat_positions.image_ids == EVENT_START_FRAME
    )
    if repeat_matches.size != 1:
        raise RuntimeError("Не найден единственный Repeat-кадр начала события")
    repeat_row = int(repeat_matches[0])
    distances = np.linalg.norm(
        teach_positions.xy_meters - repeat_positions.xy_meters[repeat_row],
        axis=1,
    )
    nearest_order = np.argsort(distances)
    teach_frames = project["alignment_session"]["frames"]
    chosen_teach_id: int | None = None
    for teach_row in nearest_order:
        candidate_id = int(teach_positions.image_ids[teach_row])
        if teach_frames[candidate_id].get("alignment") is not None:
            chosen_teach_id = candidate_id
            chosen_distance = float(distances[teach_row])
            break
    if chosen_teach_id is None:
        raise RuntimeError("Рядом с Repeat-событием нет привязанного Teach-кадра")

    teach_bgr = cv2.imread(
        str(teach_source / "images" / f"{chosen_teach_id:06d}.jpg"),
        cv2.IMREAD_COLOR,
    )
    repeat_bgr = cv2.imread(
        str(repeat_source / "images" / f"{EVENT_START_FRAME:06d}.jpg"),
        cv2.IMREAD_COLOR,
    )
    if teach_bgr is None or repeat_bgr is None:
        raise RuntimeError("Не удалось прочитать Teach/Repeat для исходной привязки")
    local_alignment = align_frame_to_reference(
        cv2.cvtColor(teach_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(repeat_bgr, cv2.COLOR_BGR2RGB),
        config=SiftRansacConfig(max_features_per_image=8_000),
    )
    chosen_teach_to_map = np.asarray(
        teach_frames[chosen_teach_id]["alignment"]["homography_frame_to_map"],
        dtype=np.float64,
    )
    repeat_to_map = (
        chosen_teach_to_map
        @ local_alignment.homography_frame_to_reference
    )
    repeat_to_map /= repeat_to_map[2, 2]
    event_repeat_px = perspective_point(event_map_px, np.linalg.inv(repeat_to_map))
    height, width = repeat_bgr.shape[:2]
    if not (0.0 <= event_repeat_px[0] < width and 0.0 <= event_repeat_px[1] < height):
        raise RuntimeError("Перенесённая точка события оказалась вне Repeat-кадра")

    evidence = {
        "teach_annotation": {
            "frame_index": TEACH_EVENT_FRAME,
            "point_px": list(TEACH_EVENT_POINT_PX),
            "origin": (
                "Оценка центра временной точки 58 по предоставленному скриншоту; "
                "точка не была сохранена в operator project"
            ),
        },
        "event_map_px": event_map_px.tolist(),
        "repeat_start_point_px": event_repeat_px.tolist(),
        "nearest_mapped_teach_frame": chosen_teach_id,
        "gps_distance_to_nearest_teach_m": chosen_distance,
        "sift_ransac": {
            "ratio_match_count": local_alignment.ratio_match_count,
            "inlier_count": local_alignment.inlier_count,
            "inlier_ratio": local_alignment.inlier_ratio,
            "inlier_spatial_coverage_fraction": (
                local_alignment.inlier_spatial_coverage_fraction
            ),
        },
        "homography_repeat_start_to_map": repeat_to_map.tolist(),
    }
    return event_repeat_px, evidence


def read_video_at(capture: cv2.VideoCapture, seconds: float) -> np.ndarray:
    """Декодирует ближайший кадр футажа к указанному времени."""

    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1_000.0)
    success, frame = capture.read()
    if not success or frame is None:
        raise RuntimeError(f"Не удалось прочитать футаж в момент {seconds:.3f} с")
    return frame


def effect_layer(
    capture: cv2.VideoCapture,
    background_crop: np.ndarray,
    *,
    seconds: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Выделяет эффект, масштабирует его и возвращает положение якоря слоя."""

    x0, y0, x1, y1 = EFFECT_CROP_XYXY
    current = read_video_at(capture, seconds)[y0:y1, x0:x1]
    if current.shape != background_crop.shape:
        raise RuntimeError("Crop футажа вышел за границы декодированного кадра")
    alpha = extract_changed_foreground_alpha(current, background_crop)
    output_width, output_height = EFFECT_OUTPUT_SIZE_WH
    layer = cv2.resize(
        current,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )
    alpha = cv2.resize(
        alpha,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )
    anchor_x = (
        (EFFECT_VISUAL_ANCHOR_SOURCE_PX[0] - x0)
        * output_width
        / (x1 - x0)
    )
    anchor_y = (
        (EFFECT_VISUAL_ANCHOR_SOURCE_PX[1] - y0)
        * output_height
        / (y1 - y0)
    )
    return layer, alpha, (anchor_x, anchor_y)


def write_image_index_subset(source: Path, target: Path, *, last_frame: int) -> None:
    """Копирует только строки индекса, соответствующие созданным кадрам."""

    with source.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        rows = [row for row in reader if int(row["Image Id"]) <= last_frame]
        fieldnames = reader.fieldnames
    if fieldnames is None or len(rows) != last_frame + 1:
        raise RuntimeError("Индекс кадров исходного фрагмента неполон")
    with target.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate(args: argparse.Namespace) -> Path:
    """Создаёт последовательность атомарно через соседний временный каталог."""

    source = args.source.expanduser().resolve()
    teach_source = args.teach_source.expanduser().resolve()
    project_path = args.operator_project.expanduser().resolve()
    effect_video = args.effect_video.expanduser().resolve()
    output = args.output.expanduser().resolve()
    last_frame = int(args.last_frame)
    if last_frame < EVENT_START_FRAME:
        raise ValueError("Последний кадр не может предшествовать событию")
    if output.exists():
        raise FileExistsError(f"Выходной каталог уже существует: {output}")
    temporary_output = output.with_name(f".{output.name}.tmp")
    if temporary_output.exists():
        raise FileExistsError(f"Остался временный каталог: {temporary_output}")
    if not effect_video.is_file():
        raise FileNotFoundError(f"Не найден футаж эффекта: {effect_video}")

    start_anchor, localization = locate_repeat_event_anchor(
        teach_source=teach_source,
        repeat_source=source,
        project_path=project_path,
    )
    clean_paths = [
        source / "images" / f"{frame_index:06d}.jpg"
        for frame_index in range(EVENT_START_FRAME, last_frame + 1)
    ]
    tracking_config = OpticalFlowAnchorConfig()
    trajectory = track_ground_anchor(
        clean_paths,
        first_frame_index=EVENT_START_FRAME,
        initial_anchor_px=(float(start_anchor[0]), float(start_anchor[1])),
        config=tracking_config,
    )

    capture = cv2.VideoCapture(str(effect_video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV не открыл футаж: {effect_video}")
    x0, y0, x1, y1 = EFFECT_CROP_XYXY
    background_crop = read_video_at(capture, EFFECT_BACKGROUND_SECONDS)[
        y0:y1, x0:x1
    ]
    temporary_images = temporary_output / "images"
    temporary_images.mkdir(parents=True)
    try:
        for frame_index in range(FIRST_FRAME, last_frame + 1):
            source_path = source / "images" / f"{frame_index:06d}.jpg"
            target_path = temporary_images / source_path.name
            if frame_index < EVENT_START_FRAME:
                shutil.copy2(source_path, target_path)
                continue
            clean_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if clean_bgr is None:
                raise RuntimeError(f"Не удалось прочитать исходник: {source_path}")
            event_seconds = (
                EFFECT_START_SECONDS
                + (frame_index - EVENT_START_FRAME) / 10.0
            )
            layer, alpha, layer_anchor = effect_layer(
                capture,
                background_crop,
                seconds=event_seconds,
            )
            trajectory_index = frame_index - EVENT_START_FRAME
            target_anchor = trajectory.points_px[trajectory_index]
            composed = composite_layer_at_anchor(
                clean_bgr,
                layer,
                alpha,
                layer_anchor_px=layer_anchor,
                target_anchor_px=(float(target_anchor[0]), float(target_anchor[1])),
            )
            if not cv2.imwrite(
                str(target_path),
                composed,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise RuntimeError(f"Не удалось записать кадр: {target_path}")
    finally:
        capture.release()

    write_image_index_subset(
        source / "image_ids.csv",
        temporary_output / "image_ids.csv",
        last_frame=last_frame,
    )
    shutil.copy2(source / "gps.csv", temporary_output / "gps.csv")
    pristine_checksums = {
        f"{frame_index:06d}.jpg": sha256_file(
            temporary_images / f"{frame_index:06d}.jpg"
        )
        for frame_index in range(FIRST_FRAME, EVENT_START_FRAME)
    }
    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "sequence_name": output.name,
        "source_sequence": str(source),
        "teach_sequence": str(teach_source),
        "operator_project": str(project_path),
        "frame_range_inclusive": [FIRST_FRAME, last_frame],
        "image_count": last_frame + 1,
        "event": {
            "start_frame": EVENT_START_FRAME,
            "end_frame": last_frame,
            "duration_frames_inclusive": last_frame - EVENT_START_FRAME + 1,
            "duration_seconds_at_10_fps": (
                last_frame - EVENT_START_FRAME + 1
            )
            / 10.0,
            "frames_before_event_are_byte_identical": True,
        },
        "localization": localization,
        "anchor_trajectory": [
            {
                "frame_index": int(frame_index),
                "point_px": point.tolist(),
            }
            for frame_index, point in zip(
                trajectory.frame_indices,
                trajectory.points_px,
                strict=True,
            )
        ],
        "tracking_config": asdict(tracking_config),
        "tracking_steps": [asdict(step) for step in trajectory.steps],
        "effect_footage": {
            **FOOTAGE_SOURCE,
            "local_file_used": str(effect_video),
            "sha256": sha256_file(effect_video),
            "background_seconds": EFFECT_BACKGROUND_SECONDS,
            "effect_start_seconds": EFFECT_START_SECONDS,
            "crop_xyxy_source_px": list(EFFECT_CROP_XYXY),
            "output_size_wh_px": list(EFFECT_OUTPUT_SIZE_WH),
            "visual_anchor_source_px": list(EFFECT_VISUAL_ANCHOR_SOURCE_PX),
        },
        "pristine_frame_sha256": pristine_checksums,
    }
    (temporary_output / "generation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_output.rename(output)
    return output


def main() -> None:
    """CLI-точка входа воспроизводимого эксперимента."""

    output = generate(parse_args())
    print(output)


if __name__ == "__main__":
    main()
