#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Собирает 190 исходных кадров Repeat-пролёта и 15 кадров надирного взрыва."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from aerial_mapper.synthetic_event import (
    composite_layer_at_anchor,
    track_ground_anchor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_IMAGES = PROJECT_ROOT / "data/queries/mars_lvig/extracted/HKairport03/images"
BLENDER_OUTPUT = PROJECT_ROOT / "outputs/synthetic_events/expl_01_test_v6"
OUTPUT = PROJECT_ROOT / "data/queries/mars_lvig/extracted/expl_01_test"
SOURCE_SEQUENCE = PROJECT_ROOT / "data/queries/mars_lvig/extracted/HKairport03"

OUTPUT_FIRST_FRAME = 0
ANCHOR_REFERENCE_FRAME = 25
EVENT_START_FRAME = 190
LAST_REPEAT_FRAME = 204
REPEAT_REFERENCE_ANCHOR_PX = (920.7292521320752, 1012.3170809543077)

# Локальный масштаб рассчитан производной Repeat→map в точке события и
# ручным масштабом карты 0,2539687368 м/px. Квадрат Blender охватывает 32 м.
METERS_PER_REPEAT_PIXEL_X = 0.06314880926117447
METERS_PER_REPEAT_PIXEL_Y = 0.06388349924442724
BLENDER_CANVAS_METERS = 32.0
LAYER_WIDTH_PX = round(BLENDER_CANVAS_METERS / METERS_PER_REPEAT_PIXEL_X)
LAYER_HEIGHT_PX = round(BLENDER_CANVAS_METERS / METERS_PER_REPEAT_PIXEL_Y)


def write_image_index(target: Path) -> None:
    """Записывает обязательный индекс для полного фрагмента 0–204."""

    source = SOURCE_SEQUENCE / "image_ids.csv"
    with source.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        rows = [
            row
            for row in reader
            if OUTPUT_FIRST_FRAME <= int(row["Image Id"]) <= LAST_REPEAT_FRAME
        ]
        fieldnames = reader.fieldnames
    expected_count = LAST_REPEAT_FRAME - OUTPUT_FIRST_FRAME + 1
    if fieldnames is None or len(rows) != expected_count:
        raise RuntimeError("Исходный индекс не содержит все кадры теста")
    with target.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Создаёт кадры 0–204: чистые 0–189 и событие 190–204."""

    if OUTPUT.exists():
        raise FileExistsError(f"Тестовый каталог уже существует: {OUTPUT}")
    rgba_directory = BLENDER_OUTPUT / "rgba"
    expected_layers = [
        rgba_directory / f"effect_{index:04d}.png"
        for index in range(1, LAST_REPEAT_FRAME - EVENT_START_FRAME + 2)
    ]
    missing = [path for path in expected_layers if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Не хватает Blender-слоёв: {missing[:3]}")

    clean_paths = [
        SOURCE_IMAGES / f"{frame_index:06d}.jpg"
        for frame_index in range(ANCHOR_REFERENCE_FRAME, LAST_REPEAT_FRAME + 1)
    ]
    trajectory = track_ground_anchor(
        clean_paths,
        first_frame_index=ANCHOR_REFERENCE_FRAME,
        initial_anchor_px=REPEAT_REFERENCE_ANCHOR_PX,
    )
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Остался временный каталог: {temporary}")
    images_directory = temporary / "images"
    images_directory.mkdir(parents=True)

    # До события сохраняем исходные JPEG-файлы без декодирования и повторного
    # сжатия. Это даёт алгоритму честный контрольный участок: любые изменения
    # пикселей на кадрах 0–189 принадлежат самому Repeat-пролёту, а не генератору.
    for frame_index in range(OUTPUT_FIRST_FRAME, EVENT_START_FRAME):
        source = SOURCE_IMAGES / f"{frame_index:06d}.jpg"
        if not source.is_file():
            raise FileNotFoundError(f"Не найден исходный кадр: {source}")
        shutil.copy2(source, images_directory / source.name)

    layer_anchor = (LAYER_WIDTH_PX / 2.0, LAYER_HEIGHT_PX / 2.0)
    for layer_index, frame_index in enumerate(
        range(EVENT_START_FRAME, LAST_REPEAT_FRAME + 1),
        start=1,
    ):
        clean = cv2.imread(str(SOURCE_IMAGES / f"{frame_index:06d}.jpg"))
        rgba = cv2.imread(
            str(rgba_directory / f"effect_{layer_index:04d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        if clean is None or rgba is None or rgba.shape[2] != 4:
            raise RuntimeError(f"Не удалось прочитать кадр {frame_index}")
        layer = cv2.resize(
            rgba[:, :, :3],
            (LAYER_WIDTH_PX, LAYER_HEIGHT_PX),
            interpolation=cv2.INTER_AREA,
        )
        alpha = cv2.resize(
            rgba[:, :, 3].astype(np.float32) / 255.0,
            (LAYER_WIDTH_PX, LAYER_HEIGHT_PX),
            interpolation=cv2.INTER_AREA,
        )
        anchor = trajectory.points_px[frame_index - ANCHOR_REFERENCE_FRAME]
        composed = composite_layer_at_anchor(
            clean,
            layer,
            alpha,
            layer_anchor_px=layer_anchor,
            target_anchor_px=(float(anchor[0]), float(anchor[1])),
        )
        target = images_directory / f"{frame_index:06d}.jpg"
        if not cv2.imwrite(
            str(target),
            composed,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        ):
            raise RuntimeError(f"Не удалось записать {target}")

    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "Контрольный участок 0–189 и 15 кадров события 190–204",
        "source_images": str(SOURCE_IMAGES),
        "blender_output": str(BLENDER_OUTPUT),
        "output_frame_range_inclusive": [OUTPUT_FIRST_FRAME, LAST_REPEAT_FRAME],
        "event_frame_range_inclusive": [EVENT_START_FRAME, LAST_REPEAT_FRAME],
        "clean_prefix_frame_count": EVENT_START_FRAME - OUTPUT_FIRST_FRAME,
        "clean_prefix_is_byte_identical_to_source": True,
        "blender_frame_range_inclusive": [1, 15],
        "projection": "orthographic_nadir",
        "layer_size_px": [LAYER_WIDTH_PX, LAYER_HEIGHT_PX],
        "meters_per_repeat_pixel_xy": [
            METERS_PER_REPEAT_PIXEL_X,
            METERS_PER_REPEAT_PIXEL_Y,
        ],
        "anchor_reference_frame_index": ANCHOR_REFERENCE_FRAME,
        "repeat_reference_anchor_px": list(REPEAT_REFERENCE_ANCHOR_PX),
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
        "tracking_steps": [asdict(step) for step in trajectory.steps],
    }
    (temporary / "generation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_image_index(temporary / "image_ids.csv")
    shutil.copy2(SOURCE_SEQUENCE / "gps.csv", temporary / "gps.csv")
    shutil.copy2(
        BLENDER_OUTPUT / "render_metadata.json",
        temporary / "blender_render_metadata.json",
    )
    temporary.rename(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
