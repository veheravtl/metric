#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Прогнать прозрачный детектор события по каталогу извлечённых кадров."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import cv2
import numpy as np

from aerial_mapper.visual_event_detection import VisualEventDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("flight_directory", type=Path)
    arguments = parser.parse_args()
    images = sorted((arguments.flight_directory / "images").glob("*.jpg"))
    if not images:
        raise SystemExit("В каталоге images нет JPEG-кадров")

    detector = VisualEventDetector()
    processing_times: list[float] = []
    candidate_frames: list[int] = []
    invalid_frames: list[int] = []
    for frame_index, path in enumerate(images):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"Не читается кадр: {path}")
        step = detector.advance(frame, frame_index=frame_index)
        processing_times.append(step.processing_time_ms)
        if step.candidate is not None:
            candidate_frames.append(frame_index)
            x, y = step.candidate.center_px
            print(
                f"candidate frame={frame_index} center=({x:.1f}, {y:.1f}) "
                f"area={step.candidate.area_fraction:.5f} "
                f"residual={step.candidate.mean_residual:.1f} "
                f"threshold={step.candidate.threshold:.1f}"
            )
        if step.status == "invalid_geometry":
            invalid_frames.append(frame_index)

    event = detector.confirmed_event
    p95 = float(np.percentile(processing_times, 95))
    print(f"frames={len(images)} candidates={candidate_frames}")
    print(f"invalid_geometry={invalid_frames}")
    print(
        f"processing_ms median={statistics.median(processing_times):.2f} "
        f"p95={p95:.2f} max={max(processing_times):.2f}"
    )
    if event is None:
        print("confirmed_event=None")
    else:
        print(
            f"confirmed onset={event.onset_frame_index} "
            f"confirmation={event.confirmation_frame_index} "
            f"point=({event.point_px[0]:.1f}, {event.point_px[1]:.1f})"
        )


if __name__ == "__main__":
    main()
