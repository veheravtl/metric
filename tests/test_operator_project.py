"""Проверки сохранения и восстановления операторского проекта."""

from pathlib import Path

import numpy as np

from aerial_mapper.alignment_session import (
    AlignmentSession,
    FrameObservation,
    MachineObservation,
)
from aerial_mapper.frame_quality import FrameImageQuality
from aerial_mapper.manual_alignment import estimate_manual_alignment
from aerial_mapper.operator_project import (
    load_operator_project,
    save_operator_project,
    source_matches_fingerprint,
)


def _filled_session() -> AlignmentSession:
    """Создать сессию со всеми типами сохраняемой покадровой диагностики."""

    session = AlignmentSession()
    frame_points = np.asarray(
        [[5.0, 5.0], [90.0, 5.0], [90.0, 70.0], [5.0, 70.0]],
        dtype=np.float64,
    )
    map_points = frame_points + np.asarray([10.0, 20.0])
    landmark_ids = [
        session.add_landmark(tuple(point)) for point in map_points
    ]
    alignment = estimate_manual_alignment(
        frame_points,
        map_points,
        frame_size_pixels=(100, 80),
        map_size_pixels=(160, 140),
    )
    quality = FrameImageQuality(
        previous_laplacian_variance=120.0,
        current_laplacian_variance=110.0,
        sharpness_ratio_to_previous=float("inf"),
        mean_luminance=100.0,
        dark_pixel_fraction=0.01,
        bright_pixel_fraction=0.02,
        warning_reasons=("пример предупреждения",),
        rejection_reasons=(),
    )
    machine = MachineObservation(
        source_frame_index=6,
        source_frame_point_px=(10.0, 10.0),
        frame_point_px=(11.0, 10.5),
        map_point_px=(20.0, 30.0),
        forward_backward_error_px=0.25,
        used_for_alignment=True,
    )
    session.remember_frame(
        7,
        tuple(
            FrameObservation(landmark_id, tuple(point))
            for landmark_id, point in zip(landmark_ids, frame_points, strict=True)
        ),
        alignment=alignment,
        status="keyframe",
        machine_observations=(machine,),
        image_quality=quality,
    )
    return session


def test_project_round_trip_preserves_session_and_workspace(tmp_path: Path) -> None:
    """Стабильные ID, матрица и позиция должны пережить закрытие приложения."""

    video_path = tmp_path / "flight.avi"
    map_path = tmp_path / "map.png"
    video_path.write_bytes(b"synthetic-video-payload")
    map_path.write_bytes(b"synthetic-map-payload")
    output = tmp_path / "work.aerial-project.json"

    saved = save_operator_project(
        output,
        video_source_path=video_path,
        video_source_type="video",
        map_source_path=map_path,
        current_frame_index=7,
        overlay_visible=False,
        session=_filled_session(),
    )
    loaded = load_operator_project(saved)

    assert loaded.current_frame_index == 7
    assert not loaded.overlay_visible
    assert loaded.video_source_path == video_path.resolve()
    assert loaded.map_source_path == map_path.resolve()
    assert loaded.session.landmark_ids == (0, 1, 2, 3)
    assert loaded.session.next_landmark_id == 4

    frame = loaded.session.frame(7)
    assert frame is not None
    assert frame.status == "keyframe"
    assert frame.alignment is not None
    np.testing.assert_allclose(
        frame.alignment.homography_frame_to_map,
        np.asarray([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]]),
        atol=1e-8,
    )
    assert frame.image_quality is not None
    assert np.isinf(frame.image_quality.sharpness_ratio_to_previous)
    assert frame.machine_observations[0].source_frame_index == 6


def test_fingerprint_detects_replaced_video(tmp_path: Path) -> None:
    """Проект не должен молча принять другой материал по прежнему пути."""

    video_path = tmp_path / "flight.avi"
    map_path = tmp_path / "map.png"
    video_path.write_bytes(b"original-video")
    map_path.write_bytes(b"map")
    output = save_operator_project(
        tmp_path / "work.json",
        video_source_path=video_path,
        video_source_type="video",
        map_source_path=map_path,
        current_frame_index=0,
        overlay_visible=True,
        session=AlignmentSession(),
    )
    project = load_operator_project(output)

    matches, reason = source_matches_fingerprint(
        video_path, project.video_fingerprint
    )
    assert matches
    assert reason == ""

    video_path.write_bytes(b"different-video")
    matches, reason = source_matches_fingerprint(
        video_path, project.video_fingerprint
    )
    assert not matches
    assert "размер" in reason or "контрольный" in reason
