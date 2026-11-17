"""Проверки автоматической цепочки Repeat → Teach → карта → метры."""

from pathlib import Path

import numpy as np
import pytest

from aerial_mapper.alignment_session import (
    AlignmentSession,
    FrameObservation,
)
from aerial_mapper.gui.flight_sequence import MarsLvigFlight
from aerial_mapper.manual_alignment import estimate_manual_alignment
from aerial_mapper.repeat_localization import (
    CLUSTER_COUNT,
    MappedTeachFrame,
    RepeatLocalizationFailure,
    _load_visual_search_index,
    _save_visual_search_index,
    compose_repeat_to_map_homography,
    limit_mapped_teach_frames,
    mapped_teach_frames,
    measure_scaled_map_segment,
)


def test_homographies_are_composed_in_repeat_teach_map_order() -> None:
    """Точка должна сначала попасть в Teach, а затем в систему карты."""

    repeat_to_teach = np.asarray(
        [[2.0, 0.0, 10.0], [0.0, 2.0, 20.0], [0.0, 0.0, 1.0]]
    )
    teach_to_map = np.asarray(
        [[1.0, 0.0, 100.0], [0.0, 1.0, 200.0], [0.0, 0.0, 1.0]]
    )
    composed = compose_repeat_to_map_homography(
        repeat_to_teach, teach_to_map
    )

    point = np.asarray([[[3.0, 4.0]]], dtype=np.float64)
    import cv2

    mapped = cv2.perspectiveTransform(point, composed).reshape(2)
    assert mapped == pytest.approx([116.0, 228.0])


def test_scaled_map_segment_uses_map_pixels_after_perspective() -> None:
    """Треугольник 3-4-5 на карте при 0,25 м/px должен дать 1,25 м."""

    measurement = measure_scaled_map_segment(
        [(10.0, 20.0), (13.0, 24.0)],
        np.eye(3),
        map_width_pixels=100,
        map_height_pixels=100,
        map_scale_meters_per_pixel=0.25,
    )

    assert measurement.map_points_px == pytest.approx(
        np.asarray([[10.0, 20.0], [13.0, 24.0]])
    )
    assert measurement.length_meters == pytest.approx(1.25)


def test_scaled_map_segment_rejects_extrapolation_outside_map() -> None:
    """Правдоподобное число за границей калиброванной карты запрещено."""

    with pytest.raises(RepeatLocalizationFailure, match="за границы карты"):
        measure_scaled_map_segment(
            [(10.0, 20.0), (110.0, 20.0)],
            np.eye(3),
            map_width_pixels=100,
            map_height_pixels=100,
            map_scale_meters_per_pixel=0.25,
        )


def test_search_database_contains_only_spaced_frames_with_map_alignment() -> None:
    """Неразмеченный Teach нельзя выбрать, даже если он визуально похож."""

    session = AlignmentSession()
    frame_points = np.asarray(
        [[0.0, 0.0], [99.0, 0.0], [99.0, 79.0], [0.0, 79.0]]
    )
    alignment = estimate_manual_alignment(
        frame_points,
        frame_points,
        frame_size_pixels=(100, 80),
        map_size_pixels=(100, 80),
    )
    observations = []
    for point in frame_points:
        landmark_id = session.add_landmark((float(point[0]), float(point[1])))
        observations.append(
            FrameObservation(
                landmark_id,
                (float(point[0]), float(point[1])),
            )
        )
    session.remember_frame(
        1,
        observations,
        alignment=alignment,
        status="keyframe",
    )

    flight = MarsLvigFlight(
        directory=Path("/tmp/teach"),
        image_paths=(
            Path("/tmp/teach/10.jpg"),
            Path("/tmp/teach/20.jpg"),
            Path("/tmp/teach/30.jpg"),
        ),
        image_ids=np.asarray([10, 20, 30], dtype=np.int64),
        timestamps_nanoseconds=np.asarray([1, 2, 3], dtype=np.int64),
        latitudes_degrees=np.asarray([1.0, 1.0, 1.0]),
        longitudes_degrees=np.asarray([1.0, 1.0, 1.0]),
        altitudes_ellipsoid_meters=np.asarray([1.0, 1.0, 1.0]),
        nominal_fps=10.0,
    )

    records = mapped_teach_frames(
        session,
        flight,
        spaced_image_ids=[10, 20, 30],
    )

    assert len(records) == 1
    assert records[0].frame_index == 1
    assert records[0].image_id == 20
    assert records[0].database_index == 0


def test_visual_index_limit_preserves_route_ends_and_renumbers_rows() -> None:
    """Разрежение не должно ломать связь строки поиска с Teach-кадром."""

    records = tuple(
        MappedTeachFrame(
            database_index=index,
            frame_index=index,
            image_id=100 + index,
            image_path=Path(f"/tmp/teach/{index}.jpg"),
            homography_teach_to_map=np.eye(3, dtype=np.float64),
        )
        for index in range(10)
    )

    selected = limit_mapped_teach_frames(records, maximum_count=3)

    assert [record.image_id for record in selected] == [100, 104, 109]
    assert [record.database_index for record in selected] == [0, 1, 2]


def test_visual_index_limit_rejects_non_positive_size() -> None:
    """Нулевой лимит является ошибкой конфигурации, а не пустой базой."""

    with pytest.raises(ValueError, match="положительным"):
        limit_mapped_teach_frames((), maximum_count=0)


def test_visual_search_index_round_trip_preserves_candidate_order(
    tmp_path: Path,
) -> None:
    """Повторный запуск должен читать готовую базу без пересборки словаря."""

    cache_path = tmp_path / "visual-index.npz"
    image_ids = np.asarray([20, 40], dtype=np.int64)
    vocabulary = np.arange(
        CLUSTER_COUNT * 3, dtype=np.float32
    ).reshape(CLUSTER_COUNT, 3)
    database = np.arange(
        image_ids.size * CLUSTER_COUNT * 3, dtype=np.float32
    ).reshape(image_ids.size, CLUSTER_COUNT * 3)

    _save_visual_search_index(
        cache_path,
        image_ids=image_ids,
        vocabulary=vocabulary,
        database_descriptors=database,
    )
    loaded = _load_visual_search_index(
        cache_path,
        expected_image_ids=image_ids,
    )

    assert loaded is not None
    loaded_vocabulary, loaded_database = loaded
    assert loaded_vocabulary == pytest.approx(vocabulary)
    assert loaded_database == pytest.approx(database)


def test_visual_search_index_rejects_different_candidate_order(
    tmp_path: Path,
) -> None:
    """Переставленные строки нельзя связывать с прежними Teach-кадрами."""

    cache_path = tmp_path / "visual-index.npz"
    _save_visual_search_index(
        cache_path,
        image_ids=np.asarray([20, 40], dtype=np.int64),
        vocabulary=np.ones((CLUSTER_COUNT, 3), dtype=np.float32),
        database_descriptors=np.ones((2, CLUSTER_COUNT * 3), dtype=np.float32),
    )

    with pytest.raises(RepeatLocalizationFailure, match="Некорректный кэш"):
        _load_visual_search_index(
            cache_path,
            expected_image_ids=np.asarray([40, 20], dtype=np.int64),
        )
