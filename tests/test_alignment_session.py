"""Проверки неразрушающей памяти маяков и кадров."""

import numpy as np
import pytest

from aerial_mapper.alignment_session import (
    AlignmentSession,
    FrameObservation,
    MachineObservation,
)
from aerial_mapper.frame_quality import FrameImageQuality
from aerial_mapper.manual_alignment import estimate_manual_alignment


def _identity_alignment():
    """Построить корректную гомографию для тестового ключевого кадра."""

    points = np.asarray(
        [[5.0, 5.0], [90.0, 5.0], [90.0, 70.0], [5.0, 70.0]],
        dtype=np.float64,
    )
    return estimate_manual_alignment(
        points,
        points,
        frame_size_pixels=(100, 80),
        map_size_pixels=(100, 80),
    )


def test_frame_overwrite_does_not_destroy_other_frames_or_landmarks() -> None:
    """Ручная правка одного кадра не должна переписывать прошлый ключевой кадр."""

    session = AlignmentSession()
    map_points = ((5.0, 5.0), (90.0, 5.0), (90.0, 70.0), (5.0, 70.0))
    landmark_ids = [session.add_landmark(point) for point in map_points]
    observations = [
        FrameObservation(landmark_id, point)
        for landmark_id, point in zip(landmark_ids, map_points, strict=True)
    ]
    alignment = _identity_alignment()

    frame_zero = session.remember_frame(
        0,
        observations,
        alignment=alignment,
        status="keyframe",
    )
    tracked_frame = session.remember_frame(
        1,
        observations,
        alignment=alignment,
        status="tracked",
        suspect_landmark_ids={landmark_ids[2]},
    )
    corrected_draft = session.remember_frame(
        1,
        observations[:2],
        alignment=None,
        status="draft",
    )

    assert session.frame_indices == (0, 1)
    assert session.frame(0) is frame_zero
    assert session.frame(0).is_keyframe  # type: ignore[union-attr]
    assert session.frame(1) is corrected_draft
    assert tracked_frame.suspect_landmark_ids == frozenset({landmark_ids[2]})
    assert len(corrected_draft.observations) == 2
    assert session.landmark_count == 4
    assert session.landmark(landmark_ids[3]).map_point_px == map_points[3]


def test_session_rejects_invalid_observations() -> None:
    """Некорректная память должна отказать явно, а не портить историю."""

    session = AlignmentSession()
    landmark_id = session.add_landmark((10.0, 20.0))
    observation = FrameObservation(landmark_id, (12.0, 22.0))

    with pytest.raises(ValueError, match="дважды"):
        session.remember_frame(
            0,
            [observation, observation],
            alignment=None,
            status="draft",
        )
    with pytest.raises(KeyError, match="Неизвестный маяк"):
        session.remember_frame(
            0,
            [FrameObservation(999, (1.0, 2.0))],
            alignment=None,
            status="draft",
        )
    with pytest.raises(ValueError, match="должна содержать гомографию"):
        session.remember_frame(
            0,
            [observation],
            alignment=None,
            status="keyframe",
        )
    with pytest.raises(ValueError, match="должен содержать причину"):
        session.remember_frame(
            1,
            [observation],
            alignment=None,
            status="rejected",
        )


def test_operator_exclusion_is_distinct_from_repairable_rejection() -> None:
    """Мусорный кадр хранится пустым и не требует ручной коррекции геометрии."""

    session = AlignmentSession()
    keyframe = session.remember_frame(
        0,
        (),
        alignment=None,
        status="draft",
    )
    excluded = session.remember_frame(
        1,
        (),
        alignment=None,
        status="excluded",
        rejection_reasons=("оператор: камера вне рабочей области",),
    )

    assert session.frame(0) is keyframe
    assert excluded.is_excluded
    assert not excluded.is_rejected
    assert excluded.observations == ()


def test_rejected_frame_keeps_diagnostics_without_changing_keyframe() -> None:
    """Отказ хранится локально и не переписывает уже удачный прошлый кадр."""

    session = AlignmentSession()
    map_points = ((5.0, 5.0), (90.0, 5.0), (90.0, 70.0), (5.0, 70.0))
    landmark_ids = [session.add_landmark(point) for point in map_points]
    observations = tuple(
        FrameObservation(landmark_id, point)
        for landmark_id, point in zip(landmark_ids, map_points, strict=True)
    )
    keyframe = session.remember_frame(
        0,
        observations,
        alignment=_identity_alignment(),
        status="keyframe",
    )
    quality = FrameImageQuality(
        previous_laplacian_variance=120.0,
        current_laplacian_variance=2.0,
        sharpness_ratio_to_previous=1.0 / 60.0,
        mean_luminance=80.0,
        dark_pixel_fraction=0.1,
        bright_pixel_fraction=0.0,
        warning_reasons=(),
        rejection_reasons=("резкость упала более чем в 20 раз",),
    )
    machine_observation = MachineObservation(
        source_frame_index=0,
        source_frame_point_px=(40.0, 30.0),
        frame_point_px=(44.0, 33.0),
        map_point_px=(40.0, 30.0),
        forward_backward_error_px=0.15,
        used_for_alignment=False,
    )

    rejected = session.remember_frame(
        1,
        observations,
        alignment=None,
        status="rejected",
        machine_observations=(machine_observation,),
        image_quality=quality,
        rejection_reasons=quality.rejection_reasons,
    )

    assert session.frame(0) is keyframe
    assert keyframe.is_keyframe
    assert rejected.is_rejected
    assert rejected.machine_observations == (machine_observation,)
    assert rejected.image_quality is quality
    assert rejected.rejection_reasons == quality.rejection_reasons


def test_correction_invalidates_only_dependent_future_until_next_keyframe() -> None:
    """Правка прошлого не затрагивает прошлое и независимую следующую опору."""

    session = AlignmentSession()
    points = ((5.0, 5.0), (90.0, 5.0), (90.0, 70.0), (5.0, 70.0))
    landmark_ids = [session.add_landmark(point) for point in points]
    observations = tuple(
        FrameObservation(landmark_id, point)
        for landmark_id, point in zip(landmark_ids, points, strict=True)
    )
    alignment = _identity_alignment()
    session.remember_frame(0, observations, alignment=alignment, status="keyframe")
    session.remember_frame(1, observations, alignment=alignment, status="tracked")
    session.remember_frame(2, observations, alignment=alignment, status="tracked")
    session.remember_frame(
        3,
        (),
        alignment=None,
        status="excluded",
        rejection_reasons=("оператор исключил кадр",),
    )
    next_keyframe = session.remember_frame(
        4, observations, alignment=alignment, status="keyframe"
    )
    later_tracked = session.remember_frame(
        5, observations, alignment=alignment, status="tracked"
    )

    invalidated = session.invalidate_derived_frames_after(0)

    assert invalidated == (1, 2)
    assert session.frame(0).is_keyframe  # type: ignore[union-attr]
    assert session.frame(1).is_stale  # type: ignore[union-attr]
    assert session.frame(1).alignment is None  # type: ignore[union-attr]
    assert session.frame(2).is_stale  # type: ignore[union-attr]
    assert session.frame(3).is_excluded  # type: ignore[union-attr]
    assert session.frame(4) is next_keyframe
    assert session.frame(5) is later_tracked
