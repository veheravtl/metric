"""Проверки наблюдаемого индикатора периодической текстуры G13-R."""

import numpy as np
import pytest

from aerial_mapper.texture_ambiguity import (
    TextureAmbiguityThresholds,
    _remote_nearest_descriptor_distances,
    assess_reference_texture_ambiguity,
)


def test_remote_search_ignores_identical_descriptor_in_local_neighbourhood() -> None:
    """Две ориентации одного угла не должны выглядеть далёким повтором."""

    descriptors = np.asarray([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0]], np.float32)
    points = np.asarray([[0.0, 0.0], [2.0, 0.0], [100.0, 0.0]], np.float32)

    distances = _remote_nearest_descriptor_distances(
        descriptors,
        points,
        remote_distance_px=20.0,
    )

    assert distances.tolist() == pytest.approx([5.0, 5.0, 5.0])


def test_remote_search_finds_repeated_descriptor_far_away() -> None:
    """Точная копия узора в другой части кадра должна дать нулевую дистанцию."""

    descriptors = np.asarray([[1.0, 2.0], [8.0, 9.0], [1.0, 2.0]], np.float32)
    points = np.asarray([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]], np.float32)

    distances = _remote_nearest_descriptor_distances(
        descriptors,
        points,
        remote_distance_px=20.0,
    )

    assert distances[0] == pytest.approx(0.0)
    assert distances[2] == pytest.approx(0.0)


def test_blank_reference_is_not_mislabeled_as_periodic() -> None:
    """Нехватку признаков обязан обработать основной gate, а не этот guard."""

    image = np.zeros((96, 128, 3), dtype=np.uint8)
    mask = np.full((96, 128), 255, dtype=np.uint8)

    result = assess_reference_texture_ambiguity(image, mask)

    assert result.keypoint_count == 0
    assert result.rejected_as_ambiguous is False


def test_invalid_fraction_threshold_is_rejected() -> None:
    """Порог больше единицы не должен создавать заведомо выключенный guard."""

    with pytest.raises(ValueError, match="диапазоне 0..1"):
        TextureAmbiguityThresholds(maximum_remote_duplicate_fraction=1.1)
