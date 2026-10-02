"""Тесты геометрии пространственной сетки исследовательского эксперимента."""

import numpy as np
import pytest

from aerial_mapper.synthetic import calculate_valid_axis_centers


def test_axis_centers_span_exact_valid_travel_without_crossing_image() -> None:
    """Крайние положения должны касаться первого и последнего пикселя."""

    centers = calculate_valid_axis_centers(
        image_size_pixels=1000,
        footprint_size_pixels=400.0,
        travel_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
    )

    # Половина следа равна 200 px. Поэтому первый центр 200 даёт левую
    # границу 0, а последний центр 799 — правую границу 999.
    assert centers[0] == pytest.approx(200.0)
    assert centers[-1] == pytest.approx(799.0)
    assert np.diff(centers) == pytest.approx(np.full(4, 149.75))


def test_axis_centers_reject_footprint_larger_than_reference() -> None:
    """Невместимый след нужно отклонить до запуска дорогого SIFT."""

    with pytest.raises(ValueError, match="След больше эталона"):
        calculate_valid_axis_centers(
            image_size_pixels=1000,
            footprint_size_pixels=1001.0,
            travel_fractions=(0.0, 1.0),
        )
