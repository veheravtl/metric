"""Проверки измерения промаха относительно цели."""

import pytest

from aerial_mapper.relative_measurement import (
    displacement_error,
    target_centered_displacement,
)


def test_target_centered_displacement_matches_operator_message() -> None:
    """Смещение 2 на 8 метров должно сохраниться без абсолютных координат."""

    displacement = target_centered_displacement(
        target_xy_m=[120.0, -35.0],
        impact_xy_m=[122.0, -27.0],
    )
    assert displacement.delta_x_m == pytest.approx(2.0)
    assert displacement.delta_y_m == pytest.approx(8.0)
    assert displacement.distance_m == pytest.approx(68.0**0.5)


def test_common_absolute_bias_cancels_from_relative_displacement() -> None:
    """Одинаковый сдвиг цели и попадания не должен портить вектор промаха."""

    truth = target_centered_displacement([10.0, 20.0], [14.0, 28.0])
    estimated = target_centered_displacement([110.0, -30.0], [114.0, -22.0])
    error = displacement_error(estimated, truth)
    assert error.vector_error_m == pytest.approx(0.0)
    assert error.absolute_distance_error_m == pytest.approx(0.0)
    assert error.direction_error_degrees == pytest.approx(0.0)


def test_differential_error_reports_components_length_and_direction() -> None:
    """Разная ошибка концов должна быть видна во всех метриках промаха."""

    truth = target_centered_displacement([0.0, 0.0], [0.0, 10.0])
    estimated = target_centered_displacement([0.0, 0.0], [1.0, 9.0])
    error = displacement_error(estimated, truth)
    assert error.error_x_m == pytest.approx(1.0)
    assert error.error_y_m == pytest.approx(-1.0)
    assert error.vector_error_m == pytest.approx(2.0**0.5)
    assert error.absolute_distance_error_m == pytest.approx(10.0 - 82.0**0.5)
    assert error.direction_error_degrees == pytest.approx(6.3401917459)
    assert error.x_sign_correct is None
    assert error.y_sign_correct is True


def test_exact_hit_has_no_defined_direction_or_component_signs() -> None:
    """Для попадания в цель направление и знаки не следует выдумывать."""

    exact = target_centered_displacement([5.0, 7.0], [5.0, 7.0])
    error = displacement_error(exact, exact)
    assert exact.distance_m == pytest.approx(0.0)
    assert error.direction_error_degrees is None
    assert error.x_sign_correct is None
    assert error.y_sign_correct is None


def test_zero_estimate_does_not_count_as_correct_positive_sign() -> None:
    """Нулевая оценка не должна притворяться положительным направлением."""

    truth = target_centered_displacement([0.0, 0.0], [2.0, -3.0])
    estimated = target_centered_displacement([0.0, 0.0], [0.0, 0.0])
    error = displacement_error(estimated, truth)

    assert error.x_sign_correct is False
    assert error.y_sign_correct is False
