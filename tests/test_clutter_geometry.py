"""Проверки детерминированного размещения трёхмерного мусора G10."""

import numpy as np
import pytest

from aerial_mapper.clutter_geometry import generate_clutter_instances


def _specification(count: int) -> dict:
    return {
        "count": count,
        "edge_margin_m": 2.0,
        "minimum_gap_m": 0.25,
        "maximum_attempts": 20_000,
        "kinds": {
            "rock": {
                "weight": 0.4,
                "height_m": [0.3, 0.9],
                "radius_m": [0.3, 0.8],
                "xy_aspect_ratio": [0.7, 1.3],
            },
            "stump": {
                "weight": 0.3,
                "height_m": [0.5, 1.6],
                "radius_m": [0.18, 0.45],
                "xy_aspect_ratio": [0.9, 1.1],
            },
            "shrub": {
                "weight": 0.3,
                "height_m": [1.0, 2.8],
                "radius_m": [0.45, 1.1],
                "xy_aspect_ratio": [0.75, 1.25],
            },
        },
    }


def test_zero_count_produces_clean_control_scene() -> None:
    """Нулевая ступень должна быть настоящим контролем без скрытых объектов."""

    assert (
        generate_clutter_instances(
            width_m=40.0,
            depth_m=30.0,
            specification=_specification(0),
            seed=17,
        )
        == ()
    )


def test_same_seed_reproduces_every_instance() -> None:
    """Seed обязан фиксировать типы, размеры, координаты и цвета."""

    first = generate_clutter_instances(
        width_m=40.0,
        depth_m=30.0,
        specification=_specification(24),
        seed=17,
    )
    second = generate_clutter_instances(
        width_m=40.0,
        depth_m=30.0,
        specification=_specification(24),
        seed=17,
    )

    assert first == second
    assert {item.kind for item in first} == {"rock", "stump", "shrub"}


def test_instances_stay_inside_bounds_and_do_not_overlap() -> None:
    """Даже плотный случай сохраняет отступ от края и зазор оснований."""

    gap = 0.25
    instances = generate_clutter_instances(
        width_m=40.0,
        depth_m=30.0,
        specification=_specification(80),
        seed=29,
    )

    for index, item in enumerate(instances):
        radius = item.footprint_radius_m
        assert abs(item.x_m) + radius <= 18.0 + 1e-12
        assert abs(item.y_m) + radius <= 13.0 + 1e-12
        assert item.height_m > 0.0
        for other in instances[index + 1 :]:
            distance = np.hypot(item.x_m - other.x_m, item.y_m - other.y_m)
            assert distance + 1e-12 >= radius + other.footprint_radius_m + gap


def test_unknown_kind_is_rejected_before_blender() -> None:
    """Опечатка в типе не должна превращаться в молчаливо иной объект."""

    specification = _specification(1)
    specification["kinds"]["tree"] = specification["kinds"].pop("rock")
    with pytest.raises(ValueError, match="Неизвестные типы"):
        generate_clutter_instances(
            width_m=40.0,
            depth_m=30.0,
            specification=specification,
            seed=3,
        )
