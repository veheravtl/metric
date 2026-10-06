"""Проверки классов синтетической текстуры земли G13."""

import numpy as np
import pytest

from aerial_mapper.ground_texture import generate_ground_texture


@pytest.mark.parametrize(
    "texture_class",
    ["rich_irregular", "low_detail", "periodic_rows"],
)
def test_texture_classes_return_bounded_rgb(texture_class: str) -> None:
    """Каждый класс должен подходить Blender Image без скрытой нормализации."""

    image = generate_ground_texture(
        {
            "width_pixels": 192,
            "height_pixels": 128,
            "class": texture_class,
        },
        seed=17,
    )

    assert image.shape == (128, 192, 3)
    assert image.dtype == np.float32
    assert np.isfinite(image).all()
    assert float(np.min(image)) >= 0.0
    assert float(np.max(image)) <= 1.0


def test_default_class_preserves_explicit_rich_texture() -> None:
    """Старые конфигурации без class не должны изменить свои пиксели."""

    base = {"width_pixels": 192, "height_pixels": 128}
    implicit = generate_ground_texture(base, seed=41)
    explicit = generate_ground_texture(
        {**base, "class": "rich_irregular"},
        seed=41,
    )

    assert np.array_equal(implicit, explicit)


def test_texture_seed_is_deterministic_but_effective() -> None:
    """Одинаковый seed повторяется точно, а другой действительно меняет сцену."""

    specification = {
        "width_pixels": 192,
        "height_pixels": 128,
        "class": "low_detail",
    }
    first = generate_ground_texture(specification, seed=8)
    repeated = generate_ground_texture(specification, seed=8)
    changed = generate_ground_texture(specification, seed=9)

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, changed)


def test_low_detail_has_less_local_contrast_than_rich_texture() -> None:
    """Название low_detail должно подтверждаться измеримым контрастом."""

    base = {"width_pixels": 320, "height_pixels": 240}
    rich = generate_ground_texture(
        {**base, "class": "rich_irregular"},
        seed=23,
    )
    low = generate_ground_texture(
        {**base, "class": "low_detail"},
        seed=23,
    )

    rich_gradient = np.abs(np.diff(rich, axis=1)).mean()
    low_gradient = np.abs(np.diff(low, axis=1)).mean()
    assert low_gradient < rich_gradient * 0.2


def test_periodic_rows_repeat_exactly_when_angle_is_zero() -> None:
    """Контроль доказывает повторяемость, а не только полосатый внешний вид."""

    period = 48
    image = generate_ground_texture(
        {
            "width_pixels": 240,
            "height_pixels": 144,
            "class": "periodic_rows",
            "row_angle_degrees": 0,
            "along_period_pixels": period,
        },
        seed=31,
    )

    assert np.array_equal(image[:, :-period], image[:, period:])


def test_unknown_texture_class_is_rejected() -> None:
    """Опечатка в frozen config не должна молча стать богатой текстурой."""

    with pytest.raises(ValueError, match="Неизвестный класс"):
        generate_ground_texture(
            {
                "width_pixels": 192,
                "height_pixels": 128,
                "class": "snow_magic",
            },
            seed=1,
        )
