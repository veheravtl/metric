"""Тесты контролируемых ошибок разметки Teach для G8."""

import numpy as np
import pytest

from aerial_mapper.teach_annotation import (
    add_contaminant_fraction,
    enclosed_non_ground,
    mask_agreement,
    offset_mask_boundary,
    retain_connected_fraction,
)


def _ground_with_hole() -> np.ndarray:
    """Создаёт землю с внутренним прямоугольником условной крыши."""

    mask = np.full((80, 100), 255, dtype=np.uint8)
    mask[25:55, 35:65] = 0
    return mask


def test_signed_boundary_offset_changes_mask_in_expected_direction() -> None:
    """Эрозия обязана терять землю, а дилатация — захватывать отверстие."""

    truth = _ground_with_hole()
    eroded = offset_mask_boundary(truth, offset_pixels=-3)
    dilated = offset_mask_boundary(truth, offset_pixels=3)

    assert np.count_nonzero(eroded) < np.count_nonzero(truth)
    assert np.count_nonzero(dilated) > np.count_nonzero(truth)
    assert mask_agreement(eroded, truth).precision == 1.0
    assert mask_agreement(dilated, truth).precision < 1.0


def test_retained_fraction_removes_ground_without_adding_false_positives() -> None:
    """Опыт полноты не должен одновременно загрязнять маску крышей."""

    truth = _ground_with_hole()
    retained = retain_connected_fraction(truth, retained_fraction=0.5)
    agreement = mask_agreement(retained, truth)

    assert agreement.precision == 1.0
    assert agreement.recall == pytest.approx(0.5, abs=0.01)


def test_enclosed_component_excludes_background_connected_to_border() -> None:
    """Внутренняя крыша отделяется от внешней неразмеченной полосы."""

    truth = _ground_with_hole()
    truth[:5, :] = 0
    enclosed = enclosed_non_ground(truth)

    assert np.all(enclosed[:5, :] == 0)
    assert np.all(enclosed[25:55, 35:65] == 255)


def test_contamination_reduces_purity_but_preserves_ground_recall() -> None:
    """Добавление крыши меняет только чистоту, а не полноту земли."""

    truth = _ground_with_hole()
    roof = enclosed_non_ground(truth)
    contaminated = add_contaminant_fraction(
        truth,
        roof,
        added_fraction=0.25,
    )
    agreement = mask_agreement(contaminated, truth)

    assert agreement.recall == 1.0
    assert agreement.precision < 1.0
    assert np.count_nonzero(contaminated & roof) == pytest.approx(
        np.count_nonzero(roof) * 0.25,
        abs=2,
    )


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
def test_invalid_fractions_are_rejected(fraction: float) -> None:
    """Ошибочный протокол не должен молча превращаться в другой опыт."""

    truth = _ground_with_hole()
    with pytest.raises(ValueError):
        retain_connected_fraction(truth, retained_fraction=fraction)
