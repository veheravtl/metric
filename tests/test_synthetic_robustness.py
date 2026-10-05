"""Тесты контролируемых искажений для синтетических G4--G6."""

import json
from pathlib import Path

import numpy as np
import pytest

from aerial_mapper.synthetic_robustness import (
    ImageDegradation,
    apply_image_degradation,
    perturb_click_points,
)


def make_image() -> np.ndarray:
    """Создаёт малый структурный RGB-кадр без внешних данных."""

    y, x = np.indices((48, 64), dtype=np.uint8)
    return np.stack((x * 3, y * 5, (x + y) * 2), axis=2)


def test_identity_degradation_preserves_every_pixel() -> None:
    """Нулевая точка sweep не должна сама вносить отличие от Blender-рендера."""

    image = make_image()

    result = apply_image_degradation(
        image,
        ImageDegradation(),
        random_seed=17,
    )

    assert np.array_equal(result, image)


def test_noise_is_reproducible_and_seed_sensitive() -> None:
    """Фиксированный seed повторяет шум, другой seed действительно меняет опыт."""

    image = make_image()
    degradation = ImageDegradation(noise_standard_deviation=20.0)

    first = apply_image_degradation(image, degradation, random_seed=11)
    repeated = apply_image_degradation(image, degradation, random_seed=11)
    different = apply_image_degradation(image, degradation, random_seed=12)

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, different)


def test_half_frame_shadow_reduces_about_half_the_pixels() -> None:
    """Параметр доли тени должен управлять площадью, а не только яркостью."""

    image = np.full((60, 80, 3), 160, dtype=np.uint8)
    result = apply_image_degradation(
        image,
        ImageDegradation(shadow_fraction=0.5, shadow_transmission=0.1),
        random_seed=1,
    )

    dark_fraction = np.mean(result[..., 0] < 100)
    assert dark_fraction == pytest.approx(0.5, abs=0.02)


def test_resolution_loss_keeps_coordinate_grid_size() -> None:
    """После эмуляции малого сенсора точки остаются в системе исходного кадра."""

    image = make_image()

    result = apply_image_degradation(
        image,
        ImageDegradation(resolution_scale=0.25),
        random_seed=1,
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert not np.array_equal(result, image)


def test_click_perturbation_is_reproducible_and_zero_is_exact() -> None:
    """Ошибка клика отделена от alignment и имеет собственный seed."""

    points = np.asarray([[10.0, 20.0], [30.0, 40.0]])

    exact = perturb_click_points(
        points,
        standard_deviation_px=0.0,
        random_seed=7,
    )
    noisy = perturb_click_points(
        points,
        standard_deviation_px=2.0,
        random_seed=7,
    )
    repeated = perturb_click_points(
        points,
        standard_deviation_px=2.0,
        random_seed=7,
    )

    assert np.array_equal(exact, points)
    assert np.array_equal(noisy, repeated)
    assert not np.array_equal(noisy, points)


@pytest.mark.parametrize(
    "degradation",
    [
        ImageDegradation(shadow_fraction=1.1),
        ImageDegradation(blur_sigma_px=-1.0),
        ImageDegradation(noise_standard_deviation=-1.0),
        ImageDegradation(resolution_scale=0.01),
    ],
)
def test_invalid_degradation_is_rejected(degradation: ImageDegradation) -> None:
    """Ошибочный протокол не должен молча превращаться в другой эксперимент."""

    with pytest.raises(ValueError):
        degradation.validate()


def test_g3_g6_protocol_keeps_calibration_and_held_out_sections_separate() -> None:
    """Отложенный seed и cases должны быть зафиксированы в репозитории заранее."""

    path = Path("experiments/configs/synthetic_3d_g3_g6_robustness.json")
    protocol = json.loads(path.read_text(encoding="utf-8"))

    widths = protocol["g3"]["object_widths_m"]
    assert widths == sorted(set(widths))
    assert widths[0] == 0.0
    assert widths[-1] == 30.0
    assert len(protocol["g5"]["cases"]) == 12
    assert protocol["g6"]["held_out_seed"] != protocol["seed"]
    assert protocol["g6"]["case_count"] == 20
    assert protocol["acceptance_thresholds"]["maximum_point_position_error_m"] == 0.2


def test_g8_protocol_separates_operational_cases_from_stress_cases() -> None:
    """Рабочая область и тяжёлые искажения G8 фиксируются до итогового прогона."""

    path = Path("experiments/configs/synthetic_3d_g8_annotation_robustness.json")
    protocol = json.loads(path.read_text(encoding="utf-8"))

    cases = protocol["mask_cases"]
    assert len(cases) == 12
    assert any(case["operational"] for case in cases)
    assert any(not case["operational"] for case in cases)
    assert {case["kind"] for case in cases} == {
        "boundary",
        "completeness",
        "contamination",
    }
    noise = protocol["calibration_pixel_noise"]
    assert noise["standard_deviations_px"] == sorted(
        set(noise["standard_deviations_px"])
    )
    assert noise["maximum_operational_standard_deviation_px"] == 1.0
