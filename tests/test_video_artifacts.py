"""Проверки воспроизводимых JPEG- и OSD-приближений G14-C."""

import numpy as np
import pytest

from aerial_mapper.video_artifacts import (
    apply_osd_occlusion,
    jpeg_round_trip,
)


def make_texture() -> np.ndarray:
    """Создаёт цветную нерегулярную текстуру без внешнего файла."""

    generator = np.random.default_rng(17)
    return generator.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)


def test_jpeg_round_trip_is_reproducible_and_lossy() -> None:
    """Настоящий JPEG должен повторяться и менять высокочастотный RGB."""

    image = make_texture()
    first = jpeg_round_trip(image, quality=30)
    repeated = jpeg_round_trip(image, quality=30)

    np.testing.assert_array_equal(first.image_rgb, repeated.image_rgb)
    assert first.encoded_bytes == repeated.encoded_bytes
    assert first.encoded_bytes > 0
    assert not np.array_equal(first.image_rgb, image)


def test_lower_jpeg_quality_reduces_buffer_for_test_texture() -> None:
    """На фиксированной шумовой текстуре низкое quality должно сжимать сильнее."""

    image = make_texture()
    assert jpeg_round_trip(image, quality=10).encoded_bytes < jpeg_round_trip(
        image, quality=90
    ).encoded_bytes


def test_osd_has_exact_reproducible_occluded_fraction() -> None:
    """Площадь OSD — управляемый фактор, а не побочный эффект рисунка."""

    image = make_texture()
    first = apply_osd_occlusion(image, fraction=0.1, random_seed=4)
    repeated = apply_osd_occlusion(image, fraction=0.1, random_seed=4)
    different = apply_osd_occlusion(image, fraction=0.1, random_seed=5)

    assert first.actual_fraction == pytest.approx(0.1, abs=1 / (120 * 160))
    np.testing.assert_array_equal(first.image_rgb, repeated.image_rgb)
    assert not np.array_equal(first.image_rgb, different.image_rgb)
    assert np.count_nonzero(first.occluded_mask) == round(0.1 * 120 * 160)


@pytest.mark.parametrize("quality", [-1, 101, 20.5, True])
def test_invalid_jpeg_quality_is_rejected(quality: object) -> None:
    """Некорректный quality не должен молча округляться."""

    with pytest.raises(ValueError):
        jpeg_round_trip(make_texture(), quality=quality)  # type: ignore[arg-type]


@pytest.mark.parametrize("fraction", [-0.1, 0.51])
def test_invalid_osd_fraction_is_rejected(fraction: float) -> None:
    """Слишком большая доля уже не соответствует узкому OSD-опыту."""

    with pytest.raises(ValueError):
        apply_osd_occlusion(make_texture(), fraction=fraction, random_seed=1)
