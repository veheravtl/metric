"""Проверки прозрачных причин отбраковки видеокадров."""

import cv2
import numpy as np

from aerial_mapper.frame_quality import assess_frame_image_quality


def _detailed_frame() -> np.ndarray:
    """Создать детальный кадр, для которого потерю резкости можно измерить."""

    random_generator = np.random.default_rng(20261107)
    gray = random_generator.integers(0, 256, size=(240, 320), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_severe_relative_blur_is_rejected_with_explanation() -> None:
    """Резкий провал деталей между соседями нельзя выдавать за хороший кадр."""

    previous = _detailed_frame()
    current = cv2.GaussianBlur(previous, (0, 0), 12.0)

    quality = assess_frame_image_quality(previous, current)

    assert quality.sharpness_ratio_to_previous < 0.05
    assert quality.is_rejected
    assert "резкость упала более чем в 20 раз" in quality.rejection_reasons


def test_moderate_blur_warns_without_rejecting_frame() -> None:
    """Пограничный кадр нужно показать оператору, но не отбраковывать молча."""

    previous = _detailed_frame()
    current = cv2.GaussianBlur(previous, (0, 0), 0.8)

    quality = assess_frame_image_quality(previous, current)

    assert 0.05 < quality.sharpness_ratio_to_previous < 0.35
    assert not quality.is_rejected
    assert quality.warning_reasons == ("резкость заметно ниже соседнего кадра",)


def test_nearly_empty_exposure_is_rejected_but_normal_frame_is_not() -> None:
    """Чёрный и белый провалы должны отличаться от обычного изображения."""

    previous = _detailed_frame()
    dark = assess_frame_image_quality(previous, np.zeros_like(previous))
    bright = assess_frame_image_quality(previous, np.full_like(previous, 255))
    normal = assess_frame_image_quality(previous, previous.copy())

    assert "кадр почти полностью чёрный" in dark.rejection_reasons
    assert "кадр почти полностью пересвечен" in bright.rejection_reasons
    assert not normal.is_rejected
    assert normal.sharpness_ratio_to_previous == 1.0
