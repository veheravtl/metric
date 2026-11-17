"""Тесты геометрии и маски синтетического события."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from aerial_mapper.synthetic_event import (
    OpticalFlowAnchorConfig,
    composite_layer_at_anchor,
    extract_changed_foreground_alpha,
    track_ground_anchor,
)


def test_ground_anchor_follows_known_image_translation(tmp_path: Path) -> None:
    """Наземный якорь должен повторить известный сдвиг текстурированного фона."""

    random_generator = np.random.default_rng(20261113)
    base = random_generator.integers(0, 256, size=(420, 560), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 1.2)
    paths: list[Path] = []
    for frame_index in range(5):
        transform = np.asarray(
            [[1.0, 0.0, 4.0 * frame_index], [0.0, 1.0, -2.0 * frame_index]],
            dtype=np.float32,
        )
        gray = cv2.warpAffine(base, transform, (560, 420))
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        path = tmp_path / f"{frame_index:06d}.png"
        assert cv2.imwrite(str(path), bgr)
        paths.append(path)

    trajectory = track_ground_anchor(
        paths,
        first_frame_index=25,
        initial_anchor_px=(250.0, 210.0),
        config=OpticalFlowAnchorConfig(
            processing_scale=1.0,
            minimum_inlier_count=50,
        ),
    )

    assert trajectory.frame_indices.tolist() == [25, 26, 27, 28, 29]
    assert trajectory.points_px[-1] == pytest.approx([266.0, 202.0], abs=0.2)
    assert min(step.inlier_ratio for step in trajectory.steps) > 0.95


def test_foreground_alpha_keeps_change_and_hides_crop_edges() -> None:
    """Маска должна выделить облако, но не проявить прямоугольник crop."""

    background = np.full((240, 320, 3), (50, 110, 60), dtype=np.uint8)
    current = background.copy()
    cv2.circle(current, (160, 125), 55, (180, 190, 210), thickness=-1)

    alpha = extract_changed_foreground_alpha(
        current,
        background,
        minimum_component_area_px=100,
    )

    assert alpha.dtype == np.float32
    assert alpha[125, 160] > 0.9
    assert alpha[0, 0] == pytest.approx(0.0)
    assert np.max(alpha[:, 0]) == pytest.approx(0.0)
    assert np.count_nonzero(alpha > 0.1) > 2_000


def test_composite_clips_layer_that_crosses_frame_boundary() -> None:
    """Частично видимый слой должен корректно обрезаться, а не менять размер."""

    background = np.zeros((40, 50, 3), dtype=np.uint8)
    layer = np.full((20, 20, 3), (10, 40, 220), dtype=np.uint8)
    alpha = np.ones((20, 20), dtype=np.float32)

    result = composite_layer_at_anchor(
        background,
        layer,
        alpha,
        layer_anchor_px=(10.0, 10.0),
        target_anchor_px=(5.0, 5.0),
    )

    assert result.shape == background.shape
    assert result[0, 0].tolist() == [10, 40, 220]
    assert result[14, 14].tolist() == [10, 40, 220]
    assert result[15, 15].tolist() == [0, 0, 0]
