"""Тесты проектной части минимального 3D-конвейера.

Blender не запускается в unit-тестах: это было бы медленно и сделало бы весь
набор зависимым от внешнего бинарного файла. Сквозной запуск оформлен отдельным
воспроизводимым smoke-экспериментом.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from aerial_mapper.synthetic_3d import (
    analyze_render,
    check_render,
    load_scene_description,
)

DESCRIPTION_PATH = Path("experiments/configs/synthetic_3d_g0_smoke.json")


def test_g0_description_has_all_camera_roles() -> None:
    """Зафиксированная G0-сцена должна покрывать весь Teach/Repeat-путь."""

    description = load_scene_description(DESCRIPTION_PATH)

    assert description["scenario_id"] == "g0_smoke"
    assert {camera["role"] for camera in description["cameras"]} == {
        "reference",
        "teach",
        "repeat",
    }



def test_g1_description_allows_exactly_two_required_poses() -> None:
    """Метрический G1 не должен требовать лишнюю Teach-камеру."""

    description = load_scene_description(
        Path("experiments/configs/synthetic_3d_g1_metric.json")
    )

    assert description["scenario_id"] == "g1_planar_metric_recovery"
    assert {camera["role"] for camera in description["cameras"]} == {
        "reference",
        "repeat",
    }
    assert len(description["metric_controls"]) == 4

def test_description_rejects_pixel_units(tmp_path: Path) -> None:
    """Мировые величины нельзя незаметно принять за пиксели вместо метров."""

    description = json.loads(DESCRIPTION_PATH.read_text(encoding="utf-8"))
    description["world"]["units"] = "pixels"
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(description), encoding="utf-8")

    with pytest.raises(ValueError, match="'metres'"):
        load_scene_description(invalid_path)


def test_structured_image_passes_smoke_metrics(tmp_path: Path) -> None:
    """Контрастная геометрическая мишень должна проходить грубую проверку."""

    image = np.full((120, 160, 3), 180, dtype=np.uint8)
    cv2.rectangle(image, (15, 15), (75, 105), (20, 60, 210), thickness=-1)
    cv2.line(image, (0, 60), (159, 60), (255, 255, 255), thickness=4)
    path = tmp_path / "structured.png"
    assert cv2.imwrite(str(path), image)

    result = check_render(path, expected_width=160, expected_height=120)

    assert result.passed
    assert result.metrics.edge_density > 0.005


def test_uniform_image_is_rejected(tmp_path: Path) -> None:
    """Однотонный рендер не должен считаться успешной генерацией сцены."""

    path = tmp_path / "uniform.png"
    assert cv2.imwrite(str(path), np.full((80, 100, 3), 127, dtype=np.uint8))

    metrics = analyze_render(path)
    result = check_render(path, expected_width=100, expected_height=80)

    assert metrics.luminance_standard_deviation == pytest.approx(0.0)
    assert not result.passed
    assert "слишком малый контраст яркости" in result.failures
