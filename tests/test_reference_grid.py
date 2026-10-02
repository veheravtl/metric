"""Тесты метрической сетки временного эталонного ортофотоплана."""

import json
from pathlib import Path

import pytest

from aerial_mapper.reference_grid import ProjectedRasterGrid

MANIFEST_PATH = Path("data/manifests/cherkasy_2021_poc.json")


def test_cherkasy_manifest_produces_expected_metric_grid() -> None:
    """Проверяем физический размер, разрешение и ориентацию выбранного растра.

    Тест не обращается в интернет. Он ловит локальные ошибки конфигурации:
    перепутанные координаты, неверную CRS, случайное изменение размера участка
    или несовпадение заявленного разрешения с количеством пикселей.
    """

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    selection = manifest["selection"]
    center = selection["center_wgs84"]

    grid = ProjectedRasterGrid.centered_on_wgs84(
        longitude=center["longitude"],
        latitude=center["latitude"],
        output_crs=selection["output_crs"],
        width_m=selection["width_m"],
        height_m=selection["height_m"],
        resolution_m_per_pixel=selection["resolution_m_per_pixel"],
    )

    assert grid.crs == "EPSG:32636"
    assert grid.width_m == pytest.approx(250.0)
    assert grid.height_m == pytest.approx(250.0)
    assert grid.width_pixels == 2500
    assert grid.height_pixels == 2500
    assert grid.resolution_x_m_per_pixel == pytest.approx(0.1)
    assert grid.resolution_y_m_per_pixel == pytest.approx(0.1)

    # Североориентированный GeoTIFF имеет положительный шаг по X и отрицательный
    # шаг по строкам: строки изображения отсчитываются сверху вниз.
    assert grid.affine_transform.a == pytest.approx(0.1)
    assert grid.affine_transform.e == pytest.approx(-0.1)


@pytest.mark.parametrize(
    ("width_m", "resolution_m_per_pixel"),
    [
        (0.0, 0.1),
        (250.0, 0.0),
        (250.0, 0.3),
    ],
)
def test_invalid_grid_parameters_are_rejected(
    width_m: float,
    resolution_m_per_pixel: float,
) -> None:
    """Некорректная геометрия должна завершаться явной ошибкой."""

    with pytest.raises(ValueError):
        ProjectedRasterGrid.centered_on_wgs84(
            longitude=32.048364,
            latitude=49.446279,
            output_crs="EPSG:32636",
            width_m=width_m,
            height_m=250.0,
            resolution_m_per_pixel=resolution_m_per_pixel,
        )
