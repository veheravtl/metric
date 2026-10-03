"""Тесты подготовки контрольных Teach/Repeat-пар из телеметрии CLOUD."""

from pathlib import Path

import numpy as np
import pytest
from pyproj import Transformer

from aerial_mapper.cloud_dataset import (
    PositionedImages,
    load_positioned_images,
    select_spaced_image_indices,
    select_teach_repeat_pairs,
)


def _write_flight_csvs(directory: Path) -> None:
    """Создаёт малый полёт с одним кадром до GPS и тремя внутри диапазона."""

    directory.mkdir()
    (directory / "image_ids.csv").write_text(
        "Timestamp, Image Id\n"
        "999000000000000000,9\n"
        "1000000000000000000,10\n"
        "1000000001000000000,11\n"
        "1000000002000000000,12\n",
        encoding="utf-8",
    )
    (directory / "gps.csv").write_text(
        "Timestamp, latitude, longitude, altitude\n"
        "1000000000000000000,43.0,-79.0,100.0\n"
        "1000000002000000000,43.002,-78.996,104.0\n",
        encoding="utf-8",
    )


def test_load_positioned_images_interpolates_without_extrapolation(
    tmp_path: Path,
) -> None:
    """Кадр вне GPS удаляется, а средний получает интерполированную координату."""

    flight_directory = tmp_path / "flight"
    _write_flight_csvs(flight_directory)

    result = load_positioned_images(flight_directory)

    assert result.image_ids.tolist() == [10, 11, 12]
    assert result.altitude_ellipsoid_meters.tolist() == [100.0, 102.0, 104.0]

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
    expected_middle = transformer.transform(-78.998, 43.001)
    assert result.xy_meters[1] == pytest.approx(expected_middle, abs=1e-6)


def test_select_pairs_uses_nearest_and_farthest_teach_positions() -> None:
    """Разметка должна выбирать геометрические крайности, а не номера кадров."""

    teach = PositionedImages(
        image_ids=np.asarray([300, 100, 200], dtype=np.int64),
        timestamps_nanoseconds=np.asarray([1.0, 2.0, 3.0]),
        xy_meters=np.asarray([[0.0, 0.0], [10.0, 0.0], [25.0, 0.0]]),
        altitude_ellipsoid_meters=np.asarray([1.0, 1.0, 1.0]),
        projected_crs="test",
    )
    repeat = PositionedImages(
        image_ids=np.asarray([900, 901, 902], dtype=np.int64),
        timestamps_nanoseconds=np.asarray([4.0, 5.0, 6.0]),
        xy_meters=np.asarray([[1.0, 0.0], [9.0, 0.0], [24.0, 0.0]]),
        altitude_ellipsoid_meters=np.asarray([1.0, 1.0, 1.0]),
        projected_crs="test",
    )

    pairs = select_teach_repeat_pairs(teach, repeat, query_count=2)

    assert [pair.repeat_image_id for pair in pairs] == [900, 902]
    assert pairs[0].positive_teach_image_id == 300
    assert pairs[0].negative_teach_image_id == 200
    assert pairs[0].positive_distance_meters == pytest.approx(1.0)
    assert pairs[0].negative_distance_meters == pytest.approx(24.0)
    assert pairs[1].positive_teach_image_id == 200
    assert pairs[1].negative_teach_image_id == 300


def test_load_positioned_images_rejects_non_monotonic_time(tmp_path: Path) -> None:
    """Перемешанная телеметрия не должна давать тихо неверную интерполяцию."""

    flight_directory = tmp_path / "flight"
    flight_directory.mkdir()
    (flight_directory / "image_ids.csv").write_text(
        "Timestamp, Image Id\n2,1\n1,2\n",
        encoding="utf-8",
    )
    (flight_directory / "gps.csv").write_text(
        "Timestamp, latitude, longitude, altitude\n1,43,-79,100\n2,43,-79,100\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="строго возрастать"):
        load_positioned_images(flight_directory)


def test_select_spaced_image_indices_keeps_route_coverage() -> None:
    """Разрежение должно убрать близкие дубли, сохранив начало и конец пути."""

    positioned = PositionedImages(
        image_ids=np.arange(6, dtype=np.int64),
        timestamps_nanoseconds=np.arange(6, dtype=np.float64),
        xy_meters=np.asarray(
            [
                [0.0, 0.0],
                [0.4, 0.0],
                [1.1, 0.0],
                [1.5, 0.0],
                [2.2, 0.0],
                [2.4, 0.0],
            ]
        ),
        altitude_ellipsoid_meters=np.zeros(6),
        projected_crs="test",
    )

    selected = select_spaced_image_indices(
        positioned,
        minimum_distance_meters=1.0,
    )

    assert selected.tolist() == [0, 2, 4, 5]
