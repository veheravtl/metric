"""Проверки пролёта как последовательности кадров с телеметрией."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from aerial_mapper.gui.flight_sequence import MarsLvigFlight


@pytest.fixture
def synthetic_flight(tmp_path: Path) -> Path:
    """Создать минимальный экспорт, повторяющий структуру MARS-LVIG."""

    flight_directory = tmp_path / "HKairport-test"
    images_directory = flight_directory / "images"
    images_directory.mkdir(parents=True)

    for image_id, value in enumerate((20, 120, 220)):
        image = np.full((24, 32, 3), value, dtype=np.uint8)
        assert cv2.imwrite(str(images_directory / f"{image_id:06d}.jpg"), image)

    origin = 1_671_606_485_000_000_000
    (flight_directory / "image_ids.csv").write_text(
        "Timestamp,Image Id\n"
        f"{origin},0\n"
        f"{origin + 100_000_000},1\n"
        f"{origin + 200_000_000},2\n",
        encoding="utf-8",
    )
    (flight_directory / "gps.csv").write_text(
        "Timestamp,latitude,longitude,altitude\n"
        f"{origin},22.0,114.0,100.0\n"
        f"{origin + 200_000_000},24.0,116.0,120.0\n",
        encoding="utf-8",
    )
    return flight_directory


def test_load_preserves_frame_index_timing_and_gps(synthetic_flight: Path) -> None:
    """Большое время не должно разрушить интервал 0,1 с и GPS-интерполяцию."""

    flight = MarsLvigFlight.load(synthetic_flight)

    assert flight.frame_count == 3
    assert flight.nominal_fps == pytest.approx(10.0)
    assert flight.interval_after_milliseconds(0) == 100

    frame = flight.read_frame_bgr(1)
    assert frame.shape == (24, 32, 3)

    telemetry = flight.telemetry(1)
    assert telemetry.image_id == 1
    assert telemetry.elapsed_seconds == pytest.approx(0.1)
    assert telemetry.has_position
    assert telemetry.latitude_degrees == pytest.approx(23.0)
    assert telemetry.longitude_degrees == pytest.approx(115.0)
    assert telemetry.altitude_ellipsoid_meters == pytest.approx(110.0)


def test_frame_outside_gps_range_has_no_invented_position(tmp_path: Path) -> None:
    """За пределами GPS нельзя молча подставлять крайнюю известную точку."""

    images_directory = tmp_path / "images"
    images_directory.mkdir()
    for image_id in range(2):
        assert cv2.imwrite(
            str(images_directory / f"{image_id:06d}.jpg"),
            np.zeros((8, 8, 3), dtype=np.uint8),
        )

    (tmp_path / "image_ids.csv").write_text(
        "Timestamp,Image Id\n100,0\n200,1\n", encoding="utf-8"
    )
    (tmp_path / "gps.csv").write_text(
        "Timestamp,latitude,longitude,altitude\n"
        "150,22.0,114.0,100.0\n"
        "200,22.1,114.1,101.0\n",
        encoding="utf-8",
    )

    flight = MarsLvigFlight.load(tmp_path)

    assert not flight.telemetry(0).has_position
    assert flight.telemetry(1).has_position


def test_missing_indexed_image_is_rejected(synthetic_flight: Path) -> None:
    """Разрыв последовательности должен обнаруживаться до воспроизведения."""

    (synthetic_flight / "images/000001.jpg").unlink()

    with pytest.raises(ValueError, match="Не найден кадр Image Id 1"):
        MarsLvigFlight.load(synthetic_flight)
