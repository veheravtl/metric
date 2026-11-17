"""Тесты геодезического измерения ROI и события от удалённого репера."""

import pytest
from pyproj import Geod

from aerial_mapper.reference_event_measurement import (
    GeographicPoint,
    NorthUpMapCalibration,
    ReferenceEventMeasurementFailure,
    map_direction_to_geographic_point,
    map_pixel_to_geographic,
    measure_target_and_event_from_reference,
)

GEOD = Geod(ellps="WGS84")


@pytest.fixture
def calibration() -> NorthUpMapCalibration:
    """Опора в центре маленькой север-ориентированной тестовой карты."""

    return NorthUpMapCalibration(
        anchor=GeographicPoint(22.416088, 114.042978),
        anchor_map_point_px=(50.0, 50.0),
        map_width_pixels=101,
        map_height_pixels=101,
    )


def test_anchor_pixel_preserves_known_geographic_coordinate(
    calibration: NorthUpMapCalibration,
) -> None:
    """Нулевое пиксельное смещение обязано вернуть исходную координату точно."""

    point = map_pixel_to_geographic(
        (50.0, 50.0),
        calibration=calibration,
        map_width_pixels=101,
        map_height_pixels=101,
        map_scale_meters_per_pixel=1.0,
    )

    assert point == calibration.anchor


def test_direction_to_distant_reference_uses_local_north_up_axes(
    calibration: NorthUpMapCalibration,
) -> None:
    """Удалённая южная точка должна задавать вниз без проекции за край карты."""

    reference_lon, reference_lat, _ = GEOD.fwd(
        calibration.anchor.longitude_degrees,
        calibration.anchor.latitude_degrees,
        180.0,
        6000.0,
    )
    direction = map_direction_to_geographic_point(
        (50.0, 50.0),
        target=GeographicPoint(reference_lat, reference_lon),
        calibration=calibration,
        map_width_pixels=101,
        map_height_pixels=101,
        map_scale_meters_per_pixel=1.0,
    )

    assert direction == pytest.approx((0.0, 1.0), abs=1e-12)


def test_rightward_event_has_positive_clockwise_deviation(
    calibration: NorthUpMapCalibration,
) -> None:
    """Событие справа от направления на ROI должно получить знак плюс."""

    # Репер помещаем ровно в 1000 м к югу от координатной опоры. Поэтому ROI
    # в пикселе опоры задаёт северное целевое направление, а сдвиг события на
    # восток однозначно проверяет соглашение о знаке угла.
    reference_lon, reference_lat, _ = GEOD.fwd(
        calibration.anchor.longitude_degrees,
        calibration.anchor.latitude_degrees,
        180.0,
        1000.0,
    )
    result = measure_target_and_event_from_reference(
        [(50.0, 50.0), (60.0, 50.0)],
        calibration=calibration,
        reference=GeographicPoint(reference_lat, reference_lon),
        map_width_pixels=101,
        map_height_pixels=101,
        map_scale_meters_per_pixel=1.0,
    )

    assert result.target.geographic_azimuth_degrees == pytest.approx(0.0, abs=1e-9)
    assert result.target.distance_meters == pytest.approx(1000.0, abs=1e-6)
    assert result.event_azimuth_deviation_degrees > 0.0
    assert result.event.distance_meters > result.target.distance_meters
    assert result.distance_difference_meters == pytest.approx(
        result.event.distance_meters - result.target.distance_meters
    )


def test_leftward_event_has_negative_deviation(
    calibration: NorthUpMapCalibration,
) -> None:
    """Зеркальный сдвиг влево должен менять знак, а не модуль отклонения."""

    reference_lon, reference_lat, _ = GEOD.fwd(
        calibration.anchor.longitude_degrees,
        calibration.anchor.latitude_degrees,
        180.0,
        1000.0,
    )
    result = measure_target_and_event_from_reference(
        [(50.0, 50.0), (40.0, 50.0)],
        calibration=calibration,
        reference=GeographicPoint(reference_lat, reference_lon),
        map_width_pixels=101,
        map_height_pixels=101,
        map_scale_meters_per_pixel=1.0,
    )

    assert result.event_azimuth_deviation_degrees < 0.0


def test_calibration_rejects_resized_map(
    calibration: NorthUpMapCalibration,
) -> None:
    """Та же точка на уменьшенной копии не должна выглядеть географически верной."""

    with pytest.raises(ReferenceEventMeasurementFailure, match="Размер карты"):
        map_pixel_to_geographic(
            (25.0, 25.0),
            calibration=calibration,
            map_width_pixels=51,
            map_height_pixels=51,
            map_scale_meters_per_pixel=1.0,
        )


def test_non_positive_scale_is_rejected(
    calibration: NorthUpMapCalibration,
) -> None:
    """Нулевой масштаб не должен превращать все пиксели в одну координату."""

    with pytest.raises(ReferenceEventMeasurementFailure, match="Масштаб карты"):
        map_pixel_to_geographic(
            (50.0, 50.0),
            calibration=calibration,
            map_width_pixels=101,
            map_height_pixels=101,
            map_scale_meters_per_pixel=0.0,
        )
