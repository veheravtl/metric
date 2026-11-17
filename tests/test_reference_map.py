"""Проверки калибровки обзорной карты и ограничения репера."""

import pytest
from pyproj import Geod

from aerial_mapper.reference_map import (
    AIRPORT_CENTER,
    AIRPORT_CENTER_PIXEL,
    DEFAULT_REFERENCE_MAP_CALIBRATION,
    MAXIMUM_REFERENCE_DISTANCE_METERS,
    SCALE_CONTROL_PIXEL,
    SCALE_CONTROL_POINT,
    ReferenceMapFailure,
    place_reference_from_geographic,
    place_reference_from_pixel,
    reference_map_meters_per_pixel,
    reference_map_pixel_to_geographic,
)

GEOD = Geod(ellps="WGS84")


def test_control_pixels_recover_both_given_geographic_points() -> None:
    """Две исходные метки обязаны замкнуть калибровку без систематического сдвига."""

    center = place_reference_from_pixel(AIRPORT_CENTER_PIXEL)
    scale = reference_map_pixel_to_geographic(SCALE_CONTROL_PIXEL)

    assert center.point == AIRPORT_CENTER
    _azimuth, _back_azimuth, error_meters = GEOD.inv(
        scale.longitude_degrees,
        scale.latitude_degrees,
        SCALE_CONTROL_POINT.longitude_degrees,
        SCALE_CONTROL_POINT.latitude_degrees,
    )
    assert error_meters == pytest.approx(0.0, abs=1e-6)
    assert reference_map_meters_per_pixel() == pytest.approx(17.632059, rel=1e-6)


def test_geographic_round_trip_preserves_pixel_and_geodesic_metadata() -> None:
    """Сохранённая координата должна вернуться в тот же пиксель интерфейса."""

    selected = place_reference_from_pixel((500.0, 450.0))
    restored = place_reference_from_geographic(selected.point)

    assert restored.map_pixel == pytest.approx((500.0, 450.0), abs=1e-7)
    assert restored.distance_from_center_meters == pytest.approx(
        selected.distance_from_center_meters,
        abs=1e-7,
    )
    assert restored.azimuth_from_center_degrees == pytest.approx(
        selected.azimuth_from_center_degrees,
        abs=1e-9,
    )


def test_pixel_outside_six_kilometre_circle_is_rejected() -> None:
    """Граница изображения не должна подменять требуемую круговую область."""

    with pytest.raises(ReferenceMapFailure, match="не дальше 6 км"):
        place_reference_from_pixel((0.0, 0.0))


def test_exact_six_kilometre_boundary_is_accepted() -> None:
    """Репер на окружности относится к разрешённой области, а не за её пределы."""

    longitude, latitude, _back_azimuth = GEOD.fwd(
        AIRPORT_CENTER.longitude_degrees,
        AIRPORT_CENTER.latitude_degrees,
        90.0,
        MAXIMUM_REFERENCE_DISTANCE_METERS,
    )
    placement = place_reference_from_geographic(
        type(AIRPORT_CENTER)(latitude, longitude)
    )

    assert placement.distance_from_center_meters == pytest.approx(
        MAXIMUM_REFERENCE_DISTANCE_METERS,
        abs=1e-6,
    )


def test_wrong_map_dimensions_are_explicit_calibration_error() -> None:
    """Калибровку нельзя молча применять к уменьшенной копии изображения."""

    calibration = type(DEFAULT_REFERENCE_MAP_CALIBRATION)(
        **{
            **DEFAULT_REFERENCE_MAP_CALIBRATION.__dict__,
            "image_width_pixels": 444,
        }
    )
    with pytest.raises(ReferenceMapFailure, match="вне карты"):
        place_reference_from_pixel((100.0, 100.0), calibration=calibration)
