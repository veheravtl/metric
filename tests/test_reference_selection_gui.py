"""Проверки вкладки ручного выбора репера."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, QSettings, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from aerial_mapper.gui.reference_selection import (
    REFERENCE_CALIBRATION_VERSION,
    REFERENCE_CALIBRATION_VERSION_KEY,
    REFERENCE_LATITUDE_KEY,
    REFERENCE_LONGITUDE_KEY,
    ReferenceSelectionWidget,
)
from aerial_mapper.reference_map import (
    AIRPORT_CENTER_PIXEL,
    DEFAULT_REFERENCE_MAP_CALIBRATION,
    reference_map_meters_per_pixel,
)


@pytest.fixture(scope="module")
def qt_application():
    application = QApplication.instance() or QApplication([])
    yield application


def _settings(tmp_path: Path, name: str) -> QSettings:
    return QSettings(str(tmp_path / name), QSettings.Format.IniFormat)


def _reference_map(tmp_path: Path, *, width: int = 888, height: int = 795) -> Path:
    path = tmp_path / f"reference-{width}x{height}.png"
    image = np.full((height, width, 3), 96, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def test_click_selects_reference_and_persists_geographic_point(
    qt_application: QApplication,
    tmp_path: Path,
) -> None:
    """Клик в сцене должен пережить масштаб вида и следующий запуск вкладки."""

    settings = _settings(tmp_path, "reference.ini")
    widget = ReferenceSelectionWidget(settings=settings)
    assert widget.load_reference_map(_reference_map(tmp_path))
    widget.resize(1200, 800)
    widget.show()
    qt_application.processEvents()

    selected_scene_pixel = QPointF(500.0, 450.0)
    selected_view_pixel = widget.map_view.mapFromScene(selected_scene_pixel)
    received = []
    widget.reference_changed.connect(received.append)
    viewport = widget.map_view.viewport()
    global_position = QPointF(viewport.mapToGlobal(selected_view_pixel))
    event = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(selected_view_pixel),
        global_position,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )

    QApplication.sendEvent(viewport, event)
    qt_application.processEvents()

    placement = widget.placement
    assert placement is not None
    assert placement.map_pixel == pytest.approx((500.0, 450.0), abs=1.0)
    assert received == [placement.point]
    assert settings.contains(REFERENCE_LATITUDE_KEY)
    assert settings.contains(REFERENCE_LONGITUDE_KEY)
    assert (
        settings.value(REFERENCE_CALIBRATION_VERSION_KEY, type=int)
        == REFERENCE_CALIBRATION_VERSION
    )

    restored = ReferenceSelectionWidget(settings=settings)
    assert restored.reference_point == placement.point
    assert restored.is_map_loaded
    restored.close()
    widget.close()


def test_legacy_reference_from_rotated_calibration_is_invalidated(
    qt_application: QApplication,
    tmp_path: Path,
) -> None:
    """Версия без знака оси Y не должна незаметно влиять на новые измерения."""

    settings = _settings(tmp_path, "legacy-reference.ini")
    settings.setValue(REFERENCE_LATITUDE_KEY, 22.4)
    settings.setValue(REFERENCE_LONGITUDE_KEY, 114.0)
    settings.sync()

    widget = ReferenceSelectionWidget(settings=settings)

    assert widget.reference_point is None
    assert not settings.contains(REFERENCE_LATITUDE_KEY)
    assert not settings.contains(REFERENCE_LONGITUDE_KEY)
    assert not settings.contains(REFERENCE_CALIBRATION_VERSION_KEY)
    widget.close()


def test_six_kilometre_circle_has_calibrated_pixel_radius(
    qt_application: QApplication,
    tmp_path: Path,
) -> None:
    """Нарисованный диаметр должен означать 12 км, а не размер виджета."""

    widget = ReferenceSelectionWidget(settings=_settings(tmp_path, "circle.ini"))
    assert widget.load_reference_map(_reference_map(tmp_path))

    expected_radius = (
        DEFAULT_REFERENCE_MAP_CALIBRATION.maximum_distance_meters
        / reference_map_meters_per_pixel()
    )
    rect = widget.radius_item.rect()
    assert rect.center().x() == pytest.approx(AIRPORT_CENTER_PIXEL[0])
    assert rect.center().y() == pytest.approx(AIRPORT_CENTER_PIXEL[1])
    assert rect.width() == pytest.approx(2.0 * expected_radius)
    assert rect.height() == pytest.approx(2.0 * expected_radius)
    widget.close()


def test_click_outside_circle_is_rejected_without_losing_previous_reference(
    qt_application: QApplication,
    tmp_path: Path,
) -> None:
    """Ошибочный клик не должен заменять последний допустимый репер."""

    widget = ReferenceSelectionWidget(settings=_settings(tmp_path, "limit.ini"))
    assert widget.load_reference_map(_reference_map(tmp_path))
    assert widget.select_reference_at_pixel((500.0, 450.0))
    previous = widget.placement

    assert not widget.select_reference_at_pixel((0.0, 0.0))
    assert widget.placement == previous
    assert "не дальше 6 км" in widget.reference_status_label.text()
    widget.close()


def test_resized_map_is_rejected_instead_of_miscalibrated(
    qt_application: QApplication,
    tmp_path: Path,
) -> None:
    """Похожая картинка другого размера не должна давать правдоподобный репер."""

    widget = ReferenceSelectionWidget(settings=_settings(tmp_path, "size.ini"))

    assert not widget.load_reference_map(
        _reference_map(tmp_path, width=444, height=398)
    )
    assert not widget.is_map_loaded
    assert "ожидался исходный размер 888×795" in widget.map_status_label.text()
    widget.close()
