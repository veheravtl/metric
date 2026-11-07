"""Проверки загрузки растровой карты в Qt-интерфейс."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QGraphicsView

from aerial_mapper.gui.main_window import MainWindow
from aerial_mapper.gui.map_image import MapImageController


@pytest.fixture(scope="module")
def qt_application():
    """Предоставить единственный QApplication для GUI-проверок модуля."""

    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def test_map(tmp_path: Path) -> Path:
    """Создать PNG с известным размером и различимыми цветовыми каналами."""

    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, :, 0] = 30
    image[:, :, 1] = 120
    image[:, :, 2] = 220
    path = tmp_path / "map.png"
    assert cv2.imwrite(str(path), image)
    return path


def test_map_controller_loads_pixels_and_reports_original_size(
    qt_application: QApplication, test_map: Path
) -> None:
    """Показ не должен менять исходный размер или порядок каналов для алгоритма."""

    controller = MapImageController(QGraphicsView())
    loaded_events: list[tuple[str, int, int]] = []
    controller.map_loaded.connect(
        lambda path, width, height: loaded_events.append((path, width, height))
    )

    assert controller.load_image(test_map)
    assert controller.is_loaded
    assert controller.source_path == test_map.resolve()
    assert loaded_events == [(str(test_map.resolve()), 60, 40)]

    image_bgr = controller.current_image_bgr
    assert image_bgr is not None
    assert image_bgr.shape == (40, 60, 3)
    assert image_bgr[0, 0].tolist() == [30, 120, 220]

    controller.clear()
    assert not controller.is_loaded
    assert controller.current_image_bgr is None


def test_main_window_reports_loaded_map(
    qt_application: QApplication, test_map: Path
) -> None:
    """Главное окно должно связать контроллер с правым mapView и статусом."""

    window = MainWindow()

    assert window.map_image.load_image(test_map)
    assert window.ui.mapView.scene() is not None
    assert window.ui.statusbar.currentMessage() == "Карта: map.png | 60×40 пикселей"

    window.close()


def test_missing_map_is_rejected(qt_application: QApplication, tmp_path: Path) -> None:
    """Отсутствующий файл должен дать явную ошибку, а не пустую чёрную область."""

    controller = MapImageController(QGraphicsView())
    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    assert not controller.load_image(tmp_path / "missing.png")
    assert errors and "Файл карты не найден" in errors[0]
