# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Вкладка ручного выбора геодезического репера на обзорной карте."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPen, QResizeEvent, QWheelEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from aerial_mapper.reference_event_measurement import GeographicPoint
from aerial_mapper.reference_map import (
    DEFAULT_REFERENCE_MAP_CALIBRATION,
    ReferenceMapFailure,
    ReferencePlacement,
    place_reference_from_geographic,
    place_reference_from_pixel,
    reference_map_meters_per_pixel,
)

from .map_image import MapImageController

REFERENCE_MAP_PATH_KEY = "reference/map_path"
REFERENCE_LATITUDE_KEY = "reference/latitude_degrees"
REFERENCE_LONGITUDE_KEY = "reference/longitude_degrees"


class ReferenceSelectionWidget(QWidget):
    """Показать круг 6 км и передать выбранный репер измерительному экрану."""

    reference_changed = Signal(object)

    def __init__(
        self,
        *,
        settings: QSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._placement: ReferencePlacement | None = None
        self._map_valid = False
        self._overlay_items: list[QGraphicsItem] = []
        self._reference_marker_items: list[QGraphicsItem] = []

        self.map_view = QGraphicsView(self)
        self.map_view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.map_view.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.map_view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.map_image = MapImageController(self.map_view, self)
        self._viewport = self.map_view.viewport()
        self._viewport.installEventFilter(self)
        self.map_view.installEventFilter(self)
        self.map_image.map_loaded.connect(self._on_map_loaded)
        self.map_image.error_occurred.connect(self._on_map_error)

        self._build_interface()
        self._restore_settings()

    @property
    def reference_point(self) -> GeographicPoint | None:
        """Вернуть последний вручную подтверждённый репер."""

        return self._placement.point if self._placement is not None else None

    @property
    def placement(self) -> ReferencePlacement | None:
        """Вернуть координату и геодезическую диагностику выбранного репера."""

        return self._placement

    @property
    def is_map_loaded(self) -> bool:
        """Соответствует ли открытый снимок ожидаемой геометрии 888×795 px."""

        return self._map_valid

    def _build_interface(self) -> None:
        root = QHBoxLayout(self)
        root.addWidget(self.map_view, 1)

        panel = QWidget(self)
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(410)
        panel_layout = QVBoxLayout(panel)

        map_group = QGroupBox("Обзорная карта", panel)
        map_layout = QVBoxLayout(map_group)
        self.open_map_button = QPushButton("Открыть чистую карту…", map_group)
        self.open_map_button.clicked.connect(self._choose_map)
        map_layout.addWidget(self.open_map_button)
        self.fit_button = QPushButton("Показать карту целиком", map_group)
        self.fit_button.clicked.connect(self.map_image.fit_to_view)
        map_layout.addWidget(self.fit_button)
        self.map_status_label = QLabel(
            "Откройте исходный чистый снимок 888×795 px. Файл не копируется "
            "в проект и не добавляется в Git.",
            map_group,
        )
        self.map_status_label.setWordWrap(True)
        map_layout.addWidget(self.map_status_label)
        panel_layout.addWidget(map_group)

        selection_group = QGroupBox("Ручной репер", panel)
        selection_layout = QVBoxLayout(selection_group)
        instruction = QLabel(
            "Левой кнопкой поставьте или перенесите репер внутри синего круга. "
            "Центр креста — аэропорт; радиус круга равен 6 км.",
            selection_group,
        )
        instruction.setWordWrap(True)
        selection_layout.addWidget(instruction)
        self.reference_status_label = QLabel(
            "Репер ещё не выбран. Автоматическое измерение азимута и "
            "дистанции будет недоступно.",
            selection_group,
        )
        self.reference_status_label.setWordWrap(True)
        self.reference_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        selection_layout.addWidget(self.reference_status_label)
        self.clear_button = QPushButton("Сбросить репер", selection_group)
        self.clear_button.clicked.connect(self.clear_reference)
        self.clear_button.setEnabled(False)
        selection_layout.addWidget(self.clear_button)
        panel_layout.addWidget(selection_group)
        panel_layout.addStretch(1)
        root.addWidget(panel)

    def load_reference_map(self, path: str | Path) -> bool:
        """Открыть карту и принять только исходный размер калибровки."""

        self._map_valid = False
        if not self.map_image.load_image(path):
            return False
        return self._map_valid

    def select_reference_at_pixel(self, pixel: tuple[float, float]) -> bool:
        """Поставить репер по исходному пикселю обзорного снимка."""

        if not self._map_valid:
            self.reference_status_label.setText(
                "Сначала откройте исходную чистую карту 888×795 px."
            )
            return False
        try:
            placement = place_reference_from_pixel(pixel)
        except ReferenceMapFailure as error:
            self.reference_status_label.setText(str(error))
            return False
        self._set_placement(placement, persist=True, emit=True)
        return True

    def clear_reference(self) -> None:
        """Удалить выбранный репер из интерфейса и настроек пользователя."""

        self._placement = None
        self._remove_items(self._reference_marker_items)
        self._settings.remove(REFERENCE_LATITUDE_KEY)
        self._settings.remove(REFERENCE_LONGITUDE_KEY)
        self._settings.sync()
        self.clear_button.setEnabled(False)
        self.reference_status_label.setText(
            "Репер ещё не выбран. Автоматическое измерение азимута и "
            "дистанции будет недоступно."
        )
        self.reference_changed.emit(None)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Преобразовать клик виджета в исходный пиксель без потери точности."""

        if watched is self._viewport and self._map_valid:
            if (
                event.type() == QEvent.Type.MouseButtonPress
                and isinstance(event, QMouseEvent)
                and event.button() == Qt.MouseButton.LeftButton
            ):
                scene_point = self.map_view.mapToScene(event.position().toPoint())
                self.select_reference_at_pixel(
                    (float(scene_point.x()), float(scene_point.y()))
                )
                return True
            if event.type() == QEvent.Type.Wheel and isinstance(event, QWheelEvent):
                factor = 1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2
                self.map_view.scale(factor, factor)
                return True
        if watched is self.map_view and event.type() == QEvent.Type.Resize:
            # Карта часто загружается до того, как layout назначит области
            # окончательный размер. Контекст map_view отменит callback, если
            # вкладка будет уничтожена до следующего прохода цикла событий.
            QTimer.singleShot(
                0, self.map_view, self.map_image.fit_to_view
            )
        return super().eventFilter(watched, event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """При изменении вкладки вернуть всю допустимую область в поле зрения."""

        super().resizeEvent(event)
        if self._map_valid:
            self.map_image.fit_to_view()

    def _choose_map(self) -> None:
        saved_path = self._settings.value(REFERENCE_MAP_PATH_KEY, "", type=str)
        start = Path(saved_path).parent if saved_path else Path.home()
        filename, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Выберите чистую обзорную карту 888×795 px",
            str(start),
            "Изображения (*.png *.jpg *.jpeg *.tif *.tiff);;Все файлы (*)",
        )
        if filename:
            self.load_reference_map(filename)

    def _on_map_loaded(self, path: str, width: int, height: int) -> None:
        calibration = DEFAULT_REFERENCE_MAP_CALIBRATION
        expected = (calibration.image_width_pixels, calibration.image_height_pixels)
        if (width, height) != expected:
            self.map_image.clear()
            self._map_valid = False
            self.map_status_label.setText(
                "Карта отклонена: ожидался исходный размер "
                f"{expected[0]}×{expected[1]} px, получен {width}×{height} px. "
                "Масштабированная или обрезанная копия даст неверные координаты."
            )
            return

        self._map_valid = True
        self._settings.setValue(REFERENCE_MAP_PATH_KEY, path)
        self._settings.sync()
        self.map_status_label.setText(
            f"Открыта {Path(path).name}: {width}×{height} px. "
            f"Масштаб ≈ {reference_map_meters_per_pixel():.3f} м/px."
        )
        self._draw_static_overlays()
        if self._placement is not None:
            self._draw_reference_marker(self._placement.map_pixel)

    def _on_map_error(self, message: str) -> None:
        self._map_valid = False
        self.map_status_label.setText(message)

    def _restore_settings(self) -> None:
        latitude = self._settings.value(REFERENCE_LATITUDE_KEY, None)
        longitude = self._settings.value(REFERENCE_LONGITUDE_KEY, None)
        if latitude is not None and longitude is not None:
            try:
                placement = place_reference_from_geographic(
                    GeographicPoint(float(latitude), float(longitude))
                )
            except (TypeError, ValueError, ReferenceMapFailure):
                self._settings.remove(REFERENCE_LATITUDE_KEY)
                self._settings.remove(REFERENCE_LONGITUDE_KEY)
            else:
                self._set_placement(placement, persist=False, emit=False)

        saved_path = self._settings.value(REFERENCE_MAP_PATH_KEY, "", type=str)
        if saved_path and Path(saved_path).is_file():
            self.load_reference_map(saved_path)

    def _set_placement(
        self,
        placement: ReferencePlacement,
        *,
        persist: bool,
        emit: bool,
    ) -> None:
        self._placement = placement
        if persist:
            self._settings.setValue(
                REFERENCE_LATITUDE_KEY, placement.point.latitude_degrees
            )
            self._settings.setValue(
                REFERENCE_LONGITUDE_KEY, placement.point.longitude_degrees
            )
            self._settings.sync()
        self.clear_button.setEnabled(True)
        self.reference_status_label.setText(
            "Репер выбран\n"
            f"Широта: {placement.point.latitude_degrees:.7f}°\n"
            f"Долгота: {placement.point.longitude_degrees:.7f}°\n"
            f"От центра: {placement.distance_from_center_meters:.1f} м\n"
            f"Азимут от центра: {placement.azimuth_from_center_degrees:.3f}°"
        )
        if self._map_valid:
            self._draw_reference_marker(placement.map_pixel)
        if emit:
            self.reference_changed.emit(placement.point)

    def _draw_static_overlays(self) -> None:
        scene = self.map_view.scene()
        if scene is None:
            return
        self._remove_items(self._overlay_items)
        calibration = DEFAULT_REFERENCE_MAP_CALIBRATION
        radius_pixels = (
            calibration.maximum_distance_meters / reference_map_meters_per_pixel()
        )
        center_x, center_y = calibration.center_pixel
        radius_pen = QPen(QColor("#00b7ff"), 2.0)
        radius_pen.setCosmetic(True)
        circle = scene.addEllipse(
            center_x - radius_pixels,
            center_y - radius_pixels,
            2.0 * radius_pixels,
            2.0 * radius_pixels,
            radius_pen,
        )
        circle.setZValue(20.0)
        self.radius_item: QGraphicsEllipseItem = circle
        self._overlay_items.append(circle)

        center_pen = QPen(QColor("#ffcc00"), 2.0)
        center_pen.setCosmetic(True)
        for x1, y1, x2, y2 in (
            (center_x - 8.0, center_y, center_x + 8.0, center_y),
            (center_x, center_y - 8.0, center_x, center_y + 8.0),
        ):
            line = scene.addLine(x1, y1, x2, y2, center_pen)
            line.setZValue(22.0)
            self._overlay_items.append(line)

    def _draw_reference_marker(self, pixel: tuple[float, float]) -> None:
        scene = self.map_view.scene()
        if scene is None:
            return
        self._remove_items(self._reference_marker_items)
        x, y = pixel
        pen = QPen(QColor("#ff304f"), 2.5)
        pen.setCosmetic(True)
        marker = scene.addEllipse(x - 7.0, y - 7.0, 14.0, 14.0, pen)
        marker.setZValue(30.0)
        self._reference_marker_items.append(marker)
        for x1, y1, x2, y2 in (
            (x - 11.0, y, x + 11.0, y),
            (x, y - 11.0, x, y + 11.0),
        ):
            line: QGraphicsLineItem = scene.addLine(x1, y1, x2, y2, pen)
            line.setZValue(31.0)
            self._reference_marker_items.append(line)

    @staticmethod
    def _remove_items(items: list[QGraphicsItem]) -> None:
        for item in items:
            scene = item.scene()
            if scene is not None:
                scene.removeItem(item)
        items.clear()
